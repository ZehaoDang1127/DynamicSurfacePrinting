"""Write lossless RGB/depth PNGs or FFV1 videos in a separate process."""

import csv
import hashlib
import io
import multiprocessing
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading

from PIL import Image


class _VideoSink:
    def __init__(self, output, width, height, fps):
        self.encoders = []
        self.logs = []
        self.readers = []
        self.buffers = []
        self.paths = []
        self.buffered_bytes = 0
        self.buffer_lock = threading.Lock()
        self.buffer_error = None
        try:
            for kind, input_format, encoded_format in (("color", "rgb24", "bgr0"), ("depth", "gray16le", "gray16le")):
                log = (output / f"ffmpeg_{kind}.log").open("wb")
                self.logs.append(log)
                path = output / f"{kind}.mkv"
                if path.exists():
                    raise FileExistsError(path)
                command = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-n",
                    "-f", "rawvideo", "-pixel_format", input_format,
                    "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0",
                    "-an", "-c:v", "ffv1", "-level", "3", "-pix_fmt", encoded_format,
                    "-threads", "1", "-fps_mode", "passthrough", "-f", "matroska", "pipe:1",
                ]
                encoder = subprocess.Popen(command, stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=log,
                                           start_new_session=True)
                self.encoders.append(encoder)
                chunks = []
                self.buffers.append(chunks)
                self.paths.append(path)
                reader = threading.Thread(target=self._collect, args=(encoder.stdout, chunks), name=f"buffer-{kind}")
                self.readers.append(reader)
                reader.start()
        except BaseException:
            self.abort()
            raise

    def _collect(self, stream, chunks):
        try:
            while True:
                chunk = stream.read(256 * 1024)
                if not chunk:
                    break
                with self.buffer_lock:
                    self.buffered_bytes += len(chunk)
                    if self.buffered_bytes > 512 * 1024**2:
                        raise RuntimeError("Compressed video exceeded the 512 MiB RAM budget; use shorter sequences")
                chunks.append(chunk)
        except BaseException as exc:
            self.buffer_error = exc
        finally:
            stream.close()

    def write(self, row, color, depth):
        if self.buffer_error:
            raise RuntimeError(str(self.buffer_error))
        for kind, array, encoder in zip(("color", "depth"), (color, depth), self.encoders):
            pixels = array.tobytes(order="C")
            row[f"{kind}_sha256"] = hashlib.sha256(pixels).hexdigest()
            encoder.stdin.write(pixels)

    def close(self):
        for encoder in self.encoders:
            encoder.stdin.close()
        codes = [encoder.wait() for encoder in self.encoders]
        for reader in self.readers:
            reader.join()
        for log in self.logs:
            log.close()
        if any(codes):
            raise RuntimeError(f"FFV1 encoder failed ({codes}); see ffmpeg_color.log / ffmpeg_depth.log")
        if self.buffer_error:
            raise RuntimeError(str(self.buffer_error))
        # The parent only sends the end marker after capture/PWM have stopped.
        for path, chunks in zip(self.paths, self.buffers):
            with path.open("xb") as handle:
                handle.writelines(chunks)
            chunks.clear()

    def abort(self):
        for encoder in self.encoders:
            if encoder.poll() is None:
                encoder.kill()
            encoder.wait()
            if encoder.stdin:
                try:
                    encoder.stdin.close()
                except (OSError, BrokenPipeError):
                    pass
        for log in self.logs:
            log.close()
        for reader in self.readers:
            reader.join()


def _write_images(output, columns, pending, errors, saved, ready, storage_format, width, height, fps):
    # The parent stops acquisition/PWM, then asks this worker to drain its queue.
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, signal.SIG_IGN)
    video = None
    try:
        if storage_format == "ffv1":
            video = _VideoSink(output, width, height, fps)
        buffered_rows = io.StringIO(newline="") if video else None
        with (output / "frames.csv").open("x", newline="", buffering=1) as handle:
            rows = csv.DictWriter(buffered_rows if video else handle, fieldnames=columns)
            rows.writeheader()
            ready.set()
            while True:
                item = pending.get()
                if item is None:
                    break
                row, color, depth = item
                if video:
                    video.write(row, color, depth)
                else:
                    Image.fromarray(color).save(output / row["color_file"], compress_level=1)
                    Image.fromarray(depth).save(output / row["depth_file"], compress_level=1)
                rows.writerow(row)
                with saved.get_lock():
                    saved.value += 1
            if video:
                video.close()
                video = None
                handle.write(buffered_rows.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException as exc:
        errors.put(f"{type(exc).__name__}: {exc}")
    finally:
        if video:
            video.abort()
        ready.set()


class ProcessDatasetWriter:
    """A bounded queue isolates image compression and file I/O from servo timing."""

    def __init__(self, output, columns, queue_size=32, storage_format="png", width=None, height=None, fps=None):
        self.output = Path(output)
        if storage_format == "png":
            (self.output / "color").mkdir()
            (self.output / "depth").mkdir()
        context = multiprocessing.get_context("spawn")
        self.pending = context.Queue(maxsize=queue_size)
        self.errors = context.Queue()
        self._saved = context.Value("Q", 0)
        self._error = None
        self.ready = context.Event()
        self.process = context.Process(
            target=_write_images,
            args=(self.output, columns, self.pending, self.errors, self._saved, self.ready,
                  storage_format, width, height, fps),
            name="png-writer",
        )
        self.process.start()
        # Imports/startup can take seconds on the Pi; finish them before PWM starts.
        while not self.ready.wait(0.1):
            if not self.process.is_alive():
                break
        if self.error:
            self.close()

    @property
    def saved(self):
        return self._saved.value

    @property
    def error(self):
        if self._error is None:
            try:
                self._error = self.errors.get_nowait()
            except queue.Empty:
                if self.process.exitcode is not None and self.process.exitcode != 0:
                    self._error = f"PNG writer exited with code {self.process.exitcode}"
        return self._error

    def submit(self, row, color, depth):
        if self.error:
            raise RuntimeError(f"Image writer failed: {self.error}")
        try:
            self.pending.put_nowait((row, color.copy(), depth.copy()))
        except queue.Full as exc:
            raise RuntimeError("Disk writer cannot keep up; recording stopped to preserve timing.") from exc

    def close(self):
        while self.process.is_alive():
            try:
                self.pending.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self.process.join()
        error = self.error
        self.pending.close()
        # A failed writer may leave queued buffers with no reader.
        self.pending.cancel_join_thread()
        self.errors.close()
        self.errors.join_thread()
        if error:
            raise RuntimeError(f"Image writer failed: {error}")
