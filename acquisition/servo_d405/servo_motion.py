"""Run timed PWM in a camera-independent process with shared command history."""

import csv
import io
import math
import multiprocessing
import signal
import time

from gobilda_servo import GoBildaServo, HardwarePWM, UPDATE_INTERVAL
from servo_patterns import play_trajectory


class MotionCancelled(Exception):
    pass


class _MemoryLoggedServo(GoBildaServo):
    def __init__(self, pwm, stop, origin, history, commands):
        super().__init__(pwm)
        self.stop_event, self.origin, self.history = stop, origin, history
        self.commands = csv.writer(commands)
        self.commands.writerow(["host_monotonic_ns", "host_unix_ns", "event",
                                "commanded_angle_deg", "pulse_us", "elapsed_s"])
        if self.angle is not None:
            self._log("initial_pwm_state", pwm.get_duty_ns() / 1000)

    def _log(self, event, pulse):
        times, angles, pulses, count, lock = self.history
        # Timestamp and publish under the same lock used by camera snapshots.
        # A snapshot must not miss a command whose timestamp precedes receipt.
        with lock:
            ns = time.monotonic_ns()
            origin = self.origin.value
            elapsed = (ns - origin) / 1e9 if origin else ""
            self.commands.writerow([ns, time.time_ns(), event, self.angle, pulse, elapsed])
            if self.angle is not None:
                index = count.value
                if index >= len(times):
                    raise RuntimeError("Servo command history capacity exceeded")
                times[index], angles[index], pulses[index] = ns, self.angle, pulse
                count.value += 1

    def set_pulse_us(self, us):
        if self.stop_event.is_set():
            raise MotionCancelled()
        pulse = super().set_pulse_us(us)
        self._log("command", pulse)
        return pulse

    def release(self):
        super().release()
        self._log("released", "")


def _run_motion(args, trajectory, command_path, stop, launch, ready, done, failed, released,
                origin, history, errors, pwm_factory):
    def request_stop(signum, frame):
        stop.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    commands = io.StringIO()
    servo = pwm = None

    def fail(exc):
        failed.set()
        errors.put(f"{type(exc).__name__}: {exc}")

    try:
        pwm = pwm_factory(channel=args.channel, chip=args.chip)
        servo = _MemoryLoggedServo(pwm, stop, origin, history, commands)
        servo.move_to(trajectory.samples[0][1], args.speed)
        if stop.wait(0.3):
            raise MotionCancelled()
        ready.set()
        while not launch.wait(0.02):
            if stop.is_set():
                raise MotionCancelled()
        if play_trajectory(servo, trajectory, origin.value, stop):
            done.set()
        stop.wait()
    except MotionCancelled:
        pass
    except BaseException as exc:
        fail(exc)
    finally:
        try:
            if servo:
                servo.release()
            elif pwm:
                pwm.release()
            released.set()
        except BaseException as exc:
            fail(exc)
        # No camera calls or regular-file writes occur in the timed PWM loop.
        # Publish the complete journal after PWM is disabled.
        try:
            with command_path.open("x", newline="") as handle:
                handle.write(commands.getvalue())
        except BaseException as exc:
            fail(exc)
        ready.set()


class PatternMotion:
    def __init__(self, args, trajectory, command_path, pwm_factory=HardwarePWM):
        self.args = args
        context = multiprocessing.get_context("spawn")
        self.stop, self.launch, self.ready, self.done, self.failed, self.released = [context.Event() for _ in range(6)]
        self.origin = context.Value("q", 0)
        capacity = len(trajectory.samples) + math.ceil(300 / (args.speed * UPDATE_INTERVAL)) + 16
        self.history = (context.RawArray("q", capacity), context.RawArray("d", capacity),
                        context.RawArray("d", capacity), context.RawValue("i", 0), context.Lock())
        self.errors = context.Queue()
        self._error = None
        self.process = context.Process(
            target=_run_motion,
            args=(args, trajectory, command_path, self.stop, self.launch, self.ready,
                  self.done, self.failed, self.released, self.origin, self.history, self.errors, pwm_factory),
            name="servo-pattern",
        )
        self.process.start()

    @property
    def error(self):
        if self._error is None and self.failed.is_set():
            try:
                self._error = self.errors.get(timeout=0.5)
            except Exception:
                self._error = "Servo worker failed without an error message"
        if self._error is None and self.process.exitcode is not None and not self.stop.is_set():
            self._error = f"Servo worker exited unexpectedly ({self.process.exitcode})"
        return self._error

    def prepare(self, capture_stop):
        while not self.ready.wait(0.02):
            if capture_stop.is_set():
                raise MotionCancelled()
            if self.error:
                raise RuntimeError(self.error)
        if capture_stop.is_set():
            raise MotionCancelled()
        if self.error:
            raise RuntimeError(self.error)

    def start(self, origin_ns):
        self.origin.value = origin_ns
        self.launch.set()

    def snapshot_at(self, host_ns):
        times, angles, pulses, count, lock = self.history
        with lock:
            for index in range(count.value - 1, -1, -1):
                if times[index] <= host_ns:
                    return times[index], angles[index], pulses[index]
        return "", "", ""

    def release(self):
        self.stop.set()
        self.launch.set()
        self.process.join(timeout=5)
        if self.process.is_alive() and self.released.is_set():
            # PWM is already off; allow slow storage to finish the command journal.
            self.process.join()
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
            HardwarePWM(channel=self.args.channel, chip=self.args.chip).release()
            raise RuntimeError("Servo worker did not stop; killed it and released PWM from the parent")
        error = self.error
        self.errors.close()
        self.errors.join_thread()
        if error:
            raise RuntimeError(error)
