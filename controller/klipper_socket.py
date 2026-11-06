"""Local Klipper host API, JSON + ETX framing. No HTTP, TCP, UART or MCU ownership.

Controller and Klippy must run on the same Unix host. Klippy alone owns the MCU
connections and retains its normal motion and thermal protection responsibilities.
"""

import asyncio
from collections import deque
import json
import os
import time

from . import __version__
from .types import ControllerFault


class KlipperSocketClient:
    MAX_MESSAGE_BYTES = 1024 * 1024

    def __init__(self, config, connector=None):
        self.config = config
        self.path = config.hardware.klipper_socket_path
        if not self.path:
            raise ControllerFault("set hardware.klipper_socket_path to Klippy's existing local API socket")
        self.timeout = config.hardware.request_timeout_s
        # An injected byte-stream connector permits protocol tests on Windows.
        self.connector = connector
        self.reader = self.writer = self.reader_task = None
        self.pending = {}
        self.next_id = 0
        self.connect_lock = asyncio.Lock()
        self.write_lock = asyncio.Lock()
        self.broken = None
        self.closed = False
        self.notifications = deque(maxlen=128)
        self.notification_drops = 0

    async def _connect(self):
        async with self.connect_lock:
            if self.closed or self.broken:
                raise ControllerFault(self.broken or "local Klipper client is closed")
            if self.writer is not None:
                return
            if self.connector is None:
                if os.name == "nt":
                    raise ControllerFault("local Klipper socket mode requires the controller and Klippy on the same Linux host")
                self.reader, self.writer = await asyncio.open_unix_connection(self.path, limit=self.MAX_MESSAGE_BYTES)
            else:
                self.reader, self.writer = await self.connector(self.path)
            self.reader_task = asyncio.create_task(self._read_messages())

    def _fail(self, reason):
        self.broken = self.broken or str(reason)
        for future in self.pending.values():
            if not future.done():
                future.set_exception(ControllerFault(self.broken))
        self.pending.clear()

    async def _read_messages(self):
        try:
            while True:
                wire = await self.reader.readuntil(b"\x03")
                if len(wire) > self.MAX_MESSAGE_BYTES:
                    raise ControllerFault("local Klipper response exceeds message limit")
                raw = json.loads(wire[:-1])
                if not isinstance(raw, dict):
                    raise ControllerFault("local Klipper response must be an object")
                if "id" not in raw:
                    if len(self.notifications) == self.notifications.maxlen:
                        self.notification_drops += 1
                    self.notifications.append(raw)
                    continue
                # Replies can be interleaved/out of order; never associate one by receive order.
                future = self.pending.pop(raw["id"], None)
                if future is None or future.done():
                    raise ControllerFault("local Klipper response has unknown or duplicate request ID")
                if "error" in raw:
                    future.set_exception(ControllerFault(f"Klipper rejected request: {raw['error']}"))
                elif "result" not in raw or not isinstance(raw["result"], dict):
                    future.set_exception(ControllerFault("local Klipper response lacks an object result"))
                else:
                    future.set_result(raw["result"])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail(f"local Klipper connection failed ({type(exc).__name__}): {exc}")

    async def rpc(self, method, params=None):
        try:
            return await asyncio.wait_for(self._rpc_request(method, params), timeout=self.timeout)
        except (asyncio.TimeoutError, OSError) as exc:
            self._fail(f"local Klipper {method} failed ({type(exc).__name__}); no retry or reconnect")
            raise ControllerFault(self.broken) from exc
        except asyncio.CancelledError:
            self._fail(f"local Klipper {method} cancelled; command acceptance may be uncertain")
            raise

    async def _rpc_request(self, method, params):
        future = None
        request_id = None
        try:
            await self._connect()
            async with self.write_lock:
                if self.broken or self.closed:
                    raise ControllerFault(self.broken or "local Klipper client is closed")
                self.next_id += 1
                request_id = self.next_id
                wire = json.dumps({"id": request_id, "method": method, "params": params or {}},
                                  allow_nan=False).encode("utf-8") + b"\x03"
                if len(wire) > self.MAX_MESSAGE_BYTES:
                    raise ControllerFault("local Klipper request exceeds message limit")
                future = asyncio.get_running_loop().create_future()
                self.pending[request_id] = future
                self.writer.write(wire)
                await self.writer.drain()
            return await future
        finally:
            if request_id is not None:
                self.pending.pop(request_id, None)
            # A drain error can set an exception before rpc has awaited its future.
            if future is not None:
                if future.done() and not future.cancelled():
                    future.exception()
                elif not future.done():
                    future.cancel()

    async def discover(self):
        printer = await self.rpc("info", {"client_info": {"name": "dynamic_surface_controller", "version": __version__}})
        available = (await self.rpc("objects/list"))["objects"]
        wanted = {name: None for name in ("webhooks", "toolhead", "gcode_move", "motion_report", "extruder",
                  "heater_bed", "virtual_sdcard", "bed_mesh", "configfile", "idle_timeout", "z_thermal_adjust")
                  if name in available}
        before = time.monotonic()
        status = await self.rpc("objects/query", {"objects": wanted})
        after = time.monotonic()
        return {"server": {"backend": "klipper_socket", "network_required": False, "socket_path": self.path},
                "printer": printer, "objects": available, "query": status,
                "request_sent_s": before, "receipt_s": after, "round_trip_s": after-before,
                "motion_schedule_known": False, "notification_drops": self.notification_drops}

    async def execute_gcode(self, script):
        return await self.rpc("gcode/script", {"script": script})

    async def emergency_stop(self):
        # A fresh local connection avoids waiting behind a pending G-code reply or
        # a failed ordinary writer. It is a new stop request, never a motion retry.
        stop_client = KlipperSocketClient(self.config, self.connector)
        try:
            return await stop_client.rpc("emergency_stop")
        finally:
            await stop_client.close()

    async def close(self):
        self.closed = True
        self._fail("local Klipper client closed")
        if self.reader_task is not None:
            self.reader_task.cancel()
            try:
                await self.reader_task
            except asyncio.CancelledError:
                pass
        if self.writer is not None:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except OSError:
                pass
