"""One serialized writer; ambiguous submissions latch the session uncertain."""

import asyncio
from dataclasses import replace
import math

from .types import ControllerFault


def gcode_for(trajectory, cold=True, synchronize=False):
    lines = []
    for move in trajectory.moves:
        if not all(math.isfinite(v) for v in (*move.xyz, move.feed_mm_s, move.extrusion_mm)) or move.feed_mm_s <= 0:
            raise ControllerFault("invalid G-code move")
        if cold and move.extrusion_mm != 0:
            raise ControllerFault("cold transport refuses extrusion")
        x, y, z = move.xyz
        line = f"G1 X{x:.6f} Y{y:.6f} Z{z:.6f} F{60*move.feed_mm_s:.6f}"
        if not cold:
            line += f" E{move.extrusion_mm:.7f}"
        lines.append(line)
    if synchronize:
        lines.append("M400")
    return "\n".join(lines)


class SerializedTransport:
    def __init__(self, max_buffer_s, config_id, log=lambda *a, **kw: None):
        self.lock = asyncio.Lock()
        self.max_buffer_s = max_buffer_s
        self.config_id = config_id
        self.attempted = set()
        self.last_id = None
        self.committed_xyz = None
        self.committed_end_s = None
        self.uncertain = False
        self.log = log
        self.cold = True

    async def submit(self, trajectory, now_s):
        async with self.lock:
            if self.uncertain:
                raise ControllerFault("motion state is uncertain; create a new reconciled session")
            key = (trajectory.segment_id, trajectory.revision)
            if key in self.attempted or trajectory.predecessor_id != self.last_id:
                raise ControllerFault("duplicate or out-of-order segment")
            if self.last_id is not None and trajectory.segment_id <= self.last_id:
                raise ControllerFault("segment IDs must increase")
            if trajectory.config_id != self.config_id or not trajectory.moves:
                raise ControllerFault("trajectory configuration differs from transport session")
            if now_s > trajectory.valid_until_s or trajectory.start_s < now_s - 1e-9:
                raise ControllerFault("trajectory expired or start deadline missed")
            if trajectory.end_s - now_s > self.max_buffer_s + 1e-9:
                raise ControllerFault("committed buffer would exceed time limit")
            if self.committed_xyz is not None and (trajectory.start_xyz != self.committed_xyz or
                    abs(trajectory.start_s - self.committed_end_s) > 1e-6):
                raise ControllerFault("trajectory does not join immutable committed endpoint")
            self.attempted.add(key)  # Before network access; never automatically replay a lost reply.
            self.log("submission", segment_id=trajectory.segment_id, revision=trajectory.revision,
                     submitted_s=now_s, script=gcode_for(trajectory, cold=self.cold), estimated_end_s=trajectory.end_s)
            try:
                reply = await self._send(trajectory)
            except BaseException:
                self.uncertain = True
                raise
            self.last_id = trajectory.segment_id
            self.committed_xyz, self.committed_end_s = trajectory.moves[-1].xyz, trajectory.end_s
            self.log("acknowledgement", segment_id=trajectory.segment_id, reply=reply,
                     physical_completion=False)
            return reply

    async def _send(self, trajectory):
        raise NotImplementedError

    async def emergency_stop(self):
        raise NotImplementedError


class FakePrinter(SerializedTransport):
    """Deterministic transport fault injection, including execution before reply loss."""

    def __init__(self, max_buffer_s, config_id, log=lambda *a, **kw: None,
                 fail_on_segment=None, failure="lost-reply", delay_s=0):
        super().__init__(max_buffer_s, config_id, log)
        self.cold = False  # Allows virtual extrusion; no hardware connection exists.
        self.fail_on_segment = fail_on_segment
        self.failure = failure
        self.delay_s = delay_s
        self.executed = []
        self.stopped = False

    async def _send(self, trajectory):
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if trajectory.segment_id == self.fail_on_segment and self.failure == "partial":
            self.executed.append(replace(trajectory, moves=trajectory.moves[:1]))
            raise ControllerFault("fake printer executed a prefix, then rejected the batch")
        self.executed.append(trajectory)
        if trajectory.segment_id == self.fail_on_segment:
            raise ControllerFault("fake printer accepted motion but its reply was lost")
        return "accepted (virtual schedule only)"

    async def emergency_stop(self):
        self.stopped = True
        return "virtual stop"
