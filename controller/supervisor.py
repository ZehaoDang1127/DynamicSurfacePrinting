"""Latched controller states: restart/reconnect never resumes an armed run."""

from .types import ControllerFault, State
import asyncio


class Supervisor:
    def __init__(self, log=lambda *a, **k: None):
        self.state = State.DISCONNECTED
        self.reason = None
        self.log = log
        self._fault_lock = asyncio.Lock()

    def transition(self, state):
        allowed = {
            State.DISCONNECTED: {State.READY, State.FAULT},
            State.READY: {State.WARMUP, State.DISCONNECTED, State.FAULT},
            State.WARMUP: {State.ARMED, State.STOPPING, State.FAULT},
            State.ARMED: {State.RUNNING, State.STOPPING, State.FAULT},
            State.RUNNING: {State.STOPPING, State.FAULT},
            State.STOPPING: {State.READY, State.FAULT},
            State.FAULT: set(),
        }
        if state not in allowed[self.state]:
            raise ControllerFault(f"invalid controller transition {self.state.value} -> {state.value}")
        previous = self.state
        self.state = state
        self.log("state", previous=previous.value, state=state.value)

    def arm(self, checks):
        failed = [name for name, satisfied in checks.items() if satisfied is not True]
        if not checks or failed:
            raise ControllerFault("arming failed: " + ", ".join(failed or ["no checks provided"]))
        self.transition(State.ARMED)

    async def fault(self, reason, transport=None):
        async with self._fault_lock:
            if self.state == State.FAULT:
                return
            self.reason = str(reason)
            if self.state in {State.WARMUP, State.ARMED, State.RUNNING}:
                self.transition(State.STOPPING)
            self.log("fault", reason=self.reason)
            if transport is not None:
                try:
                    await transport.emergency_stop()
                    self.log("emergency_stop", delivered=True)
                except Exception as exc:
                    self.log("emergency_stop", delivered=False, reason=str(exc))
            self.transition(State.FAULT)
