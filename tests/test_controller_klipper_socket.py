"""Local Klipper protocol and backend tests; in-memory streams never address a printer."""

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from controller.cold_run import dry_run
from controller.config import load_config
from controller.connection import create_printer_client
from controller.klipper_socket import KlipperSocketClient
from controller.types import ControllerFault
from test_controller_moonraker import status


ROOT = Path(__file__).resolve().parents[1]


def config():
    cfg = load_config(ROOT/"configs/controller.example.json")
    return replace(cfg, substrate_height_mm=20, hardware=replace(cfg.hardware,
                   backend="klipper_socket", klipper_socket_path="/test/klippy.sock", calibration_id="test",
                   camera_config_sha256="test", expires_utc="2100-01-01T00:00:00Z",
                   stationary_fixture_verified=True, stop_test_verified=True,
                   exclusive_control_verified=True, fixture_top_z_mm=20))


class MemoryConnection:
    def __init__(self, handler):
        self.handler = handler
        self.connections = []
        self.requests = []

    async def __call__(self, path):
        reader = asyncio.StreamReader(limit=1024*1024)
        connection = self
        class Writer:
            def write(self, wire):
                if not wire.endswith(b"\x03"):
                    raise AssertionError("missing ETX framing")
                raw = json.loads(wire[:-1])
                connection.requests.append(raw)
                connection.handler(raw, reader)
            async def drain(self):
                await asyncio.sleep(0)
            def close(self):
                reader.feed_eof()
            async def wait_closed(self):
                pass
        self.connections.append(reader)
        return reader, Writer()


def reply(reader, request_id, result):
    reader.feed_data(json.dumps({"id": request_id, "result": result}).encode()+b"\x03")


class LocalKlipperTests(unittest.IsolatedAsyncioTestCase):
    async def test_fragmented_coalesced_out_of_order_replies_and_notifications(self):
        saved = []
        def handler(raw, reader):
            saved.append(raw)
            if len(saved) == 2:
                wire = (json.dumps({"params": {"eventtime": 1}})+"\x03"+
                        json.dumps({"id": saved[1]["id"], "result": {"value": "second"}})+"\x03"+
                        json.dumps({"id": saved[0]["id"], "result": {"value": "first"}})+"\x03").encode()
                reader.feed_data(wire[:12])
                reader.feed_data(wire[12:])
        connector = MemoryConnection(handler)
        client = KlipperSocketClient(config(), connector)
        try:
            a, b = await asyncio.gather(client.rpc("first"), client.rpc("second"))
            self.assertEqual(a, {"value": "first"})
            self.assertEqual(b, {"value": "second"})
            self.assertEqual(len(client.notifications), 1)
            self.assertEqual(len(connector.connections), 1)
        finally:
            await client.close()

    async def test_lost_motion_reply_does_not_retry_or_reconnect(self):
        cfg = config()
        cfg = replace(cfg, hardware=replace(cfg.hardware, request_timeout_s=.02))
        connector = MemoryConnection(lambda raw, reader: None)
        client = KlipperSocketClient(cfg, connector)
        try:
            with self.assertRaisesRegex(ControllerFault, "no retry"):
                await client.execute_gcode("G90")
            with self.assertRaises(ControllerFault):
                await client.execute_gcode("G90")
            self.assertEqual(len(connector.requests), 1)
            self.assertEqual(len(connector.connections), 1)
        finally:
            await client.close()

    async def test_stop_uses_a_separate_connection_while_script_is_pending(self):
        def handler(raw, reader):
            if raw["method"] == "emergency_stop":
                reply(reader, raw["id"], {})
        connector = MemoryConnection(handler)
        client = KlipperSocketClient(config(), connector)
        ordinary = asyncio.create_task(client.execute_gcode("G1 X111\nM400"))
        try:
            while not connector.requests:
                await asyncio.sleep(0)
            self.assertEqual(await client.emergency_stop(), {})
            self.assertEqual(len(connector.connections), 2)
            self.assertEqual([r["method"] for r in connector.requests], ["gcode/script", "emergency_stop"])
            self.assertFalse(ordinary.done())
        finally:
            ordinary.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await ordinary
            await client.close()

    async def test_errors_eof_and_malformed_messages_fail_pending_calls(self):
        for wire in (b'[]\x03', b'{"id":999,"result":{}}\x03', b'bad json\x03', None):
            def handler(raw, reader):
                reader.feed_eof() if wire is None else reader.feed_data(wire)
            client = KlipperSocketClient(config(), MemoryConnection(handler))
            try:
                with self.assertRaises(ControllerFault):
                    await client.rpc("info")
                self.assertTrue(client.broken)
            finally:
                await client.close()

    async def test_read_only_discovery_and_cold_run_use_local_endpoints(self):
        values = status()
        def handler(raw, reader):
            method = raw["method"]
            if method == "info":
                result = {"state": "ready", "hostname": "fake-linux", "software_version": "test"}
            elif method == "objects/list":
                result = {"objects": list(values)}
            elif method == "objects/query":
                result = {"status": values, "eventtime": 10}
            elif method == "gcode/script":
                if "G1 " in raw["params"]["script"]:
                    for obj, field in (("motion_report", "live_position"), ("gcode_move", "position"), ("gcode_move", "gcode_position")):
                        values[obj][field] = [111, 120, 25, 0]
                result = {}
            else:
                self.fail(method)
            reply(reader, raw["id"], result)
        connector = MemoryConnection(handler)
        client = KlipperSocketClient(config(), connector)
        report = await client.discover()
        self.assertFalse(report["server"]["network_required"])
        self.assertFalse(any(r["method"] == "gcode/script" for r in connector.requests))
        with tempfile.TemporaryDirectory() as directory, patch("controller.cold_run.create_printer_client", return_value=client):
            result = await dry_run(config(), np.array([[110, 120], [111, 120]]), Path(directory)/"run", execute=True)
        self.assertTrue(result["path_complete"], result)
        self.assertEqual(len(connector.connections), 1)

    async def test_local_backend_selection_never_constructs_http_client(self):
        with patch("controller.moonraker.MoonrakerClient") as network:
            client = create_printer_client(config())
            network.assert_not_called()
            self.assertIsInstance(client, KlipperSocketClient)
            await client.close()

    @unittest.skipUnless(os.name == "nt", "Windows-specific diagnostic")
    async def test_native_windows_host_is_rejected_without_a_network_fallback(self):
        client = KlipperSocketClient(config())
        try:
            with self.assertRaisesRegex(ControllerFault, "same Linux host"):
                await client.discover()
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()
