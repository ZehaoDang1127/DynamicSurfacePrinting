"""The deployment shortcuts must remain read-only or offline."""

import json
from pathlib import Path
import tempfile
import unittest

from pi_support.runner import command_for, parser


class PiLauncherTests(unittest.TestCase):
    def test_preview_cannot_enable_execution(self):
        args = parser().parse_args(["preview", "--output", "output/test"])
        command = command_for(args, Path("/bundle"))
        self.assertEqual(command[1:4], ["-m", "controller.stationary", "run"])
        self.assertNotIn("--execute", command)
        with self.assertRaises(SystemExit):
            parser().parse_args(["preview", "--execute"])

    def test_diagnostics_reads_selected_profile_and_preserves_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "custom.json"
            profile.write_text(json.dumps({"hardware": {"moonraker_url": "http://printer.local:7125"}}))
            session = root / "old run"
            session.mkdir()
            (session / "execution_events.jsonl").write_text('{"event": "preflight"}\n')
            args = parser().parse_args(["diagnose", "--config", "custom.json", "--session", "old run"])
            command = command_for(args, root)
            self.assertEqual(command[command.index("--url") + 1], "http://printer.local:7125")
            self.assertEqual(command[command.index("--session") + 1], str(root / "old run"))
            override = parser().parse_args(["diagnose", "--config", "missing.json", "--url", "http://other:7125"])
            self.assertIn("http://other:7125", command_for(override, root))

    def test_explicit_session_must_exist_and_contain_run_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "trial"
            args = parser().parse_args(["diagnose", "--offline", "--session", "trial"])
            with self.assertRaisesRegex(ValueError, "existing execution directory"):
                command_for(args, root)
            session.mkdir()
            (session / "unrelated.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "no recognized execution records"):
                command_for(args, root)
            (session / "last_status_snapshot.json").write_text("{}")
            self.assertIn(str(session.resolve()), command_for(args, root))

    def test_offline_diagnostics_needs_no_profile_or_url(self):
        args = parser().parse_args(["diagnose", "--offline", "--config", "missing.json"])
        command = command_for(args, Path("/missing-bundle"))
        self.assertIn("--offline", command)
        self.assertNotIn("--url", command)

    def test_capture_and_baseline_never_invoke_controller(self):
        for name, module in (("capture", "surface_mapper"), ("baseline", "prediction")):
            with self.subTest(name=name):
                args = parser().parse_args([name, "--frames", "9"])
                command = command_for(args, Path("/bundle"))
                self.assertEqual(command[1:3], ["-m", module])
                self.assertIn("realsense", command)
                self.assertNotIn("--execute", command)
                self.assertEqual(command[command.index("--frames") + 1], "9")
        args = parser().parse_args(["capture", "--frames", "-1"])
        with self.assertRaises(ValueError):
            command_for(args)


if __name__ == "__main__":
    unittest.main()
