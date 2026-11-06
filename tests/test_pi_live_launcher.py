"""Execution cannot be enabled by read-only and shadow Pi entry points."""
import unittest
from pathlib import Path

from pi_support.runner import command_for, parser


class LiveLauncherTests(unittest.TestCase):
    def test_only_explicit_live_print_enables_execute(self):
        root = Path(__file__).resolve().parents[1]
        for command in ("live-check", "live-shadow", "live-print"):
            args = parser().parse_args([command])
            argv = command_for(args, root)
            self.assertIn("controller.live_print", argv)
            self.assertNotIn("--execute", argv)
            self.assertIn(str(root / "profiles/live/settings.json"), argv)
        args = parser().parse_args(["live-print", "--execute"])
        self.assertEqual(command_for(args, root).count("--execute"), 1)
        for command in ("live-check", "live-shadow"):
            with self.assertRaises(SystemExit):
                parser().parse_args([command, "--execute"])

    def test_live_test_has_no_hardware_command(self):
        argv = command_for(parser().parse_args(["live-self-test"]))
        self.assertIn("unittest", argv)
        self.assertNotIn("--execute", argv)


if __name__ == "__main__":
    unittest.main()
