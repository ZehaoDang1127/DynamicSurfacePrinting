#!/usr/bin/env python3
"""
goBILDA 2000 Series Dual Mode Servo (25-2, Torque) on a Raspberry Pi 5, GPIO18.

Uses the Pi 5's hardware PWM (RP1 PWM0, channel 2) through the kernel's sysfs
interface, so pulses are jitter-free and no extra Python packages are needed.
(RPi.GPIO and pigpio do not work on the Pi 5.)

ONE-TIME SETUP
  1. Add this line to /boot/firmware/config.txt (under [all]) and reboot:
         dtoverlay=pwm-2chan
  2. Confirm GPIO18 is in PWM mode:
         pinctrl get 18     ->  should show "a3 ... PWM0_CHAN2"

WIRING
  Servo signal -> GPIO18 (physical pin 12). 3.3 V logic is fine for this servo.
  Servo +V     -> separate 4.8-7.4 V supply rated 3 A or more
                  (stall current is about 2.5 A at 6 V).
  Servo GND    -> supply GND *and* a Pi GND pin (grounds must be shared).
  Do NOT power the servo from the Pi's 5 V pin.

SPEED
  Every move is slowed to SPEED_DEG_PER_SEC (set just below the imports).
  Change that number to change the default, or override it for one run with
  --speed, e.g. --speed 10 (very slow) or --speed 120 (fast).
  The servo's own top speed is about 240 deg/s, so bigger numbers won't go faster.
  The first move after power-up or 'off' runs at full speed, because the
  program can't know where the servo is until it has sent it somewhere.

USAGE
  python3 gobilda_servo.py sweep                  # 0 -> 300 -> 0 deg over and over (Ctrl+C stops)
  python3 gobilda_servo.py sweep --speed 10       # same, but slower
  python3 gobilda_servo.py sweep --start 60 --end 240 --speed 20 --times 10 --pause 1
  python3 gobilda_servo.py angle 150              # move slowly to 150 deg and hold
  python3 gobilda_servo.py pulse 1500             # send a raw pulse width (us) and hold
  python3 gobilda_servo.py interactive            # type angles; servo moves to each one slowly
  python3 gobilda_servo.py speed 0.5 -t 3         # continuous mode only: half speed for 3 s
  python3 gobilda_servo.py off                    # stop pulses; servo goes limp

  If you get "Permission denied", run with sudo.
"""

import argparse
import glob
import math
import os
import time

# ---- Change this to make the servo faster or slower by default --------------
SPEED_DEG_PER_SEC = 30.0   # degrees per second (lower = slower). 30 -> 10 s for 300 deg
# ------------------------------------------------------------------------------

UPDATE_INTERVAL = 0.02        # seconds between position updates (one 50 Hz servo frame)
PWM_CLASS = "/sys/class/pwm"
RP1_PWM0_ADDR = "1f00098000"  # RP1 PWM0 block, which drives GPIO18 on the Pi 5
GPIO18_CHANNEL = 2            # GPIO18 = PWM0_CHAN2 on the Pi 5
SERVO_FREQ_HZ = 50            # standard 20 ms servo frame


# ---------------------------------------------------------------------------
# Low-level sysfs PWM
# ---------------------------------------------------------------------------

def _write(path, value, retries=50):
    """Write to a sysfs file, retrying briefly while udev fixes permissions after export."""
    for attempt in range(retries):
        try:
            with open(path, "w") as f:
                f.write(str(value))
            return
        except (PermissionError, FileNotFoundError):
            if attempt == retries - 1:
                raise
            time.sleep(0.02)


def _read_int(path):
    with open(path) as f:
        return int(f.read().strip())


def find_pwmchip():
    """Find the sysfs pwmchip for RP1 PWM0 (its number varies between kernel versions)."""
    chips = sorted(glob.glob(os.path.join(PWM_CLASS, "pwmchip*")))
    for chip in chips:
        if RP1_PWM0_ADDR in os.path.realpath(chip):
            return chip
    # Fallback: if there's exactly one chip with enough channels, use it.
    candidates = [c for c in chips if _read_int(os.path.join(c, "npwm")) > GPIO18_CHANNEL]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        "Couldn't identify the PWM controller for GPIO18.\n"
        "Is 'dtoverlay=pwm-2chan' in /boot/firmware/config.txt, and did you reboot?\n"
        f"Chips found: {chips or 'none'}  (pick one with --chip)"
    )


class HardwarePWM:
    """Minimal wrapper around one sysfs PWM channel."""

    def __init__(self, channel=GPIO18_CHANNEL, chip=None, freq_hz=SERVO_FREQ_HZ):
        self.chip = chip or find_pwmchip()
        self.channel = channel
        self.path = os.path.join(self.chip, f"pwm{channel}")

        if not os.path.isdir(self.path):
            _write(os.path.join(self.chip, "export"), channel)
            for _ in range(100):  # wait for the channel directory to appear
                if os.path.isdir(self.path):
                    break
                time.sleep(0.01)

        self.period_ns = round(1e9 / freq_hz)
        # The kernel rejects a period shorter than the current duty cycle.
        if _read_int(self._f("duty_cycle")) > self.period_ns:
            _write(self._f("duty_cycle"), 0)
        _write(self._f("period"), self.period_ns)

    def _f(self, name):
        return os.path.join(self.path, name)

    def set_duty_ns(self, ns):
        _write(self._f("duty_cycle"), int(round(ns)))

    def get_duty_ns(self):
        return _read_int(self._f("duty_cycle"))

    def is_enabled(self):
        return _read_int(self._f("enable")) == 1

    def enable(self):
        _write(self._f("enable"), 1)

    def disable(self):
        _write(self._f("enable"), 0)

    def release(self):
        """Stop the output and unexport the channel."""
        try:
            self.disable()
        finally:
            _write(os.path.join(self.chip, "unexport"), self.channel)


# ---------------------------------------------------------------------------
# Servo
# ---------------------------------------------------------------------------

class GoBildaServo:
    """
    goBILDA 2000-0025-0002 (25-2, Torque).

    Default (position) mode: 500-2500 us  ->  0-300 deg, 1500 us = 150 deg (center).
    Continuous mode (set with the goBILDA servo programmer):
        900-2100 us  ->  full speed one way .. full speed the other, 1500 us = stop.
    Increasing pulse width turns the servo clockwise.
    """

    def __init__(self, pwm, min_us=500, max_us=2500, range_deg=300.0,
                 cr_min_us=900, cr_max_us=2100, center_us=1500):
        self.pwm = pwm
        self.min_us, self.max_us = min_us, max_us
        self.range_deg = range_deg
        self.cr_min_us, self.cr_max_us = cr_min_us, cr_max_us
        self.center_us = center_us  # trim this if a continuous servo creeps at "stop"
        self._enabled = False
        self.angle = None  # last commanded angle; None = unknown (limp or just started)

        # If an earlier run left the servo holding a position, continue from there.
        try:
            if pwm.is_enabled():
                self._enabled = True
                self._track(pwm.get_duty_ns() / 1000)
        except (AttributeError, OSError, ValueError):
            pass

    def _track(self, us):
        if self.min_us <= us <= self.max_us:
            self.angle = (us - self.min_us) / (self.max_us - self.min_us) * self.range_deg
        else:
            self.angle = None

    def set_pulse_us(self, us):
        us = max(self.min_us, min(self.max_us, us))
        self.pwm.set_duty_ns(us * 1000)
        if not self._enabled:
            self.pwm.enable()
            self._enabled = True
        self._track(us)
        return us

    def set_angle(self, degrees):
        """Jump straight to an angle (the servo moves at its full speed)."""
        degrees = max(0.0, min(self.range_deg, degrees))
        us = self.min_us + (self.max_us - self.min_us) * degrees / self.range_deg
        return self.set_pulse_us(us)

    def move_to(self, degrees, speed=SPEED_DEG_PER_SEC):
        """Move smoothly to an angle at `speed` degrees per second. Returns when done."""
        target = max(0.0, min(self.range_deg, degrees))
        if self.angle is None:          # position unknown: nothing to ramp from
            us = self.set_angle(target)
            time.sleep(abs(target - self.range_deg / 2) / 240 + 0.3)  # let it get there
            return us

        start = self.angle
        steps = max(1, math.ceil(abs(target - start) / (speed * UPDATE_INTERVAL)))
        t0 = time.monotonic()
        for i in range(1, steps + 1):
            us = self.set_angle(start + (target - start) * i / steps)
            wait = t0 + i * UPDATE_INTERVAL - time.monotonic()  # keep steady timing
            if wait > 0:
                time.sleep(wait)
        return us

    def set_speed(self, speed):
        """Continuous mode: -1.0 (full CCW) .. 0 (stop) .. +1.0 (full CW)."""
        speed = max(-1.0, min(1.0, speed))
        if speed >= 0:
            us = self.center_us + speed * (self.cr_max_us - self.center_us)
        else:
            us = self.center_us + speed * (self.center_us - self.cr_min_us)
        return self.set_pulse_us(us)

    def stop(self):
        """Continuous mode: stop spinning."""
        return self.set_pulse_us(self.center_us)

    def release(self):
        """Stop sending pulses; the servo goes limp."""
        self.pwm.release()
        self._enabled = False
        self.angle = None


# ---------------------------------------------------------------------------
# Command-line demo
# ---------------------------------------------------------------------------

def run_sweep(servo, speed, start=0.0, end=None, times=0, pause=0.5):
    """Sweep start -> end -> start slowly, repeating `times` times (0 = until Ctrl+C)."""
    end = servo.range_deg if end is None else end
    print(f"Moving to start ({start:g} deg)...")
    servo.move_to(start, speed)
    time.sleep(pause)

    print(f"Sweeping {start:g} <-> {end:g} deg at {speed:g} deg/s "
          f"(about {abs(end - start) / speed:.1f} s each way)")
    cycle = 0
    while times == 0 or cycle < times:
        cycle += 1
        print(f"Cycle {cycle}" + (f" of {times}" if times else "   (Ctrl+C to stop)"))
        for target in (end, start):
            servo.move_to(target, speed)
            time.sleep(pause)  # brief stop at each end


def run_interactive(servo, continuous, speed):
    what = "speed (-1 to 1)" if continuous else f"angle (0 to {servo.range_deg:g})"
    print(f"Enter {what}, 'p<us>' for a raw pulse (e.g. p1500), or 'q' to quit.")
    while True:
        try:
            s = input("> ").strip().lower()
        except EOFError:
            break
        if s in ("q", "quit", "exit"):
            break
        try:
            if s.startswith("p"):
                us = servo.set_pulse_us(float(s[1:]))
            elif continuous:
                us = servo.set_speed(float(s))
            else:
                us = servo.move_to(float(s), speed)
            print(f"  pulse = {us:.0f} us")
        except ValueError:
            print("  ?")


def positive_float(text):
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def main():
    p = argparse.ArgumentParser(description="goBILDA 2000 Series servo on Raspberry Pi 5 GPIO18")
    p.add_argument("--chip", help="override pwmchip path, e.g. /sys/class/pwm/pwmchip0")
    p.add_argument("--channel", type=int, default=GPIO18_CHANNEL, help="PWM channel (GPIO18 = 2)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_speed(parser):
        parser.add_argument("--speed", type=positive_float, default=SPEED_DEG_PER_SEC,
                            help=f"degrees per second, lower = slower (default {SPEED_DEG_PER_SEC:g})")

    an = sub.add_parser("angle", help="move slowly to an angle and hold")
    an.add_argument("degrees", type=float)
    add_speed(an)
    sub.add_parser("pulse", help="send a raw pulse width and hold").add_argument("us", type=float)
    sw = sub.add_parser("sweep", help="sweep back and forth repeatedly")
    sw.add_argument("--start", type=float, default=0.0, help="start angle (deg)")
    sw.add_argument("--end", type=float, default=300.0, help="end angle (deg)")
    sw.add_argument("--times", type=int, default=0, help="number of back-and-forth cycles (0 = forever)")
    sw.add_argument("--pause", type=float, default=0.5, help="seconds to wait at each end")
    add_speed(sw)
    sp = sub.add_parser("speed", help="continuous mode only: spin at a speed")
    sp.add_argument("value", type=float, help="-1.0 .. 1.0")
    sp.add_argument("-t", "--duration", type=float, default=2.0, help="seconds")
    it = sub.add_parser("interactive", help="type positions (or speeds)")
    it.add_argument("--continuous", action="store_true", help="servo is in continuous mode")
    add_speed(it)
    sub.add_parser("off", help="stop pulses and release the servo")
    args = p.parse_args()

    servo = GoBildaServo(HardwarePWM(channel=args.channel, chip=args.chip))

    if args.cmd == "off":
        servo.release()
        print("PWM stopped, servo released.")
        return

    # These leave the PWM running after the script exits so the servo holds position.
    if args.cmd == "angle":
        if servo.angle is None:
            print("Position unknown (servo was off), so this first move runs at full speed.")
        try:
            servo.move_to(args.degrees, args.speed)
        except KeyboardInterrupt:
            print("\nStopped early.")  # the servo holds wherever it got to
        print(f"Holding {servo.angle:.0f} deg. Run 'off' to release.")
        return
    if args.cmd == "pulse":
        us = servo.set_pulse_us(args.us)
        print(f"Holding {us:.0f} us pulse. Run 'off' to release.")
        return

    continuous = args.cmd == "speed" or (args.cmd == "interactive" and args.continuous)
    try:
        if args.cmd == "sweep":
            run_sweep(servo, args.speed, args.start, args.end, args.times, args.pause)
        elif args.cmd == "speed":
            us = servo.set_speed(args.value)
            print(f"Spinning at {args.value:+.2f} ({us:.0f} us) for {args.duration:g} s")
            time.sleep(args.duration)
        elif args.cmd == "interactive":
            run_interactive(servo, args.continuous, args.speed)
    except KeyboardInterrupt:
        print()
    finally:
        if continuous:
            servo.stop()
            time.sleep(0.2)
        servo.release()


if __name__ == "__main__":
    main()
