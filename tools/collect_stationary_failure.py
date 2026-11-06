"""Collect stationary-job diagnostics using local reads and HTTP GET only.

Run in the Pi bundle directory. This does not restart Klipper, move axes, heat,
upload files, start/resume/cancel jobs, or send an emergency stop.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import urllib.error
import urllib.request


def get(url, *, tail=False):
    headers = {}
    if os.environ.get("MOONRAKER_API_KEY"):
        headers["X-Api-Key"] = os.environ["MOONRAKER_API_KEY"]
    if tail:
        headers["Range"] = "bytes=-65536"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=6) as response:
            if tail:
                if response.status != 206:
                    return {"error": "server did not honor bounded log-tail request", "http_status": response.status}
                return {"text": response.read(65536).decode("utf-8", errors="replace")}
            return json.loads(response.read(4_000_000).decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return {"error": str(exc), "type": type(exc).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://192.168.50.2:7125")
    parser.add_argument("--session", type=Path)
    parser.add_argument("--offline", action="store_true", help="collect local files only")
    args = parser.parse_args()
    root = Path.cwd()
    out = root / "output" / ("stationary_diagnostic_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    record = {"captured_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "post-incident diagnostics; current readings are not the historical triggering sample",
              "printer_commands_sent": 0, "automatic_recovery_attempted": False}
    if not args.offline:
        record["printer_info"] = get(args.url.rstrip("/") + "/printer/info")
        record["current_status"] = get(args.url.rstrip("/") +
            "/printer/objects/query?gcode_move=position,gcode_position,homing_origin&motion_report=live_position,live_velocity"
            "&toolhead=position,homed_axes&print_stats=state,filename&virtual_sdcard=file_position,is_active")
        tail = get(args.url.rstrip("/") + "/server/files/logs/klippy.log", tail=True)
        if "text" in tail:
            (out / "klippy_tail.log").write_text(tail["text"], encoding="utf-8")
            record["klippy_tail_saved"] = True
        else:
            record["klippy_tail_result"] = tail
    session = args.session
    if session is None:
        candidates = []
        for file in (root / "output").rglob("execution.json"):
            if not file.resolve().is_relative_to((root / "output").resolve()):
                continue
            try:
                execution = json.loads(file.read_text(encoding="utf-8"))
                if execution.get("state") == "FAULT" and execution.get("start_requested"):
                    candidates.append(file)
            except (OSError, ValueError):
                continue
        if candidates:
            session = max(candidates, key=lambda file: file.stat().st_mtime).parent
    if session is not None:
        record["source_session"] = str(session.resolve())
        for name in ("execution.json", "job.json", "last_status_snapshot.json"):
            file = session / name
            if file.is_file():
                record[name] = json.loads(file.read_text(encoding="utf-8"))
        events = session / "execution_events.jsonl"
        if events.is_file():
            with events.open("rb") as stream:
                stream.seek(max(0, events.stat().st_size - 131072))
                data = stream.read().decode("utf-8", errors="replace")
            (out / "execution_events_tail.txt").write_text(data, encoding="utf-8")
        record["historical_triggering_pose_available"] = "last_status_snapshot.json" in record
    else:
        record["local_failure"] = "no started FAULT execution.json found under this directory's output folder"
    (out / "diagnostic.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"diagnostic_directory": str(out),
        "printer_info": record.get("printer_info"), "current_status": record.get("current_status"),
        "last_execution": record.get("execution.json"),
        "historical_triggering_pose_available": record.get("historical_triggering_pose_available", False),
        "printer_commands_sent": 0}, indent=2))


if __name__ == "__main__":
    main()
