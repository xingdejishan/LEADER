import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from tools.resource_guard import DiskGuard, GIB


def _gpu_pids(gpu_uuid):
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15,
    )
    if result.returncode:
        raise RuntimeError("nvidia-smi process query failed: {}".format(result.stderr.strip()))
    pids = set()
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2 and fields[1] == gpu_uuid:
            pids.add(int(fields[0]))
    return pids


def _process_group(pid):
    try:
        value = Path("/proc/{}/stat".format(pid)).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return int(value.rsplit(")", 1)[1].split()[2])


def _event(stream, event, **values):
    record = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **values}
    stream.write(json.dumps(record, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _stop_owned_group(child, stream, reason):
    if child.poll() is not None:
        return
    if _process_group(child.pid) != child.pid:
        raise RuntimeError("guard refused to signal an unexpected process group")
    _event(stream, "stopping_owned_run", reason=reason, pid=child.pid)
    os.killpg(child.pid, signal.SIGINT)
    try:
        child.wait(timeout=60)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(child.pid, signal.SIGKILL)
    child.wait()


def run(args):
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("guarded command is empty")
    guard = DiskGuard(args.watch_path, int(args.minimum_free_gib * GIB))
    available = guard.available_bytes()
    if available < int(args.start_free_gib * GIB):
        raise RuntimeError("filesystem free space is below the guarded start threshold")
    if _gpu_pids(args.gpu_uuid):
        raise RuntimeError("selected GPU is no longer idle before launch")
    log_path = Path(args.log).resolve()
    if log_path.exists():
        raise FileExistsError("guard log already exists")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8", newline="\n") as stream:
        _event(stream, "start", free_bytes=available, gpu_uuid=args.gpu_uuid, command=command)
        child = subprocess.Popen(command, start_new_session=True)
        _event(stream, "child_started", pid=child.pid)
        next_gpu = time.monotonic()
        next_snapshot = time.monotonic()
        gpu_errors = 0
        while child.poll() is None:
            available = guard.available_bytes()
            if available < guard.minimum_free_bytes:
                _stop_owned_group(child, stream, "disk_free_below_floor")
                break
            now = time.monotonic()
            if now >= next_gpu:
                try:
                    foreign = sorted(pid for pid in _gpu_pids(args.gpu_uuid)
                                     if _process_group(pid) != child.pid)
                    gpu_errors = 0
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                    gpu_errors += 1
                    _event(stream, "gpu_query_error", count=gpu_errors, error=str(error))
                    foreign = []
                if foreign:
                    _stop_owned_group(child, stream, "foreign_gpu_processes={}".format(foreign))
                    break
                if gpu_errors >= 3:
                    _stop_owned_group(child, stream, "gpu_visibility_lost")
                    break
                next_gpu = now + 30
            if now >= next_snapshot:
                _event(stream, "resource_snapshot", free_bytes=available)
                next_snapshot = now + 60
            time.sleep(5)
        returncode = child.wait()
        _event(stream, "child_exited", returncode=returncode, free_bytes=guard.available_bytes())
        return returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch-path", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=4.0)
    parser.add_argument("--start-free-gib", type=float, default=7.0)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.minimum_free_gib < 2 or args.start_free_gib < args.minimum_free_gib + 2:
        raise ValueError("guard thresholds do not preserve a safe write buffer")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
