#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Watchdog for OOM risks in RL/LLM training.
- Monitors: system mem, target process RSS, GPU mem/util, disk usage.
- Logs to CSV; prints alerts; optional alert command.
"""

import argparse, csv, os, re, shlex, signal, subprocess, sys, time
from datetime import datetime

try:
    import psutil
except ImportError:
    print("Missing dependency: psutil. Install via `pip install psutil`", file=sys.stderr); sys.exit(1)

# ---- GPU helpers ------------------------------------------------------------
def _nvml_available():
    try:
        import pynvml  # type: ignore
        return True
    except Exception:
        return False

def gpu_stats():
    """Return list of dicts: [{'index':0,'mem_used':MB,'mem_total':MB,'util':%}, ...]"""
    stats = []
    if _nvml_available():
        import pynvml
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                stats.append({
                    "index": i,
                    "mem_used": mem.used / (1024**2),
                    "mem_total": mem.total / (1024**2),
                    "util": float(util)
                })
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    else:
        # fallback: parse nvidia-smi
        try:
            q = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                text=True
            ).strip().splitlines()
            for line in q:
                idx, mu, mt, u = [x.strip() for x in line.split(",")]
                stats.append({
                    "index": int(idx),
                    "mem_used": float(mu),
                    "mem_total": float(mt),
                    "util": float(u)
                })
        except Exception:
            # No GPU or nvidia-smi not found
            pass
    return stats

# ---- Process helpers --------------------------------------------------------
def find_pid_by_regex(pattern):
    rx = re.compile(pattern)
    matches = []
    for p in psutil.process_iter(attrs=["pid","name","cmdline"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
            name = p.info.get("name") or ""
            if rx.search(cmd) or rx.search(name):
                matches.append(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return matches

def human(n):
    for unit in ["","K","M","G","T"]:
        if abs(n) < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}P"

# ---- Alert ------------------------------------------------------------------
def run_alert_cmd(cmd, msg):
    if not cmd: return
    try:
        # pass message via env var ALERT_MSG
        env = os.environ.copy()
        env["ALERT_MSG"] = msg
        subprocess.Popen(cmd if isinstance(cmd, list) else shlex.split(cmd), env=env)
    except Exception as e:
        print(f"[alert-cmd error] {e}", file=sys.stderr)

# ---- Main loop --------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Watch OOM risks and resource spikes.")
    ag = ap.add_mutually_exclusive_group(required=True)
    ag.add_argument("--pid", type=int, help="Target process PID to watch")
    ag.add_argument("--proc-regex", type=str, help="Regex to match process name/cmdline")

    ap.add_argument("--interval", type=float, default=5.0, help="Sampling interval (s)")
    ap.add_argument("--log", type=str, default="oom_watch.csv", help="CSV log path")
    ap.add_argument("--alert-cmd", type=str, default="", help="Cmd to run on alert (uses env ALERT_MSG)")
    ap.add_argument("--sys-mem-thresh", type=float, default=90.0, help="System memory %% alert threshold")
    ap.add_argument("--proc-mem-thresh-mb", type=float, default=300*1024, help="Process RSS MB alert threshold")
    ap.add_argument("--gpu-mem-thresh-pct", type=float, default=95.0, help="Per-GPU memory %% alert threshold")
    ap.add_argument("--disk-path", type=str, default="/", help="Path to check disk usage")
    ap.add_argument("--disk-thresh", type=float, default=92.0, help="Disk usage %% alert threshold")
    ap.add_argument("--capture-dmesg", action="store_true", help="On alert, dump `dmesg | tail -n 50`")
    args = ap.parse_args()

    # Resolve PID
    pid = args.pid
    if args.proc_regex:
        pids = find_pid_by_regex(args.proc_regex)
        if not pids:
            print(f"[watch] No process matches regex: {args.proc_regex}", file=sys.stderr); sys.exit(2)
        pid = pids[0]
        print(f"[watch] Matched PID={pid} for regex: {args.proc_regex}")

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"[watch] PID {pid} not found", file=sys.stderr); sys.exit(2)

    # Prepare CSV
    new_file = not os.path.exists(args.log)
    f = open(args.log, "a", newline="")
    w = csv.writer(f)
    if new_file:
        w.writerow([
            "time_iso","pid","proc_mem_mb","proc_cpu_pct",
            "sys_mem_pct","disk_pct","gpu_idx","gpu_mem_used_mb","gpu_mem_total_mb","gpu_util_pct"
        ])
        f.flush()

    print(f"[watch] logging to {args.log} | interval={args.interval}s | thresholds:"
          f" sys_mem>{args.sys_mem_thresh}%, proc_mem>{args.proc_mem_thresh_mb}MB,"
          f" gpu_mem>{args.gpu_mem_thresh_pct}%, disk>{args.disk_thresh}%")

    # Smooth CPU %
    proc.cpu_percent(None)

    while True:
        try:
            now = datetime.now().isoformat(timespec="seconds")
            # System mem & disk
            sys_mem = psutil.virtual_memory().percent
            disk_pct = psutil.disk_usage(args.disk_path).percent

            # Process stats
            proc_mem_mb = proc.memory_info().rss / (1024**2)
            proc_cpu = proc.cpu_percent(None)

            # GPU stats
            gstats = gpu_stats() or [{"index": -1, "mem_used": 0.0, "mem_total": 0.0, "util": 0.0}]

            # Log rows per GPU
            for gs in gstats:
                w.writerow([now, pid, f"{proc_mem_mb:.0f}", f"{proc_cpu:.1f}",
                            f"{sys_mem:.1f}", f"{disk_pct:.1f}", gs["index"],
                            f"{gs['mem_used']:.0f}", f"{gs['mem_total']:.0f}", f"{gs['util']:.1f}"])
            f.flush()

            # Alerts
            alerts = []
            if sys_mem >= args.sys_mem_thresh:
                alerts.append(f"System memory {sys_mem:.1f}% >= {args.sys_mem_thresh}%")
            if proc_mem_mb >= args.proc_mem_thresh_mb:
                alerts.append(f"Process RSS {proc_mem_mb:.0f}MB >= {args.proc_mem_thresh_mb:.0f}MB (PID {pid})")
            if disk_pct >= args.disk_thresh:
                alerts.append(f"Disk {args.disk_path} usage {disk_pct:.1f}% >= {args.disk_thresh}%")
            for gs in gstats:
                if gs["mem_total"] > 0:
                    pct = gs["mem_used"] / gs["mem_total"] * 100.0
                    if pct >= args.gpu_mem_thresh_pct:
                        alerts.append(f"GPU{gs['index']} mem {pct:.1f}% (used {gs['mem_used']:.0f}MB / total {gs['mem_total']:.0f}MB)")

            if alerts:
                msg = f"[ALERT {now}] " + " | ".join(alerts)
                print("\033[91m" + msg + "\033[0m", file=sys.stderr)
                if args.capture_dmesg:
                    try:
                        tail = subprocess.check_output(["bash","-lc","dmesg -T | tail -n 50"], text=True, stderr=subprocess.STDOUT)
                        print("--- dmesg tail ---\n"+tail, file=sys.stderr)
                    except Exception as e:
                        print(f"[dmesg error] {e}", file=sys.stderr)
                run_alert_cmd(args.alert_cmd, msg)

            time.sleep(args.interval)

        except psutil.NoSuchProcess:
            print(f"[watch] PID {pid} ended.", file=sys.stderr)
            break
        except KeyboardInterrupt:
            print("[watch] stopped by user.")
            break
        except Exception as e:
            print(f"[watch] error: {e}", file=sys.stderr)
            time.sleep(max(1.0, args.interval))

if __name__ == "__main__":
    main()
