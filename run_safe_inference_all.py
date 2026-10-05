#!/usr/bin/env python3
"""Runs inference.py over every SwinCVS mode (e2e, frozen), one run at a time. Each run is a
single pass that writes throughput/<sync_mode>sync.csv and
resources/<sync_mode>sync/ (peak.csv, resource.csv), plus result.csv when it
is the inference run, into one directory per mode:

    outputs/pretrained/<MACHINE_TAG>/safe/all/<mode>/

RUNS below lists every mode explicitly, so you can comment out or delete
individual lines to run a subset.

Usage: ./run_safe_inference_all.py [--throughput [{frame,video} ...]] [--inference]
With no arguments, runs inference plus both throughput modes. Inference is
measured in the video sync run (the frame sync run if only frame was asked
for), so it never costs an extra pass.
    ./run_safe_inference_all.py --throughput frame        # frame sync throughput only
    ./run_safe_inference_all.py --throughput video        # video sync throughput only
    ./run_safe_inference_all.py --inference --throughput video
The output-dir machine tag is auto-detected (jetson_orin_nano / macbook_m5_pro / nitro5_1650ti); override with: MACHINE_TAG=my_gpu ./run_safe_inference_all.py
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
# Jetson (L4T) ships /etc/nv_tegra_release; macOS is the MacBook M5 Pro; anything else is the Nitro 5 / GTX 1650 Ti.
if Path("/etc/nv_tegra_release").exists():
    DEFAULT_MACHINE_TAG = "jetson_orin_nano"
elif sys.platform == "darwin":
    DEFAULT_MACHINE_TAG = "macbook_m5_pro"
else:
    DEFAULT_MACHINE_TAG = "nitro5_1650ti"
MACHINE_TAG = os.environ.get("MACHINE_TAG", DEFAULT_MACHINE_TAG)
SPLIT = "all"
OUT_BASE = REPO_ROOT / "outputs" / "pretrained" / MACHINE_TAG / "safe" 

CONFIG = "config/infer_safe.yaml"  # SAFE 1fps csv/frames: SwinCVS needs the is_ds_keyframe
                                   # flag and 4 contiguous preceding frames per keyframe, which
                                   # only the 1fps csv provides (no 5fps equivalent).
WARMUP = "10"
SYNC_MODES = ("frame", "video")

RUNS = [
    "e2e",
    "frozen",
]

slug_dir = {
    "e2e": "swincvs_end-to-end",
    "frozen": "swincvs_frozen"
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference.py over every mode in RUNS. With no "
        "arguments, runs inference plus both throughput modes."
    )
    parser.add_argument(
        "--throughput",
        nargs="*",
        choices=SYNC_MODES,
        help="Throughput sync modes to measure (no values = both).",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Write result.csv (measured in the video sync run, or the frame sync run if only frame is selected).",
    )
    args = parser.parse_args()
    if args.throughput is None and not args.inference:
        args.throughput, args.inference = list(SYNC_MODES), True
    elif args.throughput == []:
        args.throughput = list(SYNC_MODES)
    return args


def plan_runs(sync_modes, inference):
    """(sync_mode, throughput_only) per inference.py call."""
    sync_modes = [mode for mode in SYNC_MODES if mode in (sync_modes or [])]
    runs = []
    if inference:
        runs.append(("frame" if sync_modes == ["frame"] else "video", False))
    runs += [(mode, True) for mode in sync_modes if mode not in (run[0] for run in runs)]
    return runs


def run(slug, sync_mode, throughput_only):
    out_dir = OUT_BASE / slug_dir[slug] / SPLIT
    out_dir.mkdir(parents=True, exist_ok=True)

    task = "throughput" if throughput_only else "inference + throughput"
    print(f"==> [{slug}] {task} + resource usage ({sync_mode} sync, 1fps, warmup={WARMUP})")
    cmd = [
        sys.executable, "inference.py",
        "--dataset", "safe",
        "--config_path", CONFIG,
        "--mode", slug,
        "--output_dir", str(out_dir),
        "--warmup", WARMUP,
        "--sync_mode", sync_mode,
    ]
    if throughput_only:
        cmd.append("--throughput_only")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


LOG_PATH = REPO_ROOT / "run_safe_inference_all.log"


def main():
    args = parse_args()
    runs = plan_runs(args.throughput, args.inference)
    failures = []
    with open(LOG_PATH, "w") as log:
        for slug in RUNS:
            for sync_mode, throughput_only in runs:
                try:
                    run(slug, sync_mode, throughput_only)
                except subprocess.CalledProcessError as e:
                    label = f"{slug} ({sync_mode} sync{', throughput only' if throughput_only else ''})"
                    msg = f"[{label}] FAILED (exit {e.returncode})"
                    print(f"!! {msg}", file=sys.stderr)
                    log.write(msg + "\n")
                    log.flush()
                    failures.append(label)

        if failures:
            log.write("\nFailed runs:\n")
            for slug in failures:
                log.write(f"  {slug}\n")

    if failures:
        print(f"\n{len(failures)} run(s) failed, see {LOG_PATH}:")
        for slug in failures:
            print(f"  {slug}")
    print(f"All runs done. Outputs under: {OUT_BASE}")


if __name__ == "__main__":
    main()
