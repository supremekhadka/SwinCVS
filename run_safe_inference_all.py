#!/usr/bin/env python3
"""Runs inference.py over every SwinCVS mode (e2e, frozen), one run at a time. Each run is a
single pass that writes result.csv, throughput.csv and resources/ (peak.csv,
resource.csv) into one directory per mode:

    outputs/pretrained/<MACHINE_TAG>/safe/all/<mode>/

RUNS below lists every mode explicitly, so you can comment out or delete
individual lines to run a subset.

Usage: ./run_safe_inference_all.py
The output-dir machine tag is auto-detected (jetson_orin_nano / nitro5_1650ti); override with: MACHINE_TAG=my_gpu ./run_safe_inference_all.py
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
# Jetson (L4T) ships /etc/nv_tegra_release; anything else is the Nitro 5 / GTX 1650 Ti.
DEFAULT_MACHINE_TAG = "jetson_orin_nano" if Path("/etc/nv_tegra_release").exists() else "nitro5_1650ti"
MACHINE_TAG = os.environ.get("MACHINE_TAG", DEFAULT_MACHINE_TAG)
OUT_BASE = REPO_ROOT / "outputs" / "pretrained" / MACHINE_TAG / "safe" / "all"

CONFIG = "config/infer_safe.yaml"  # SAFE 1fps csv/frames: SwinCVS needs the is_ds_keyframe
                                   # flag and 4 contiguous preceding frames per keyframe, which
                                   # only the 1fps csv provides (no 5fps equivalent).
WARMUP = "10"

RUNS = [
    "e2e",
    "frozen",
]


def run(slug):
    out_dir = OUT_BASE / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"==> [{slug}] inference + throughput + resource usage (1fps, warmup={WARMUP})")
    cmd = [
        sys.executable, "inference.py",
        "--dataset", "safe",
        "--config_path", CONFIG,
        "--mode", slug,
        "--output_dir", str(out_dir),
        "--warmup", WARMUP,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


LOG_PATH = REPO_ROOT / "run_safe_inference_all.log"


def main():
    failures = []
    with open(LOG_PATH, "w") as log:
        for slug in RUNS:
            try:
                run(slug)
            except subprocess.CalledProcessError as e:
                msg = f"[{slug}] FAILED (exit {e.returncode})"
                print(f"!! {msg}", file=sys.stderr)
                log.write(msg + "\n")
                log.flush()
                failures.append(slug)

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
