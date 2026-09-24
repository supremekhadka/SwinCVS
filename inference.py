"""
Run SwinCVS inference on either the Endoscapes dataset (original pipeline,
with ground-truth metrics) or the SAFE dataset (keyframe-flag csv, no
ground-truth metrics assumed - confidences only), selected via --dataset.

Frozen vs. E2E mode, the output directory, and which evaluation to run are
all CLI flags so one config per dataset can be reused across runs:

    python3 inference.py --dataset endoscapes --config_path config/infer.yaml \
        --mode frozen --output_dir results/endoscapes_inference_01 --eval inference

    python3 inference.py --dataset endoscapes --config_path config/infer.yaml \
        --mode frozen --output_dir results/endoscapes_throughput_01 --eval throughput --throughput_level image

    python3 inference.py --dataset safe --config_path config/infer_safe.yaml \
        --mode frozen --output_dir results/safe_inference_01 --eval inference

--eval inference  -> result.csv (endoscapes: also metrics.json with mAP / balanced accuracy)
--eval throughput -> throughput.csv (--throughput_level image, per-frame, forces batch size 1) or
                     throughput_video.csv (--throughput_level video, timed with CUDA events,
                     synchronized once per video, forces batch size 1)

Inference and throughput are always run as separate invocations - they are
different measurements and are never produced in one pass.

Both modes read from the same VIDEO_ROOT/CSV_PATH: for SAFE this is the
1fps-sampled frames, since each prediction needs the 4 frames preceding the
current one to build its input sequence, and 5fps sampling wouldn't give a
valid contiguous window.
"""

print("Importing libraries...")
# Standard library imports
import argparse
import json
import time
from pathlib import Path
import warnings

# Third-party imports
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Local imports
from scripts.f_environment import get_config, set_deterministic_behaviour
from scripts.f_build import build_inference_model
from scripts.f_metrics import get_map, get_balanced_accuracies

# Endoscapes pipeline (untouched)
from scripts.f_dataset import get_inference_dataset, get_inference_dataloader

# SAFE pipeline
from scripts.f_dataset_safe import get_safe_inference_dataset, get_safe_inference_dataloader

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SwinCVS inference")
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["endoscapes", "safe"],
        help="Which dataset/csv format this run is for.",
    )
    parser.add_argument(
        "--config_path", type=str, required=True,
        help="Path to inference config YAML (config/infer.yaml for endoscapes, config/infer_safe.yaml for safe).",
    )
    parser.add_argument(
        "--mode", type=str, required=True, choices=["e2e", "frozen"],
        help="e2e = un-frozen backbone weights (config.WEIGHTS_E2E); frozen = frozen backbone weights (config.WEIGHTS_FROZEN).",
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Override the weights file to load (defaults to config.WEIGHTS_E2E / WEIGHTS_FROZEN based on --mode). "
             "May be an absolute/relative path or a filename inside ./weights/",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write outputs into. Created if it doesn't exist.",
    )
    parser.add_argument(
        "--eval", type=str, required=True, choices=["inference", "throughput"],
        help="inference -> result.csv (+ metrics.json for endoscapes); throughput -> throughput.csv/throughput_video.csv. "
             "Run each as a separate invocation.",
    )
    parser.add_argument(
        "--throughput_level", type=str, default="image", choices=["image", "video"],
        help="Only used with --eval throughput. image -> synchronize CUDA every frame, write throughput.csv "
             "(default). video -> synchronize once per video via CUDA events, write throughput_video.csv.",
    )
    parser.add_argument(
        "--warmup", type=int, default=0,
        help="Run this many forward passes (batch size 1) before the timed/evaluated run, to warm up "
             "CUDA kernels and caches. Images used for warmup are NOT excluded afterwards - the full "
             "evaluation still runs over every row, including whichever ones were used to warm up.",
    )
    # Optional per-dataset path overrides, so the config doesn't need editing between runs
    parser.add_argument("--csv_path", type=str, default=None, help="Override config.CSV_PATH")
    parser.add_argument("--images_path", type=str, default=None, help="Endoscapes only: override config.IMAGES_PATH")
    parser.add_argument("--video_root", type=str, default=None, help="Safe only: override config.VIDEO_ROOT")
    return parser.parse_args()


def resolve_weights_path(config, args):
    if args.weights is not None:
        weights_file = args.weights
    else:
        weights_file = config.WEIGHTS_E2E if args.mode == "e2e" else config.WEIGHTS_FROZEN

    weights_path = Path(weights_file)
    if not weights_path.is_file():
        weights_path = Path("weights") / weights_file
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Could not find weights file '{weights_file}' (looked for it as given, and under ./weights/)."
        )
    return weights_path


def load_config(args):
    config = get_config(args.config_path, mode="inference")
    config.defrost()
    if args.csv_path is not None:
        config.CSV_PATH = args.csv_path
    if args.dataset == "endoscapes" and args.images_path is not None:
        config.IMAGES_PATH = args.images_path
    if args.dataset == "safe" and args.video_root is not None:
        config.VIDEO_ROOT = args.video_root

    config.MODEL.E2E = (args.mode == "e2e")
    config.MODEL.INFERENCE = True
    config.freeze()

    if config.CSV_PATH is None:
        raise ValueError("CSV_PATH must be set (via config or --csv_path).")
    if args.dataset == "endoscapes" and config.IMAGES_PATH is None:
        raise ValueError("IMAGES_PATH must be set (via config or --images_path).")
    if args.dataset == "safe" and config.VIDEO_ROOT is None:
        raise ValueError("VIDEO_ROOT must be set (via config or --video_root).")

    return config


def run_inference_loop(model, dataloader, device, has_targets, do_throughput, meta_df=None):
    """
    Shared timing/forward-pass loop for both datasets.
    has_targets=True: dataloader yields (samples, targets); targets are collected too (endoscapes).
    has_targets=False: dataloader yields samples only (safe).
    """
    all_probs = []
    all_targets = [] if has_targets else None
    throughput_rows = []

    len_dataloader = len(dataloader)
    row_cursor = 0
    data_iter = iter(dataloader)
    with torch.inference_mode():
        for step in range(len_dataloader):
            print(f"Processing batch: {step + 1:04}/{len_dataloader:04}", end="\r")

            t_pre_start = time.perf_counter()
            batch = next(data_iter)  # preprocessing happens here (num_workers=0)
            if has_targets:
                samples, targets = batch
            else:
                samples, targets = batch, None
            samples = samples.to(device, non_blocking=True)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t_model_start = time.perf_counter()

            outputs = model(samples)

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t_model_end = time.perf_counter()

            probs = torch.sigmoid(outputs)
            probs_cpu = probs.to("cpu")
            t_post_end = time.perf_counter()

            all_probs.append(probs_cpu)
            if has_targets:
                all_targets.append(targets.to("cpu"))

            if do_throughput:
                inference_time_ms = (t_model_end - t_model_start) * 1000.0
                latency_time_ms = (t_post_end - t_pre_start) * 1000.0
                n = probs_cpu.shape[0]
                for i in range(n):
                    meta_row = meta_df.iloc[row_cursor + i]
                    throughput_rows.append(
                        {
                            "vid_id": meta_row.get("vid_id"),
                            "vid": meta_row.get("vid"),
                            "frame": meta_row.get("frame"),
                            # batch_size is forced to 1 whenever throughput is measured,
                            # so these are true per-sample timings.
                            "inference_time_ms": round(inference_time_ms, 3),
                            "latency_time_ms": round(latency_time_ms, 3),
                        }
                    )
            row_cursor += probs_cpu.shape[0]

    print()
    all_probs = torch.cat(all_probs, dim=0)
    if has_targets:
        all_targets = torch.cat(all_targets, dim=0)
    throughput_df = pd.DataFrame(throughput_rows) if do_throughput else None
    return all_probs, all_targets, throughput_df


def warmup_model(model, dataset, device, n_warmup):
    """
    Run `n_warmup` forward passes (batch size 1, untimed, predictions discarded)
    before the real evaluation loop, to warm up CUDA kernels/caches so the
    first few timed samples aren't penalised by one-off initialisation cost.

    Deliberately reuses samples straight out of `dataset` rather than dummy
    tensors, so the warmup forward passes see real images end-to-end. Those
    same samples are NOT dropped from the evaluation afterwards - the run
    that follows still iterates over the whole dataset.
    """
    if n_warmup <= 0:
        return

    n_warmup = min(n_warmup, len(dataset))
    print(f"Warming up with {n_warmup} sample(s)...")
    warmup_loader = DataLoader(
        Subset(dataset, list(range(n_warmup))),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    with torch.inference_mode():
        for batch in warmup_loader:
            samples = batch[0] if isinstance(batch, (list, tuple)) else batch
            samples = samples.to(device, non_blocking=True)
            model(samples)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    print("Warmup complete.")


def run_throughput_video_level(model, dataset, device, meta_df, has_targets):
    """
    Video-level throughput: times each frame's forward pass with CUDA events
    (no per-frame synchronize), synchronizing only once per video. Returns one
    row per `vid` with total and per-frame-amortized timings, matching the
    throughput_video.csv format used by the cvs/CVS-AdaptNet repos.
    """
    is_cuda = device.startswith("cuda")
    rows = []
    for vid, group in meta_df.groupby("vid", sort=False):
        indices = group.index.tolist()
        vid_id = group["vid_id"].iloc[0] if "vid_id" in group.columns else None

        start_events, end_events = [], []
        t_video_start = time.perf_counter()
        with torch.inference_mode():
            for idx in indices:
                item = dataset[idx]
                sample = item[0] if has_targets else item
                sample = sample.unsqueeze(0).to(device, non_blocking=True)

                if is_cuda:
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                    start_evt.record()
                    model(sample)
                    end_evt.record()
                    start_events.append(start_evt)
                    end_events.append(end_evt)
                else:
                    t0 = time.perf_counter()
                    model(sample)
                    start_events.append(t0)
                    end_events.append(time.perf_counter())

        if is_cuda:
            torch.cuda.synchronize()
            inference_times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        else:
            inference_times_ms = [(e - s) * 1000.0 for s, e in zip(start_events, end_events)]
        t_video_end = time.perf_counter()

        num_frames = len(indices)
        total_inference_time_ms = sum(inference_times_ms)
        total_latency_time_ms = (t_video_end - t_video_start) * 1000.0
        rows.append(
            {
                "vid_id": vid_id,
                "vid": vid,
                "num_frames": num_frames,
                "inference_time_ms": round(total_inference_time_ms, 3),
                "latency_time_ms": round(total_latency_time_ms, 3),
                "frame_inference_time_ms": round(total_inference_time_ms / num_frames, 3),
                "frame_latency_time_ms": round(total_latency_time_ms / num_frames, 3),
            }
        )
    return pd.DataFrame(rows)


def write_throughput(args, output_dir, model, dataset, device, dataloader, meta_df, has_targets):
    if args.throughput_level == "image":
        _, _, throughput_df = run_inference_loop(
            model, dataloader, device, has_targets=has_targets, do_throughput=True, meta_df=meta_df
        )
        throughput_path = output_dir / "throughput.csv"
        throughput_df.to_csv(throughput_path, index=False)
        print(f"Throughput saved to: {throughput_path}")
        print(
            f"Mean inference time: {throughput_df['inference_time_ms'].mean():.2f} ms | "
            f"Mean latency: {throughput_df['latency_time_ms'].mean():.2f} ms"
        )
    else:
        throughput_video_df = run_throughput_video_level(model, dataset, device, meta_df, has_targets)
        throughput_path = output_dir / "throughput_video.csv"
        throughput_video_df.to_csv(throughput_path, index=False)
        print(f"Throughput saved to: {throughput_path}")
        print(
            f"Mean per-frame inference time: {throughput_video_df['frame_inference_time_ms'].mean():.2f} ms | "
            f"Mean per-frame latency: {throughput_video_df['frame_latency_time_ms'].mean():.2f} ms"
        )


def run_endoscapes(config, args, model, device, output_dir):
    dataset = get_inference_dataset(config)
    warmup_model(model, dataset, device, args.warmup)

    input_df = pd.read_csv(config.CSV_PATH)
    last_index = (len(input_df) // 5) * 5
    label_rows = input_df.iloc[4:last_index:5].reset_index(drop=True)  # row index 4, 9, 14, ...

    meta_df = label_rows[["vid", "frame"]].copy()
    if "vid_id" in input_df.columns:
        meta_df["vid_id"] = input_df.iloc[4:last_index:5]["vid_id"].reset_index(drop=True)

    if args.eval == "inference":
        dataloader = get_inference_dataloader(
            type("C", (), {"BATCH_SIZE": config.BATCH_SIZE})(), dataset
        )
        probs, targets, _ = run_inference_loop(
            model, dataloader, device, has_targets=True, do_throughput=False
        )
        preds = torch.round(probs)

        (C1_bacc, C2_bacc, C3_bacc, total_bacc) = get_balanced_accuracies([targets], [preds])
        C1_ap, C2_ap, C3_ap, mAP = get_map([targets], [probs])

        metrics = {
            "avg_bal_acc": round(total_bacc, 4),
            "C1_bacc": round(C1_bacc, 4),
            "C2_bacc": round(C2_bacc, 4),
            "C3_bacc": round(C3_bacc, 4),
            "avg_map": round(mAP, 4),
            "C1_map": round(C1_ap, 4),
            "C2_map": round(C2_ap, 4),
            "C3_map": round(C3_ap, 4),
        }
        print("\nTesting results:", metrics)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        result_df = label_rows.copy()
        result_df["Conf_C1"] = probs[:, 0].tolist()
        result_df["Conf_C2"] = probs[:, 1].tolist()
        result_df["Conf_C3"] = probs[:, 2].tolist()
        result_path = output_dir / "result.csv"
        result_df.to_csv(result_path, index=False)
        print(f"Predictions saved to: {result_path}")
        print(f"Metrics saved to: {output_dir / 'metrics.json'}")
    else:
        dataloader = get_inference_dataloader(type("C", (), {"BATCH_SIZE": 1})(), dataset)
        write_throughput(args, output_dir, model, dataset, device, dataloader, meta_df, has_targets=True)


def run_safe(config, args, model, device, output_dir):
    dataset, meta_df = get_safe_inference_dataset(config)
    if len(dataset) == 0:
        raise RuntimeError("No usable 5-frame keyframe sequences were found in the csv.")
    warmup_model(model, dataset, device, args.warmup)

    if args.eval == "inference":
        dataloader = get_safe_inference_dataloader(config, dataset, batch_size=config.BATCH_SIZE, num_workers=0)
        probs, _, _ = run_inference_loop(
            model, dataloader, device, has_targets=False, do_throughput=False
        )
        assert probs.shape[0] == len(meta_df), "Prediction / metadata row count mismatch"

        result_df = meta_df.copy()
        result_df["Conf_C1"] = probs[:, 0].tolist()
        result_df["Conf_C2"] = probs[:, 1].tolist()
        result_df["Conf_C3"] = probs[:, 2].tolist()
        result_path = output_dir / "result.csv"
        result_df.to_csv(result_path, index=False)
        print(f"Predictions saved to: {result_path}")
    else:
        dataloader = get_safe_inference_dataloader(config, dataset, batch_size=1, num_workers=0)
        write_throughput(args, output_dir, model, dataset, device, dataloader, meta_df, has_targets=False)


def main():
    args = parse_args()

    config = load_config(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_deterministic_behaviour(config.SEED)

    device = f"cuda:{config.CUDA_ID}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset} | Mode: {args.mode}")

    weights_path = resolve_weights_path(config, args)
    model = build_inference_model(config)
    print("Full model initialised successfully!\n")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    print(f"Trained SwinCVS weights loaded successfully for INFERENCE - path: {weights_path}")
    model.to(device)
    model.eval()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    print(f"\nRunning {args.eval}" + (f" ({args.throughput_level}-level)" if args.eval == "throughput" else ""))
    if args.dataset == "endoscapes":
        run_endoscapes(config, args, model, device, output_dir)
    else:
        run_safe(config, args, model, device, output_dir)


if __name__ == "__main__":
    main()