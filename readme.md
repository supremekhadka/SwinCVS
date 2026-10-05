# SwinCVS: A Unified Approach to Classifying Critical View of Safety Structures in Laparoscopic Cholecystectomy

**Authors**:
Franciszek Nowak, Evangelos B. Mazomenos, Brian Davidson, Matthew J. Clarkson

This repository is a fork of [franeknowak/SwinCVS](https://github.com/franeknowak/SwinCVS). The model and training pipeline are unchanged from the original publication (trained on the **Endoscapes2023** dataset). The original repo could only run inference on Endoscapes-format annotations. This fork extends it to run inference on our **SAFE** dataset and adds a separate script for fine-tuning on SAFE (`train_safe.py`, see [Fine-tuning on SAFE](#fine-tuning-on-safe)). The Endoscapes pipeline and training script are unchanged apart from small metric fixes.

---

## Overview

This repository provides code necessary for reproduction of the SwinCVS publication. The work proposes a SwinV2+LSTM based architecture called SwinCVS, to classify three Critical View of Safety (CVS) criteria from an open access Endoscapes2023 dataset.

## Implemented models

- **SwinV2 Backbone**: Pure SwinV2 backbone. Can be run on random weights or initialised using provided ImageNet weights.
- **SwinCVS (E2E, with multiclassifier)**: SwinCVS with end-to-end training and multiclassifier. Backbone weights initialised on ImageNet.
- **SwinCVS (E2E, without multiclassifier)**: SwinCVS with end-to-end training, but without multiclassifier. Backbone weights initialised on ImageNet.
- **SwinCVS (Frozen, without multiclassifier)**: SwinCVS where the image encoding backbone is frozen. Suggested backbone weights pretrained on Endoscapes.

All released weights were **trained on Endoscapes2023 only**. Running inference on SAFE data (below) is a distribution-shift question the weights were not fine-tuned for — treat SAFE results accordingly.

## Installation

- Clone this repository
- Confirm you have cuda enabled. In console type nvidia-smi. Our driver API details are:
NVIDIA-SMI 550.120 | Driver Version: 550.120 | CUDA Version: 12.4
- Install runtime API cuda 12.1 - Remember to add to path!
- Install dependencies. You may use any python version above 3.9 and PyTorch that supports your cuda version:
```
conda create --name swincvs python=3.12
conda activate swincvs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
```
- Download the model weights:
  ```bash
  python3 download_weights.py
  ```
  This fetches the same weights zip that `SwinCVS.py` and `inference.py` would otherwise download implicitly on first run (see `verify_results_weights_folder` in `scripts/f_environment.py`), plus `SwinCVS_frozen_ENDP_sd5_bestMAP.pt` from [Hugging Face](https://huggingface.co/supremekhadka/SwinCVS), and extracts/places them into `weights/`. Running it during setup avoids a large, silent download the first time you kick off training or inference — those scripts still check for the weights and will download them if missing, but you shouldn't need to rely on that anymore.

### Setup on macOS (Apple silicon)

Target: MacBook Pro M5 Pro. `inference.py` runs on the GPU through
PyTorch's MPS backend, which is picked automatically when CUDA is absent.
Training (`SwinCVS.py`, `train_safe.py`) is CUDA-only and not supported here.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision   # plain PyPI; the macOS wheels include MPS
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
python3 download_weights.py
```

`run_safe_inference_all.py` writes to `outputs/pretrained/macbook_m5_pro/` on macOS.

Power comes from `powermetrics`, which needs root. The scripts call it with
`sudo -n` (no password prompt), so allow it once:

```bash
echo "$USER ALL=(root) NOPASSWD: /usr/bin/powermetrics" | sudo tee /etc/sudoers.d/powermetrics
sudo chmod 440 /etc/sudoers.d/powermetrics
```

Without this, a warning is printed and `power_w` stays empty; everything else
still runs.

On the Mac, `inference_time_ms` uses `torch.mps.Event` timing (the MPS
counterpart of the CUDA events), and the sync points use
`torch.mps.synchronize()`. `memory_mb` is `torch.mps.current_allocated_memory()`
(tensor memory held on the GPU, like `torch.cuda.memory_allocated`). MPS has no
peak counter, so `peak_memory_mb` is the maximum of the 0.2 s samples and of
per-module readings taken during the warm-up passes; keep `--warmup` above 0.
`power_w` is the `GPU Power` estimate from `powermetrics` (GPU only, modelled
by macOS rather than measured on a power rail). For comparable numbers, run
plugged in with Low Power Mode off.

### Setup on Jetson Orin Nano

The desktop install above (conda + a fixed pytorch-cuda build) doesn't apply on Jetson — PyTorch/torchvision there are tied to the JetPack version, not a generic CUDA version, so check your JetPack version first and match the Python and PyTorch versions to it rather than following the versions above.

- Check your JetPack version (`apt-cache show nvidia-jetpack` or `cat /etc/nv_tegra_release`), then look up the Python and PyTorch build it expects — JetPack ships a specific Python version and needs a matching PyTorch wheel from NVIDIA/PyTorch's Jetson index.
- For **JetPack 7.2**, that means **Python 3.12**, and PyTorch/torchvision installed via:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
  ```
- After that, install the rest of the dependencies as usual: `pip install -r requirements.txt`.
- If you're on a different JetPack version, don't reuse the `cu132` wheel above as-is — re-check the matching Python/CUDA combination for your JetPack release before installing.
- Then download the weights as above: `python3 download_weights.py`.

---

## Training

`SwinCVS.py` training is unchanged from the original repository and only runs on Endoscapes2023. To fine-tune on SAFE, see [Fine-tuning on SAFE](#fine-tuning-on-safe).

Script is run by executing `SwinCVS.py` from the root of the repository. Model and training parameters are set within `config/SwinCVS_config.yaml`. Settings that specify model selection are:
- `MODEL.LSTM`: `False` - just SwinV2 backbone training, `True` - SwinCVS = SwinV2 with LSTM
- `MODEL.E2E`: `False` - backbone weights frozen, `True` - end-to-end training
- `MODEL.MULTICLASSIFIER`: `False` - does not add an additional classifier after backbone, `True` - adds a classifier after backbone, before LSTM
- `MODEL.INFERENCE`: `False` - allows for training, `True` - skips all training, performs only testing on provided weights (via this script, not `inference.py`)
- `BACKBONE.PRETRAINED`: `str` - which backbone weights to load, ImageNet or Endoscapes

The script automatically downloads the Endoscapes dataset and the ImageNet/Endoscapes-pretrained backbone weights into the repo directory. If you already have the dataset downloaded, set `DATASET_DIR` in the config to the folder that contains the `endoscapes` subfolder.

After changing the config, run:
```bash
python3 SwinCVS.py --config_path config/SwinCVS_config.yaml
```

Outputs (per-epoch metrics, predictions, best-epoch weights) are written to `results/` and `weights/` under the repo root, named from the model settings (e.g. `SwinCVS_frozen_ENDP_sd5`).

---

## Inference

Inference is handled by `inference.py`, a single entrypoint for both supported datasets, selected with `--dataset`:

- `--dataset endoscapes` — the original Endoscapes evaluation-csv format, with ground-truth labels and reported balanced accuracy / mAP metrics.
- `--dataset safe` — our SAFE dataset format, keyframe-flag csv, no ground-truth labels assumed, confidences only.

Model mode (frozen vs. end-to-end backbone) and where outputs get written are always CLI arguments, not config settings, so one config per dataset can be reused across runs:

```bash
python3 inference.py \
  --dataset {endoscapes,safe} \
  --config_path <path to config> \
  --mode {e2e,frozen} \
  --output_dir <output directory>
```

Each run makes a single pass (batch size 1, over every video) and measures per-video throughput and resource usage in one of two sync modes (`--sync_mode frame` or `--sync_mode video`, see [Throughput modes](#throughput-modes)). It also writes predictions (`result.csv`, plus `metrics.json` for Endoscapes) unless `--throughput_only` is given. Inference is normally measured together with video sync, while frame sync is run throughput-only.

| Flag | Required | Meaning |
|---|---|---|
| `--dataset` | yes | `endoscapes` or `safe` — selects the csv format, image-path convention, and whether ground-truth metrics are computed. |
| `--config_path` | yes | `config/infer.yaml` for Endoscapes, `config/infer_safe.yaml` for SAFE. |
| `--mode` | yes | `e2e` (un-frozen backbone) or `frozen` (frozen backbone). Picks `WEIGHTS_E2E` / `WEIGHTS_FROZEN` from the config. |
| `--weights` | no | Override the weights file (path, or filename under `weights/`). Defaults to the config's `WEIGHTS_E2E`/`WEIGHTS_FROZEN` for the chosen `--mode`. |
| `--output_dir` | yes | Directory outputs are written to. Created automatically if it doesn't exist. |
| `--warmup` | no (default `0`) | Run this many forward passes (batch size 1, untimed) before the measured pass, to warm up CUDA kernels/caches. The warmup samples are **not** excluded afterwards — the full dataset is still processed and included in the outputs. Resource sampling covers the warmup. |
| `--csv_path` | no | Override `CSV_PATH` from the config. |
| `--images_path` | no | Endoscapes only — override `IMAGES_PATH` from the config. |
| `--video_root` | no | SAFE only — override `VIDEO_ROOT` from the config. |
| `--sync_mode` | no (default `video`) | Throughput sync mode: `frame` (sync after every frame, writes `throughput/framesync.csv`) or `video` (sync only at video boundaries, writes `throughput/videosync.csv`). |
| `--throughput_only` | no | Only measure throughput and resource usage; do not write `result.csv` / `metrics.json`. |

Inference always runs at batch size 1 so per-frame timing is exact; batching would only give an averaged number.

For SAFE, `VIDEO_ROOT`/`CSV_PATH` (`config/infer_safe.yaml`) are the 1fps-sampled frames and CSV. This differs from the `cvs`/`CVS-AdaptNet` repos, which use the 5fps frames: here, every prediction needs the 4 frames immediately preceding the keyframe to build its input sequence, and only `metadata_1fps.csv` marks keyframes (`is_ds_keyframe`), so 5fps sampling wouldn't give a valid contiguous window.

To run both modes on SAFE in one go, use `run_safe_inference_all.py`. It writes to `outputs/pretrained/<device>/safe/all/<mode>/` (`<device>` is auto-detected as `jetson_orin_nano`, `macbook_m5_pro` or `nitro5_1650ti`; override with `MACHINE_TAG`). Select what to run with `--throughput` and `--inference`:

```bash
./run_safe_inference_all.py                                # inference + frame and video sync (default)
./run_safe_inference_all.py --throughput                   # frame and video sync, no result.csv
./run_safe_inference_all.py --throughput frame             # frame sync only
./run_safe_inference_all.py --throughput video             # video sync only
./run_safe_inference_all.py --inference --throughput video # inference + video sync in one pass
./run_safe_inference_all.py --inference                    # inference (measured with video sync)
```

Inference always shares a pass with a throughput mode: video sync, or frame sync if `--throughput frame` is the only mode selected. Every other selected mode gets its own throughput-only pass.

### Examples

```bash
# Endoscapes, frozen backbone
python3 inference.py --dataset endoscapes --config_path config/infer.yaml \
  --mode frozen --output_dir results/endoscapes_01

# SAFE, end-to-end backbone, inference + video sync throughput
python3 inference.py --dataset safe --config_path config/infer_safe.yaml \
  --mode e2e --output_dir results/safe_01 --sync_mode video

# SAFE, end-to-end backbone, frame sync throughput only (no result.csv)
python3 inference.py --dataset safe --config_path config/infer_safe.yaml \
  --mode e2e --output_dir results/safe_01 --sync_mode frame --throughput_only

# SAFE, weights overridden at the CLI
python3 inference.py --dataset safe --config_path config/infer_safe.yaml \
  --mode frozen --weights SwinCVS_frozen_ENDP_sd5_bestMAP.pt \
  --output_dir results/safe_02

# SAFE, both modes, into outputs/pretrained/<device>/safe/all/{e2e,frozen}/
./run_safe_inference_all.py
```

### Outputs

Written to `--output_dir`:

```text
<output_dir>/
├── result.csv                  # not written with --throughput_only
├── metrics.json                # Endoscapes only, not written with --throughput_only
├── throughput/
│   ├── framesync.csv           # --sync_mode frame
│   └── videosync.csv           # --sync_mode video
└── resources/
    ├── framesync/{peak,resource}.csv
    └── videosync/{peak,resource}.csv
```

- **`result.csv`**: every input column preserved for each row evaluated, plus `Conf_C1`, `Conf_C2`, `Conf_C3` — the model's sigmoid confidences for each CVS criterion.
  - Endoscapes: one row per 5-frame-window label (as in the original pipeline).
  - SAFE: one row per `is_ds_keyframe == True` row that had a full, contiguous 5-frame window available (see format below); keyframes without one are dropped and reported in the run log.
- **`metrics.json`** (Endoscapes only): `avg_bal_acc`, `C1_bacc`/`C2_bacc`/`C3_bacc`, `avg_map`, `C1_map`/`C2_map`/`C3_map`, computed against ground truth.
- **`throughput/<mode>sync.csv`**: one row per frame (`framesync.csv`) or per `vid` (`videosync.csv`); see [Throughput modes](#throughput-modes) for the columns.
- **`resources/<mode>sync/peak.csv`**: one row, `peak_memory_mb`, `peak_power_w`.
- **`resources/<mode>sync/resource.csv`**: the sampled time series `time_s`, `memory_mb`, `power_w`, covering warm-up and the full run.

Resource usage is sampled in the background over the whole run (including
warm-up), GPU-only. Memory comes from CUDA's allocator counters
(`torch.cuda.memory_allocated`/`max_memory_allocated`), so it is only what
this process holds on the GPU, including on Jetson's unified memory. Power is
auto-detected: on Jetson (`tegrastats` present) it is the `VDD_CPU_GPU_CV`
rail, the closest proxy available on Orin Nano (GPU + CPU + deep learning
accelerator cores, not pure GPU power); on a desktop GPU (`nvidia-smi`
present) it is `power.draw`, which is GPU-only. If neither binary is on
`PATH`, power samples stay empty.

### Throughput modes

Both modes time each frame the same way:

- `inference_time_ms`: CUDA-event elapsed time around the model forward only (no pre/postprocessing), under `torch.inference_mode()`.
- `latency_time_ms`: `perf_counter` from the start of preprocessing (image load + transform + device transfer) to the end of postprocessing (sigmoid, copy to CPU) after a final CUDA sync.

`--sync_mode frame` synchronizes after every frame and reads that frame's event time right after. `throughput/framesync.csv` has one row per frame: `vid_id`, `vid`, `frame`, `inference_time_ms`, `latency_time_ms`. A row is one forward pass, i.e. one prediction, so it has the same rows as `result.csv`: each prediction takes a 5-frame window, so `frame` is the keyframe's number and both times cover the whole window. On SAFE only `is_ds_keyframe == True` rows with a full contiguous window get a row (the other 1fps rows are only window context); on Endoscapes there is one row per 5-frame labelled window. Warm-up passes are not logged; those samples are measured again in the full pass.

`--sync_mode video` records an event pair per frame without per-frame syncs (postprocessing runs after the last frame), and syncs only once before and once after each video. The inference total is the sum of the event times; the latency total is the whole-video `perf_counter` time. `throughput/videosync.csv` columns: `vid_id`, `vid`, `num_frames`, `inference_time_ms`, `latency_time_ms` (video totals), `frame_inference_time_ms`, `frame_latency_time_ms` (total / `num_frames`).

### Data formats

**Endoscapes** (`config/infer.yaml`, `CSV_PATH` + `IMAGES_PATH`): the original evaluation csv, `vid`/`frame`/`C1`/`C2`/`C3` columns, one row per frame, annotated on a fixed stride of 5. Images are read flat from `IMAGES_PATH` as `<vid>_<frame>.jpg`.

**SAFE** (`config/infer_safe.yaml`, `CSV_PATH` + `VIDEO_ROOT`): one row per frame, with an explicit `is_ds_keyframe` flag instead of a fixed stride:

```
vid_id,vid,frame,is_ds_keyframe,avg_cvs,C1,C2,C3,annotator_...
482f2643-...,2025-11-28_053936_de4b6d58-...,0001,False,,,,,
...
482f2643-...,2025-11-28_053936_de4b6d58-...,0005,True,"[0.67,0.0,0.0]",1.0,0.0,0.0,...
```

For every row with `is_ds_keyframe == True`, the 4 immediately preceding rows in the same `vid` (by frame number) are used to build the 5-frame input sequence; the frame numbers must be contiguous (`n-4 .. n`) or the keyframe is skipped. Images are read from `VIDEO_ROOT/<vid>/<frame>.jpg` (one sub-folder per video) — update `build_image_path()` in `scripts/f_dataset_safe.py` if your data is laid out differently. `C1`/`C2`/`C3`/`avg_cvs`/annotator columns, when present, are passed through to `result.csv` unused (no ground-truth metrics are computed for SAFE).

---

## Fine-tuning on SAFE

`train_safe.py` fine-tunes either released SwinCVS variant on SAFE. It is separate from `SwinCVS.py`, which still trains on Endoscapes only. Each variant has its own config:

| Variant | Config | Initialised from (`MODEL.INIT_WEIGHTS`) | Trained parameters |
|---|---|---|---|
| End-to-end | `config/SwinCVS_safe_finetune_e2e.yaml` | `SwinV2LSTM_e2e_raw_mc_V3_sd5_bestMAP.pt` | backbone, `fc_swin`, LSTM, `fc_lstm` |
| Frozen | `config/SwinCVS_safe_finetune_frozen.yaml` | `SwinCVS_frozen_ENDP_sd5_bestMAP.pt` | LSTM, `fc_lstm` (backbone frozen) |

```bash
python3 train_safe.py --config config/SwinCVS_safe_finetune_e2e.yaml
python3 train_safe.py --config config/SwinCVS_safe_finetune_frozen.yaml

# common overrides
python3 train_safe.py --config config/SwinCVS_safe_finetune_frozen.yaml \
  --data_root /path/to/SAFE --device cuda:0 --epochs 20 --batch_size 8 --num_workers 8 \
  --eval_test --opts TRAIN.OPTIMIZER.CLASSIFIER_LR 5e-5 DATA.AUGMENT.ENABLE False
```

**Initialisation.** The full SwinCVS checkpoint is loaded with a strict `load_state_dict`. A missing file, a missing or unexpected key, or a shape mismatch stops the run. `INIT_WEIGHTS` can be a path or a file name under `weights/`. `BACKBONE.PRETRAINED` (the Endoscapes path, which ignores load errors) is not used here.

**Data.** `DATA.ROOT` defaults to `../data/SAFE`. The script reads `labels/1fps/splits/{train,val,test}.csv` and the images at `images/1fps/<vid>/<vid>_<frame>.jpg`. There is one sample for each `is_ds_keyframe` row: the 5 consecutive frames that end at the keyframe, within the same video (the same window logic as SAFE inference). The targets are `C1`/`C2`/`C3`. If a split csv has no `is_ds_keyframe` or label columns, they are joined from `labels/1fps/metadata.csv`. Preprocessing matches `inference.py --dataset safe` exactly: Resize(384, 683), CenterCrop(384), Normalize, then per-image min-max. The train split also gets light augmentation (`DATA.AUGMENT`: horizontal flip and mild colour jitter). The random parameters are drawn once per 5-frame sequence, so every frame in a sequence gets the same transform. Set `DATA.AUGMENT.ENABLE: False` to turn it off. The per-image min-max scaling mostly cancels brightness and contrast jitter.

**Loss.** `BCEWithLogitsLoss(pos_weight=TRAIN.POS_WEIGHT)`, one weight per criterion, set in the fine-tuning configs. With `POS_WEIGHT: null` it is computed from the train split as #neg / #pos. For E2E the loss is `alpha * L(fc_swin) + (1 - alpha) * L(fc_lstm)`. `TRAIN.MULTICLASSIFIER_ALPHA` defaults to 0.6, which is the value at the end of the original 10-epoch schedule. Set `MULTICLASSIFIER_ALPHA_DECAY: True` to use the original per-epoch decay instead. Validation uses the same criterion and alpha, and losses are averaged per sample.

**Optimisation.** AdamW with the original betas, eps and weight decay. AMP and gradient clipping (`CLIP_GRAD: 5`) match `SwinCVS.py`. The original training used constant learning rates of 1e-5 (encoder) and 1e-3 (LSTM/classifier). Fine-tuning uses lower defaults:

- E2E: `ENCODER_LR` 5e-6 (backbone + `fc_swin`), `CLASSIFIER_LR` 1e-4.
- Frozen: `CLASSIFIER_LR` 1e-4.
- Both: a 1-epoch linear warmup, then a per-step cosine decay to 1% (`TRAIN.LR_SCHEDULER`; `NAME: 'none'` keeps the rate constant).

Gradient accumulation (`TRAIN.ACCUMULATION_STEPS`) divides the loss by N and steps and zeroes the gradients only every N iterations.

**Memory (24GB GPU).** E2E defaults to `BACKBONE.USE_CHECKPOINT: True` (gradient checkpointing in the SwinV2 blocks), batch 2 and accumulation 2, which gives an effective batch of 4 (the original batch size). Frozen uses batch 8 without checkpointing, since no gradients flow through the backbone. `MODEL.FROZEN_BACKBONE_EVAL: True` keeps the frozen backbone in eval mode, which disables drop-path. The default `False` matches the original training.

**Logging (wandb).** The project is `safe-cvs-finetune` by default, and the run name is `EXPERIMENT_NAME` (`safe_finetune_e2e` / `safe_finetune_frozen`). Configure it under the `WANDB` block, or with `--wandb_project`, `--wandb_entity` and `--no_wandb`. The `WANDB_MODE` environment variable (`online`/`offline`/`disabled`) is respected.

| Key | When | Contents |
|---|---|---|
| `train/loss`, `train/loss_lstm`, `train/loss_swin` (E2E), `train/lr_*`, `train/grad_norm` | every iteration | |
| `train/epoch_loss`, `train/epoch_loss_lstm`, `train/epoch_loss_swin` (E2E) | every epoch | per-sample means |
| `val/loss`, `val/loss_lstm`, `val/loss_swin` (E2E) | every epoch | same criterion and mix as train |
| `val/mAP`, `val/AP_C1..C3`, `val/bacc_C1..C3`, `val/bacc_mean`, `val/recall_mean` | every epoch | from the LSTM head, the one used at inference |

No metrics are computed on the train split. `val/recall_mean` is the value that `scripts/f_metrics.py` returns, and `SwinCVS.py` reports it as `avg_bal_acc`. `val/bacc_mean` is the true mean of the three balanced accuracies.

**Outputs.** Each run gets a new directory, `<OUTPUT_DIR>/<run_name>_<timestamp>/` (default `work_dirs/safe_finetune/`). Nothing in `weights/` is written. The directory contains:

- `config.yaml`
- `results.json` (per-epoch losses, metrics and val probabilities)
- `best.pt` (best val mAP) and `last.pt`, both plain state dicts that `inference.py --weights <path>` can load (use `--mode e2e` or `--mode frozen` to match the variant)
- the wandb files

With `--eval_test` (or `TEST.ENABLE: True`), the best checkpoint is then evaluated on the test split, and `test_metrics.json` and `test_result.csv` are written.

**Debugging / smoke tests.** `--max_train_iters N` and `--max_val_iters N` cap the iterations per epoch. `--max_val_iters` also limits the test pass. If a capped val subset has no positives for a criterion, that criterion's AP and recall are `nan`, and the means are taken over the criteria that are defined.

```bash
WANDB_MODE=offline python3 train_safe.py --config config/SwinCVS_safe_finetune_frozen.yaml \
  --batch_size 1 --val_batch_size 2 --num_workers 2 --epochs 1 \
  --max_train_iters 6 --max_val_iters 54 --output_dir /tmp/safe_smoke
```

---

## Repository layout (additions in this fork)

- `download_weights.py` — explicit setup-time weights download (calls the same `verify_results_weights_folder` used implicitly by `SwinCVS.py` / `inference.py`).
- `inference.py` — unified inference entrypoint (`--dataset endoscapes|safe`), described above.
- `run_safe_inference_all.py` — runs `inference.py` on SAFE for `e2e` and `frozen`.
- `scripts/resource_monitor.py` — background GPU memory/power sampler used by `inference.py`.
- `scripts/f_dataset_safe.py` — SAFE-format dataset/dataloader construction (keyframe-based sequencing, `VIDEO_ROOT/<vid>/<frame>.jpg` image paths). Independent of `scripts/f_dataset.py`, which remains the untouched Endoscapes pipeline used by both `SwinCVS.py` (training) and `inference.py --dataset endoscapes`.
- `config/infer.yaml` — Endoscapes inference config. `MODEL.E2E` was removed in favour of the `--mode` CLI flag; `WEIGHTS_E2E`/`WEIGHTS_FROZEN` added so `--mode` can select the right weights file.
- `config/infer_safe.yaml` — SAFE inference config, same `--mode`/weights convention, `CSV_PATH` + `VIDEO_ROOT` instead of `CSV_PATH` + `IMAGES_PATH`.

- `train_safe.py`: SAFE fine-tuning entry point, described above.
- `config/SwinCVS_safe_finetune_e2e.yaml` and `config/SwinCVS_safe_finetune_frozen.yaml`: the SAFE fine-tuning configs.
- `scripts/f_dataset_safe.py` also contains the labelled SAFE fine-tuning datasets (`get_safe_datasets` and the sequence-consistent augmentation). The 5-frame window logic and the transform are shared helpers (`build_safe_sequences`, `build_safe_transform`), so inference and training preprocess frames identically. Inference behaviour is unchanged.
- `scripts/f_build.py`: `build_finetune_model` (strict full-checkpoint init via `MODEL.INIT_WEIGHTS`).
- `scripts/f_training_utils.py`: `NativeScalerWithGradNormCount(enabled=...)`, so AMP can be turned off. The default behaviour is unchanged.
- `scripts/f_metrics.py`: zero-division guards. A recall, specificity or AP with no positives or no negatives is `nan`, and the means skip it.
- `SwinCVS.py`: fixed the printed "Average balanced accuracy", which averaged `C1 + C1 + C3` instead of `C1 + C2 + C3`.

Everything else (`scripts/m_swinv2.py`, `scripts/m_swincvs.py`, `scripts/f_environment.py`, `config/SwinCVS_config.yaml`) is unmodified from upstream.

## Citation

If you use this work in your research, please cite the original paper:
Nowak, F., Mazomenos, E., Davidson, B., Clarkson, M., SwinCVS: A Unified Approach to Classifying Critical View of Safety Structures in Laparoscopic Cholecystectomy. Int J CARS (2025). https://doi.org/10.1007/s11548-025-03354-9