# SwinCVS: A Unified Approach to Classifying Critical View of Safety Structures in Laparoscopic Cholecystectomy

**Authors**:
Franciszek Nowak, Evangelos B. Mazomenos, Brian Davidson, Matthew J. Clarkson

This repository is a fork of [franeknowak/SwinCVS](https://github.com/franeknowak/SwinCVS). The model and training pipeline are unchanged from the original publication (trained on the **Endoscapes2023** dataset). The changes in this fork are limited to inference: the original repo could only run inference on Endoscapes-format annotations, and it has been extended here to also run on our **SAFE** dataset, without touching the Endoscapes pipeline or the training code.

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
- Install dependencies:<br>
conda create --name swincvs python=3.9.19<br>
conda activate swincvs<br>
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia<br>
pip install -r requirements.txt<br>
- Download the model weights:
  ```bash
  python3 download_weights.py
  ```
  This fetches the same weights zip that `SwinCVS.py` and `inference.py` would otherwise download implicitly on first run (see `verify_results_weights_folder` in `scripts/f_environment.py`), and extracts it into `weights/`. Running it during setup avoids a large, silent download the first time you kick off training or inference — those scripts still check for the weights and will download them if missing, but you shouldn't need to rely on that anymore.

### Setup on Jetson Orin Nano

The desktop install above (conda + a fixed pytorch-cuda build) doesn't apply on Jetson — PyTorch/torchvision there are tied to the JetPack version, not a generic CUDA version, so check your JetPack version first and match the Python and PyTorch versions to it rather than following the versions above.

- Check your JetPack version (`apt-cache show nvidia-jetpack` or `cat /etc/nv_tegra_release`), then look up the Python and PyTorch build it expects — JetPack ships a specific Python version and needs a matching PyTorch wheel from NVIDIA/PyTorch's Jetson index, not the desktop `pytorch-cuda` conda package.
- For **JetPack 7.2**, that means **Python 3.12**, and PyTorch/torchvision installed via:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
  ```
- After that, install the rest of the dependencies as usual: `pip install -r requirements.txt` (drop `torch`/`torchvision` from that step if `requirements.txt` pins different versions, so the Jetson-specific wheel above isn't overwritten).
- If you're on a different JetPack version, don't reuse the `cu132` wheel above as-is — re-check the matching Python/CUDA combination for your JetPack release before installing.
- Then download the weights as above: `python3 download_weights.py`.

---

## Training

Training is unchanged from the original repository and only ever runs on Endoscapes2023.

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
  --output_dir <output directory> \
  --eval {inference,throughput} \
  --throughput_level {image,video}
```

| Flag | Required | Meaning |
|---|---|---|
| `--dataset` | yes | `endoscapes` or `safe` — selects the csv format, image-path convention, and whether ground-truth metrics are computed. |
| `--config_path` | yes | `config/infer.yaml` for Endoscapes, `config/infer_safe.yaml` for SAFE. |
| `--mode` | yes | `e2e` (un-frozen backbone) or `frozen` (frozen backbone). Picks `WEIGHTS_E2E` / `WEIGHTS_FROZEN` from the config. |
| `--weights` | no | Override the weights file (path, or filename under `weights/`). Defaults to the config's `WEIGHTS_E2E`/`WEIGHTS_FROZEN` for the chosen `--mode`. |
| `--output_dir` | yes | Directory outputs are written to. Created automatically if it doesn't exist. |
| `--eval` | yes | `inference` → `result.csv` (+ `metrics.json` for Endoscapes); `throughput` → `throughput.csv` or `throughput_video.csv`. Run each as a separate invocation — inference and throughput are different measurements over different data paths and are never produced in one pass. |
| `--throughput_level` | no (default `image`) | Only used with `--eval throughput`. `image` synchronizes CUDA around every frame and writes `throughput.csv`. `video` groups frames by `vid`, times each frame's forward pass with CUDA events, and synchronizes only once per video, writing `throughput_video.csv`. |
| `--csv_path` | no | Override `CSV_PATH` from the config. |
| `--images_path` | no | Endoscapes only — override `IMAGES_PATH` from the config. |
| `--video_root` | no | SAFE only — override `VIDEO_ROOT` from the config. |

`--eval throughput` forces a batch size of 1 internally regardless of `--throughput_level`, since timing accurately requires it — batching would only give an averaged number.

For SAFE, both `--eval inference` and `--eval throughput` read from the same 1fps-sampled `VIDEO_ROOT`/`CSV_PATH` (`config/infer_safe.yaml`). This differs from the `cvs`/`CVS-AdaptNet` repos, which run inference on 5fps frames and throughput on 1fps frames: here, every prediction needs the 4 frames immediately preceding the current one to build its input sequence, so 5fps sampling wouldn't give a valid contiguous window for either measurement.

### Examples

```bash
# Endoscapes, frozen backbone, inference
python3 inference.py --dataset endoscapes --config_path config/infer.yaml \
  --mode frozen --output_dir results/endoscapes_inference_01 --eval inference

# Endoscapes, frozen backbone, image-level throughput
python3 inference.py --dataset endoscapes --config_path config/infer.yaml \
  --mode frozen --output_dir results/endoscapes_throughput_01 --eval throughput --throughput_level image

# SAFE, end-to-end backbone, inference
python3 inference.py --dataset safe --config_path config/infer_safe.yaml \
  --mode e2e --output_dir results/safe_inference_01 --eval inference

# SAFE, video-level throughput, weights overridden at the CLI
python3 inference.py --dataset safe --config_path config/infer_safe.yaml \
  --mode frozen --weights SwinCVS_frozen_ENDP_sd5_bestMAP.pt \
  --output_dir results/safe_throughput_01 --eval throughput --throughput_level video
```

### Outputs

- **`result.csv`** (`--eval inference`): every input column preserved for each row evaluated, plus `Conf_C1`, `Conf_C2`, `Conf_C3` — the model's sigmoid confidences for each CVS criterion.
  - Endoscapes: one row per 5-frame-window label (as in the original pipeline).
  - SAFE: one row per `is_ds_keyframe == True` row that had a full, contiguous 5-frame window available (see format below); keyframes without one are dropped and reported in the run log.
- **`metrics.json`** (Endoscapes only, `--eval inference`): `avg_bal_acc`, `C1_bacc`/`C2_bacc`/`C3_bacc`, `avg_map`, `C1_map`/`C2_map`/`C3_map`, computed against ground truth.
- **`throughput.csv`** (`--eval throughput --throughput_level image`): `vid_id`, `vid`, `frame`, `inference_time_ms`, `latency_time_ms` — one row per evaluated frame. `inference_time_ms` is the model forward pass only (GPU-synchronised every frame); `latency_time_ms` is end-to-end (image load + transform + device transfer + forward + sigmoid).
- **`throughput_video.csv`** (`--eval throughput --throughput_level video`): one row per `vid` with `vid_id`, `vid`, `num_frames`, `inference_time_ms`, `latency_time_ms`, `frame_inference_time_ms`, `frame_latency_time_ms` — the last two are the video totals divided by `num_frames`. `inference_time_ms` is measured per frame with CUDA events but only synchronized once at the end of the video; `latency_time_ms` covers the whole video end-to-end.

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

## Repository layout (additions in this fork)

- `download_weights.py` — explicit setup-time weights download (calls the same `verify_results_weights_folder` used implicitly by `SwinCVS.py` / `inference.py`).
- `inference.py` — unified inference entrypoint (`--dataset endoscapes|safe`), described above.
- `scripts/f_dataset_safe.py` — SAFE-format dataset/dataloader construction (keyframe-based sequencing, `VIDEO_ROOT/<vid>/<frame>.jpg` image paths). Independent of `scripts/f_dataset.py`, which remains the untouched Endoscapes pipeline used by both `SwinCVS.py` (training) and `inference.py --dataset endoscapes`.
- `config/infer.yaml` — Endoscapes inference config. `MODEL.E2E` was removed in favour of the `--mode` CLI flag; `WEIGHTS_E2E`/`WEIGHTS_FROZEN` added so `--mode` can select the right weights file.
- `config/infer_safe.yaml` — SAFE inference config, same `--mode`/weights convention, `CSV_PATH` + `VIDEO_ROOT` instead of `CSV_PATH` + `IMAGES_PATH`.

Everything else (`SwinCVS.py`, `scripts/m_swinv2.py`, `scripts/m_swincvs.py`, `scripts/f_build.py`, `scripts/f_training*.py`, `scripts/f_environment.py`, `scripts/f_metrics.py`, `config/SwinCVS_config.yaml`) is unmodified from upstream.

## Citation

If you use this work in your research, please cite the original paper:
Nowak, F., Mazomenos, E., Davidson, B., Clarkson, M., SwinCVS: A Unified Approach to Classifying Critical View of Safety Structures in Laparoscopic Cholecystectomy. Int J CARS (2025). https://doi.org/10.1007/s11548-025-03354-9