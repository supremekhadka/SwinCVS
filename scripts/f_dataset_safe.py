"""
Dataset utilities for running SwinCVS inference on the SAFE dataset.

Kept separate from `scripts/f_dataset.py` on purpose: that module is the
Endoscapes-specific pipeline (fixed stride-5 sequences, `IMAGES_PATH` +
"<vid>_<frame>.jpg" naming) and is left untouched. The SAFE CSV instead
marks the keyframe explicitly via `is_ds_keyframe`, and frames live under a
per-video folder (`VIDEO_ROOT/<vid>/<frame>.jpg`), so sequence construction
and image path resolution both need their own logic.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from torchvision import transforms
import pandas as pd


def build_image_path(video_root, vid, frame):
    """
    SAFE dataset convention: one sub-folder per video, frames named
    "<vid>_<frame>.jpg" inside it (frame is already zero-padded in the csv,
    e.g. "0005") -> VIDEO_ROOT/<vid>/<vid>_<frame>.jpg
    """
    return str(Path(video_root) / str(vid) / f"{vid}_{frame}.jpg")


def get_safe_inference_dataset(config):
    """
    Build 5-frame sequences ending on each `is_ds_keyframe == True` row.

    Unlike the Endoscapes pipeline (fixed stride of 5), the SAFE csv marks
    keyframes explicitly, so for every keyframe row we look back at the 4
    preceding rows *within the same video* and require the frame numbers to
    be contiguous (idx-4 .. idx). Keyframes without 4 valid preceding frames
    (e.g. at the very start of a video, or after a gap in the frame numbers)
    are skipped with a warning and dropped from the outputs.

    Returns
    -------
    dataset : SafeSwinCVS_Inference_Dataset
    meta_df : pd.DataFrame
        One row per usable sequence, carrying every original csv column for
        the keyframe row (so results can be re-merged 1:1), in the same
        order the dataset yields sequences.
    """
    csv_path = Path(config.CSV_PATH)
    if not (str(csv_path).endswith(".csv") and csv_path.is_file()):
        raise FileNotFoundError(f"Invalid CSV Path: {config.CSV_PATH}")

    df = pd.read_csv(csv_path)
    df["frame"] = df["frame"].astype(str).str.zfill(4)
    df["_frame_int"] = df["frame"].astype(int)

    sequences = []
    meta_rows = []
    skipped = 0

    for vid, vid_df in df.groupby("vid", sort=False):
        vid_df = vid_df.sort_values("_frame_int").reset_index(drop=True)
        keyframe_positions = vid_df.index[vid_df["is_ds_keyframe"] == True].tolist()

        for pos in keyframe_positions:
            if pos < 4:
                skipped += 1
                continue

            window = vid_df.iloc[pos - 4 : pos + 1]
            frame_ints = window["_frame_int"].tolist()
            # Require exactly consecutive frame numbers for a valid 5-frame sequence
            if frame_ints != list(range(frame_ints[0], frame_ints[0] + 5)):
                skipped += 1
                continue

            frames = window["frame"].tolist()
            sequences.append({"vid": vid, "frames": frames})
            meta_rows.append(vid_df.iloc[pos].drop(labels=["_frame_int"]).to_dict())

    if skipped:
        print(f"Warning: skipped {skipped} keyframe(s) without a full 5-frame preceding window.")

    meta_df = pd.DataFrame(meta_rows).reset_index(drop=True)

    mean, std = config.ENDOSCAPES_MEAN, config.ENDOSCAPES_STD
    transform_sequence = transforms.Compose(
        [
            transforms.Resize(tuple(config.RESIZE)),
            transforms.CenterCrop(config.CENTER_CROP),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
    )

    dataset = SafeSwinCVS_Inference_Dataset(
        sequences, transform_sequence, video_root=config.VIDEO_ROOT
    )

    return dataset, meta_df


class SafeSwinCVS_Inference_Dataset(Dataset):
    """5-frame sequence dataset for SAFE-format inference (no labels required)."""

    def __init__(self, sequences, transform_sequence, video_root):
        self.sequences = sequences
        self.transforms = transform_sequence
        self.video_root = video_root

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        vid = seq["vid"]
        paths = [build_image_path(self.video_root, vid, frame) for frame in seq["frames"]]

        image_list = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            image = self.transforms(image)
            image = (image - torch.min(image)) / (-torch.min(image) + torch.max(image))
            image_list.append(image)

        images = torch.stack(image_list)
        return images


def get_safe_inference_dataloader(config, dataset, batch_size=None, num_workers=0):
    """
    num_workers defaults to 0 so that preprocessing runs synchronously on the
    main thread - required for accurate end-to-end (pre-to-post-process)
    latency measurement in inference_safe.py.
    """
    bs = batch_size if batch_size is not None else config.BATCH_SIZE
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )