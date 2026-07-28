import os

import numpy as np
import tifffile as tiff
from einops import rearrange
from torch.utils.data import Dataset
import logging

logger = logging.getLogger("Dataset")

# Number of files reserved from the front of the sorted train list for validation.
# Fixed constant — guarantees the same split across every run.
_VAL_SIZE = 100


class MexicoDataset(Dataset):
    """Real InSAR dataset from Mexico.

    The underlying data lives in mirrored directory trees::

        root_dir/
        ├── raw/
        │   ├── train/   # noisy wrapped interferogram patches (*.tif)
        │   └── test/
        ├── clean/
        │   ├── train/   # clean wrapped interferogram patches (*.tif)
        │   └── test/
        ├── raw_image/
        │   ├── train/   # noisy SAR image patches — complex64 (*.npy)
        │   └── test/
        ├── clean_image/
        │   ├── train/   # clean SAR image patches — complex64 (*.npy)
        │   └── test/
        └── metadata/
            ├── train/   # per-patch metadata (*.txt), contains time_diff
            └── test/
    """

    def __init__(self, split: str, root_dir: str, img_size: int, transform=None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        self.split    = split
        self.root_dir = root_dir
        self.img_size = img_size

        # Both train and val share the raw/train + clean/train directories.
        # The test split lives in raw/test + clean/test.
        data_dir = "train" if split in ("train", "val") else "test"
        self.raw_dir         = os.path.join(root_dir, "raw",         data_dir)
        self.clean_dir       = os.path.join(root_dir, "clean",       data_dir)
        self.raw_image_dir   = os.path.join(root_dir, "raw_image",   data_dir)
        self.clean_image_dir = os.path.join(root_dir, "clean_image", data_dir)
        self.metadata_dir    = os.path.join(root_dir, "metadata",    data_dir)

        all_files   = sorted(os.listdir(self.clean_dir))

        if split == "test":
            self.files = all_files
        elif split == "val":
            self.files = all_files[:_VAL_SIZE]
        elif split == "train":
            self.files = all_files[_VAL_SIZE:]

        # shuffle the files
        np.random.seed(42)
        np.random.shuffle(self.files)

        logger.info(f"MexicoDataset | split={split} | {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _read_time_diff(meta_path: str) -> float:
        """Parse ``time_diff=<value>`` from a metadata text file."""
        with open(meta_path) as f:
            for line in f:
                if line.startswith("time_diff="):
                    return float(line.strip().split("=", 1)[1])
        return 0.0

    def __getitem__(self, idx):
        filename   = self.files[idx]
        raw_path   = os.path.join(self.raw_dir,   filename)
        clean_path = os.path.join(self.clean_dir, filename)
        flat = self.img_size * self.img_size
        try:
            raw   = rearrange(tiff.imread(raw_path),   "h w -> (h w)")
            clean = rearrange(tiff.imread(clean_path), "h w -> (h w)")
        except Exception as e:
            logger.info(f"Error reading file {filename}: {e}")
            raw   = np.zeros((flat,), dtype=np.float32)
            clean = np.zeros((flat,), dtype=np.float32)

        sample = {"x0": raw, "x1": clean}

        # For val / test, also return the SAR images and temporal baseline.
        if self.split in ("val", "test"):
            npy_filename   = filename.replace(".tif", ".npy")
            raw_img_path   = os.path.join(self.raw_image_dir,   npy_filename)
            clean_img_path = os.path.join(self.clean_image_dir, npy_filename)
            meta_filename  = filename.replace(".tif", ".txt")
            meta_path      = os.path.join(self.metadata_dir, meta_filename)
            try:
                raw_image   = rearrange(np.load(raw_img_path),   "h w -> (h w)")
                clean_image = rearrange(np.load(clean_img_path), "h w -> (h w)")
            except Exception as e:
                logger.info(f"Error reading image file {filename}: {e}")
                raw_image   = np.zeros((flat,), dtype=np.complex64)
                clean_image = np.zeros((flat,), dtype=np.complex64)

            time_diff = self._read_time_diff(meta_path)

            sample["clean_image"] = clean_image
            sample["raw_image"]   = raw_image
            sample["time_diff"]   = np.float32(time_diff)

        return sample
