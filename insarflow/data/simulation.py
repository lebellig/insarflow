import os

from tqdm import tqdm
import numpy as np
import tifffile as tiff
from einops import rearrange
from torch.utils.data import Dataset
import logging

logger = logging.getLogger("Dataset")

# Number of files reserved from the front of the sorted train list for validation.
# Fixed constant — guarantees the same split across every run.
_VAL_SIZE = 100

# Last N files in the sorted list are always used as the test set.
_TEST_SIZE = 10000

# Known corrupted file to exclude.
_CORRUPTED = {"195033.tif"}


class SimulationInSARDataset(Dataset):
    """Synthetic InSAR dataset.

    Expected directory structure::

        root_dir/
        ├── interf/          # noisy wrapped interferograms (*.tif)
        └── originWrapped/   # clean wrapped phase (*.tif)

    Splits
    ------
    ``"train"``
        All files except the last ``_TEST_SIZE`` (10 000), minus the first
        ``_VAL_SIZE`` (100) which are reserved for validation.
    ``"val"``
        The first ``_VAL_SIZE`` files of the train pool.
        Deterministic across all runs (based on sorted file names).
    ``"test"``
        The last ``_TEST_SIZE`` files of the full sorted list.
    """

    def __init__(self, split: str, root_dir: str, img_size: int, transform=None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        self.split = split
        self.root_dir = root_dir
        self.img_size = img_size

        raw_dir   = os.path.join(root_dir, "interf")
        clean_dir = os.path.join(root_dir, "originWrapped")

        clean_files = set(os.listdir(clean_dir))
        raw_files   = set(os.listdir(raw_dir))
        all_files   = sorted(clean_files & raw_files - _CORRUPTED)

        train_pool = all_files[:-_TEST_SIZE]
        test_pool  = all_files[-_TEST_SIZE:]

        if split == "val":
            self.files = train_pool[:_VAL_SIZE]
        elif split == "train":
            self.files = train_pool[_VAL_SIZE:]
        else:  # test
            self.files = test_pool[:250]

        logger.info(f"SimulationInSARDataset | split={split} | {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        raw_path   = os.path.join(self.root_dir, "interf",        filename)
        clean_path = os.path.join(self.root_dir, "originWrapped", filename)
        try:
            raw   = rearrange(tiff.imread(raw_path),   "h w -> (h w)")
            clean = rearrange(tiff.imread(clean_path), "h w -> (h w)")
        except Exception as e:
            logger.info(f"Error reading file {filename}: {e}")
            raw   = np.zeros((self.img_size * self.img_size,), dtype=np.float32)
            clean = np.zeros((self.img_size * self.img_size,), dtype=np.float32)

        clean = (clean + np.pi) % (2 * np.pi)
        raw   = (raw   + np.pi) % (2 * np.pi)
        return {"x0": raw, "x1": clean}


if __name__ == "__main__":
    dataset = SimulationInSARDataset(
        split="test",
        root_dir="/share/DEEPLEARNING/datasets/InSARFlowSimulations",
        img_size=256,
    )

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]

    print(sample["x0"].shape, sample["x1"].shape)
