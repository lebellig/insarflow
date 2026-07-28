import os as os

from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
from einops import rearrange
from skimage.io import imsave
from tqdm import trange

# VARIABLES
CLEAN_PATH = "/lustre/fsn1/projects/rech/zpa/uia62vp/insar_dataset/clean_phases.npy"
RAW_PATH = "/lustre/fsn1/projects/rech/zpa/uia62vp/insar_dataset/raw_images.npy"
OUT_DIR = "/lustre/fsn1/projects/rech/zpa/uia62vp/insar_dataset/"


RAW_PATH = "/users/l/lebellig/insar_dataset/raw_images.npy"
CLEAN_PATH = "/users/l/lebellig/insar_dataset/clean_phases.npy"
OUT_DIR = "/users/l/lebellig/insar_dataset/"


IMG_SIZE = 256
DOWNSAMPLING = 4
OVERLAP = 0.0
DATASET_DIR = f"extended_insar_{IMG_SIZE}_ov{OVERLAP}_dwn{DOWNSAMPLING}"

# load raw images
raw_images = np.load(RAW_PATH)
raw_images = rearrange(raw_images, "h w b -> b h w")
pad_left, pad_right = 3, 4
raw_images = raw_images[
    :, 500 + pad_left : 4138 - pad_right, 500 + pad_left : 17209 - pad_right
]
print("Raw image loaded ", raw_images.shape)
processed_phases = np.load(CLEAN_PATH)
processed_phases = rearrange(processed_phases, "h w b -> b h w")
print("Processed phases loaded ", processed_phases.shape)
DATASET_PATH = os.path.join(OUT_DIR, DATASET_DIR)
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "clean"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "clean", "train"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "clean", "test"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "raw"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "raw", "train"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "raw", "test"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "clean_image"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "clean_image", "train"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "clean_image", "test"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "raw_image"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "raw_image", "train"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "raw_image", "test"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "metadata"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "metadata", "train"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "metadata", "test"), exist_ok=True)
print(processed_phases.shape)
# print(processed_phases.min(), processed_phases.max())

# apply downsampling
dwn_clean = processed_phases[:, ::DOWNSAMPLING, ::DOWNSAMPLING]
dwn_raw = raw_images[:, ::DOWNSAMPLING, ::DOWNSAMPLING]
# TEST RANGES
xtest_start, xtest_end = 2000 // DOWNSAMPLING, 3000 // DOWNSAMPLING
ytest_start, ytest_end = 6000 // DOWNSAMPLING, 7000 // DOWNSAMPLING
stride = int(IMG_SIZE * (1 - OVERLAP))
nphases = len(dwn_clean)
ncrops = 0
for i in trange(1, nphases):
    for j in trange(i + 1, nphases):
        raw_delta = np.angle(dwn_raw[i] * np.conj(dwn_raw[j])) % (2 * np.pi)
        clean_delta = (dwn_clean[i] - dwn_clean[j]) % (2 * np.pi)
        time_diff = j - i
        for x in range(0, raw_delta.shape[0] - IMG_SIZE, stride):
            for y in range(0, raw_delta.shape[1] - IMG_SIZE, stride):
                if (x > xtest_start and x < xtest_end) and (
                    y > ytest_start and y < ytest_end
                ):
                    split = "test"
                else:
                    split = "train"
                # delta crops
                raw_crop = raw_delta[x : x + IMG_SIZE, y : y + IMG_SIZE]
                clean_crop = clean_delta[x : x + IMG_SIZE, y : y + IMG_SIZE]
                # image crops
                raw_img_crop = dwn_raw[i][x : x + IMG_SIZE, y : y + IMG_SIZE]
                clean_img_crop = dwn_clean[i][x : x + IMG_SIZE, y : y + IMG_SIZE]
                # save deltas
                imsave(
                    os.path.join(DATASET_PATH, "raw", split, f"{x}_{y}_{i}_{j}.tif"),
                    raw_crop,
                )
                imsave(
                    os.path.join(DATASET_PATH, "clean", split, f"{x}_{y}_{i}_{j}.tif"),
                    clean_crop,
                )
                # save images (complex tensors)
                np.save(
                    os.path.join(DATASET_PATH, "raw_image", split, f"{x}_{y}_{i}_{j}.npy"),
                    raw_img_crop,
                )
                np.save(
                    os.path.join(DATASET_PATH, "clean_image", split, f"{x}_{y}_{i}_{j}.npy"),
                    clean_img_crop,
                )
                # save metadata
                meta_path = os.path.join(
                    DATASET_PATH, "metadata", split, f"{x}_{y}_{i}_{j}.txt"
                )
                with open(meta_path, "w") as f:
                    f.write(f"time_diff={time_diff}\n")
                ncrops += 1

print("Saved ", ncrops)
