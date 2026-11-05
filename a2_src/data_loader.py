from pathlib import Path
from typing import Tuple, Dict, Any
from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

from a2_src.config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    BATCH_SIZE,
    IMG_SIZE,
    CLASSES,
    DEVICE,
)
from a2_src.artaug import (
    ArtAugDatasetAugmentation,
    get_eval_transforms,
    generate_stress_environment,
    polish_low_exposure_frame,
)


class StressTestDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        exposure: float = 1.0,
        blur_kernel: int = 1,
        noise_level: float = 0.0,
        polish_frames: bool = False,
    ):
        self.image_folder = datasets.ImageFolder(str(root_dir))
        self.exposure = exposure
        self.blur_kernel = blur_kernel
        self.noise_level = noise_level
        self.polish_frames = polish_frames
        self.eval_transforms = get_eval_transforms()

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        pil_img, label = self.image_folder[idx]
        img_np = np.array(pil_img)

        if len(img_np.shape) == 2:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        stressed_bgr = generate_stress_environment(
            img_bgr,
            exposure=self.exposure,
            blur_kernel=self.blur_kernel,
            noise_level=self.noise_level,
        )

        if self.polish_frames and self.exposure < 0.7:
            stressed_bgr = polish_low_exposure_frame(stressed_bgr)

        stressed_rgb = cv2.cvtColor(stressed_bgr, cv2.COLOR_BGR2RGB)
        stressed_pil = Image.fromarray(stressed_rgb)

        tensor_img = self.eval_transforms(stressed_pil)
        return tensor_img, label


def get_dataloaders(use_artaug_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = ArtAugDatasetAugmentation(apply_artaug=use_artaug_train)
    eval_transform = get_eval_transforms()

    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=eval_transform)
    test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=eval_transform)

    pin_memory = DEVICE.type != "cpu"
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def get_stress_test_loader(
    exposure: float = 1.0,
    blur_kernel: int = 1,
    noise_level: float = 0.0,
    polish_frames: bool = False,
) -> DataLoader:
    dataset = StressTestDataset(
        root_dir=TEST_DIR,
        exposure=exposure,
        blur_kernel=blur_kernel,
        noise_level=noise_level,
        polish_frames=polish_frames,
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
