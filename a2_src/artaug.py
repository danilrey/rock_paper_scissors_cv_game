import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def polish_low_exposure_frame(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    mean_val = np.mean(l_channel) / 255.0
    if mean_val < 0.4:
        gamma = max(0.4, 0.8 + np.log(max(mean_val, 0.05)) * 0.15)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        l_enhanced = cv2.LUT(l_enhanced, table)

    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    polished = cv2.bilateralFilter(enhanced_bgr, d=5, sigmaColor=35, sigmaSpace=35)
    return polished


def apply_exposure_stress(image_bgr: np.ndarray, factor: float = 0.3) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_motion_blur_stress(image_bgr: np.ndarray, kernel_size: int = 15, angle: float = 45.0) -> np.ndarray:
    if kernel_size <= 1:
        return image_bgr
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    theta = np.deg2rad(angle)
    for i in range(kernel_size):
        offset = i - center
        x = int(center + offset * np.cos(theta))
        y = int(center + offset * np.sin(theta))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    k_sum = np.sum(kernel)
    if k_sum > 0:
        kernel /= k_sum
    else:
        kernel[center, center] = 1.0

    return cv2.filter2D(image_bgr, -1, kernel)


def apply_sensor_noise_stress(image_bgr: np.ndarray, intensity: float = 30.0) -> np.ndarray:
    if intensity <= 0:
        return image_bgr
    noise = np.random.normal(0, intensity, image_bgr.shape).astype(np.float32)
    noisy = np.clip(image_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def generate_stress_environment(
    image_bgr: np.ndarray,
    exposure: float = 1.0,
    blur_kernel: int = 1,
    noise_level: float = 0.0,
) -> np.ndarray:
    stressed = image_bgr.copy()
    if abs(exposure - 1.0) > 0.01:
        stressed = apply_exposure_stress(stressed, exposure)
    if blur_kernel > 1:
        stressed = apply_motion_blur_stress(stressed, blur_kernel)
    if noise_level > 0:
        stressed = apply_sensor_noise_stress(stressed, noise_level)
    return stressed


class ArtAugDatasetAugmentation:
    def __init__(self, apply_artaug: bool = True):
        self.apply_artaug = apply_artaug
        self.spatial_transforms = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
        ])
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1
        )
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, pil_image: Image.Image) -> torch.Tensor:
        img_np = np.array(pil_image)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        if self.apply_artaug and np.random.rand() < 0.45:
            factor = np.random.uniform(0.25, 1.75)
            img_bgr = apply_exposure_stress(img_bgr, factor)
            if factor < 0.6 and np.random.rand() < 0.5:
                img_bgr = polish_low_exposure_frame(img_bgr)

        if self.apply_artaug and np.random.rand() < 0.25:
            k = np.random.choice([5, 7, 9, 11])
            img_bgr = apply_motion_blur_stress(img_bgr, kernel_size=k)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_transformed = Image.fromarray(img_rgb)

        pil_transformed = self.spatial_transforms(pil_transformed)
        pil_transformed = self.color_jitter(pil_transformed)
        return self.to_tensor_norm(pil_transformed)


def get_eval_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
