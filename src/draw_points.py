import numpy as np
import numpy.typing as npt
from torch import Tensor
import torch
from PIL import Image


def compute_image(points: Tensor, image_width: int, image_height: int) -> npt.NDArray[np.float32]:
    scaled_points = points * torch.tensor([image_width, image_height], dtype=torch.float32)
    full_pixels = scaled_points.round().to(torch.int32)
    image = np.zeros((image_height, image_width), dtype=np.float32)
    for x, y in full_pixels:
        if 0 <= x < image_width and 0 <= y < image_height:
            image[y, x] = 1
    return image


def draw(image: npt.NDArray[np.float32], file_path: str):
    """Draws images with pixel values in the range [0, 1]."""

    Image.fromarray(image * 255).convert('RGB').save(file_path)


def draw_points(points: Tensor, image_width: int, image_height: int, file_path: str):
    image = compute_image(points, image_width, image_height)
    draw(image, file_path)
