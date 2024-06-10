import torch
from PIL import Image
from torch import Tensor

from rectangle import Rectangle, to_affine_function


DEFAULT_IMAGE_WIDTH = 256
DEFAULT_IMAGE_HEIGHT = 256


def compute_image(points: Tensor, image_width: int = DEFAULT_IMAGE_WIDTH, image_height: int = DEFAULT_IMAGE_HEIGHT) -> Tensor:
    assert len(points.shape) == 2, "Points should be a 2D tensor."
    assert points.shape[1] == 2, "Points should have 2 columns (x, y)."
    scaled_points = points * torch.tensor([image_width, image_height], dtype=torch.float32)
    full_pixels = scaled_points.round().to(torch.int32)
    image = torch.zeros((image_height, image_width), dtype=torch.float32)
    for x, y in full_pixels:
        if 0 <= x < image_width and 0 <= y < image_height:
            image[y, x] = 1
    return image


def draw(image: Tensor, file_path: str):
    """Draws images with pixel values in the range [0, 1]."""

    Image.fromarray(image.numpy() * 255).convert('RGB').save(file_path)


def draw_points(points: Tensor, file_path: str, image_width: int = DEFAULT_IMAGE_WIDTH, image_height: int = DEFAULT_IMAGE_HEIGHT):
    assert len(points.shape) == 2, "Points should be a 2D tensor."
    assert points.shape[1] == 2, "Points should have 2 columns (x, y)."
    image = compute_image(points, image_width, image_height)
    draw(image, file_path)


def draw_rectangle(rectangle: Rectangle, file_path: str, image_width: int = DEFAULT_IMAGE_WIDTH, image_height: int = DEFAULT_IMAGE_HEIGHT):
    # create all coordinates in the unit square
    coordinates = torch.meshgrid(torch.linspace(0, 1, image_width), torch.linspace(0, 1, image_height), indexing="ij")
    # and put them in a list
    points_unit_square = torch.stack(coordinates).reshape(2, -1).T
    # apply the affine function defined by the rectangle to all these points
    transformation_matrix, offset = to_affine_function(rectangle)
    points_mapped = points_unit_square @ transformation_matrix.T + offset
    draw_points(points_mapped, file_path, image_width, image_height)
