import numpy as np
import numpy.typing as npt

from PIL import Image

from rectangle import Rectangle, to_affine_function
from compute_image_interface import compute_image


DEFAULT_IMAGE_WIDTH = 256
DEFAULT_IMAGE_HEIGHT = 256


def draw_image(image: npt.NDArray[np.float32], file_path: str):
    """Draws images with pixel values in the range [0, 1]."""

    Image.fromarray(image * 255).convert('RGB').save(file_path)


def draw_points(points: npt.NDArray[np.float32], file_path: str, image_width: int = DEFAULT_IMAGE_WIDTH, image_height: int = DEFAULT_IMAGE_HEIGHT):
    assert len(points.shape) == 2, "Points should be a 2D tensor."
    assert points.shape[1] == 2, "Points should have 2 columns (x, y)."
    image = compute_image(points, image_width, image_height)
    draw_image(image, file_path)


def draw_rectangle(rectangle: Rectangle, file_path: str, image_width: int = DEFAULT_IMAGE_WIDTH, image_height: int = DEFAULT_IMAGE_HEIGHT):
    # create all coordinates in the unit square
    coordinates = np.meshgrid(np.linspace(0, 1, image_width), np.linspace(0, 1, image_height), indexing="ij")
    # and put them in a list
    points_unit_square = np.stack(coordinates).reshape(2, -1).T.astype(np.float32)
    # apply the affine function defined by the rectangle to all these points
    transformation_matrix, offset = to_affine_function(rectangle)
    points_mapped = points_unit_square @ transformation_matrix.T + offset
    draw_points(points_mapped, file_path, image_width, image_height)
