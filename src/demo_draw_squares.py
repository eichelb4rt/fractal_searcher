import numpy as np
import torch
from draw_points import compute_image, draw
from rectangle import Rectangle, to_affine_function

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100

points_unit_square = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
points_unit_square = torch.from_numpy(np.stack(points_unit_square, axis=-1).reshape(-1, 2).astype(np.float32))
image_unit_square = compute_image(points_unit_square, IMAGE_WIDTH, IMAGE_HEIGHT)
draw(image_unit_square, "images/unit_square.png")
rectangle = Rectangle(
    center_x=0.25,
    center_y=0.25,
    width=0.5,
    height=0.5,
    angle=90,
)
transformation_matrix, offset = to_affine_function(rectangle)
points_mapped = points_unit_square @ transformation_matrix.T + offset
image_mapped = compute_image(points_mapped, IMAGE_WIDTH, IMAGE_HEIGHT)
draw(image_mapped, "images/mapped.png")

rectangle_2 = Rectangle(
    center_x=0.35,
    center_y=0.35,
    width=0.5,
    height=0.5,
    angle=70,
)
transformation_matrix_2, offset_2 = to_affine_function(rectangle_2)
points_mapped_2 = points_unit_square @ transformation_matrix_2.T + offset_2
image_mapped_2 = compute_image(points_mapped_2, IMAGE_WIDTH, IMAGE_HEIGHT)
draw(image_mapped_2, "images/mapped_2.png")
