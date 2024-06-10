from dataclasses import dataclass
import torch
from torch import Tensor


UNIT_SQUARE_CENTER = torch.tensor([0.5, 0.5], dtype=torch.float32)


@dataclass
class Rectangle:
    """An affine function defined by what rectangle the unit square is mapped to."""

    center_x: float
    center_y: float
    width: float
    height: float
    angle: float

    def __post_init__(self):
        assert 0 <= self.center_x <= 1
        assert 0 <= self.center_y <= 1
        assert 0 < self.width <= 1
        assert 0 < self.height <= 1
        assert 0 <= self.angle <= 360


def to_affine_function(rectangle: Rectangle) -> tuple[Tensor, Tensor]:
    """Convert a rectangle to a transformation matrix plus an offset that transforms a point p = (x, y)^T to a different point Ap + b."""

    # compute the transformation matrix
    scale_matrix = torch.tensor([[rectangle.width, 0], [0, rectangle.height]], dtype=torch.float32)
    angle_radians = torch.deg2rad(torch.tensor(rectangle.angle))
    rotation_matrix = torch.tensor([[torch.cos(angle_radians), -torch.sin(angle_radians)], [torch.sin(angle_radians), torch.cos(angle_radians)]], dtype=torch.float32)
    transformation_matrix = rotation_matrix @ scale_matrix
    # offset for A (p - c) + b where c is the unit square center
    offset = torch.tensor([rectangle.center_x, rectangle.center_y])
    # offset for Ap + (b - Ac)
    offset -= transformation_matrix @ UNIT_SQUARE_CENTER
    return transformation_matrix, offset


def main():
    point_center = torch.tensor([0.5, 0.5], dtype=torch.float32)
    point_sw = torch.tensor([0, 0], dtype=torch.float32)
    point_nw = torch.tensor([0, 1], dtype=torch.float32)
    point_ne = torch.tensor([1, 1], dtype=torch.float32)
    point_se = torch.tensor([1, 0], dtype=torch.float32)
    # maps the unit square to the bottom left quadrant and rotates it to the left once
    rectangle = Rectangle(
        center_x=0.25,
        center_y=0.25,
        width=0.5,
        height=0.5,
        angle=90,
    )

    expected_map_center = torch.tensor([rectangle.center_x, rectangle.center_y], dtype=torch.float32)
    expected_map_sw = torch.tensor([0.5, 0], dtype=torch.float32)
    expected_map_nw = torch.tensor([0, 0], dtype=torch.float32)
    expected_map_ne = torch.tensor([0, 0.5], dtype=torch.float32)
    expected_map_se = torch.tensor([0.5, 0.5], dtype=torch.float32)

    transformation_matrix, offset = to_affine_function(rectangle)
    assert torch.allclose(transformation_matrix @ point_center + offset, expected_map_center, atol=1e-3)
    assert torch.allclose(transformation_matrix @ point_sw + offset, expected_map_sw, atol=1e-3)
    assert torch.allclose(transformation_matrix @ point_nw + offset, expected_map_nw, atol=1e-3)
    assert torch.allclose(transformation_matrix @ point_ne + offset, expected_map_ne, atol=1e-3)
    assert torch.allclose(transformation_matrix @ point_se + offset, expected_map_se, atol=1e-3)


if __name__ == "__main__":
    main()
