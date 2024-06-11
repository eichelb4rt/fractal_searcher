from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


UNIT_SQUARE_CENTER = np.array([0.5, 0.5], dtype=np.float32)


@dataclass
class Rectangle:
    """An affine function defined by what rectangle the unit square is mapped to. Angle is in the range [0, 1] which maps to [0, 360] degrees."""

    center_x: float
    center_y: float
    width: float
    height: float
    angle: float

    def __post_init__(self):
        assert 0 <= self.center_x <= 1
        assert 0 <= self.center_y <= 1
        assert 0 <= self.width <= 1
        assert 0 <= self.height <= 1
        assert 0 <= self.angle <= 1


def to_affine_function(rectangle: Rectangle) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Convert a rectangle to a transformation matrix plus an offset that transforms a point p = (x, y)^T to a different point Ap + b."""

    # compute the transformation matrix
    scale_matrix = np.array([[rectangle.width, 0], [0, rectangle.height]], dtype=np.float32)
    angle_radians = 2 * np.pi * rectangle.angle
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)], [np.sin(angle_radians), np.cos(angle_radians)]], dtype=np.float32)
    transformation_matrix = rotation_matrix @ scale_matrix
    # offset for A (p - c) + b where c is the unit square center
    offset = np.array([rectangle.center_x, rectangle.center_y], dtype=np.float32)
    # offset for Ap + (b - Ac)
    offset -= transformation_matrix @ UNIT_SQUARE_CENTER
    return transformation_matrix, offset


def main():
    point_center = np.array([0.5, 0.5], dtype=np.float32)
    point_sw = np.array([0, 0], dtype=np.float32)
    point_nw = np.array([0, 1], dtype=np.float32)
    point_ne = np.array([1, 1], dtype=np.float32)
    point_se = np.array([1, 0], dtype=np.float32)
    # maps the unit square to the bottom left quadrant and rotates it to the left once
    rectangle = Rectangle(
        center_x=0.25,
        center_y=0.25,
        width=0.5,
        height=0.5,
        angle=90 / 360,
    )

    expected_map_center = np.array([rectangle.center_x, rectangle.center_y], dtype=np.float32)
    expected_map_sw = np.array([0.5, 0], dtype=np.float32)
    expected_map_nw = np.array([0, 0], dtype=np.float32)
    expected_map_ne = np.array([0, 0.5], dtype=np.float32)
    expected_map_se = np.array([0.5, 0.5], dtype=np.float32)

    transformation_matrix, offset = to_affine_function(rectangle)
    assert np.allclose(transformation_matrix @ point_center + offset, expected_map_center, atol=1e-3)
    assert np.allclose(transformation_matrix @ point_sw + offset, expected_map_sw, atol=1e-3)
    assert np.allclose(transformation_matrix @ point_nw + offset, expected_map_nw, atol=1e-3)
    assert np.allclose(transformation_matrix @ point_ne + offset, expected_map_ne, atol=1e-3)
    assert np.allclose(transformation_matrix @ point_se + offset, expected_map_se, atol=1e-3)


if __name__ == "__main__":
    main()
