import numpy as np
from compute_image_interface import compute_image
from draw import draw_points
from gen_fractal import generate_fractal
from rectangle import Rectangle, rectangles_to_function_system, to_affine_function, rectangle_to_contiguous_affine_function

# generating_rectangles = [
#     Rectangle(
#         center_x=0.25,
#         center_y=0.75,
#         width=0.5,
#         height=0.5,
#         rotate_angle=0,
#     ),
#     Rectangle(
#         center_x=0.75,
#         center_y=0.75,
#         width=0.5,
#         height=0.5,
#         rotate_angle=0,
#     ),
#     Rectangle(
#         center_x=0.5,
#         center_y=0.25,
#         width=0.5,
#         height=0.5,
#         rotate_angle=0,
#     ),
# ]

# function_system = rectangles_to_function_system(rectangles_sierpinski)
# function_system = np.array([0.5, 0., 0., 0.5, 0., 0.5, 0.5, 0., 0., 0.5, 0.5, 0.5, 0.5, 0., 0., 0.5, 0.25, 0], dtype=np.float32)
# points = generate_fractal(function_system, seed=0, num_points=1_000_000)
# draw_points(points, image_width=1024, image_height=1024, file_path="images/sierpinski.png")


generating_rectangles = [Rectangle(center_x=0.4836706, center_y=0.6347658, width=0.8470122, height=0.61462164, rotate_angle=0.6777338), Rectangle(center_x=0.24755667, center_y=0.7460904, width=0.49858105, height=0.50419086, rotate_angle=1.0)]
function_system = rectangles_to_function_system(generating_rectangles)
points = generate_fractal(function_system, seed=0, num_points=2_000_000)
draw_points(points, image_width=1024, image_height=1024, file_path="images/generated.png")
