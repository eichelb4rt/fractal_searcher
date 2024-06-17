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


generating_rectangles = [Rectangle(center_x=0.31483945, center_y=0.7129634, width=0.4889633, height=0.6013781, rotate_angle=0.8016038), Rectangle(center_x=0.57289195, center_y=0.38880455, width=0.41044068, height=0.9426252, rotate_angle=0.5978181), Rectangle(center_x=0.14312154, center_y=0.74318475, width=0.576238, height=0.3695073, rotate_angle=0.24424502), Rectangle(center_x=0.6002584, center_y=0.4113936, width=0.83180624, height=0.45762888, rotate_angle=0.4121104)]
function_system = rectangles_to_function_system(generating_rectangles)
points = generate_fractal(function_system, seed=0, num_points=2_000_000)
draw_points(points, image_width=1024, image_height=1024, file_path="images/generated.png")
