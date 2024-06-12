import numpy as np
from compute_image_interface import compute_image
from draw import draw_points
from gen_fractal import generate_fractal
from rectangle import Rectangle, rectangles_to_function_system, to_affine_function, rectangle_to_contiguous_affine_function

rectangles_sierpinski = [
    Rectangle(
        center_x=0.25,
        center_y=0.75,
        width=0.5,
        height=0.5,
        rotate_angle=0,
    ),
    Rectangle(
        center_x=0.75,
        center_y=0.75,
        width=0.5,
        height=0.5,
        rotate_angle=0,
    ),
    Rectangle(
        center_x=0.5,
        center_y=0.25,
        width=0.5,
        height=0.5,
        rotate_angle=0,
    ),
]

# function_system = rectangles_to_function_system(rectangles_sierpinski)
# function_system = np.array([0.5, 0., 0., 0.5, 0., 0.5, 0.5, 0., 0., 0.5, 0.5, 0.5, 0.5, 0., 0., 0.5, 0.25, 0], dtype=np.float32)
# points = generate_fractal(function_system, seed=0, num_points=1_000_000)
# draw_points(points, image_width=1024, image_height=1024, file_path="images/sierpinski.png")


rectangles_botched_sierpinski = [Rectangle(center_x=0.43331647, center_y=0.3955077, width=0.52147627, height=0.46386707, rotate_angle=0.66796845), Rectangle(center_x=0.34994644, center_y=0.8730466, width=0.5898434, height=0.5004258, rotate_angle=0.3173819), Rectangle(center_x=0.79011494, center_y=0.8679324, width=0.5420968, height=0.53437376, rotate_angle=0.33699903)]
function_system_botched = rectangles_to_function_system(rectangles_botched_sierpinski)
points_botched = generate_fractal(function_system_botched, seed=0, num_points=1_000_000)
draw_points(points_botched, image_width=1024, image_height=1024, file_path="images/sierpinski_botched.png")
