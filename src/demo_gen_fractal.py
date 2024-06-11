import numpy as np
from compute_image_interface import compute_image
from draw import draw_points
from gen_fractal import generate_fractal
from rectangle import Rectangle, rectangles_to_function_system, to_affine_function, to_contiguous_affine_function

rectangles_sierpinski = [
    Rectangle(
        center_x=0.25,
        center_y=0.75,
        width=0.5,
        height=0.5,
        angle=0,
        mirror=False,
    ),
    Rectangle(
        center_x=0.75,
        center_y=0.75,
        width=0.5,
        height=0.5,
        angle=0,
        mirror=False,
    ),
    Rectangle(
        center_x=0.5,
        center_y=0.25,
        width=0.5,
        height=0.5,
        angle=0,
        mirror=False,
    ),
]

# function_system = rectangles_to_function_system(rectangles_sierpinski)
# function_system = np.array([0.5, 0., 0., 0.5, 0., 0.5, 0.5, 0., 0., 0.5, 0.5, 0.5, 0.5, 0., 0., 0.5, 0.25, 0], dtype=np.float32)
# points = generate_fractal(function_system, seed=0, num_points=1_000_000)
# draw_points(points, image_width=1024, image_height=1024, file_path="images/sierpinski.png")


# rectangles_botched_sierpinski = [Rectangle(center_x=0.40451179215505545, center_y=0.4046360841851236, width=0.5595878400033942, height=0.4606784931615354, angle=0.6384846972296522), Rectangle(center_x=0.20613531071812397, center_y=0.7816655771655312, width=0.551516048453975, height=0.38404748499690977, angle=1.0), Rectangle(center_x=0.36274372049024567, center_y=0.7841856975687671, width=1.0, height=0.8473707985538472, angle=0.6698225030216304)]
# function_system_botched = rectangles_to_function_system(rectangles_botched_sierpinski)
function_system_botched = np.array([-0.534734, -0.2514664, 0.2781784, -0.48338634, 0.9419137,
                                    0.8177933, 0.24975565, 0.37033963, -0.93073523, 0.0993778,
                                    0.12753958, 1.3074517, 0.07046364, -0.0110388, 0.00899988,
                                    0.0864272, 0.5383321, 0.87788314], dtype=np.float32)
points_botched = generate_fractal(function_system_botched, seed=0, num_points=1_000_000)
draw_points(points_botched, image_width=1024, image_height=1024, file_path="images/sierpinski_botched.png")
