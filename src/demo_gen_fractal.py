from compute_image_interface import compute_image
from draw import draw_points
from gen_fractal import generate_fractal
from rectangle import Rectangle, to_affine_function

rectangles_sierpinski = [
    Rectangle(
        center_x=0.25,
        center_y=0.75,
        width=0.5,
        height=0.5,
        angle=0,
    ),
    Rectangle(
        center_x=0.75,
        center_y=0.75,
        width=0.5,
        height=0.5,
        angle=0,
    ),
    Rectangle(
        center_x=0.5,
        center_y=0.25,
        width=0.5,
        height=0.5,
        angle=0,
    ),
]

# function_system = [to_affine_function(rectangle) for rectangle in rectangles_sierpinski]
# points = generate_fractal(function_system, seed=0, num_points=1_000_000)
# draw_points(points, image_width=1024, image_height=1024, file_path="images/sierpinski.png")


rectangles_botched_sierpinski = [Rectangle(center_x=0.49549866480573873, center_y=0.24731512864041308, width=0.4692865748828883, height=0.4894721382156372, angle=1.0), Rectangle(center_x=0.43495436075878546, center_y=1.0, width=0.8814858602660186, height=0.2506769751042075, angle=1.0), Rectangle(center_x=0.21157241653999595, center_y=0.8732515341762715, width=0.6971007788939101, height=1.0, angle=0.6945462638053035)]
function_system_botched = [to_affine_function(rectangle) for rectangle in rectangles_botched_sierpinski]
points_botched = generate_fractal(function_system_botched, seed=0, num_points=1_000_000)
draw_points(points_botched, image_width=1024, image_height=1024, file_path="images/sierpinski_botched.png")
