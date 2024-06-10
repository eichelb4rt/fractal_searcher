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

function_system = [to_affine_function(rectangle) for rectangle in rectangles_sierpinski]


points = generate_fractal(function_system, seed=0, num_points=1_000_000)
draw_points(points, image_width=1024, image_height=1024, file_path="images/sierpinski.png")

points = generate_fractal(function_system, seed=0, num_points=5000)
draw_points(points, image_width=32, image_height=32, file_path="images/sierpinski_small.png")
