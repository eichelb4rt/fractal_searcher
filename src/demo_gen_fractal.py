from rectangle import Rectangle
from gen_fractal import generate_fractal
from draw_points import draw_points

rectangles = [
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

points = generate_fractal(rectangles, seed=0, num_points=10_000)
draw_points(points, image_width=100, image_height=100, file_path="images/fractal.png")
