from rectangle import Rectangle
from gen_fractal import generate_fractal
from draw_points import draw_points, draw_rectangle

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


points = generate_fractal(rectangles, seed=0, num_points=100_000)
draw_points(points, image_width=256, image_height=256, file_path="images/fractal.png")

for rectangle, name in zip(rectangles, ["rectangle_1.png", "rectangle_2.png", "rectangle_3.png"]):
    draw_rectangle(rectangle, image_width=256, image_height=256, file_path=f"images/{name}")
