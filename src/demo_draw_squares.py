from draw_points import draw_rectangle
from rectangle import Rectangle

rectangle = Rectangle(
    center_x=0.25,
    center_y=0.25,
    width=0.5,
    height=0.5,
    angle=90,
)
draw_rectangle(rectangle, "images/mapped.png")

rectangle_2 = Rectangle(
    center_x=0.35,
    center_y=0.35,
    width=0.5,
    height=0.5,
    angle=70,
)
draw_rectangle(rectangle_2, "images/mapped_2.png")

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

for rectangle, name in zip(rectangles, ["rectangle_1.png", "rectangle_2.png", "rectangle_3.png"]):
    draw_rectangle(rectangle, f"images/{name}")
