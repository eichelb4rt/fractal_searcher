from draw import draw_rectangle
from rectangle import Rectangle

rectangle = Rectangle(
    center_x=0.25,
    center_y=0.25,
    width=0.5,
    height=0.5,
    angle=90 / 360,
)
draw_rectangle(rectangle, "images/mapped.png")

rectangle_2 = Rectangle(
    center_x=0.35,
    center_y=0.35,
    width=0.5,
    height=0.5,
    angle=70 / 360,
)
draw_rectangle(rectangle_2, "images/mapped_2.png")

# rectangles = [
#     Rectangle(
#         center_x=0.25,
#         center_y=0.75,
#         width=0.5,
#         height=0.5,
#         angle=0,
#     ),
#     Rectangle(
#         center_x=0.75,
#         center_y=0.75,
#         width=0.5,
#         height=0.5,
#         angle=0,
#     ),
#     Rectangle(
#         center_x=0.5,
#         center_y=0.25,
#         width=0.5,
#         height=0.5,
#         angle=0,
#     ),
# ]

rectangles = [Rectangle(center_x=0.49549866480573873, center_y=0.24731512864041308, width=0.4692865748828883, height=0.4894721382156372, angle=1.0), Rectangle(center_x=0.43495436075878546, center_y=1.0, width=0.8814858602660186, height=0.2506769751042075, angle=1.0), Rectangle(center_x=0.21157241653999595, center_y=0.8732515341762715, width=0.6971007788939101, height=1.0, angle=0.6945462638053035)]

for rectangle, name in zip(rectangles, ["rectangle_1.png", "rectangle_2.png", "rectangle_3.png"]):
    draw_rectangle(rectangle, f"images/{name}")
