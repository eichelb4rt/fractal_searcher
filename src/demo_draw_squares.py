from draw import draw_rectangle
from rectangle import Rectangle

rectangle = Rectangle(
    center_x=0.25,
    center_y=0.25,
    width=0.5,
    height=0.5,
    rotate_angle=90 / 360,
)
draw_rectangle(rectangle, "images/mapped.png")

rectangle_2 = Rectangle(
    center_x=0.35,
    center_y=0.35,
    width=0.5,
    height=0.5,
    rotate_angle=70 / 360,
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

rectangles = [Rectangle(center_x=0.25112194, center_y=0.74369967, width=0.50011814, height=0.51339436, rotate_angle=1.0), Rectangle(center_x=0.6824844, center_y=0.86980325, width=0.60112315, height=0.45811594, rotate_angle=0.6686232), Rectangle(center_x=0.4944999, center_y=0.25298727, width=0.5047151, height=0.49528512, rotate_angle=0.9975586)]

for rectangle, name in zip(rectangles, ["rectangle_1.png", "rectangle_2.png", "rectangle_3.png"]):
    draw_rectangle(rectangle, f"images/{name}")
