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

rectangles = [Rectangle(center_x=0.43331647, center_y=0.3955077, width=0.52147627, height=0.46386707, rotate_angle=0.66796845), Rectangle(center_x=0.34994644, center_y=0.8730466, width=0.5898434, height=0.5004258, rotate_angle=0.3173819), Rectangle(center_x=0.79011494, center_y=0.8679324, width=0.5420968, height=0.53437376, rotate_angle=0.33699903)]

for rectangle, name in zip(rectangles, ["rectangle_1.png", "rectangle_2.png", "rectangle_3.png"]):
    draw_rectangle(rectangle, f"images/{name}")
