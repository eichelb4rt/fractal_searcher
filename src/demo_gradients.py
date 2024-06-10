import numpy as np
import torch
from PIL import Image

from draw import compute_image, draw
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

torch.autograd.set_detect_anomaly(True)
function_system = [(transformation_matrix.requires_grad_(True), offset.requires_grad_(True)) for transformation_matrix, offset in function_system]
points = generate_fractal(function_system, seed=0, num_points=5000)
# some_scalar = torch.sum(points)
# some_scalar.backward()
image = compute_image(points, image_width=32, image_height=32)
some_scalar = torch.sum(image)
some_scalar.backward()
# # read a grayscale image
# target = Image.open("images/sierpinski_small.png").convert('L')
# target = torch.from_numpy(np.array(target)).to(torch.float32) / 255
# assert target.shape == image.shape
# # compute the loss
# error = torch.mean((image - target) ** 2)
# # compute the gradients
# error.backward()


# print the gradients
for transformation_matrix, offset in function_system:
    print(transformation_matrix.grad)
    print(offset.grad)
