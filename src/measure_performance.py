import numpy as np
from tqdm import tqdm
from PIL import Image

from draw import draw_points
from gen_fractal import generate_fractal
from rectangle import Rectangle, rectangles_to_function_system
from gradients import gradient_approximation
from find_fractal import error

rectangles_sierpinski = [
    Rectangle(
        center_x=0.25,
        center_y=0.75,
        width=0.5,
        height=0.5,
        rotate_angle=0,
        mirror_angle=False,
    ),
    Rectangle(
        center_x=0.75,
        center_y=0.75,
        width=0.5,
        height=0.5,
        rotate_angle=0,
        mirror_angle=False,
    ),
    Rectangle(
        center_x=0.5,
        center_y=0.25,
        width=0.5,
        height=0.5,
        rotate_angle=0,
        mirror_angle=False,
    ),
]

target_image = Image.open("images/sierpinski.png").convert('L')
target_image = np.array(target_image).astype(np.float32) / 255
np.random.seed(0)
n_rectangles = 3
starting_point = np.random.rand(2)
N_POINTS = 1_000_000
selected_indices = np.random.randint(0, n_rectangles, (N_POINTS,)).astype(np.uint32)
objective_function = lambda x: error(x, target_image, num_points=N_POINTS, starting_point=starting_point, selected_indices=selected_indices)
function_system = rectangles_to_function_system(rectangles_sierpinski)
for _ in tqdm(range(20)):
    gradient_approximation(objective_function, function_system, 1e-2)
