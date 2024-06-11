import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image
from tqdm import tqdm

from compute_image_interface import compute_image
from draw import draw_image
from find_fractal import subsample_image
from gen_fractal import generate_fractal
from gradients import gradient_descent, greedy_descent, greedy_descent_2
from rectangle import Rectangle, to_affine_function

SUBSAMPLE_SIZE = 16
NUM_STEPS = 200
LEARNING_RATE = 1e-2
GRADIENT_APPROXIMATION_EPSILON = 1e-1
GREEDY_STEP_SIZE = 1e-2
N_RUNS = 10

STD_STRAY = 0.1

N_POINTS = 5 * SUBSAMPLE_SIZE * SUBSAMPLE_SIZE


def to_vector(rectangles: list[Rectangle]) -> npt.NDArray[np.float32]:
    return np.array([[rectangle.center_x, rectangle.center_y, rectangle.width, rectangle.height, rectangle.angle] for rectangle in rectangles]).flatten()


def error(rectangle_vectors: npt.NDArray[np.float32], target_image: npt.NDArray[np.float32], num_points: int, starting_point: npt.NDArray[np.float32], selected_indices: list[int]) -> float:
    # convert vectors -> rectangles -> function system
    assert len(rectangle_vectors) % 5 == 0
    rectangles = [Rectangle(*rectangle_vectors[i:i + 5]) for i in range(0, len(rectangle_vectors), 5)]
    function_system = [to_affine_function(rectangle) for rectangle in rectangles]
    # generate the fractal
    points = generate_fractal(function_system, num_points=num_points, starting_point=starting_point, selected_indices=selected_indices)
    # compute the difference to the target image
    image = compute_image(points, target_image.shape[1], target_image.shape[0])
    return np.mean((image - target_image) ** 2).item()


# read a grayscale image
target_image = Image.open("images/sierpinski.png").convert('L')
target_image = np.array(target_image).astype(np.float32) / 255
subsampled_target_image = subsample_image(target_image, SUBSAMPLE_SIZE, SUBSAMPLE_SIZE)
# set some hyperparameters
np.random.seed(0)
n_rectangles = 3
starting_point = np.random.rand(2)
selected_indices = np.random.randint(0, n_rectangles, (N_POINTS,))
# build the objective function with that fixed seed
objective_function = lambda x: error(x, subsampled_target_image, num_points=N_POINTS, starting_point=starting_point, selected_indices=selected_indices)

# create a few random starts for the gradient descent
np.random.seed(0)
x_size = 5 * n_rectangles
# initial_x_s = np.random.uniform(0, 1, N_RUNS * x_size).reshape((N_RUNS, x_size)).astype(np.float32).clip(0, 1)
old_top_rectangles = [Rectangle(center_x=0.419581859501152, center_y=0.31470083211469646, width=0.521265774977827, height=0.4039322703376769, angle=0.5650563808152962), Rectangle(center_x=0.21878087516677852, center_y=0.96, width=0.6786466492640256, height=0.32828351617156976, angle=0.98), Rectangle(center_x=0.413453728038578, center_y=0.6374711738547325, width=0.7895365900559044, height=0.6945811611074447, angle=0.6737167009624576)]
old_top_vector = to_vector(old_top_rectangles)
initial_x_s = (np.random.normal(0, STD_STRAY, N_RUNS * x_size).reshape((N_RUNS, x_size)).astype(np.float32) + old_top_vector).clip(0, 1)

# optimize
# top_x_s, error_courses = zip(*[greedy_descent(objective_function, initial_x, num_steps=NUM_STEPS, step_size=GREEDY_STEP_SIZE) for initial_x in tqdm(initial_x_s, leave=False)])
top_x_s, error_courses = zip(*[gradient_descent(objective_function, initial_x, learning_rate=LEARNING_RATE, num_steps=NUM_STEPS, gradient_approximation_epsilon=GRADIENT_APPROXIMATION_EPSILON) for initial_x in tqdm(initial_x_s, leave=False)])
# see what we optimized
scores = [objective_function(top_x) for top_x in top_x_s]
ranks = np.argsort(scores)
x_s_sorted = [top_x_s[i] for i in ranks]
scores_sorted = [scores[i] for i in ranks]
rectangles_sorted = [[Rectangle(*x_s_sorted[i][j:j + 5]) for j in range(0, len(x_s_sorted[i]), 5)] for i in range(N_RUNS)]
for rank, (x, score, rectangles) in enumerate(zip(x_s_sorted, scores_sorted, rectangles_sorted), 1):
    print(f"Top {rank} score: {score}, rectangles: {rectangles}")
# plot the errors
for i, error_course in enumerate(error_courses):
    plt.plot(error_course, label=f"Run {i}")
plt.title(f"learning rate: {LEARNING_RATE}, num steps: {NUM_STEPS}, epsilon: {GRADIENT_APPROXIMATION_EPSILON}")
plt.legend()
plt.savefig(f"images/greedy_sierpinski_small_error_n_runs_{N_RUNS}_lr_{LEARNING_RATE:.1e}_n_steps_{NUM_STEPS}_gd_epsilon_{GRADIENT_APPROXIMATION_EPSILON:.1e}_subsample_size_{SUBSAMPLE_SIZE}.png")
# draw the results
draw_image(subsampled_target_image, f"images/sierpinski_subsampled_{subsampled_target_image.shape[0]}x{subsampled_target_image.shape[1]}.png")
for rank, rectangles in enumerate(rectangles_sorted, 1):
    function_system = [to_affine_function(rectangle) for rectangle in rectangles]
    points = generate_fractal(function_system, num_points=N_POINTS, starting_point=starting_point, selected_indices=selected_indices)
    image = compute_image(points, subsampled_target_image.shape[1], subsampled_target_image.shape[0])
    draw_image(image, f"images/sierpinski_{subsampled_target_image.shape[0]}x{subsampled_target_image.shape[1]}_small_reconstructed_top_{rank}.png")

true_rectangles = [
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

true_parameters = to_vector(true_rectangles)
print(f"True parameter error: {objective_function(true_parameters)}")
