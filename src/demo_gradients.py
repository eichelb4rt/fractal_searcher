from multiprocessing import Pool

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
from rectangle import Rectangle, rectangles_to_function_system, to_contiguous_affine_function, rectangles_to_vector, vectors_to_rectangles
from find_fractal import error

SUBSAMPLE_SIZE = 32
NUM_STEPS = 200
LEARNING_RATE = 1e-0
GRADIENT_APPROXIMATION_EPSILON = 1e-2
GREEDY_STEP_SIZE = 1e-1
N_RUNS = 20

STD_STRAY = 0.1

N_POINTS = 5 * SUBSAMPLE_SIZE * SUBSAMPLE_SIZE

USED_METHOD = "gradient_descent"


# read a grayscale image
target_image = Image.open("images/sierpinski.png").convert('L')
target_image = np.array(target_image).astype(np.float32) / 255
subsampled_target_image = subsample_image(target_image, SUBSAMPLE_SIZE, SUBSAMPLE_SIZE)
# set some hyperparameters
np.random.seed(0)
n_rectangles = 3
starting_point = np.random.rand(2)
selected_indices = np.random.randint(0, n_rectangles, (N_POINTS,)).astype(np.uint32)
# build the objective function with that fixed seed
objective_function = lambda function_system: error(function_system, subsampled_target_image, num_points=N_POINTS, starting_point=starting_point, selected_indices=selected_indices)

# create a few random starts for the gradient descent
np.random.seed(0)
x_size = 6 * n_rectangles
# initial_function_systems = np.random.normal(0, 0.05, N_RUNS * x_size).reshape((N_RUNS, x_size)).astype(np.float32)
initial_vectors = np.random.uniform(0, 1, N_RUNS * x_size).reshape((N_RUNS, n_rectangles, 6)).astype(np.float32)
initial_rectangles = [vectors_to_rectangles(initial_vectors[i]) for i in range(N_RUNS)]
initial_function_systems = [rectangles_to_function_system(initial_rectangles[i]) for i in range(N_RUNS)]
# old_top_rectangles = [Rectangle(center_x=0.419581859501152, center_y=0.31470083211469646, width=0.521265774977827, height=0.4039322703376769, angle=0.5650563808152962), Rectangle(center_x=0.21878087516677852, center_y=0.96, width=0.6786466492640256, height=0.32828351617156976, angle=0.98), Rectangle(center_x=0.413453728038578, center_y=0.6374711738547325, width=0.7895365900559044, height=0.6945811611074447, angle=0.6737167009624576)]
# old_top_vector = rectangles_to_vector(old_top_rectangles)
# initial_rectangles = vectors_to_rectangles((np.random.normal(0, STD_STRAY, N_RUNS * x_size).reshape((N_RUNS, x_size)).astype(np.float32) + old_top_vector).clip(0, 1))
# initial_function_systems = [rectangles_to_function_system(initial_rectangles[i]) for i in range(N_RUNS)]

# optimize


def gradient_descent_wrapper(initial_function_system, show_progress):
    return gradient_descent(objective_function, initial_function_system, learning_rate=LEARNING_RATE, num_steps=NUM_STEPS, gradient_approximation_epsilon=GRADIENT_APPROXIMATION_EPSILON, show_progress=show_progress)


def greedy_descent_wrapper(initial_function_system, show_progress):
    return greedy_descent(objective_function, initial_function_system, num_steps=NUM_STEPS, step_size=GREEDY_STEP_SIZE, show_progress=show_progress)


def greedy_descent_2_wrapper(initial_function_system, show_progress):
    return greedy_descent_2(objective_function, initial_function_system, num_steps=NUM_STEPS, step_size=GREEDY_STEP_SIZE, show_progress=show_progress)


optimization_methods = {
    "gradient_descent": gradient_descent_wrapper,
    "greedy_descent": greedy_descent_wrapper,
    "greedy_descent_2": greedy_descent_2_wrapper
}
pool = Pool()
args = [(initial_function_system, i == 0) for i, initial_function_system in enumerate(initial_function_systems)]
results = pool.starmap(optimization_methods[USED_METHOD], args)
# results = [optimization_methods[USED_METHOD](initial_function_system, True) for i, initial_function_system in enumerate(initial_function_systems)]
top_x_s, error_courses = zip(*results)
# see what we optimized
scores = [objective_function(top_x) for top_x in top_x_s]
ranks = np.argsort(scores)
function_systems_sorted = [top_x_s[i] for i in ranks]
scores_sorted = [scores[i] for i in ranks]
for rank, (score, function_system) in enumerate(zip(scores_sorted, function_systems_sorted), 1):
    print(f"Top {rank} score: {score}, function_system: {repr(function_system)}")
# plot the errors
for i, error_course in enumerate(error_courses):
    plt.plot(error_course, label=f"Run {i}")
plt.title(f"learning rate: {LEARNING_RATE}, num steps: {NUM_STEPS}, epsilon: {GRADIENT_APPROXIMATION_EPSILON}")
# plt.legend()
plt.savefig(f"images/greedy_sierpinski_small_error_n_runs_{N_RUNS}_lr_{LEARNING_RATE:.1e}_n_steps_{NUM_STEPS}_gd_epsilon_{GRADIENT_APPROXIMATION_EPSILON:.1e}_subsample_size_{SUBSAMPLE_SIZE}.png")
# draw the results
draw_image(subsampled_target_image, f"images/sierpinski_subsampled_{subsampled_target_image.shape[0]}x{subsampled_target_image.shape[1]}.png")
for rank, function_system in enumerate(function_systems_sorted, 1):
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
        mirror=False,
    ),
    Rectangle(
        center_x=0.75,
        center_y=0.75,
        width=0.5,
        height=0.5,
        angle=0,
        mirror=False,
    ),
    Rectangle(
        center_x=0.5,
        center_y=0.25,
        width=0.5,
        height=0.5,
        angle=0,
        mirror=False,
    ),
]

true_parameters = rectangles_to_function_system(true_rectangles)
print(f"True parameter error: {objective_function(true_parameters)}")
