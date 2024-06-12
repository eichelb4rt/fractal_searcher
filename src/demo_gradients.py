from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image
from tqdm import tqdm

from compute_image_interface import compute_image
from draw import draw_image, draw_points
from find_fractal import subsample_image
from gen_fractal import generate_fractal
from gradients import gradient_descent, greedy_descent, greedy_descent_2
from rectangle import Rectangle, rectangles_to_function_system, rectangle_to_contiguous_affine_function, rectangles_to_vector, vectors_to_function_system, vectors_to_rectangles
from find_fractal import error

SUBSAMPLE_SIZE = 256
NUM_STEPS = 800
LEARNING_RATE = 2e-2
GRADIENT_APPROXIMATION_EPSILON = 1e-2
GREEDY_STEP_SIZE = 1e-2
N_RUNS = 20

STD_STRAY = 0.1

N_POINTS = 5 * SUBSAMPLE_SIZE * SUBSAMPLE_SIZE

USED_METHOD = "gradient_descent"


# read a grayscale image
target_image = Image.open("images/sierpinski.png").convert('L')
target_image = np.array(target_image).astype(np.float32) / 255
subsampled_target_image = subsample_image(target_image, SUBSAMPLE_SIZE, SUBSAMPLE_SIZE)
draw_image(subsampled_target_image, f"images/sierpinski_subsampled_{subsampled_target_image.shape[0]}x{subsampled_target_image.shape[1]}.png")
# set some hyperparameters
np.random.seed(0)
n_rectangles = 3
starting_point = np.random.rand(2)
selected_indices = np.random.randint(0, n_rectangles, (N_POINTS,)).astype(np.uint32)
# build the objective function with that fixed seed
objective_function = lambda rectangles_param_vector: error(rectangles_param_vector, subsampled_target_image, num_points=N_POINTS, starting_point=starting_point, selected_indices=selected_indices)

# create a few random starts for the gradient descent
np.random.seed(0)
param_vector_size = 5 * n_rectangles
# initial_param_vectors = np.random.uniform(0, 1, N_RUNS * param_vector_size).reshape((N_RUNS, param_vector_size)).astype(np.float32)
old_top_rectangles = [Rectangle(center_x=0.43559211, center_y=0.37553188, width=0.54621595, height=0.45281804, rotate_angle=0.66374946), Rectangle(center_x=0.30484405, center_y=0.8763722, width=0.5664403, height=0.4548299, rotate_angle=0.3348315), Rectangle(center_x=0.8143797, center_y=0.87052804, width=0.5524245, height=0.45184985, rotate_angle=0.3303465)]
old_top_vector = rectangles_to_vector(old_top_rectangles)
initial_param_vectors = (np.random.normal(0, STD_STRAY, N_RUNS * param_vector_size).reshape((N_RUNS, param_vector_size)).astype(np.float32) + old_top_vector).clip(0, 1)

# optimize


def gradient_descent_wrapper(initial_param_vector, show_progress):
    return gradient_descent(objective_function, initial_param_vector, learning_rate=LEARNING_RATE, num_steps=NUM_STEPS, gradient_approximation_epsilon=GRADIENT_APPROXIMATION_EPSILON, show_progress=show_progress)


def greedy_descent_wrapper(initial_param_vector, show_progress):
    return greedy_descent(objective_function, initial_param_vector, num_steps=NUM_STEPS, step_size=GREEDY_STEP_SIZE, show_progress=show_progress)


def greedy_descent_2_wrapper(initial_param_vector, show_progress):
    return greedy_descent_2(objective_function, initial_param_vector, num_steps=NUM_STEPS, step_size=GREEDY_STEP_SIZE, show_progress=show_progress)


optimization_methods = {
    "gradient_descent": gradient_descent_wrapper,
    "greedy_descent": greedy_descent_wrapper,
    "greedy_descent_2": greedy_descent_2_wrapper
}
pool = Pool()
args = [(initial_param_vector, i == 0) for i, initial_param_vector in enumerate(initial_param_vectors)]
results = pool.starmap(optimization_methods[USED_METHOD], args)
# results = [optimization_methods[USED_METHOD](initial_function_system, True) for i, initial_function_system in enumerate(initial_function_systems)]
param_vectors, error_courses = zip(*results)
# see what we optimized
scores = [objective_function(param_vector) for param_vector in param_vectors]
ranks = np.argsort(scores)
param_vectors_sorted = [param_vectors[i] for i in ranks]
scores_sorted = [scores[i] for i in ranks]
for rank, (score, param_vector) in enumerate(zip(scores_sorted, param_vectors_sorted), 1):
    print(f"Top {rank} score: {score}, rectangles: {vectors_to_rectangles(param_vector)}")
# plot the errors
y_min = np.min(error_courses)
y_max = np.quantile(error_courses, 0.99)
for i, error_course in enumerate(error_courses):
    plt.plot(error_course, label=f"Run {i}")
plt.ylim(bottom=y_min, top=y_max)
plt.title(f"learning rate: {LEARNING_RATE}, num steps: {NUM_STEPS}, epsilon: {GRADIENT_APPROXIMATION_EPSILON}")
# plt.legend()
plt.savefig(f"images/{USED_METHOD}_sierpinski_small_error_n_runs_{N_RUNS}_lr_{LEARNING_RATE:.1e}_n_steps_{NUM_STEPS}_gd_epsilon_{GRADIENT_APPROXIMATION_EPSILON:.1e}_subsample_size_{SUBSAMPLE_SIZE}.png")
# draw the results
# for rank, param_vector in enumerate(param_vectors_sorted, 1):
#     function_system = vectors_to_function_system(param_vector)
#     points = generate_fractal(function_system, num_points=N_POINTS, starting_point=starting_point, selected_indices=selected_indices)
#     image = compute_image(points, subsampled_target_image.shape[1], subsampled_target_image.shape[0])
#     draw_image(image, f"images/sierpinski_{subsampled_target_image.shape[0]}x{subsampled_target_image.shape[1]}_small_reconstructed_top_{rank}.png")

# draw the best result in the subsampled resolution
best_param_vector = param_vectors_sorted[0]
function_system = vectors_to_function_system(best_param_vector)
points = generate_fractal(function_system, num_points=N_POINTS, starting_point=starting_point, selected_indices=selected_indices)
draw_points(points, image_width=subsampled_target_image.shape[1], image_height=subsampled_target_image.shape[0], file_path=f"images/sierpinski_reconstructed_small_{SUBSAMPLE_SIZE}x{SUBSAMPLE_SIZE}.png")


# draw the best result in the full resolution
points = generate_fractal(function_system, num_points=1_000_000, seed=0)
draw_points(points, image_width=target_image.shape[1], image_height=target_image.shape[0], file_path=f"images/sierpinski_reconstructed_full_from_{SUBSAMPLE_SIZE}x{SUBSAMPLE_SIZE}.png")

true_rectangles = [
    Rectangle(
        center_x=0.25,
        center_y=0.75,
        width=0.5,
        height=0.5,
        rotate_angle=0,
    ),
    Rectangle(
        center_x=0.75,
        center_y=0.75,
        width=0.5,
        height=0.5,
        rotate_angle=0,
    ),
    Rectangle(
        center_x=0.5,
        center_y=0.25,
        width=0.5,
        height=0.5,
        rotate_angle=0,
    ),
]

true_parameters = rectangles_to_vector(true_rectangles)
print(f"True parameter error: {objective_function(true_parameters)}")
true_function_system = vectors_to_function_system(true_parameters)
true_points = generate_fractal(true_function_system, num_points=N_POINTS, starting_point=starting_point, selected_indices=selected_indices)
draw_points(true_points, image_width=subsampled_target_image.shape[1], image_height=subsampled_target_image.shape[0], file_path="images/sierpinski_true.png")
