from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image
from tqdm import tqdm

from compute_image_interface import compute_image
from draw import draw_image, draw_points
from find_fractal import glue_params, subsample_image
from gen_fractal import generate_fractal
from gradients import gradient_descent, gradient_descent_2, greedy_descent, greedy_descent_2, parallel_gradient_descent
from rectangle import Rectangle, rectangles_to_function_system, rectangle_to_contiguous_affine_function, rectangles_to_vector, vectors_to_function_system, vectors_to_rectangles
from find_fractal import error, fixed_params_error

SUBSAMPLE_SIZE = 32
NUM_STEPS = 4000
GRADIENT_APPROXIMATION_EPSILON = 1e-2
N_RUNS = 64
USED_METHOD = "gradient_descent"

# gradient_descent parameters
LEARNING_RATE = 3e-2

# greedy parameters
GREEDY_STEP_SIZE = 1e-2

# backtracking line search parameters
INITIAL_STEP_SIZE = 0.1
MAX_STEP_SIZE_TRIES = 10
STEP_SIZE_DECREASE = 0.75
EXPECTED_GRADIENT_GAIN = 0.5

# stray from the original parameters
STD_STRAY = 1e-1

N_POINTS = 5 * SUBSAMPLE_SIZE * SUBSAMPLE_SIZE


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

# old_top_rectangles = [Rectangle(center_x=0.18183221, center_y=0.8518872, width=0.5413586, height=0.46516478, rotate_angle=0.6695025), Rectangle(center_x=0.67693424, center_y=0.86053306, width=0.57093024, height=0.46157643, rotate_angle=0.66512793), Rectangle(center_x=0.49311495, center_y=0.24331623, width=0.50257236, height=0.5156987, rotate_angle=1.0)]
# old_top_vector = rectangles_to_vector(old_top_rectangles)
# initial_param_vectors = (np.random.normal(0, STD_STRAY, N_RUNS * param_vector_size).reshape((N_RUNS, param_vector_size)).astype(np.float32) + old_top_vector).clip(0, 1)

old_top_rectangles = [Rectangle(center_x=0.43007988, center_y=0.37300894, width=0.571716, height=0.45852715, rotate_angle=0.6679186), Rectangle(center_x=0.7524599, center_y=0.7510932, width=0.51635915, height=0.49861547, rotate_angle=0.0), Rectangle(center_x=0.24755667, center_y=0.7460904, width=0.49858105, height=0.50419086, rotate_angle=1.0)]
old_top_vector = rectangles_to_vector(old_top_rectangles)

# optimize
# for i in range(n_rectangles):
optimized_rectangle_index = 0
fixed_params = np.concatenate([old_top_vector[:5 * optimized_rectangle_index], old_top_vector[5 * (optimized_rectangle_index + 1):]])
objective_function = partial(fixed_params_error, variable_rectangle_index=optimized_rectangle_index, fixed_param_vector=fixed_params, target_image=subsampled_target_image, num_points=N_POINTS, starting_point=starting_point, selected_indices=selected_indices)
initial_variable_params = np.random.uniform(0, 1, 5 * N_RUNS).reshape(N_RUNS, 5).astype(np.float32)
args = [initial_variable_params[i] for i in range(N_RUNS)]
variable_param_vectors, error_courses, scores = parallel_gradient_descent(objective_function, args, learning_rate=LEARNING_RATE, num_steps=NUM_STEPS, gradient_approximation_epsilon=GRADIENT_APPROXIMATION_EPSILON)
# glue together the param vectors again
param_vectors = [glue_params(variable_param_vector, optimized_rectangle_index, fixed_params) for variable_param_vector in variable_param_vectors]

# args = [initial_param_vectors[i] for i in range(N_RUNS)]
# objective_function = partial(error, target_image=subsampled_target_image, num_points=N_POINTS, starting_point=starting_point, selected_indices=selected_indices)
# param_vectors, error_courses, scores = parallel_gradient_descent(objective_function, args, learning_rate=LEARNING_RATE, num_steps=NUM_STEPS, gradient_approximation_epsilon=GRADIENT_APPROXIMATION_EPSILON)
# see what we optimized
for rank, (score, param_vector) in enumerate(zip(scores, param_vectors), 1):
    print(f"Top {rank} score: {score}, rectangles: {vectors_to_rectangles(param_vector)}")
# plot the errors
plotted_courses = error_courses[::4]
y_min = np.min(plotted_courses)
y_max = np.quantile(plotted_courses, 0.99)
for i, error_course in enumerate(plotted_courses):
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
best_param_vector = param_vectors[0]
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
