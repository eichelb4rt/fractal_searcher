# linux version (compiler directives for cython compiler)
# distutils: language = c++
# distutils: sources = gen_fractal.cpp
# distutils: extra_compile_args = -O3 -ffast-math -march=native -fopenmp
# distutils: extra_link_args = -fopenmp
# cython: language_level = 3

import numpy as np
import numpy.typing as npt

cdef extern from "gen_fractal.hpp":
    struct gen_fractal_result:
        float* points
        unsigned int n_points

    void free_gen_fractal_result(gen_fractal_result* result)
    gen_fractal_result* gen_fractal_c(const float* function_system, unsigned int n_points, float start_x, float start_y, const unsigned int* selected_indices)


def gen_fractal_wrapper(float[:] function_system, n_points: int, starting_point: npt.NDArray[np.float32], const unsigned int[:] selected_indices) -> npt.NDArray[np.float32]:
    assert selected_indices.shape[0] == n_points
    cdef gen_fractal_result* result = gen_fractal_c(&function_system[0], n_points, starting_point[0], starting_point[1], &selected_indices[0])
    points = np.empty((n_points, 2), dtype=np.float32)
    points[:, :] = <float[:n_points, :2]> result.points
    free_gen_fractal_result(result)
    return points