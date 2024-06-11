#include "gen_fractal.hpp"
#include <cstdlib>
#include <stdio.h>

void free_gen_fractal_result(gen_fractal_result* result) {
    free(const_cast<float*> (result->points));
    delete result;

}
gen_fractal_result* gen_fractal_c(const float* function_system, unsigned int n_points, float start_x, float start_y, const unsigned int* selected_indices) {
    auto points = (float*) calloc(2 * n_points, sizeof(float));
    float x = start_x;
    float y = start_y;
    for (unsigned int i = 0; i < n_points; ++i) {
        unsigned int function_index = selected_indices[i];
        float matrix_11 = function_system[6 * function_index];
        float matrix_12 = function_system[6 * function_index + 1];
        float matrix_21 = function_system[6 * function_index + 2];
        float matrix_22 = function_system[6 * function_index + 3];
        float offset_1 = function_system[6 * function_index + 4];
        float offset_2 = function_system[6 * function_index + 5];
        float new_x = matrix_11 * x + matrix_12 * y + offset_1;
        float new_y = matrix_21 * x + matrix_22 * y + offset_2;
        x = new_x;
        y = new_y;
        points[2 * i] = x;
        points[2 * i + 1] = y;
    }
    return new gen_fractal_result{ points, n_points };
}