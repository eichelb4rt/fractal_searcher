#pragma once

struct gen_fractal_result {
    float* points;
    unsigned int n_points;
};

void free_gen_fractal_result(gen_fractal_result* result);
gen_fractal_result* gen_fractal_c(const float* function_system, unsigned int n_points, float start_x, float start_y, const unsigned int* selected_indices);