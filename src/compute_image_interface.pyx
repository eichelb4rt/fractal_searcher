# linux version (compiler directives for cython compiler)
# distutils: language = c++
# distutils: sources = compute_image.cpp
# distutils: extra_compile_args = -O3 -ffast-math -march=native -fopenmp
# distutils: extra_link_args = -fopenmp
# cython: language_level = 3

import numpy as np

cdef extern from "compute_image.hpp":
    struct Image:
        int width
        int height
        float* data
    
    void free_image(Image* image)

    Image* compute_image_c(const long* filled_pixels, unsigned int n_filled_pixels, unsigned int width, unsigned int height)

def compute_image(const float[:, :] points, unsigned int width, unsigned int height):
    # NOTE: clip desperately wants to output long for some reason and shit goes to the fan if it's casted to uint
    cdef long[:, :] filled_pixels = np.round(points * np.array([width, height], dtype=np.float32)).clip(0, [width - 1, height - 1]).astype(np.int64).clip(0, [width - 1, height - 1])
    assert filled_pixels.shape[1] == 2
    assert np.min(filled_pixels[:, 0]) >= 0 and np.max(filled_pixels[:, 0]) < width, f"min x: {np.min(filled_pixels[:, 0])}, max x: {np.max(filled_pixels[:, 0])}, max allowed: {width}"
    assert np.min(filled_pixels[:, 1]) >= 0 and np.max(filled_pixels[:, 1]) < height, f"min y: {np.min(filled_pixels[:, 1])}, max y: {np.max(filled_pixels[:, 1])}, max allowed: {height}"
    cdef unsigned int n_filled_pixels = filled_pixels.shape[0]
    cdef Image* image = compute_image_c(&filled_pixels[0, 0], n_filled_pixels, width, height)
    result = np.empty((image.height, image.width), dtype=np.float32)
    result[:, :] = <float[:height, :width]> image.data
    free_image(image)
    return result