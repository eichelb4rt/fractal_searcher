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

    Image* compute_image_c(const unsigned int* filled_pixels, unsigned int n_filled_pixels, unsigned int width, unsigned int height)

def compute_image(float[:, :] points, unsigned int width, unsigned int height):
    cdef unsigned int[:, :] filled_pixels = np.round(points * np.array([width, height], dtype=np.float32)).clip(0, [width - 1, height - 1]).astype(np.uint32)
    assert filled_pixels.shape[1] == 2
    assert np.min(filled_pixels[:, 0]) >= 0 and np.max(filled_pixels[:, 0]) < width
    assert np.min(filled_pixels[:, 1]) >= 0 and np.max(filled_pixels[:, 1]) < height
    cdef unsigned int n_filled_pixels = filled_pixels.shape[0]
    cdef Image* image = compute_image_c(&filled_pixels[0, 0], n_filled_pixels, width, height)
    result = np.empty((image.height, image.width), dtype=np.float32)
    result[:, :] = <float[:height, :width]> image.data
    free_image(image)
    return result