#include "compute_image.hpp"
#include <cstdlib>

void free_image(Image* image) {
    free(const_cast<float*> (image->data));
    delete image;
}

Image* compute_image_c(const long* filled_pixels, unsigned int n_filled_pixels, unsigned int width, unsigned int height) {
    auto data = (float*) calloc(width * height, sizeof(float));
    for (unsigned int i = 0; i < n_filled_pixels; ++i) {
        unsigned int x = filled_pixels[2 * i];
        unsigned int y = filled_pixels[2 * i + 1];
        data[y * width + x] = 1.0f;
    }
    return new Image{ width, height, data };
}