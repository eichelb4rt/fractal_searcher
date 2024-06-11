#pragma once

struct Image {
    unsigned int width;
    unsigned int height;
    float* data;
};

void free_image(Image* image);

Image* compute_image_c(const long* filled_pixels, unsigned int n_filled_pixels, unsigned int width, unsigned int height);
