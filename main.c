#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(void)
{
    int w, h, channels;
    uint8_t* img = stbi_load("input.png", &w, &h, &channels, 1);
    if (img == NULL) {
        printf("Could not load input.png\n");
        return 1;
    }
    printf("Loaded %d x %d\n", w, h);

    uint8_t* img_out = (uint8_t*)calloc((size_t)w * h, 1);
    if (img_out == NULL) {
        printf("Memory not allocated.\n");
        stbi_image_free(img);
        return 1;
    }

    for (int y = 1; y <= h - 2; y++) {
        for (int x = 1; x <= w - 2; x++) {
            int gx =
                -1 * img[(y - 1) * w + (x - 1)] + 1 * img[(y - 1) * w + (x + 1)] +
                -2 * img[(y)*w + (x - 1)] + 2 * img[(y)*w + (x + 1)] +
                -1 * img[(y + 1) * w + (x - 1)] + 1 * img[(y + 1) * w + (x + 1)];

            int gy =
                -1 * img[(y - 1) * w + (x - 1)] + 1 * img[(y + 1) * w + (x - 1)] +
                -2 * img[(y - 1) * w + (x)] + 2 * img[(y + 1) * w + (x)] +
                -1 * img[(y - 1) * w + (x + 1)] + 1 * img[(y + 1) * w + (x + 1)];

            if (gx < 0) gx = -gx;
            if (gy < 0) gy = -gy;

            int mag = gx + gy;
            if (mag > 255) mag = 255;
            img_out[y * w + x] = (uint8_t)mag;
        }
    }

    stbi_write_png("output.png", w, h, 1, img_out, w);
    printf("Wrote output.png\n");

    stbi_image_free(img);
    free(img_out);
    return 0;
}