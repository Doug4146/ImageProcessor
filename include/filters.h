#ifndef FILTERS_H
#define FILTERS_H


#include <stdint.h> // For type uint8_t
#include "pool.h"  // For struct MemoryPool


/**
 * @brief Structure for representing a window (square matrix) used for convolution-based filtering.
 * Contains fields for matrix size and a pointer array of matrix entries (uint8_t).
 */
typedef struct Window {
        int size;
        uint8_t *entries;
} Window;


/**
 * @brief Structure for representing a kernel (square matrix) used for convolution-based filtering.
 * Contains fields for matrix size and a pointer array of matrix entries (float).
 */
typedef struct Kernel {
        int size;
        float *entries;
} Kernel;


/**
 * @brief Enumeration for the different intensity levels of the gaussian blur filter. Each intensity level
 * is assigned a value representing the matrix size of the gaussian kernel. 
 */
typedef enum GaussianBlurIntensity {
        BLUR_INTENSITY_VERYLIGHT = 3,  // size = 03, stddev = 0.8
        BLUR_INTENSITY_LIGHT = 5,      // size = 05, stddev = 1
        BLUR_INTENSITY_MEDIUM = 9,     // size = 09, stddev = 2
        BLUR_INTENSITY_HIGH = 13,      // size = 13, stddev = 3
        BLUR_INTENSITY_VERYHIGH = 25,  // size = 25, stddev = 4
} GaussianBlurIntensity;



/**
 * @brief Utility function that prints out the elements of the `entries` array in a `Window` structure
 * in rectangular form.
 * @param window Pointer to the Window structure to print.
 */
void print_window(struct Window *window);

/**
 * @brief Utility function that prints out the elements of the `entries` array in a `Kernelx` structure
 * in rectangular form. 
 * @param Kernel Pointer to the Kernel structure to print.
 */
void print_kernel(struct Kernel *kernel);


struct Window *create_window(int y, int x, int windowSize, int imageHeight, int imageWidth, uint8_t *imageChannelArray, struct MemoryPool *pool);


//void shift_window_right(struct ImageRGB *image, struct Window *window, int x, int y, enum ChannelTypeRGB channelType);


struct Kernel *create_gaussian_kernel(enum GaussianBlurIntensity blurIntensity);


uint8_t compute_convolution(struct Kernel *kernel, struct Window *window);


int apply_convolution_pipeline(struct ImageRGB *inputImage, struct ImageRGB *outputImage , struct Kernel *kernel);


struct ImageRGB *apply_filter_gaussian_blur(struct ImageRGB **inputImage, enum GaussianBlurIntensity blurIntensity);




#endif //FILTERS_H