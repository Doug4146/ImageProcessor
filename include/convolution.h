#ifndef CONVOLUTION_H
#define CONVOLUTION_H


#include <stdint.h>  // For type uint8_t
#include "image.h"  // For struct ImageRGB
#include "pool.h"  // For struct MemoryPool


/**
 * @brief Structure for representing a window (square matrix) used for convolution-based filtering.
 * Contains fields for matrix size and a pointer array of matrix entries (uint8_t).
 */
typedef struct Window {
        int size;
        float *entries;
} Window;


/**
 * @brief Structure for representing a kernel (square matrix) used for convolution-based filtering.
 * Contains fields for matrix size and a pointer array of matrix entries (float).
 */
typedef struct Kernel {
        int size;
        float *entries;
} Kernel;

// Enumeration for the general different intensity levels of common filters (sharpen, emboss, etc).
typedef enum GeneralFilterIntensity {
        FILTER_INTENSITY_LIGHT,      
        FILTER_INTENSITY_MEDIUM,     
        FILTER_INTENSITY_HIGH,
        FILTER_INTENSITY_INVALID
} GeneralFilterIntensity;





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
void shift_window_right(int y, int x, struct Window *window, int imageHeight, int imageWidth, uint8_t *inputChannelArray);


struct Kernel *create_gaussian_kernel(enum GeneralFilterIntensity filterIntensity);
struct Kernel *create_box_blur_kernel(enum GeneralFilterIntensity filterIntensity);

struct Kernel *create_sharpen_kernel(enum GeneralFilterIntensity filterIntensity);
struct Kernel *create_emboss_kernel(enum GeneralFilterIntensity filterIntensity);

struct Kernel *create_sobel_vertical_kernel(enum GeneralFilterIntensity filterIntensity);
struct Kernel *create_sobel_horizontal_kernel(enum GeneralFilterIntensity filterIntensity);
void free_kernel(struct Kernel *kernel);


uint8_t compute_convolution(float *kernelEntriesArray, float *windowEntriesArray, int arrayLength);
int apply_convolution_pipeline_channel(uint8_t *inputChannels, uint8_t *outputChannels, struct Kernel *kernel, int imageHeight, int imageWidth);
int apply_convolution_pipeline_RGB(struct ImageRGB *inputImage, struct ImageRGB *outputImage, struct Kernel *kernel);




#endif //CONVOLUTION_H