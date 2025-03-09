#ifndef FILTERS_H
#define FILTERS_H


#include "convolution.h"  // For enum GaussianBlurIntensity


// Enumeration for the different filter types 
typedef enum TypeFilter {
        FILTER_GREYSCALE,
        FILTER_GAUSSIAN_BLUR,
        FILTER_BOX_BLUR,
        FILTER_EMBOSS,
        FILTER_SHARPEN,
        FILTER_SOBEL_EDGE_DETECTION,
        FILTER_INVALID
} TypeFilter;


// Applies the greyscale filter to an RGB image. Saves results in a created ImageOneChannel struct and frees the input image
struct ImageOneChannel *apply_filter_greyscale(struct ImageRGB **inputImage);

// Applies a generic convolution based filter (e.g. emboss, sharpen) on an input image. Both input and output image are RGB
struct ImageRGB *apply_filter_generic_convolution(struct ImageRGB **inputImage, enum TypeFilter typeFilter, enum GeneralFilterIntensity filterIntensity);

// Applies the sobel operator filter to an RGB image. Saves results in a created ImageOneChannel struct and frees the input image
struct ImageOneChannel *apply_filter_sobel_edge_detection(struct ImageRGB **inputImage, enum GeneralFilterIntensity filterIntensity);







#endif //FILTERS_H