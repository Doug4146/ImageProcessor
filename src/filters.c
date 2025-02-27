#include <stdio.h>
#include <stdlib.h>
#include <math.h> // For round()
#include <stdint.h> // For type uint8_t
#include "image.h"
#include "pool.h"
#include "filters.h"


static const double CONST_PI = 3.141592653589793f;


void print_window(struct Window *window) {
        printf("\n");
        for (int y = 0; y < window->size; y++) {
		for (int x = 0; x < window->size; x++) {
			printf(" %d ", window->entries[y*window->size + x]);
		}
		printf("\n");
	}
        printf("\n"); 
}

void print_kernel(struct Kernel *kernel) {
        printf("\n");
        for (int y = 0; y < kernel->size; y++) {
		for (int x = 0; x < kernel->size; x++) {
			printf(" %.2f ", kernel->entries[y*kernel->size + x]);
		}
		printf("\n");
	}
        printf("\n"); 
}



struct Window *create_window(int y, int x, int windowSize, int imageHeight, int imageWidth, uint8_t *imageChannelArray, struct MemoryPool *pool) {

        // Create a Window struct and initialize size field
        struct Window *window = (struct Window*)allocate_from_pool(pool, sizeof(struct Window));
        if (window == NULL) {
                fprintf(stderr, "\nFatal error: could not allocate memory for Window structure.\n");
		return NULL;
        }
        window->size = windowSize;

        // Allocate memory for the entries array
        window->entries = (uint8_t*)allocate_from_pool(pool, sizeof(uint8_t)*(windowSize*windowSize));
        if (window->entries == NULL) {
                free_from_pool(pool, (void*) window,  sizeof(struct Window));
                fprintf(stderr, "\nFatal error: could not allocate memory for Window structure.\n");
		return NULL;
        }

        int halfWindowSize = windowSize / 2;  // Half the window size (used for offset calculations)
        int windowIndex = 0;  // Current index in the window's entries array

        // Loop over the defined window region of the image in row-major order
        for (int windowY = y - halfWindowSize; windowY <= y + halfWindowSize; windowY++) {
                for (int windowX = x - halfWindowSize; windowX <= x + halfWindowSize; windowX++) {

                        // Apply zero-padding for out-of-bounds indices
                        if (windowY < 0 || windowY >= imageHeight || windowX < 0 || windowX >= imageWidth) {
                                window->entries[windowIndex] = 0;
                        } 
                        // Determine the index of the Image's channelArray and capture into the Window's entries array
                        else {
                                int imageChannelIndex = (windowY * imageWidth) + windowX;
                                window->entries[windowIndex] = imageChannelArray[imageChannelIndex];
                        }

                        // Increment the index in the window's entries array 
                        windowIndex++;
                }
        }

        return window;

}



struct Kernel *create_gaussian_kernel(enum GaussianBlurIntensity blurIntensity) {

        // Create a Kernel struct and initialize size field
        struct Kernel *kernel = (struct Kernel*)malloc(sizeof(struct Kernel));
        if (kernel == NULL) {
                fprintf(stderr, "\nFatal error: could not allocate memory for kernel structure.\n");
		return NULL;
        }
        kernel->size = blurIntensity;

        // Allocate memory for the entries array
        kernel->entries = (float*)malloc((kernel->size*kernel->size)*sizeof(float));
        if (kernel->entries == NULL) {
                free(kernel);
                fprintf(stderr, "\nFatal error: could not allocate memory for kernel structure.\n");
		return NULL;
        }

        // Determine standard deviation value  
        float stddev;
        switch (blurIntensity) {
                case BLUR_INTENSITY_VERYLIGHT: 
                        stddev = 0.8; break;
                case BLUR_INTENSITY_LIGHT: 
                        stddev = 1; break;
                case BLUR_INTENSITY_MEDIUM: 
                        stddev = 2; break;
                case BLUR_INTENSITY_HIGH: 
                        stddev = 3; break;
                case BLUR_INTENSITY_VERYHIGH: 
                        stddev = 4; break;
        }

        int halfWindowSize = kernel->size / 2;  // Half the kernel size (used for offset calculations)
        float sumEntries = 0.0;  // For kernel normalization

        // Loop over the kernel's entries array in row-major order. The index of the center entry is
        // defined to be at i=0, j=0
        for (int j = -halfWindowSize; j <= halfWindowSize; j++) {
                for (int i = -halfWindowSize; i <= halfWindowSize; i++) {

                        // Apply the gaussian function to the current entry
                        float result = (1.0/(2.0*CONST_PI*stddev*stddev))*exp(-((i*i + j*j)/(2.0*stddev*stddev)));
                        sumEntries += result;

                        // Calculate index in the entries array and capture the result
                        int index = (j+halfWindowSize)*kernel->size + (i+halfWindowSize);
                        kernel->entries[index] = result;
                
                }
        }

        // Normalize the kernel
        for (int i = 0; i < kernel->size*kernel->size; i++) {
                kernel->entries[i] /= sumEntries;
        }

        return kernel;

}



uint8_t compute_convolution(struct Kernel *kernel, struct Window *window) {

        // Apply the convolution operation to the matrices
        float temp = 0.0f;
        for (int i = 0; i < kernel->size*kernel->size; i++) {
                temp += kernel->entries[i] * window->entries[i];
        }

        // Return the clamped and rounded convolution result
        if (temp <= 0) {
                return (uint8_t) 0;
        } else if (temp >= 255) {
                return (uint8_t) 255;
        } else {
                return (uint8_t) round(temp);
        }

}



int apply_convolution_pipeline(struct ImageRGB *inputImage, struct ImageRGB *outputImage, struct Kernel *kernel) {

        // Initializing useful values
        int imageHeight = inputImage->height;
        int imageWidth = inputImage->width;
        int windowSize = kernel->size;

        // Create a MemoryPool struct with enough preallocated memory to hold one Window struct and its entries array
        int alignedWindowSize = memory_size_alignment(sizeof(struct Window)) +
                                memory_size_alignment(sizeof(uint8_t)*(windowSize*windowSize));
        struct MemoryPool *pool = init_memory_pool(alignedWindowSize);
        if (pool == NULL) {
                return 0;
        }

        // Loop over the contiguous redChannels array of the input image struct in row-major order
        for (int channelY = 0; channelY < imageHeight; channelY++) {
                for (int channelX = 0; channelX < imageWidth; channelX++) {

                        // Create a red channel Window struct centered at the current channel
                        struct Window *windowRed; 
                        windowRed = create_window(channelY, channelX, windowSize, imageHeight, imageWidth, inputImage->redChannels, pool);
                        if (windowRed == NULL) return 0;

                        // Compute the convolution for each channel type and capture into output image struct
                        int channelIndex = (channelY * imageWidth) + channelX;
                        outputImage->redChannels[channelIndex] = compute_convolution(kernel, windowRed);

                        // Empty the memory pool to "free" the Window struct
                        empty_pool(pool);
                }
        }

        // Loop over the contiguous greenChannels array of the input image struct in row-major order
        for (int channelY = 0; channelY < imageHeight; channelY++) {
                for (int channelX = 0; channelX < imageWidth; channelX++) {

                        // Create a green channel Window struct centered at the current channel
                        struct Window *windowGreen; 
                        windowGreen = create_window(channelY, channelX, windowSize, imageHeight, imageWidth, inputImage->greenChannels, pool);
                        if (windowGreen == NULL) return 0;

                        // Compute the convolution for each channel type and capture into output image struct
                        int channelIndex = (channelY * imageWidth) + channelX;
                        outputImage->greenChannels[channelIndex] = compute_convolution(kernel, windowGreen);

                        // Empty the memory pool to "free" the Window struct
                        empty_pool(pool);
                }
        }

        // Loop over the contiguous greenChannels array of the input image struct in row-major order
        for (int channelY = 0; channelY < imageHeight; channelY++) {
                for (int channelX = 0; channelX < imageWidth; channelX++) {

                        // Create a blue channel Window struct centered at the current channel
                        struct Window *windowBlue; 
                        windowBlue = create_window(channelY, channelX, windowSize, imageHeight, imageWidth, inputImage->blueChannels, pool);
                        if (windowBlue == NULL) return 0;

                        // Compute the convolution for each channel type and capture into output image struct
                        int channelIndex = (channelY * imageWidth) + channelX;
                        outputImage->blueChannels[channelIndex] = compute_convolution(kernel, windowBlue);

                        // Empty the memory pool to "free" the Window struct
                        empty_pool(pool);
                }
        }

        // Completely free the memory allocated for the memory pool and nullify pointer
        release_entire_memory_pool(pool); pool = NULL;

        // Indicate that convolution pipeline executed successfully
        return 1;

}



struct ImageRGB *apply_filter_gaussian_blur(struct ImageRGB **inputImage, enum GaussianBlurIntensity blurIntensity) {

        // Verify input image parameter
        if (inputImage == NULL || *inputImage == NULL) {
                fprintf(stderr, "\nFatal error: input image structure could not be processed in the gaussian blur filter.\n");
                return NULL;
        }

        // Create a blank Image struct for the output image
        struct ImageRGB *outputImage = load_empty_imageRGB((*inputImage)->width, (*inputImage)->height);
        if (outputImage == NULL) {
                free_imageRGB(*inputImage); *inputImage = NULL;
                return NULL;
        }

        // Create a gaussian Kernel struct
        struct Kernel *kernel = create_gaussian_kernel(blurIntensity);
        if (kernel == NULL) {
                free_imageRGB(*inputImage); *inputImage = NULL; 
                free_imageRGB(outputImage);
                return NULL;
        }

        // Apply the convolution pipeline to the input image and capture the result in the output image
        int convolutionPipeline = apply_convolution_pipeline(*inputImage, outputImage, kernel);
        if (convolutionPipeline == 0) {
                free_imageRGB(*inputImage); *inputImage = NULL; 
                free_imageRGB(outputImage);
                free(kernel->entries); free(kernel);
                return NULL;
        }

        // Free and nullify the input image struct, free the kernel struct
        free_imageRGB(*inputImage); *inputImage = NULL; 
        free(kernel->entries); free(kernel);

        return outputImage;

}










