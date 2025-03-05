#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // For roundf()
#include <stdint.h>  // For type uint8_t
#include <immintrin.h>  // For AVX2 intrinsics
#include <omp.h>  //  For multithreading
#include "image.h"
#include "pool.h"
#include "filters.h"


static const double CONST_PI = 3.141592653589793f;


void print_window(struct Window *window) {
        printf("\n");
        for (int y = 0; y < window->size; y++) {
		for (int x = 0; x < window->size; x++) {
			printf(" %.2f ", window->entries[y*window->size + x]);
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

        // Create a Window struct (on heap) and initialize size field
        struct Window *window = (struct Window*)allocate_from_pool(pool, sizeof(struct Window));
        if (window == NULL) {
                fprintf(stderr, "\nFatal error: could not allocate memory for Window structure.\n");
		return NULL;
        }
        window->size = windowSize;

        // Allocate memory for the entries array from the memory pool
        window->entries = (float*)allocate_from_pool(pool, (windowSize*windowSize)*sizeof(float));
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

                        if (windowY < 0 || windowY >= imageHeight || windowX < 0 || windowX >= imageWidth) {
                                window->entries[windowIndex] = 0; // Apply zero-padding for out-of-bounds indices
                        } 
                        else {
                                // Determine the index of the Image's channelArray and capture into the Window's entries array
                                int imageChannelIndex = (windowY * imageWidth) + windowX;
                                window->entries[windowIndex] = imageChannelArray[imageChannelIndex];
                        }

                        windowIndex++;  // Move to the next element in the window's entries array 
                }
        }

        return window;

}


void shift_window_right(int y, int x, struct Window *window, int imageHeight, int imageWidth, uint8_t *inputChannelArray) {

        // Initialize useful values
        int windowSize = window->size;
        int halfWindowSize = windowSize / 2;
        int xRightmost = (x + halfWindowSize) + 1;  // The rightmost column of the shifted window 

        // Shift all columns of the window one step to the left to reuse data locally
        // Loop over each row of the window and use memmove (highly optimized compared to brute force loop)
        for (int windowRow = 0; windowRow < windowSize; windowRow++) {
                memmove(&window->entries[windowRow*windowSize],  // Destination (first column)
                        &window->entries[windowRow*windowSize + 1],  // Source (second column)
                        (windowSize-1)*sizeof(*window->entries));  // Size, in bytes, of data to move
        }

        // Initialize index for inserting data into the rightmost column of the window
        int colInsertIndex = windowSize - 1;

        // Loop to insert data into the new rightmost column data (or zero-padding if out of bounds)
        for (int row = y - halfWindowSize; row <= y + halfWindowSize; row++) {
                
                if (row < 0 || row >= imageHeight || xRightmost < 0 || xRightmost >= imageWidth) {
                        window->entries[colInsertIndex] = 0;  // Zero-padding for out-of-bounds pixels
                } else {  
                        // Determine the index of the Image's channelArray and capture into the Window's entries array
                        int imageIndex = (row * imageWidth) + xRightmost;
                        window->entries[colInsertIndex] = (float) inputChannelArray[imageIndex];
                }

                colInsertIndex += windowSize;  // Move to the next row's rightmost column
        }

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
        #ifdef _WIN32
                // For Windows and MinGW, use _aligned_malloc
                kernel->entries = (float*)_aligned_malloc((kernel->size*kernel->size)*sizeof(float), MEMORY_ALIGNMENT);
                if (kernel->entries == NULL) {
                        free(kernel);
                        fprintf(stderr, "\nFatal error: could not allocate memory for kernel structure.\n");
                        return NULL;
                }
        #else
                // For POSIX systems (Linux, macOS), use posix_memalign
                if (posix_memalign(&kernel->entries, MEMORY_ALIGNMENT, (kernel->size*kernel->size)*sizeof(float)) != 0) {
                        free(kernel);
                        fprintf(stderr, "\nFatal error: could not allocate memory for kernel structure.\n");
                        return NULL;
                }
        #endif

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


// Frees kernel struct properly (for specific memory alignment functions)
void free_kernel(struct Kernel *kernel) {
        
#ifdef _WIN32
        // For Windows and MinGW, use _aligned_free
        _aligned_free(kernel->entries);
#else
        // For POSIX systems, use free
        free(kernel->entries);
#endif
        free(kernel);
}


// Use AVX2 vectorization (SIMD intrinsics) to boost convolution algorithm on kernel and window
uint8_t compute_convolution(float *kernelEntriesArray, float *windowEntriesArray, int arrayLength) {

        // Set an AVX2 register (of type single precision floating point) to 0
        __m256 temp_ps = _mm256_setzero_ps();

        // Loop over kernel and window entries in increments of 8 (for AVX2)
        int i = 0;
        for (; i < arrayLength - 8; i += 8) {

                // Load 8 bytes (single precison floating point) into an AVX2 register from the kernel entries array
                __m256 kernel_reg_ps = _mm256_load_ps(&(kernelEntriesArray[i]));

                // Load 8 bytes (single precison floating point) into an AVX2 register from the window entries array
                __m256 window_reg_ps = _mm256_load_ps(&(windowEntriesArray[i]));

                // Use fused-multiply add instruction to multiply kernel and window entries and add to temp_ps register
                temp_ps = _mm256_fmadd_ps(kernel_reg_ps, window_reg_ps, temp_ps);

        }

        // Logic for reducing the temp_ps register (summing all of its single precision floating point elements)
        temp_ps = _mm256_hadd_ps(temp_ps, temp_ps);  // Apply horizontal add instruction to temp_ps with itself
        temp_ps = _mm256_hadd_ps(temp_ps, temp_ps);  // Apply horizontal add instruction to temp_ps with itself
        __m256 tempFlip_ps = _mm256_permute2f128_ps(temp_ps, temp_ps, 1);  // Flip the temp_ps register
        temp_ps = _mm256_add_ps(temp_ps, tempFlip_ps);  // Add the tempFlip_ps regsiter with the regular temp_ps register

        // Retrieve the lower (first) single precision floating point element of the temp_ps register
        float convolutionResult = _mm256_cvtss_f32(temp_ps); 

        // Loop over the remaining kernel and window entries
        for (; i < arrayLength; i++) {
                convolutionResult += (kernelEntriesArray[i] * windowEntriesArray[i]);
        }

        // Return the clamped and rounded convolution result
        if (convolutionResult <= 0) {
                return 0;
        } else if (convolutionResult >= 255) {
                return 255;
        } else { 
                return (uint8_t) roundf(convolutionResult);
        }

}


// Carries out the parallized convolution pipeline for given channelsArray of input image and stores result into output image
int apply_convolution_pipeline_channel(uint8_t *inputChannels, uint8_t *outputChannels, struct Kernel *kernel, int imageHeight, 
        int imageWidth) {

        // Initialize useful values
        int windowSize = kernel->size;
        int haloSize = windowSize / 2;
        int tileSize = 64 - 2*haloSize;
        int windowEntriesArrayLength = windowSize*windowSize; 

        // Flag to indicate error in the parallel processing
        int errorFlag = 0;

        // Parallelize over tiles in row-major order
        #pragma omp parallel for collapse(2) schedule(static) shared(errorFlag)
        for (int yy = 0; yy < imageHeight; yy += tileSize) {
                for (int xx = 0; xx < imageWidth; xx += tileSize) {

                        // Create a MemoryPool to store one Window struct and its entries array PER TILE
                        int alignedWindowSize = memory_size_alignment(sizeof(struct Window)) +
                                                memory_size_alignment(sizeof(float)*(windowSize*windowSize));
                        struct MemoryPool *pool = init_memory_pool(alignedWindowSize);
                        if (pool == NULL) {
                                #pragma omp atomic write
                                errorFlag = 1;
                                continue; // Skip to the next iteration
                        }
                        

                        // Loop over the channels in current tile in row-major order
                        for (int y = yy; y < (yy + tileSize) && y < imageHeight; y++) {

                                // Create a Window struct centered at the start of the current row in the tile
                                struct Window *window; 
                                window = create_window(y, xx, windowSize, imageHeight, imageWidth, inputChannels, pool);
                                if (window == NULL) {
                                        #pragma omp atomic write
                                        errorFlag = 1;
                                        break; // Exit the processing of this tile
                                }

                                for (int x = xx; x < (xx + tileSize) && x < imageWidth; x++) {

                                        // Compute the convolution between the window and kernel and capture into output image struct
                                        int channelIndex = (y * imageWidth) + x;
                                        outputChannels[channelIndex] = compute_convolution(kernel->entries, window->entries, 
                                                windowEntriesArrayLength);
                                        
                                        // Shift the Window right if not currently at last column in the tile
                                        if (x < (xx + tileSize - 1) && x < (imageWidth - 1)) {
                                                shift_window_right(y, x, window, imageHeight, imageWidth, inputChannels);
                                        }
                                
                                }

                                empty_pool(pool);  // Empty the memory pool to "free" the window 
                        }

                        release_entire_memory_pool(pool);  // Completely free the memory pool struct
                }
        }

        // Check if an error occurred during the parallel processing and return 0
        if (errorFlag) return 0;

        // Indicate that convolution pipeline executed successfully for given channel
        return 1;

}


// Applies the image_convolution_pipeline_channel for each of three (RGB) channels of a given image
int apply_convolution_pipeline_RGB(struct ImageRGB *inputImage, struct ImageRGB *outputImage, struct Kernel *kernel) {

        // Initializing useful values
        int imageHeight = inputImage->height;
        int imageWidth = inputImage->width;
        int windowSize = kernel->size;

        // Apply convolution pipeline for redChannels array of the input image struct
        int convolutionRed = apply_convolution_pipeline_channel(inputImage->redChannels, outputImage->redChannels,
                kernel, imageHeight, imageWidth);        
        if (convolutionRed == 0) return 0;

        // Apply convolution pipeline for greenChannels array of the input image struct
        int convolutionGreen = apply_convolution_pipeline_channel(inputImage->greenChannels, outputImage->greenChannels,
                kernel, imageHeight, imageWidth);        
        if (convolutionGreen == 0) return 0;

        // Apply convolution pipeline for blueChannels array of the input image struct
        int convolutionBlue = apply_convolution_pipeline_channel(inputImage->blueChannels, outputImage->blueChannels,
                kernel, imageHeight, imageWidth);        
        if (convolutionBlue == 0) return 0;

        // Indicate that convolution pipeline executed successfully for all channels
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
        int convolutionPipeline = apply_convolution_pipeline_RGB(*inputImage, outputImage, kernel);
        if (convolutionPipeline == 0) {
                free_imageRGB(*inputImage); *inputImage = NULL; 
                free_imageRGB(outputImage);
                free_kernel(kernel); 
                return NULL;
        }

        // Free and nullify the input image struct, free the kernel struct
        free_imageRGB(*inputImage); *inputImage = NULL; 
        free_kernel(kernel);

        return outputImage;

}