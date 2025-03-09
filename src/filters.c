#include <stdio.h>
#include <math.h>  // For roundf()
#include <immintrin.h>  // For AVX2 support
#include <omp.h>  // For parallel processing
#include "image.h"
#include "convolution.h"
#include "filters.h"



struct ImageOneChannel *apply_filter_greyscale(struct ImageRGB **inputImage) {

        // Verify input image parameter
        if (inputImage == NULL || *inputImage == NULL || (*inputImage)->redChannels == NULL ||
                        (*inputImage)->greenChannels == NULL || (*inputImage)->blueChannels == NULL ) {
                fprintf(stderr, "\nFatal error: input image structure could not be processed in the greyscale filter.\n");
                return NULL;
        }
        
        // Create a blank ImageOneChannel struct for the output image
        struct ImageOneChannel *outputImage = load_empty_imageOneChannel((*inputImage)->width, (*inputImage)->height);
        if (outputImage == NULL) {
                free_imageRGB(*inputImage); *inputImage = NULL;
                return NULL;
        }

        // Initialize useful values
        int width = (*inputImage)->width;
        int height = (*inputImage)->height;

        // Make the cache-handling better by prefetching and stuff (thats the best-ish that can do)
        // Add SIMD acceleration to process 8 greyscale calculations at one

        // Parallelize the loop over image PIXELS in row-major order
        #pragma omp parallel for
        for (int pixelY = 0; pixelY < height; pixelY++) {
                for (int pixelX = 0; pixelX < width; pixelX++) {

                        // Index in the channels/pixels array
                        int index = (pixelY * width) + pixelX;

                        // Initialize input images current RGB channel values
                        int redChannel = (*inputImage)->redChannels[index];
                        int greenChannel = (*inputImage)->greenChannels[index];
                        int blueChannel = (*inputImage)->blueChannels[index];

                        // Perform the greyscale calculation and capture the clamped and rounded value 
                        float greyscaleResult = (0.299*redChannel + 0.587*greenChannel + 0.114*blueChannel);

                        // Clamp and round the result
                        if (greyscaleResult <= 0) {
                                outputImage->pixels[index] = 0;
                        } else if (greyscaleResult >= 255) {
                                outputImage->pixels[index] = 255;
                        } else {
                                outputImage->pixels[index] = roundf(greyscaleResult);
                        }
                        
                }
        }

        // Free and nullify the input image struct
        free_imageRGB(*inputImage); *inputImage = NULL; 

        return outputImage;

}


struct ImageRGB *apply_filter_generic_convolution(struct ImageRGB **inputImage, enum TypeFilter typeFilter, enum GeneralFilterIntensity filterIntensity) {

        // Verify input image parameter
        if (inputImage == NULL || *inputImage == NULL || (*inputImage)->redChannels == NULL ||
                        (*inputImage)->greenChannels == NULL || (*inputImage)->blueChannels == NULL ) {
                fprintf(stderr, "\nFatal error: input image structure could not be processed in the blur filter.\n");
                free_imageRGB(*inputImage); *inputImage = NULL;
                return NULL;
        }

        // Create a blank Image struct for the output image
        struct ImageRGB *outputImage = load_empty_imageRGB((*inputImage)->width, (*inputImage)->height);
        if (outputImage == NULL) {
                free_imageRGB(*inputImage); *inputImage = NULL;
                return NULL;
        }

        // Create the desired filter's convolution kernel
        struct Kernel *kernel;
        switch (typeFilter) {
                case FILTER_GAUSSIAN_BLUR: kernel = create_gaussian_kernel(filterIntensity); break;
                case FILTER_BOX_BLUR:      kernel = create_box_blur_kernel(filterIntensity); break;
                case FILTER_EMBOSS:        kernel = create_emboss_kernel(filterIntensity);   break;
                case FILTER_SHARPEN:       kernel = create_sharpen_kernel(filterIntensity);  break;
        }
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


struct ImageOneChannel *apply_filter_sobel_edge_detection(struct ImageRGB **inputImage, enum GeneralFilterIntensity filterIntensity) {

        // Verify input image parameter
        if (inputImage == NULL || *inputImage == NULL || (*inputImage)->redChannels == NULL ||
                        (*inputImage)->greenChannels == NULL || (*inputImage)->blueChannels == NULL ) {
                fprintf(stderr, "\nFatal error: input image structure could not be processed in the greyscale filter.\n");
                free_imageRGB(*inputImage); *inputImage = NULL;
                return NULL;
        }

        // Initialize useful values
        int width = (*inputImage)->width;
        int height = (*inputImage)->height;

        // Create 2 blank temporary Image structs and one output blank image struct
        struct ImageOneChannel *outputImage = load_empty_imageOneChannel(width, height);
        struct ImageOneChannel *tempImageOne = load_empty_imageOneChannel(width, height);
        struct ImageOneChannel *tempImageTwo = load_empty_imageOneChannel(width, height);
        if (outputImage == NULL || tempImageOne == NULL || tempImageTwo == NULL) {
                free_imageRGB(*inputImage); *inputImage = NULL;
                free_imageOneChannel(outputImage); free_imageOneChannel(tempImageOne); free_imageOneChannel(tempImageTwo);
                return NULL;
        }

        // Apply the greyscale filter to the input RGB image, result will be an ImageOneChannel struct
        // This function also takes care of freeing and nullifying the input image
        struct ImageOneChannel *inputGreyscaleImage = apply_filter_greyscale(inputImage);
        if (inputGreyscaleImage == NULL) {
                free_imageOneChannel(outputImage); free_imageOneChannel(tempImageOne); free_imageOneChannel(tempImageTwo);
                return NULL;
        }

        // Create horizontal and vertical sobel kernels
        struct Kernel *horizontalSobel = create_sobel_horizontal_kernel(filterIntensity);
        struct Kernel *verticalSobel = create_sobel_vertical_kernel(filterIntensity);
        if (horizontalSobel == NULL || verticalSobel == NULL) {
                free_imageOneChannel(outputImage); free_imageOneChannel(tempImageOne); free_imageOneChannel(tempImageTwo);
                free_imageOneChannel(inputGreyscaleImage);
                free_kernel(horizontalSobel); free_kernel(verticalSobel);
                return NULL;
        }

        // Apply the horizontalSobel Kernel to the greyscaleInputImage and save results into tempImageOne
        int horizontalConvolution = apply_convolution_pipeline_channel(inputGreyscaleImage->pixels, tempImageOne->pixels,
                horizontalSobel, height, width);
        if (horizontalConvolution == 0) {
                free_imageOneChannel(outputImage); free_imageOneChannel(tempImageOne); free_imageOneChannel(tempImageTwo);
                free_imageOneChannel(inputGreyscaleImage);
                free_kernel(horizontalSobel); free_kernel(verticalSobel);
                return NULL;
        }
        
        // Apply the verticalSobel Kernel to the greyscaleInputImage and save results into tempImageTwo
        int verticalConvolution = apply_convolution_pipeline_channel(inputGreyscaleImage->pixels, tempImageTwo->pixels,
                verticalSobel, height, width);
        if (verticalConvolution == 0) {
                free_imageOneChannel(outputImage); free_imageOneChannel(tempImageOne); free_imageOneChannel(tempImageTwo);
                free_imageOneChannel(inputGreyscaleImage);
                free_kernel(horizontalSobel); free_kernel(verticalSobel);
                return NULL;
        }

        // Combine the effects of each sobel kernel
        // Optimize the balls out of this
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {

                        int index = y * width + x;

                        int magnitude = sqrt(pow(tempImageOne->pixels[index], 2) + pow(tempImageTwo->pixels[index], 2));

                        // Clamp and round the result
                        if (magnitude <= 0) {
                                outputImage->pixels[index] = 0;
                        } else if (magnitude >= 255) {
                                outputImage->pixels[index] = 255;
                        } else {
                                outputImage->pixels[index] = roundf(magnitude);
                        }

                }
        }

        // Free all structs
        free_imageOneChannel(tempImageOne); free_imageOneChannel(tempImageTwo);
        free_imageOneChannel(inputGreyscaleImage);
        free_kernel(horizontalSobel); free_kernel(verticalSobel);

        return outputImage;

}



