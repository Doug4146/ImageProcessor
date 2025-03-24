#include <stdio.h>
#include <math.h>  // For roundf()
#include <immintrin.h>  // For AVX2 support
#include <omp.h>  // For parallel processing
#include <stdint.h>  // For uint8_t
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

        // Grayscale channel weights (scaled to int16 for precision)
        int16_t redWeight_int16 = (int16_t) (0.299 * 128);
        int16_t greenWeight_int16 = (int16_t) (0.587 * 128);
        int16_t blueWeight_int16 = (int16_t) (0.114 * 128);

        // Load weights into 256 bit AVX2 registers
        __m256i redWeight_vec16i = _mm256_set1_epi16(redWeight_int16);
        __m256i greenWeight_vec16i = _mm256_set1_epi16(greenWeight_int16);
        __m256i blueWeight_vec16i = _mm256_set1_epi16(blueWeight_int16);

        // Parallelize the loop over image PIXELS in row-major order
        #pragma omp parallel for schedule(static)
        for (int pixelY = 0; pixelY < height; pixelY++) {
                
                int pixelX = 0;
                for (; pixelX + 16 < width; pixelX += 16) {
                        
                        // Index in the channels/pixels array
                        int index = (pixelY * width) + pixelX;

                        // Load 16 pixels from each channel of type uint8 and convert to int16
                        __m256i redChannels_vec16i = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*) &(*inputImage)->redChannels[index]));
                        __m256i greenChannels_vec16i = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*) &(*inputImage)->greenChannels[index]));
                        __m256i blueChannels_vec16i = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*) &(*inputImage)->blueChannels[index]));

                        // Perform the greycale operation
                        __m256i greyscale_vec16i = _mm256_add_epi16(
                                _mm256_add_epi16(
                                        _mm256_mullo_epi16(redChannels_vec16i, redWeight_vec16i),
                                        _mm256_mullo_epi16(greenChannels_vec16i, greenWeight_vec16i)
                                ),
                                _mm256_mullo_epi16(blueChannels_vec16i, blueWeight_vec16i)
                        );

                        // Normalize results by dividing greyscale values by 128 (right bitshift by 7)
                        greyscale_vec16i = _mm256_srli_epi16(greyscale_vec16i, 7);

                        // Convert int16 result values to clamped uint8 values
                        greyscale_vec16i = _mm256_packus_epi16(greyscale_vec16i, 
                                _mm256_permute2x128_si256(greyscale_vec16i, greyscale_vec16i, 0x11));
                        
                        // Extract the first 128 bits (16 uint8) from result vector and store into output image
                        __m128i greyscale_vec8u = _mm256_extracti128_si256(greyscale_vec16i, 0);
                        _mm_storeu_si128((__m128i*) &(outputImage->pixels[index]), greyscale_vec8u);

                }

                // Process the remaining pixels
                for (; pixelX < width; pixelX++) {
                        
                        int index = (pixelY * width) + pixelX;

                        int redChannel = (*inputImage)->redChannels[index];
                        int greenChannel = (*inputImage)->greenChannels[index];
                        int blueChannel = (*inputImage)->blueChannels[index];

                        float greyscaleResult = 0.299*redChannel + 0.587*greenChannel + 0.114*blueChannel;

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
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++) {

                int x = 0;
                for (; x + 16 < width; x += 16) {

                        // Index in the pixels array
                        int index = (y * width) + x;

                        // Load 16 pixels, of type uint8, from each tempImage and convert to int16
                        __m256i tempOnePixels_vec16i = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*) &(tempImageOne->pixels[index])));
                        __m256i tempTwoPixels_vec16i = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*) &(tempImageTwo->pixels[index])));

                        // Extract first 8 pixels (currently int16) and convert to precision-single floating point for each tempImage
                        __m256i tempOnePixelsFirstEight_32i = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tempOnePixels_vec16i, 0));
                        __m256i tempTwoPixelsFirstEight_32i = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tempTwoPixels_vec16i, 0));
                        __m256 tempOnePixelsFirstEight_ps = _mm256_cvtepi32_ps(tempOnePixelsFirstEight_32i);
                        __m256 tempTwoPixelsFirstEight_ps = _mm256_cvtepi32_ps(tempTwoPixelsFirstEight_32i);

                        // Extract last 8 pixels (currently int16) and convert to precision-single floating point for each tempImage
                        __m256i tempOnePixelsLastEight_32i = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tempOnePixels_vec16i, 1));
                        __m256i tempTwoPixelsLastEight_32i = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tempTwoPixels_vec16i, 1));
                        __m256 tempOnePixelsLastEight_ps = _mm256_cvtepi32_ps(tempOnePixelsLastEight_32i);
                        __m256 tempTwoPixelsLastEight_ps = _mm256_cvtepi32_ps(tempTwoPixelsLastEight_32i);

                        // Perform the "magnitude" operation for each group of 8 pixels (currently precision-single floating point)
                        __m256 lowerHalf_ps = _mm256_sqrt_ps(
                                _mm256_add_ps(
                                        _mm256_mul_ps(tempOnePixelsFirstEight_ps, tempOnePixelsFirstEight_ps),
                                        _mm256_mul_ps(tempTwoPixelsFirstEight_ps, tempTwoPixelsFirstEight_ps)
                                )
                        );
                        __m256 upperHalf_ps = _mm256_sqrt_ps(
                                _mm256_add_ps(
                                        _mm256_mul_ps(tempOnePixelsLastEight_ps, tempOnePixelsLastEight_ps),
                                        _mm256_mul_ps(tempTwoPixelsLastEight_ps, tempTwoPixelsLastEight_ps)
                                )
                        );

                        // Convert each half from precision-single floating point to int32 and clamp
                        __m256i lowerHalf_32i = _mm256_cvtps_epi32(lowerHalf_ps);
                        lowerHalf_32i = _mm256_min_epi32(_mm256_max_epi32(lowerHalf_32i, _mm256_set1_epi32(0)), _mm256_set1_epi32(255));
                        __m256i upperHalf_32i = _mm256_cvtps_epi32(upperHalf_ps);
                        upperHalf_32i = _mm256_min_epi32(_mm256_max_epi32(upperHalf_32i, _mm256_set1_epi32(0)), _mm256_set1_epi32(255));

                        // Convert each half from int32 to int16 values
                        __m128i lowerHalf_16i = _mm256_extracti128_si256(_mm256_packus_epi32(lowerHalf_32i, _mm256_permute2x128_si256(lowerHalf_32i, lowerHalf_32i, 0x11)), 0);
                        __m128i upperHalf_16i = _mm256_extracti128_si256(_mm256_packus_epi32(upperHalf_32i, _mm256_permute2x128_si256(upperHalf_32i, upperHalf_32i, 0x11)), 0);

                         // Combine halves, convert result to clamped uint8 values, and store into output image
                        __m256i result_16i = _mm256_insertf128_si256(_mm256_castsi128_si256(lowerHalf_16i), upperHalf_16i, 1);
                        __m128i result_8u = _mm256_extracti128_si256(_mm256_packus_epi16(result_16i, _mm256_permute2x128_si256(result_16i, result_16i, 0x11)), 0);
                        _mm_storeu_si128((__m128i*) &(outputImage->pixels[index]), result_8u);
                                
                }

                // Process remaining pixels in row
                for (; x < width; x++) {

                        int index = y * width + x;

                        int magnitude = sqrt(pow(tempImageOne->pixels[index], 2) + pow(tempImageTwo->pixels[index], 2));

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



