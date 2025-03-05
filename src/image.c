#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "image.h"



// Constant global integer for number of channels per pixel in RGB Image format
const int RGB_NUM_CHANNELS = 3;



struct ImageRGB *load_imageRGB(const char *filename) {

        // Create an ImageRGB struct
        struct ImageRGB *image = (struct ImageRGB*)malloc(sizeof(struct ImageRGB));
        if (image == NULL) {
                fprintf(stderr, "\nFatal error: image could not be loaded.\n\n");
		return NULL;
        }

        // Load image data in RGB format into a temporary array
        int width, height, numChannels;
        uint8_t *tempArray = stbi_load(filename, &width, &height, &numChannels, RGB_NUM_CHANNELS);
        if (tempArray == NULL) {
                free(image);
                fprintf(stderr, "\nFatal error: image could not be loaded. Reason: %s.\n\n", stbi_failure_reason());
		return NULL;
        }
        
        // Initialize ImageRGB struct fields
        image->width = width;
        image->height = height;
        image->numChannels = RGB_NUM_CHANNELS;

        // Allocate a single contiguous memory block for SoA channel layout
        uint8_t *pixelMemory = (uint8_t*)malloc(((height * width)*RGB_NUM_CHANNELS)*sizeof(uint8_t));
        if (pixelMemory == NULL) {
                free(image);
                stbi_image_free(tempArray);
                fprintf(stderr, "\nFatal error: image could not be loaded.\n\n");
		return NULL;
        }

        // Assign channel pointers
        image->redChannels = pixelMemory;
        image->greenChannels = pixelMemory + (width * height);
        image->blueChannels = pixelMemory + (2 * (width * height));

        // Convert from AoS channel layout (RGBRGBRGB) to SoA channel layout (RRRGGGBBB)
        for (int i = 0; i < (height * width); i++) {

                int tempIndex = i * RGB_NUM_CHANNELS; // Index in tempArray

                image->redChannels[i] = tempArray[tempIndex];
                image->greenChannels[i] = tempArray[tempIndex+1];
                image->blueChannels[i] = tempArray[tempIndex+2];

        }

        // Free temporary array
        stbi_image_free(tempArray);

        return image;

}



struct ImageRGB *load_empty_imageRGB(int width, int height) {
        
        // Create an ImageRGB struct
        struct ImageRGB *image = (struct ImageRGB*)malloc(sizeof(struct ImageRGB));
        if (image == NULL) {
                fprintf(stderr, "\nFatal error: empty image could not be loaded.\n\n");
		return NULL;
        }

        // Initialize ImageRGB struct fields
        image->width = width;
        image->height = height;
        image->numChannels = RGB_NUM_CHANNELS;

        // Allocate a single contiguous memory block for SoA channel layout
        uint8_t *pixelMemory = (uint8_t*)malloc(((width*height)*RGB_NUM_CHANNELS)*sizeof(uint8_t));
        if (pixelMemory == NULL) {
                free(image);
                fprintf(stderr, "\nFatal error: empty image could not be loaded.\n\n");
		return NULL;
        }

        // Assign channel pointers
        image->redChannels = pixelMemory;
        image->greenChannels = pixelMemory + (width * height);
        image->blueChannels = pixelMemory + (2 * (width * height));
        
        return image;

}



int save_imageRGB(struct ImageRGB *image, const char *filename, ImageFileType fileType) {

        // Validate Image struct parameter
        if (image == NULL || image->redChannels == NULL || image->greenChannels == NULL || image->blueChannels == NULL) {
                fprintf(stderr, "\nFatal error: image could not be saved.\n\n");
		return 0;
        }

        // Allocate a single contiguous memory block for AoS channel layout
        uint8_t *tempArray = (uint8_t*)malloc(((image->height * image->width)*RGB_NUM_CHANNELS)*sizeof(uint8_t));
        if (tempArray == NULL) {
                free(tempArray);
                fprintf(stderr, "\nFatal error: image could not be saved.\n\n");
		return 0;
        }
        
        // Convert from SoA channel layout (RRRGGGBBB) to AoS channel layout (RGBRGBRGB)
        for (int i = 0; i < (image->height * image->width); i++) {
                
                int tempIndex = i * RGB_NUM_CHANNELS; // Index in tempArray

                tempArray[tempIndex] = image->redChannels[i];
                tempArray[tempIndex+1] = image->greenChannels[i];
                tempArray[tempIndex+2] = image->blueChannels[i];
                
        }

        // Switch-case statement for saving images to different file types
        int imageWrite = 0;
        switch (fileType) {
                case FILE_TYPE_PNG:
                        int strideBytes = image->width * image->numChannels;
                        imageWrite = stbi_write_png(filename, image->width, image->height, image->numChannels, tempArray, strideBytes);
                        break;
                case FILE_TYPE_JPG:
                        int jpgQuality = 100; // For same image quality compared to png and bmp
                        imageWrite = stbi_write_jpg(filename, image->width, image->height, image->numChannels, tempArray, jpgQuality);
                case FILE_TYPE_BMP:
                        imageWrite = stbi_write_bmp(filename, image->width, image->height, image->numChannels, tempArray);
        }
        if (imageWrite == 0) {
		fprintf(stderr, "\nFatal error: Image could not be saved. Reason: %s.\n\n", stbi_failure_reason());
		return 0;
	}

        // Free temporary array
        free(tempArray);

        return 1;

}



void free_imageRGB(struct ImageRGB *image) {
        
        // Free the contiguous data array if it is not already NULL
        if (image->redChannels != NULL) {
                free(image->redChannels);
                image->redChannels = NULL;
                image->greenChannels = NULL;
                image->blueChannels = NULL;
        }

        // Free the image structure itself if it is not already NULL
        if (image != NULL) {
                free(image);
        }

}