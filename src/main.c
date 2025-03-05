#include <stdio.h>
#include <windows.h> // For performance benchmarking
#include <math.h>
#include "image.h"
#include "filters.h"
#include "pool.h"


int main(int argc, char *argv[]) {


        const char *inputFilename = "..\\input\\HighQuality.jpg";
        const char *outputFilename = "..\\output\\OutputImage.png";


        // Performance benchmarking
	LARGE_INTEGER frequency, start, end;
	double elapsedTime = 0.0f;
	QueryPerformanceFrequency(&frequency); // Get the high-resolution counter's frequency (ticks per second)

        // Load image
        struct ImageRGB *image = load_imageRGB(inputFilename);
        if (image == NULL) {
                return 1;
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /////////////////////////     START OF CODE TO BENCHMARK     ///////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////


        QueryPerformanceCounter(&start); // Start timing

        image = apply_filter_gaussian_blur(&image, BLUR_INTENSITY_HIGH);
        if (image == NULL) {
                //fprintf(stderr, "\nGaussian blur failed.\n");
                return 0;
        }

        QueryPerformanceCounter(&end); // End timing
        elapsedTime += (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart; // Calculate elapsed time in seconds

        // For benchmark simple print
        printf("Runtime: %.10lf\n", elapsedTime);
        
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /////////////////////////     END OF CODE TO BENCHMARK     /////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Indicate image is being saved
        printf("Saving image...\n");

        // Save image (png for testing)
        int saveImage = save_imageRGB(image, outputFilename, FILE_TYPE_PNG);
        if (saveImage == 0) {
                free_imageRGB(image);
                return 1;
        }

        // Free image
        free_imageRGB(image);
        image = NULL;

        // Indicate image saved correctly
        printf("Image saved correctly.\n");

        // Successful program call
        return 0;

}