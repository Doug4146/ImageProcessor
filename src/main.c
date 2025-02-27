#include <stdio.h>
// #include <windows.h> // For performance benchmarking
#include <math.h>
#include "image.h"
#include "filters.h"
#include "pool.h"


int main(int argc, char *argv[]) {


        const char *inputFilename = "..\\input\\NewYork.jpg";
        const char *outputFilename = "..\\output\\outputImage.png";


        // // Performance benchmarking
	// LARGE_INTEGER frequency, start, end;
	// double elapsedTime = 0.0f;
	// QueryPerformanceFrequency(&frequency); // Get the high-resolution counter's frequency (ticks per second)

        // Load image
        struct ImageRGB *image = load_imageRGB(inputFilename);
        if (image == NULL) {
                return 1;
        }

        

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /////////////////////////     START OF CODE TO BENCHMARK     ///////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////


        // QueryPerformanceCounter(&start); // Start timing


        image = apply_filter_gaussian_blur(&image, BLUR_INTENSITY_HIGH);
        if (image == NULL) {
                fprintf(stderr, "\nGaussian blur failed.\n");
                return 0;
        }


        // QueryPerformanceCounter(&end); // End timing
        // elapsedTime += (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart; // Calculate elapsed time in seconds

        
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /////////////////////////     END OF CODE TO BENCHMARK     /////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////



        // // Save image (png for testing)
        // int saveImage = save_imageRGB(image, outputFilename, FILE_TYPE_PNG);
        // if (saveImage == 0) {
        //         free_imageRGB(image);
        //         return 1;
        // }


        // Free image
        free_imageRGB(image);
        image = NULL;


        // // For benchmark simple print
        // printf("Runtime: %.10lf", elapsedTime);


        // Successful program call
        return 0;

}