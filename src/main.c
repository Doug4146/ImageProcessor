#include <stdio.h>
#include <windows.h> // For performance benchmarking
#include <math.h>
#include <string.h>
#include "image.h"
#include "filters.h"
#include "convolution.h"



void print_correct_program_usage();

int validate_path_arguments(const char *inputPath,  const char *outputPath);

enum TypeFilter determine_filter(const char *filterName);

enum GeneralFilterIntensity determine_filter_intensity(const char *intensityName);



int main(int argc, char *argv[]) {


        // Performance benchmarking
	LARGE_INTEGER frequency, start, end;
	double elapsedTime = 0.0f;
	QueryPerformanceFrequency(&frequency); // Get the high-resolution counter's frequency (ticks per second)


        // Check for invalid number of command line arguments
        if (argc != 5) {
                print_correct_program_usage();
                return 1;
        }

        // Command line argument strings
        const char *inputImagePath = argv[1];
        const char *outputImagePath = argv[2];
        const char *filterName = argv[3];
        const char *filterIntensityName = argv[4];

        // Validate input and output file paths from command-line arguments
        int validatePaths = validate_path_arguments(inputImagePath, outputImagePath);
        if (validatePaths == 0) return 1;

        // Determine the type of the requested filter from command-line argument
        enum TypeFilter filter = determine_filter(filterName);
        if (filter == FILTER_INVALID) return 1;

        // Determine the desired filter intensity from command-line argument
        enum GeneralFilterIntensity intensity = determine_filter_intensity(filterIntensityName);
        if (intensity == FILTER_INTENSITY_INVALID) return 1;
        
        // Load the input image
        struct ImageRGB *inputImage = load_imageRGB(inputImagePath);
        if (inputImage == NULL) return 1;

        // Start timing
        QueryPerformanceCounter(&start);

        // Apply the desired filter
        struct ImageRGB *outputImageRGB = NULL;
        struct ImageOneChannel *outputImageOneChannel = NULL;
        enum ImageType outputImageType;
        switch (filter) {
                case FILTER_GAUSSIAN_BLUR:
                        outputImageRGB = apply_filter_generic_convolution(&inputImage, filter, intensity); 
                        outputImageType = IMAGE_TYPE_THREE_CHANNEL; break;
                case FILTER_BOX_BLUR:
                        outputImageRGB = apply_filter_generic_convolution(&inputImage, filter, intensity); 
                        outputImageType = IMAGE_TYPE_THREE_CHANNEL; break;
                case FILTER_EMBOSS:
                        outputImageRGB = apply_filter_generic_convolution(&inputImage, filter, intensity); 
                        outputImageType = IMAGE_TYPE_THREE_CHANNEL; break;
                case FILTER_SHARPEN:
                        outputImageRGB = apply_filter_generic_convolution(&inputImage, filter, intensity); 
                        outputImageType = IMAGE_TYPE_THREE_CHANNEL; break;
                case FILTER_GREYSCALE:
                        outputImageOneChannel = apply_filter_greyscale(&inputImage);
                        outputImageType = IMAGE_TYPE_ONE_CHANNEL; break;
                case FILTER_SOBEL_EDGE_DETECTION:
                        outputImageOneChannel = apply_filter_sobel_edge_detection(&inputImage, intensity);
                        outputImageType = IMAGE_TYPE_ONE_CHANNEL; break;
        }

        QueryPerformanceCounter(&end); // End timing
        elapsedTime += (double) (end.QuadPart - start.QuadPart) / frequency.QuadPart;
        printf("Runtime: %.5lf milliseconds.\n", 1000 * elapsedTime);


        // // Save the output image to the output path
        // if (outputImageType == IMAGE_TYPE_ONE_CHANNEL) {

        //         if (outputImageOneChannel == NULL) return 1;
        //         int saveImage = save_imageOneChannel(outputImageOneChannel, outputImagePath, FILE_TYPE_PNG);
        //         if (saveImage == 0) {
        //                 free_imageOneChannel(outputImageOneChannel);
        //                 return 1;
        //         }

        //         free_imageOneChannel(outputImageOneChannel);

        // } else if (outputImageType == IMAGE_TYPE_THREE_CHANNEL) {

        //         if (outputImageRGB == NULL) return 1;
        //         int saveImage = save_imageRGB(outputImageRGB, outputImagePath, FILE_TYPE_PNG);
        //         if (saveImage == 0) {
        //                 free_imageRGB(outputImageRGB);
        //                 return 1;
        //         }

        //         free_imageRGB(outputImageRGB);
                
        // }

        // Program executed successfully
        return 0;

}



void print_correct_program_usage() {
        printf("\nFatal error: invalid program arguments.\n");
        printf("Correct usage:  \"..\\ImageProcessor.exe\"  \"..\\input\\INPUT_FILENAME\"  \"..\\output\\OUTPUT_FILENAME\"  \"FILTER\" \"FILTER_INTENSITY\"\n");
        printf("Accepted image filetypes: \"png\", \"jpg\", \"bmp\".\n");
        printf("Accepted filters: \"Greyscale\", \"Gaussian Blur\", \"Box Blur\", \"Emboss\", \"Sharpen\", \"Sobel Edge Detection\".\n");
        printf("Accepted filter intensities: \"Light\", \"Medium\", \"High\".\n\n");
}

int validate_path_arguments(const char *inputPath,  const char *outputPath) {

        // Check for invalid length input image path
        int minumumInputPathLength = 14;   // e.g. ..\input\f.png
        int inputPathLength = strlen(inputPath);
        if (inputPathLength < minumumInputPathLength) {
                printf("\nFatal error: incorrect input path name.\n");
                printf("Correct input path name: \"..\\input\\INPUT_FILENAME\".\n\n");
                return 0;
        }

        // Check for incorrect input directory name
        if (strncmp(inputPath, "..\\input\\", 9) != 0) {
                printf("\nFatal error: incorrect input path name.\n");
                printf("Correct input path name: \"..\\input\\INPUT_FILENAME\".\n\n");
                return 0;
        }

        // Check for incorrect input image filetype
        if (strncmp(inputPath + (inputPathLength - 4), ".png", 4) != 0 &&
            strncmp(inputPath + (inputPathLength - 4), ".jpg", 4) != 0 &&
            strncmp(inputPath + (inputPathLength - 4), ".bmp", 4) != 0) {
                printf("\nFatal error: incorrect input image filetype.\n");
                printf("Accepted image filetypes: \"png\", \"jpg\", \"bmp\".\n\n");
                return 0;
        }

        // Check for invalid output image path length
        int minumumOutputPathLength = 15;  // e.g. ..\output\f.png
        int outputPathLength = strlen(outputPath);
        if (outputPathLength < minumumOutputPathLength) {
                printf("\nFatal error: incorrect output path name.\n");
                printf("Correct output path name: \"..\\output\\OUTPUT_FILENAME\".\n\n");
                return 0;
        }

        // Check for incorrect output directory name
        if (strncmp(outputPath, "..\\output\\", 10) != 0) {
                printf("\nFatal error: incorrect output path name.\n");
                printf("Correct output path name: \"..\\output\\INPUT_FILENAME\".\n\n");
                return 0;
        }

        // Check for incorrect output image filetype
        if (strncmp(outputPath + (outputPathLength - 4), ".png", 4) != 0 &&
            strncmp(outputPath + (outputPathLength - 4), ".jpg", 4) != 0 &&
            strncmp(outputPath + (outputPathLength - 4), ".bmp", 4) != 0) {
                printf("\nFatal error: incorrect output image filetype.\n");
                printf("Accepted image filetypes: \"png\", \"jpg\", \"bmp\".\n\n");
                return 0;
        }

        return 1;

}

enum TypeFilter determine_filter(const char *filterName) {

        // Determine the length of the filter name string
        int filterNameLength = strlen(filterName);

        // Determine the filter type

        if (filterNameLength == 6 && (strncmp(filterName, "Emboss", 6) == 0)) {
                return FILTER_EMBOSS;
        } else if (filterNameLength == 7 && (strncmp(filterName, "Sharpen", 7) == 0)) {
                return FILTER_SHARPEN;
        } else if (filterNameLength == 8 && (strncmp(filterName, "Box Blur", 8) == 0)) {
                return FILTER_BOX_BLUR;
        } else if (filterNameLength == 9 && (strncmp(filterName, "Greyscale", 9) == 0)) {
                return FILTER_GREYSCALE;
        } else if (filterNameLength == 13 && (strncmp(filterName, "Gaussian Blur", 13) == 0)) {
                return FILTER_GAUSSIAN_BLUR;
        } else if (filterNameLength == 20 && (strncmp(filterName, "Sobel Edge Detection", 20) == 0)) {
                return FILTER_SOBEL_EDGE_DETECTION;
        } else {
                printf("\nFatal error: invalid filter.\n");
                printf("Accepted filters: \"Greyscale\", \"Gaussian Blur\", \"Box Blur\", \"Emboss\", \"Sharpen\", \"Sobel Edge Detection\".\n\n");
                return FILTER_INVALID;
        }

}

enum GeneralFilterIntensity determine_filter_intensity(const char *intensityName) {

        // Determine the length of the filter intensity name string
        int filterIntensityNameLength = strlen(intensityName);

        // Determine the filter intensity type

        if (filterIntensityNameLength == 4 && (strncmp(intensityName, "High", 4) == 0)) {
                return FILTER_INTENSITY_HIGH;
        } else if (filterIntensityNameLength == 5 && (strncmp(intensityName, "Light", 5) == 0)) {
                return FILTER_INTENSITY_LIGHT;
        } else if (filterIntensityNameLength == 6 && (strncmp(intensityName, "Medium", 6) == 0)) {
                return FILTER_INTENSITY_MEDIUM;
        } else {
                printf("\nFatal error: invalid filter intensity.\n");
                printf("Accepted filter intensities: \"Light\", \"Medium\", \"High\".\n\n");
                return FILTER_INTENSITY_INVALID;
        }

}


