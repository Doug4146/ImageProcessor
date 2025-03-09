# ImageProcessor

This is a simple image processing tool written in C which supports image processing filters including greyscale conversion,
box blurring, gaussian blurring, embossing, sharpening and sobel edge detection. Utilizes stb_image.h and stb_image_write.h libraries
for image loading and saving. Currently only works with png, jpg and bmp image filetypes.

## Requirements
- **MinGW** (tested with version 14.2.0, includes GCC as the C compiler)
- **CMake** (tested with version 3.30.1)
- **mingw32-make** (for building)

## Build Instructions

1. **Fork the repository**:
   - Go to https://github.com/Doug4146/ImageProcessor
   - Click on the "Fork" button at the top-right corner to create your own copy of the repository
2. **Clone the forked repository**:
   ```bash
   git clone https://github.com/yourusername/your-forked-repo.git
   cd your-forked-repo
3. **Create a build directory**:
   ```bash
   mkdir build
   cd build
4. **Generate build files with CMake**
   ```bash
   cmake .. -G "MinGW Makefiles"
5. **Build the program**
   ```bash
   mingw32-make

## Usage
  - To apply a filter to an input image, the image must be stored within the input\ directory. The output image will be stored in the output\ directory.
  - Ensure to properly write the relative input image path and the desired output image path with a valid image filetype (png, jpg, bmp).
  - The following is the proper usage to run the program (use fewer than 5 arguments for more detailed instructions):
    ```bash
    .\ImageProcessor.exe "..\input\INPUT_IMAGE" "..\output\OUTPUT_IMAGE" "FILTER" "FILTER INTENSITY"
  - Example:
    ```bash
    .\ImageProcessor.exe "..\input\myInputImage.jpg" "..\output\myOutputImage.png" "Gaussian Blur" "Medium"
    

