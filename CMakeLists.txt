cmake_minimum_required(VERSION 3.30.1)  #Specify the minimum CMake version required
project(image_processor C)  #Define the project name and specify that using C

#Set C standard
set(CMAKE_C_STANDARD 11)

#Add executable (source files are in src/)
add_executable(ImageProcessor src/main.c src/image.c src/filters.c src/pool.c)

# Include the header files from /include directory
target_include_directories(ImageProcessor PRIVATE include) 

# enable openMP and AVX2 compiler intrinsics including "multiply-fuse add"
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -mavx2 -mfma")

#target_compile_options(ImageProcessor PRIVATE -O3 -Wall -Wextra -Wpedantic) # Add after completing own optimizations