cmake_minimum_required(VERSION 3.30.1)  #Specify the minimum CMake version required
project(image_processor C)  #Define the project name and specify that using C

set(CMAKE_C_STANDARD 11)  #Set C standard

add_executable(ImageProcessor src/main.c src/image.c src/filters.c src/pool.c)  #Add executable (source files are in src/)

#target_compile_options(my_project PRIVATE -O1 -Wall -Wextra -Wpedantic) # Add after making memory optimzations and stuff
#cmake .. -G "MinGW Makefiles"

target_include_directories(ImageProcessor PRIVATE include)  # Include the header files from /include directory