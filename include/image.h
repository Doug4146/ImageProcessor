#ifndef IMAGE_H
#define IMAGE_H

#include <stdint.h>


// Constant global integer for number of channels per pixel in RGB Image format
extern const int RGB_NUM_CHANNELS;


// Enumeration for valid image file types including png, jpg, and bmp.
typedef enum FileType {
        FILE_TYPE_PNG,
        FILE_TYPE_JPG,
        FILE_TYPE_BMP
} ImageFileType;


// Enumeration for the channel types of a three-channeled image (R, G, B).
typedef enum ChannelTypeRGB {
        CHANNEL_TYPE_RED,
        CHANNEL_TYPE_GREEN,
        CHANNEL_TYPE_BLUE
} ChannelTypeRGB;


// Structure for a 3-channeled Image in structure of arrays form. 
// The different channel arrays are stored in one contiguous memory region
typedef struct ImageRGB {
        int width, height, numChannels;
        uint8_t *redChannels;
        uint8_t *greenChannels;
        uint8_t *blueChannels;
} ImageRGB;




// Loads image from disk using stb_image.h. Transforms the loaded image into a struct ImageRGB
// and returns pointer to the struct
struct ImageRGB *load_imageRGB(const char *filename);


struct ImageRGB *load_empty_imageRGB(int width, int height);


int save_imageRGB(struct ImageRGB *image, const char *filename, ImageFileType fileType);


void free_imageRGB(struct ImageRGB *image);






#endif //IMAGE_H