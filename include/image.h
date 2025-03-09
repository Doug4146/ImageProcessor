#ifndef IMAGE_H
#define IMAGE_H

#include <stdint.h>


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


// Structure for a 1-channeled Image (black and white) in structure of arrays form.
typedef struct ImageOneChannel {
        int width, height, numChannels;
        uint8_t *pixels;
} Image;


// Enumeration for the type of image - three channeled (RGB) or one channaled (black/white)
typedef enum ImageType {
        IMAGE_TYPE_ONE_CHANNEL,
        IMAGE_TYPE_THREE_CHANNEL
} ImageType;





// Loads image from disk using stb_image.h. Transforms the loaded image into a struct ImageRGB
// and returns pointer to the struct
struct ImageRGB *load_imageRGB(const char *filename);
struct ImageOneChannel *load_imageOneChannel(const char *filename);


struct ImageRGB *load_empty_imageRGB(int width, int height);
struct ImageOneChannel *load_empty_imageOneChannel(int width, int height);


int save_imageRGB(struct ImageRGB *image, const char *filename, ImageFileType fileType);
int save_imageOneChannel(struct ImageOneChannel *image, const char *filename, ImageFileType fileType);


void free_imageRGB(struct ImageRGB *image);
void free_imageOneChannel(struct ImageOneChannel *image);






#endif //IMAGE_H