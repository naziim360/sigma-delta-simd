#ifndef _LIB_BMP_H_
#define _LIB_BMP_H_

#include <stdint.h>

#define ALIGNMENT       16


typedef struct __attribute__ ((__packed__)) 
{
    uint16_t signature;
    uint32_t file_size;
    uint32_t reserved;
    uint32_t offset;
} type_bmp_file_header;

typedef struct __attribute__ ((__packed__)) 
{
    uint32_t header_size;
    uint32_t width;
    uint32_t height;
    uint16_t nb_plans;
    uint16_t depth;
    uint32_t compression;
    uint32_t image_size;
    uint32_t h_resolution;
    uint32_t v_resolution;
    uint32_t nb_cols_palette;
    uint32_t nb_imp_cols_palette;
} type_bmp_picture_header;

typedef struct __attribute__ ((__packed__)) 
{
    type_bmp_file_header    file_header;
    type_bmp_picture_header picture_header;
    uint8_t*                pixels;
} type_bitmap;

uint8_t bmp_open(char* path, type_bitmap *ptr_bmp);
uint8_t bmp_write(char* path, type_bitmap bmp);
void    bmp_free(type_bitmap bmp);

#endif