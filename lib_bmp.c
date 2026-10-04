#include "lib_bmp.h"
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

uint8_t bmp_open(char* path, type_bitmap *ptr_bmp)
{
    size_t data_length, headers_size;
    int file = open(path, O_RDONLY);
    if( file < 0 )
    {
        printf("Erreur lecture fichier : \"%s\"\n", path);
        return -1;
    }

    read(file, &(ptr_bmp->file_header), sizeof(type_bmp_file_header));
    read(file, &(ptr_bmp->picture_header), sizeof(type_bmp_picture_header));

    headers_size = sizeof(type_bmp_file_header) + sizeof(type_bmp_picture_header);
    data_length = (ptr_bmp->file_header).file_size - headers_size;
    ptr_bmp->pixels = (uint8_t*) aligned_alloc(ALIGNMENT, data_length);
    read(file, ptr_bmp->pixels, data_length);
    
    close(file);
    return 0;
}

uint8_t bmp_write(char* path, type_bitmap bmp)
{
    size_t data_length, headers_size;
    int file = open(path, O_CREAT | O_WRONLY, S_IRWXU | S_IRWXG | S_IRWXO);
    if( file < 0 )
    {
        printf("Erreur écriture fichier : \"%s\"\n", path);
        return -1;
    }

    write(file, &(bmp.file_header), sizeof(type_bmp_file_header));
    write(file, &(bmp.picture_header), sizeof(type_bmp_picture_header));
    headers_size = sizeof(type_bmp_file_header) + sizeof(type_bmp_picture_header);
    data_length = bmp.file_header.file_size - headers_size;
    write(file, bmp.pixels, data_length);

    close(file);
    return 0;
}

void    bmp_free(type_bitmap bmp)
{
    free(bmp.pixels);
}