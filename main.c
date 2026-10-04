#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>
#include "lib_bmp.h"


#define N 4
#define BLACK 0x00
#define WHITE 0xFF
#define NB_IMAGES 200
#define IMAGE_NAME "Image"
#define PATH_SRC "./images_src/"
#ifdef SIMD_VERSION
#define IMAGE_SD "Image_simd_"
#define IMAGE_SD_E "Image_simd_e_"
#define IMAGE_SD_E_D "Image_simd_e_d_"
#define PATH_SD "./images_simd/"
#else
#define IMAGE_SD "Image_sca_"
#define IMAGE_SD_E "Image_sca_e_"
#define IMAGE_SD_E_D "Image_sca_e_d_"
#define PATH_SD "./images_scalaire/"
#endif

uint64_t system_nanoTime()
{
    struct timespec now;
    clock_gettime(0, &now);
    return now.tv_sec * 1000000000LL + now.tv_nsec;
}

void SD_Initialization(int32_t nb_pixels, uint8_t *pixels_src, uint8_t *moy, uint16_t *var, uint8_t *pixels_sd)
{
#ifndef SIMD_VERSION
    uint32_t i;
    for (i = 0; i < nb_pixels; i++)
    {
        moy[i] = pixels_src[i];
        var[i] = 0;
        pixels_sd[i] = BLACK;
    }
#else

    uint32_t i;
    uint8x16_t vblack = vdupq_n_u8(BLACK);
    uint16x8_t vzero = vdupq_n_u16(0);

    for (i = 0; i < nb_pixels; i += 16)
    {
        vst1q_u8(moy + i, vld1q_u8(pixels_src + i));
        vst1q_u8(pixels_sd + i, vblack);
        vst1q_u16(var + i, vzero);
        vst1q_u16(var + i + 8, vzero);
    }

#endif
}

void Sigma_Delta(int32_t nb_pixels, uint8_t *pixels_in, uint8_t *moy, uint16_t *var, uint8_t *pixels_out)
{
#ifndef SIMD_VERSION
    uint32_t i;
    uint8_t tmp_moy, tmp_pixel, delta;
    uint16_t tmp_var, delta_N;

    for (i = 0; i < nb_pixels; i++)
    {
        tmp_moy = moy[i];
        tmp_pixel = pixels_in[i];
        tmp_var = var[i];

        if (tmp_moy < tmp_pixel)
            tmp_moy++;
        if (tmp_moy > tmp_pixel)
            tmp_moy--;

        delta = abs(tmp_moy - tmp_pixel);
        delta_N = delta * N;

        if (delta != 0)
        {
            if (tmp_var < delta_N)
                tmp_var++;
            if (tmp_var > delta_N)
                tmp_var--;
        }

        if (delta < tmp_var)
            tmp_pixel = WHITE;
        else
            tmp_pixel = BLACK;

        moy[i] = tmp_moy;
        pixels_out[i] = tmp_pixel;
        var[i] = tmp_var;
    }

#else

    uint32_t i;
    uint8x8_t tmp_moy, tmp_pixel, delta, mask_lt, mask_gt, mask3_8, var_sh, mask_delta_non_null, mask2_lt, mask2_gt, mask3_16, tmp_var;
    uint16x8_t delta_N, delta_16;

    uint8x8_t v_white = vdup_n_u8(WHITE), v_black = vdup_n_u8(BLACK);

    uint8x8_t vones = vdup_n_u8(1), vn = vdup_n_u8(N), vzero = vdup_n_u8(0);

    for (i = 0; i < nb_pixels; i += 8)
    {

        tmp_moy = vld1_u8(moy + i);
        tmp_pixel = vld1_u8(pixels_in + i);
        tmp_var = vld1_u8((uint8_t *)var + 2 * i);

        mask_lt = vand_u8(vclt_u8(tmp_moy, tmp_pixel), vones); // un masque avec [0x01] si tmp_moy < tmp_pixel, [0x00] sinon
        mask_gt = vand_u8(vcgt_u8(tmp_moy, tmp_pixel), vones); // un masque avec [0x01] si tmp_moy > tmp_pixel, [0x00] sinon

        tmp_moy = vqadd_u8(mask_lt, tmp_moy); // tmp_moy = [tmp_moy + 1] si tmp_moy < tmp_pixel,  [tmp_moy] sinon (avec saturation)
        tmp_moy = vqsub_u8(tmp_moy, mask_gt); // tmp_moy = [tmp_moy - 1] si tmp_moy > tmp_pixel,  [tmp_moy] sinon (avec saturation)

        delta = vsub_u8(vmax_u8(tmp_moy, tmp_pixel),
                        vmin_u8(tmp_moy, tmp_pixel)); // delta = max(moy, pixel) - min(moy,pixel) = |moy-pixel|

        mask_delta_non_null = vadd_u8(mask_gt, mask_lt); // (tmp_moy < tmp_pixel) || (tmp_moy >tmp_pixel) sur 16 bits

        var_sh = vshr_n_u8(tmp_var, N);

        // un masque avec [0x01] si ((tmp_var / N) < delta) && delta != 0), [0x0000] sinon
        mask2_lt = vand_u8(vclt_u8(var_sh, delta), mask_delta_non_null);

        // un masque avec [0x01] si ((tmp_var / N) > delta) && delta != 0), [0x0000] sinon
        mask2_gt = vand_u8(vcgt_u8(var_sh, delta), mask_delta_non_null);

        tmp_var = vqadd_u8(mask2_lt, tmp_var); // tmp_var = [tmp_var + 1] si (tmp_var < (N* delta) && delta != 0),  [tmp_var] sinon
        tmp_var = vqsub_u8(tmp_var, mask2_gt); // tmp_var = [tmp_var - 1] si (tmp_var > (N* delta) && delta != 0),  [tmp_var] sinon

        mask3_8 = vclt_u8(delta, tmp_var);

        tmp_pixel = vbsl_u8(mask3_8, v_white, v_black);

        vst1_u8(moy + i, tmp_moy);
        vst1_u8(pixels_out + i, tmp_pixel);
        vst1_u8(((uint8_t *)var + 2 * i), tmp_var);
    }

#endif
}

void Morpho_Initialization(uint32_t Width, uint32_t Height, uint8_t *pixels)
{
#ifndef SIMD_VERSION
    uint32_t i;

    for (i = 0; i < Width; i++)
        pixels[i] = pixels[i + (Height - 1) * Width] = WHITE;
    for (i = 1; i < (Height - 1); i++)
        pixels[i * Width] = pixels[(i + 1) * Width - 1] = WHITE;
#else

    uint32_t i;
    uint8x8_t v_white = vdup_n_u8(WHITE);

    for (i = 0; i < Width; i += 8){
        vst1_u8(pixels + i, v_white);
        vst1_u8(pixels + (i + (Height - 1) * Width), v_white);
    }

    for (i = 1; i < (Height - 1); i++)
        pixels[i * Width] = pixels[(i + 1) * Width - 1] = WHITE;

#endif
}

void Erosion(uint32_t Width, uint32_t Height, uint8_t *pixels_in, uint8_t *pixels_out)
{
#ifndef SIMD_VERSION
    uint32_t i, j;

    for (i = 1; i < (Height - 1); i++)
        for (j = 1; j < (Width - 1); j++)
            if ((pixels_in[(i - 1) * Width + j - 1] == BLACK) &&
                (pixels_in[(i - 1) * Width + j] == BLACK) &&
                (pixels_in[(i - 1) * Width + j + 1] == BLACK) &&
                (pixels_in[i * Width + j - 1] == BLACK) &&
                (pixels_in[i * Width + j] == BLACK) &&
                (pixels_in[i * Width + j + 1] == BLACK) &&
                (pixels_in[(i + 1) * Width + j - 1] == BLACK) &&
                (pixels_in[(i + 1) * Width + j] == BLACK) &&
                (pixels_in[(i + 1) * Width + j + 1] == BLACK))
                pixels_out[i * Width + j] = BLACK;
            else
                pixels_out[i * Width + j] = WHITE;
#else

    uint32_t i, j;
    uint8x16_t top_cur, top_nxt, mid_cur, mid_nxt, low_cur, low_nxt;
    uint8x16_t tl, tc, tr, ml, mc, mr, ll, lc, lr, mask;

    for (i = 1; i < (Height - 1); i++)
    {
        for (j = 1; j < (Width - 1 - 16); j += 16)
        {
            top_cur = vld1q_u8(pixels_in + (i - 1) * Width + j);
            top_nxt = vld1q_u8(pixels_in + (i - 1) * Width + j + 16);
            mid_cur = vld1q_u8(pixels_in +  i      * Width + j);
            mid_nxt = vld1q_u8(pixels_in +  i      * Width + j + 16);
            low_cur = vld1q_u8(pixels_in + (i + 1) * Width + j);
            low_nxt = vld1q_u8(pixels_in + (i + 1) * Width + j + 16);

            tl = top_cur;  tc = vextq_u8(top_cur, top_nxt, 1);  tr = vextq_u8(top_cur, top_nxt, 2);
            ml = mid_cur;  mc = vextq_u8(mid_cur, mid_nxt, 1);  mr = vextq_u8(mid_cur, mid_nxt, 2);
            ll = low_cur;  lc = vextq_u8(low_cur, low_nxt, 1);  lr = vextq_u8(low_cur, low_nxt, 2);

            uint8x16_t m1 = vandq_u8(tl, tc);
            uint8x16_t m2 = vandq_u8(tr, ml);
            uint8x16_t m3 = vandq_u8(mc, mr);
            uint8x16_t m4 = vandq_u8(ll, lc);

            uint8x16_t m5 = vandq_u8(m1, m2);
            uint8x16_t m6 = vandq_u8(m3, m4);

            mask = vandq_u8(vandq_u8(m5, m6), lr);

            vst1q_u8(pixels_out + (i * Width + j), mask);
        }
    }

#endif
}

void Dilatation(uint32_t Width, uint32_t Height, uint8_t *pixels_in, uint8_t *pixels_out)
{
#ifndef SIMD_VERSION
    uint32_t i, j;

    for (i = 1; i < (Height - 1); i++)
        for (j = 1; j < (Width - 1); j++)
            if ((pixels_in[(i - 1) * Width + j - 1] == BLACK) ||
                (pixels_in[(i - 1) * Width + j] == BLACK) ||
                (pixels_in[(i - 1) * Width + j + 1] == BLACK) ||
                (pixels_in[i * Width + j - 1] == BLACK) ||
                (pixels_in[i * Width + j] == BLACK) ||
                (pixels_in[i * Width + j + 1] == BLACK) ||
                (pixels_in[(i + 1) * Width + j - 1] == BLACK) ||
                (pixels_in[(i + 1) * Width + j] == BLACK) ||
                (pixels_in[(i + 1) * Width + j + 1] == BLACK))
                pixels_out[i * Width + j] = BLACK;
            else
                pixels_out[i * Width + j] = WHITE;
#else

    uint32_t i, j;
    uint8x16_t top_cur, top_nxt, mid_cur, mid_nxt, low_cur, low_nxt;
    uint8x16_t tl, tc, tr, ml, mc, mr, ll, lc, lr, mask;

    for (i = 1; i < (Height - 1); i++)
    {
        for (j = 1; j < (Width - 1 - 16); j += 16)
        {
            top_cur = vld1q_u8(pixels_in + (i - 1) * Width + j);
            top_nxt = vld1q_u8(pixels_in + (i - 1) * Width + j + 16);
            mid_cur = vld1q_u8(pixels_in +  i      * Width + j);
            mid_nxt = vld1q_u8(pixels_in +  i      * Width + j + 16);
            low_cur = vld1q_u8(pixels_in + (i + 1) * Width + j);
            low_nxt = vld1q_u8(pixels_in + (i + 1) * Width + j + 16);

            tl = top_cur;  tc = vextq_u8(top_cur, top_nxt, 1);  tr = vextq_u8(top_cur, top_nxt, 2);
            ml = mid_cur;  mc = vextq_u8(mid_cur, mid_nxt, 1);  mr = vextq_u8(mid_cur, mid_nxt, 2);
            ll = low_cur;  lc = vextq_u8(low_cur, low_nxt, 1);  lr = vextq_u8(low_cur, low_nxt, 2);

            uint8x16_t m1 = vorrq_u8(tl, tc);
            uint8x16_t m2 = vorrq_u8(tr, ml);
            uint8x16_t m3 = vorrq_u8(mc, mr);
            uint8x16_t m4 = vorrq_u8(ll, lc);

            uint8x16_t m5 = vorrq_u8(m1, m2);
            uint8x16_t m6 = vorrq_u8(m3, m4);

            mask = vorrq_u8(vorrq_u8(m5, m6), lr);

            vst1q_u8(pixels_out + (i * Width + j), mask);
        }
    }

#endif
}

int main()
{
    long duree_totale, duree;
    uint32_t i, nb_pixels, offset_pixels, headers_size;
    type_bitmap bmp_src, bmp_dst;
    uint8_t *Moy;
    uint16_t *Var;
    char path_src[100], path_dst[100];

    // Création du chemin pour l'image 0 et lecture de celle-ci
    sprintf(path_src, "%s%s%d.bmp", PATH_SRC, IMAGE_NAME, 0);
    bmp_open(path_src, &bmp_src);
    // Préparation du bitmap pour stocker le résultat
    headers_size = sizeof(bmp_src.file_header) + sizeof(bmp_src.picture_header);
    nb_pixels = bmp_src.picture_header.image_size;
    offset_pixels = bmp_src.file_header.offset - headers_size;
    memcpy(&(bmp_dst.file_header), &(bmp_src.file_header), headers_size);
    bmp_dst.pixels = (uint8_t *)aligned_alloc(ALIGNMENT, bmp_src.file_header.file_size - headers_size);
    memcpy(bmp_dst.pixels, bmp_src.pixels, offset_pixels);
    // Allocation des matrices Moy et Var stockant respectivement les moyennes et les variances
    Moy = (uint8_t *)aligned_alloc(ALIGNMENT, nb_pixels);
    Var = (uint16_t *)aligned_alloc(ALIGNMENT, nb_pixels << 1);

    // Initialisation de l'algorithme avec l'image 0
    SD_Initialization(nb_pixels, bmp_src.pixels + offset_pixels, Moy, Var, bmp_dst.pixels + offset_pixels);
    // Création du chemin pour l'image 0 et écriture de celle-ci
    sprintf(path_dst, "%s%s%d.bmp", PATH_SD, IMAGE_SD, 0);
    bmp_write(path_dst, bmp_dst);

    duree_totale = 0;
    for (i = 1; i < NB_IMAGES; i++)
    {
        sprintf(path_src, "%s%s%d.bmp", PATH_SRC, IMAGE_NAME, i);
        bmp_open(path_src, &bmp_src);
        duree = system_nanoTime();
        Sigma_Delta(nb_pixels, bmp_src.pixels + offset_pixels, Moy, Var, bmp_dst.pixels + offset_pixels);
        duree_totale += system_nanoTime() - duree;
        sprintf(path_dst, "%s%s%d.bmp", PATH_SD, IMAGE_SD, i);
        bmp_write(path_dst, bmp_dst);
    }
    printf("Sigma/Delta - Temps de traitement moyen par image : %.2f ns\n", ((double)duree_totale) / (NB_IMAGES - 1));

    Morpho_Initialization(bmp_dst.picture_header.width, bmp_dst.picture_header.height, bmp_dst.pixels + offset_pixels);
    duree_totale = 0;
    for (i = 0; i < NB_IMAGES; i++)
    {
        sprintf(path_src, "%s%s%d.bmp", PATH_SD, IMAGE_SD, i);
        bmp_open(path_src, &bmp_src);
        duree = system_nanoTime();
        Erosion(bmp_src.picture_header.width, bmp_src.picture_header.height, bmp_src.pixels + offset_pixels, bmp_dst.pixels + offset_pixels);
        duree_totale += system_nanoTime() - duree;
        sprintf(path_dst, "%s%s%d.bmp", PATH_SD, IMAGE_SD_E, i);
        bmp_write(path_dst, bmp_dst);
    }
    printf("Erosion     - Temps de traitement moyen par image : %.2f ns\n", ((double)duree_totale) / NB_IMAGES);

    duree_totale = 0;
    for (i = 0; i < NB_IMAGES; i++)
    {
        sprintf(path_src, "%s%s%d.bmp", PATH_SD, IMAGE_SD_E, i);
        bmp_open(path_src, &bmp_src);
        duree = system_nanoTime();
        Dilatation(bmp_src.picture_header.width, bmp_src.picture_header.height, bmp_src.pixels + offset_pixels, bmp_dst.pixels + offset_pixels);
        duree_totale += system_nanoTime() - duree;
        sprintf(path_dst, "%s%s%d.bmp", PATH_SD, IMAGE_SD_E_D, i);
        bmp_write(path_dst, bmp_dst);
    }
    printf("Dilatation  - Temps de traitement moyen par image : %.2f ns\n", ((double)duree_totale) / NB_IMAGES);
    return 0;
}
