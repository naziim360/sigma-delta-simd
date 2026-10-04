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

// Sets up the background model before processing any frames:
// - moy (mean) starts equal to the first frame
// - var (variance) starts at 0
// - the output mask starts all black (no motion detected yet)
void SD_Initialization(int32_t nb_pixels, uint8_t *pixels_src, uint8_t *moy, uint16_t *var, uint8_t *pixels_sd)
{
#ifndef SIMD_VERSION
    // Scalar version: one pixel at a time.
    uint32_t i;
    for (i = 0; i < nb_pixels; i++)
    {
        moy[i] = pixels_src[i];
        var[i] = 0;
        pixels_sd[i] = BLACK;
    }
#else
    // SIMD version: same thing but 16 pixels at a time using NEON registers.
    uint32_t i;
    uint8x16_t vblack = vdupq_n_u8(BLACK);
    uint16x8_t vzero = vdupq_n_u16(0);

    for (i = 0; i < nb_pixels; i += 16)
    {
        vst1q_u8(moy + i, vld1q_u8(pixels_src + i)); // copy 16 pixels into moy
        vst1q_u8(pixels_sd + i, vblack);             // 16 pixels set to black
        vst1q_u16(var + i, vzero);                   // first 8 variances to 0
        vst1q_u16(var + i + 8, vzero);                // next 8 variances to 0
    }

#endif
}

// This is the actual Sigma-Delta algorithm: for every pixel, slowly
// adjust a running "mean" towards the current pixel value, and use the
// difference (delta) compared to a running "variance" threshold to decide
// if the pixel is background (WHITE) or motion (BLACK).
void Sigma_Delta(int32_t nb_pixels, uint8_t *pixels_in, uint8_t *moy, uint16_t *var, uint8_t *pixels_out)
{
#ifndef SIMD_VERSION
    // Scalar version: straightforward, one pixel at a time.
    uint32_t i;
    uint8_t tmp_moy, tmp_pixel, delta;
    uint16_t tmp_var, delta_N;

    for (i = 0; i < nb_pixels; i++)
    {
        tmp_moy = moy[i];
        tmp_pixel = pixels_in[i];
        tmp_var = var[i];

        // Move the mean by 1 step towards the current pixel.
        if (tmp_moy < tmp_pixel)
            tmp_moy++;
        if (tmp_moy > tmp_pixel)
            tmp_moy--;

        // How far the pixel is from the updated mean.
        delta = abs(tmp_moy - tmp_pixel);
        delta_N = delta * N;

        // Move the variance/threshold by 1 step towards N * delta.
        if (delta != 0)
        {
            if (tmp_var < delta_N)
                tmp_var++;
            if (tmp_var > delta_N)
                tmp_var--;
        }

        // If the difference is smaller than the threshold, it's background.
        // Otherwise it's motion.
        if (delta < tmp_var)
            tmp_pixel = WHITE;
        else
            tmp_pixel = BLACK;

        // Save the updated mean, variance and output pixel.
        moy[i] = tmp_moy;
        pixels_out[i] = tmp_pixel;
        var[i] = tmp_var;
    }

#else
    // SIMD version: same logic as above, but done on 8 pixels at once.
    // Since we can't use "if" on a whole vector, every decision becomes
    // a comparison that produces a mask (0x01 / 0x00), and we use that
    // mask to conditionally add/subtract.
    uint32_t i;
    uint8x8_t tmp_moy, tmp_pixel, delta, mask_lt, mask_gt, mask3_8, var_sh, mask_delta_non_null, mask2_lt, mask2_gt, mask3_16, tmp_var;
    uint16x8_t delta_N, delta_16;

    uint8x8_t v_white = vdup_n_u8(WHITE), v_black = vdup_n_u8(BLACK);

    uint8x8_t vones = vdup_n_u8(1), vn = vdup_n_u8(N), vzero = vdup_n_u8(0);

    for (i = 0; i < nb_pixels; i += 8)
    {
        // Load 8 pixels of mean, current frame, and variance.
        tmp_moy = vld1_u8(moy + i);
        tmp_pixel = vld1_u8(pixels_in + i);
        tmp_var = vld1_u8((uint8_t *)var + 2 * i);

        mask_lt = vand_u8(vclt_u8(tmp_moy, tmp_pixel), vones); // mask with [0x01] where tmp_moy < tmp_pixel, [0x00] otherwise
        mask_gt = vand_u8(vcgt_u8(tmp_moy, tmp_pixel), vones); // mask with [0x01] where tmp_moy > tmp_pixel, [0x00] otherwise

        // Move the mean by 1 step, same idea as the scalar version but
        // applied to all 8 pixels at the same time.
        tmp_moy = vqadd_u8(mask_lt, tmp_moy); // tmp_moy = tmp_moy + 1 where tmp_moy < tmp_pixel, unchanged otherwise (saturating)
        tmp_moy = vqsub_u8(tmp_moy, mask_gt); // tmp_moy = tmp_moy - 1 where tmp_moy > tmp_pixel, unchanged otherwise (saturating)

        // |mean - pixel|, computed as max-min since these are unsigned bytes.
        delta = vsub_u8(vmax_u8(tmp_moy, tmp_pixel),
                        vmin_u8(tmp_moy, tmp_pixel)); // delta = max(moy, pixel) - min(moy, pixel) = |moy - pixel|

        mask_delta_non_null = vadd_u8(mask_gt, mask_lt); // true where (tmp_moy < tmp_pixel) || (tmp_moy > tmp_pixel), i.e. delta != 0

        var_sh = vshr_n_u8(tmp_var, N);

        // mask with [0x01] where ((tmp_var / N) < delta) && delta != 0, [0x00] otherwise
        mask2_lt = vand_u8(vclt_u8(var_sh, delta), mask_delta_non_null);

        // mask with [0x01] where ((tmp_var / N) > delta) && delta != 0, [0x00] otherwise
        mask2_gt = vand_u8(vcgt_u8(var_sh, delta), mask_delta_non_null);

        // Move the variance/threshold by 1 step, same idea as scalar.
        tmp_var = vqadd_u8(mask2_lt, tmp_var); // tmp_var = tmp_var + 1 where (tmp_var < N*delta && delta != 0), unchanged otherwise
        tmp_var = vqsub_u8(tmp_var, mask2_gt); // tmp_var = tmp_var - 1 where (tmp_var > N*delta && delta != 0), unchanged otherwise

        // Decide black or white per pixel without branching: build a mask,
        // then pick white or black for each lane based on that mask.
        mask3_8 = vclt_u8(delta, tmp_var);

        tmp_pixel = vbsl_u8(mask3_8, v_white, v_black);

        // Write the 8 results back.
        vst1_u8(moy + i, tmp_moy);
        vst1_u8(pixels_out + i, tmp_pixel);
        vst1_u8(((uint8_t *)var + 2 * i), tmp_var);
    }

#endif
}

// Draws a white border around the mask so that Erosion
// and Dilatation never have to read outside the image.
void Morpho_Initialization(uint32_t Width, uint32_t Height, uint8_t *pixels)
{
#ifndef SIMD_VERSION
    uint32_t i;

    // Top and bottom rows.
    for (i = 0; i < Width; i++)
        pixels[i] = pixels[i + (Height - 1) * Width] = WHITE;
    // Left and right columns.
    for (i = 1; i < (Height - 1); i++)
        pixels[i * Width] = pixels[(i + 1) * Width - 1] = WHITE;
#else

    uint32_t i;
    uint8x8_t v_white = vdup_n_u8(WHITE);

    // Top and bottom rows, 8 pixels at a time.
    for (i = 0; i < Width; i += 8){
        vst1_u8(pixels + i, v_white);
        vst1_u8(pixels + (i + (Height - 1) * Width), v_white);
    }

    // Left and right columns are just single pixels per row, not worth SIMD.
    for (i = 1; i < (Height - 1); i++)
        pixels[i * Width] = pixels[(i + 1) * Width - 1] = WHITE;

#endif
}

// 3x3 erosion
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
        // Process 16 columns at a time.
        for (j = 1; j < (Width - 1 - 16); j += 16)
        {
            // Load the current 16-pixel tile and the next one, for each
            // of the 3 rows (top, middle, bottom) of the neighborhood.
            top_cur = vld1q_u8(pixels_in + (i - 1) * Width + j);
            top_nxt = vld1q_u8(pixels_in + (i - 1) * Width + j + 16);
            mid_cur = vld1q_u8(pixels_in +  i      * Width + j);
            mid_nxt = vld1q_u8(pixels_in +  i      * Width + j + 16);
            low_cur = vld1q_u8(pixels_in + (i + 1) * Width + j);
            low_nxt = vld1q_u8(pixels_in + (i + 1) * Width + j + 16);

            // Get the left/center/right neighbor view of each row by
            // sliding a 16-byte window across the two loaded tiles.
            tl = top_cur;  tc = vextq_u8(top_cur, top_nxt, 1);  tr = vextq_u8(top_cur, top_nxt, 2);
            ml = mid_cur;  mc = vextq_u8(mid_cur, mid_nxt, 1);  mr = vextq_u8(mid_cur, mid_nxt, 2);
            ll = low_cur;  lc = vextq_u8(low_cur, low_nxt, 1);  lr = vextq_u8(low_cur, low_nxt, 2);

            // AND all 9 neighbors together: result is black (0x00) only
            // if every single one of them was black.
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

// 3x3 dilation
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
        // Same tiling/neighbor trick as Erosion above.
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

            // OR all 9 neighbors together: result is black
            //  as soon as a single one of them was black.
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

    // Open image 0, just to read its header info (width, height, etc.)
    sprintf(path_src, "%s%s%d.bmp", PATH_SRC, IMAGE_NAME, 0);
    bmp_open(path_src, &bmp_src);

    // Prepare the output bitmap (same headers as the source, fresh pixel buffer).
    headers_size = sizeof(bmp_src.file_header) + sizeof(bmp_src.picture_header);
    nb_pixels = bmp_src.picture_header.image_size;
    offset_pixels = bmp_src.file_header.offset - headers_size;
    memcpy(&(bmp_dst.file_header), &(bmp_src.file_header), headers_size);
    bmp_dst.pixels = (uint8_t *)aligned_alloc(ALIGNMENT, bmp_src.file_header.file_size - headers_size);
    memcpy(bmp_dst.pixels, bmp_src.pixels, offset_pixels); // copy the color palette as-is

    // Allocate the running mean (Moy) and variance (Var) buffers.
    Moy = (uint8_t *)aligned_alloc(ALIGNMENT, nb_pixels);
    Var = (uint16_t *)aligned_alloc(ALIGNMENT, nb_pixels << 1);

    // Seed the background model with the very first frame.
    SD_Initialization(nb_pixels, bmp_src.pixels + offset_pixels, Moy, Var, bmp_dst.pixels + offset_pixels);
    sprintf(path_dst, "%s%s%d.bmp", PATH_SD, IMAGE_SD, 0);
    bmp_write(path_dst, bmp_dst);

    // Stage 1: Sigma-Delta on every frame, timed
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

    // Stage 2: Erosion on every Sigma-Delta output, timed
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

    // Stage 3: Dilatation on every Erosion output, timed
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