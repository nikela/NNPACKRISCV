/**** if deadling with EPI-0.7 intrinsics then use this file instead of convolutional-inference.c***//


#include <stdio.h>
#include <pthread.h>
#include "init.h"

typedef void (*pthreadpool_task_2d_tile_2d_t)(void *, size_t, size_t, size_t, size_t);
typedef void (*pthreadpool_task_2d_tile_2d_t1)(void *, size_t, size_t, size_t, size_t);

struct NNP_CACHE_ALIGN kernel_transform_context
{
        nnp_transform_2d_with_offset transform_function;
        const float *kernel;
        void *kernel_transform;

        size_t tuple_size;
        size_t input_channels;
        size_t input_channels_block_size;
        size_t output_channels;
        struct nnp_size kernel_size;
};

struct NNP_CACHE_ALIGN output_transform_context
{
        nnp_transform_2d_with_bias transform_function;
        float *output;
        const void *output_transform;
        const float *bias;

        size_t tuple_size;
        size_t tiles_count;
        struct fxdiv_divisor_size_t tiles_x_count;
        struct fxdiv_divisor_size_t tiles_block_max;
        size_t output_channels;
        struct nnp_size output_size;
        struct nnp_size output_tile;
};

struct NNP_CACHE_ALIGN tuple_multiplication_context
{
        size_t tuple_elements;
        size_t tuple_size;
        size_t tiles_subblock_max;
        size_t input_channels_block_size;
        size_t input_channels_block_start;
        size_t output_channels;
        size_t output_channels_subblock_max;
        size_t output_channels_block_start;

        const void *input_transform;
        const void *kernel_transform;
        void *output_transform;

        nnp_fast_tuple_gemm_function fast_gemm;
        nnp_full_tuple_gemm_function full_gemm;
};

struct NNP_CACHE_ALIGN input_transform_context
{
        const float *input;
        void *input_transform;
        nnp_transform_2d_with_offset transform_function;

        const size_t tuple_size;
        const size_t tiles_count;
        const struct fxdiv_divisor_size_t tiles_x_count;
        const size_t input_channels_block_start;
        const size_t input_channels_block_size;
        const struct nnp_size input_size;
        const size_t input_padding_left;
        const size_t input_padding_top;
        const struct nnp_size input_tile;
        const struct nnp_size input_tile_step;
};

static void compute_input_transform(
    const struct input_transform_context context[restrict static 1],
    size_t input_channels_block_offset, size_t tiles_subblock_start,
    size_t input_channels_block_range, size_t tiles_subblock_size, size_t interchannels);

static void compute_output_transform(
    const struct output_transform_context context[restrict static 1],
    size_t output_channels_subblock_start, size_t tiles_subblock_start,
    size_t output_channels_subblock_size, size_t tiles_subblock_size);

static void compute_kernel_transform(
    const struct kernel_transform_context context[restrict static 1],
    size_t output_channels_subblock_start, size_t input_channels_block_offset,
    size_t output_channels_subblock_size, size_t input_channels_block_increment, size_t interchannles);
static void compute_tuple_multiplication(
    const struct tuple_multiplication_context context[restrict static 1],
    size_t tiles_block_start, size_t output_channels_subblock_start,
    size_t tiles_block_size, size_t output_channels_subblock_size);

void nnp_s4gemm_only_3x3__neon(
    size_t k, size_t update,
    const float a[restrict static 1],
    const float b[restrict static 1],
    float c[restrict static 1],
    size_t row_stride_c);
void nnp_s4gemm_upto_3x3__neon(
    uint32_t mr, uint32_t nr,
    size_t k, size_t update,
    const float a[restrict static 1],
    const float b[restrict static 1],
    float c[restrict static 1],
    size_t row_stride_c);

struct hardware_info nnp_hwinfo = {};

int subsampling;

void nnp_s4gemm_upto_3x3__neon(
    uint32_t mr, uint32_t nr,
    size_t k, size_t update,
    const float a[restrict static 1],
    const float b[restrict static 1],
    float c[restrict static 1],
    size_t row_stride_c)
{
        int simd_width = 4;
        int vl = nnp_hwinfo.sxgemm.mr * 4;
        int simd_width1 = __builtin_epi_vsetvlmax(__epi_e32, __epi_m1);
        int rem = __builtin_epi_vsetvlmax(__epi_e32, __epi_m1) / 4; //__builtin_epi_vsetvlmax(__epi_e32, __epi_m1);
        int index1_host[simd_width1];
        int inc = 0;
        int four = 4;
        for (int i = 0; i < rem; i++)
        {
                for (int ind = 0; ind < 4; ind++)
                {
                        index1_host[inc] = ind;
                        inc++;
                }
        }
        switch (mr)
        {
        case 1:
        {
                for (int i1 = 0; i1 < vl;)
                {
                        unsigned long gvl = __builtin_epi_vsetvl(((long)vl - (long)i1), __epi_e32, __epi_m1);
                        __epi_2xf32 acc00 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc10 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc20 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc30 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc40 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc50 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc60 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc70 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc80 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        const float *b0_ptr = b + i1;
                        const float *a1_ptr = a + 0;
                        const size_t b_increment = nr * simd_width;
                        for (int j = 0; j < k; j++)
                        {

                                const __epi_2xi32 FOUR = __builtin_epi_vbroadcast_2xi32(four, gvl);
                                __epi_2xi32 index1 = __builtin_epi_vload_2xi32(&index1_host[0], gvl);
                                __epi_2xi32 index11 = __builtin_epi_vmul_2xi32(index1, FOUR, gvl);
                                const __epi_2xf32 a0 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;

                                const __epi_2xf32 b0 = __builtin_epi_vload_2xf32(b0_ptr, gvl);
                                b0_ptr += b_increment;
                                acc00 = __builtin_epi_vfmacc_2xf32(acc00, a0, b0, gvl);
                        }
                        if (update != 0)
                        {
                                __builtin_epi_vstore_2xf32(c + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(c + i1, gvl), acc00, gvl), gvl);
                        }
                        else
                        {
                                __builtin_epi_vstore_2xf32(c + i1, acc00, gvl);
                        }

                        i1 += gvl;
                }
                break;
        }
        case 2:
        {
                for (int i1 = 0; i1 < vl;)
                {
                        unsigned long gvl = __builtin_epi_vsetvl(((long)vl - (long)i1), __epi_e32, __epi_m1);
                        __epi_2xf32 acc00 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc10 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc20 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc30 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc40 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc50 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc60 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc70 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc80 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        const float *b0_ptr = b + i1;
                        const float *a1_ptr = a + 0;
                        const size_t b_increment = nr * simd_width;
                        for (int j = 0; j < k; j++)
                        {

                                const __epi_2xi32 FOUR = __builtin_epi_vbroadcast_2xi32(four, gvl);
                                __epi_2xi32 index1 = __builtin_epi_vload_2xi32(&index1_host[0], gvl);
                                __epi_2xi32 index11 = __builtin_epi_vmul_2xi32(index1, FOUR, gvl);
                                const __epi_2xf32 a0 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a1 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;

                                const __epi_2xf32 b0 = __builtin_epi_vload_2xf32(b0_ptr, gvl);
                                b0_ptr += b_increment;
                                acc00 = __builtin_epi_vfmacc_2xf32(acc00, a0, b0, gvl);
                                acc10 = __builtin_epi_vfmacc_2xf32(acc10, a1, b0, gvl);
                        }
                        float *restrict crow0 = c;
                        float *restrict crow1 = crow0 + row_stride_c;
                        if (update != 0)
                        {
                                __builtin_epi_vstore_2xf32(crow0 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow0 + i1, gvl), acc00, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow1 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow1 + i1, gvl), acc10, gvl), gvl);
                        }
                        else
                        {
                                __builtin_epi_vstore_2xf32(crow0 + i1, acc00, gvl);
                                __builtin_epi_vstore_2xf32(crow1 + i1, acc10, gvl);
                        }

                        i1 += gvl;
                }
                break;
        }
        case 4:
        {
                for (int i1 = 0; i1 < vl;)
                {
                        unsigned long gvl = __builtin_epi_vsetvl(((long)vl - (long)i1), __epi_e32, __epi_m1);
                        __epi_2xf32 acc00 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc10 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc20 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc30 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc40 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc50 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc60 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc70 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc80 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        const float *b0_ptr = b + i1;
                        const float *a1_ptr = a + 0;
                        const size_t b_increment = nr * simd_width;
                        for (int j = 0; j < k; j++)
                        {

                                const __epi_2xi32 FOUR = __builtin_epi_vbroadcast_2xi32(four, gvl);
                                __epi_2xi32 index1 = __builtin_epi_vload_2xi32(&index1_host[0], gvl);
                                __epi_2xi32 index11 = __builtin_epi_vmul_2xi32(index1, FOUR, gvl);
                                const __epi_2xf32 a0 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a1 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a2 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a3 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;

                                const __epi_2xf32 b0 = __builtin_epi_vload_2xf32(b0_ptr, gvl);
                                b0_ptr += b_increment;
                                acc00 = __builtin_epi_vfmacc_2xf32(acc00, a0, b0, gvl);
                                acc10 = __builtin_epi_vfmacc_2xf32(acc10, a1, b0, gvl);
                                acc20 = __builtin_epi_vfmacc_2xf32(acc20, a2, b0, gvl);
                                acc30 = __builtin_epi_vfmacc_2xf32(acc30, a3, b0, gvl);
                        }
                        float *restrict crow0 = c;
                        float *restrict crow1 = crow0 + row_stride_c;
                        float *restrict crow2 = crow1 + row_stride_c;
                        float *restrict crow3 = crow2 + row_stride_c;
                        if (update != 0)
                        {
                                __builtin_epi_vstore_2xf32(crow0 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow0 + i1, gvl), acc00, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow1 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow1 + i1, gvl), acc10, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow2 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow2 + i1, gvl), acc20, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow3 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow3 + i1, gvl), acc30, gvl), gvl);
                        }
                        else
                        {
                                __builtin_epi_vstore_2xf32(crow0 + i1, acc00, gvl);
                                __builtin_epi_vstore_2xf32(crow1 + i1, acc10, gvl);
                                __builtin_epi_vstore_2xf32(crow2 + i1, acc20, gvl);
                                __builtin_epi_vstore_2xf32(crow3 + i1, acc30, gvl);
                        }

                        i1 += gvl;
                }
                break;
        }
        case 9:
        {
                for (int i1 = 0; i1 < vl;)
                {
                        unsigned long gvl = __builtin_epi_vsetvl(((long)vl - (long)i1), __epi_e32, __epi_m1);
                        __epi_2xf32 acc00 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc10 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc20 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc30 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc40 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc50 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc60 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc70 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc80 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        const float *b0_ptr = b + i1;
                        const float *a1_ptr = a + 0;
                        const size_t b_increment = nr * simd_width;
                        for (int j = 0; j < k; j++)
                        {

                                const __epi_2xi32 FOUR = __builtin_epi_vbroadcast_2xi32(four, gvl);
                                __epi_2xi32 index1 = __builtin_epi_vload_2xi32(&index1_host[0], gvl);
                                __epi_2xi32 index11 = __builtin_epi_vmul_2xi32(index1, FOUR, gvl);
                                const __epi_2xf32 a0 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a1 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a2 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a3 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a4 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a5 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a6 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a7 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a8 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;

                                const __epi_2xf32 b0 = __builtin_epi_vload_2xf32(b0_ptr, gvl);
                                b0_ptr += b_increment;
                                acc00 = __builtin_epi_vfmacc_2xf32(acc00, a0, b0, gvl);
                                acc10 = __builtin_epi_vfmacc_2xf32(acc10, a1, b0, gvl);
                                acc20 = __builtin_epi_vfmacc_2xf32(acc20, a2, b0, gvl);
                                acc30 = __builtin_epi_vfmacc_2xf32(acc30, a3, b0, gvl);
                                acc40 = __builtin_epi_vfmacc_2xf32(acc40, a4, b0, gvl);
                                acc50 = __builtin_epi_vfmacc_2xf32(acc50, a5, b0, gvl);
                                acc60 = __builtin_epi_vfmacc_2xf32(acc60, a6, b0, gvl);
                                acc70 = __builtin_epi_vfmacc_2xf32(acc70, a7, b0, gvl);
                                acc80 = __builtin_epi_vfmacc_2xf32(acc80, a8, b0, gvl);
                        }
                        float *restrict crow0 = c;
                        float *restrict crow1 = crow0 + row_stride_c;
                        float *restrict crow2 = crow1 + row_stride_c;
                        float *restrict crow3 = crow2 + row_stride_c;
                        float *restrict crow4 = crow3 + row_stride_c;
                        float *restrict crow5 = crow4 + row_stride_c;
                        float *restrict crow6 = crow5 + row_stride_c;
                        float *restrict crow7 = crow6 + row_stride_c;
                        float *restrict crow8 = crow7 + row_stride_c;
                        if (update != 0)
                        {
                                __builtin_epi_vstore_2xf32(crow0 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow0 + i1, gvl), acc00, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow1 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow1 + i1, gvl), acc10, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow2 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow2 + i1, gvl), acc20, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow3 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow3 + i1, gvl), acc30, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow4 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow4 + i1, gvl), acc40, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow5 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow5 + i1, gvl), acc50, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow6 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow6 + i1, gvl), acc60, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow7 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow7 + i1, gvl), acc70, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow8 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow8 + i1, gvl), acc80, gvl), gvl);
                        }
                        else
                        {
                                __builtin_epi_vstore_2xf32(crow0 + i1, acc00, gvl);
                                __builtin_epi_vstore_2xf32(crow1 + i1, acc10, gvl);
                                __builtin_epi_vstore_2xf32(crow2 + i1, acc20, gvl);
                                __builtin_epi_vstore_2xf32(crow3 + i1, acc30, gvl);
                                __builtin_epi_vstore_2xf32(crow4 + i1, acc40, gvl);
                                __builtin_epi_vstore_2xf32(crow5 + i1, acc50, gvl);
                                __builtin_epi_vstore_2xf32(crow6 + i1, acc60, gvl);
                                __builtin_epi_vstore_2xf32(crow7 + i1, acc70, gvl);
                                __builtin_epi_vstore_2xf32(crow8 + i1, acc80, gvl);
                        }

                        i1 += gvl;
                }
                break;
        }
                case 17:
        {
                for (int i1 = 0; i1 < vl;)
                {
                        unsigned long gvl = __builtin_epi_vsetvl(((long)vl - (long)i1), __epi_e32, __epi_m1);
                        __epi_2xf32 acc00 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc10 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc20 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc30 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc40 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc50 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc60 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc70 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc80 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc90 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc100 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc110 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc120 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc130 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc140 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc150 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc160 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc170 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc180 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc190 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc200 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc210 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc220 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc230 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc240 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc250 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc260 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc270 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc280 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc290 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc300 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc310 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        const float *b0_ptr = b + i1;
                        const float *a1_ptr = a + 0;
                        const size_t b_increment = nr * simd_width;
                        for (int j = 0; j < k; j++)
                        {

                                const __epi_2xi32 FOUR = __builtin_epi_vbroadcast_2xi32(four, gvl);
                                __epi_2xi32 index1 = __builtin_epi_vload_2xi32(&index1_host[0], gvl);
                                __epi_2xi32 index11 = __builtin_epi_vmul_2xi32(index1, FOUR, gvl);
                                const __epi_2xf32 a0 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a1 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a2 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a3 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a4 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a5 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a6 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a7 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a8 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a9 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a10 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a11 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a12 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a13 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a14 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a15 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a16 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;

                                const __epi_2xf32 b0 = __builtin_epi_vload_2xf32(b0_ptr, gvl);
                                b0_ptr += b_increment;
                                acc00 = __builtin_epi_vfmacc_2xf32(acc00, a0, b0, gvl);
                                acc10 = __builtin_epi_vfmacc_2xf32(acc10, a1, b0, gvl);
                                acc20 = __builtin_epi_vfmacc_2xf32(acc20, a2, b0, gvl);
                                acc30 = __builtin_epi_vfmacc_2xf32(acc30, a3, b0, gvl);
                                acc40 = __builtin_epi_vfmacc_2xf32(acc40, a4, b0, gvl);
                                acc50 = __builtin_epi_vfmacc_2xf32(acc50, a5, b0, gvl);
                                acc60 = __builtin_epi_vfmacc_2xf32(acc60, a6, b0, gvl);
                                acc70 = __builtin_epi_vfmacc_2xf32(acc70, a7, b0, gvl);
                                acc80 = __builtin_epi_vfmacc_2xf32(acc80, a8, b0, gvl);
                                acc90 = __builtin_epi_vfmacc_2xf32(acc90, a9, b0, gvl);
                                acc100 = __builtin_epi_vfmacc_2xf32(acc100, a10, b0, gvl);
                                acc110 = __builtin_epi_vfmacc_2xf32(acc110, a11, b0, gvl);
                                acc120 = __builtin_epi_vfmacc_2xf32(acc120, a12, b0, gvl);
                                acc130 = __builtin_epi_vfmacc_2xf32(acc130, a13, b0, gvl);
                                acc140 = __builtin_epi_vfmacc_2xf32(acc140, a14, b0, gvl);
                                acc150 = __builtin_epi_vfmacc_2xf32(acc150, a15, b0, gvl);
                                acc160 = __builtin_epi_vfmacc_2xf32(acc160, a16, b0, gvl);
                        }
                        float *restrict crow0 = c;
                        float *restrict crow1 = crow0 + row_stride_c;
                        float *restrict crow2 = crow1 + row_stride_c;
                        float *restrict crow3 = crow2 + row_stride_c;
                        float *restrict crow4 = crow3 + row_stride_c;
                        float *restrict crow5 = crow4 + row_stride_c;
                        float *restrict crow6 = crow5 + row_stride_c;
                        float *restrict crow7 = crow6 + row_stride_c;
                        float *restrict crow8 = crow7 + row_stride_c;
                        float *restrict crow9 = crow8 + row_stride_c;
                        float *restrict crow10 = crow9 + row_stride_c;
                        float *restrict crow11 = crow10 + row_stride_c;
                        float *restrict crow12 = crow11 + row_stride_c;
                        float *restrict crow13 = crow12 + row_stride_c;
                        float *restrict crow14 = crow13 + row_stride_c;
                        float *restrict crow15 = crow14 + row_stride_c;
                        float *restrict crow16 = crow15 + row_stride_c;
                        if (update != 0)
                        {
                                __builtin_epi_vstore_2xf32(crow0 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow0 + i1, gvl), acc00, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow1 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow1 + i1, gvl), acc10, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow2 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow2 + i1, gvl), acc20, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow3 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow3 + i1, gvl), acc30, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow4 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow4 + i1, gvl), acc40, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow5 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow5 + i1, gvl), acc50, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow6 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow6 + i1, gvl), acc60, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow7 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow7 + i1, gvl), acc70, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow8 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow8 + i1, gvl), acc80, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow9 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow9 + i1, gvl), acc90, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow10 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow10 + i1, gvl), acc100, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow11 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow11 + i1, gvl), acc110, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow12 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow12 + i1, gvl), acc120, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow13 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow13 + i1, gvl), acc130, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow14 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow14 + i1, gvl), acc140, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow15 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow15 + i1, gvl), acc150, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow16 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow16 + i1, gvl), acc160, gvl), gvl);
                        }
                        else
                        {
                                __builtin_epi_vstore_2xf32(crow0 + i1, acc00, gvl);
                                __builtin_epi_vstore_2xf32(crow1 + i1, acc10, gvl);
                                __builtin_epi_vstore_2xf32(crow2 + i1, acc20, gvl);
                                __builtin_epi_vstore_2xf32(crow3 + i1, acc30, gvl);
                                __builtin_epi_vstore_2xf32(crow4 + i1, acc40, gvl);
                                __builtin_epi_vstore_2xf32(crow5 + i1, acc50, gvl);
                                __builtin_epi_vstore_2xf32(crow6 + i1, acc60, gvl);
                                __builtin_epi_vstore_2xf32(crow7 + i1, acc70, gvl);
                                __builtin_epi_vstore_2xf32(crow8 + i1, acc80, gvl);
                                __builtin_epi_vstore_2xf32(crow9 + i1, acc90, gvl);
                                __builtin_epi_vstore_2xf32(crow10 + i1, acc100, gvl);
                                __builtin_epi_vstore_2xf32(crow11 + i1, acc110, gvl);
                                __builtin_epi_vstore_2xf32(crow12 + i1, acc120, gvl);
                                __builtin_epi_vstore_2xf32(crow13 + i1, acc130, gvl);
                                __builtin_epi_vstore_2xf32(crow14 + i1, acc140, gvl);
                                __builtin_epi_vstore_2xf32(crow15 + i1, acc150, gvl);
                                __builtin_epi_vstore_2xf32(crow16 + i1, acc160, gvl);
                        }

                        i1 += gvl;
                }
                break;
        }
        case 25:
        {
                for (int i1 = 0; i1 < vl;)
                {
                        unsigned long gvl = __builtin_epi_vsetvl(((long)vl - (long)i1), __epi_e32, __epi_m1);
                        __epi_2xf32 acc00 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc10 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc20 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc30 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc40 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc50 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc60 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc70 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc80 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc90 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc100 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc110 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc120 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc130 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc140 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc150 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc160 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc170 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc180 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc190 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc200 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc210 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc220 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc230 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc240 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc250 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc260 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc270 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc280 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc290 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc300 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc310 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        const float *b0_ptr = b + i1;
                        const float *a1_ptr = a + 0;
                        const size_t b_increment = nr * simd_width;
                        for (int j = 0; j < k; j++)
                        {

                                const __epi_2xi32 FOUR = __builtin_epi_vbroadcast_2xi32(four, gvl);
                                __epi_2xi32 index1 = __builtin_epi_vload_2xi32(&index1_host[0], gvl);
                                __epi_2xi32 index11 = __builtin_epi_vmul_2xi32(index1, FOUR, gvl);
                                const __epi_2xf32 a0 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a1 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a2 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a3 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a4 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a5 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a6 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a7 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a8 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a9 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a10 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a11 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a12 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a13 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a14 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a15 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a16 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a17 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a18 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a19 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a20 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a21 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a22 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a23 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a24 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;

                                const __epi_2xf32 b0 = __builtin_epi_vload_2xf32(b0_ptr, gvl);
                                b0_ptr += b_increment;
                                acc00 = __builtin_epi_vfmacc_2xf32(acc00, a0, b0, gvl);
                                acc10 = __builtin_epi_vfmacc_2xf32(acc10, a1, b0, gvl);
                                acc20 = __builtin_epi_vfmacc_2xf32(acc20, a2, b0, gvl);
                                acc30 = __builtin_epi_vfmacc_2xf32(acc30, a3, b0, gvl);
                                acc40 = __builtin_epi_vfmacc_2xf32(acc40, a4, b0, gvl);
                                acc50 = __builtin_epi_vfmacc_2xf32(acc50, a5, b0, gvl);
                                acc60 = __builtin_epi_vfmacc_2xf32(acc60, a6, b0, gvl);
                                acc70 = __builtin_epi_vfmacc_2xf32(acc70, a7, b0, gvl);
                                acc80 = __builtin_epi_vfmacc_2xf32(acc80, a8, b0, gvl);
                                acc90 = __builtin_epi_vfmacc_2xf32(acc90, a9, b0, gvl);
                                acc100 = __builtin_epi_vfmacc_2xf32(acc100, a10, b0, gvl);
                                acc110 = __builtin_epi_vfmacc_2xf32(acc110, a11, b0, gvl);
                                acc120 = __builtin_epi_vfmacc_2xf32(acc120, a12, b0, gvl);
                                acc130 = __builtin_epi_vfmacc_2xf32(acc130, a13, b0, gvl);
                                acc140 = __builtin_epi_vfmacc_2xf32(acc140, a14, b0, gvl);
                                acc150 = __builtin_epi_vfmacc_2xf32(acc150, a15, b0, gvl);
                                acc160 = __builtin_epi_vfmacc_2xf32(acc160, a16, b0, gvl);
                                acc170 = __builtin_epi_vfmacc_2xf32(acc170, a17, b0, gvl);
                                acc180 = __builtin_epi_vfmacc_2xf32(acc180, a18, b0, gvl);
                                acc190 = __builtin_epi_vfmacc_2xf32(acc190, a19, b0, gvl);
                                acc200 = __builtin_epi_vfmacc_2xf32(acc200, a20, b0, gvl);
                                acc210 = __builtin_epi_vfmacc_2xf32(acc210, a21, b0, gvl);
                                acc220 = __builtin_epi_vfmacc_2xf32(acc220, a22, b0, gvl);
                                acc230 = __builtin_epi_vfmacc_2xf32(acc230, a23, b0, gvl);
                                acc240 = __builtin_epi_vfmacc_2xf32(acc240, a24, b0, gvl);
                        }
                        float *restrict crow0 = c;
                        float *restrict crow1 = crow0 + row_stride_c;
                        float *restrict crow2 = crow1 + row_stride_c;
                        float *restrict crow3 = crow2 + row_stride_c;
                        float *restrict crow4 = crow3 + row_stride_c;
                        float *restrict crow5 = crow4 + row_stride_c;
                        float *restrict crow6 = crow5 + row_stride_c;
                        float *restrict crow7 = crow6 + row_stride_c;
                        float *restrict crow8 = crow7 + row_stride_c;
                        float *restrict crow9 = crow8 + row_stride_c;
                        float *restrict crow10 = crow9 + row_stride_c;
                        float *restrict crow11 = crow10 + row_stride_c;
                        float *restrict crow12 = crow11 + row_stride_c;
                        float *restrict crow13 = crow12 + row_stride_c;
                        float *restrict crow14 = crow13 + row_stride_c;
                        float *restrict crow15 = crow14 + row_stride_c;
                        float *restrict crow16 = crow15 + row_stride_c;
                        float *restrict crow17 = crow16 + row_stride_c;
                        float *restrict crow18 = crow17 + row_stride_c;
                        float *restrict crow19 = crow18 + row_stride_c;
                        float *restrict crow20 = crow19 + row_stride_c;
                        float *restrict crow21 = crow20 + row_stride_c;
                        float *restrict crow22 = crow21 + row_stride_c;
                        float *restrict crow23 = crow22 + row_stride_c;
                        float *restrict crow24 = crow23 + row_stride_c;
                        if (update != 0)
                        {
                                __builtin_epi_vstore_2xf32(crow0 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow0 + i1, gvl), acc00, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow1 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow1 + i1, gvl), acc10, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow2 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow2 + i1, gvl), acc20, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow3 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow3 + i1, gvl), acc30, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow4 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow4 + i1, gvl), acc40, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow5 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow5 + i1, gvl), acc50, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow6 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow6 + i1, gvl), acc60, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow7 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow7 + i1, gvl), acc70, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow8 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow8 + i1, gvl), acc80, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow9 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow9 + i1, gvl), acc90, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow10 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow10 + i1, gvl), acc100, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow11 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow11 + i1, gvl), acc110, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow12 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow12 + i1, gvl), acc120, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow13 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow13 + i1, gvl), acc130, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow14 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow14 + i1, gvl), acc140, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow15 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow15 + i1, gvl), acc150, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow16 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow16 + i1, gvl), acc160, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow17 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow17 + i1, gvl), acc170, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow18 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow18 + i1, gvl), acc180, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow19 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow19 + i1, gvl), acc190, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow20 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow20 + i1, gvl), acc200, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow21 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow21 + i1, gvl), acc210, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow22 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow22 + i1, gvl), acc220, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow23 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow23 + i1, gvl), acc230, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow24 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow24 + i1, gvl), acc240, gvl), gvl);
                        }
                        else
                        {
                                __builtin_epi_vstore_2xf32(crow0 + i1, acc00, gvl);
                                __builtin_epi_vstore_2xf32(crow1 + i1, acc10, gvl);
                                __builtin_epi_vstore_2xf32(crow2 + i1, acc20, gvl);
                                __builtin_epi_vstore_2xf32(crow3 + i1, acc30, gvl);
                                __builtin_epi_vstore_2xf32(crow4 + i1, acc40, gvl);
                                __builtin_epi_vstore_2xf32(crow5 + i1, acc50, gvl);
                                __builtin_epi_vstore_2xf32(crow6 + i1, acc60, gvl);
                                __builtin_epi_vstore_2xf32(crow7 + i1, acc70, gvl);
                                __builtin_epi_vstore_2xf32(crow8 + i1, acc80, gvl);
                                __builtin_epi_vstore_2xf32(crow9 + i1, acc90, gvl);
                                __builtin_epi_vstore_2xf32(crow10 + i1, acc100, gvl);
                                __builtin_epi_vstore_2xf32(crow11 + i1, acc110, gvl);
                                __builtin_epi_vstore_2xf32(crow12 + i1, acc120, gvl);
                                __builtin_epi_vstore_2xf32(crow13 + i1, acc130, gvl);
                                __builtin_epi_vstore_2xf32(crow14 + i1, acc140, gvl);
                                __builtin_epi_vstore_2xf32(crow15 + i1, acc150, gvl);
                                __builtin_epi_vstore_2xf32(crow16 + i1, acc160, gvl);
                                __builtin_epi_vstore_2xf32(crow17 + i1, acc170, gvl);
                                __builtin_epi_vstore_2xf32(crow18 + i1, acc180, gvl);
                                __builtin_epi_vstore_2xf32(crow19 + i1, acc190, gvl);
                                __builtin_epi_vstore_2xf32(crow20 + i1, acc200, gvl);
                                __builtin_epi_vstore_2xf32(crow21 + i1, acc210, gvl);
                                __builtin_epi_vstore_2xf32(crow22 + i1, acc220, gvl);
                                __builtin_epi_vstore_2xf32(crow23 + i1, acc230, gvl);
                                __builtin_epi_vstore_2xf32(crow24 + i1, acc240, gvl);
                        }

                        i1 += gvl;
                }
                break;
        }
        case 32:
        {
                for (int i1 = 0; i1 < vl;)
                {
                        unsigned long gvl = __builtin_epi_vsetvl(((long)vl - (long)i1), __epi_e32, __epi_m1);
                        __epi_2xf32 acc00 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc10 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc20 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc30 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc40 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc50 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc60 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc70 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc80 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc90 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc100 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc110 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc120 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc130 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc140 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc150 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc160 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc170 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc180 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc190 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc200 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc210 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc220 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc230 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc240 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc250 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc260 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc270 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc280 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc290 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc300 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc310 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        const float *b0_ptr = b + i1;
                        const float *a1_ptr = a + 0;
                        const size_t b_increment = nr * simd_width;
                        for (int j = 0; j < k; j++)
                        {

                                const __epi_2xi32 FOUR = __builtin_epi_vbroadcast_2xi32(four, gvl);
                                __epi_2xi32 index1 = __builtin_epi_vload_2xi32(&index1_host[0], gvl);
                                __epi_2xi32 index11 = __builtin_epi_vmul_2xi32(index1, FOUR, gvl);
                                const __epi_2xf32 a0 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a1 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a2 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a3 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a4 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a5 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a6 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a7 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a8 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a9 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a10 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a11 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a12 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a13 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a14 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a15 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a16 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a17 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a18 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a19 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a20 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a21 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a22 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a23 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a24 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a25 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a26 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a27 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a28 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a29 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a30 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;
                                const __epi_2xf32 a31 = __builtin_epi_vload_indexed_2xf32(a1_ptr, index11, gvl);
                                a1_ptr += 4;

                                const __epi_2xf32 b0 = __builtin_epi_vload_2xf32(b0_ptr, gvl);
                                b0_ptr += b_increment;
                                acc00 = __builtin_epi_vfmacc_2xf32(acc00, a0, b0, gvl);
                                acc10 = __builtin_epi_vfmacc_2xf32(acc10, a1, b0, gvl);
                                acc20 = __builtin_epi_vfmacc_2xf32(acc20, a2, b0, gvl);
                                acc30 = __builtin_epi_vfmacc_2xf32(acc30, a3, b0, gvl);
                                acc40 = __builtin_epi_vfmacc_2xf32(acc40, a4, b0, gvl);
                                acc50 = __builtin_epi_vfmacc_2xf32(acc50, a5, b0, gvl);
                                acc60 = __builtin_epi_vfmacc_2xf32(acc60, a6, b0, gvl);
                                acc70 = __builtin_epi_vfmacc_2xf32(acc70, a7, b0, gvl);
                                acc80 = __builtin_epi_vfmacc_2xf32(acc80, a8, b0, gvl);
                                acc90 = __builtin_epi_vfmacc_2xf32(acc90, a9, b0, gvl);
                                acc100 = __builtin_epi_vfmacc_2xf32(acc100, a10, b0, gvl);
                                acc110 = __builtin_epi_vfmacc_2xf32(acc110, a11, b0, gvl);
                                acc120 = __builtin_epi_vfmacc_2xf32(acc120, a12, b0, gvl);
                                acc130 = __builtin_epi_vfmacc_2xf32(acc130, a13, b0, gvl);
                                acc140 = __builtin_epi_vfmacc_2xf32(acc140, a14, b0, gvl);
                                acc150 = __builtin_epi_vfmacc_2xf32(acc150, a15, b0, gvl);
                                acc160 = __builtin_epi_vfmacc_2xf32(acc160, a16, b0, gvl);
                                acc170 = __builtin_epi_vfmacc_2xf32(acc170, a17, b0, gvl);
                                acc180 = __builtin_epi_vfmacc_2xf32(acc180, a18, b0, gvl);
                                acc190 = __builtin_epi_vfmacc_2xf32(acc190, a19, b0, gvl);
                                acc200 = __builtin_epi_vfmacc_2xf32(acc200, a20, b0, gvl);
                                acc210 = __builtin_epi_vfmacc_2xf32(acc210, a21, b0, gvl);
                                acc220 = __builtin_epi_vfmacc_2xf32(acc220, a22, b0, gvl);
                                acc230 = __builtin_epi_vfmacc_2xf32(acc230, a23, b0, gvl);
                                acc240 = __builtin_epi_vfmacc_2xf32(acc240, a24, b0, gvl);
                                acc250 = __builtin_epi_vfmacc_2xf32(acc250, a25, b0, gvl);
                                acc260 = __builtin_epi_vfmacc_2xf32(acc260, a26, b0, gvl);
                                acc270 = __builtin_epi_vfmacc_2xf32(acc270, a27, b0, gvl);
                                acc280 = __builtin_epi_vfmacc_2xf32(acc280, a28, b0, gvl);
                                acc290 = __builtin_epi_vfmacc_2xf32(acc290, a29, b0, gvl);
                                acc300 = __builtin_epi_vfmacc_2xf32(acc300, a30, b0, gvl);
                                acc310 = __builtin_epi_vfmacc_2xf32(acc310, a31, b0, gvl);
                        }
                        float *restrict crow0 = c;
                        float *restrict crow1 = crow0 + row_stride_c;
                        float *restrict crow2 = crow1 + row_stride_c;
                        float *restrict crow3 = crow2 + row_stride_c;
                        float *restrict crow4 = crow3 + row_stride_c;
                        float *restrict crow5 = crow4 + row_stride_c;
                        float *restrict crow6 = crow5 + row_stride_c;
                        float *restrict crow7 = crow6 + row_stride_c;
                        float *restrict crow8 = crow7 + row_stride_c;
                        float *restrict crow9 = crow8 + row_stride_c;
                        float *restrict crow10 = crow9 + row_stride_c;
                        float *restrict crow11 = crow10 + row_stride_c;
                        float *restrict crow12 = crow11 + row_stride_c;
                        float *restrict crow13 = crow12 + row_stride_c;
                        float *restrict crow14 = crow13 + row_stride_c;
                        float *restrict crow15 = crow14 + row_stride_c;
                        float *restrict crow16 = crow15 + row_stride_c;
                        float *restrict crow17 = crow16 + row_stride_c;
                        float *restrict crow18 = crow17 + row_stride_c;
                        float *restrict crow19 = crow18 + row_stride_c;
                        float *restrict crow20 = crow19 + row_stride_c;
                        float *restrict crow21 = crow20 + row_stride_c;
                        float *restrict crow22 = crow21 + row_stride_c;
                        float *restrict crow23 = crow22 + row_stride_c;
                        float *restrict crow24 = crow23 + row_stride_c;
                        float *restrict crow25 = crow24 + row_stride_c;
                        float *restrict crow26 = crow25 + row_stride_c;
                        float *restrict crow27 = crow26 + row_stride_c;
                        float *restrict crow28 = crow27 + row_stride_c;
                        float *restrict crow29 = crow28 + row_stride_c;
                        float *restrict crow30 = crow29 + row_stride_c;
                        float *restrict crow31 = crow30 + row_stride_c;
                        if (update != 0)
                        {
                                __builtin_epi_vstore_2xf32(crow0 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow0 + i1, gvl), acc00, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow1 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow1 + i1, gvl), acc10, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow2 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow2 + i1, gvl), acc20, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow3 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow3 + i1, gvl), acc30, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow4 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow4 + i1, gvl), acc40, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow5 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow5 + i1, gvl), acc50, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow6 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow6 + i1, gvl), acc60, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow7 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow7 + i1, gvl), acc70, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow8 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow8 + i1, gvl), acc80, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow9 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow9 + i1, gvl), acc90, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow10 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow10 + i1, gvl), acc100, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow11 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow11 + i1, gvl), acc110, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow12 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow12 + i1, gvl), acc120, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow13 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow13 + i1, gvl), acc130, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow14 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow14 + i1, gvl), acc140, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow15 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow15 + i1, gvl), acc150, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow16 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow16 + i1, gvl), acc160, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow17 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow17 + i1, gvl), acc170, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow18 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow18 + i1, gvl), acc180, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow19 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow19 + i1, gvl), acc190, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow20 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow20 + i1, gvl), acc200, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow21 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow21 + i1, gvl), acc210, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow22 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow22 + i1, gvl), acc220, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow23 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow23 + i1, gvl), acc230, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow24 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow24 + i1, gvl), acc240, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow25 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow25 + i1, gvl), acc250, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow26 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow26 + i1, gvl), acc260, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow27 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow27 + i1, gvl), acc270, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow28 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow28 + i1, gvl), acc280, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow29 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow29 + i1, gvl), acc290, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow30 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow30 + i1, gvl), acc300, gvl), gvl);
                                __builtin_epi_vstore_2xf32(crow31 + i1, __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(crow31 + i1, gvl), acc310, gvl), gvl);
                        }
                        else
                        {
                                __builtin_epi_vstore_2xf32(crow0 + i1, acc00, gvl);
                                __builtin_epi_vstore_2xf32(crow1 + i1, acc10, gvl);
                                __builtin_epi_vstore_2xf32(crow2 + i1, acc20, gvl);
                                __builtin_epi_vstore_2xf32(crow3 + i1, acc30, gvl);
                                __builtin_epi_vstore_2xf32(crow4 + i1, acc40, gvl);
                                __builtin_epi_vstore_2xf32(crow5 + i1, acc50, gvl);
                                __builtin_epi_vstore_2xf32(crow6 + i1, acc60, gvl);
                                __builtin_epi_vstore_2xf32(crow7 + i1, acc70, gvl);
                                __builtin_epi_vstore_2xf32(crow8 + i1, acc80, gvl);
                                __builtin_epi_vstore_2xf32(crow9 + i1, acc90, gvl);
                                __builtin_epi_vstore_2xf32(crow10 + i1, acc100, gvl);
                                __builtin_epi_vstore_2xf32(crow11 + i1, acc110, gvl);
                                __builtin_epi_vstore_2xf32(crow12 + i1, acc120, gvl);
                                __builtin_epi_vstore_2xf32(crow13 + i1, acc130, gvl);
                                __builtin_epi_vstore_2xf32(crow14 + i1, acc140, gvl);
                                __builtin_epi_vstore_2xf32(crow15 + i1, acc150, gvl);
                                __builtin_epi_vstore_2xf32(crow16 + i1, acc160, gvl);
                                __builtin_epi_vstore_2xf32(crow17 + i1, acc170, gvl);
                                __builtin_epi_vstore_2xf32(crow18 + i1, acc180, gvl);
                                __builtin_epi_vstore_2xf32(crow19 + i1, acc190, gvl);
                                __builtin_epi_vstore_2xf32(crow20 + i1, acc200, gvl);
                                __builtin_epi_vstore_2xf32(crow21 + i1, acc210, gvl);
                                __builtin_epi_vstore_2xf32(crow22 + i1, acc220, gvl);
                                __builtin_epi_vstore_2xf32(crow23 + i1, acc230, gvl);
                                __builtin_epi_vstore_2xf32(crow24 + i1, acc240, gvl);
                                __builtin_epi_vstore_2xf32(crow25 + i1, acc250, gvl);
                                __builtin_epi_vstore_2xf32(crow26 + i1, acc260, gvl);
                                __builtin_epi_vstore_2xf32(crow27 + i1, acc270, gvl);
                                __builtin_epi_vstore_2xf32(crow28 + i1, acc280, gvl);
                                __builtin_epi_vstore_2xf32(crow29 + i1, acc290, gvl);
                                __builtin_epi_vstore_2xf32(crow30 + i1, acc300, gvl);
                                __builtin_epi_vstore_2xf32(crow31 + i1, acc310, gvl);
                        }

                        i1 += gvl;
                }
                break;
        }
        default:
                printf("not an option");
        }
}

void nnp_s4gemm_only_3x3__neon(
    size_t k, size_t update,
    const float a[restrict static 1],
    const float b[restrict static 1],
    float c[restrict static 1],
    size_t row_stride_c)
{
        // printf("%d", row_stride_c);
        const int simd_width = __builtin_epi_vsetvlmax(__epi_e32, __epi_m1); // nnp_hwinfo.sve_simd_width;//nnp_hwinfo.simd_width;
        int index_host[simd_width];
        int index1_host[simd_width];
        int rem = __builtin_epi_vsetvlmax(__epi_e32, __epi_m1) / 4;
        int vl = nnp_hwinfo.sxgemm.mr * 4;
        int index_max = vl / 64;
        for (int i1 = 0; i1 < vl;)
        {
                unsigned long gvl = __builtin_epi_vsetvl(((long)vl - (long)i1), __epi_e32, __epi_m1);
                int inc = 0;
                int four = 4;
                const __epi_2xi32 FOUR = __builtin_epi_vbroadcast_2xi32(four, gvl);
                for (int i = 0; i < rem; i++)
                {
                        for (int ind = 0; ind < 4; ind++)
                        {
                                index1_host[inc] = ind;
                                inc++;
                        }
                }

                __epi_2xi32 index1 = __builtin_epi_vload_2xi32(&index1_host[0], gvl);
                __epi_2xi32 index11 = __builtin_epi_vmul_2xi32(index1, FOUR, gvl);

                for (int index = 0; index < index_max; index++)
                {
                        const float *a1 = &a[0];
                        // svbool_t pg = svwhilelt_b32(i1,64);
                        __epi_2xf32 acc00 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc10 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc20 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc30 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc40 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc50 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc60 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc70 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc80 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc90 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc100 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc110 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc120 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc130 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc140 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc150 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc01 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc11 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc21 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc31 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc02 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc12 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc22 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc32 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc03 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc13 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc23 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc33 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc41 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc51 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc61 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc71 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc42 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc52 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc62 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc72 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        __epi_2xf32 acc43 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc53 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc63 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl), acc73 = __builtin_epi_vbroadcast_2xf32(0.0f, gvl);
                        for (int j = 0; j < k; j++)
                        {
                                __epi_2xf32 b0 = __builtin_epi_vload_2xf32(&b[i1 + (j * vl)], gvl);
                                const __epi_2xf32 a0 = __builtin_epi_vload_indexed_2xf32(&a[(0 + (64 * index)) + (j * vl)], index11, gvl);
                                acc00 = __builtin_epi_vfmacc_2xf32(acc00, a0, b0, gvl);
                                const __epi_2xf32 a1 = __builtin_epi_vload_indexed_2xf32(&a[(4 + (64 * index)) + (j * vl)], index11, gvl);
                                acc10 = __builtin_epi_vfmacc_2xf32(acc10, a1, b0, gvl);
                                const __epi_2xf32 a2 = __builtin_epi_vload_indexed_2xf32(&a[(8 + (index * 64)) + (j * vl)], index11, gvl);
                                acc20 = __builtin_epi_vfmacc_2xf32(acc20, a2, b0, gvl);
                                const __epi_2xf32 a3 = __builtin_epi_vload_indexed_2xf32(&a[(12 + (64 * index)) + (j * vl)], index11, gvl);
                                acc30 = __builtin_epi_vfmacc_2xf32(acc30, a3, b0, gvl);
                                const __epi_2xf32 a4 = __builtin_epi_vload_indexed_2xf32(&a[(16 + (64 * index)) + (j * vl)], index11, gvl);
                                acc40 = __builtin_epi_vfmacc_2xf32(acc40, a4, b0, gvl);
                                const __epi_2xf32 a5 = __builtin_epi_vload_indexed_2xf32(&a[(20 + (64 * index)) + (j * vl)], index11, gvl);
                                acc50 = __builtin_epi_vfmacc_2xf32(acc50, a5, b0, gvl);
                                const __epi_2xf32 a6 = __builtin_epi_vload_indexed_2xf32(&a[(24 + (64 * index)) + (j * vl)], index11, gvl);
                                acc60 = __builtin_epi_vfmacc_2xf32(acc60, a6, b0, gvl);
                                const __epi_2xf32 a7 = __builtin_epi_vload_indexed_2xf32(&a[(28 + (64 * index)) + (j * vl)], index11, gvl);
                                acc70 = __builtin_epi_vfmacc_2xf32(acc70, a7, b0, gvl);
                                const __epi_2xf32 a8 = __builtin_epi_vload_indexed_2xf32(&a[(32 + (64 * index)) + (j * vl)], index11, gvl);
                                acc80 = __builtin_epi_vfmacc_2xf32(acc80, a8, b0, gvl);
                                const __epi_2xf32 a9 = __builtin_epi_vload_indexed_2xf32(&a[(36 + (64 * index)) + (j * vl)], index11, gvl);
                                acc90 = __builtin_epi_vfmacc_2xf32(acc90, a9, b0, gvl);
                                const __epi_2xf32 a10 = __builtin_epi_vload_indexed_2xf32(&a[(40 + (64 * index)) + (j * vl)], index11, gvl);
                                acc100 = __builtin_epi_vfmacc_2xf32(acc100, a10, b0, gvl);
                                const __epi_2xf32 a11 = __builtin_epi_vload_indexed_2xf32(&a[(44 + (64 * index)) + (j * vl)], index11, gvl);
                                acc110 = __builtin_epi_vfmacc_2xf32(acc110, a11, b0, gvl);
                                const __epi_2xf32 a12 = __builtin_epi_vload_indexed_2xf32(&a[(48 + (64 * index)) + (j * vl)], index11, gvl);
                                acc120 = __builtin_epi_vfmacc_2xf32(acc120, a12, b0, gvl);
                                const __epi_2xf32 a13 = __builtin_epi_vload_indexed_2xf32(&a[(52 + (64 * index)) + (j * vl)], index11, gvl);
                                acc130 = __builtin_epi_vfmacc_2xf32(acc130, a13, b0, gvl);
                                const __epi_2xf32 a14 = __builtin_epi_vload_indexed_2xf32(&a[(56 + (64 * index)) + (j * vl)], index11, gvl);
                                acc140 = __builtin_epi_vfmacc_2xf32(acc140, a14, b0, gvl);
                                const __epi_2xf32 a15 = __builtin_epi_vload_indexed_2xf32(&a[(60 + (64 * index)) + (j * vl)], index11, gvl);
                                acc150 = __builtin_epi_vfmacc_2xf32(acc150, a15, b0, gvl);
                        }
                        if (update != 0)
                        {
                                __builtin_epi_vstore_2xf32(&c[i1 + (16 * index * row_stride_c)], __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(&c[i1 + ((16 * index) * row_stride_c)], gvl), acc00, gvl), gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 1) * row_stride_c)], __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(&c[i1 + ((16 * index + 1) * row_stride_c)], gvl), acc10, gvl), gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 2) * row_stride_c)], __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(&c[i1 + ((16 * index + 2) * row_stride_c)], gvl), acc20, gvl), gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 3) * row_stride_c)], __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(&c[i1 + ((16 * index + 3) * row_stride_c)], gvl), acc30, gvl), gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 4) * row_stride_c)], __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(&c[i1 + ((16 * index + 4) * row_stride_c)], gvl), acc40, gvl), gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 5) * row_stride_c)], __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(&c[i1 + ((16 * index + 5) * row_stride_c)], gvl), acc50, gvl), gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 6) * row_stride_c)], __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(&c[i1 + ((16 * index + 6) * row_stride_c)], gvl), acc60, gvl), gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 7) * row_stride_c)], __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(&c[i1 + ((16 * index + 7) * row_stride_c)], gvl), acc70, gvl), gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 8) * row_stride_c)], __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(&c[i1 + ((16 * index + 8) * row_stride_c)], gvl), acc80, gvl), gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 9) * row_stride_c)], __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(&c[i1 + ((16 * index + 9) * row_stride_c)], gvl), acc90, gvl), gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 10) * row_stride_c)], __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(&c[i1 + ((16 * index + 10) * row_stride_c)], gvl), acc100, gvl), gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 11) * row_stride_c)], __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(&c[i1 + ((16 * index + 11) * row_stride_c)], gvl), acc110, gvl), gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 12) * row_stride_c)], __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(&c[i1 + ((16 * index + 12) * row_stride_c)], gvl), acc120, gvl), gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 13) * row_stride_c)], __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(&c[i1 + ((16 * index + 13) * row_stride_c)], gvl), acc130, gvl), gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 14) * row_stride_c)], __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(&c[i1 + ((16 * index + 14) * row_stride_c)], gvl), acc140, gvl), gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 15) * row_stride_c)], __builtin_epi_vfadd_2xf32(__builtin_epi_vload_2xf32(&c[i1 + ((16 * index + 15) * row_stride_c)], gvl), acc150, gvl), gvl);
                        }
                        else
                        {
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index) * row_stride_c)], acc00, gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 1) * row_stride_c)], acc10, gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 2) * row_stride_c)], acc20, gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 3) * row_stride_c)], acc30, gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 4) * row_stride_c)], acc40, gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 5) * row_stride_c)], acc50, gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 6) * row_stride_c)], acc60, gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 7) * row_stride_c)], acc70, gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 8) * row_stride_c)], acc80, gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 9) * row_stride_c)], acc90, gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 10) * row_stride_c)], acc100, gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 11) * row_stride_c)], acc110, gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 12) * row_stride_c)], acc120, gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 13) * row_stride_c)], acc130, gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 14) * row_stride_c)], acc140, gvl);
                                __builtin_epi_vstore_2xf32(&c[i1 + ((16 * index + 15) * row_stride_c)], acc150, gvl);
                        }
                }
                i1 += gvl;
        }
}
bool rescale_coefficients = true;

static void winograd_f6k3_input_transform1(
    const __epi_2xf32 d0,
    const __epi_2xf32 d1,
    const __epi_2xf32 d2,
    const __epi_2xf32 d3,
    const __epi_2xf32 d4,
    const __epi_2xf32 d5,
    const __epi_2xf32 d6,
    const __epi_2xf32 d7,
    __epi_2xf32 transform0,
    __epi_2xf32 transform1,
    __epi_2xf32 transform2,
    __epi_2xf32 transform3,
    __epi_2xf32 transform4,
    __epi_2xf32 transform5,
    __epi_2xf32 transform6,
    __epi_2xf32 transform7)
{
        int simd_width = 4; // nnp_hwinfo.simd_width;

        for (int i1 = 0; i1 < simd_width;)
        {
                unsigned long gvl = __builtin_epi_vsetvl(((long)simd_width - (long)i1), __epi_e32, __epi_m1);
                // svfloat32_t const_0_25__5_00 = svld1rq(pg, datatmp);

                float const_0_25 = 0.25f;
                float const_0_5 = 5.00f;
                __epi_2xf32 vconst_0_25 = __builtin_epi_vbroadcast_2xf32(const_0_25, gvl);
                __epi_2xf32 vconst_0_5 = __builtin_epi_vbroadcast_2xf32(const_0_5, gvl);
                // const float32x4_t const_0_25 = vmovq_n_f32(0.25f);

                // Compute wd0 := d0 - d6
                __epi_2xf32 wd0 = __builtin_epi_vfsub_2xf32(d0, d6, gvl);
                const __epi_2xf32 d4_sub_d2 = __builtin_epi_vfsub_2xf32(d4, d2, gvl);

                // Compute wd7 := d7 - d1
                __epi_2xf32 wd7 = __builtin_epi_vfsub_2xf32(d7, d1, gvl);
                const __epi_2xf32 d3_sub_d5 = __builtin_epi_vfsub_2xf32(d3, d5, gvl);
                // float32x4_t wd1 := d2 + d6
                __epi_2xf32 wd1 = __builtin_epi_vfadd_2xf32(d2, d6, gvl);
                // Compute wd2 := d1 + d5
                __epi_2xf32 wd2 = __builtin_epi_vfadd_2xf32(d1, d5, gvl);
                // Compute wd4 := d5 + 0.25 * d1
                __epi_2xf32 wd4 = __builtin_epi_vfmacc_2xf32(d5, vconst_0_25, d1, gvl);
                // Compute wd5 := d6 - 5.0 * d4
                __epi_2xf32 wd5 = __builtin_epi_vfmsac_2xf32(d6, vconst_0_5, d4, gvl);
                // Compute wd3 := d6 + 0.25 * d2
                __epi_2xf32 wd3 = __builtin_epi_vfmacc_2xf32(d6, vconst_0_25, d2, gvl);
                // Compute wd6 := d1 + 0.25 * d5
                __epi_2xf32 wd6 = __builtin_epi_vfmacc_2xf32(d1, vconst_0_25, d5, gvl);
                // const svfloat32_t const_5_25__4_25 = svld1rq(pg, datatmp1);
                const float const_5_25 = 5.25f;
                const float const_4_25 = 4.25f;
                __epi_2xf32 vconst_5_25 = __builtin_epi_vbroadcast_2xf32(const_5_25, gvl);
                __epi_2xf32 vconst_4_25 = __builtin_epi_vbroadcast_2xf32(const_4_25, gvl);
                // Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
                wd0 = __builtin_epi_vfmacc_2xf32(wd0, d4_sub_d2, vconst_5_25, gvl);
                // Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
                wd7 = __builtin_epi_vfmacc_2xf32(wd7, d3_sub_d5, vconst_5_25, gvl);

                // Compute
                //   wd1 := (d6 + d2) - 4.25 * d4
                //   wd2 := (d1 + d5) - 4.25 * d3
                wd1 = __builtin_epi_vfmsac_2xf32(wd1, vconst_4_25, d4, gvl);
                wd2 = __builtin_epi_vfmsac_2xf32(wd2, vconst_4_25, d3, gvl);
                // const svfloat32_t const_1_25__4_00 = svld1(pg, datatmp2);
                const float const_1_25 = 1.25f;
                const float const_4_00 = 4.00f;
                __epi_2xf32 vconst_1_25 = __builtin_epi_vbroadcast_2xf32(const_1_25, gvl);
                __epi_2xf32 vconst_4_00 = __builtin_epi_vbroadcast_2xf32(const_4_00, gvl);
                // Compute
                //   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
                //   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
                //   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
                //   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
                wd3 = __builtin_epi_vfmsub_2xf32(wd3, vconst_1_25, d4, gvl);
                wd5 = __builtin_epi_vfmacc_2xf32(wd5, vconst_4_00, d2, gvl);
                wd4 = __builtin_epi_vfmsub_2xf32(wd4, vconst_1_25, d3, gvl);
                wd6 = __builtin_epi_vfmsub_2xf32(wd6, vconst_1_25, d3, gvl);

                const float const_2 = 2.0f;
                __epi_2xf32 vconst_2 = __builtin_epi_vbroadcast_2xf32(const_2, gvl);

                transform0 = wd0;
                transform1 = __builtin_epi_vfadd_2xf32(wd1, wd2, gvl);
                transform2 = __builtin_epi_vfsub_2xf32(wd1, wd2, gvl);
                transform3 = __builtin_epi_vfmacc_2xf32(wd3, vconst_2, wd4, gvl);
                transform4 = __builtin_epi_vfmsac_2xf32(wd3, vconst_2, wd4, gvl);
                transform5 = __builtin_epi_vfmacc_2xf32(wd5, vconst_2, wd6, gvl);
                transform6 = __builtin_epi_vfmsac_2xf32(wd5, vconst_2, wd6, gvl);
                transform7 = wd7;
                // for(int i=0;i<4;i++)
                //       printf("value \n\n%f", buff[i]);
                i1 += gvl;
        }
}

static void winograd_f6k3_input_transform_intertile(
    const __epi_2xf32 d0,
    const __epi_2xf32 d1,
    const __epi_2xf32 d2,
    const __epi_2xf32 d3,
    const __epi_2xf32 d4,
    const __epi_2xf32 d5,
    const __epi_2xf32 d6,
    const __epi_2xf32 d7,
    __epi_2xf32 transform0,
    __epi_2xf32 transform1,
    __epi_2xf32 transform2,
    __epi_2xf32 transform3,
    __epi_2xf32 transform4,
    __epi_2xf32 transform5,
    __epi_2xf32 transform6,
    __epi_2xf32 transform7)
{
        int simd_width = nnp_hwinfo.sve_simd_width;

        for (int i1 = 0; i1 < simd_width;)
        {
                unsigned long gvl = __builtin_epi_vsetvl(((long)simd_width - (long)i1), __epi_e32, __epi_m1);
                // svfloat32_t const_0_25__5_00 = svld1rq(pg, datatmp);

                float const_0_25 = 0.25f;
                float const_0_5 = 5.00f;
                __epi_2xf32 vconst_0_25 = __builtin_epi_vbroadcast_2xf32(const_0_25, gvl);
                __epi_2xf32 vconst_0_5 = __builtin_epi_vbroadcast_2xf32(const_0_5, gvl);
                // const float32x4_t const_0_25 = vmovq_n_f32(0.25f);

                // Compute wd0 := d0 - d6
                __epi_2xf32 wd0 = __builtin_epi_vfsub_2xf32(d0, d6, gvl);
                const __epi_2xf32 d4_sub_d2 = __builtin_epi_vfsub_2xf32(d4, d2, gvl);

                // Compute wd7 := d7 - d1
                __epi_2xf32 wd7 = __builtin_epi_vfsub_2xf32(d7, d1, gvl);
                const __epi_2xf32 d3_sub_d5 = __builtin_epi_vfsub_2xf32(d3, d5, gvl);
                // float32x4_t wd1 := d2 + d6
                __epi_2xf32 wd1 = __builtin_epi_vfadd_2xf32(d2, d6, gvl);
                // Compute wd2 := d1 + d5
                __epi_2xf32 wd2 = __builtin_epi_vfadd_2xf32(d1, d5, gvl);
                // Compute wd4 := d5 + 0.25 * d1
                __epi_2xf32 wd4 = __builtin_epi_vfmacc_2xf32(d5, vconst_0_25, d1, gvl);
                // Compute wd5 := d6 - 5.0 * d4
                __epi_2xf32 wd5 = __builtin_epi_vfmsac_2xf32(d6, vconst_0_5, d4, gvl);
                // Compute wd3 := d6 + 0.25 * d2
                __epi_2xf32 wd3 = __builtin_epi_vfmacc_2xf32(d6, vconst_0_25, d2, gvl);
                // Compute wd6 := d1 + 0.25 * d5
                __epi_2xf32 wd6 = __builtin_epi_vfmacc_2xf32(d1, vconst_0_25, d5, gvl);
                // const svfloat32_t const_5_25__4_25 = svld1rq(pg, datatmp1);
                const float const_5_25 = 5.25f;
                const float const_4_25 = 4.25f;
                __epi_2xf32 vconst_5_25 = __builtin_epi_vbroadcast_2xf32(const_5_25, gvl);
                __epi_2xf32 vconst_4_25 = __builtin_epi_vbroadcast_2xf32(const_4_25, gvl);
                // Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
                wd0 = __builtin_epi_vfmacc_2xf32(wd0, d4_sub_d2, vconst_5_25, gvl);
                // Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
                wd7 = __builtin_epi_vfmacc_2xf32(wd7, d3_sub_d5, vconst_5_25, gvl);

                // Compute
                //   wd1 := (d6 + d2) - 4.25 * d4
                //   wd2 := (d1 + d5) - 4.25 * d3
                wd1 = __builtin_epi_vfmsac_2xf32(wd1, vconst_4_25, d4, gvl);
                wd2 = __builtin_epi_vfmsac_2xf32(wd2, vconst_4_25, d3, gvl);
                // const svfloat32_t const_1_25__4_00 = svld1(pg, datatmp2);
                const float const_1_25 = 1.25f;
                const float const_4_00 = 4.00f;
                __epi_2xf32 vconst_1_25 = __builtin_epi_vbroadcast_2xf32(const_1_25, gvl);
                __epi_2xf32 vconst_4_00 = __builtin_epi_vbroadcast_2xf32(const_4_00, gvl);
                // Compute
                //   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
                //   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
                //   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
                //   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
                wd3 = __builtin_epi_vfmsub_2xf32(wd3, vconst_1_25, d4, gvl);
                wd5 = __builtin_epi_vfmacc_2xf32(wd5, vconst_4_00, d2, gvl);
                wd4 = __builtin_epi_vfmsub_2xf32(wd4, vconst_1_25, d3, gvl);
                wd6 = __builtin_epi_vfmsub_2xf32(wd6, vconst_1_25, d3, gvl);

                const float const_2 = 2.0f;
                __epi_2xf32 vconst_2 = __builtin_epi_vbroadcast_2xf32(const_2, gvl);

                transform0 = wd0;
                transform1 = __builtin_epi_vfadd_2xf32(wd1, wd2, gvl);
                transform2 = __builtin_epi_vfsub_2xf32(wd1, wd2, gvl);
                transform3 = __builtin_epi_vfmacc_2xf32(wd3, vconst_2, wd4, gvl);
                transform4 = __builtin_epi_vfmsac_2xf32(wd3, vconst_2, wd4, gvl);
                transform5 = __builtin_epi_vfmacc_2xf32(wd5, vconst_2, wd6, gvl);
                transform6 = __builtin_epi_vfmsac_2xf32(wd5, vconst_2, wd6, gvl);
                transform7 = wd7;
                // for(int i=0;i<4;i++)
                //       printf("value \n\n%f", buff[i]);
                i1 += gvl;
        }
}

void nnp_iwt8x8_3x3_with_offset__neon(
    const float data[restrict static 1],
    void *transform,
    size_t data_stride,
    size_t transform_stride,
    uint32_t row_count,
    uint32_t column_count,
    uint32_t row_offset,
    uint32_t column_offset)
{

        int interchannels = 1; // nnp.hwinfo.globalinterchannels;
        int simd_width = nnp_hwinfo.simd_width;

        float vin0123[4 * simd_width], vin4567[4 * simd_width], vin01231[4 * simd_width], vin45671[4 * simd_width];
        //      printf("row_count=%d column_count=%d, transform_stride=%d, data_stride=%d\n", row_count, column_count, transform_stride, data_stride);
        __epi_2xf32 wd0, wd1, wd2, wd3, wd4, wd5, wd6, wd7, wd8, wd9, wd10, wd11, wd12, wd13, wd14, wd15;

        for (int i1 = 0; i1 < simd_width;)
        {
                unsigned long gvl = __builtin_epi_vsetvl(((long)simd_width - (long)i1), __epi_e32, __epi_m1);
                int four = 4;
                const __epi_2xi32 FOUR = __builtin_epi_vbroadcast_2xi32(four, gvl);

                if (row_count == 8 && column_count == 8 && row_offset == 0 && column_offset == 0)
                {
                        // Fast path where we can directly load `data` into `wd`.

                        __epi_2xf32 d0 = __builtin_epi_vload_2xf32(&data[0 * data_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d1 = __builtin_epi_vload_2xf32(&data[1 * data_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d2 = __builtin_epi_vload_2xf32(&data[2 * data_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d3 = __builtin_epi_vload_2xf32(&data[3 * data_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d4 = __builtin_epi_vload_2xf32(&data[4 * data_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d5 = __builtin_epi_vload_2xf32(&data[5 * data_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d6 = __builtin_epi_vload_2xf32(&data[6 * data_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d7 = __builtin_epi_vload_2xf32(&data[7 * data_stride + 0 * simd_width], gvl);
                        __epi_2xf32 wd0, wd1, wd2, wd3, wd8, wd9, wd10, wd11;
                        float const_0_25 = 0.25f;
                        float const_0_5 = 5.00f;
                        __epi_2xf32 vconst_0_25 = __builtin_epi_vbroadcast_2xf32(const_0_25, gvl);
                        __epi_2xf32 vconst_0_5 = __builtin_epi_vbroadcast_2xf32(const_0_5, gvl);
                        // const float32x4_t const_0_25 = vmovq_n_f32(0.25f);

                        // Compute wd0 := d0 - d6
                        __epi_2xf32 wd00 = __builtin_epi_vfsub_2xf32(d0, d6, gvl);
                        const __epi_2xf32 d4_sub_d2 = __builtin_epi_vfsub_2xf32(d4, d2, gvl);

                        // Compute wd7 := d7 - d1
                        __epi_2xf32 wd07 = __builtin_epi_vfsub_2xf32(d7, d1, gvl);
                        const __epi_2xf32 d3_sub_d5 = __builtin_epi_vfsub_2xf32(d3, d5, gvl);
                        // float32x4_t wd1 := d2 + d6
                        __epi_2xf32 wd01 = __builtin_epi_vfadd_2xf32(d2, d6, gvl);
                        // Compute wd2 := d1 + d5
                        __epi_2xf32 wd02 = __builtin_epi_vfadd_2xf32(d1, d5, gvl);
                        // Compute wd4 := d5 + 0.25 * d1
                        __epi_2xf32 wd04 = __builtin_epi_vfmacc_2xf32(d5, vconst_0_25, d1, gvl);
                        // Compute wd5 := d6 - 5.0 * d4
                        __epi_2xf32 wd05 = __builtin_epi_vfsub_2xf32(d6, __builtin_epi_vfmul_2xf32(vconst_0_5, d4, gvl), gvl);
                        // Compute wd3 := d6 + 0.25 * d2
                        __epi_2xf32 wd03 = __builtin_epi_vfmacc_2xf32(d6, vconst_0_25, d2, gvl);
                        // Compute wd6 := d1 + 0.25 * d5
                        __epi_2xf32 wd06 = __builtin_epi_vfmacc_2xf32(d1, vconst_0_25, d5, gvl);
                        // const svfloat32_t const_5_25__4_25 = svld1rq(pg, datatmp1);
                        const float const_5_25 = 5.25f;
                        const float const_4_25 = 4.25f;
                        __epi_2xf32 vconst_5_25 = __builtin_epi_vbroadcast_2xf32(const_5_25, gvl);
                        __epi_2xf32 vconst_4_25 = __builtin_epi_vbroadcast_2xf32(const_4_25, gvl);
                        // Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
                        wd00 = __builtin_epi_vfmacc_2xf32(wd00, d4_sub_d2, vconst_5_25, gvl);
                        // Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
                        wd07 = __builtin_epi_vfmacc_2xf32(wd07, d3_sub_d5, vconst_5_25, gvl);

                        // Compute
                        //   wd1 := (d6 + d2) - 4.25 * d4
                        //   wd2 := (d1 + d5) - 4.25 * d3
                        wd01 = __builtin_epi_vfsub_2xf32(wd01, __builtin_epi_vfmul_2xf32(vconst_4_25, d4, gvl), gvl);
                        wd02 = __builtin_epi_vfsub_2xf32(wd02, __builtin_epi_vfmul_2xf32(vconst_4_25, d3, gvl), gvl);
                        // const svfloat32_t const_1_25__4_00 = svld1(pg, datatmp2);
                        const float const_1_25 = 1.25f;
                        const float const_4_00 = 4.00f;
                        __epi_2xf32 vconst_1_25 = __builtin_epi_vbroadcast_2xf32(const_1_25, gvl);
                        __epi_2xf32 vconst_4_00 = __builtin_epi_vbroadcast_2xf32(const_4_00, gvl);
                        // Compute
                        //   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
                        //   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
                        //   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
                        //   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
                        wd03 = __builtin_epi_vfsub_2xf32(wd03, __builtin_epi_vfmul_2xf32(vconst_1_25, d4, gvl), gvl);
                        wd05 = __builtin_epi_vfmacc_2xf32(wd05, vconst_4_00, d2, gvl);
                        wd04 = __builtin_epi_vfsub_2xf32(wd04, __builtin_epi_vfmul_2xf32(vconst_1_25, d3, gvl), gvl);
                        wd06 = __builtin_epi_vfsub_2xf32(wd06, __builtin_epi_vfmul_2xf32(vconst_1_25, d3, gvl), gvl);

                        const float const_2 = 2.0f;
                        __epi_2xf32 vconst_2 = __builtin_epi_vbroadcast_2xf32(const_2, gvl);

                        wd0 = wd00;
                        wd1 = __builtin_epi_vfadd_2xf32(wd01, wd02, gvl);
                        wd2 = __builtin_epi_vfsub_2xf32(wd01, wd02, gvl);
                        wd3 = __builtin_epi_vfmacc_2xf32(wd03, vconst_2, wd04, gvl);
                        wd8 = __builtin_epi_vfsub_2xf32(wd03, __builtin_epi_vfmul_2xf32(vconst_2, wd04, gvl), gvl);
                        wd9 = __builtin_epi_vfmacc_2xf32(wd05, vconst_2, wd06, gvl);
                        wd10 = __builtin_epi_vfsub_2xf32(wd05, __builtin_epi_vfmul_2xf32(vconst_2, wd06, gvl), gvl);
                        wd11 = wd07;
                        __builtin_epi_vstore_2xf32(&vin0123[0], wd0, gvl);
                        __builtin_epi_vstore_2xf32(&vin0123[simd_width], wd1, gvl);
                        __builtin_epi_vstore_2xf32(&vin0123[2 * simd_width], wd2, gvl);
                        __builtin_epi_vstore_2xf32(&vin0123[3 * simd_width], wd3, gvl);
                        __builtin_epi_vstore_2xf32(&vin01231[0], wd8, gvl);
                        __builtin_epi_vstore_2xf32(&vin01231[simd_width], wd9, gvl);
                        __builtin_epi_vstore_2xf32(&vin01231[2 * simd_width], wd10, gvl);
                        __builtin_epi_vstore_2xf32(&vin01231[3 * simd_width], wd11, gvl);

                        __epi_2xf32 d8 = __builtin_epi_vload_2xf32(&data[0 * data_stride + 1 * simd_width], gvl);
                        __epi_2xf32 d9 = __builtin_epi_vload_2xf32(&data[1 * data_stride + 1 * simd_width], gvl);
                        __epi_2xf32 d10 = __builtin_epi_vload_2xf32(&data[2 * data_stride + 1 * simd_width], gvl);
                        __epi_2xf32 d11 = __builtin_epi_vload_2xf32(&data[3 * data_stride + 1 * simd_width], gvl);

                        __epi_2xf32 d12 = __builtin_epi_vload_2xf32(&data[4 * data_stride + 1 * simd_width], gvl);
                        __epi_2xf32 d13 = __builtin_epi_vload_2xf32(&data[5 * data_stride + 1 * simd_width], gvl);
                        __epi_2xf32 d14 = __builtin_epi_vload_2xf32(&data[6 * data_stride + 1 * simd_width], gvl);
                        __epi_2xf32 d15 = __builtin_epi_vload_2xf32(&data[7 * data_stride + 1 * simd_width], gvl);
                        __epi_2xf32 wd4, wd5, wd6, wd7, wd12, wd13, wd14, wd15;
                        wd00 = __builtin_epi_vfsub_2xf32(d8, d14, gvl);
                        const __epi_2xf32 d12_sub_d10 = __builtin_epi_vfsub_2xf32(d12, d10, gvl);

                        // Compute wd7 := d7 - d1
                        wd07 = __builtin_epi_vfsub_2xf32(d15, d9, gvl);
                        const __epi_2xf32 d11_sub_d13 = __builtin_epi_vfsub_2xf32(d11, d13, gvl);
                        // float32x4_t wd1 := d2 + d6
                        wd01 = __builtin_epi_vfadd_2xf32(d10, d14, gvl);
                        // Compute wd2 := d1 + d5
                        wd02 = __builtin_epi_vfadd_2xf32(d9, d13, gvl);
                        // Compute wd4 := d5 + 0.25 * d1
                        wd04 = __builtin_epi_vfmacc_2xf32(d13, vconst_0_25, d9, gvl);
                        // Compute wd5 := d6 - 5.0 * d4
                        wd05 = __builtin_epi_vfsub_2xf32(d14, __builtin_epi_vfmul_2xf32(vconst_0_5, d12, gvl), gvl);
                        // Compute wd3 := d6 + 0.25 * d2
                        wd03 = __builtin_epi_vfmacc_2xf32(d14, vconst_0_25, d10, gvl);
                        // Compute wd6 := d1 + 0.25 * d5
                        wd06 = __builtin_epi_vfmacc_2xf32(d9, vconst_0_25, d13, gvl);
                        // const svfloat32_t const_5_25__4_25 = svld1rq(pg, datatmp1);
                        // const float const_5_25 = 5.25f;
                        // const float const_4_25 = 4.25f;
                        //__epi_2xf32 vconst_5_25 = __builtin_epi_vbroadcast_2xf32(const_5_25, gvl);
                        //__epi_2xf32 vconst_4_25 = __builtin_epi_vbroadcast_2xf32(const_4_25, gvl);
                        // Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
                        wd00 = __builtin_epi_vfmacc_2xf32(wd00, d12_sub_d10, vconst_5_25, gvl);
                        // Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
                        wd07 = __builtin_epi_vfmacc_2xf32(wd07, d11_sub_d13, vconst_5_25, gvl);

                        // Compute
                        //   wd1 := (d6 + d2) - 4.25 * d4
                        //   wd2 := (d1 + d5) - 4.25 * d3
                        wd01 = __builtin_epi_vfsub_2xf32(wd01, __builtin_epi_vfmul_2xf32(vconst_4_25, d12, gvl), gvl);
                        wd02 = __builtin_epi_vfsub_2xf32(wd02, __builtin_epi_vfmul_2xf32(vconst_4_25, d11, gvl), gvl);
                        // const svfloat32_t const_1_25__4_00 = svld1(pg, datatmp2);
                        // const float const_1_25 = 1.25f;
                        // const float const_4_00 =4.00f;
                        //__epi_2xf32 vconst_1_25 = __builtin_epi_vbroadcast_2xf32(const_1_25, gvl);
                        //__epi_2xf32 vconst_4_00 = __builtin_epi_vbroadcast_2xf32(const_4_00, gvl);
                        //  Compute
                        //    wd3 := (d6 + 0.25 * d2) - 1.25 * d4
                        //    wd4 := (d5 + 0.25 * d1) - 1.25 * d3
                        //    wd6 := (d1 + 0.25 * d5) - 1.25 * d3
                        //    wd5 := (d6 - 5.0 * d4) + 4.0 * d2
                        wd03 = __builtin_epi_vfsub_2xf32(wd03, __builtin_epi_vfmul_2xf32(vconst_1_25, d12, gvl), gvl);
                        wd05 = __builtin_epi_vfmacc_2xf32(wd05, vconst_4_00, d10, gvl);
                        wd04 = __builtin_epi_vfsub_2xf32(wd04, __builtin_epi_vfmul_2xf32(vconst_1_25, d11, gvl), gvl);
                        wd06 = __builtin_epi_vfsub_2xf32(wd06, __builtin_epi_vfmul_2xf32(vconst_1_25, d11, gvl), gvl);

                        //   const float const_2 = 2.0f;
                        // __epi_2xf32 vconst_2 = __builtin_epi_vbroadcast_2xf32(const_2, gvl);

                        wd4 = wd00;
                        wd5 = __builtin_epi_vfadd_2xf32(wd01, wd02, gvl);
                        wd6 = __builtin_epi_vfsub_2xf32(wd01, wd02, gvl);
                        wd7 = __builtin_epi_vfmacc_2xf32(wd03, vconst_2, wd04, gvl);
                        wd12 = __builtin_epi_vfsub_2xf32(wd03, __builtin_epi_vfmul_2xf32(vconst_2, wd04, gvl), gvl);
                        wd13 = __builtin_epi_vfmacc_2xf32(wd05, vconst_2, wd06, gvl);
                        wd14 = __builtin_epi_vfsub_2xf32(wd05, __builtin_epi_vfmul_2xf32(vconst_2, wd06, gvl), gvl);
                        wd15 = wd07;

                        __builtin_epi_vstore_2xf32(&vin4567[0], wd4, gvl);
                        __builtin_epi_vstore_2xf32(&vin4567[simd_width], wd5, gvl);
                        __builtin_epi_vstore_2xf32(&vin4567[2 * simd_width], wd6, gvl);
                        __builtin_epi_vstore_2xf32(&vin4567[3 * simd_width], wd7, gvl);
                        __builtin_epi_vstore_2xf32(&vin45671[0], wd12, gvl);
                        __builtin_epi_vstore_2xf32(&vin45671[simd_width], wd13, gvl);
                        __builtin_epi_vstore_2xf32(&vin45671[2 * simd_width], wd14, gvl);
                        __builtin_epi_vstore_2xf32(&vin45671[3 * simd_width], wd15, gvl);

                        // for(int i=0;i<16;i++)
                        //      printf("vin0123 = %f\n", vin0123[i]);
                }
                else
                {
                        float block[8][simd_width * 2];
                        {
                                float zero = 0.0f;
                                const __epi_2xf32 vzero = __builtin_epi_vbroadcast_2xf32(zero, gvl);
                                for (float *block_ptr = &block[0][0], *block_end = &block[8][0]; block_ptr != block_end; block_ptr += simd_width)
                                {
                                        __builtin_epi_vstore_2xf32(block_ptr, vzero, gvl);
                                }
                        }
                        for (size_t i = 0; i < row_count; i++)
                        {
                                for (size_t j = 0; j < column_count; j++)
                                {
                                        block[row_offset + i][column_offset + j] = data[i * data_stride + j];
                                }
                        }
                        for (size_t col = 0; col < 1; col++)
                        {
                                __epi_2xf32 d0 = __builtin_epi_vload_2xf32(&block[0][0 * simd_width], gvl);
                                __epi_2xf32 d1 = __builtin_epi_vload_2xf32(&block[1][0 * simd_width], gvl);
                                __epi_2xf32 d2 = __builtin_epi_vload_2xf32(&block[2][0 * simd_width], gvl);
                                __epi_2xf32 d3 = __builtin_epi_vload_2xf32(&block[3][0 * simd_width], gvl);
                                __epi_2xf32 d4 = __builtin_epi_vload_2xf32(&block[4][0 * simd_width], gvl);
                                __epi_2xf32 d5 = __builtin_epi_vload_2xf32(&block[5][0 * simd_width], gvl);
                                __epi_2xf32 d6 = __builtin_epi_vload_2xf32(&block[6][0 * simd_width], gvl);
                                __epi_2xf32 d7 = __builtin_epi_vload_2xf32(&block[7][0 * simd_width], gvl);
                                __epi_2xf32 wd00, wd01, wd02, wd03, wd08, wd09, wd010, wd011;
                                float const_0_25 = 0.25f;
                                float const_0_5 = 5.00f;
                                __epi_2xf32 vconst_0_25 = __builtin_epi_vbroadcast_2xf32(const_0_25, gvl);
                                __epi_2xf32 vconst_0_5 = __builtin_epi_vbroadcast_2xf32(const_0_5, gvl);
                                // const float32x4_t const_0_25 = vmovq_n_f32(0.25f);

                                // Compute wd0 := d0 - d6
                                __epi_2xf32 wd0 = __builtin_epi_vfsub_2xf32(d0, d6, gvl);
                                const __epi_2xf32 d4_sub_d2 = __builtin_epi_vfsub_2xf32(d4, d2, gvl);

                                // Compute wd7 := d7 - d1
                                __epi_2xf32 wd7 = __builtin_epi_vfsub_2xf32(d7, d1, gvl);
                                const __epi_2xf32 d3_sub_d5 = __builtin_epi_vfsub_2xf32(d3, d5, gvl);
                                // float32x4_t wd1 := d2 + d6
                                __epi_2xf32 wd1 = __builtin_epi_vfadd_2xf32(d2, d6, gvl);
                                // Compute wd2 := d1 + d5
                                __epi_2xf32 wd2 = __builtin_epi_vfadd_2xf32(d1, d5, gvl);
                                // Compute wd4 := d5 + 0.25 * d1
                                __epi_2xf32 wd4 = __builtin_epi_vfmacc_2xf32(d5, vconst_0_25, d1, gvl);
                                // Compute wd5 := d6 - 5.0 * d4
                                __epi_2xf32 wd5 = __builtin_epi_vfsub_2xf32(d6, __builtin_epi_vfmul_2xf32(vconst_0_5, d4, gvl), gvl);
                                // Compute wd3 := d6 + 0.25 * d2
                                __epi_2xf32 wd3 = __builtin_epi_vfmacc_2xf32(d6, vconst_0_25, d2, gvl);
                                // Compute wd6 := d1 + 0.25 * d5
                                __epi_2xf32 wd6 = __builtin_epi_vfmacc_2xf32(d1, vconst_0_25, d5, gvl);
                                // const svfloat32_t const_5_25__4_25 = svld1rq(pg, datatmp1);
                                const float const_5_25 = 5.25f;
                                const float const_4_25 = 4.25f;
                                __epi_2xf32 vconst_5_25 = __builtin_epi_vbroadcast_2xf32(const_5_25, gvl);
                                __epi_2xf32 vconst_4_25 = __builtin_epi_vbroadcast_2xf32(const_4_25, gvl);
                                // Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
                                wd0 = __builtin_epi_vfmacc_2xf32(wd0, d4_sub_d2, vconst_5_25, gvl);
                                // Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
                                wd7 = __builtin_epi_vfmacc_2xf32(wd7, d3_sub_d5, vconst_5_25, gvl);

                                // Compute
                                //   wd1 := (d6 + d2) - 4.25 * d4
                                //   wd2 := (d1 + d5) - 4.25 * d3
                                wd1 = __builtin_epi_vfsub_2xf32(wd1, __builtin_epi_vfmul_2xf32(vconst_4_25, d4, gvl), gvl);
                                wd2 = __builtin_epi_vfsub_2xf32(wd2, __builtin_epi_vfmul_2xf32(vconst_4_25, d3, gvl), gvl);
                                // const svfloat32_t const_1_25__4_00 = svld1(pg, datatmp2);
                                const float const_1_25 = 1.25f;
                                const float const_4_00 = 4.00f;
                                __epi_2xf32 vconst_1_25 = __builtin_epi_vbroadcast_2xf32(const_1_25, gvl);
                                __epi_2xf32 vconst_4_00 = __builtin_epi_vbroadcast_2xf32(const_4_00, gvl);
                                // Compute
                                //   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
                                //   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
                                //   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
                                //   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
                                wd3 = __builtin_epi_vfsub_2xf32(wd3, __builtin_epi_vfmul_2xf32(vconst_1_25, d4, gvl), gvl);
                                wd5 = __builtin_epi_vfmacc_2xf32(wd5, vconst_4_00, d2, gvl);
                                wd4 = __builtin_epi_vfsub_2xf32(wd4, __builtin_epi_vfmul_2xf32(vconst_1_25, d3, gvl), gvl);
                                wd6 = __builtin_epi_vfsub_2xf32(wd6, __builtin_epi_vfmul_2xf32(vconst_1_25, d3, gvl), gvl);

                                const float const_2 = 2.0f;
                                __epi_2xf32 vconst_2 = __builtin_epi_vbroadcast_2xf32(const_2, gvl);

                                wd00 = wd0;
                                wd01 = __builtin_epi_vfadd_2xf32(wd1, wd2, gvl);
                                wd02 = __builtin_epi_vfsub_2xf32(wd1, wd2, gvl);
                                wd03 = __builtin_epi_vfmacc_2xf32(wd3, vconst_2, wd4, gvl);
                                wd08 = __builtin_epi_vfsub_2xf32(wd3, __builtin_epi_vfmul_2xf32(vconst_2, wd4, gvl), gvl);
                                wd09 = __builtin_epi_vfmacc_2xf32(wd5, vconst_2, wd6, gvl);
                                wd010 = __builtin_epi_vfsub_2xf32(wd5, __builtin_epi_vfmul_2xf32(vconst_2, wd6, gvl), gvl);
                                wd011 = wd7;
                                // float tmp[4];
                                //__builtin_epi_vstore_2xf32(&tmp[0], wd01, gvl);
                                //                for(int i=0;i<4;i++)
                                //                      printf("wd1 tmp value %f\n", tmp[i]);
                                __builtin_epi_vstore_2xf32(&vin0123[0], wd00, gvl);
                                __builtin_epi_vstore_2xf32(&vin0123[simd_width], wd01, gvl);
                                __builtin_epi_vstore_2xf32(&vin0123[2 * simd_width], wd02, gvl);
                                __builtin_epi_vstore_2xf32(&vin0123[3 * simd_width], wd03, gvl);
                                __builtin_epi_vstore_2xf32(&vin01231[0], wd08, gvl);
                                __builtin_epi_vstore_2xf32(&vin01231[simd_width], wd09, gvl);
                                __builtin_epi_vstore_2xf32(&vin01231[2 * simd_width], wd010, gvl);
                                __builtin_epi_vstore_2xf32(&vin01231[3 * simd_width], wd011, gvl);
                        }
                        {
                                __epi_2xf32 d0 = __builtin_epi_vload_2xf32(&block[0][1 * simd_width], gvl);
                                __epi_2xf32 d1 = __builtin_epi_vload_2xf32(&block[1][1 * simd_width], gvl);
                                __epi_2xf32 d2 = __builtin_epi_vload_2xf32(&block[2][1 * simd_width], gvl);
                                __epi_2xf32 d3 = __builtin_epi_vload_2xf32(&block[3][1 * simd_width], gvl);
                                __epi_2xf32 d4 = __builtin_epi_vload_2xf32(&block[4][1 * simd_width], gvl);
                                __epi_2xf32 d5 = __builtin_epi_vload_2xf32(&block[5][1 * simd_width], gvl);
                                __epi_2xf32 d6 = __builtin_epi_vload_2xf32(&block[6][1 * simd_width], gvl);
                                __epi_2xf32 d7 = __builtin_epi_vload_2xf32(&block[7][1 * simd_width], gvl);
                                __epi_2xf32 wd04, wd05, wd06, wd07, wd012, wd013, wd014, wd015;

                                float const_0_25 = 0.25f;
                                float const_0_5 = 5.00f;
                                __epi_2xf32 vconst_0_25 = __builtin_epi_vbroadcast_2xf32(const_0_25, gvl);
                                __epi_2xf32 vconst_0_5 = __builtin_epi_vbroadcast_2xf32(const_0_5, gvl);
                                // const float32x4_t const_0_25 = vmovq_n_f32(0.25f);

                                // Compute wd0 := d0 - d6
                                __epi_2xf32 wd0 = __builtin_epi_vfsub_2xf32(d0, d6, gvl);
                                const __epi_2xf32 d4_sub_d2 = __builtin_epi_vfsub_2xf32(d4, d2, gvl);

                                // Compute wd7 := d7 - d1
                                __epi_2xf32 wd7 = __builtin_epi_vfsub_2xf32(d7, d1, gvl);
                                const __epi_2xf32 d3_sub_d5 = __builtin_epi_vfsub_2xf32(d3, d5, gvl);
                                // float32x4_t wd1 := d2 + d6
                                __epi_2xf32 wd1 = __builtin_epi_vfadd_2xf32(d2, d6, gvl);
                                // Compute wd2 := d1 + d5
                                __epi_2xf32 wd2 = __builtin_epi_vfadd_2xf32(d1, d5, gvl);
                                // Compute wd4 := d5 + 0.25 * d1
                                __epi_2xf32 wd4 = __builtin_epi_vfmacc_2xf32(d5, vconst_0_25, d1, gvl);
                                // Compute wd5 := d6 - 5.0 * d4
                                __epi_2xf32 wd5 = __builtin_epi_vfsub_2xf32(d6, __builtin_epi_vfmul_2xf32(vconst_0_5, d4, gvl), gvl);
                                // Compute wd3 := d6 + 0.25 * d2
                                __epi_2xf32 wd3 = __builtin_epi_vfmacc_2xf32(d6, vconst_0_25, d2, gvl);
                                // Compute wd6 := d1 + 0.25 * d5
                                __epi_2xf32 wd6 = __builtin_epi_vfmacc_2xf32(d1, vconst_0_25, d5, gvl);
                                // const svfloat32_t const_5_25__4_25 = svld1rq(pg, datatmp1);
                                const float const_5_25 = 5.25f;
                                const float const_4_25 = 4.25f;
                                __epi_2xf32 vconst_5_25 = __builtin_epi_vbroadcast_2xf32(const_5_25, gvl);
                                __epi_2xf32 vconst_4_25 = __builtin_epi_vbroadcast_2xf32(const_4_25, gvl);
                                // Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
                                wd0 = __builtin_epi_vfmacc_2xf32(wd0, d4_sub_d2, vconst_5_25, gvl);
                                // Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
                                wd7 = __builtin_epi_vfmacc_2xf32(wd7, d3_sub_d5, vconst_5_25, gvl);

                                // Compute
                                //   wd1 := (d6 + d2) - 4.25 * d4
                                //   wd2 := (d1 + d5) - 4.25 * d3
                                wd1 = __builtin_epi_vfsub_2xf32(wd1, __builtin_epi_vfmul_2xf32(vconst_4_25, d4, gvl), gvl);
                                wd2 = __builtin_epi_vfsub_2xf32(wd2, __builtin_epi_vfmul_2xf32(vconst_4_25, d3, gvl), gvl);
                                // const svfloat32_t const_1_25__4_00 = svld1(pg, datatmp2);
                                const float const_1_25 = 1.25f;
                                const float const_4_00 = 4.00f;
                                __epi_2xf32 vconst_1_25 = __builtin_epi_vbroadcast_2xf32(const_1_25, gvl);
                                __epi_2xf32 vconst_4_00 = __builtin_epi_vbroadcast_2xf32(const_4_00, gvl);
                                // Compute
                                //   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
                                //   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
                                //   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
                                //   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
                                wd3 = __builtin_epi_vfsub_2xf32(wd3, __builtin_epi_vfmul_2xf32(vconst_1_25, d4, gvl), gvl);
                                wd5 = __builtin_epi_vfmacc_2xf32(wd5, vconst_4_00, d2, gvl);
                                wd4 = __builtin_epi_vfsub_2xf32(wd4, __builtin_epi_vfmul_2xf32(vconst_1_25, d3, gvl), gvl);
                                wd6 = __builtin_epi_vfsub_2xf32(wd6, __builtin_epi_vfmul_2xf32(vconst_1_25, d3, gvl), gvl);

                                const float const_2 = 2.0f;
                                __epi_2xf32 vconst_2 = __builtin_epi_vbroadcast_2xf32(const_2, gvl);

                                wd04 = wd0;
                                wd05 = __builtin_epi_vfadd_2xf32(wd1, wd2, gvl);
                                wd06 = __builtin_epi_vfsub_2xf32(wd1, wd2, gvl);
                                wd07 = __builtin_epi_vfmacc_2xf32(wd3, vconst_2, wd4, gvl);
                                wd012 = __builtin_epi_vfsub_2xf32(wd3, __builtin_epi_vfmul_2xf32(vconst_2, wd4, gvl), gvl);
                                wd013 = __builtin_epi_vfmacc_2xf32(wd5, vconst_2, wd6, gvl);
                                wd014 = __builtin_epi_vfsub_2xf32(wd5, __builtin_epi_vfmul_2xf32(vconst_2, wd6, gvl), gvl);
                                wd015 = wd7;
                                __builtin_epi_vstore_2xf32(&vin4567[0], wd04, gvl);
                                __builtin_epi_vstore_2xf32(&vin4567[simd_width], wd05, gvl);
                                __builtin_epi_vstore_2xf32(&vin4567[2 * simd_width], wd06, gvl);
                                __builtin_epi_vstore_2xf32(&vin4567[3 * simd_width], wd07, gvl);
                                __builtin_epi_vstore_2xf32(&vin45671[0], wd012, gvl);
                                __builtin_epi_vstore_2xf32(&vin45671[simd_width], wd013, gvl);
                                __builtin_epi_vstore_2xf32(&vin45671[2 * simd_width], wd014, gvl);
                                __builtin_epi_vstore_2xf32(&vin45671[3 * simd_width], wd015, gvl);
                                // for(int i=0;i<16;i++)
                                //      printf("vin0123 = %f\n", vin0123[i]);
                        }
                }
                for (size_t col = 0; col < 1; col++)
                {
                        int icol = 0;
                        int ind1 = 0;
                        int index1_host[simd_width];
                        for (int j = 0; j < 4; j++)
                        {
                                index1_host[ind1] = ((j * simd_width));
                                ind1++;
                        }
                        int one = 1;
                        const __epi_2xi32 vone = __builtin_epi_vbroadcast_2xi32(one, gvl);
                        __epi_2xi32 index1, index2, index3, index4;
                        index1 = __builtin_epi_vload_2xi32(&index1_host[0], gvl);
                        index2 = __builtin_epi_vadd_2xi32(index1, vone, gvl);
                        index3 = __builtin_epi_vadd_2xi32(index2, vone, gvl);
                        index4 = __builtin_epi_vadd_2xi32(index3, vone, gvl);
                        __epi_2xi32 index11 = __builtin_epi_vmul_2xi32(index1, FOUR, gvl);
                        __epi_2xi32 index12 = __builtin_epi_vmul_2xi32(index2, FOUR, gvl);
                        __epi_2xi32 index13 = __builtin_epi_vmul_2xi32(index3, FOUR, gvl);
                        __epi_2xi32 index14 = __builtin_epi_vmul_2xi32(index4, FOUR, gvl);
                        // float tmp[4];
                        // int index_tmp[4];
                        // __epi_2xf32 vtmp = __builtin_epi_vload_indexed_2xf32(  &vin0123[0],  index11, gvl);
                        // __builtin_epi_vstore_2xf32(&tmp[0], vtmp, gvl);
                        // __builtin_epi_vstore_2xi32(&index_tmp[0], index2, gvl);
                        //   for(int i=0;i<4;i++)
                        //         printf("indexed load single = %d %f %f \n", index_tmp[i], tmp[i], vin0123[index_tmp[i]]);

                        {
                                __epi_2xf32 d0 = __builtin_epi_vload_indexed_2xf32(&vin0123[0], index11, gvl);
                                __epi_2xf32 d1 = __builtin_epi_vload_indexed_2xf32(&vin0123[0], index12, gvl);
                                __epi_2xf32 d2 = __builtin_epi_vload_indexed_2xf32(&vin0123[0], index13, gvl);
                                __epi_2xf32 d3 = __builtin_epi_vload_indexed_2xf32(&vin0123[0], index14, gvl);
                                __epi_2xf32 d4 = __builtin_epi_vload_indexed_2xf32(&vin4567[0], index11, gvl);
                                __epi_2xf32 d5 = __builtin_epi_vload_indexed_2xf32(&vin4567[0], index12, gvl);
                                __epi_2xf32 d6 = __builtin_epi_vload_indexed_2xf32(&vin4567[0], index13, gvl);
                                __epi_2xf32 d7 = __builtin_epi_vload_indexed_2xf32(&vin4567[0], index14, gvl);
                                __epi_2xf32 vout0, vout1, vout2, vout3, vout4, vout5, vout6, vout7;

                                float const_0_25 = 0.25f;
                                float const_0_5 = 5.00f;
                                __epi_2xf32 vconst_0_25 = __builtin_epi_vbroadcast_2xf32(const_0_25, gvl);
                                __epi_2xf32 vconst_0_5 = __builtin_epi_vbroadcast_2xf32(const_0_5, gvl);
                                // const float32x4_t const_0_25 = vmovq_n_f32(0.25f);

                                // Compute wd0 := d0 - d6
                                __epi_2xf32 wd0 = __builtin_epi_vfsub_2xf32(d0, d6, gvl);
                                const __epi_2xf32 d4_sub_d2 = __builtin_epi_vfsub_2xf32(d4, d2, gvl);

                                // Compute wd7 := d7 - d1
                                __epi_2xf32 wd7 = __builtin_epi_vfsub_2xf32(d7, d1, gvl);
                                const __epi_2xf32 d3_sub_d5 = __builtin_epi_vfsub_2xf32(d3, d5, gvl);
                                // float32x4_t wd1 := d2 + d6
                                __epi_2xf32 wd1 = __builtin_epi_vfadd_2xf32(d2, d6, gvl);
                                // Compute wd2 := d1 + d5
                                __epi_2xf32 wd2 = __builtin_epi_vfadd_2xf32(d1, d5, gvl);
                                // Compute wd4 := d5 + 0.25 * d1
                                __epi_2xf32 wd4 = __builtin_epi_vfmacc_2xf32(d5, vconst_0_25, d1, gvl);
                                // Compute wd5 := d6 - 5.0 * d4
                                __epi_2xf32 wd5 = __builtin_epi_vfsub_2xf32(d6, __builtin_epi_vfmul_2xf32(vconst_0_5, d4, gvl), gvl);
                                // Compute wd3 := d6 + 0.25 * d2
                                __epi_2xf32 wd3 = __builtin_epi_vfmacc_2xf32(d6, vconst_0_25, d2, gvl);
                                // Compute wd6 := d1 + 0.25 * d5
                                __epi_2xf32 wd6 = __builtin_epi_vfmacc_2xf32(d1, vconst_0_25, d5, gvl);
                                // const svfloat32_t const_5_25__4_25 = svld1rq(pg, datatmp1);
                                const float const_5_25 = 5.25f;
                                const float const_4_25 = 4.25f;
                                __epi_2xf32 vconst_5_25 = __builtin_epi_vbroadcast_2xf32(const_5_25, gvl);
                                __epi_2xf32 vconst_4_25 = __builtin_epi_vbroadcast_2xf32(const_4_25, gvl);

                                // Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
                                wd0 = __builtin_epi_vfmacc_2xf32(wd0, d4_sub_d2, vconst_5_25, gvl);
                                // Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
                                wd7 = __builtin_epi_vfmacc_2xf32(wd7, d3_sub_d5, vconst_5_25, gvl);
                                // Compute
                                //   wd1 := (d6 + d2) - 4.25 * d4
                                //   wd2 := (d1 + d5) - 4.25 * d3
                                wd1 = __builtin_epi_vfsub_2xf32(wd1, __builtin_epi_vfmul_2xf32(vconst_4_25, d4, gvl), gvl);
                                wd2 = __builtin_epi_vfsub_2xf32(wd2, __builtin_epi_vfmul_2xf32(vconst_4_25, d3, gvl), gvl);
                                // const svfloat32_t const_1_25__4_00 = svld1(pg, datatmp2);
                                const float const_1_25 = 1.25f;
                                const float const_4_00 = 4.00f;
                                __epi_2xf32 vconst_1_25 = __builtin_epi_vbroadcast_2xf32(const_1_25, gvl);
                                __epi_2xf32 vconst_4_00 = __builtin_epi_vbroadcast_2xf32(const_4_00, gvl);
                                // Compute
                                //   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
                                //   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
                                //   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
                                //   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
                                wd3 = __builtin_epi_vfsub_2xf32(wd3, __builtin_epi_vfmul_2xf32(vconst_1_25, d4, gvl), gvl);
                                wd5 = __builtin_epi_vfmacc_2xf32(wd5, vconst_4_00, d2, gvl);
                                wd4 = __builtin_epi_vfsub_2xf32(wd4, __builtin_epi_vfmul_2xf32(vconst_1_25, d3, gvl), gvl);
                                wd6 = __builtin_epi_vfsub_2xf32(wd6, __builtin_epi_vfmul_2xf32(vconst_1_25, d3, gvl), gvl);

                                const float const_2 = 2.0f;
                                __epi_2xf32 vconst_2 = __builtin_epi_vbroadcast_2xf32(const_2, gvl);

                                vout0 = wd0;
                                vout1 = __builtin_epi_vfadd_2xf32(wd1, wd2, gvl);
                                vout2 = __builtin_epi_vfsub_2xf32(wd1, wd2, gvl);
                                vout3 = __builtin_epi_vfmacc_2xf32(wd3, vconst_2, wd4, gvl);
                                vout4 = __builtin_epi_vfsub_2xf32(wd3, __builtin_epi_vfmul_2xf32(vconst_2, wd4, gvl), gvl);
                                vout5 = __builtin_epi_vfmacc_2xf32(wd5, vconst_2, wd6, gvl);
                                vout6 = __builtin_epi_vfsub_2xf32(wd5, __builtin_epi_vfmul_2xf32(vconst_2, wd6, gvl), gvl);
                                vout7 = wd7;

                                // float tmp[4];
                                // __builtin_epi_vstore_2xf32(&tmp[0], vout2,gvl);
                                // for(int i=0;i<4;i++)
                                //      printf("vout 2 4 simd single = %f\n", tmp[i]);
                                __builtin_epi_vstore_2xf32(&transform[0], vout0, gvl);
                                transform += transform_stride;
                                __builtin_epi_vstore_2xf32(&transform[0], vout1, gvl);
                                transform += transform_stride;
                                __builtin_epi_vstore_2xf32(&transform[0], vout2, gvl);
                                transform += transform_stride;
                                __builtin_epi_vstore_2xf32(&transform[0], vout3, gvl);
                                transform += transform_stride;
                                __builtin_epi_vstore_2xf32(transform, vout4, gvl);
                                transform += transform_stride;
                                __builtin_epi_vstore_2xf32(transform, vout5, gvl);
                                transform += transform_stride;
                                __builtin_epi_vstore_2xf32(transform, vout6, gvl);
                                transform += transform_stride;
                                __builtin_epi_vstore_2xf32(transform, vout7, gvl);
                                transform += transform_stride;
                        }
                        {
                                __epi_2xf32 vout10, vout11, vout12, vout13, vout14, vout15, vout16, vout17;
                                __epi_2xf32 d0 = __builtin_epi_vload_indexed_2xf32(&vin01231[0], index11, gvl);
                                __epi_2xf32 d1 = __builtin_epi_vload_indexed_2xf32(&vin01231[0], index12, gvl);
                                __epi_2xf32 d2 = __builtin_epi_vload_indexed_2xf32(&vin01231[0], index13, gvl);
                                __epi_2xf32 d3 = __builtin_epi_vload_indexed_2xf32(&vin01231[0], index14, gvl);
                                __epi_2xf32 d4 = __builtin_epi_vload_indexed_2xf32(&vin45671[0], index11, gvl);
                                __epi_2xf32 d5 = __builtin_epi_vload_indexed_2xf32(&vin45671[0], index12, gvl);
                                __epi_2xf32 d6 = __builtin_epi_vload_indexed_2xf32(&vin45671[0], index13, gvl);
                                __epi_2xf32 d7 = __builtin_epi_vload_indexed_2xf32(&vin45671[0], index14, gvl);

                                float const_0_25 = 0.25f;
                                float const_0_5 = 5.00f;
                                __epi_2xf32 vconst_0_25 = __builtin_epi_vbroadcast_2xf32(const_0_25, gvl);
                                __epi_2xf32 vconst_0_5 = __builtin_epi_vbroadcast_2xf32(const_0_5, gvl);
                                // const float32x4_t const_0_25 = vmovq_n_f32(0.25f);

                                // Compute wd0 := d0 - d6
                                __epi_2xf32 wd0 = __builtin_epi_vfsub_2xf32(d0, d6, gvl);
                                const __epi_2xf32 d4_sub_d2 = __builtin_epi_vfsub_2xf32(d4, d2, gvl);

                                // Compute wd7 := d7 - d1
                                __epi_2xf32 wd7 = __builtin_epi_vfsub_2xf32(d7, d1, gvl);
                                const __epi_2xf32 d3_sub_d5 = __builtin_epi_vfsub_2xf32(d3, d5, gvl);
                                // float32x4_t wd1 := d2 + d6
                                __epi_2xf32 wd1 = __builtin_epi_vfadd_2xf32(d2, d6, gvl);
                                // Compute wd2 := d1 + d5
                                __epi_2xf32 wd2 = __builtin_epi_vfadd_2xf32(d1, d5, gvl);
                                // Compute wd4 := d5 + 0.25 * d1
                                __epi_2xf32 wd4 = __builtin_epi_vfmacc_2xf32(d5, vconst_0_25, d1, gvl);
                                // Compute wd5 := d6 - 5.0 * d4
                                __epi_2xf32 wd5 = __builtin_epi_vfsub_2xf32(d6, __builtin_epi_vfmul_2xf32(vconst_0_5, d4, gvl), gvl);
                                // Compute wd3 := d6 + 0.25 * d2
                                __epi_2xf32 wd3 = __builtin_epi_vfmacc_2xf32(d6, vconst_0_25, d2, gvl);
                                // Compute wd6 := d1 + 0.25 * d5
                                __epi_2xf32 wd6 = __builtin_epi_vfmacc_2xf32(d1, vconst_0_25, d5, gvl);
                                // const svfloat32_t const_5_25__4_25 = svld1rq(pg, datatmp1);
                                const float const_5_25 = 5.25f;
                                const float const_4_25 = 4.25f;
                                __epi_2xf32 vconst_5_25 = __builtin_epi_vbroadcast_2xf32(const_5_25, gvl);
                                __epi_2xf32 vconst_4_25 = __builtin_epi_vbroadcast_2xf32(const_4_25, gvl);

                                // Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
                                wd0 = __builtin_epi_vfmacc_2xf32(wd0, d4_sub_d2, vconst_5_25, gvl);
                                // Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
                                wd7 = __builtin_epi_vfmacc_2xf32(wd7, d3_sub_d5, vconst_5_25, gvl);
                                // Compute
                                //   wd1 := (d6 + d2) - 4.25 * d4
                                //   wd2 := (d1 + d5) - 4.25 * d3
                                wd1 = __builtin_epi_vfsub_2xf32(wd1, __builtin_epi_vfmul_2xf32(vconst_4_25, d4, gvl), gvl);
                                wd2 = __builtin_epi_vfsub_2xf32(wd2, __builtin_epi_vfmul_2xf32(vconst_4_25, d3, gvl), gvl);
                                // const svfloat32_t const_1_25__4_00 = svld1(pg, datatmp2);
                                const float const_1_25 = 1.25f;
                                const float const_4_00 = 4.00f;
                                __epi_2xf32 vconst_1_25 = __builtin_epi_vbroadcast_2xf32(const_1_25, gvl);
                                __epi_2xf32 vconst_4_00 = __builtin_epi_vbroadcast_2xf32(const_4_00, gvl);
                                // Compute
                                //   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
                                //   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
                                //   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
                                //   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
                                wd3 = __builtin_epi_vfsub_2xf32(wd3, __builtin_epi_vfmul_2xf32(vconst_1_25, d4, gvl), gvl);
                                wd5 = __builtin_epi_vfmacc_2xf32(wd5, vconst_4_00, d2, gvl);
                                wd4 = __builtin_epi_vfsub_2xf32(wd4, __builtin_epi_vfmul_2xf32(vconst_1_25, d3, gvl), gvl);
                                wd6 = __builtin_epi_vfsub_2xf32(wd6, __builtin_epi_vfmul_2xf32(vconst_1_25, d3, gvl), gvl);

                                const float const_2 = 2.0f;
                                __epi_2xf32 vconst_2 = __builtin_epi_vbroadcast_2xf32(const_2, gvl);

                                vout10 = wd0;
                                vout11 = __builtin_epi_vfadd_2xf32(wd1, wd2, gvl);
                                vout12 = __builtin_epi_vfsub_2xf32(wd1, wd2, gvl);
                                vout13 = __builtin_epi_vfmacc_2xf32(wd3, vconst_2, wd4, gvl);
                                vout14 = __builtin_epi_vfsub_2xf32(wd3, __builtin_epi_vfmul_2xf32(vconst_2, wd4, gvl), gvl);
                                vout15 = __builtin_epi_vfmacc_2xf32(wd5, vconst_2, wd6, gvl);
                                vout16 = __builtin_epi_vfsub_2xf32(wd5, __builtin_epi_vfmul_2xf32(vconst_2, wd6, gvl), gvl);
                                vout17 = wd7;
                                // float tmp[4];
                                //__builtin_epi_vstore_2xf32(&tmp[0], vout14,gvl);
                                // for(int i=0;i<4;i++)
                                //       printf("vout 14 4 simd single= %f\n", tmp[i]);
                                __builtin_epi_vstore_2xf32(transform, vout10, gvl);
                                transform += transform_stride;
                                __builtin_epi_vstore_2xf32(transform, vout11, gvl);
                                transform += transform_stride;
                                __builtin_epi_vstore_2xf32(transform, vout12, gvl);
                                transform += transform_stride;
                                __builtin_epi_vstore_2xf32(transform, vout13, gvl);
                                transform += transform_stride;
                                __builtin_epi_vstore_2xf32(transform, vout14, gvl);
                                transform += transform_stride;
                                __builtin_epi_vstore_2xf32(transform, vout15, gvl);
                                transform += transform_stride;
                                __builtin_epi_vstore_2xf32(transform, vout16, gvl);
                                transform += transform_stride;
                                __builtin_epi_vstore_2xf32(transform, vout17, gvl);
                                transform += transform_stride;
                        }
                }
                i1 += gvl;
        }
}

////////////////////////////////////////

////////////////////////////////////////////////////
void nnp_iwt8x8_3x3_with_offset__neon_intertile1(
    const float *data[restrict static 1],
    void **transform,
    size_t data_stride,
    size_t transform_stride,
    uint32_t row_count,
    uint32_t column_count,
    uint32_t row_offset,
    uint32_t column_offset, size_t interchannels)
{
        // printf("I am in intertile neo function\n");
        int simd_width = interchannels * 4; // nnp_hwinfo.sve_simd_width;//nnp_hwinfo.simd_width;
        // printf("sve simd_width = %d", nnp_hwinfo.sve_simd_width);
        float vin0123[4 * simd_width], vin4567[4 * simd_width], vin01231[4 * simd_width], vin45671[4 * simd_width];
        // const int interchannels  = nnp_hwinfo.globalinterchannels;
        float *new_data, *new_data1;
        new_data = (float *)malloc(sizeof(float) * 8 * 8 * interchannels);
        new_data1 = (float *)malloc(sizeof(float) * 8 * 8 * interchannels);
        if (new_data == NULL)
        {
                fprintf(stderr, "error in allocating memory\n");
                exit(-1);
        }
        if (new_data1 == NULL)
        {
                fprintf(stderr, "error in allocating memory\n");
                exit(-1);
        }
        if (row_count == 8 && column_count == 8 && row_offset == 0 && column_offset == 0)
        {
                int tmp_width = 4;
#pragma loop unroll_count(interchannels)
                for (int k = 0; k < interchannels; k++)
                {
                        for (int i = 0; i < 8; i++)
                        {
                                for (int j = 0; j < 4; j++)
                                {
                                        new_data[(i * simd_width) + ((0 * tmp_width) + (j + k * 4))] = data[k][(i * data_stride) + ((0 * tmp_width) + j)];
                                        new_data1[(i * simd_width) + ((0 * tmp_width) + (j + k * 4))] = data[k][(i * data_stride) + ((1 * tmp_width) + j)];
                                }
                        }
                }
        }
        int tmp_stride = simd_width; // for tmp new_data and new_data1

        for (int i1 = 0; i1 < simd_width;)
        {
                unsigned long gvl = __builtin_epi_vsetvl(((long)simd_width - (long)i1), __epi_e32, __epi_m1);
                int four = 4;
                const __epi_2xi32 FOUR = __builtin_epi_vbroadcast_2xi32(four, gvl);

                if (row_count == 8 && column_count == 8 && row_offset == 0 && column_offset == 0)
                {
                        // Fast path where we can directly load `data` into `wd`.
                        __epi_2xf32 wd0, wd1, wd2, wd3, wd4, wd5, wd6, wd7, wd8, wd9, wd10, wd11, wd12, wd13, wd14, wd15;

                        __epi_2xf32 d0 = __builtin_epi_vload_2xf32(&new_data[0 * tmp_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d1 = __builtin_epi_vload_2xf32(&new_data[1 * tmp_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d2 = __builtin_epi_vload_2xf32(&new_data[2 * tmp_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d3 = __builtin_epi_vload_2xf32(&new_data[3 * tmp_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d4 = __builtin_epi_vload_2xf32(&new_data[4 * tmp_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d5 = __builtin_epi_vload_2xf32(&new_data[5 * tmp_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d6 = __builtin_epi_vload_2xf32(&new_data[6 * tmp_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d7 = __builtin_epi_vload_2xf32(&new_data[7 * tmp_stride + 0 * simd_width], gvl);
                        float const_0_25 = 0.25f;
                        float const_0_5 = 5.00f;
                        __epi_2xf32 vconst_0_25 = __builtin_epi_vbroadcast_2xf32(const_0_25, gvl);
                        __epi_2xf32 vconst_0_5 = __builtin_epi_vbroadcast_2xf32(const_0_5, gvl);
                        // const float32x4_t const_0_25 = vmovq_n_f32(0.25f);

                        // Compute wd0 := d0 - d6
                        __epi_2xf32 wd00 = __builtin_epi_vfsub_2xf32(d0, d6, gvl);
                        const __epi_2xf32 d4_sub_d2 = __builtin_epi_vfsub_2xf32(d4, d2, gvl);

                        // Compute wd7 := d7 - d1
                        __epi_2xf32 wd07 = __builtin_epi_vfsub_2xf32(d7, d1, gvl);
                        const __epi_2xf32 d3_sub_d5 = __builtin_epi_vfsub_2xf32(d3, d5, gvl);
                        // float32x4_t wd1 := d2 + d6
                        __epi_2xf32 wd01 = __builtin_epi_vfadd_2xf32(d2, d6, gvl);
                        // Compute wd2 := d1 + d5
                        __epi_2xf32 wd02 = __builtin_epi_vfadd_2xf32(d1, d5, gvl);
                        // Compute wd4 := d5 + 0.25 * d1
                        __epi_2xf32 wd04 = __builtin_epi_vfmacc_2xf32(d5, vconst_0_25, d1, gvl);
                        // Compute wd5 := d6 - 5.0 * d4
                        __epi_2xf32 wd05 = __builtin_epi_vfsub_2xf32(d6, __builtin_epi_vfmul_2xf32(vconst_0_5, d4, gvl), gvl);
                        // Compute wd3 := d6 + 0.25 * d2
                        __epi_2xf32 wd03 = __builtin_epi_vfmacc_2xf32(d6, vconst_0_25, d2, gvl);
                        // Compute wd6 := d1 + 0.25 * d5
                        __epi_2xf32 wd06 = __builtin_epi_vfmacc_2xf32(d1, vconst_0_25, d5, gvl);
                        // const svfloat32_t const_5_25__4_25 = svld1rq(pg, datatmp1);
                        const float const_5_25 = 5.25f;
                        const float const_4_25 = 4.25f;
                        __epi_2xf32 vconst_5_25 = __builtin_epi_vbroadcast_2xf32(const_5_25, gvl);
                        __epi_2xf32 vconst_4_25 = __builtin_epi_vbroadcast_2xf32(const_4_25, gvl);
                        // Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
                        wd00 = __builtin_epi_vfmacc_2xf32(wd00, d4_sub_d2, vconst_5_25, gvl);
                        // Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
                        wd07 = __builtin_epi_vfmacc_2xf32(wd07, d3_sub_d5, vconst_5_25, gvl);

                        // Compute
                        //   wd1 := (d6 + d2) - 4.25 * d4
                        //   wd2 := (d1 + d5) - 4.25 * d3
                        wd01 = __builtin_epi_vfsub_2xf32(wd01, __builtin_epi_vfmul_2xf32(vconst_4_25, d4, gvl), gvl);
                        wd02 = __builtin_epi_vfsub_2xf32(wd02, __builtin_epi_vfmul_2xf32(vconst_4_25, d3, gvl), gvl);
                        // const svfloat32_t const_1_25__4_00 = svld1(pg, datatmp2);
                        const float const_1_25 = 1.25f;
                        const float const_4_00 = 4.00f;
                        __epi_2xf32 vconst_1_25 = __builtin_epi_vbroadcast_2xf32(const_1_25, gvl);
                        __epi_2xf32 vconst_4_00 = __builtin_epi_vbroadcast_2xf32(const_4_00, gvl);
                        // Compute
                        //   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
                        //   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
                        //   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
                        //   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
                        wd03 = __builtin_epi_vfsub_2xf32(wd03, __builtin_epi_vfmul_2xf32(vconst_1_25, d4, gvl), gvl);
                        wd05 = __builtin_epi_vfmacc_2xf32(wd05, vconst_4_00, d2, gvl);
                        wd04 = __builtin_epi_vfsub_2xf32(wd04, __builtin_epi_vfmul_2xf32(vconst_1_25, d3, gvl), gvl);
                        wd06 = __builtin_epi_vfsub_2xf32(wd06, __builtin_epi_vfmul_2xf32(vconst_1_25, d3, gvl), gvl);

                        const float const_2 = 2.0f;
                        __epi_2xf32 vconst_2 = __builtin_epi_vbroadcast_2xf32(const_2, gvl);

                        wd0 = wd00;
                        wd1 = __builtin_epi_vfadd_2xf32(wd01, wd02, gvl);
                        wd2 = __builtin_epi_vfsub_2xf32(wd01, wd02, gvl);
                        wd3 = __builtin_epi_vfmacc_2xf32(wd03, vconst_2, wd04, gvl);
                        wd8 = __builtin_epi_vfsub_2xf32(wd03, __builtin_epi_vfmul_2xf32(vconst_2, wd04, gvl), gvl);
                        wd9 = __builtin_epi_vfmacc_2xf32(wd05, vconst_2, wd06, gvl);
                        wd10 = __builtin_epi_vfsub_2xf32(wd05, __builtin_epi_vfmul_2xf32(vconst_2, wd06, gvl), gvl);
                        wd11 = wd07;

                        __builtin_epi_vstore_2xf32(&vin0123[0], wd0, gvl);
                        __builtin_epi_vstore_2xf32(&vin0123[simd_width], wd1, gvl);
                        __builtin_epi_vstore_2xf32(&vin0123[2 * simd_width], wd2, gvl);
                        __builtin_epi_vstore_2xf32(&vin0123[3 * simd_width], wd3, gvl);
                        __builtin_epi_vstore_2xf32(&vin01231[0], wd8, gvl);
                        __builtin_epi_vstore_2xf32(&vin01231[simd_width], wd9, gvl);
                        __builtin_epi_vstore_2xf32(&vin01231[2 * simd_width], wd10, gvl);
                        __builtin_epi_vstore_2xf32(&vin01231[3 * simd_width], wd11, gvl);

                        __epi_2xf32 d8 = __builtin_epi_vload_2xf32(&new_data1[0 * tmp_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d9 = __builtin_epi_vload_2xf32(&new_data1[1 * tmp_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d10 = __builtin_epi_vload_2xf32(&new_data1[2 * tmp_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d11 = __builtin_epi_vload_2xf32(&new_data1[3 * tmp_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d12 = __builtin_epi_vload_2xf32(&new_data1[4 * tmp_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d13 = __builtin_epi_vload_2xf32(&new_data1[5 * tmp_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d14 = __builtin_epi_vload_2xf32(&new_data1[6 * tmp_stride + 0 * simd_width], gvl);
                        __epi_2xf32 d15 = __builtin_epi_vload_2xf32(&new_data1[7 * tmp_stride + 0 * simd_width], gvl);

                        wd00 = __builtin_epi_vfsub_2xf32(d8, d14, gvl);
                        const __epi_2xf32 d12_sub_d10 = __builtin_epi_vfsub_2xf32(d12, d10, gvl);

                        // Compute wd7 := d7 - d1
                        wd07 = __builtin_epi_vfsub_2xf32(d15, d9, gvl);
                        const __epi_2xf32 d11_sub_d13 = __builtin_epi_vfsub_2xf32(d11, d13, gvl);
                        // float32x4_t wd1 := d2 + d6
                        wd01 = __builtin_epi_vfadd_2xf32(d10, d14, gvl);
                        // Compute wd2 := d1 + d5
                        wd02 = __builtin_epi_vfadd_2xf32(d9, d13, gvl);
                        // Compute wd4 := d5 + 0.25 * d1
                        wd04 = __builtin_epi_vfmacc_2xf32(d13, vconst_0_25, d9, gvl);
                        // Compute wd5 := d6 - 5.0 * d4
                        wd05 = __builtin_epi_vfsub_2xf32(d14, __builtin_epi_vfmul_2xf32(vconst_0_5, d12, gvl), gvl);
                        // Compute wd3 := d6 + 0.25 * d2
                        wd03 = __builtin_epi_vfmacc_2xf32(d14, vconst_0_25, d10, gvl);
                        // Compute wd6 := d1 + 0.25 * d5
                        wd06 = __builtin_epi_vfmacc_2xf32(d9, vconst_0_25, d13, gvl);
                        // const svfloat32_t const_5_25__4_25 = svld1rq(pg, datatmp1);
                        // const float const_5_25 = 5.25f;
                        // const float const_4_25 = 4.25f;
                        //__epi_2xf32 vconst_5_25 = __builtin_epi_vbroadcast_2xf32(const_5_25, gvl);
                        //__epi_2xf32 vconst_4_25 = __builtin_epi_vbroadcast_2xf32(const_4_25, gvl);
                        // Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
                        wd00 = __builtin_epi_vfmacc_2xf32(wd00, d12_sub_d10, vconst_5_25, gvl);
                        // Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
                        wd07 = __builtin_epi_vfmacc_2xf32(wd07, d11_sub_d13, vconst_5_25, gvl);

                        // Compute
                        //   wd1 := (d6 + d2) - 4.25 * d4
                        //   wd2 := (d1 + d5) - 4.25 * d3
                        wd01 = __builtin_epi_vfsub_2xf32(wd01, __builtin_epi_vfmul_2xf32(vconst_4_25, d12, gvl), gvl);
                        wd02 = __builtin_epi_vfsub_2xf32(wd02, __builtin_epi_vfmul_2xf32(vconst_4_25, d11, gvl), gvl);
                        // const svfloat32_t const_1_25__4_00 = svld1(pg, datatmp2);
                        // const float const_1_25 = 1.25f;
                        // const float const_4_00 =4.00f;
                        //__epi_2xf32 vconst_1_25 = __builtin_epi_vbroadcast_2xf32(const_1_25, gvl);
                        //__epi_2xf32 vconst_4_00 = __builtin_epi_vbroadcast_2xf32(const_4_00, gvl);
                        //  Compute
                        //    wd3 := (d6 + 0.25 * d2) - 1.25 * d4
                        //    wd4 := (d5 + 0.25 * d1) - 1.25 * d3
                        //    wd6 := (d1 + 0.25 * d5) - 1.25 * d3
                        //    wd5 := (d6 - 5.0 * d4) + 4.0 * d2
                        wd03 = __builtin_epi_vfsub_2xf32(wd03, __builtin_epi_vfmul_2xf32(vconst_1_25, d12, gvl), gvl);
                        wd05 = __builtin_epi_vfmacc_2xf32(wd05, vconst_4_00, d10, gvl);
                        wd04 = __builtin_epi_vfsub_2xf32(wd04, __builtin_epi_vfmul_2xf32(vconst_1_25, d11, gvl), gvl);
                        wd06 = __builtin_epi_vfsub_2xf32(wd06, __builtin_epi_vfmul_2xf32(vconst_1_25, d11, gvl), gvl);

                        //   const float const_2 = 2.0f;
                        // __epi_2xf32 vconst_2 = __builtin_epi_vbroadcast_2xf32(const_2, gvl);

                        wd4 = wd00;
                        wd5 = __builtin_epi_vfadd_2xf32(wd01, wd02, gvl);
                        wd6 = __builtin_epi_vfsub_2xf32(wd01, wd02, gvl);
                        wd7 = __builtin_epi_vfmacc_2xf32(wd03, vconst_2, wd04, gvl);
                        wd12 = __builtin_epi_vfsub_2xf32(wd03, __builtin_epi_vfmul_2xf32(vconst_2, wd04, gvl), gvl);
                        wd13 = __builtin_epi_vfmacc_2xf32(wd05, vconst_2, wd06, gvl);
                        wd14 = __builtin_epi_vfsub_2xf32(wd05, __builtin_epi_vfmul_2xf32(vconst_2, wd06, gvl), gvl);
                        wd15 = wd07;

                        __builtin_epi_vstore_2xf32(&vin4567[0], wd4, gvl);
                        __builtin_epi_vstore_2xf32(&vin4567[simd_width], wd5, gvl);
                        __builtin_epi_vstore_2xf32(&vin4567[2 * simd_width], wd6, gvl);
                        __builtin_epi_vstore_2xf32(&vin4567[3 * simd_width], wd7, gvl);
                        __builtin_epi_vstore_2xf32(&vin45671[0], wd12, gvl);
                        __builtin_epi_vstore_2xf32(&vin45671[simd_width], wd13, gvl);
                        __builtin_epi_vstore_2xf32(&vin45671[2 * simd_width], wd14, gvl);
                        __builtin_epi_vstore_2xf32(&vin45671[3 * simd_width], wd15, gvl);

                        // for(int i=0;i<16;i++)
                        //      fprintf(stderr, "intertilevin0123 = %f\n", vin0123[i]);
                }
                else
                {
                        float block[16][simd_width * 2];
                        {
                                float zero = 0.0f;
                                const __epi_2xf32 vzero = __builtin_epi_vbroadcast_2xf32(zero, gvl);
                                for (float *block_ptr = &block[0][0], *block_end = &block[16][0]; block_ptr != block_end; block_ptr += simd_width)
                                {
                                        __builtin_epi_vstore_2xf32(block_ptr, vzero, gvl);
                                }
                        }
#pragma loop unroll_count(interchannels)
                        for (int k = 0; k < interchannels; k++)
                        {
                                for (size_t i = 0; i < row_count; i++)
                                {
                                        for (size_t j = 0; j < (column_count); j++)
                                        {
                                                if (j < 4)
                                                {
                                                        block[row_offset + i][column_offset + (j + k * 4)] = data[k][i * data_stride + j];
                                                }
                                                else
                                                {
                                                        block[row_offset + (i + 8)][column_offset + ((j + k * 4) - 4)] = data[k][i * data_stride + (j)];
                                                }
                                        }
                                }
                        }

                        for (size_t col = 0; col < 1; col++)
                        {
                                __epi_2xf32 d0 = __builtin_epi_vload_2xf32(&block[0][0 * simd_width], gvl);
                                __epi_2xf32 d1 = __builtin_epi_vload_2xf32(&block[1][0 * simd_width], gvl);
                                __epi_2xf32 d2 = __builtin_epi_vload_2xf32(&block[2][0 * simd_width], gvl);
                                __epi_2xf32 d3 = __builtin_epi_vload_2xf32(&block[3][0 * simd_width], gvl);
                                __epi_2xf32 d4 = __builtin_epi_vload_2xf32(&block[4][0 * simd_width], gvl);
                                __epi_2xf32 d5 = __builtin_epi_vload_2xf32(&block[5][0 * simd_width], gvl);
                                __epi_2xf32 d6 = __builtin_epi_vload_2xf32(&block[6][0 * simd_width], gvl);
                                __epi_2xf32 d7 = __builtin_epi_vload_2xf32(&block[7][0 * simd_width], gvl);
                                __epi_2xf32 wd00, wd01, wd02, wd03, wd08, wd09, wd010, wd011;
                                float const_0_25 = 0.25f;
                                float const_0_5 = 5.00f;
                                __epi_2xf32 vconst_0_25 = __builtin_epi_vbroadcast_2xf32(const_0_25, gvl);
                                __epi_2xf32 vconst_0_5 = __builtin_epi_vbroadcast_2xf32(const_0_5, gvl);
                                // const float32x4_t const_0_25 = vmovq_n_f32(0.25f);

                                // Compute wd0 := d0 - d6
                                __epi_2xf32 wd0 = __builtin_epi_vfsub_2xf32(d0, d6, gvl);
                                const __epi_2xf32 d4_sub_d2 = __builtin_epi_vfsub_2xf32(d4, d2, gvl);

                                // Compute wd7 := d7 - d1
                                __epi_2xf32 wd7 = __builtin_epi_vfsub_2xf32(d7, d1, gvl);
                                const __epi_2xf32 d3_sub_d5 = __builtin_epi_vfsub_2xf32(d3, d5, gvl);
                                // float32x4_t wd1 := d2 + d6
                                __epi_2xf32 wd1 = __builtin_epi_vfadd_2xf32(d2, d6, gvl);
                                // Compute wd2 := d1 + d5
                                __epi_2xf32 wd2 = __builtin_epi_vfadd_2xf32(d1, d5, gvl);
                                // Compute wd4 := d5 + 0.25 * d1
                                __epi_2xf32 wd4 = __builtin_epi_vfmacc_2xf32(d5, vconst_0_25, d1, gvl);
                                // Compute wd5 := d6 - 5.0 * d4
                                __epi_2xf32 wd5 = __builtin_epi_vfsub_2xf32(d6, __builtin_epi_vfmul_2xf32(vconst_0_5, d4, gvl), gvl);
                                // Compute wd3 := d6 + 0.25 * d2
                                __epi_2xf32 wd3 = __builtin_epi_vfmacc_2xf32(d6, vconst_0_25, d2, gvl);
                                // Compute wd6 := d1 + 0.25 * d5
                                __epi_2xf32 wd6 = __builtin_epi_vfmacc_2xf32(d1, vconst_0_25, d5, gvl);
                                // const svfloat32_t const_5_25__4_25 = svld1rq(pg, datatmp1);
                                const float const_5_25 = 5.25f;
                                const float const_4_25 = 4.25f;
                                __epi_2xf32 vconst_5_25 = __builtin_epi_vbroadcast_2xf32(const_5_25, gvl);
                                __epi_2xf32 vconst_4_25 = __builtin_epi_vbroadcast_2xf32(const_4_25, gvl);
                                // Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
                                wd0 = __builtin_epi_vfmacc_2xf32(wd0, d4_sub_d2, vconst_5_25, gvl);
                                // Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
                                wd7 = __builtin_epi_vfmacc_2xf32(wd7, d3_sub_d5, vconst_5_25, gvl);

                                // Compute
                                //   wd1 := (d6 + d2) - 4.25 * d4
                                //   wd2 := (d1 + d5) - 4.25 * d3
                                wd1 = __builtin_epi_vfsub_2xf32(wd1, __builtin_epi_vfmul_2xf32(vconst_4_25, d4, gvl), gvl);
                                wd2 = __builtin_epi_vfsub_2xf32(wd2, __builtin_epi_vfmul_2xf32(vconst_4_25, d3, gvl), gvl);
                                // const svfloat32_t const_1_25__4_00 = svld1(pg, datatmp2);
                                const float const_1_25 = 1.25f;
                                const float const_4_00 = 4.00f;
                                __epi_2xf32 vconst_1_25 = __builtin_epi_vbroadcast_2xf32(const_1_25, gvl);
                                __epi_2xf32 vconst_4_00 = __builtin_epi_vbroadcast_2xf32(const_4_00, gvl);
                                // Compute
                                //   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
                                //   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
                                //   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
                                //   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
                                wd3 = __builtin_epi_vfsub_2xf32(wd3, __builtin_epi_vfmul_2xf32(vconst_1_25, d4, gvl), gvl);
                                wd5 = __builtin_epi_vfmacc_2xf32(wd5, vconst_4_00, d2, gvl);
                                wd4 = __builtin_epi_vfsub_2xf32(wd4, __builtin_epi_vfmul_2xf32(vconst_1_25, d3, gvl), gvl);
                                wd6 = __builtin_epi_vfsub_2xf32(wd6, __builtin_epi_vfmul_2xf32(vconst_1_25, d3, gvl), gvl);

                                const float const_2 = 2.0f;
                                __epi_2xf32 vconst_2 = __builtin_epi_vbroadcast_2xf32(const_2, gvl);

                                wd00 = wd0;
                                wd01 = __builtin_epi_vfadd_2xf32(wd1, wd2, gvl);
                                wd02 = __builtin_epi_vfsub_2xf32(wd1, wd2, gvl);
                                wd03 = __builtin_epi_vfmacc_2xf32(wd3, vconst_2, wd4, gvl);
                                wd08 = __builtin_epi_vfsub_2xf32(wd3, __builtin_epi_vfmul_2xf32(vconst_2, wd4, gvl), gvl);
                                wd09 = __builtin_epi_vfmacc_2xf32(wd5, vconst_2, wd6, gvl);
                                wd010 = __builtin_epi_vfsub_2xf32(wd5, __builtin_epi_vfmul_2xf32(vconst_2, wd6, gvl), gvl);
                                wd011 = wd7;
                                __builtin_epi_vstore_2xf32(&vin0123[0], wd00, gvl);
                                __builtin_epi_vstore_2xf32(&vin0123[simd_width], wd01, gvl);
                                __builtin_epi_vstore_2xf32(&vin0123[2 * simd_width], wd02, gvl);
                                __builtin_epi_vstore_2xf32(&vin0123[3 * simd_width], wd03, gvl);
                                __builtin_epi_vstore_2xf32(&vin01231[0], wd08, gvl);
                                __builtin_epi_vstore_2xf32(&vin01231[simd_width], wd09, gvl);
                                __builtin_epi_vstore_2xf32(&vin01231[2 * simd_width], wd010, gvl);
                                __builtin_epi_vstore_2xf32(&vin01231[3 * simd_width], wd011, gvl);

                                // next itertaion

                                __epi_2xf32 d8 = __builtin_epi_vload_2xf32(&block[8][0 * simd_width], gvl);
                                __epi_2xf32 d9 = __builtin_epi_vload_2xf32(&block[9][0 * simd_width], gvl);
                                __epi_2xf32 d10 = __builtin_epi_vload_2xf32(&block[10][0 * simd_width], gvl);
                                __epi_2xf32 d11 = __builtin_epi_vload_2xf32(&block[11][0 * simd_width], gvl);
                                __epi_2xf32 d12 = __builtin_epi_vload_2xf32(&block[12][0 * simd_width], gvl);
                                __epi_2xf32 d13 = __builtin_epi_vload_2xf32(&block[13][0 * simd_width], gvl);
                                __epi_2xf32 d14 = __builtin_epi_vload_2xf32(&block[14][0 * simd_width], gvl);
                                __epi_2xf32 d15 = __builtin_epi_vload_2xf32(&block[15][0 * simd_width], gvl);
                                //__epi_2xf32 wd00, wd01, wd02, wd03,wd08, wd09, wd010, wd011;
                                __epi_2xf32 wd04, wd05, wd06, wd07, wd012, wd013, wd014, wd015; // upto here
                                // float const_0_25 = 0.25f;
                                // float const_0_5 = 5.00f;
                                //__epi_2xf32 vconst_0_25 = __builtin_epi_vbroadcast_2xf32(const_0_25, gvl);
                                //__epi_2xf32 vconst_0_5 = __builtin_epi_vbroadcast_2xf32(const_0_5, gvl);
                                // const float32x4_t const_0_25 = vmovq_n_f32(0.25f);

                                // Compute wd0 := d0 - d6
                                wd0 = __builtin_epi_vfsub_2xf32(d8, d14, gvl);
                                const __epi_2xf32 d12_sub_d10 = __builtin_epi_vfsub_2xf32(d12, d10, gvl);

                                // Compute wd7 := d7 - d1
                                wd7 = __builtin_epi_vfsub_2xf32(d15, d9, gvl);
                                const __epi_2xf32 d11_sub_d13 = __builtin_epi_vfsub_2xf32(d11, d13, gvl);
                                // float32x4_t wd1 := d2 + d6
                                wd1 = __builtin_epi_vfadd_2xf32(d10, d14, gvl);
                                // Compute wd2 := d1 + d5
                                wd2 = __builtin_epi_vfadd_2xf32(d9, d13, gvl);
                                // Compute wd4 := d5 + 0.25 * d1
                                wd4 = __builtin_epi_vfmacc_2xf32(d13, vconst_0_25, d9, gvl);
                                // Compute wd5 := d6 - 5.0 * d4
                                wd5 = __builtin_epi_vfsub_2xf32(d14, __builtin_epi_vfmul_2xf32(vconst_0_5, d12, gvl), gvl);
                                // Compute wd3 := d6 + 0.25 * d2
                                wd3 = __builtin_epi_vfmacc_2xf32(d14, vconst_0_25, d10, gvl);
                                // Compute wd6 := d1 + 0.25 * d5
                                wd6 = __builtin_epi_vfmacc_2xf32(d9, vconst_0_25, d13, gvl);
                                // const svfloat32_t const_5_25__4_25 = svld1rq(pg, datatmp1);
                                // const float const_5_25 = 5.25f;
                                // const float const_4_25 = 4.25f;
                                //__epi_2xf32 vconst_5_25 = __builtin_epi_vbroadcast_2xf32(const_5_25, gvl);
                                //__epi_2xf32 vconst_4_25 = __builtin_epi_vbroadcast_2xf32(const_4_25, gvl);
                                // Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
                                wd0 = __builtin_epi_vfmacc_2xf32(wd0, d12_sub_d10, vconst_5_25, gvl);
                                // Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
                                wd7 = __builtin_epi_vfmacc_2xf32(wd7, d11_sub_d13, vconst_5_25, gvl);

                                // Compute
                                //   wd1 := (d6 + d2) - 4.25 * d4
                                //   wd2 := (d1 + d5) - 4.25 * d3
                                wd1 = __builtin_epi_vfsub_2xf32(wd1, __builtin_epi_vfmul_2xf32(vconst_4_25, d12, gvl), gvl);
                                wd2 = __builtin_epi_vfsub_2xf32(wd2, __builtin_epi_vfmul_2xf32(vconst_4_25, d11, gvl), gvl);
                                // const svfloat32_t const_1_25__4_00 = svld1(pg, datatmp2);
                                // const float const_1_25 = 1.25f;
                                // const float const_4_00 =4.00f;
                                //__epi_2xf32 vconst_1_25 = __builtin_epi_vbroadcast_2xf32(const_1_25, gvl);
                                //__epi_2xf32 vconst_4_00 = __builtin_epi_vbroadcast_2xf32(const_4_00, gvl);
                                //  Compute
                                //    wd3 := (d6 + 0.25 * d2) - 1.25 * d4
                                //    wd4 := (d5 + 0.25 * d1) - 1.25 * d3
                                //    wd6 := (d1 + 0.25 * d5) - 1.25 * d3
                                //    wd5 := (d6 - 5.0 * d4) + 4.0 * d2
                                wd3 = __builtin_epi_vfsub_2xf32(wd3, __builtin_epi_vfmul_2xf32(vconst_1_25, d12, gvl), gvl);
                                wd5 = __builtin_epi_vfmacc_2xf32(wd5, vconst_4_00, d10, gvl);
                                wd4 = __builtin_epi_vfsub_2xf32(wd4, __builtin_epi_vfmul_2xf32(vconst_1_25, d11, gvl), gvl);
                                wd6 = __builtin_epi_vfsub_2xf32(wd6, __builtin_epi_vfmul_2xf32(vconst_1_25, d11, gvl), gvl);

                                //   const float const_2 = 2.0f;
                                // __epi_2xf32 vconst_2 = __builtin_epi_vbroadcast_2xf32(const_2, gvl);

                                wd04 = wd0;
                                wd05 = __builtin_epi_vfadd_2xf32(wd1, wd2, gvl);
                                wd06 = __builtin_epi_vfsub_2xf32(wd1, wd2, gvl);
                                wd07 = __builtin_epi_vfmacc_2xf32(wd3, vconst_2, wd4, gvl);
                                wd012 = __builtin_epi_vfsub_2xf32(wd3, __builtin_epi_vfmul_2xf32(vconst_2, wd4, gvl), gvl);
                                wd013 = __builtin_epi_vfmacc_2xf32(wd5, vconst_2, wd6, gvl);
                                wd014 = __builtin_epi_vfsub_2xf32(wd5, __builtin_epi_vfmul_2xf32(vconst_2, wd6, gvl), gvl);
                                wd015 = wd7;

                                __builtin_epi_vstore_2xf32(&vin4567[0], wd04, gvl);
                                __builtin_epi_vstore_2xf32(&vin4567[simd_width], wd05, gvl);
                                __builtin_epi_vstore_2xf32(&vin4567[2 * simd_width], wd06, gvl);
                                __builtin_epi_vstore_2xf32(&vin4567[3 * simd_width], wd07, gvl);
                                __builtin_epi_vstore_2xf32(&vin45671[0], wd012, gvl);
                                __builtin_epi_vstore_2xf32(&vin45671[simd_width], wd013, gvl);
                                __builtin_epi_vstore_2xf32(&vin45671[2 * simd_width], wd014, gvl);
                                __builtin_epi_vstore_2xf32(&vin45671[3 * simd_width], wd015, gvl);

                                //	for(int i=0;i<32;i++)
                                //		fprintf(stderr, "intertile =%f\n", vin0123[i]);
                        }
                }
                float tmp_transform[16 * simd_width];
                //		printf( "value of simd_width=%d", simd_width);
                int index1_host[simd_width];
                int ind1 = 0;
                for (size_t col = 0; col < 1; col++)
                {
                        int icol = 0;
                        for (int i = 0; i < interchannels; i++)
                        {
                                for (int j = 0; j < 4; j++)
                                {
                                        index1_host[ind1] = ((j * simd_width) + 4 * i);
                                        ind1++;
                                }
                        }
                        int one = 1;
                        const __epi_2xi32 vone = __builtin_epi_vbroadcast_2xi32(one, gvl);
                        __epi_2xi32 index1, index2, index3, index4;

                        index1 = __builtin_epi_vload_2xi32(&index1_host[0], gvl);
                        index2 = __builtin_epi_vadd_2xi32(index1, vone, gvl);
                        index3 = __builtin_epi_vadd_2xi32(index2, vone, gvl);
                        index4 = __builtin_epi_vadd_2xi32(index3, vone, gvl);
                        __epi_2xi32 index11 = __builtin_epi_vmul_2xi32(index1, FOUR, gvl);
                        __epi_2xi32 index12 = __builtin_epi_vmul_2xi32(index2, FOUR, gvl);
                        __epi_2xi32 index13 = __builtin_epi_vmul_2xi32(index3, FOUR, gvl);
                        __epi_2xi32 index14 = __builtin_epi_vmul_2xi32(index4, FOUR, gvl);

                        {
                                __epi_2xf32 d0 = __builtin_epi_vload_indexed_2xf32(&vin0123[0], index11, gvl);
                                __epi_2xf32 d1 = __builtin_epi_vload_indexed_2xf32(&vin0123[0], index12, gvl);
                                __epi_2xf32 d2 = __builtin_epi_vload_indexed_2xf32(&vin0123[0], index13, gvl);
                                __epi_2xf32 d3 = __builtin_epi_vload_indexed_2xf32(&vin0123[0], index14, gvl);
                                __epi_2xf32 d4 = __builtin_epi_vload_indexed_2xf32(&vin4567[0], index11, gvl);
                                __epi_2xf32 d5 = __builtin_epi_vload_indexed_2xf32(&vin4567[0], index12, gvl);
                                __epi_2xf32 d6 = __builtin_epi_vload_indexed_2xf32(&vin4567[0], index13, gvl);
                                __epi_2xf32 d7 = __builtin_epi_vload_indexed_2xf32(&vin4567[0], index14, gvl);
                                __epi_2xf32 vout0, vout1, vout2, vout3, vout4, vout5, vout6, vout7;

                                float const_0_25 = 0.25f;
                                float const_0_5 = 5.00f;
                                __epi_2xf32 vconst_0_25 = __builtin_epi_vbroadcast_2xf32(const_0_25, gvl);
                                __epi_2xf32 vconst_0_5 = __builtin_epi_vbroadcast_2xf32(const_0_5, gvl);
                                // const float32x4_t const_0_25 = vmovq_n_f32(0.25f);

                                // Compute wd0 := d0 - d6
                                __epi_2xf32 wd0 = __builtin_epi_vfsub_2xf32(d0, d6, gvl);
                                const __epi_2xf32 d4_sub_d2 = __builtin_epi_vfsub_2xf32(d4, d2, gvl);

                                // Compute wd7 := d7 - d1
                                __epi_2xf32 wd7 = __builtin_epi_vfsub_2xf32(d7, d1, gvl);
                                const __epi_2xf32 d3_sub_d5 = __builtin_epi_vfsub_2xf32(d3, d5, gvl);
                                // float32x4_t wd1 := d2 + d6
                                __epi_2xf32 wd1 = __builtin_epi_vfadd_2xf32(d2, d6, gvl);
                                // Compute wd2 := d1 + d5
                                __epi_2xf32 wd2 = __builtin_epi_vfadd_2xf32(d1, d5, gvl);
                                // Compute wd4 := d5 + 0.25 * d1
                                __epi_2xf32 wd4 = __builtin_epi_vfmacc_2xf32(d5, vconst_0_25, d1, gvl);
                                // Compute wd5 := d6 - 5.0 * d4
                                __epi_2xf32 wd5 = __builtin_epi_vfsub_2xf32(d6, __builtin_epi_vfmul_2xf32(vconst_0_5, d4, gvl), gvl);
                                // Compute wd3 := d6 + 0.25 * d2
                                __epi_2xf32 wd3 = __builtin_epi_vfmacc_2xf32(d6, vconst_0_25, d2, gvl);
                                // Compute wd6 := d1 + 0.25 * d5
                                __epi_2xf32 wd6 = __builtin_epi_vfmacc_2xf32(d1, vconst_0_25, d5, gvl);
                                // const svfloat32_t const_5_25__4_25 = svld1rq(pg, datatmp1);
                                const float const_5_25 = 5.25f;
                                const float const_4_25 = 4.25f;
                                __epi_2xf32 vconst_5_25 = __builtin_epi_vbroadcast_2xf32(const_5_25, gvl);
                                __epi_2xf32 vconst_4_25 = __builtin_epi_vbroadcast_2xf32(const_4_25, gvl);

                                // Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
                                wd0 = __builtin_epi_vfmacc_2xf32(wd0, d4_sub_d2, vconst_5_25, gvl);
                                // Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
                                wd7 = __builtin_epi_vfmacc_2xf32(wd7, d3_sub_d5, vconst_5_25, gvl);
                                // Compute
                                //   wd1 := (d6 + d2) - 4.25 * d4
                                //   wd2 := (d1 + d5) - 4.25 * d3
                                wd1 = __builtin_epi_vfsub_2xf32(wd1, __builtin_epi_vfmul_2xf32(vconst_4_25, d4, gvl), gvl);
                                wd2 = __builtin_epi_vfsub_2xf32(wd2, __builtin_epi_vfmul_2xf32(vconst_4_25, d3, gvl), gvl);
                                // const svfloat32_t const_1_25__4_00 = svld1(pg, datatmp2);
                                const float const_1_25 = 1.25f;
                                const float const_4_00 = 4.00f;
                                __epi_2xf32 vconst_1_25 = __builtin_epi_vbroadcast_2xf32(const_1_25, gvl);
                                __epi_2xf32 vconst_4_00 = __builtin_epi_vbroadcast_2xf32(const_4_00, gvl);
                                // Compute
                                //   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
                                //   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
                                //   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
                                //   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
                                wd3 = __builtin_epi_vfsub_2xf32(wd3, __builtin_epi_vfmul_2xf32(vconst_1_25, d4, gvl), gvl);
                                wd5 = __builtin_epi_vfmacc_2xf32(wd5, vconst_4_00, d2, gvl);
                                wd4 = __builtin_epi_vfsub_2xf32(wd4, __builtin_epi_vfmul_2xf32(vconst_1_25, d3, gvl), gvl);
                                wd6 = __builtin_epi_vfsub_2xf32(wd6, __builtin_epi_vfmul_2xf32(vconst_1_25, d3, gvl), gvl);

                                const float const_2 = 2.0f;
                                __epi_2xf32 vconst_2 = __builtin_epi_vbroadcast_2xf32(const_2, gvl);

                                vout0 = wd0;
                                vout1 = __builtin_epi_vfadd_2xf32(wd1, wd2, gvl);
                                vout2 = __builtin_epi_vfsub_2xf32(wd1, wd2, gvl);
                                vout3 = __builtin_epi_vfmacc_2xf32(wd3, vconst_2, wd4, gvl);
                                vout4 = __builtin_epi_vfsub_2xf32(wd3, __builtin_epi_vfmul_2xf32(vconst_2, wd4, gvl), gvl);
                                vout5 = __builtin_epi_vfmacc_2xf32(wd5, vconst_2, wd6, gvl);
                                vout6 = __builtin_epi_vfsub_2xf32(wd5, __builtin_epi_vfmul_2xf32(vconst_2, wd6, gvl), gvl);
                                vout7 = wd7;

                                __builtin_epi_vstore_2xf32(&tmp_transform[0], vout0, gvl);
                                __builtin_epi_vstore_2xf32(&tmp_transform[1 * simd_width], vout1, gvl);
                                __builtin_epi_vstore_2xf32(&tmp_transform[2 * simd_width], vout2, gvl);
                                __builtin_epi_vstore_2xf32(&tmp_transform[3 * simd_width], vout3, gvl);
                                __builtin_epi_vstore_2xf32(&tmp_transform[4 * simd_width], vout4, gvl);
                                __builtin_epi_vstore_2xf32(&tmp_transform[5 * simd_width], vout5, gvl);
                                __builtin_epi_vstore_2xf32(&tmp_transform[6 * simd_width], vout6, gvl);
                                __builtin_epi_vstore_2xf32(&tmp_transform[7 * simd_width], vout7, gvl);
                        }
                        {
                                __epi_2xf32 vout10, vout11, vout12, vout13, vout14, vout15, vout16, vout17;
                                __epi_2xf32 d0 = __builtin_epi_vload_indexed_2xf32(&vin01231[0], index11, gvl);
                                __epi_2xf32 d1 = __builtin_epi_vload_indexed_2xf32(&vin01231[0], index12, gvl);
                                __epi_2xf32 d2 = __builtin_epi_vload_indexed_2xf32(&vin01231[0], index13, gvl);
                                __epi_2xf32 d3 = __builtin_epi_vload_indexed_2xf32(&vin01231[0], index14, gvl);
                                __epi_2xf32 d4 = __builtin_epi_vload_indexed_2xf32(&vin45671[0], index11, gvl);
                                __epi_2xf32 d5 = __builtin_epi_vload_indexed_2xf32(&vin45671[0], index12, gvl);
                                __epi_2xf32 d6 = __builtin_epi_vload_indexed_2xf32(&vin45671[0], index13, gvl);
                                __epi_2xf32 d7 = __builtin_epi_vload_indexed_2xf32(&vin45671[0], index14, gvl);

                                float const_0_25 = 0.25f;
                                float const_0_5 = 5.00f;
                                __epi_2xf32 vconst_0_25 = __builtin_epi_vbroadcast_2xf32(const_0_25, gvl);
                                __epi_2xf32 vconst_0_5 = __builtin_epi_vbroadcast_2xf32(const_0_5, gvl);
                                // const float32x4_t const_0_25 = vmovq_n_f32(0.25f);

                                // Compute wd0 := d0 - d6
                                __epi_2xf32 wd0 = __builtin_epi_vfsub_2xf32(d0, d6, gvl);
                                const __epi_2xf32 d4_sub_d2 = __builtin_epi_vfsub_2xf32(d4, d2, gvl);

                                // Compute wd7 := d7 - d1
                                __epi_2xf32 wd7 = __builtin_epi_vfsub_2xf32(d7, d1, gvl);
                                const __epi_2xf32 d3_sub_d5 = __builtin_epi_vfsub_2xf32(d3, d5, gvl);
                                // float32x4_t wd1 := d2 + d6
                                __epi_2xf32 wd1 = __builtin_epi_vfadd_2xf32(d2, d6, gvl);
                                // Compute wd2 := d1 + d5
                                __epi_2xf32 wd2 = __builtin_epi_vfadd_2xf32(d1, d5, gvl);
                                // Compute wd4 := d5 + 0.25 * d1
                                __epi_2xf32 wd4 = __builtin_epi_vfmacc_2xf32(d5, vconst_0_25, d1, gvl);
                                // Compute wd5 := d6 - 5.0 * d4
                                __epi_2xf32 wd5 = __builtin_epi_vfsub_2xf32(d6, __builtin_epi_vfmul_2xf32(vconst_0_5, d4, gvl), gvl);
                                // Compute wd3 := d6 + 0.25 * d2
                                __epi_2xf32 wd3 = __builtin_epi_vfmacc_2xf32(d6, vconst_0_25, d2, gvl);
                                // Compute wd6 := d1 + 0.25 * d5
                                __epi_2xf32 wd6 = __builtin_epi_vfmacc_2xf32(d1, vconst_0_25, d5, gvl);
                                // const svfloat32_t const_5_25__4_25 = svld1rq(pg, datatmp1);
                                const float const_5_25 = 5.25f;
                                const float const_4_25 = 4.25f;
                                __epi_2xf32 vconst_5_25 = __builtin_epi_vbroadcast_2xf32(const_5_25, gvl);
                                __epi_2xf32 vconst_4_25 = __builtin_epi_vbroadcast_2xf32(const_4_25, gvl);

                                // Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)
                                wd0 = __builtin_epi_vfmacc_2xf32(wd0, d4_sub_d2, vconst_5_25, gvl);
                                // Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)
                                wd7 = __builtin_epi_vfmacc_2xf32(wd7, d3_sub_d5, vconst_5_25, gvl);
                                // Compute
                                //   wd1 := (d6 + d2) - 4.25 * d4
                                //   wd2 := (d1 + d5) - 4.25 * d3
                                wd1 = __builtin_epi_vfsub_2xf32(wd1, __builtin_epi_vfmul_2xf32(vconst_4_25, d4, gvl), gvl);
                                wd2 = __builtin_epi_vfsub_2xf32(wd2, __builtin_epi_vfmul_2xf32(vconst_4_25, d3, gvl), gvl);
                                // const svfloat32_t const_1_25__4_00 = svld1(pg, datatmp2);
                                const float const_1_25 = 1.25f;
                                const float const_4_00 = 4.00f;
                                __epi_2xf32 vconst_1_25 = __builtin_epi_vbroadcast_2xf32(const_1_25, gvl);
                                __epi_2xf32 vconst_4_00 = __builtin_epi_vbroadcast_2xf32(const_4_00, gvl);
                                // Compute
                                //   wd3 := (d6 + 0.25 * d2) - 1.25 * d4
                                //   wd4 := (d5 + 0.25 * d1) - 1.25 * d3
                                //   wd6 := (d1 + 0.25 * d5) - 1.25 * d3
                                //   wd5 := (d6 - 5.0 * d4) + 4.0 * d2
                                wd3 = __builtin_epi_vfsub_2xf32(wd3, __builtin_epi_vfmul_2xf32(vconst_1_25, d4, gvl), gvl);
                                wd5 = __builtin_epi_vfmacc_2xf32(wd5, vconst_4_00, d2, gvl);
                                wd4 = __builtin_epi_vfsub_2xf32(wd4, __builtin_epi_vfmul_2xf32(vconst_1_25, d3, gvl), gvl);
                                wd6 = __builtin_epi_vfsub_2xf32(wd6, __builtin_epi_vfmul_2xf32(vconst_1_25, d3, gvl), gvl);

                                const float const_2 = 2.0f;
                                __epi_2xf32 vconst_2 = __builtin_epi_vbroadcast_2xf32(const_2, gvl);

                                vout10 = wd0;
                                vout11 = __builtin_epi_vfadd_2xf32(wd1, wd2, gvl);
                                vout12 = __builtin_epi_vfsub_2xf32(wd1, wd2, gvl);
                                vout13 = __builtin_epi_vfmacc_2xf32(wd3, vconst_2, wd4, gvl);
                                vout14 = __builtin_epi_vfsub_2xf32(wd3, __builtin_epi_vfmul_2xf32(vconst_2, wd4, gvl), gvl);
                                vout15 = __builtin_epi_vfmacc_2xf32(wd5, vconst_2, wd6, gvl);
                                vout16 = __builtin_epi_vfsub_2xf32(wd5, __builtin_epi_vfmul_2xf32(vconst_2, wd6, gvl), gvl);
                                vout17 = wd7;
                                //__builtin_epi_vstore_2xf32(&tmp[0], vout14,gvl);
                                // for(int i=0;i<16;i++)
                                //       printf("vout 14 = %f\n", tmp[i]);
                                __builtin_epi_vstore_2xf32(&tmp_transform[8 * simd_width], vout10, gvl);
                                __builtin_epi_vstore_2xf32(&tmp_transform[9 * simd_width], vout11, gvl);
                                __builtin_epi_vstore_2xf32(&tmp_transform[10 * simd_width], vout12, gvl);
                                __builtin_epi_vstore_2xf32(&tmp_transform[11 * simd_width], vout13, gvl);
                                __builtin_epi_vstore_2xf32(&tmp_transform[12 * simd_width], vout14, gvl);
                                __builtin_epi_vstore_2xf32(&tmp_transform[13 * simd_width], vout15, gvl);
                                __builtin_epi_vstore_2xf32(&tmp_transform[14 * simd_width], vout16, gvl);
                                __builtin_epi_vstore_2xf32(&tmp_transform[15 * simd_width], vout17, gvl);
                        }
                        //			for(int i=0;i<48;i++)
                        //			{
                        //				printf(" value of trans%f\n", tmp_transform[i]);
                        //			}
                        for (size_t i = 0; i < 16; i++)
                        {
#pragma loop unroll_count(interchannels)
                                for (int k = 0; k < interchannels; k++)
                                {
                                        for (size_t j = 0; j < 4; j++)
                                        {
                                                *(((float *)transform[k]) + j) = tmp_transform[i * simd_width + (j + k * 4)];
                                        }
                                        transform[k] += transform_stride;
                                }
                        }

                        /* for (size_t i = 0; i < 16; i++)
                         {
                                 for (size_t j = 0; j < 4; j++)
                                 {
                                 *(((float *)transform[0])+j)   = tmp_transform[i*simd_width+j];
                                 *(((float *)transform[1])+j)   = tmp_transform[i*simd_width + (j+4)];
                                 if(interchannels > 2) {*(((float *)transform[2])+j)   = tmp_transform[i*simd_width + (j+8)];}
                                 if(interchannels > 3) {*(((float *)transform[3])+j)   = tmp_transform[i*simd_width + (j+12)];}
                                 if(interchannels > 4) {*(((float *)transform[4])+j)   = tmp_transform[i*simd_width + (j+16)];}
                                 if(interchannels > 5){*(((float *)transform[5])+j)   = tmp_transform[i*simd_width + (j+20)];}
                                 if(interchannels > 6) {*(((float *)transform[6])+j)   = tmp_transform[i*simd_width + (j+24)];}
                                 if(interchannels > 7) { *(((float *)transform[7])+j)   = tmp_transform[i*simd_width + (j+28)];}
                                 if(interchannels > 8) {*(((float *)transform[8])+j)   = tmp_transform[i*simd_width + (j+32)];}
                                 if(interchannels > 9) {*(((float *)transform[9])+j)   = tmp_transform[i*simd_width + (j+36)];}
                                 if(interchannels > 10) {*(((float *)transform[10])+j)   = tmp_transform[i*simd_width + (j+40)];}
                                 if(interchannels > 11){*(((float *)transform[11])+j)   = tmp_transform[i*simd_width + (j+44)];}
                                 if(interchannels > 12) {*(((float *)transform[12])+j)   = tmp_transform[i*simd_width + (j+48)];}
                                 if(interchannels > 13) { *(((float *)transform[13])+j)   = tmp_transform[i*simd_width + (j+52)];}
                                 if(interchannels > 14) { *(((float *)transform[14])+j)   = tmp_transform[i*simd_width + (j+56)];}
                                 if(interchannels > 15) { *(((float *)transform[15])+j)   = tmp_transform[i*simd_width + (j+60)];}

                                 }
                                 transform[0] += transform_stride;
                                 transform[1] += transform_stride;
                                 if(interchannels > 2) {transform[2] += transform_stride;}
                                 if(interchannels > 3) {transform[3] += transform_stride;}
                                  if(interchannels > 4) {transform[4] += transform_stride;}
                                 if(interchannels > 5) {transform[5] += transform_stride;}
                                 if(interchannels > 6) {transform[6] += transform_stride;}
                                 if(interchannels > 7) {transform[7] += transform_stride;}
                                  if(interchannels > 8) {transform[8] += transform_stride;}
                                 if(interchannels > 9) {transform[9] += transform_stride;}
                                  if(interchannels > 10) {transform[10] += transform_stride;}
                                 if(interchannels > 11) {transform[11] += transform_stride;}
                                 if(interchannels > 12) {transform[12] += transform_stride;}
                                 if(interchannels > 13) {transform[13] += transform_stride;}
                                 if(interchannels > 14) {transform[14] += transform_stride;}
                                 if(interchannels > 15) {transform[15] += transform_stride;}

                         }*/
                }
                i1 += gvl;
        }
        if (new_data != NULL)
        {
                free(new_data);
                new_data = NULL;
        }
        if (new_data1 != NULL)
        {
                free(new_data1);
                new_data1 = NULL;
        }
}
// intertile input transform

void nnp_kwt8x8_3x3__neon(
    const float g[restrict static 9],
    float transform[restrict static 1],
    size_t stride_g,
    size_t transform_stride,
    uint32_t row_count,
    uint32_t column_count,
    uint32_t row_offset,
    uint32_t column_offset)
{

        int simd_width = nnp_hwinfo.simd_width; // 16;//nnp_hwinfo.simd_width;

        transform_stride /= sizeof(float);

        for (int i1 = 0; i1 < simd_width;)
        {
                unsigned long gvl = __builtin_epi_vsetvl(((long)simd_width - (long)i1), __epi_e32, __epi_m1);
                __epi_2xf32 g0_vec = __builtin_epi_vload_2xf32(g, gvl);
                __epi_2xf32 g1_vec = __builtin_epi_vload_2xf32(g + 3, gvl);
                // g2[3] is junk
                float tmpbuff[simd_width];
                for (int i = 0; i < 3; i++)
                {
                        tmpbuff[i] = g[6 + i];
                }
                tmpbuff[3] = g[5];
                //__epi_2xf32 g2_vec = __builtin_epi_vload_2xf32( &g[6], gvl);
                __epi_2xf32 g2_vec = __builtin_epi_vload_2xf32(&tmpbuff[0], gvl);

                __epi_2xf32 transform0, transform1, transform2, transform3, transform4, transform5, transform6, transform7;
                float four = 4.0;
                const __epi_2xf32 const_4 = __builtin_epi_vbroadcast_2xf32(four, gvl);
                __epi_2xf32 w2 = __builtin_epi_vfadd_2xf32(g0_vec, g2_vec, gvl);
                __epi_2xf32 w4 = __builtin_epi_vfmacc_2xf32(g0_vec, const_4, g2_vec, gvl);
                __epi_2xf32 w6 = __builtin_epi_vfmacc_2xf32(g2_vec, const_4, g0_vec, gvl);
                float two = 2.0f;
                __epi_2xf32 v_two = __builtin_epi_vbroadcast_2xf32(two, gvl);
                const __epi_2xf32 two_g1 = __builtin_epi_vfmul_2xf32(g1_vec, v_two, gvl);
                __epi_2xf32 w1 = __builtin_epi_vfadd_2xf32(w2, g1_vec, gvl);
                w2 = __builtin_epi_vfsub_2xf32(w2, g1_vec, gvl);
                // w2 = w2 - g1;
                __epi_2xf32 w3 = __builtin_epi_vfadd_2xf32(w4, two_g1, gvl);
                w4 = __builtin_epi_vfsub_2xf32(w4, two_g1, gvl);
                __epi_2xf32 w5 = __builtin_epi_vfadd_2xf32(w6, two_g1, gvl);
                w6 = __builtin_epi_vfsub_2xf32(w6, two_g1, gvl);

                if (rescale_coefficients)
                {
                        float var = -0x1.C71C72p-3f;
                        const __epi_2xf32 minus_2_over_9 = __builtin_epi_vbroadcast_2xf32(var, gvl);

                        w1 = __builtin_epi_vfmul_2xf32(w1, minus_2_over_9, gvl);
                        w2 = __builtin_epi_vfmul_2xf32(w2, minus_2_over_9, gvl);
                        float var1 = 0x1.6C16C2p-7f;
                        const __epi_2xf32 rcp_90 = __builtin_epi_vbroadcast_2xf32(var1, gvl);
                        w3 = __builtin_epi_vfmul_2xf32(w3, rcp_90, gvl);
                        w4 = __builtin_epi_vfmul_2xf32(w4, rcp_90, gvl);
                        float var2 = 0x1.6C16C2p-8f;
                        const __epi_2xf32 rcp_180 = __builtin_epi_vbroadcast_2xf32(var2, gvl);
                        w5 = __builtin_epi_vfmul_2xf32(w5, rcp_180, gvl);
                        w6 = __builtin_epi_vfmul_2xf32(w6, rcp_180, gvl);
                }

                transform0 = g0_vec;
                transform1 = w1;
                transform2 = w2;
                transform3 = w3;
                transform4 = w4;
                transform5 = w5;
                transform6 = w6;
                transform7 = g2_vec;
                float buff[4 * simd_width], buff1[4 * simd_width];
                __builtin_epi_vstore_2xf32(&buff[0], transform0, gvl);
                __builtin_epi_vstore_2xf32(&buff[simd_width], transform1, gvl);
                __builtin_epi_vstore_2xf32(&buff[2 * simd_width], transform2, gvl);
                __builtin_epi_vstore_2xf32(&buff[3 * simd_width], transform3, gvl);
                __builtin_epi_vstore_2xf32(&buff1[0], transform4, gvl);
                __builtin_epi_vstore_2xf32(&buff1[simd_width], transform5, gvl);
                __builtin_epi_vstore_2xf32(&buff1[2 * simd_width], transform6, gvl);
                __builtin_epi_vstore_2xf32(&buff1[3 * simd_width], transform7, gvl);

                // for(int i=0;i<4;i++)
                //       printf("intermediate = %f \n", buff[i]);

                int ind1 = 0;
                int index1_host[simd_width];
                for (int j = 0; j < 4; j++)
                {
                        index1_host[ind1] = ((j * simd_width));
                        ind1++;
                }
                int one = 1;
                const __epi_2xi32 vone = __builtin_epi_vbroadcast_2xi32(one, gvl);
                __epi_2xi32 index1, index2, index3, index4;
                index1 = __builtin_epi_vload_2xi32(&index1_host[0], gvl);
                index2 = __builtin_epi_vadd_2xi32(index1, vone, gvl);
                index3 = __builtin_epi_vadd_2xi32(index2, vone, gvl);
                index4 = __builtin_epi_vadd_2xi32(index3, vone, gvl);
                int fourv = 4;
                const __epi_2xi32 FOUR = __builtin_epi_vbroadcast_2xi32(fourv, gvl);
                __epi_2xi32 index11 = __builtin_epi_vmul_2xi32(index1, FOUR, gvl);
                __epi_2xi32 index12 = __builtin_epi_vmul_2xi32(index2, FOUR, gvl);
                __epi_2xi32 index13 = __builtin_epi_vmul_2xi32(index3, FOUR, gvl);
                __epi_2xi32 index14 = __builtin_epi_vmul_2xi32(index4, FOUR, gvl);
                __epi_2xf32 row0 = __builtin_epi_vload_indexed_2xf32(&buff[0], index11, gvl);
                __epi_2xf32 row1 = __builtin_epi_vload_indexed_2xf32(&buff[0], index12, gvl);
                __epi_2xf32 row2 = __builtin_epi_vload_indexed_2xf32(&buff[0], index13, gvl);
                __epi_2xf32 row3 = __builtin_epi_vload_indexed_2xf32(&buff[0], index14, gvl);

                __epi_2xf32 row4 = __builtin_epi_vload_indexed_2xf32(&buff1[0], index11, gvl);
                __epi_2xf32 row5 = __builtin_epi_vload_indexed_2xf32(&buff1[0], index12, gvl);
                __epi_2xf32 row6 = __builtin_epi_vload_indexed_2xf32(&buff1[0], index13, gvl);
                __epi_2xf32 row7 = __builtin_epi_vload_indexed_2xf32(&buff1[0], index14, gvl);

                __epi_2xf32 wg00, wg10, wg20, wg30, wg40, wg50, wg60, wg70, wg01, wg11, wg21, wg31, wg41, wg51, wg61, wg71;

                __epi_2xf32 w2_tmp = __builtin_epi_vfadd_2xf32(row0, row2, gvl);
                __epi_2xf32 w4_tmp = __builtin_epi_vfmacc_2xf32(row0, const_4, row2, gvl);
                __epi_2xf32 w6_tmp = __builtin_epi_vfmacc_2xf32(row2, const_4, row0, gvl);
                const __epi_2xf32 two_g1_tmp = __builtin_epi_vfmul_2xf32(row1, v_two, gvl);
                __epi_2xf32 w1_tmp = __builtin_epi_vfadd_2xf32(w2_tmp, row1, gvl);
                w2_tmp = __builtin_epi_vfsub_2xf32(w2_tmp, row1, gvl);
                // w2 = w2 - g1;
                __epi_2xf32 w3_tmp = __builtin_epi_vfadd_2xf32(w4_tmp, two_g1_tmp, gvl);
                w4_tmp = __builtin_epi_vfsub_2xf32(w4_tmp, two_g1_tmp, gvl);
                __epi_2xf32 w5_tmp = __builtin_epi_vfadd_2xf32(w6_tmp, two_g1_tmp, gvl);
                w6_tmp = __builtin_epi_vfsub_2xf32(w6_tmp, two_g1_tmp, gvl);

                if (rescale_coefficients)
                {
                        float var = -0x1.C71C72p-3f;
                        const __epi_2xf32 minus_2_over_9 = __builtin_epi_vbroadcast_2xf32(var, gvl);

                        w1_tmp = __builtin_epi_vfmul_2xf32(w1_tmp, minus_2_over_9, gvl);
                        w2_tmp = __builtin_epi_vfmul_2xf32(w2_tmp, minus_2_over_9, gvl);
                        float var1 = 0x1.6C16C2p-7f;
                        const __epi_2xf32 rcp_90 = __builtin_epi_vbroadcast_2xf32(var1, gvl);
                        w3_tmp = __builtin_epi_vfmul_2xf32(w3_tmp, rcp_90, gvl);
                        w4_tmp = __builtin_epi_vfmul_2xf32(w4_tmp, rcp_90, gvl);
                        float var2 = 0x1.6C16C2p-8f;
                        const __epi_2xf32 rcp_180 = __builtin_epi_vbroadcast_2xf32(var2, gvl);
                        w5_tmp = __builtin_epi_vfmul_2xf32(w5_tmp, rcp_180, gvl);
                        w6_tmp = __builtin_epi_vfmul_2xf32(w6_tmp, rcp_180, gvl);
                }

                wg00 = row0;
                wg10 = w1_tmp;
                wg20 = w2_tmp;
                wg30 = w3_tmp;
                wg40 = w4_tmp;
                wg50 = w5_tmp;
                wg60 = w6_tmp;
                wg70 = row2;

                __epi_2xf32 w2_tmp1 = __builtin_epi_vfadd_2xf32(row4, row6, gvl);
                __epi_2xf32 w4_tmp1 = __builtin_epi_vfmacc_2xf32(row4, const_4, row6, gvl);
                __epi_2xf32 w6_tmp1 = __builtin_epi_vfmacc_2xf32(row6, const_4, row4, gvl);
                const __epi_2xf32 two_g1_tmp1 = __builtin_epi_vfmul_2xf32(row5, v_two, gvl);
                __epi_2xf32 w1_tmp1 = __builtin_epi_vfadd_2xf32(w2_tmp1, row5, gvl);
                w2_tmp1 = __builtin_epi_vfsub_2xf32(w2_tmp1, row5, gvl);
                // w2 = w2 - g1;
                __epi_2xf32 w3_tmp1 = __builtin_epi_vfadd_2xf32(w4_tmp1, two_g1_tmp1, gvl);
                w4_tmp1 = __builtin_epi_vfsub_2xf32(w4_tmp1, two_g1_tmp1, gvl);
                __epi_2xf32 w5_tmp1 = __builtin_epi_vfadd_2xf32(w6_tmp1, two_g1_tmp1, gvl);
                w6_tmp1 = __builtin_epi_vfsub_2xf32(w6_tmp1, two_g1_tmp1, gvl);

                if (rescale_coefficients)
                {
                        float var = -0x1.C71C72p-3f;
                        const __epi_2xf32 minus_2_over_9 = __builtin_epi_vbroadcast_2xf32(var, gvl);

                        w1_tmp1 = __builtin_epi_vfmul_2xf32(w1_tmp1, minus_2_over_9, gvl);
                        w2_tmp1 = __builtin_epi_vfmul_2xf32(w2_tmp1, minus_2_over_9, gvl);
                        float var1 = 0x1.6C16C2p-7f;
                        const __epi_2xf32 rcp_90 = __builtin_epi_vbroadcast_2xf32(var1, gvl);
                        w3_tmp1 = __builtin_epi_vfmul_2xf32(w3_tmp1, rcp_90, gvl);
                        w4_tmp1 = __builtin_epi_vfmul_2xf32(w4_tmp1, rcp_90, gvl);
                        float var2 = 0x1.6C16C2p-8f;
                        const __epi_2xf32 rcp_180 = __builtin_epi_vbroadcast_2xf32(var2, gvl);
                        w5_tmp1 = __builtin_epi_vfmul_2xf32(w5_tmp1, rcp_180, gvl);
                        w6_tmp1 = __builtin_epi_vfmul_2xf32(w6_tmp1, rcp_180, gvl);
                }

                wg01 = row4;
                wg11 = w1_tmp1;
                wg21 = w2_tmp1;
                wg31 = w3_tmp1;
                wg41 = w4_tmp1;
                wg51 = w5_tmp1;
                wg61 = w6_tmp1;
                wg71 = row6;

                __builtin_epi_vstore_2xf32(&transform[0], wg00, gvl);
                transform += transform_stride;
                __builtin_epi_vstore_2xf32(&transform[0], wg10, gvl);
                transform += transform_stride;
                __builtin_epi_vstore_2xf32(&transform[0], wg20, gvl);
                transform += transform_stride;
                __builtin_epi_vstore_2xf32(&transform[0], wg30, gvl);
                transform += transform_stride;
                __builtin_epi_vstore_2xf32(&transform[0], wg40, gvl);
                transform += transform_stride;
                __builtin_epi_vstore_2xf32(&transform[0], wg50, gvl);
                transform += transform_stride;
                __builtin_epi_vstore_2xf32(&transform[0], wg60, gvl);
                transform += transform_stride;
                __builtin_epi_vstore_2xf32(&transform[0], wg70, gvl);
                transform += transform_stride;
                __builtin_epi_vstore_2xf32(&transform[0], wg01, gvl);
                transform += transform_stride;
                __builtin_epi_vstore_2xf32(&transform[0], wg11, gvl);
                transform += transform_stride;
                __builtin_epi_vstore_2xf32(&transform[0], wg21, gvl);
                transform += transform_stride;
                __builtin_epi_vstore_2xf32(&transform[0], wg31, gvl);
                transform += transform_stride;
                __builtin_epi_vstore_2xf32(&transform[0], wg41, gvl);
                transform += transform_stride;
                __builtin_epi_vstore_2xf32(&transform[0], wg51, gvl);
                transform += transform_stride;
                __builtin_epi_vstore_2xf32(&transform[0], wg61, gvl);
                transform += transform_stride;
                __builtin_epi_vstore_2xf32(&transform[0], wg71, gvl);
                transform += transform_stride;

                i1 += gvl;
        }
        // free(g0);
}

// intertile kernel transform
void nnp_kwt8x8_3x3__neon_intertile(
    const float *g[restrict static 9],
    float *transform[restrict static 1],
    size_t stride_g,
    size_t transform_stride,
    uint32_t row_count,
    uint32_t column_count,
    uint32_t row_offset,
    uint32_t column_offset, size_t interchannels)
{
        int simd_width = interchannels * 4; // nnp_hwinfo.sve_simd_width;//16;//nnp_hwinfo.simd_width;
        //     int interchannels  = nnp_hwinfo.globalinterchannels;//4;

        transform_stride /= sizeof(float);
        float *g0;
        g0 = (float *)malloc(sizeof(float) * 8 * 8 * interchannels);
        if (g0 == NULL)
        {
                fprintf(stderr, "Error in allocating g0 \n");
                exit(-1);
        }
        for (int k = 0; k < interchannels; k++)
        {
                for (int j = 0; j < 4; j++)
                {
                        g0[(0 * simd_width) + (j + (k * 4))] = g[k][j];
                        g0[(1 * simd_width) + (j + (k * 4))] = g[k][3 + j];

                        if (j < 3)
                        {
                                g0[(2 * simd_width) + (j + (k * 4))] = g[k][6 + j];
                        }
                }
                g0[(2 * simd_width) + (3 + (k * 4))] = g[k][5];
        }

        for (int i1 = 0; i1 < simd_width;)
        {
                unsigned long gvl = __builtin_epi_vsetvl(((long)simd_width - (long)i1), __epi_e32, __epi_m1);
                __epi_2xf32 g0_vec = __builtin_epi_vload_2xf32(&g0[0], gvl);
                __epi_2xf32 g1_vec = __builtin_epi_vload_2xf32(&g0[1 * simd_width + 0], gvl);
                // g2[3] is junk
                __epi_2xf32 g2_vec = __builtin_epi_vload_2xf32(&g0[2 * simd_width + 0], gvl);
                __epi_2xf32 transform0, transform1, transform2, transform3, transform4, transform5, transform6, transform7;
                float four = 4.0;
                const __epi_2xf32 const_4 = __builtin_epi_vbroadcast_2xf32(four, gvl);
                __epi_2xf32 w2 = __builtin_epi_vfadd_2xf32(g0_vec, g2_vec, gvl);
                __epi_2xf32 w4 = __builtin_epi_vfmacc_2xf32(g0_vec, const_4, g2_vec, gvl);
                __epi_2xf32 w6 = __builtin_epi_vfmacc_2xf32(g2_vec, const_4, g0_vec, gvl);
                float two = 2.0f;
                __epi_2xf32 v_two = __builtin_epi_vbroadcast_2xf32(two, gvl);
                const __epi_2xf32 two_g1 = __builtin_epi_vfmul_2xf32(g1_vec, v_two, gvl);
                __epi_2xf32 w1 = __builtin_epi_vfadd_2xf32(w2, g1_vec, gvl);
                w2 = __builtin_epi_vfsub_2xf32(w2, g1_vec, gvl);
                // w2 = w2 - g1;
                __epi_2xf32 w3 = __builtin_epi_vfadd_2xf32(w4, two_g1, gvl);
                w4 = __builtin_epi_vfsub_2xf32(w4, two_g1, gvl);
                __epi_2xf32 w5 = __builtin_epi_vfadd_2xf32(w6, two_g1, gvl);
                w6 = __builtin_epi_vfsub_2xf32(w6, two_g1, gvl);

                if (rescale_coefficients)
                {
                        float var = -0x1.C71C72p-3f;
                        const __epi_2xf32 minus_2_over_9 = __builtin_epi_vbroadcast_2xf32(var, gvl);

                        w1 = __builtin_epi_vfmul_2xf32(w1, minus_2_over_9, gvl);
                        w2 = __builtin_epi_vfmul_2xf32(w2, minus_2_over_9, gvl);
                        float var1 = 0x1.6C16C2p-7f;
                        const __epi_2xf32 rcp_90 = __builtin_epi_vbroadcast_2xf32(var1, gvl);
                        w3 = __builtin_epi_vfmul_2xf32(w3, rcp_90, gvl);
                        w4 = __builtin_epi_vfmul_2xf32(w4, rcp_90, gvl);
                        float var2 = 0x1.6C16C2p-8f;
                        const __epi_2xf32 rcp_180 = __builtin_epi_vbroadcast_2xf32(var2, gvl);
                        w5 = __builtin_epi_vfmul_2xf32(w5, rcp_180, gvl);
                        w6 = __builtin_epi_vfmul_2xf32(w6, rcp_180, gvl);
                }

                transform0 = g0_vec;
                transform1 = w1;
                transform2 = w2;
                transform3 = w3;
                transform4 = w4;
                transform5 = w5;
                transform6 = w6;
                transform7 = g2_vec;
                float buff[4 * simd_width], buff1[4 * simd_width];
                __builtin_epi_vstore_2xf32(&buff[0], transform0, gvl);
                __builtin_epi_vstore_2xf32(&buff[simd_width], transform1, gvl);
                __builtin_epi_vstore_2xf32(&buff[2 * simd_width], transform2, gvl);
                __builtin_epi_vstore_2xf32(&buff[3 * simd_width], transform3, gvl);
                __builtin_epi_vstore_2xf32(&buff1[0], transform4, gvl);
                __builtin_epi_vstore_2xf32(&buff1[simd_width], transform5, gvl);
                __builtin_epi_vstore_2xf32(&buff1[2 * simd_width], transform6, gvl);
                __builtin_epi_vstore_2xf32(&buff1[3 * simd_width], transform7, gvl);

                // for(int i=0;i<4;i++)
                //       printf("intermediate = %f \n", buff[i]);

                int ind1 = 0;
                int index1_host[simd_width];
                for (int i = 0; i < interchannels; i++)
                {
                        for (int j = 0; j < 4; j++)
                        {
                                index1_host[ind1] = ((j * simd_width) + 4 * i);
                                ind1++;
                        }
                }
                int one = 1;
                const __epi_2xi32 vone = __builtin_epi_vbroadcast_2xi32(one, gvl);
                __epi_2xi32 index1, index2, index3, index4;
                index1 = __builtin_epi_vload_2xi32(&index1_host[0], gvl);
                index2 = __builtin_epi_vadd_2xi32(index1, vone, gvl);
                index3 = __builtin_epi_vadd_2xi32(index2, vone, gvl);
                index4 = __builtin_epi_vadd_2xi32(index3, vone, gvl);
                int fourv = 4;
                const __epi_2xi32 FOUR = __builtin_epi_vbroadcast_2xi32(fourv, gvl);
                __epi_2xi32 index11 = __builtin_epi_vmul_2xi32(index1, FOUR, gvl);
                __epi_2xi32 index12 = __builtin_epi_vmul_2xi32(index2, FOUR, gvl);
                __epi_2xi32 index13 = __builtin_epi_vmul_2xi32(index3, FOUR, gvl);
                __epi_2xi32 index14 = __builtin_epi_vmul_2xi32(index4, FOUR, gvl);
                __epi_2xf32 row0 = __builtin_epi_vload_indexed_2xf32(&buff[0], index11, gvl);
                __epi_2xf32 row1 = __builtin_epi_vload_indexed_2xf32(&buff[0], index12, gvl);
                __epi_2xf32 row2 = __builtin_epi_vload_indexed_2xf32(&buff[0], index13, gvl);
                __epi_2xf32 row3 = __builtin_epi_vload_indexed_2xf32(&buff[0], index14, gvl);

                __epi_2xf32 row4 = __builtin_epi_vload_indexed_2xf32(&buff1[0], index11, gvl);
                __epi_2xf32 row5 = __builtin_epi_vload_indexed_2xf32(&buff1[0], index12, gvl);
                __epi_2xf32 row6 = __builtin_epi_vload_indexed_2xf32(&buff1[0], index13, gvl);
                __epi_2xf32 row7 = __builtin_epi_vload_indexed_2xf32(&buff1[0], index14, gvl);

                __epi_2xf32 wg00, wg10, wg20, wg30, wg40, wg50, wg60, wg70, wg01, wg11, wg21, wg31, wg41, wg51, wg61, wg71;

                __epi_2xf32 w2_tmp = __builtin_epi_vfadd_2xf32(row0, row2, gvl);
                __epi_2xf32 w4_tmp = __builtin_epi_vfmacc_2xf32(row0, const_4, row2, gvl);
                __epi_2xf32 w6_tmp = __builtin_epi_vfmacc_2xf32(row2, const_4, row0, gvl);
                const __epi_2xf32 two_g1_tmp = __builtin_epi_vfmul_2xf32(row1, v_two, gvl);
                __epi_2xf32 w1_tmp = __builtin_epi_vfadd_2xf32(w2_tmp, row1, gvl);
                w2_tmp = __builtin_epi_vfsub_2xf32(w2_tmp, row1, gvl);
                // w2 = w2 - g1;
                __epi_2xf32 w3_tmp = __builtin_epi_vfadd_2xf32(w4_tmp, two_g1_tmp, gvl);
                w4_tmp = __builtin_epi_vfsub_2xf32(w4_tmp, two_g1_tmp, gvl);
                __epi_2xf32 w5_tmp = __builtin_epi_vfadd_2xf32(w6_tmp, two_g1_tmp, gvl);
                w6_tmp = __builtin_epi_vfsub_2xf32(w6_tmp, two_g1_tmp, gvl);

                if (rescale_coefficients)
                {
                        float var = -0x1.C71C72p-3f;
                        const __epi_2xf32 minus_2_over_9 = __builtin_epi_vbroadcast_2xf32(var, gvl);

                        w1_tmp = __builtin_epi_vfmul_2xf32(w1_tmp, minus_2_over_9, gvl);
                        w2_tmp = __builtin_epi_vfmul_2xf32(w2_tmp, minus_2_over_9, gvl);
                        float var1 = 0x1.6C16C2p-7f;
                        const __epi_2xf32 rcp_90 = __builtin_epi_vbroadcast_2xf32(var1, gvl);
                        w3_tmp = __builtin_epi_vfmul_2xf32(w3_tmp, rcp_90, gvl);
                        w4_tmp = __builtin_epi_vfmul_2xf32(w4_tmp, rcp_90, gvl);
                        float var2 = 0x1.6C16C2p-8f;
                        const __epi_2xf32 rcp_180 = __builtin_epi_vbroadcast_2xf32(var2, gvl);
                        w5_tmp = __builtin_epi_vfmul_2xf32(w5_tmp, rcp_180, gvl);
                        w6_tmp = __builtin_epi_vfmul_2xf32(w6_tmp, rcp_180, gvl);
                }

                wg00 = row0;
                wg10 = w1_tmp;
                wg20 = w2_tmp;
                wg30 = w3_tmp;
                wg40 = w4_tmp;
                wg50 = w5_tmp;
                wg60 = w6_tmp;
                wg70 = row2;

                __epi_2xf32 w2_tmp1 = __builtin_epi_vfadd_2xf32(row4, row6, gvl);
                __epi_2xf32 w4_tmp1 = __builtin_epi_vfmacc_2xf32(row4, const_4, row6, gvl);
                __epi_2xf32 w6_tmp1 = __builtin_epi_vfmacc_2xf32(row6, const_4, row4, gvl);
                const __epi_2xf32 two_g1_tmp1 = __builtin_epi_vfmul_2xf32(row5, v_two, gvl);
                __epi_2xf32 w1_tmp1 = __builtin_epi_vfadd_2xf32(w2_tmp1, row5, gvl);
                w2_tmp1 = __builtin_epi_vfsub_2xf32(w2_tmp1, row5, gvl);
                // w2 = w2 - g1;
                __epi_2xf32 w3_tmp1 = __builtin_epi_vfadd_2xf32(w4_tmp1, two_g1_tmp1, gvl);
                w4_tmp1 = __builtin_epi_vfsub_2xf32(w4_tmp1, two_g1_tmp1, gvl);
                __epi_2xf32 w5_tmp1 = __builtin_epi_vfadd_2xf32(w6_tmp1, two_g1_tmp1, gvl);
                w6_tmp1 = __builtin_epi_vfsub_2xf32(w6_tmp1, two_g1_tmp1, gvl);

                if (rescale_coefficients)
                {
                        float var = -0x1.C71C72p-3f;
                        const __epi_2xf32 minus_2_over_9 = __builtin_epi_vbroadcast_2xf32(var, gvl);

                        w1_tmp1 = __builtin_epi_vfmul_2xf32(w1_tmp1, minus_2_over_9, gvl);
                        w2_tmp1 = __builtin_epi_vfmul_2xf32(w2_tmp1, minus_2_over_9, gvl);
                        float var1 = 0x1.6C16C2p-7f;
                        const __epi_2xf32 rcp_90 = __builtin_epi_vbroadcast_2xf32(var1, gvl);
                        w3_tmp1 = __builtin_epi_vfmul_2xf32(w3_tmp1, rcp_90, gvl);
                        w4_tmp1 = __builtin_epi_vfmul_2xf32(w4_tmp1, rcp_90, gvl);
                        float var2 = 0x1.6C16C2p-8f;
                        const __epi_2xf32 rcp_180 = __builtin_epi_vbroadcast_2xf32(var2, gvl);
                        w5_tmp1 = __builtin_epi_vfmul_2xf32(w5_tmp1, rcp_180, gvl);
                        w6_tmp1 = __builtin_epi_vfmul_2xf32(w6_tmp1, rcp_180, gvl);
                }

                wg01 = row4;
                wg11 = w1_tmp1;
                wg21 = w2_tmp1;
                wg31 = w3_tmp1;
                wg41 = w4_tmp1;
                wg51 = w5_tmp1;
                wg61 = w6_tmp1;
                wg71 = row6;

                float tmp_transform[16 * simd_width];
                __builtin_epi_vstore_2xf32(&tmp_transform[0], wg00, gvl);
                __builtin_epi_vstore_2xf32(&tmp_transform[1 * simd_width], wg10, gvl);
                __builtin_epi_vstore_2xf32(&tmp_transform[2 * simd_width], wg20, gvl);
                __builtin_epi_vstore_2xf32(&tmp_transform[3 * simd_width], wg30, gvl);
                __builtin_epi_vstore_2xf32(&tmp_transform[4 * simd_width], wg40, gvl);
                __builtin_epi_vstore_2xf32(&tmp_transform[5 * simd_width], wg50, gvl);
                __builtin_epi_vstore_2xf32(&tmp_transform[6 * simd_width], wg60, gvl);
                __builtin_epi_vstore_2xf32(&tmp_transform[7 * simd_width], wg70, gvl);
                __builtin_epi_vstore_2xf32(&tmp_transform[8 * simd_width], wg01, gvl);
                __builtin_epi_vstore_2xf32(&tmp_transform[9 * simd_width], wg11, gvl);
                __builtin_epi_vstore_2xf32(&tmp_transform[10 * simd_width], wg21, gvl);
                __builtin_epi_vstore_2xf32(&tmp_transform[11 * simd_width], wg31, gvl);
                __builtin_epi_vstore_2xf32(&tmp_transform[12 * simd_width], wg41, gvl);
                __builtin_epi_vstore_2xf32(&tmp_transform[13 * simd_width], wg51, gvl);
                __builtin_epi_vstore_2xf32(&tmp_transform[14 * simd_width], wg61, gvl);
                __builtin_epi_vstore_2xf32(&tmp_transform[15 * simd_width], wg71, gvl);
                for (size_t i = 0; i < 16; i++)
                {
#pragma loop unroll_count(interchannels)
                        for (int k = 0; k < interchannels; k++)
                        {
                                for (size_t j = 0; j < 4; j++)
                                {
                                        *(transform[k] + j) = tmp_transform[i * simd_width + (j + k * 4)];
                                }
                                transform[k] += transform_stride;
                        }
                }
                i1 += gvl;
        }
        if (g0 != NULL)
        {
                free(g0);
                g0 = NULL;
        }
}

#if !NNP_INFERENCE_ONLY
void nnp_kwt8x8_3Rx3R__neon(
    const float g[restrict static 9],
    float transform[restrict static 1],
    size_t stride_g,
    size_t transform_stride,
    int row_count,
    int column_count,
    int row_offset,
    int column_offset)
{
}

void nnp_owt8x8_3x3__neon(
    const void *restrict transform,
    float output[restrict static 1],
    size_t transform_stride,
    size_t output_stride,
    int row_count,
    int column_count,
    int row_offset,
    int column_offset)
{
        printf("hi");
}
#endif /* !NNP_INFERENCE_ONLY */

void nnp_owt8x8_3x3_with_bias__neon(
    void *restrict transform,
    float output[restrict static 1],
    const float bias[restrict static 1],
    size_t transform_stride,
    size_t output_stride,
    uint32_t row_count,
    uint32_t column_count)
{
        printf("hi");
}

void nnp_owt8x8_3x3_with_bias__neon_intertile(
    const void **restrict transform,
    float *output[restrict static 1],
    const float bias[restrict static 1],
    size_t transform_stride,
    size_t output_stride,
    uint32_t row_count,
    uint32_t column_count, size_t interchannels)
{
        int simd_width = interchannels * 4; // nnp_hwinfo.sve_simd_width;//nnp_hwinfo.simd_width;
                                            //      int interchannels  = nnp_hwinfo.globalinterchannels;

        float *new_data;
        new_data = (float *)malloc(sizeof(float) * 8 * 8 * 8 * interchannels);
        if (new_data == NULL)
        {
                fprintf(stderr, "Error in allocating new_data \n");
                exit(-1);
        }

        for (int i = 0; i < 16; i++)
        {
#pragma loop unroll_count(interchannels)
                for (int k = 0; k < interchannels; k++)
                {
                        for (int j = 0; j < 4; j++)
                        {
                                new_data[(i * simd_width) + (j + k * 4)] = *(((float *)transform[k]) + j);
                        }
                        transform[k] += transform_stride;
                }
        }

        for (int i1 = 0; i1 < simd_width;)
        {
                unsigned long gvl = __builtin_epi_vsetvl(((long)simd_width - (long)i1), __epi_e32, __epi_m1);

                __epi_2xf32 m0, m1, m2, m3, m4, m5, m6, m7;

                // first iteration
                m0 = __builtin_epi_vload_2xf32(&new_data[0 * simd_width], gvl);
                m1 = __builtin_epi_vload_2xf32(&new_data[1 * simd_width], gvl);
                m2 = __builtin_epi_vload_2xf32(&new_data[2 * simd_width], gvl);
                m3 = __builtin_epi_vload_2xf32(&new_data[3 * simd_width], gvl);
                m4 = __builtin_epi_vload_2xf32(&new_data[4 * simd_width], gvl);
                m5 = __builtin_epi_vload_2xf32(&new_data[5 * simd_width], gvl);
                m6 = __builtin_epi_vload_2xf32(&new_data[6 * simd_width], gvl);
                m7 = __builtin_epi_vload_2xf32(&new_data[7 * simd_width], gvl);
                __epi_2xf32 o0, o1, o2, o3, o4, o5;

                __epi_2xf32 m1_add_m2 = __builtin_epi_vfadd_2xf32(m1, m2, gvl);
                __epi_2xf32 m1_sub_m2 = __builtin_epi_vfsub_2xf32(m1, m2, gvl);
                __epi_2xf32 m3_add_m4 = __builtin_epi_vfadd_2xf32(m3, m4, gvl);
                __epi_2xf32 m3_sub_m4 = __builtin_epi_vfsub_2xf32(m3, m4, gvl);
                __epi_2xf32 m5_add_m6 = __builtin_epi_vfadd_2xf32(m5, m6, gvl);
                __epi_2xf32 m5_sub_m6 = __builtin_epi_vfsub_2xf32(m5, m6, gvl);

                __epi_2xf32 s0 = __builtin_epi_vfadd_2xf32(m0, m1_add_m2, gvl);
                __epi_2xf32 s5 = __builtin_epi_vfadd_2xf32(m7, m1_sub_m2, gvl);

                float var = 16.0f;
                float var1 = 8.0f;
                __epi_2xf32 const_16 = __builtin_epi_vbroadcast_2xf32(var, gvl);
                __epi_2xf32 const_8 = __builtin_epi_vbroadcast_2xf32(var1, gvl);

                __epi_2xf32 s1 = __builtin_epi_vfmacc_2xf32(m1_sub_m2, m5_sub_m6, const_16, gvl);
                __epi_2xf32 s4 = __builtin_epi_vfmacc_2xf32(m1_add_m2, m3_add_m4, const_16, gvl);
                __epi_2xf32 s2 = __builtin_epi_vfmacc_2xf32(m1_add_m2, m5_add_m6, const_8, gvl);
                __epi_2xf32 s3 = __builtin_epi_vfmacc_2xf32(m1_sub_m2, m3_sub_m4, const_8, gvl);

                float var2 = 32.0f, var3 = 2.0f;
                __epi_2xf32 const_32 = __builtin_epi_vbroadcast_2xf32(var2, gvl);
                __epi_2xf32 const_2 = __builtin_epi_vbroadcast_2xf32(var3, gvl);

                s0 = __builtin_epi_vfmacc_2xf32(s0, m5_add_m6, const_32, gvl);
                s5 = __builtin_epi_vfmacc_2xf32(s5, m3_sub_m4, const_32, gvl);
                s1 = __builtin_epi_vfmacc_2xf32(s1, m3_sub_m4, const_2, gvl);
                s4 = __builtin_epi_vfmacc_2xf32(s4, m5_add_m6, const_2, gvl);

                s0 = __builtin_epi_vfadd_2xf32(s0, m3_add_m4, gvl);
                s5 = __builtin_epi_vfadd_2xf32(s5, m5_sub_m6, gvl);
                float var4 = 4.0f;
                const __epi_2xf32 const_4 = __builtin_epi_vbroadcast_2xf32(var4, gvl);
                s2 = __builtin_epi_vfmacc_2xf32(s2, m3_add_m4, const_4, gvl);
                s3 = __builtin_epi_vfmacc_2xf32(s3, m5_sub_m6, const_4, gvl);
                o0 = s0;
                o1 = s1;
                o2 = s2;
                o3 = s3;
                o4 = s4;
                o5 = s5;

                // second iteration
                __epi_2xf32 o6, o7, o8, o9, o10, o11;
                m0 = __builtin_epi_vload_2xf32(&new_data[8 * simd_width], gvl);
                m1 = __builtin_epi_vload_2xf32(&new_data[9 * simd_width], gvl);
                m2 = __builtin_epi_vload_2xf32(&new_data[10 * simd_width], gvl);
                m3 = __builtin_epi_vload_2xf32(&new_data[11 * simd_width], gvl);
                m4 = __builtin_epi_vload_2xf32(&new_data[12 * simd_width], gvl);
                m5 = __builtin_epi_vload_2xf32(&new_data[13 * simd_width], gvl);
                m6 = __builtin_epi_vload_2xf32(&new_data[14 * simd_width], gvl);
                m7 = __builtin_epi_vload_2xf32(&new_data[15 * simd_width], gvl);

                m1_add_m2 = __builtin_epi_vfadd_2xf32(m1, m2, gvl);
                m1_sub_m2 = __builtin_epi_vfsub_2xf32(m1, m2, gvl);
                m3_add_m4 = __builtin_epi_vfadd_2xf32(m3, m4, gvl);
                m3_sub_m4 = __builtin_epi_vfsub_2xf32(m3, m4, gvl);
                m5_add_m6 = __builtin_epi_vfadd_2xf32(m5, m6, gvl);
                m5_sub_m6 = __builtin_epi_vfsub_2xf32(m5, m6, gvl);

                s0 = __builtin_epi_vfadd_2xf32(m0, m1_add_m2, gvl);
                s5 = __builtin_epi_vfadd_2xf32(m7, m1_sub_m2, gvl);

                s1 = __builtin_epi_vfmacc_2xf32(m1_sub_m2, m5_sub_m6, const_16, gvl);
                s4 = __builtin_epi_vfmacc_2xf32(m1_add_m2, m3_add_m4, const_16, gvl);
                s2 = __builtin_epi_vfmacc_2xf32(m1_add_m2, m5_add_m6, const_8, gvl);
                s3 = __builtin_epi_vfmacc_2xf32(m1_sub_m2, m3_sub_m4, const_8, gvl);

                s0 = __builtin_epi_vfmacc_2xf32(s0, m5_add_m6, const_32, gvl);
                s5 = __builtin_epi_vfmacc_2xf32(s5, m3_sub_m4, const_32, gvl);
                s1 = __builtin_epi_vfmacc_2xf32(s1, m3_sub_m4, const_2, gvl);
                s4 = __builtin_epi_vfmacc_2xf32(s4, m5_add_m6, const_2, gvl);

                s0 = __builtin_epi_vfadd_2xf32(s0, m3_add_m4, gvl);
                s5 = __builtin_epi_vfadd_2xf32(s5, m5_sub_m6, gvl);
                s2 = __builtin_epi_vfmacc_2xf32(s2, m3_add_m4, const_4, gvl);
                s3 = __builtin_epi_vfmacc_2xf32(s3, m5_sub_m6, const_4, gvl);
                o6 = s0;
                o7 = s1;
                o8 = s2;
                o9 = s3;
                o10 = s4;
                o11 = s5;

                if (row_count == 6 && column_count == 6 && output_stride >= 6)
                {

                        float vin4567[4 * simd_width], vin45671[4 * simd_width];
                        __builtin_epi_vstore_2xf32(&vin4567[0], o0, gvl);
                        __builtin_epi_vstore_2xf32(&vin4567[simd_width], o1, gvl);
                        __builtin_epi_vstore_2xf32(&vin4567[2 * simd_width], o2, gvl);
                        __builtin_epi_vstore_2xf32(&vin4567[3 * simd_width], o3, gvl);
                        __builtin_epi_vstore_2xf32(&vin45671[0], o6, gvl);
                        __builtin_epi_vstore_2xf32(&vin45671[simd_width], o7, gvl);
                        __builtin_epi_vstore_2xf32(&vin45671[2 * simd_width], o8, gvl);
                        __builtin_epi_vstore_2xf32(&vin45671[3 * simd_width], o9, gvl);

                        __epi_2xf32 a0, a1, a2, a3, a4, a5, a6, a7;

                        int ind1 = 0;
                        int index1_host[simd_width];
                        for (int i = 0; i < interchannels; i++)
                        {
                                for (int j = 0; j < 4; j++)
                                {
                                        index1_host[ind1] = ((j * simd_width) + 4 * i);
                                        ind1++;
                                }
                        }
                        int one = 1;
                        const __epi_2xi32 vone = __builtin_epi_vbroadcast_2xi32(one, gvl);
                        int four = 4;
                        const __epi_2xi32 FOUR = __builtin_epi_vbroadcast_2xi32(four, gvl);
                        __epi_2xi32 index1, index2, index3, index4;
                        index1 = __builtin_epi_vload_2xi32(&index1_host[0], gvl);
                        index2 = __builtin_epi_vadd_2xi32(index1, vone, gvl);
                        index3 = __builtin_epi_vadd_2xi32(index2, vone, gvl);
                        index4 = __builtin_epi_vadd_2xi32(index3, vone, gvl);
                        __epi_2xi32 index11 = __builtin_epi_vmul_2xi32(index1, FOUR, gvl);
                        __epi_2xi32 index12 = __builtin_epi_vmul_2xi32(index2, FOUR, gvl);
                        __epi_2xi32 index13 = __builtin_epi_vmul_2xi32(index3, FOUR, gvl);
                        __epi_2xi32 index14 = __builtin_epi_vmul_2xi32(index4, FOUR, gvl);

                        __epi_2xf32 qout0, qout1, qout2, qout3, qout4, qout5;
                        __epi_2xf32 d0 = __builtin_epi_vload_indexed_2xf32(&vin4567[0], index11, gvl);
                        __epi_2xf32 d1 = __builtin_epi_vload_indexed_2xf32(&vin4567[0], index12, gvl);
                        __epi_2xf32 d2 = __builtin_epi_vload_indexed_2xf32(&vin4567[0], index13, gvl);
                        __epi_2xf32 d3 = __builtin_epi_vload_indexed_2xf32(&vin4567[0], index14, gvl);
                        __epi_2xf32 d4 = __builtin_epi_vload_indexed_2xf32(&vin45671[0], index11, gvl);
                        __epi_2xf32 d5 = __builtin_epi_vload_indexed_2xf32(&vin45671[0], index12, gvl);
                        __epi_2xf32 d6 = __builtin_epi_vload_indexed_2xf32(&vin45671[0], index13, gvl);
                        __epi_2xf32 d7 = __builtin_epi_vload_indexed_2xf32(&vin45671[0], index14, gvl);
                        __epi_2xf32 vout0, vout1, vout2, vout3, vout4, vout5, vout6, vout7;

                        m1_add_m2 = __builtin_epi_vfadd_2xf32(d1, d2, gvl);
                        m1_sub_m2 = __builtin_epi_vfsub_2xf32(d1, d2, gvl);
                        m3_add_m4 = __builtin_epi_vfadd_2xf32(d3, d4, gvl);
                        m3_sub_m4 = __builtin_epi_vfsub_2xf32(d3, d4, gvl);
                        m5_add_m6 = __builtin_epi_vfadd_2xf32(d5, d6, gvl);
                        m5_sub_m6 = __builtin_epi_vfsub_2xf32(d5, d6, gvl);

                        s0 = __builtin_epi_vfadd_2xf32(d0, m1_add_m2, gvl);
                        s5 = __builtin_epi_vfadd_2xf32(d7, m1_sub_m2, gvl);

                        s1 = __builtin_epi_vfmacc_2xf32(m1_sub_m2, m5_sub_m6, const_16, gvl);
                        s4 = __builtin_epi_vfmacc_2xf32(m1_add_m2, m3_add_m4, const_16, gvl);
                        s2 = __builtin_epi_vfmacc_2xf32(m1_add_m2, m5_add_m6, const_8, gvl);
                        s3 = __builtin_epi_vfmacc_2xf32(m1_sub_m2, m3_sub_m4, const_8, gvl);

                        s0 = __builtin_epi_vfmacc_2xf32(s0, m5_add_m6, const_32, gvl);
                        s5 = __builtin_epi_vfmacc_2xf32(s5, m3_sub_m4, const_32, gvl);
                        s1 = __builtin_epi_vfmacc_2xf32(s1, m3_sub_m4, const_2, gvl);
                        s4 = __builtin_epi_vfmacc_2xf32(s4, m5_add_m6, const_2, gvl);

                        s0 = __builtin_epi_vfadd_2xf32(s0, m3_add_m4, gvl);
                        s5 = __builtin_epi_vfadd_2xf32(s5, m5_sub_m6, gvl);
                        s2 = __builtin_epi_vfmacc_2xf32(s2, m3_add_m4, const_4, gvl);
                        s3 = __builtin_epi_vfmacc_2xf32(s3, m5_sub_m6, const_4, gvl);
                        qout0 = s0;
                        qout1 = s1;
                        qout2 = s2;
                        qout3 = s3;
                        qout4 = s4;
                        qout5 = s5;
                        float output_ptr[16 * simd_width];
                        __builtin_epi_vstore_2xf32(&output_ptr[0], qout0, gvl);
                        __builtin_epi_vstore_2xf32(&output_ptr[simd_width], qout1, gvl);
                        __builtin_epi_vstore_2xf32(&output_ptr[2 * simd_width], qout2, gvl);
                        __builtin_epi_vstore_2xf32(&output_ptr[3 * simd_width], qout3, gvl);
                        __builtin_epi_vstore_2xf32(&output_ptr[4 * simd_width], qout4, gvl);
                        __builtin_epi_vstore_2xf32(&output_ptr[5 * simd_width], qout5, gvl);

                        unsigned long gvl_new = __builtin_epi_vsetvl(((long)simd_width / 2 - (long)i1), __epi_e32, __epi_m1);

                        float vin1234[2 * simd_width], vin12341[2 * simd_width];

                        __builtin_epi_vstore_2xf32(&vin1234[0], o4, gvl);
                        __builtin_epi_vstore_2xf32(&vin1234[simd_width], o5, gvl);
                        __builtin_epi_vstore_2xf32(&vin12341[0], o10, gvl);
                        __builtin_epi_vstore_2xf32(&vin12341[simd_width], o11, gvl);
                        // printf("upto here");

                        int ind11 = 0;
                        int index11_host[simd_width / 2];
                        for (int i = 0; i < interchannels; i++)
                        {
                                for (int j = 0; j < 2; j++)
                                {
                                        index11_host[ind11] = ((j * simd_width) + 2 * 2 * i);
                                        ind11++;
                                }
                        }
                        __epi_2xi32 index1_new, index2_new, index3_new, index4_new;
                        index1_new = __builtin_epi_vload_2xi32(&index11_host[0], gvl_new);
                        index2_new = __builtin_epi_vadd_2xi32(index1_new, vone, gvl_new);
                        index3_new = __builtin_epi_vadd_2xi32(index2_new, vone, gvl_new);
                        index4_new = __builtin_epi_vadd_2xi32(index3_new, vone, gvl_new);
                        __epi_2xi32 index11_new = __builtin_epi_vmul_2xi32(index1_new, FOUR, gvl_new);
                        __epi_2xi32 index12_new = __builtin_epi_vmul_2xi32(index2_new, FOUR, gvl_new);
                        __epi_2xi32 index13_new = __builtin_epi_vmul_2xi32(index3_new, FOUR, gvl_new);
                        __epi_2xi32 index14_new = __builtin_epi_vmul_2xi32(index4_new, FOUR, gvl_new);

                        __epi_2xf32 d0_new = __builtin_epi_vload_indexed_2xf32(&vin1234[0], index11_new, gvl_new);
                        __epi_2xf32 d1_new = __builtin_epi_vload_indexed_2xf32(&vin1234[0], index12_new, gvl_new);
                        __epi_2xf32 d2_new = __builtin_epi_vload_indexed_2xf32(&vin1234[0], index13_new, gvl_new);
                        __epi_2xf32 d3_new = __builtin_epi_vload_indexed_2xf32(&vin1234[0], index14_new, gvl_new);
                        __epi_2xf32 d4_new = __builtin_epi_vload_indexed_2xf32(&vin12341[0], index11_new, gvl_new);
                        __epi_2xf32 d5_new = __builtin_epi_vload_indexed_2xf32(&vin12341[0], index12_new, gvl_new);
                        __epi_2xf32 d6_new = __builtin_epi_vload_indexed_2xf32(&vin12341[0], index13_new, gvl_new);
                        __epi_2xf32 d7_new = __builtin_epi_vload_indexed_2xf32(&vin12341[0], index14_new, gvl_new);

                        //////stop here
                        __epi_2xf32 dout0, dout1, dout2, dout3, dout4, dout5;

                        const __epi_2xf32 m1_add_m2_new = __builtin_epi_vfadd_2xf32(d1_new, d2_new, gvl_new);
                        const __epi_2xf32 m1_sub_m2_new = __builtin_epi_vfsub_2xf32(d1_new, d2_new, gvl_new);
                        const __epi_2xf32 m3_add_m4_new = __builtin_epi_vfadd_2xf32(d3_new, d4_new, gvl_new);
                        const __epi_2xf32 m3_sub_m4_new = __builtin_epi_vfsub_2xf32(d3_new, d4_new, gvl_new);
                        const __epi_2xf32 m5_add_m6_new = __builtin_epi_vfadd_2xf32(d5_new, d6_new, gvl_new);
                        const __epi_2xf32 m5_sub_m6_new = __builtin_epi_vfsub_2xf32(d5_new, d6_new, gvl_new);

                        __epi_2xf32 s0_new = __builtin_epi_vfadd_2xf32(d0_new, m1_add_m2_new, gvl_new);
                        __epi_2xf32 s5_new = __builtin_epi_vfadd_2xf32(d7_new, m1_sub_m2_new, gvl_new);

                        __epi_2xf32 s1_new = __builtin_epi_vfmacc_2xf32(m1_sub_m2_new, m5_sub_m6_new, const_16, gvl_new);
                        __epi_2xf32 s4_new = __builtin_epi_vfmacc_2xf32(m1_add_m2_new, m3_add_m4_new, const_16, gvl_new);
                        __epi_2xf32 s2_new = __builtin_epi_vfmacc_2xf32(m1_add_m2_new, m5_add_m6_new, const_8, gvl_new);
                        __epi_2xf32 s3_new = __builtin_epi_vfmacc_2xf32(m1_sub_m2_new, m3_sub_m4_new, const_8, gvl_new);

                        s0_new = __builtin_epi_vfmacc_2xf32(s0_new, m5_add_m6_new, const_32, gvl_new);
                        s5_new = __builtin_epi_vfmacc_2xf32(s5_new, m3_sub_m4_new, const_32, gvl_new);
                        s1_new = __builtin_epi_vfmacc_2xf32(s1_new, m3_sub_m4_new, const_2, gvl_new);
                        s4_new = __builtin_epi_vfmacc_2xf32(s4_new, m5_add_m6_new, const_2, gvl_new);

                        s0_new = __builtin_epi_vfadd_2xf32(s0_new, m3_add_m4_new, gvl_new);
                        s5_new = __builtin_epi_vfadd_2xf32(s5_new, m5_sub_m6_new, gvl_new);
                        s2_new = __builtin_epi_vfmacc_2xf32(s2_new, m3_add_m4_new, const_4, gvl_new);
                        s3_new = __builtin_epi_vfmacc_2xf32(s3_new, m5_sub_m6_new, const_4, gvl_new);
                        dout0 = s0_new;
                        dout1 = s1_new;
                        dout2 = s2_new;
                        dout3 = s3_new;
                        dout4 = s4_new;
                        dout5 = s5_new;

                        float output_col45ptr[16 * simd_width];
                        __builtin_epi_vstore_2xf32(&output_col45ptr[0], dout0, gvl_new);
                        __builtin_epi_vstore_2xf32(&output_col45ptr[simd_width / 2], dout1, gvl_new);
                        __builtin_epi_vstore_2xf32(&output_col45ptr[2 * (simd_width / 2)], dout2, gvl_new);
                        __builtin_epi_vstore_2xf32(&output_col45ptr[3 * (simd_width / 2)], dout3, gvl_new);
                        __builtin_epi_vstore_2xf32(&output_col45ptr[4 * (simd_width / 2)], dout4, gvl_new);
                        __builtin_epi_vstore_2xf32(&output_col45ptr[5 * (simd_width / 2)], dout5, gvl_new);

                        float *output_col0123[interchannels];
                        float *output_col45[interchannels];
                        for (int k = 0; k < interchannels; k++)
                        {
                                output_col0123[k] = output[k];
                                output_col45[k] = output[k] + 4;
                        }

                        for (size_t i = 0; i < 6; i++)
                        {
#pragma loop unroll_count(interchannels)
                                for (int k = 0; k < interchannels; k++)
                                {
                                        for (size_t j = 0; j < 4; j++)
                                        {
                                                output_col0123[k][j] = output_ptr[i * simd_width + (j + k * 4)];
                                        }
                                        if (i < 5)
                                        {
                                                output_col0123[k] += output_stride;
                                        }
                                }
                        }

                        for (size_t i = 0; i < 6; i++)
                        {
#pragma loop unroll_count(interchannels)
                                for (int k = 0; k < interchannels; k++)
                                {
                                        for (size_t j = 0; j < 2; j++)
                                        {
                                                output_col45[k][j] = output_col45ptr[i * (simd_width / 2) + (j + k * 2)];
                                        }
                                        if (i < 5)
                                        {
                                                output_col45[k] += output_stride;
                                        }
                                }
                        }
                        //          printf("sve\n");
                        // for(int i=0;i<48;i++)
                        //          printf("output_col45[0]= %f\n", output[0][i]);
                        //    printf("sve\n");
                        //  for(int i=0;i<48;i++)
                        //           printf("output_col45[0]= %f\n", output[1][i]);
                }
                else
                {
                        float block[6 * simd_width];
                        float block1[6 * simd_width];
                        float vin4567[4 * simd_width], vin45671[4 * simd_width];
                        __builtin_epi_vstore_2xf32(&vin4567[0], o0, gvl);
                        __builtin_epi_vstore_2xf32(&vin4567[simd_width], o1, gvl);
                        __builtin_epi_vstore_2xf32(&vin4567[2 * simd_width], o2, gvl);
                        __builtin_epi_vstore_2xf32(&vin4567[3 * simd_width], o3, gvl);
                        __builtin_epi_vstore_2xf32(&vin45671[0], o6, gvl);
                        __builtin_epi_vstore_2xf32(&vin45671[simd_width], o7, gvl);
                        __builtin_epi_vstore_2xf32(&vin45671[2 * simd_width], o8, gvl);
                        __builtin_epi_vstore_2xf32(&vin45671[3 * simd_width], o9, gvl);

                        __epi_2xf32 a0, a1, a2, a3, a4, a5, a6, a7;

                        int ind1 = 0;
                        int index1_host[simd_width];
                        for (int i = 0; i < interchannels; i++)
                        {
                                for (int j = 0; j < 4; j++)
                                {
                                        index1_host[ind1] = ((j * simd_width) + 4 * i);
                                        ind1++;
                                }
                        }
                        int one = 1;
                        const __epi_2xi32 vone = __builtin_epi_vbroadcast_2xi32(one, gvl);
                        int four = 4;
                        const __epi_2xi32 FOUR = __builtin_epi_vbroadcast_2xi32(four, gvl);
                        __epi_2xi32 index1, index2, index3, index4;
                        index1 = __builtin_epi_vload_2xi32(&index1_host[0], gvl);
                        index2 = __builtin_epi_vadd_2xi32(index1, vone, gvl);
                        index3 = __builtin_epi_vadd_2xi32(index2, vone, gvl);
                        index4 = __builtin_epi_vadd_2xi32(index3, vone, gvl);
                        __epi_2xi32 index11 = __builtin_epi_vmul_2xi32(index1, FOUR, gvl);
                        __epi_2xi32 index12 = __builtin_epi_vmul_2xi32(index2, FOUR, gvl);
                        __epi_2xi32 index13 = __builtin_epi_vmul_2xi32(index3, FOUR, gvl);
                        __epi_2xi32 index14 = __builtin_epi_vmul_2xi32(index4, FOUR, gvl);
                        __epi_2xf32 qout0, qout1, qout2, qout3, qout4, qout5;
                        __epi_2xf32 d0 = __builtin_epi_vload_indexed_2xf32(&vin4567[0], index11, gvl);
                        __epi_2xf32 d1 = __builtin_epi_vload_indexed_2xf32(&vin4567[0], index12, gvl);
                        __epi_2xf32 d2 = __builtin_epi_vload_indexed_2xf32(&vin4567[0], index13, gvl);
                        __epi_2xf32 d3 = __builtin_epi_vload_indexed_2xf32(&vin4567[0], index14, gvl);
                        __epi_2xf32 d4 = __builtin_epi_vload_indexed_2xf32(&vin45671[0], index11, gvl);
                        __epi_2xf32 d5 = __builtin_epi_vload_indexed_2xf32(&vin45671[0], index12, gvl);
                        __epi_2xf32 d6 = __builtin_epi_vload_indexed_2xf32(&vin45671[0], index13, gvl);
                        __epi_2xf32 d7 = __builtin_epi_vload_indexed_2xf32(&vin45671[0], index14, gvl);
                        __epi_2xf32 vout0, vout1, vout2, vout3, vout4, vout5, vout6, vout7;

                        m1_add_m2 = __builtin_epi_vfadd_2xf32(d1, d2, gvl);
                        m1_sub_m2 = __builtin_epi_vfsub_2xf32(d1, d2, gvl);
                        m3_add_m4 = __builtin_epi_vfadd_2xf32(d3, d4, gvl);
                        m3_sub_m4 = __builtin_epi_vfsub_2xf32(d3, d4, gvl);
                        m5_add_m6 = __builtin_epi_vfadd_2xf32(d5, d6, gvl);
                        m5_sub_m6 = __builtin_epi_vfsub_2xf32(d5, d6, gvl);

                        s0 = __builtin_epi_vfadd_2xf32(d0, m1_add_m2, gvl);
                        s5 = __builtin_epi_vfadd_2xf32(d7, m1_sub_m2, gvl);

                        s1 = __builtin_epi_vfmacc_2xf32(m1_sub_m2, m5_sub_m6, const_16, gvl);
                        s4 = __builtin_epi_vfmacc_2xf32(m1_add_m2, m3_add_m4, const_16, gvl);
                        s2 = __builtin_epi_vfmacc_2xf32(m1_add_m2, m5_add_m6, const_8, gvl);
                        s3 = __builtin_epi_vfmacc_2xf32(m1_sub_m2, m3_sub_m4, const_8, gvl);

                        s0 = __builtin_epi_vfmacc_2xf32(s0, m5_add_m6, const_32, gvl);
                        s5 = __builtin_epi_vfmacc_2xf32(s5, m3_sub_m4, const_32, gvl);
                        s1 = __builtin_epi_vfmacc_2xf32(s1, m3_sub_m4, const_2, gvl);
                        s4 = __builtin_epi_vfmacc_2xf32(s4, m5_add_m6, const_2, gvl);

                        s0 = __builtin_epi_vfadd_2xf32(s0, m3_add_m4, gvl);
                        s5 = __builtin_epi_vfadd_2xf32(s5, m5_sub_m6, gvl);
                        s2 = __builtin_epi_vfmacc_2xf32(s2, m3_add_m4, const_4, gvl);
                        s3 = __builtin_epi_vfmacc_2xf32(s3, m5_sub_m6, const_4, gvl);
                        qout0 = s0;
                        qout1 = s1;
                        qout2 = s2;
                        qout3 = s3;
                        qout4 = s4;
                        qout5 = s5;

                        __builtin_epi_vstore_2xf32(&block[0], qout0, gvl);
                        __builtin_epi_vstore_2xf32(&block[simd_width], qout1, gvl);
                        __builtin_epi_vstore_2xf32(&block[2 * simd_width], qout2, gvl);
                        __builtin_epi_vstore_2xf32(&block[3 * simd_width], qout3, gvl);
                        __builtin_epi_vstore_2xf32(&block[4 * simd_width], qout4, gvl);
                        __builtin_epi_vstore_2xf32(&block[5 * simd_width], qout5, gvl);

                        unsigned long gvl_new = __builtin_epi_vsetvl(((long)simd_width / 2 - (long)i1), __epi_e32, __epi_m1);

                        float vin1234[2 * simd_width], vin12341[2 * simd_width];

                        __builtin_epi_vstore_2xf32(&vin1234[0], o4, gvl);
                        __builtin_epi_vstore_2xf32(&vin1234[simd_width], o5, gvl);
                        __builtin_epi_vstore_2xf32(&vin12341[0], o10, gvl);
                        __builtin_epi_vstore_2xf32(&vin12341[simd_width], o11, gvl);
                        // printf("upto here");

                        int ind11 = 0;
                        int index11_host[simd_width / 2];
                        for (int i = 0; i < interchannels; i++)
                        {
                                for (int j = 0; j < 2; j++)
                                {
                                        index11_host[ind11] = ((j * simd_width) + 2 * 2 * i); // added extra 2*
                                        ind11++;
                                }
                        }
                        __epi_2xi32 index1_new, index2_new, index3_new, index4_new;
                        index1_new = __builtin_epi_vload_2xi32(&index11_host[0], gvl_new);
                        index2_new = __builtin_epi_vadd_2xi32(index1_new, vone, gvl_new);
                        index3_new = __builtin_epi_vadd_2xi32(index2_new, vone, gvl_new);
                        index4_new = __builtin_epi_vadd_2xi32(index3_new, vone, gvl_new);
                        __epi_2xi32 index11_new = __builtin_epi_vmul_2xi32(index1_new, FOUR, gvl_new);
                        __epi_2xi32 index12_new = __builtin_epi_vmul_2xi32(index2_new, FOUR, gvl_new);
                        __epi_2xi32 index13_new = __builtin_epi_vmul_2xi32(index3_new, FOUR, gvl_new);
                        __epi_2xi32 index14_new = __builtin_epi_vmul_2xi32(index4_new, FOUR, gvl_new);
                        __epi_2xf32 d0_new = __builtin_epi_vload_indexed_2xf32(&vin1234[0], index11_new, gvl_new);
                        __epi_2xf32 d1_new = __builtin_epi_vload_indexed_2xf32(&vin1234[0], index12_new, gvl_new);
                        __epi_2xf32 d2_new = __builtin_epi_vload_indexed_2xf32(&vin1234[0], index13_new, gvl_new);
                        __epi_2xf32 d3_new = __builtin_epi_vload_indexed_2xf32(&vin1234[0], index14_new, gvl_new);
                        __epi_2xf32 d4_new = __builtin_epi_vload_indexed_2xf32(&vin12341[0], index11_new, gvl_new);
                        __epi_2xf32 d5_new = __builtin_epi_vload_indexed_2xf32(&vin12341[0], index12_new, gvl_new);
                        __epi_2xf32 d6_new = __builtin_epi_vload_indexed_2xf32(&vin12341[0], index13_new, gvl_new);
                        __epi_2xf32 d7_new = __builtin_epi_vload_indexed_2xf32(&vin12341[0], index14_new, gvl_new);

                        //////stop here
                        __epi_2xf32 dout0, dout1, dout2, dout3, dout4, dout5;

                        const __epi_2xf32 m1_add_m2_new = __builtin_epi_vfadd_2xf32(d1_new, d2_new, gvl_new);
                        const __epi_2xf32 m1_sub_m2_new = __builtin_epi_vfsub_2xf32(d1_new, d2_new, gvl_new);
                        const __epi_2xf32 m3_add_m4_new = __builtin_epi_vfadd_2xf32(d3_new, d4_new, gvl_new);
                        const __epi_2xf32 m3_sub_m4_new = __builtin_epi_vfsub_2xf32(d3_new, d4_new, gvl_new);
                        const __epi_2xf32 m5_add_m6_new = __builtin_epi_vfadd_2xf32(d5_new, d6_new, gvl_new);
                        const __epi_2xf32 m5_sub_m6_new = __builtin_epi_vfsub_2xf32(d5_new, d6_new, gvl_new);

                        __epi_2xf32 s0_new = __builtin_epi_vfadd_2xf32(d0_new, m1_add_m2_new, gvl_new);
                        __epi_2xf32 s5_new = __builtin_epi_vfadd_2xf32(d7_new, m1_sub_m2_new, gvl_new);

                        __epi_2xf32 s1_new = __builtin_epi_vfmacc_2xf32(m1_sub_m2_new, m5_sub_m6_new, const_16, gvl_new);
                        __epi_2xf32 s4_new = __builtin_epi_vfmacc_2xf32(m1_add_m2_new, m3_add_m4_new, const_16, gvl_new);
                        __epi_2xf32 s2_new = __builtin_epi_vfmacc_2xf32(m1_add_m2_new, m5_add_m6_new, const_8, gvl_new);
                        __epi_2xf32 s3_new = __builtin_epi_vfmacc_2xf32(m1_sub_m2_new, m3_sub_m4_new, const_8, gvl_new);

                        s0_new = __builtin_epi_vfmacc_2xf32(s0_new, m5_add_m6_new, const_32, gvl_new);
                        s5_new = __builtin_epi_vfmacc_2xf32(s5_new, m3_sub_m4_new, const_32, gvl_new);
                        s1_new = __builtin_epi_vfmacc_2xf32(s1_new, m3_sub_m4_new, const_2, gvl_new);
                        s4_new = __builtin_epi_vfmacc_2xf32(s4_new, m5_add_m6_new, const_2, gvl_new);
                        s0_new = __builtin_epi_vfadd_2xf32(s0_new, m3_add_m4_new, gvl_new);
                        s5_new = __builtin_epi_vfadd_2xf32(s5_new, m5_sub_m6_new, gvl_new);
                        s2_new = __builtin_epi_vfmacc_2xf32(s2_new, m3_add_m4_new, const_4, gvl_new);
                        s3_new = __builtin_epi_vfmacc_2xf32(s3_new, m5_sub_m6_new, const_4, gvl_new);
                        dout0 = s0_new;
                        dout1 = s1_new;
                        dout2 = s2_new;
                        dout3 = s3_new;
                        dout4 = s4_new;
                        dout5 = s5_new;
                        __builtin_epi_vstore_2xf32(&block1[0], dout0, gvl_new);
                        __builtin_epi_vstore_2xf32(&block1[simd_width / 2], dout1, gvl_new);
                        __builtin_epi_vstore_2xf32(&block1[2 * (simd_width / 2)], dout2, gvl_new);
                        __builtin_epi_vstore_2xf32(&block1[3 * (simd_width / 2)], dout3, gvl_new);
                        __builtin_epi_vstore_2xf32(&block1[4 * (simd_width / 2)], dout4, gvl_new);
                        __builtin_epi_vstore_2xf32(&block1[5 * (simd_width / 2)], dout5, gvl_new);

                        for (size_t i = 0; i < row_count; i++)
                        {
#pragma loop unroll_count(interchannels)
                                for (int k = 0; k < interchannels; k++)
                                {

                                        for (size_t j = 0; j < column_count; j++) // original trying to change it
                                        {
                                                if (j < 4)
                                                { // original trying to change it
                                                        output[k][i * output_stride + j] = block[i * simd_width + (j + k * 4)];
                                                }
                                        }
                                }
#pragma loop unroll_count(interchannels)
                                for (int k = 0; k < interchannels; k++)
                                {
                                        for (size_t j = 0; j < column_count; j++)
                                        {
                                                if (j > 3 && j < 6)
                                                {
                                                        output[k][i * output_stride + j] = block1[(i * (simd_width / 2)) + ((j + k * 2) - 4)];
                                                }
                                        }
                                }
                        }

                        //  printf("sve\n");
                        //  for(int i=0;i<48;i++)
                        ///{
                        //        printf("output_col45[0]= %f for i= %d\n",  output[0][i], i);
                        //        printf("output_col45[3]= %f for i = %d\n",  output[3][i], i);
                        //	}
                }
                i1 += gvl;
        }
        if (new_data != NULL)
        {
                free(new_data);
                new_data = NULL;
        }
}

void nnp_owt8x8_3x3s2_with_bias__neon(
    const void *restrict transform,
    float output[restrict static 1],
    const float bias[restrict static 1],
    size_t transform_stride,
    size_t output_stride,
    int row_count,
    int column_count)
{
}

void nnp_owt8x8_3x3s2_with_bias__neon_intertile(
    const void **restrict transform,
    float *output[restrict static 1],
    const float bias[restrict static 1],
    size_t transform_stride,
    size_t output_stride,
    int row_count,
    int column_count)
{
}
void nnp_owt8x8_3x3_with_bias_with_relu__neon(
    const void *restrict transform,
    float output[restrict static 1],
    const float bias[restrict static 1],
    size_t transform_stride, size_t output_stride,
    int row_count, int column_count)
{
}

void nnp_owt8x8_3x3s2_with_bias_with_relu__neon(
    const void *restrict transform,
    float output[restrict static 1],
    const float bias[restrict static 1],
    size_t transform_stride, size_t output_stride,
    int row_count, int column_count)
{
}

////////////////////////////////////////////////////////////////////////// convolutional file
void tuple_parallelize_2d_tile_2d(
    // pthreadpool_t threadpool,
    //  pthreadpool_task_2d_tile_2d_t task,
    void *argument,
    size_t range_i,
    size_t range_j,
    size_t tile_i,
    size_t tile_j,
    int flags)
{
        //       printf("I am going in correct no-intertile logic with range_i = %d\n", range_i);
        /* No thread pool used: execute task sequentially on the calling thread */
        for (size_t i = 0; i < range_i; i += tile_i)
        {
                for (size_t j = 0; j < range_j; j += tile_j)
                {
                        compute_tuple_multiplication(argument, i, j, min(range_i - i, tile_i), min(range_j - j, tile_j));
                }
        }
}

/////////////////////////////
void output_parallelize_2d_tile_2d(
    // pthreadpool_t threadpool,
    // pthreadpool_task_2d_tile_2d_t task,
    void *argument,
    size_t range_i,
    size_t range_j,
    size_t tile_i,
    size_t tile_j,
    int flags)
{
        //           printf("I am going in correct no-intertile logic with range_i = %d %d %d %d\n\n", range_i, range_j, tile_i, tile_j);
        /* No thread pool used: execute task sequentially on the calling thread */
        for (size_t i = 0; i < range_i; i += tile_i)
        {
                for (size_t j = 0; j < range_j; j += tile_j)
                {
                        compute_output_transform(argument, i, j, min(range_i - i, tile_i), min(range_j - j, tile_j));
                }
        }
}

/////////////////
void kernel_parallelize_2d_tile_2d_intertile(
    void *argument,
    size_t range_i,
    size_t range_j,
    size_t tile_i,
    size_t tile_j, size_t tiles,
    int flags)
{

        // fprintf(stderr, "I am in kernel function with range_i = %d tiles = %d\n", range_i, tiles);
        // printf("range_i = %d, range j = %d\n", range_i, range_j);
        size_t threads_count;
        //        int interchannels =tiles;//16 ;  // for 1024-bit
        int interchannels; // 16 ;  // for 1024-bit
        if (range_j >= tiles)
        {
                interchannels = tiles; // 16 ;  // for 1024-bit
        }
        else
        {
                interchannels = range_j;
        }
        //	fprintf(stderr, "I am after my logic kernel function with range_i = %d interchannels = %d\n", range_i, interchannels);

        // added by sonia- starting with sequential
        if ((range_i >= 4) && (range_j >= 4))
        {
                // fprintf(stderr, "I am going in correct logic with range_i in intertile= %d\n", range_i);
                if (tile_i == 1)
                {
                        tile_i = interchannels;
                }
                else if (tile_j == 1)
                {
                        tile_j = interchannels;
                }
                /* No thread pool used: execute task sequentially on the calling thread */
                for (size_t i = 0; i < range_i; i += tile_i)
                { // need to load 8x8 tiles from 4 channels at a time
                        for (size_t j = 0; j < range_j; j += tile_j)
                        {
                                //              printf(" tile i = %d tile j = %d, min(range_i - i, tile_i) = %d min(range_j - j, tile_j) = %d\n", tile_i, tile_j, min(range_i - i, tile_i),  min(range_j - j, tile_j));
                                compute_kernel_transform(argument, i, j, min(range_i - i, tile_i), min(range_j - j, tile_j), interchannels);
                        }
                }

        } // added by sonia - end
        else
        {
                // fprintf(stderr, "I am going in correct else if logic with range_i = %d", range_i);
                /* No thread pool used: execute task sequentially on the calling thread */
                for (size_t i = 0; i < range_i; i += tile_i)
                {
                        for (size_t j = 0; j < range_j; j += tile_j)
                        {
                                compute_kernel_transform(argument, i, j, min(range_i - i, tile_i), min(range_j - j, tile_j), interchannels);
                        }
                }
        }
}
///////////////////////////////
void input_parallelize_2d_tile_2d_intertile(
    //   pthreadpool_t threadpool,
    void *argument,
    size_t range_i,
    size_t range_j,
    size_t tile_i,
    size_t tile_j, size_t tiles,
    int flags)
{

        //	fprintf(stderr, "I am in input pthread function");
        //       printf("range_i = %d, range j = %d\n", range_i, range_j);
        size_t threads_count;
        int interchannels; // 16 ;  // for 1024-bit
        if (range_i >= tiles)
        {
                interchannels = tiles; // 16 ;  // for 1024-bit
        }
        else
        {
                interchannels = range_i;
        }

        if ((range_i >= 4) && (range_j >= 4))
        {
                // fprintf(stderr, "I am going in correct logic with range_i in intertile= %d\n", range_i);
                if (tile_i == 1)
                {
                        tile_i = interchannels;
                }
                else if (tile_j == 1)
                {
                        tile_j = interchannels;
                }
                /* No thread pool used: execute task sequentially on the calling thread */
                for (size_t i = 0; i < range_i; i += tile_i)
                { // need to load 8x8 tiles from 4 channels at a time
                        for (size_t j = 0; j < range_j; j += tile_j)
                        {
                                //              printf(" tile i = %d tile j = %d, min(range_i - i, tile_i) = %d min(range_j - j, tile_j) = %d\n", tile_i, tile_j, min(range_i - i, tile_i),  min(range_j - j, tile_j));
                                compute_input_transform(argument, i, j, min(range_i - i, tile_i), min(range_j - j, tile_j), interchannels);
                        }
                }
        } // added by sonia - end
        else
        {
                // fprintf(stderr, "I am going in correct else if logic with range_i = %d", range_i);
                /* No thread pool used: execute task sequentially on the calling thread */
                for (size_t i = 0; i < range_i; i += tile_i)
                {
                        for (size_t j = 0; j < range_j; j += tile_j)
                        {
                                compute_input_transform(argument, i, j, min(range_i - i, tile_i), min(range_j - j, tile_j), interchannels);
                        }
                }
        }
}
//////////////////////////////

static void compute_kernel_transform(
    const struct kernel_transform_context context[restrict static 1],
    size_t output_channels_subblock_start, size_t input_channels_block_offset,
    size_t output_channels_subblock_size, size_t input_channels_block_increment, size_t interchannels)
{
        // fprintf(stderr, "I am in kernel");
        const size_t tuple_size = context->tuple_size;
        const size_t input_channels = context->input_channels;
        const size_t input_channels_block_size = context->input_channels_block_size;
        const size_t output_channels = context->output_channels;
        const struct nnp_size kernel_size = context->kernel_size;

        const float(*kernel)[input_channels][kernel_size.width * kernel_size.height] =
            (const float(*)[input_channels][kernel_size.width * kernel_size.height]) context->kernel;
        void *kernel_transform = context->kernel_transform;
        nnp_transform_2d_with_offset transform_function = context->transform_function;

        for (size_t output_channels_subblock_offset = 0; output_channels_subblock_offset < output_channels_subblock_size; output_channels_subblock_offset += 1)
        {
                const size_t output_channel = output_channels_subblock_start + output_channels_subblock_offset;

                if (input_channels_block_size <= 3)
                {
                        nnp_kwt8x8_3x3__neon(
                            kernel[output_channel][input_channels_block_offset],
                            kernel_transform +
                                (output_channels_subblock_start * input_channels_block_size + input_channels_block_offset * output_channels_subblock_size + output_channels_subblock_offset) * tuple_size,
                            kernel_size.width,
                            input_channels_block_size * output_channels * tuple_size,
                            kernel_size.height, kernel_size.width, 0, 0);
                }
                else
                {
                        const float *kernel_ptr[interchannels];
                        float *kernel_transform_ptr[interchannels];
                        for (int k = 0; k < interchannels; k++)
                        {
                                kernel_ptr[k] = &kernel[output_channel][input_channels_block_offset + k];
                                kernel_transform_ptr[k] = kernel_transform + (output_channels_subblock_start * input_channels_block_size + (input_channels_block_offset + k) * output_channels_subblock_size + output_channels_subblock_offset) * tuple_size;
                        }
                        nnp_kwt8x8_3x3__neon_intertile(
                            kernel_ptr,
                            kernel_transform_ptr,
                            kernel_size.width,
                            input_channels_block_size * output_channels * tuple_size,
                            kernel_size.height, kernel_size.width, 0, 0, interchannels);
                }
        }
}

static void compute_input_transform(
    const struct input_transform_context context[restrict static 1],
    size_t input_channels_block_offset, size_t tiles_subblock_start,
    size_t input_channels_block_range, size_t tiles_subblock_size, size_t interchannels)

{
        // fprintf(stderr, "I am in input");
        const size_t tuple_size = context->tuple_size;
        const size_t tiles_count = context->tiles_count;
        const struct fxdiv_divisor_size_t tiles_x_count = context->tiles_x_count;
        const size_t input_channels_block_start = context->input_channels_block_start;
        const size_t input_channels_block_size = context->input_channels_block_size;
        const struct nnp_size input_size = context->input_size;
        const size_t input_padding_left = context->input_padding_left;
        const size_t input_padding_top = context->input_padding_top;
        const struct nnp_size input_tile = context->input_tile;
        const struct nnp_size input_tile_step = context->input_tile_step;

        const float(*input)[input_size.height][input_size.width] =
            (const float(*)[input_size.height][input_size.width])context->input;
        void *input_transform = context->input_transform;
        nnp_transform_2d_with_offset transform_function = context->transform_function;

        const size_t input_channel = input_channels_block_start + input_channels_block_offset;
        for (size_t tiles_subblock_offset = 0; tiles_subblock_offset < tiles_subblock_size; tiles_subblock_offset += 1)
        {
                const size_t tile = tiles_subblock_start + tiles_subblock_offset;
                const struct fxdiv_result_size_t tile_xy = fxdiv_divide_size_t(tile, tiles_x_count);
                const size_t tile_x = tile_xy.remainder;
                const size_t tile_y = tile_xy.quotient;

                const size_t output_x = tile_x * input_tile_step.width;
                const size_t output_y = tile_y * input_tile_step.height;

                const size_t input_x = min(doz(output_x, input_padding_left), input_size.width);
                const size_t input_y = min(doz(output_y, input_padding_top), input_size.height);

                const size_t row_offset = doz(input_padding_top, output_y);
                const size_t row_count = min(input_size.height - input_y, input_tile.height - row_offset);
                const size_t column_offset = doz(input_padding_left, output_x);
                const size_t column_count = min(input_size.width - input_x, input_tile.width - column_offset);
                if (input_channels_block_size <= 3)
                {
                        // printf("transform stride = %f\n", (input_channels_block_size * tiles_count * tuple_size));
                        // printf(" input channels start = %d, index =  %d\n",input_channels_block_offset,  ((tiles_subblock_start * input_channels_block_size + input_channels_block_offset * tiles_subblock_size + tiles_subblock_offset) * tuple_size));
                        nnp_iwt8x8_3x3_with_offset__neon(
                            &input[input_channel][input_y][input_x],
                            input_transform + (tiles_subblock_start * input_channels_block_size + input_channels_block_offset * tiles_subblock_size + tiles_subblock_offset) * tuple_size,
                            input_size.width,
                            input_channels_block_size * tiles_count * tuple_size,
                            row_count, column_count, row_offset, column_offset);
                }
                else
                {
                        // fprintf(stderr, "going in interchannels %d\n", interchannels);
                        const float *input_ptr[interchannels];
                        void *transform_ptr[interchannels];
                        for (int k = 0; k < interchannels; k++)
                        {
                                input_ptr[k] = &input[input_channel + k][input_y][input_x];
                                transform_ptr[k] = input_transform + (tiles_subblock_start * input_channels_block_size + (input_channels_block_offset + k) * tiles_subblock_size + tiles_subblock_offset) * tuple_size;
                        }
                        nnp_iwt8x8_3x3_with_offset__neon_intertile1(
                            input_ptr,
                            transform_ptr,
                            input_size.width,
                            input_channels_block_size * tiles_count * (tuple_size),
                            row_count, column_count, row_offset, column_offset, interchannels);
                        // fprintf(stderr, "coming out\n");
                }
        }
}

static void compute_output_transform(
    const struct output_transform_context context[restrict static 1],
    size_t output_channels_subblock_start, size_t tiles_subblock_start,
    size_t output_channels_subblock_size, size_t tiles_subblock_size)
{
        int interchannels;
        if (output_channels_subblock_size >= nnp_hwinfo.globalinterchannels)
        {
                interchannels = nnp_hwinfo.globalinterchannels;
        }
        else
        {
                interchannels = output_channels_subblock_size;
        }
        //	fprintf(stderr, "value of interchannels in output = %d", interchannels);
        const size_t tuple_size = context->tuple_size;
        const size_t tiles_count = context->tiles_count;
        const struct fxdiv_divisor_size_t tiles_x_count = context->tiles_x_count;
        const struct fxdiv_divisor_size_t tiles_block_max = context->tiles_block_max;
        const size_t output_channels = context->output_channels;
        const struct nnp_size output_size = context->output_size;
        const struct nnp_size output_tile = context->output_tile;

        const size_t tiles_block_start = fxdiv_round_down_size_t(tiles_subblock_start, tiles_block_max);
        const size_t tiles_block_size = min(tiles_count - tiles_block_start, tiles_block_max.value);

        float(*output)[output_size.height][output_size.width] =
            (float(*)[output_size.height][output_size.width])context->output;
        const void *output_transform = context->output_transform;
        const float *bias = context->bias;
        nnp_transform_2d_with_bias transform_function = context->transform_function;
        for (size_t tiles_subblock_offset = 0; tiles_subblock_offset < tiles_subblock_size; tiles_subblock_offset += 1)
        {
                const size_t tile = tiles_subblock_start + tiles_subblock_offset;
                const struct fxdiv_result_size_t tile_xy = fxdiv_divide_size_t(tile, tiles_x_count);
                const size_t tile_x = tile_xy.remainder;
                const size_t tile_y = tile_xy.quotient;

                const size_t output_x = tile_x * output_tile.width;
                const size_t output_y = tile_y * output_tile.height;
                if (subsampling == 2 && output_channels_subblock_size < 4)
                {
                        printf("in owt");
                        for (size_t output_channels_subblock_offset = 0; output_channels_subblock_offset < output_channels_subblock_size; output_channels_subblock_offset += 1)
                        {
                                const size_t output_channel = output_channels_subblock_start + output_channels_subblock_offset;
                                nnp_owt8x8_3x3s2_with_bias__neon(
                                    output_transform +
                                        (tiles_block_start * output_channels + output_channels_subblock_start * tiles_block_size + ((tiles_subblock_start - tiles_block_start) + tiles_subblock_offset) * output_channels_subblock_size + output_channels_subblock_offset) * tuple_size,
                                    &output[output_channel][output_y][output_x],
                                    &bias[output_channel],
                                    tiles_count * output_channels * tuple_size,
                                    output_size.width,
                                    min(output_tile.height, output_size.height - output_y),
                                    min(output_tile.width, output_size.width - output_x));
                        }
                }
                else if (output_channels_subblock_size >= 4)
                {
                        //	printf("nnp_hwinfo.globalinterchannels = %d",nnp_hwinfo.globalinterchannels);

                        //      printf("output_channels_subblock_size = %d\n", output_channels_subblock_size);
                        for (size_t output_channels_subblock_offset = 0; output_channels_subblock_offset < output_channels_subblock_size; output_channels_subblock_offset += interchannels)
                        {
                                float *output_ptr[interchannels];
                                void *output_transform_ptr[interchannels];
                                const size_t output_channel = output_channels_subblock_start + output_channels_subblock_offset;
                                //      const size_t output_channel1 = output_channels_subblock_start + (output_channels_subblock_offset+1);
                                //      const size_t output_channel2 = output_channels_subblock_start + (output_channels_subblock_offset+2);
                                //      const size_t output_channel3 = output_channels_subblock_start + (output_channels_subblock_offset+3);
                                for (int k = 0; k < interchannels; k++)
                                {
                                        output_ptr[k] = &output[output_channel + k][output_y][output_x];
                                        output_transform_ptr[k] = output_transform +
                                                                  (tiles_block_start * output_channels + output_channels_subblock_start * tiles_block_size + ((tiles_subblock_start - tiles_block_start) + tiles_subblock_offset) * output_channels_subblock_size + (output_channels_subblock_offset + k)) * tuple_size;
                                }
                                if (subsampling == 2)
                                {
                                        printf("in owt s2");
                                        nnp_owt8x8_3x3s2_with_bias__neon_intertile(
                                            output_transform_ptr,
                                            output_ptr,
                                            &bias[output_channel],
                                            tiles_count * output_channels * tuple_size,
                                            output_size.width,
                                            min(output_tile.height, output_size.height - output_y),
                                            min(output_tile.width, output_size.width - output_x));
                                }
                                else
                                {
                                        nnp_owt8x8_3x3_with_bias__neon_intertile(
                                            output_transform_ptr,
                                            output_ptr,
                                            &bias[output_channel],
                                            tiles_count * output_channels * tuple_size,
                                            output_size.width,
                                            min(output_tile.height, output_size.height - output_y),
                                            min(output_tile.width, output_size.width - output_x), interchannels);
                                }
                        }
                }
                else
                {
                        printf("inside neon owt");
                        for (size_t output_channels_subblock_offset = 0; output_channels_subblock_offset < output_channels_subblock_size; output_channels_subblock_offset += 1)
                        {
                                const size_t output_channel = output_channels_subblock_start + output_channels_subblock_offset;
                                nnp_owt8x8_3x3s2_with_bias__neon(
                                    output_transform +
                                        (tiles_block_start * output_channels + output_channels_subblock_start * tiles_block_size + ((tiles_subblock_start - tiles_block_start) + tiles_subblock_offset) * output_channels_subblock_size + output_channels_subblock_offset) * tuple_size,
                                    &output[output_channel][output_y][output_x],
                                    &bias[output_channel],
                                    tiles_count * output_channels * tuple_size,
                                    output_size.width,
                                    min(output_tile.height, output_size.height - output_y),
                                    min(output_tile.width, output_size.width - output_x));
                        }
                }
        }
}

static void compute_tuple_multiplication(
    const struct tuple_multiplication_context context[restrict static 1],
    size_t tiles_block_start, size_t output_channels_subblock_start,
    size_t tiles_block_size, size_t output_channels_subblock_size)
{

        const size_t tuple_elements = context->tuple_elements;
        const size_t tuple_size = context->tuple_size;
        const size_t tiles_subblock_max = context->tiles_subblock_max;
        const size_t input_channels_block_size = context->input_channels_block_size;
        const size_t input_channels_block_start = context->input_channels_block_start;
        const size_t output_channels = context->output_channels;
        const size_t output_channels_subblock_max = context->output_channels_subblock_max;
        const size_t output_channels_block_start = context->output_channels_block_start;

        const void *input_transform = context->input_transform +
                                      tiles_block_start * input_channels_block_size * tuple_size;
        const void *kernel_transform = context->kernel_transform +
                                       (output_channels_block_start + output_channels_subblock_start) * input_channels_block_size * tuple_size;
        void *output_transform = context->output_transform +
                                 (tiles_block_start * output_channels + (output_channels_block_start + output_channels_subblock_start) * tiles_block_size) * tuple_size;

        //      printf("tuple elements = %d, output_channels_subblock_size=%d output_channels_subblock_max=%d input_channels_block_size=%d\n", tuple_elements, output_channels_subblock_size, output_channels_subblock_max, input_channels_block_size);
        if (output_channels_subblock_size == output_channels_subblock_max)
        {
                const nnp_fast_tuple_gemm_function fast_gemm = context->fast_gemm;
                while (tiles_block_size >= tiles_subblock_max)
                {
                        tiles_block_size -= tiles_subblock_max;
                        // printf("fast gemm\n");
                        nnp_s4gemm_only_3x3__neon(
                            input_channels_block_size, input_channels_block_start,
                            input_transform, kernel_transform, output_transform,
                            output_channels_subblock_size * tuple_elements);

                        input_transform += tiles_subblock_max * input_channels_block_size * tuple_size;
                        output_transform += tiles_subblock_max * output_channels_subblock_size * tuple_size;
                }
        }
        //		printf("intermediate %f\n", *(((float *)output_transform)+3) );

        const nnp_full_tuple_gemm_function full_gemm = context->full_gemm;
        while (tiles_block_size != 0)
        {
                const size_t tiles_subblock_size = min(tiles_block_size, tiles_subblock_max);
                tiles_block_size -= tiles_subblock_size;

                //      printf("full gemm\n");
                nnp_s4gemm_upto_3x3__neon(
                    tiles_subblock_size, output_channels_subblock_size,
                    input_channels_block_size, input_channels_block_start,
                    input_transform, kernel_transform, output_transform,
                    output_channels_subblock_size * tuple_elements);

                input_transform += tiles_subblock_max * input_channels_block_size * tuple_size;
                output_transform += tiles_subblock_max * output_channels_subblock_size * tuple_size;
        } // sonia - need to uncomment uncomment
        //		printf("intermediate %f\n", *(((float *)output_transform)+3) );
}

struct NNP_CACHE_ALIGN kernel_packing_context
{
        const float *kernel;
        float *packed_kernel;

        size_t reduction_size;
        size_t reduction_block_start;
        size_t reduction_block_size;
};

static enum nnp_status compute_fast_convolution_inference(
    const bool fourier_transform,
    const enum nnp_convolution_transform_strategy transform_strategy,
    const size_t transform_element_size,
    const size_t input_channels,
    const size_t output_channels,
    const struct nnp_size tile_size,
    const struct nnp_size input_size,
    const struct nnp_padding input_padding,
    const struct nnp_size kernel_size,
    const struct nnp_size output_size,
    const struct nnp_size output_subsampling,
    const float *input,
    const float *kernel,
    const float *bias,
    float *output,
    void *workspace_buffer,
    size_t *workspace_size,
    const nnp_transform_2d_with_offset input_transform_function,
    const nnp_transform_2d_with_offset kernel_transform_function,
    const nnp_transform_2d_with_bias output_transform_function,
    pthreadpool_t threadpool,
    struct nnp_profile *profile)
{
        // fprintf(stderr, "I am in 2");
        void *memory_block = NULL;
        size_t memory_size = 0;
        size_t simd_width;
        simd_width = nnp_hwinfo.simd_width;
        //       if (fourier_transform) {
        //      simd_width = nnp_hwinfo.simd_width;
        //      }
        //      else{
        //      simd_width = 8;
        //      }
        const size_t tuple_elements = 4;                                   // simd_width;
        const size_t tuple_size = tuple_elements * transform_element_size; // transform_element_size = sizeof(float)
        const size_t tile_elements = tile_size.height * tile_size.width;
        const size_t tuple_count = tile_elements / tuple_elements;
        const struct nnp_size output_tile_size = {
            .width = (tile_size.width - kernel_size.width) / output_subsampling.width + 1,
            .height = (tile_size.height - kernel_size.height) / output_subsampling.height + 1};
        const struct nnp_size tile_step = {
            .width = tile_size.width - kernel_size.width + 1,
            .height = tile_size.height - kernel_size.height + 1};

        const size_t tiles_y_count = divide_round_up(output_size.height, output_tile_size.height);
        const size_t tiles_x_count = divide_round_up(output_size.width, output_tile_size.width);
        const size_t tiles_count = tiles_x_count * tiles_y_count;

        printf("tuple count=%d tuple size = %d tuple elements = %d nnp_hwinfo.blocking.l1=%d nnp_hwinfo.blocking.l2=%d nnp_hwinfo.blocking.l3 =%d\n", tuple_count, tuple_size, tuple_elements, nnp_hwinfo.blocking.l1, nnp_hwinfo.blocking.l2, nnp_hwinfo.blocking.l3);
        /* Calculate cache blocking parameters */
        const size_t cache_elements_l1 = nnp_hwinfo.blocking.l1 / tuple_size;
        const size_t cache_elements_l2 = nnp_hwinfo.blocking.l2 / tuple_size;
        const size_t cache_elements_l3 = nnp_hwinfo.blocking.l3 / tuple_size;

        const size_t tiles_subblock_max = (nnp_hwinfo.sxgemm.mr);
        fprintf(stderr, "tiles_sublock_max %d nnp_hwinfo.globalinterchannels - %d", tiles_subblock_max, nnp_hwinfo.globalinterchannels);
        const size_t output_channels_subblock_max = (nnp_hwinfo.sxgemm.nr);
        fprintf(stderr, "cache_elements_l1 = %d value = %d", cache_elements_l1, tiles_subblock_max);
        const size_t input_channels_block_max =
            round_down(cache_elements_l1 / (tiles_subblock_max + output_channels_subblock_max), 2);
        const size_t tiles_block_max =
            round_down(cache_elements_l2 / input_channels_block_max, tiles_subblock_max);
        const size_t output_channels_block_max =
            round_down(cache_elements_l3 / input_channels_block_max, output_channels_subblock_max);

        const size_t transform_tile_size = tile_elements * transform_element_size;
        const size_t input_transform_size = tiles_count * min(input_channels, input_channels_block_max) * transform_tile_size;
        const size_t output_transform_size = tiles_count * output_channels * transform_tile_size;
        switch (transform_strategy)
        {
        case nnp_convolution_transform_strategy_compute:
        case nnp_convolution_transform_strategy_reuse:
        {
                memory_size = input_transform_size + output_transform_size;
                const size_t kernel_transform_size = output_channels * min(input_channels, input_channels_block_max) * transform_tile_size;
                if (transform_strategy == nnp_convolution_transform_strategy_compute)
                {
                        memory_size += kernel_transform_size;
                }
                if (workspace_buffer == NULL)
                {
                        if (workspace_size == NULL)
                        {
                                memory_block = allocate_memory(memory_size);
                                if (memory_block == NULL)
                                {
                                        return nnp_status_out_of_memory;
                                }
                        }
                        else
                        {
                                *workspace_size = memory_size;
                                return nnp_status_success;
                        }
                }
                else
                {
                        if (*workspace_size < memory_size)
                        {
                                return nnp_status_insufficient_buffer;
                        }
                        memory_block = workspace_buffer;
                }
                // fprintf(stderr, "input_channels = %d", input_channels);
                //	fprintf(stderr, "I am in 2 before transforms");
                void *input_transform = memory_block;
                void *output_transform = memory_block + input_transform_size;
                void *kernel_transform = memory_block + input_transform_size + output_transform_size; // commented by sonia

                for (size_t input_channels_block_start = 0; input_channels_block_start < input_channels; input_channels_block_start += input_channels_block_max)
                {
                        const size_t input_channels_block_size = min(input_channels - input_channels_block_start, input_channels_block_max);

                        if (transform_strategy == nnp_convolution_transform_strategy_compute)
                        {
                                //        NNP_KERNEL_TRANSFORM_START(profile)
                                struct kernel_transform_context kernel_transform_context = {
                                    .transform_function = kernel_transform_function,
                                    .kernel = kernel + input_channels_block_start * kernel_size.height * kernel_size.width,
                                    .kernel_transform = kernel_transform,
                                    .tuple_size = tuple_size,
                                    .input_channels = input_channels,
                                    .input_channels_block_size = input_channels_block_size,
                                    .output_channels = output_channels,
                                    .kernel_size = kernel_size,
                                };
                                struct timeval starttime2, endtime2;
                                fprintf(stderr, "I am in 2 before kernel transforms");
                                // gettimeofday(&starttime2,NULL);
                                kernel_parallelize_2d_tile_2d_intertile( // threadpool,
                                                                         //         (pthreadpool_task_2d_tile_2d_t1) compute_kernel_transform,
                                    &kernel_transform_context,
                                    output_channels, input_channels_block_size,
                                    output_channels_subblock_max, 1, nnp_hwinfo.globalinterchannels,
                                    PTHREADPOOL_FLAG_DISABLE_DENORMALS);
                                //       NNP_KERNEL_TRANSFORM_END(profile)
                                gettimeofday(&endtime2, NULL);
                                double totaltime = ((endtime2.tv_sec + .000001 * endtime2.tv_usec) - (starttime2.tv_sec + .000001 * starttime2.tv_usec));
                                // new changes for kernel reuse
                                //      kernel = kernel_transform;
                        }
                        else
                        {
                                kernel_transform = (void *)kernel + input_channels_block_start * output_channels * transform_tile_size;
                                // kernel_transform = ((void*) kernel) + input_channels_block_start * output_channels * transform_tile_size,
                                // input_channels_block_start * kernel_size.height * kernel_size.width;
                                // input_channels_block_start * output_channels * transform_tile_size;
                        }
                        //				printf("\n after kernel transformation\n");
                        //				for(int i=0;i<10;i++)
                        //					printf("kernel transformation = %f", *(((float *)kernel_transform)+i));
                        //  NNP_INPUT_TRANSFORM_START(profile);
                        fprintf(stderr, "I am in 2 after kernel transforms");
                        struct input_transform_context input_transform_context = {
                            .input = input,
                            .input_transform = input_transform,
                            .transform_function = input_transform_function,
                            .tuple_size = tuple_size,
                            .tiles_count = tiles_count,
                            .tiles_x_count = fxdiv_init_size_t(tiles_x_count),
                            .input_channels_block_start = input_channels_block_start,
                            .input_channels_block_size = input_channels_block_size,
                            .input_size = input_size,
                            .input_padding_left = input_padding.left,
                            .input_padding_top = input_padding.top,
                            .input_tile = tile_size,
                            .input_tile_step = tile_step,
                        };
                        //                              pthreadpool_parallelize_2d_tile_2d(threadpool,
                        //                                      (pthreadpool_task_2d_tile_2d_t) compute_input_transform,
                        //                                      &input_transform_context,
                        //                                      input_channels_block_size, tiles_count,
                        //                                      1,                         tiles_subblock_max,
                        //                                      PTHREADPOOL_FLAG_DISABLE_DENORMALS);
                        fprintf(stderr, "I am in before input transforms");

                        input_parallelize_2d_tile_2d_intertile( // threadpool,  //need o uncomet
                            //(pthreadpool_task_2d_tile_2d_t1) compute_input_transform,
                            &input_transform_context,
                            input_channels_block_size, tiles_count, 1, tiles_subblock_max, nnp_hwinfo.globalinterchannels,
                            PTHREADPOOL_FLAG_DISABLE_DENORMALS);
                        fprintf(stderr, "after input");
                        //				printf("\n after input transformation\n");
                        //				for(int i=0;i<10;i++)
                        //					printf("input transformation = %f\t ", *(((float *)input_transform)+i));

                        //    NNP_INPUT_TRANSFORM_END(profile)

                        //      NNP_BLOCK_MULTIPLICATION_START(profile)
                        for (size_t tuple_index = 0; tuple_index < tuple_count; tuple_index += 1)
                        {
                                nnp_full_tuple_gemm_function full_gemm_function;
                                nnp_fast_tuple_gemm_function fast_gemm_function;
                                fast_gemm_function = nnp_hwinfo.sxgemm.only_mr_x_nr;
                                full_gemm_function = nnp_hwinfo.sxgemm.upto_mr_x_nr;
                                for (size_t output_channels_block_start = 0; output_channels_block_start < output_channels; output_channels_block_start += output_channels_block_max)
                                {
                                        const size_t output_channels_block_size = min(output_channels - output_channels_block_start, output_channels_block_max);
                                        struct tuple_multiplication_context tuple_multiplication_context = {
                                            .tuple_elements = tuple_elements,
                                            .tuple_size = tuple_size,
                                            .tiles_subblock_max = tiles_subblock_max,
                                            .input_channels_block_start = input_channels_block_start,
                                            .input_channels_block_size = input_channels_block_size,
                                            .output_channels = output_channels,
                                            .output_channels_subblock_max = output_channels_subblock_max,
                                            .output_channels_block_start = output_channels_block_start,
                                            .input_transform = input_transform +
                                                               tuple_index * tiles_count * input_channels_block_size * tuple_size,
                                            .kernel_transform = kernel_transform +
                                                                tuple_index * output_channels * input_channels_block_size * tuple_size,
                                            .output_transform = output_transform +
                                                                tuple_index * tiles_count * output_channels * tuple_size,
                                            .fast_gemm = fast_gemm_function,
                                            .full_gemm = full_gemm_function,
                                        };
                                        tuple_parallelize_2d_tile_2d( // threadpool,  //tuple multiplication - fast gemm function
                                            //           (pthreadpool_task_2d_tile_2d_t) compute_tuple_multiplication,
                                            &tuple_multiplication_context,
                                            tiles_count, output_channels_block_size,
                                            tiles_block_max, output_channels_subblock_max,
                                            PTHREADPOOL_FLAG_DISABLE_DENORMALS);
                                        //				printf("\n after output transformationbefore output\n");
                                        //				for(int i=0;i<10;i++)
                                        //					printf("output transformation = %f %f %f \t ", *(((float *)tuple_multiplication_context.output_transform)+i), *(((float *)tuple_multiplication_context.input_transform)+i), *(((float *)tuple_multiplication_context.kernel_transform)+i));
                                        fprintf(stderr, "after tuple");
                                }
                        }
                        //        NNP_BLOCK_MULTIPLICATION_END(profile)
                }
                //  NNP_OUTPUT_TRANSFORM_START(profile)
                struct output_transform_context output_transform_context = {
                    .transform_function = output_transform_function,
                    .output = output,
                    .output_transform = output_transform,
                    .bias = bias,
                    .tuple_size = tuple_size,
                    .tiles_count = tiles_count,
                    .tiles_x_count = fxdiv_init_size_t(tiles_x_count),
                    .tiles_block_max = fxdiv_init_size_t(tiles_block_max),
                    .output_channels = output_channels,
                    .output_size = output_size,
                    .output_tile = output_tile_size,
                };
                output_parallelize_2d_tile_2d( // threadpool,
                                               //   (pthreadpool_task_2d_tile_2d_t) compute_output_transform,
                    &output_transform_context,
                    output_channels, tiles_count,
                    output_channels_subblock_max, tiles_subblock_max,
                    PTHREADPOOL_FLAG_DISABLE_DENORMALS);

                fprintf(stderr, "after output transform\n");
                //  NNP_OUTPUT_TRANSFORM_END(profile);
                //				printf("\n after output\n");
                //				for(int i=0;i<10;i++)
                //					printf("output = %f\t ", *(((float *)output)+i));

                break;
        }
        case nnp_convolution_transform_strategy_precompute:
        {
                const size_t kernel_transform_size = output_channels * input_channels * transform_tile_size;
                if (workspace_buffer == NULL)
                {
                        *workspace_size = kernel_transform_size;
                        return nnp_status_success;
                }
                else
                {
                        if (*workspace_size < kernel_transform_size)
                        {
                                return nnp_status_insufficient_buffer;
                        }
                        memory_block = workspace_buffer;
                }
                for (size_t input_channels_block_start = 0; input_channels_block_start < input_channels; input_channels_block_start += input_channels_block_max)
                {
                        const size_t input_channels_block_size = min(input_channels - input_channels_block_start, input_channels_block_max);

                        // NNP_KERNEL_TRANSFORM_START(profile)
                        struct kernel_transform_context kernel_transform_context = {
                            .transform_function = kernel_transform_function,
                            .kernel = kernel + input_channels_block_start * kernel_size.height * kernel_size.width,
                            .kernel_transform = (void *)workspace_buffer + input_channels_block_start * output_channels * transform_tile_size,
                            //.kernel_transform = (void*) workspace_buffer + input_channels_block_start * output_channels * tile_elements,
                            .tuple_size = tuple_size,
                            .input_channels = input_channels,
                            .input_channels_block_size = input_channels_block_size,
                            .output_channels = output_channels,
                            .kernel_size = kernel_size,
                        };
                        kernel_parallelize_2d_tile_2d_intertile( // threadpool,
                                                                 //          (pthreadpool_task_2d_tile_2d_t1) compute_kernel_transform,
                            &kernel_transform_context,
                            output_channels, input_channels_block_size,
                            output_channels_subblock_max, 1, nnp_hwinfo.globalinterchannels,
                            PTHREADPOOL_FLAG_DISABLE_DENORMALS);

                        // pthreadpool_parallelize_2d_tile_2d(threadpool,
                        //       (pthreadpool_task_2d_tile_2d_t) compute_kernel_transform,
                        //       &kernel_transform_context,
                        //       output_channels,              input_channels_block_size,
                        //       output_channels_subblock_max, 1,
                        //       PTHREADPOOL_FLAG_DISABLE_DENORMALS);
                        // NNP_KERNEL_TRANSFORM_END(profile)
                }
                break;
        }
        default:
                return nnp_status_invalid_transform_strategy;
        }
        if (memory_block != workspace_buffer)
        {
                release_memory(memory_block, memory_size);
        }
        return nnp_status_success;
}

enum nnp_status nnp_convolution_inference(
    enum nnp_convolution_algorithm algorithm,
    enum nnp_convolution_transform_strategy transform_strategy,
    size_t input_channels,
    size_t output_channels,
    struct nnp_size input_size,
    struct nnp_padding input_padding,
    struct nnp_size kernel_size,
    struct nnp_size output_subsampling,
    const float *input,
    const float *kernel,
    const float *bias,
    float *output,
    void *workspace_buffer,
    size_t *workspace_size,
    enum nnp_activation activation,
    const void *activation_parameters,
    pthreadpool_t threadpool,
    struct nnp_profile *profile)
{
        fprintf(stderr, "I am in 1");
        struct nnp_size tile_size;
        size_t transform_element_size;
        bool fourier_transform;
        nnp_transform_2d_with_offset input_transform_function = NULL;
        nnp_transform_2d_with_offset kernel_transform_function = NULL;
        nnp_transform_2d_with_bias output_transform_function = NULL;
        fourier_transform = false;

        const struct nnp_size output_size = {
            .width = (input_padding.left + input_size.width + input_padding.right - kernel_size.width) / output_subsampling.width + 1,
            .height = (input_padding.top + input_size.height + input_padding.bottom - kernel_size.height) / output_subsampling.height + 1};
        tile_size = (struct nnp_size){.height = 8, .width = 8};
        transform_element_size = sizeof(float);
        input_transform_function = nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream;
        kernel_transform_function = nnp_hwinfo.transforms.kwt_f6x6_3x3;

        if (output_subsampling.height == 1 && output_subsampling.width == 1)
        {
                output_transform_function = nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias;
        }
        else if (output_subsampling.height == 2 && output_subsampling.width == 2)
        {
                output_transform_function = nnp_hwinfo.transforms.owt_f6x6_3x3s2_with_bias;
        }
        status = compute_fast_convolution_inference(
            fourier_transform, transform_strategy, transform_element_size,
            input_channels, output_channels,
            tile_size, input_size, input_padding, kernel_size, output_size, output_subsampling,
            input, kernel, bias, output, workspace_buffer, workspace_size,
            input_transform_function, kernel_transform_function, output_transform_function,
            threadpool, profile);
}

void init_hwinfo(void)
{
        nnp_hwinfo.simd_width = 4; // original
        nnp_hwinfo.sve_simd_width = __builtin_epi_vsetvlmax(__epi_e32, __epi_m1);
        nnp_hwinfo.globalinterchannels = nnp_hwinfo.sve_simd_width / 4; // 16;
        fprintf(stderr, "vector lebgth =%d", __builtin_epi_vsetvlmax(__epi_e32, __epi_m1));
        // nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_store = (nnp_transform_2d_with_offset) nnp_iwt8x8_3x3_with_offset__neon;
        nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_iwt8x8_3x3_with_offset__neon;
        nnp_hwinfo.transforms.kwt_f6x6_3x3 = (nnp_transform_2d_with_offset)nnp_kwt8x8_3x3__neon;
        nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_with_bias__neon;
        nnp_hwinfo.transforms.owt_f6x6_3x3s2_with_bias = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3s2_with_bias__neon;
        nnp_hwinfo.sxgemm = (struct sxgemm)
        {
                .mr = 16,
                .nr = 16,
#if CPUINFO_ARCH_ARM
#else
                .only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_s4gemm_only_3x3__neon,
#endif
                .upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_s4gemm_upto_3x3__neon,
        };
        if (nnp_hwinfo.sve_simd_width >= 128)
        {
                nnp_hwinfo.sxgemm.mr = 32; // 3 original
                nnp_hwinfo.sxgemm.nr = 32; // 3 original
        }
        else
        {
                nnp_hwinfo.sxgemm.mr = 16;
                nnp_hwinfo.sxgemm.nr = 16;
        }
        init_static_ios_hwinfo();
}
