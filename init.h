#pragma once
#include<stdio.h>
#include<stdlib.h>
#include"fxdiv.h"
#include <stddef.h>
#include <assert.h>
#include<sys/time.h>
#include<stdbool.h>
#define NNP_ALIGN(alignment) __attribute__((__aligned__(alignment)))
#define NNP_SIMD_ALIGN NNP_ALIGN(64)
#define NNP_CACHE_ALIGN NNP_ALIGN(64)
#define nnp_convolution_transform_strategy_tuple_based nnp_convolution_transform_strategy_compute
#if defined(__linux__) || defined(__native_client__)
        #include <time.h>
        #include <unistd.h>
        #include <sys/mman.h>
#elif defined(__MACH__)
        #include <mach/mach.h>
        #include <mach/mach_time.h>
#elif defined(EMSCRIPTEN)
        #include <emscripten.h>
#endif
typedef struct pthreadpool* pthreadpool_t;
#define PTHREADPOOL_FLAG_DISABLE_DENORMALS 0x00000001


void init_hwinfo(void);
struct nnp_padding {
        /** Padding above the image data */
        size_t top;
        /** Padding on the right of image data */
        size_t right;
        /** Padding below the image data */
        size_t bottom;
        /** Padding on the left of image data */
        size_t left;
};

//enum nnp_status status;
enum nnp_convolution_algorithm {
        /** Let NNPACK choose the algorithm depending on layer parameters */
        nnp_convolution_algorithm_auto = 0,
        /** Tiled convolution based on 2D Fourier transform with 8x8 blocks. Supports kernels up to 8x8. */
        nnp_convolution_algorithm_ft8x8 = 1,
        /** Tiled convolution based on 2D Fourier transform with 16x16 blocks. Supports kernels up to 16x16. */
        nnp_convolution_algorithm_ft16x16 = 2,
        /** Tiled convolution based on 2D Winograd transform F(3x3, 6x6) with 8x8 blocks. Supports only 3x3 kernels. */
        nnp_convolution_algorithm_wt8x8 = 3,
        /** Direct convolution via implicit GEMM. */
        nnp_convolution_algorithm_implicit_gemm = 4,
        /** Direct convolution implementation. */
        nnp_convolution_algorithm_direct = 5,
        /**
         * Tiled convolution based on 2D Winograd transform F(3x3, 6x6) with 8x8 blocks in FP16.
         * Supports only 3x3 kernels. Implemented only for new ARM processors (with NEON-HP),
         * on non-supported processors falls back to nnp_convolution_algorithm_wt8x8.
         */
        nnp_convolution_algorithm_wt8x8_fp16 = 6,
};
enum nnp_activation {
        /** Identity activation f(x) := x, i.e. no transformation */
        nnp_activation_identity = 0,
        /** ReLU activation f(x) := max(0, x) */
        nnp_activation_relu = 1,
};
typedef void (*nnp_transform_2d_with_bias)(const void*, void*, const void*, size_t, size_t, uint32_t, uint32_t);
typedef void (*nnp_transform_2d_with_offset)(const void*, void*, size_t, size_t, uint32_t, uint32_t, uint32_t, uint32_t);

struct transforms {




       
        nnp_transform_2d_with_offset iwt_f6x6_3x3_with_offset_and_stream;
        nnp_transform_2d_with_offset kwt_f6x6_3x3;
#if !NNP_INFERENCE_ONLY
      
        nnp_transform_2d_with_offset owt_f6x6_3x3;
#endif
        nnp_transform_2d_with_bias owt_f6x6_3x3_with_bias;
        nnp_transform_2d_with_bias owt_f6x6_3x3s2_with_bias;


};



enum nnp_status {
	/** The call succeeded, and all output arguments now contain valid data. */
	nnp_status_success = 0,
	/** NNPACK function was called with batch_size == 0. */
	nnp_status_invalid_batch_size = 2,
	/** NNPACK function was called with channels == 0. */
	nnp_status_invalid_channels = 3,
	/** NNPACK function was called with input_channels == 0. */
	nnp_status_invalid_input_channels = 4,
	/** NNPACK function was called with output_channels == 0. */
	nnp_status_invalid_output_channels = 5,
	/** NNPACK function was called with input_size.height == 0 or input_size.width == 0 */
	nnp_status_invalid_input_size = 10,
	/** NNPACK function was called with input_stride.height == 0 or input_stride.width == 0 */
	nnp_status_invalid_input_stride = 11,
	/** NNPACK function was called with input_padding not less than respective kernel (or pooling) size, i.e.:
	 *
	 *  - input_padding.left   >= kernel_size.width  (>= pooling_size.width)
	 *  - input_padding.right  >= kernel_size.width  (>= pooling_size.width)
	 *  - input_padding.top    >= kernel_size.height (>= pooling_size.height)
	 *  - input_padding.bottom >= kernel_size.height (>= pooling_size.height)
	 */
	nnp_status_invalid_input_padding = 12,
	/** NNPACK function was called with kernel_size.height == 0 or kernel_size.width == 0 */
	nnp_status_invalid_kernel_size = 13,
	/** NNPACK function was called with pooling_size.height == 0 or pooling_size.width == 0 */
	nnp_status_invalid_pooling_size = 14,
	/** NNPACK function was called with pooling_stride.height == 0 or pooling_stride.width == 0 */
	nnp_status_invalid_pooling_stride = 15,
	/** NNPACK function was called with convolution algorithm not in nnp_convolution_algorithm enumeration */
	nnp_status_invalid_algorithm = 16,
	/** NNPACK function was called with convolution transform strategy not in nnp_convolution_transform_strategy enum */
	nnp_status_invalid_transform_strategy = 17,
	/** NNPACK function was called with output_subsampling.height == 0 or output_subsampling.width == 0 */
	nnp_status_invalid_output_subsampling = 13,
	/** NNPACK function was called with activation not in nnp_activation enum */
	nnp_status_invalid_activation = 14,
	/** NNPACK function was called with invalid activation parameters */
	nnp_status_invalid_activation_parameters = 15,

	/** NNPACK does not support the particular input size for the function */
	nnp_status_unsupported_input_size = 20,
	/** NNPACK does not support the particular input stride for the function */
	nnp_status_unsupported_input_stride = 21,
	/** NNPACK does not support the particular input padding for the function */
	nnp_status_unsupported_input_padding = 22,
	/** NNPACK does not support the particular kernel size for the function */
	nnp_status_unsupported_kernel_size = 23,
	/** NNPACK does not support the particular pooling size for the function */
	nnp_status_unsupported_pooling_size = 24,
       nnp_status_unsupported_pooling_stride = 25,
        /** NNPACK does not support the particular convolution algorithm for the function */
        nnp_status_unsupported_algorithm = 26,
        /** NNPACK does not support the particular convolution transform strategy for the algorithm */
        nnp_status_unsupported_transform_strategy = 27,
        /** NNPACK does not support the particular activation function for the function */
        nnp_status_unsupported_activation = 28,
        /** NNPACK does not support the particular activation function parameters for the function */
        nnp_status_unsupported_activation_parameters = 29,

        /** NNPACK function was called before the library was initialized */
        nnp_status_uninitialized = 50,
        /** NNPACK does not implement this function for the host CPU */
        nnp_status_unsupported_hardware = 51,
        /** NNPACK failed to allocate memory for temporary buffers */
        nnp_status_out_of_memory = 52,
        /** Scratch space buffer is too small */
        nnp_status_insufficient_buffer = 53,
        /** Scratch space buffer is not properly aligned */
        nnp_status_misaligned_buffer = 54
};

typedef void (*nnp_transform_2d_with_bias)(const void*, void*, const void*, size_t, size_t, uint32_t, uint32_t);


typedef void (*nnp_fast_tuple_gemm_function)(size_t, size_t, const void*, const void*, void*, size_t);
typedef void (*nnp_full_tuple_gemm_function)(uint32_t, uint32_t, size_t, size_t, const void*, const void*, void*, size_t);

struct hxgemm {
        int mr;
        int nr;
};


struct sxgemm {
               nnp_fast_tuple_gemm_function only_mr_x_nr;
        nnp_full_tuple_gemm_function upto_mr_x_nr;

        int mr;
        int nr;
};



struct sgemm {
        int mr;
        int nr;
};



struct cxgemm { int mr; int nr; };
struct cache_info {
        int size;
        int associativity;
        int threads;
        bool inclusive;
};

inline static void release_memory(void* memory_block, size_t memory_size) {
#if defined(__linux__)
        if (memory_block != NULL) {
                munmap(memory_block, memory_size);
        }
#else
        free(memory_block);
#endif
}

inline static void* allocate_memory(size_t memory_size) {
#if defined(__linux__)
        #if !defined(__ANDROID__)
                /* Try to use large page TLB */
                void* memory_block = mmap(NULL, memory_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | MAP_HUGETLB, -1, 0);
        #else
                void* memory_block = MAP_FAILED;
        #endif
        if (memory_block == MAP_FAILED) {
                /* Fallback to standard pages */
                memory_block = mmap(NULL, memory_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
                if (memory_block == MAP_FAILED) {
                        return NULL;
                }
        }
        return memory_block;
#else
        void* memory_block = NULL;
        int allocation_result = posix_memalign(&memory_block, 64, memory_size);
        return (allocation_result == 0) ? memory_block : NULL;
#endif
}


enum nnp_convolution_transform_strategy {
        nnp_convolution_transform_strategy_compute = 1,
        nnp_convolution_transform_strategy_precompute = 2,
        nnp_convolution_transform_strategy_reuse = 3
};


static inline size_t doz(size_t a, size_t b) {
        return a > b ? a - b : 0;
}
typedef void (*nnp_full_tuple_gemm_function)(uint32_t, uint32_t, size_t, size_t, const void*, const void*, void*, size_t);

//typedef void (*nnp_transform_2d_with_offset)(const void*, void*, size_t, size_t, int, int, int, int);

static inline size_t min(size_t a, size_t b) {
        return a > b ? b : a;
}

static inline size_t divide_round_up(size_t dividend, size_t divisor) {
        if (dividend % divisor == 0) {
                return dividend / divisor;
        } else {
                return dividend / divisor + 1;
        }
}


static inline size_t round_up(size_t number, size_t factor) {
        return (number + factor - 1) / factor * factor;
}

static inline size_t round_up_by_power_of_2(size_t number, size_t power_of_2_factor) {
        return (number + power_of_2_factor - 1) & ~(power_of_2_factor - 1);
}

static inline size_t round_down(size_t number, size_t factor) {
        return number / factor * factor;
}

struct nnp_size {
        /** Width (horizontal size) of an image, kernel, or pooling filter. */
        size_t width;
        /** Height (vertical size) of an image, kernel, or pooling filter. */
        size_t height;
};




struct cache_hierarchy_info {
        struct cache_info l1;
        struct cache_info l2;
        struct cache_info l3;
        struct cache_info l4;
};


struct cache_blocking_info {
        size_t l1;
        size_t l2;
        size_t l3;
        size_t l4;
};
struct hardware_info {
        bool initialized;
        bool supported;
        int simd_width;
        int sve_simd_width;
        int globalinterchannels;
        struct transforms transforms;

        struct cache_hierarchy_info cache;
        struct cache_blocking_info blocking;

        struct sgemm sgemm;
        struct sxgemm sxgemm;
#if NNP_BACKEND_ARM
        struct hxgemm hxgemm;
#endif /* NNP_BACKEND_ARM */



};

extern struct hardware_info nnp_hwinfo;

static void init_static_ios_hwinfo(void) {
        nnp_hwinfo.cache.l1 = (struct cache_info) {
                .size = 32 * 1024,
                       .associativity = 1,
                       .threads = 1,
                     .inclusive = false,
        };
        nnp_hwinfo.cache.l2 = (struct cache_info) {
                       .size = 1 * 1024 * 1024,
                      .associativity = 1,
                        .threads = 1,
                        .inclusive = false,
        };
        nnp_hwinfo.cache.l3 = (struct cache_info) {
                        .size = 2 * 1024 * 1024,
                        .associativity = 8,
                        .threads = 1,
                        .inclusive = false,
        };
	nnp_hwinfo.blocking.l1 = nnp_hwinfo.cache.l1.size;
	nnp_hwinfo.blocking.l2 = nnp_hwinfo.cache.l2.size;
	nnp_hwinfo.blocking.l3 = nnp_hwinfo.cache.l3.size;
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
        const float* input,
        const float* kernel,
        const float* bias,
        float* output,
        void* workspace_buffer,
        size_t* workspace_size,
        enum nnp_activation activation,
        const void* activation_parameters,
        pthreadpool_t threadpool,
        struct nnp_profile* profile);
