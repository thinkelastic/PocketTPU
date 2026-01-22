/*
 * 8-Element DSP Dot Product Accelerator Driver
 * Dedicated hardware for head_size=8 attention computation
 * Uses 8 parallel DSP multipliers for ~5 cycle dot products
 */

#ifndef DOT8_ACCEL_H
#define DOT8_ACCEL_H

#include <stdint.h>

// Accelerator base address
#define DOT8_ACCEL_BASE   0x51000000UL

// Register offsets
#define DOT8_A_DATA      (DOT8_ACCEL_BASE + 0x00)  // Write A[0-7] (auto-increment)
#define DOT8_B_DATA      (DOT8_ACCEL_BASE + 0x04)  // Write B[0-7], triggers compute on 8th
#define DOT8_CTRL        (DOT8_ACCEL_BASE + 0x08)  // [0]=busy (read), [1]=reset_idx (write)
#define DOT8_RESULT_LO   (DOT8_ACCEL_BASE + 0x0C)  // Low 32 bits of result
#define DOT8_RESULT_HI   (DOT8_ACCEL_BASE + 0x10)  // High 32 bits of result

// Memory-mapped register access
#define REG32(addr) (*(volatile uint32_t *)(addr))

// Fixed-point Q16.16 conversion (shared with dma_dot_accel.h)
#ifndef FLOAT_TO_Q16
#define FLOAT_TO_Q16(f) ((int32_t)((f) * 65536.0f))
#endif
#ifndef Q32_TO_FLOAT
#define Q32_TO_FLOAT(q) ((float)(q) * 2.3283064365386963e-10f)
#endif

// Q8.24 format: 8 bits integer, 24 bits fraction
// Better precision for small values (attention scores are typically < 1.0)
// Range: -128 to +127, Precision: ~6e-8
#define FLOAT_TO_Q8_24(f) ((int32_t)((f) * 16777216.0f))  // 2^24
#define Q8_24_TO_FLOAT(q) ((float)(q) * 5.9604644775390625e-8f)  // 2^-24
// Product of two Q8.24 numbers is Q16.48 (need 64-bit)
// To convert Q16.48 to float: multiply by 2^-48
#define Q16_48_TO_FLOAT(q) ((float)(q) * 3.552713678800501e-15f)  // 2^-48

// Check if accelerator is busy
static inline int dot8_busy(void) {
    return REG32(DOT8_CTRL) & 1;
}

// Wait for accelerator to finish (typically ~5 cycles)
static inline void dot8_wait(void) {
    while (dot8_busy());
}

// Reset write indices (call before loading new A vector)
static inline void dot8_reset(void) {
    REG32(DOT8_CTRL) = 2;
}

// Load A vector (8 elements, Q16.16 format)
// Call once per attention head, reuse for all K vectors
static inline void dot8_load_a(const int32_t* a) {
    REG32(DOT8_A_DATA) = a[0];
    REG32(DOT8_A_DATA) = a[1];
    REG32(DOT8_A_DATA) = a[2];
    REG32(DOT8_A_DATA) = a[3];
    REG32(DOT8_A_DATA) = a[4];
    REG32(DOT8_A_DATA) = a[5];
    REG32(DOT8_A_DATA) = a[6];
    REG32(DOT8_A_DATA) = a[7];
}

// Load A vector from floats (converts to Q16)
static inline void dot8_load_a_float(const float* a) {
    REG32(DOT8_A_DATA) = FLOAT_TO_Q16(a[0]);
    REG32(DOT8_A_DATA) = FLOAT_TO_Q16(a[1]);
    REG32(DOT8_A_DATA) = FLOAT_TO_Q16(a[2]);
    REG32(DOT8_A_DATA) = FLOAT_TO_Q16(a[3]);
    REG32(DOT8_A_DATA) = FLOAT_TO_Q16(a[4]);
    REG32(DOT8_A_DATA) = FLOAT_TO_Q16(a[5]);
    REG32(DOT8_A_DATA) = FLOAT_TO_Q16(a[6]);
    REG32(DOT8_A_DATA) = FLOAT_TO_Q16(a[7]);
}

// Compute dot product: load B and trigger compute
// A must already be loaded. Returns immediately, use dot8_wait() or dot8_get_result().
static inline void dot8_compute(const int32_t* b) {
    REG32(DOT8_B_DATA) = b[0];
    REG32(DOT8_B_DATA) = b[1];
    REG32(DOT8_B_DATA) = b[2];
    REG32(DOT8_B_DATA) = b[3];
    REG32(DOT8_B_DATA) = b[4];
    REG32(DOT8_B_DATA) = b[5];
    REG32(DOT8_B_DATA) = b[6];
    REG32(DOT8_B_DATA) = b[7];  // Triggers compute
}

// Compute from floats (converts B to Q16)
static inline void dot8_compute_float(const float* b) {
    REG32(DOT8_B_DATA) = FLOAT_TO_Q16(b[0]);
    REG32(DOT8_B_DATA) = FLOAT_TO_Q16(b[1]);
    REG32(DOT8_B_DATA) = FLOAT_TO_Q16(b[2]);
    REG32(DOT8_B_DATA) = FLOAT_TO_Q16(b[3]);
    REG32(DOT8_B_DATA) = FLOAT_TO_Q16(b[4]);
    REG32(DOT8_B_DATA) = FLOAT_TO_Q16(b[5]);
    REG32(DOT8_B_DATA) = FLOAT_TO_Q16(b[6]);
    REG32(DOT8_B_DATA) = FLOAT_TO_Q16(b[7]);  // Triggers compute
}

// Get result as 64-bit Q32.32
static inline int64_t dot8_get_result(void) {
    dot8_wait();
    int64_t lo = REG32(DOT8_RESULT_LO);
    int64_t hi = REG32(DOT8_RESULT_HI);
    return lo | (hi << 32);
}

// Get result as float
static inline float dot8_get_result_float(void) {
    return Q32_TO_FLOAT(dot8_get_result());
}

/*
 * Full dot product: load A, load B, compute, return result as float
 * For repeated use with same A, use dot8_load_a_float() once then
 * dot8_compute_float() + dot8_get_result_float() for each B.
 */
static inline float dot8_dot_product(const float* a, const float* b) {
    dot8_reset();
    dot8_load_a_float(a);
    dot8_compute_float(b);
    return dot8_get_result_float();
}

/*
 * Attention-optimized API:
 * 1. dot8_attention_begin(q) - Load Q vector once per head
 * 2. For each K: score = dot8_attention_score(k)
 */
static inline void dot8_attention_begin(const float* q) {
    dot8_reset();
    dot8_load_a_float(q);
}

static inline float dot8_attention_score(const float* k) {
    dot8_compute_float(k);
    return dot8_get_result_float();
}

#endif // DOT8_ACCEL_H
