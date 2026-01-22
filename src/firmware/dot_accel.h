/*
 * Dot Product Accelerator Driver
 * Hardware-accelerated fixed-point dot product using DSP blocks
 */

#ifndef DOT_ACCEL_H
#define DOT_ACCEL_H

#include <stdint.h>
#include <stddef.h>

// Accelerator base address
#define DOT_ACCEL_BASE   0x50000000UL

// Register offsets
#define DOT_ACCEL_CTRL      (DOT_ACCEL_BASE + 0x00)
#define DOT_ACCEL_LENGTH    (DOT_ACCEL_BASE + 0x04)
#define DOT_ACCEL_RESULT_LO (DOT_ACCEL_BASE + 0x08)
#define DOT_ACCEL_RESULT_HI (DOT_ACCEL_BASE + 0x0C)
#define DOT_ACCEL_VEC_A     (DOT_ACCEL_BASE + 0x10)
#define DOT_ACCEL_VEC_B     (DOT_ACCEL_BASE + 0x50)

// Maximum vector size per batch
#define DOT_ACCEL_VEC_SIZE  16

// Memory-mapped register access
#define REG32(addr) (*(volatile uint32_t *)(addr))

// Fixed-point Q16.16 conversion macros
#define FLOAT_TO_Q16(f) ((int32_t)((f) * 65536.0f))
#define Q16_TO_FLOAT(q) ((float)(q) * (1.0f / 65536.0f))
// Result is Q32.32 after multiplication of two Q16.16 numbers
// Use multiply by reciprocal instead of divide - faster on soft CPU
#define Q32_TO_FLOAT(q) ((float)(q) * 2.3283064365386963e-10f)

// Check if accelerator is busy
static inline int dot_accel_busy(void) {
    return REG32(DOT_ACCEL_CTRL) & 1;
}

// Wait for accelerator to finish
static inline void dot_accel_wait(void) {
    while (dot_accel_busy()) {
        // Spin
    }
}

// Start accelerator computation
static inline void dot_accel_start(void) {
    REG32(DOT_ACCEL_CTRL) = 1;
}

// Set vector length
static inline void dot_accel_set_length(int len) {
    REG32(DOT_ACCEL_LENGTH) = len;
}

// Load a value into vector A
static inline void dot_accel_load_a(unsigned int idx, int32_t val) {
    REG32(DOT_ACCEL_VEC_A + idx * 4UL) = val;
}

// Load a value into vector B
static inline void dot_accel_load_b(unsigned int idx, int32_t val) {
    REG32(DOT_ACCEL_VEC_B + idx * 4UL) = val;
}

// Get result (64-bit)
static inline int64_t dot_accel_get_result(void) {
    int64_t lo = REG32(DOT_ACCEL_RESULT_LO);
    int64_t hi = REG32(DOT_ACCEL_RESULT_HI);
    return lo | (hi << 32);
}

/*
 * Compute dot product of two float vectors using accelerator
 * Handles vectors larger than DOT_ACCEL_VEC_SIZE by batching
 * Returns float result
 */
static inline float dot_product_accel(float *a, float *b, int n) {
    int64_t total = 0;
    int offset = 0;

    while (offset < n) {
        int batch = (n - offset > DOT_ACCEL_VEC_SIZE) ? DOT_ACCEL_VEC_SIZE : (n - offset);

        // Load vectors (convert float to Q16.16)
        for (int i = 0; i < batch; i++) {
            dot_accel_load_a(i, FLOAT_TO_Q16(a[offset + i]));
            dot_accel_load_b(i, FLOAT_TO_Q16(b[offset + i]));
        }

        // Set length and start
        dot_accel_set_length(batch);
        dot_accel_start();

        // Wait for completion
        dot_accel_wait();

        // Accumulate result
        total += dot_accel_get_result();

        offset += batch;
    }

    // Convert from Q32.32 to float
    return Q32_TO_FLOAT(total);
}

#endif // DOT_ACCEL_H
