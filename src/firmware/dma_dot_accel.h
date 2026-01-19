/*
 * DMA Dot Product Accelerator Driver
 * Hardware-accelerated fixed-point dot product using DMA from SDRAM
 */

#ifndef DMA_DOT_ACCEL_H
#define DMA_DOT_ACCEL_H

#include <stdint.h>
#include <stddef.h>

// Accelerator base address
#define DMA_DOT_ACCEL_BASE   0x50000000UL

// Register offsets
#define DMA_DOT_CTRL      (DMA_DOT_ACCEL_BASE + 0x00)
#define DMA_DOT_LENGTH    (DMA_DOT_ACCEL_BASE + 0x04)
#define DMA_DOT_RESULT_LO (DMA_DOT_ACCEL_BASE + 0x08)
#define DMA_DOT_RESULT_HI (DMA_DOT_ACCEL_BASE + 0x0C)
#define DMA_DOT_ADDR_A    (DMA_DOT_ACCEL_BASE + 0x10)
#define DMA_DOT_ADDR_B    (DMA_DOT_ACCEL_BASE + 0x14)

// Maximum vector size per batch
#define DMA_DOT_MAX_LEN   256

// Memory-mapped register access
#define REG32(addr) (*(volatile uint32_t *)(addr))

// Fixed-point Q16.16 conversion macros
#define FLOAT_TO_Q16(f) ((int32_t)((f) * 65536.0f))
#define Q16_TO_FLOAT(q) ((float)(q) / 65536.0f)
// Result is Q32.32 after multiplication of two Q16.16 numbers
#define Q32_TO_FLOAT(q) ((float)(q) / (65536.0f * 65536.0f))

// SDRAM base address (for converting pointers to SDRAM word addresses)
#ifndef SDRAM_BASE
#define SDRAM_BASE 0x10000000UL
#endif

// Convert SDRAM pointer to word address for accelerator
static inline uint32_t ptr_to_sdram_word_addr(void* ptr) {
    return ((uintptr_t)ptr - SDRAM_BASE) >> 2;  // Word address (divide by 4)
}

// Check if accelerator is busy
static inline int dma_dot_busy(void) {
    return REG32(DMA_DOT_CTRL) & 1;
}

// Wait for accelerator to finish
static inline void dma_dot_wait(void) {
    while (dma_dot_busy()) {
        // Spin
    }
}

// Start accelerator computation
static inline void dma_dot_start(void) {
    REG32(DMA_DOT_CTRL) = 1;
}

// Set vector length
static inline void dma_dot_set_length(int len) {
    REG32(DMA_DOT_LENGTH) = len;
}

// Set SDRAM word address for vector A
static inline void dma_dot_set_addr_a(uint32_t word_addr) {
    REG32(DMA_DOT_ADDR_A) = word_addr;
}

// Set SDRAM word address for vector B
static inline void dma_dot_set_addr_b(uint32_t word_addr) {
    REG32(DMA_DOT_ADDR_B) = word_addr;
}

// Get result (64-bit)
static inline int64_t dma_dot_get_result(void) {
    int64_t lo = REG32(DMA_DOT_RESULT_LO);
    int64_t hi = REG32(DMA_DOT_RESULT_HI);
    return lo | (hi << 32);
}

/*
 * Compute dot product of two Q16.16 vectors in SDRAM using DMA accelerator.
 * Vectors must be pre-converted to Q16.16 format.
 * Handles vectors larger than DMA_DOT_MAX_LEN by batching.
 * Returns Q32.32 result (needs Q32_TO_FLOAT to convert to float).
 *
 * Parameters:
 *   a: Pointer to first vector in SDRAM (Q16.16 format)
 *   b: Pointer to second vector in SDRAM (Q16.16 format)
 *   n: Number of elements
 *
 * Returns: 64-bit Q32.32 accumulated result
 */
static inline int64_t dma_dot_product_q16(int32_t* a, int32_t* b, int n) {
    int64_t total = 0;
    int offset = 0;

    while (offset < n) {
        int batch = (n - offset > DMA_DOT_MAX_LEN) ? DMA_DOT_MAX_LEN : (n - offset);

        // Set addresses (word addresses in SDRAM)
        dma_dot_set_addr_a(ptr_to_sdram_word_addr(a + offset));
        dma_dot_set_addr_b(ptr_to_sdram_word_addr(b + offset));

        // Set length and start DMA
        dma_dot_set_length(batch);
        dma_dot_start();

        // Wait for completion
        dma_dot_wait();

        // Accumulate result
        total += dma_dot_get_result();

        offset += batch;
    }

    return total;
}

#endif // DMA_DOT_ACCEL_H
