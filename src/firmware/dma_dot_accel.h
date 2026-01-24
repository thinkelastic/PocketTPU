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
#define DMA_DOT_ADDR_A_NEXT (DMA_DOT_ACCEL_BASE + 0x18)

// Weight cache registers
#define DMA_DOT_CACHE_CTRL   (DMA_DOT_ACCEL_BASE + 0x20)  // [3:0]=slot, [8]=start_load
#define DMA_DOT_CACHE_VALID  (DMA_DOT_ACCEL_BASE + 0x24)  // [15:0]=slot valid bitmask
#define DMA_DOT_CACHE_ADDR   (DMA_DOT_ACCEL_BASE + 0x28)  // SDRAM source address
#define DMA_DOT_CACHE_LEN    (DMA_DOT_ACCEL_BASE + 0x2C)  // Elements to load (up to 4096)
#define DMA_DOT_CACHE_OFFSET (DMA_DOT_ACCEL_BASE + 0x30)  // Row offset within cache slot

// Maximum vector size per batch (increased to 512 for B-caching optimization)
#define DMA_DOT_MAX_LEN   512

// Weight cache constants
#define DMA_CACHE_SLOTS       16
#define DMA_CACHE_SLOT_SIZE   4096  // Max elements per slot

// Memory-mapped register access
#define REG32(addr) (*(volatile uint32_t *)(addr))

// Fixed-point Q16.16 conversion macros
#define FLOAT_TO_Q16(f) ((int32_t)((f) * 65536.0f))
#define Q16_TO_FLOAT(q) ((float)(q) * (1.0f / 65536.0f))
// Result is Q32.32 after multiplication of two Q16.16 numbers
// Use multiply by reciprocal instead of divide - faster on soft CPU
#define Q32_TO_FLOAT(q) ((float)(q) * 2.3283064365386963e-10f)

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

// Control register bits
#define DMA_DOT_CTRL_START       (1 << 0)  // Start operation
#define DMA_DOT_CTRL_USE_CACHED  (1 << 1)  // Use cached B vector
#define DMA_DOT_CTRL_PRELOAD_B   (1 << 2)  // Preload B only, no compute
#define DMA_DOT_CTRL_PIPELINE    (1 << 3)  // Pipeline mode: prefetch next A while computing
#define DMA_DOT_CTRL_USE_CACHE   (1 << 4)  // Use BRAM weight cache for A (skip SDRAM fetch)
#define DMA_DOT_CTRL_STREAMING   (1 << 5)  // Streaming mode: compute as A arrives (no buffer)

// Status register bits (read from CTRL)
#define DMA_DOT_STATUS_BUSY      (1 << 0)  // Accelerator is busy
#define DMA_DOT_STATUS_READY     (1 << 4)  // Ready for next operation (prefetch done)

// Start accelerator computation (normal mode)
static inline void dma_dot_start(void) {
    REG32(DMA_DOT_CTRL) = DMA_DOT_CTRL_START;
}

// Preload B vector into cache (CTRL=5: preload_b_only + start)
static inline void dma_dot_preload_b(void) {
    REG32(DMA_DOT_CTRL) = DMA_DOT_CTRL_START | DMA_DOT_CTRL_PRELOAD_B;
}

// Start computation using cached B (CTRL=3: use_cached_b + start)
static inline void dma_dot_start_cached(void) {
    REG32(DMA_DOT_CTRL) = DMA_DOT_CTRL_START | DMA_DOT_CTRL_USE_CACHED;
}

// Start streaming computation - compute as A arrives, no buffering
// Requires B preloaded. Fastest mode: overlaps DMA with compute completely.
static inline void dma_dot_start_streaming(void) {
    REG32(DMA_DOT_CTRL) = DMA_DOT_CTRL_START | DMA_DOT_CTRL_USE_CACHED | DMA_DOT_CTRL_STREAMING;
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

/*
 * Preload B vector into hardware cache for repeated use.
 * Call this once before calling dma_dot_product_q16_cached() multiple times.
 * Only works for vectors <= DMA_DOT_MAX_LEN (256 elements).
 *
 * Parameters:
 *   b: Pointer to B vector in SDRAM (Q16.16 format)
 *   n: Number of elements (must be <= 256)
 */
static inline void dma_dot_preload_b_vector(int32_t* b, int n) {
    dma_dot_set_addr_b(ptr_to_sdram_word_addr(b));
    dma_dot_set_length(n);
    dma_dot_preload_b();
    dma_dot_wait();
}

/*
 * Compute dot product using cached B vector (preloaded with dma_dot_preload_b_vector).
 * Only fetches vector A from SDRAM - uses previously cached B.
 * Only works for vectors <= DMA_DOT_MAX_LEN (256 elements).
 *
 * Parameters:
 *   a: Pointer to A vector in SDRAM (Q16.16 format)
 *   n: Number of elements (must match preloaded B length)
 *
 * Returns: 64-bit Q32.32 result
 */
static inline int64_t dma_dot_product_q16_cached(int32_t* a, int n) {
    dma_dot_set_addr_a(ptr_to_sdram_word_addr(a));
    dma_dot_set_length(n);
    dma_dot_start_cached();
    dma_dot_wait();
    return dma_dot_get_result();
}

/*
 * ============================================================================
 * PIPELINED DOUBLE-BUFFERING API
 * ============================================================================
 * For maximum throughput, use the pipelined API which overlaps DMA with compute:
 *
 * 1. dma_dot_preload_b_vector(x, n)     - Preload x vector once
 * 2. dma_dot_pipeline_first(w0, w1, n)  - Start first row, prefetch second
 * 3. Loop: result = dma_dot_pipeline_next(w_next, n) - Get result, start next
 * 4. result = dma_dot_pipeline_last(n)  - Get final result
 */

// Set next A address for prefetching
static inline void dma_dot_set_addr_a_next(uint32_t word_addr) {
    REG32(DMA_DOT_ADDR_A_NEXT) = word_addr;
}

// Start pipelined computation: fetch A, compute with cached B, prefetch next A
static inline void dma_dot_start_pipeline(void) {
    REG32(DMA_DOT_CTRL) = DMA_DOT_CTRL_START | DMA_DOT_CTRL_USE_CACHED | DMA_DOT_CTRL_PIPELINE;
}

/*
 * Start first pipelined dot product.
 * Fetches first A, starts compute, prefetches second A.
 */
static inline void dma_dot_pipeline_first(int32_t* a_first, int32_t* a_next, int n) {
    dma_dot_set_addr_a(ptr_to_sdram_word_addr(a_first));
    dma_dot_set_addr_a_next(ptr_to_sdram_word_addr(a_next));
    dma_dot_set_length(n);
    dma_dot_start_pipeline();
}

/*
 * Continue pipelined operation: wait for previous, get result, start next.
 * The next A was already prefetched, so this just starts compute and prefetches a_next.
 */
static inline int64_t dma_dot_pipeline_next(int32_t* a_next, int n) {
    (void)n;  /* Length already set, kept for API consistency */
    dma_dot_wait();
    int64_t result = dma_dot_get_result();

    // Set up next prefetch and start
    dma_dot_set_addr_a_next(ptr_to_sdram_word_addr(a_next));
    dma_dot_start_pipeline();

    return result;
}

/*
 * Finish pipelined operation: wait for last compute, get result.
 * No more prefetching needed.
 */
static inline int64_t dma_dot_pipeline_last(void) {
    dma_dot_wait();
    return dma_dot_get_result();
}

/*
 * ============================================================================
 * BRAM WEIGHT CACHE API
 * ============================================================================
 * For maximum throughput on frequently-used weights, load them into BRAM once
 * at startup. Cache provides single-cycle access vs ~100+ cycle SDRAM DMA.
 *
 * Usage:
 * 1. dma_cache_load(slot, weights_ptr, n)  - Load weights into cache slot
 * 2. dma_cache_select(slot)                - Select active slot
 * 3. dma_dot_start_cached_weights()        - Compute using cached weights + cached B
 */

// Check if cache load is in progress
static inline int dma_cache_busy(void) {
    return REG32(DMA_DOT_CACHE_CTRL) & (1 << 4);  // Bit 4 = cache_load_busy
}

// Wait for cache load to complete
static inline void dma_cache_wait(void) {
    while (dma_cache_busy());
}

// Get bitmask of valid cache slots
static inline uint16_t dma_cache_get_valid(void) {
    return REG32(DMA_DOT_CACHE_VALID) & 0xFFFF;
}

// Select active cache slot (0-15)
static inline void dma_cache_select(int slot) {
    REG32(DMA_DOT_CACHE_CTRL) = slot & 0xF;
}

// Set row offset within cache slot (for matmul: row * n)
static inline void dma_cache_set_row_offset(int offset) {
    REG32(DMA_DOT_CACHE_OFFSET) = offset & 0xFFF;
}

/*
 * Load weights from SDRAM into cache slot.
 * Blocks until load is complete.
 *
 * Parameters:
 *   slot: Cache slot (0-15)
 *   src:  Pointer to weights in SDRAM (Q16.16 format)
 *   n:    Number of elements (up to 4096)
 */
static inline void dma_cache_load(int slot, int32_t* src, int n) {
    // Wait for any previous operation to complete
    dma_dot_wait();
    dma_cache_wait();

    // Set source address and length
    REG32(DMA_DOT_CACHE_ADDR) = ptr_to_sdram_word_addr(src);
    REG32(DMA_DOT_CACHE_LEN) = n;

    // Start load: slot in [3:0], start bit in [8]
    REG32(DMA_DOT_CACHE_CTRL) = (slot & 0xF) | (1 << 8);

    // Wait for load to complete
    dma_cache_wait();
}

/*
 * Start computation using cached weights (from BRAM) and cached B vector.
 * Much faster than SDRAM path - weights read in single cycle.
 * Must have called dma_cache_select() to choose slot first.
 */
static inline void dma_dot_start_cached_weights(void) {
    REG32(DMA_DOT_CTRL) = DMA_DOT_CTRL_START | DMA_DOT_CTRL_USE_CACHED | DMA_DOT_CTRL_USE_CACHE;
}

/*
 * Compute dot product using cached weights from BRAM.
 * B vector fetched from SDRAM (normal path).
 * Weights must have been loaded with dma_cache_load().
 *
 * Parameters:
 *   slot: Cache slot containing weights (0-15)
 *   b:    Pointer to B vector in SDRAM (Q16.16 format)
 *   n:    Number of elements
 *
 * Returns: 64-bit Q32.32 result
 */
static inline int64_t dma_dot_product_cached_weights(int slot, int32_t* b, int n) {
    dma_cache_select(slot);
    dma_dot_set_addr_b(ptr_to_sdram_word_addr(b));
    dma_dot_set_length(n);
    REG32(DMA_DOT_CTRL) = DMA_DOT_CTRL_START | DMA_DOT_CTRL_USE_CACHE;
    dma_dot_wait();
    return dma_dot_get_result();
}

/*
 * Compute dot product using both cached weights (BRAM) and cached B vector.
 * Fastest path - no SDRAM access during compute.
 * B must have been preloaded with dma_dot_preload_b_vector().
 * Weights must have been loaded with dma_cache_load().
 *
 * Parameters:
 *   slot: Cache slot containing weights (0-15)
 *   n:    Number of elements
 *
 * Returns: 64-bit Q32.32 result
 */
static inline int64_t dma_dot_product_full_cached(int slot, int n) {
    dma_cache_select(slot);
    dma_dot_set_length(n);
    dma_dot_start_cached_weights();
    dma_dot_wait();
    return dma_dot_get_result();
}

/*
 * ============================================================================
 * Q8 DIRECT MODE API
 * ============================================================================
 * Q8_0 format: 34 bytes per 32 elements (2-byte FP16 scale + 32 int8 values)
 * Hardware dequantizes Q8 to Q16.16 on-the-fly during DMA fetch.
 * This allows using Q8 weights directly from GGUF without CPU conversion.
 *
 * Q8 byte address calculation:
 *   For element index i, Q8 block is at: (i / 32) * 34 bytes from weight start
 */

// Calculate Q8 byte offset for a row of weights
// Each row of n elements spans (n / 32) Q8 blocks = (n / 32) * 34 bytes
static inline uint32_t q8_row_bytes(int n_elements) {
    return ((n_elements + 31) / 32) * 34;
}

// Convert Q8 data pointer to SDRAM byte address for accelerator
// Note: Q8 uses byte addresses since blocks are 34 bytes (not word-aligned)
static inline uint32_t ptr_to_sdram_byte_addr(void* ptr) {
    return (uintptr_t)ptr - SDRAM_BASE;
}

/*
 * Compute dot product using streaming mode - overlaps DMA with compute.
 * B must be preloaded. Computes as A arrives - no intermediate buffer.
 *
 * Timing: ~n cycles (vs ~1.5n for buffered mode) = ~33% faster
 */
static inline int64_t dma_dot_product_q16_streaming(int32_t* a, int n) {
    dma_dot_set_addr_a(ptr_to_sdram_word_addr(a));
    dma_dot_set_length(n);
    dma_dot_start_streaming();
    dma_dot_wait();
    return dma_dot_get_result();
}

#endif // DMA_DOT_ACCEL_H
