/*
 * VexRiscv Firmware for Analogue Pocket
 * Entry point - select test or llama mode
 */

#include "terminal.h"
#include "dma_dot_accel.h"

/* External entry points */
extern void llama_main(void);
extern void memtest_main(void);

/* Set to 1 for memory test, 0 for llama */
#define RUN_MEMTEST 0
/* Set to 1 to test DMA dot product accelerator */
#define TEST_DMA_ACCEL 0

/* System registers */
#define SYS_BASE      0x40000000UL
#define SYS_STATUS    (*(volatile uint32_t*)(SYS_BASE + 0x00))

/* SDRAM base */
#define SDRAM_BASE    0x10000000UL

#if TEST_DMA_ACCEL
static void test_dma_accel(void) {
    printf("=== DMA ACCEL TEST ===\n");

    /* Wait for SDRAM ready */
    while (!(SYS_STATUS & 1)) {}
    printf("SDRAM ready\n");

    /* Allocate test vectors in SDRAM */
    volatile int32_t* vec_a = (volatile int32_t*)(SDRAM_BASE + 0x100000);  /* 1MB offset */
    volatile int32_t* vec_b = (volatile int32_t*)(SDRAM_BASE + 0x100100);  /* 256 bytes later */

    /* Test: vec_a = [1,2,3,4], vec_b = [1,2,3,4] in Q16.16
     * Dot product = 1*1 + 2*2 + 3*3 + 4*4 = 30
     */
    printf("Writing test vectors to SDRAM...\n");
    for (int i = 0; i < 4; i++) {
        vec_a[i] = FLOAT_TO_Q16((float)(i + 1));
        vec_b[i] = FLOAT_TO_Q16((float)(i + 1));
    }

    /* Verify writes */
    printf("vec_a[0] = 0x%08x (expect 0x10000)\n", (unsigned)vec_a[0]);
    printf("vec_a[1] = 0x%08x (expect 0x20000)\n", (unsigned)vec_a[1]);

    /* Set DMA addresses (word addresses in SDRAM) */
    uint32_t addr_a = (0x100000) >> 2;  /* Word address */
    uint32_t addr_b = (0x100100) >> 2;

    printf("Setting DMA addresses: A=0x%06x B=0x%06x\n", addr_a, addr_b);
    dma_dot_set_addr_a(addr_a);
    dma_dot_set_addr_b(addr_b);
    dma_dot_set_length(4);

    /* Start DMA and computation */
    printf("Starting DMA...\n");
    dma_dot_start();

    /* Wait for completion */
    int timeout = 100000;
    while (dma_dot_busy() && timeout > 0) {
        timeout--;
    }

    if (timeout == 0) {
        printf("TIMEOUT!\n");
    } else {
        /* Read result */
        int64_t result = dma_dot_get_result();
        float fresult = Q32_TO_FLOAT(result);
        printf("Result: %d (expect 30)\n", (int)fresult);
        printf("Raw result: 0x%08x%08x\n",
               (unsigned)(result >> 32), (unsigned)(result & 0xFFFFFFFF));
    }

    printf("=== TEST DONE ===\n\n");
}
#endif

int main(void) {
    term_init();

    printf("VexRiscv on Analogue Pocket\n");
    printf("===========================\n");
    printf("\n");

#if TEST_DMA_ACCEL
    test_dma_accel();
#endif

#if RUN_MEMTEST
    /* Run memory test suite */
    memtest_main();
#else
    /* Run Llama-2 inference */
    llama_main();
#endif

    /* Should not return, but if it does, idle */
    while (1) {
        /* Firmware idle loop */
    }

    return 0;
}
