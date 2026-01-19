/*
 * VexRiscv Firmware for Analogue Pocket
 * Entry point - select test or llama mode
 */

#include "terminal.h"
#include "dot_accel.h"

/* External entry points */
extern void llama_main(void);
extern void memtest_main(void);

/* Set to 1 for memory test, 0 for llama */
#define RUN_MEMTEST 0
/* Set to 1 to test dot product accelerator */
#define TEST_DOT_ACCEL 0

#if TEST_DOT_ACCEL
static void test_dot_accel(void) {
    printf("=== DOT ACCEL TEST ===\n");

    /* Test: vec_a = [1,2,3,4], vec_b = [1,2,3,4]
     * Dot product = 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
     * In Q16.16: 30 * 65536 = 1966080 = 0x1E0000
     * But result is Q32.32 (two Q16.16 multiplied), so need Q32_TO_FLOAT
     */

    /* Set length */
    REG32(DOT_ACCEL_LENGTH) = 4;

    /* Load vectors: a = [1,2,3,4], b = [1,2,3,4] */
    for (int i = 0; i < 4; i++) {
        REG32(DOT_ACCEL_VEC_A + i * 4) = FLOAT_TO_Q16((float)(i + 1));
        REG32(DOT_ACCEL_VEC_B + i * 4) = FLOAT_TO_Q16((float)(i + 1));
    }

    /* Start computation */
    REG32(DOT_ACCEL_CTRL) = 1;

    /* Wait for completion */
    int timeout = 10000;
    while ((REG32(DOT_ACCEL_CTRL) & 1) && timeout > 0) {
        timeout--;
    }

    if (timeout == 0) {
        printf("TIMEOUT!\n");
    } else {
        /* Read result */
        int64_t result = dot_accel_get_result();
        float fresult = Q32_TO_FLOAT(result);
        printf("Result: %d (expect 30)\n", (int)fresult);
    }

    printf("=== TEST DONE ===\n\n");
}
#endif

int main(void) {
    term_init();

    printf("VexRiscv on Analogue Pocket\n");
    printf("===========================\n");
    printf("\n");

#if TEST_DOT_ACCEL
    test_dot_accel();
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
