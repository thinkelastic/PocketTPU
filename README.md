# VexRiscv RISC-V Core for Analogue Pocket

A demonstration core featuring a VexRiscv RISC-V CPU running on the Analogue Pocket FPGA, capable of running LLaMA-2 inference for text generation. The CPU executes custom firmware and displays output on a 40x30 character text terminal.

## Features

- **VexRiscv RISC-V CPU** - Open source RV32IM processor with instruction/data caches
- **LLaMA-2 Inference** - Run small language models directly on the Pocket
- **Dot Product Accelerator** - Hardware DSP-based vector operations for neural network inference
- **64KB BRAM** - Program and data storage
- **64MB SDRAM** - External memory for model weights and heap
- **16MB PSRAM** - Fast pseudo-SRAM for KV cache
- **40x30 Text Terminal** - 1200 character display with 8x8 font
- **Printf Support** - Standard printf-style formatted output
- **320x240 @ 60Hz** - Native video output
- **Fully Open Source** - Redistributable under MIT license

## Quick Start

1. Copy the `release/Cores` and `release/Platforms` folders to your Analogue Pocket SD card root
2. Power on your Analogue Pocket
3. Navigate to Cores menu and select "Homebrew"
4. The core will display output on screen

## Project Structure

```
.
├── src/
│   ├── fpga/                     # FPGA source code
│   │   ├── apf/                  # Analogue Pocket Framework
│   │   ├── core/                 # Core implementation
│   │   │   ├── core_top.v        # Top-level module
│   │   │   ├── cpu_system.v      # VexRiscv + RAM + peripherals
│   │   │   ├── dot_product_accel.v # Hardware dot product accelerator
│   │   │   ├── text_terminal.v   # Text rendering
│   │   │   └── io_sdram.v        # SDRAM controller
│   │   └── vexriscv/             # VexRiscv CPU core
│   │       └── VexRiscv_Full.v
│   │
│   └── firmware/                 # RISC-V firmware
│       ├── main.c                # Main program
│       ├── terminal.c            # Terminal driver
│       ├── terminal.h            # Terminal API
│       ├── llama_embedded.c      # LLM inference engine
│       ├── dot_accel.h           # Dot product accelerator driver
│       ├── start.S               # Startup code
│       ├── linker.ld             # Linker script
│       └── Makefile              # Build system
│
├── release/                      # Ready for SD card
│   ├── Cores/ThinkElastic.Homebrew/
│   └── Platforms/
│
├── dist/                         # Distribution assets
│   └── platforms/
│
└── *.json                        # Core configuration files
```

## Technical Specifications

### Hardware

| Parameter | Value |
|-----------|-------|
| Target FPGA | Cyclone V 5CEBA4F23C8 |
| CPU Clock | 50 MHz |
| Video Output | 320x240 @ 60Hz |
| Program RAM | 64KB BRAM |
| External RAM | 64MB SDRAM |
| PSRAM | 16MB (CRAM0) |
| VRAM | 1200 bytes (40x30 chars) |

### VexRiscv Configuration

- RV32IM instruction set (base integer + multiply/divide)
- 32 registers (x0-x31)
- 5-stage pipeline
- Instruction and data caches
- Wishbone bus interface
- Hardware multiply/divide

### Memory Map

| Address Range | Size | Description |
|--------------|------|-------------|
| `0x00000000 - 0x0000FFFF` | 64KB | BRAM (program + data) |
| `0x10000000 - 0x13FFFFFF` | 64MB | SDRAM |
| `0x20000000 - 0x200004AF` | 1200B | VRAM (text terminal) |
| `0x30000000 - 0x30FFFFFF` | 16MB | PSRAM (CRAM0) |
| `0x50000000 - 0x500000FF` | 256B | Dot Product Accelerator |

## LLaMA-2 Inference

This core includes an embedded implementation of LLaMA-2 inference, adapted from [karpathy/llama2.c](https://github.com/karpathy/llama2.c).

### How It Works

- **Model Loading**: Model weights are loaded from the SD card via Analogue's data slot system into SDRAM
- **Tokenizer**: BPE tokenizer with vocabulary embedded in firmware
- **Inference**: Full transformer forward pass with attention, RMSNorm, and SwiGLU FFN
- **Sampling**: Temperature and top-p (nucleus) sampling for text generation
- **KV Cache**: Key/value cache stored in PSRAM for faster random access during attention

### Memory Layout

| Region | Address | Usage |
|--------|---------|-------|
| Model Weights | `0x10000000` | Transformer parameters from .bin file |
| Tokenizer Data | `0x12000000` | Vocabulary and scores |
| Runtime Heap | `0x12100000` | Activations, logits |
| KV Cache | `0x30000000` | Key/value cache in PSRAM |

### Supported Models

The implementation supports small LLaMA-2 architecture models that fit in 64MB SDRAM:
- stories15M (15M parameters)
- stories42M (42M parameters)
- stories110M (110M parameters)

### Configuration

Default settings in `llama_embedded.c`:
- `DEFAULT_STEPS`: 64 tokens max generation
- `DEFAULT_TEMPERATURE`: 1.0
- `DEFAULT_TOPP`: 0.9 (nucleus sampling)
- `DEFAULT_PROMPT`: "Once upon a time"
- `USE_DOT_ACCEL`: 1 (enable hardware accelerator)

## DMA Dot Product Accelerator

A custom hardware accelerator for dot product operations using DMA from SDRAM, used to speed up matrix-vector multiplications in the neural network.

### Hardware Design

- **DMA burst reads** directly from SDRAM for zero-copy vector access
- **512-element vector buffers** in BRAM for large batch processing
- **Q16.16 fixed-point arithmetic** for weights and activations
- **64-bit accumulator** to handle overflow from large vectors
- **3-stage pipelined MAC** for high throughput
- **B-caching optimization** - preload input vector once, reuse for all weight rows

### Performance

| Configuration | Speed |
|--------------|-------|
| Software only | 24 tok/min |
| DMA accelerator | 77 tok/min |
| DMA + B-caching | 84 tok/min |

### Register Map

| Offset | Register | Description |
|--------|----------|-------------|
| `0x00` | CTRL | Control/status: bit 0=start, bit 1=use_cached_b, bit 2=preload_b_only |
| `0x04` | LENGTH | Vector length (up to 512 elements) |
| `0x08` | RESULT_LO | Low 32 bits of 64-bit result |
| `0x0C` | RESULT_HI | High 32 bits of 64-bit result |
| `0x10` | ADDR_A | SDRAM word address for vector A |
| `0x14` | ADDR_B | SDRAM word address for vector B |

### Usage

```c
#include "dma_dot_accel.h"

// Preload x vector for B-caching (call once per matmul)
dma_dot_preload_b_vector(x_q16, n);

// Compute dot products using cached B
for (int i = 0; i < d; i++) {
    int64_t result = dma_dot_product_q16_cached(weights + i*n, n);
    output[i] = Q32_TO_FLOAT(result);
}
```

The B-caching optimization preloads the input vector once into hardware BRAM, then reuses it for all weight rows in the matmul. This reduces DMA bandwidth by ~50% for the input vector fetches.

## Firmware Development

### Prerequisites

Install the RISC-V toolchain:

```bash
# Arch Linux
sudo pacman -S riscv64-elf-gcc

# Ubuntu/Debian
sudo apt install gcc-riscv64-unknown-elf

# macOS (Homebrew)
brew install riscv64-elf-gcc
```

### Building Firmware

```bash
cd src/firmware
make clean all
make install    # Copy MIF to FPGA core directory
```

### Example Program

```c
#include "terminal.h"

int main(void) {
    term_init();

    printf("VexRiscv on Analogue Pocket\n");
    printf("===========================\n");
    printf("\n");
    printf("Hello World!\n");

    while (1) {
        // Main loop
    }

    return 0;
}
```

### Terminal API

```c
void term_init(void);              // Initialize terminal
void term_clear(void);             // Clear screen
void term_setpos(int row, int col); // Set cursor position
void term_putchar(char c);         // Write single character
void term_puts(const char *s);     // Write string
void term_println(const char *s);  // Write string with newline
void term_printf(const char *fmt, ...); // Formatted output

// Printf supports: %d, %u, %x, %X, %s, %c, %%, width specifiers
#define printf term_printf         // Use printf() syntax
```

## Building the FPGA

### Prerequisites

- Intel Quartus Prime 25.1 or later
- Analogue Pocket development files

### Build Steps

1. Open `src/fpga/ap_core.qpf` in Quartus
2. Run compilation (Processing > Start Compilation)
3. The bitstream will be generated in `src/fpga/output_files/`

### Packaging for Analogue Pocket

After compilation, the release folder contains everything needed:
- `release/Cores/ThinkElastic.Homebrew/` - Core files
- `release/Platforms/` - Platform definition

Copy both folders to your SD card root.

## Key Design Decisions

### CPU Selection

VexRiscv was chosen over PicoRV32 for:
- Hardware multiply/divide support (RV32IM)
- Instruction and data caches for better performance
- Pipelined design (50 MHz clock)
- Well-tested in LiteX projects

### Video Timing

Native 320x240 @ 60Hz output with 12.288 MHz pixel clock. The text terminal uses an 8x8 pixel font, giving 40 columns x 30 rows = 1200 characters.

### Memory Architecture

- Firmware BRAM uses altsyncram with MIF initialization
- SDRAM provides 64MB for model weights and dynamic heap
- PSRAM (CRAM0) provides 16MB for KV cache with faster random access
- VRAM is memory-mapped and directly written by CPU
- Wishbone bus with arbiter for instruction/data access
- Dot product accelerator mapped to uncacheable I/O region (0x5xxxxxxx)

## License

- **VexRiscv**: MIT License (SpinalHDL)
- **Core Implementation**: ThinkElastic
- **Analogue Pocket Framework**: Provided by Analogue

## Resources

- [VexRiscv GitHub](https://github.com/SpinalHDL/VexRiscv)
- [Analogue Developer Portal](https://www.analogue.co/developer)
- [RISC-V Specifications](https://riscv.org/specifications/)
