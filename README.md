# PocketTPU - LLM Inference on Analogue Pocket

A RISC-V core for the Analogue Pocket FPGA capable of running LLM inference for text generation. Features hardware-accelerated dot product operations using DSP blocks and DMA for neural network computation.

## Features

- **VexRiscv RISC-V CPU** - RV32IM processor at 50 MHz with instruction/data caches
- **LLM Inference Engine** - Supports both GPT-2 and LLaMA architectures
- **GGUF Model Support** - Load models in GGUF format directly
- **Q8_0 Quantization** - Hardware dequantizes Q8 weights on-the-fly during DMA
- **DMA Dot Product Accelerator** - Hardware DSP-based matmul with double-buffering
- **8-Element DSP Accelerator** - Dedicated hardware for attention computation
- **BRAM Weight Cache** - 16KB on-chip cache for frequently-used weights
- **64KB BRAM** - Program and data storage
- **64MB SDRAM** - External memory for model weights
- **16MB PSRAM** - Fast KV cache storage
- **40x30 Text Terminal** - 1200 character display with 8x8 font

## Performance

| Configuration       | Speed        |
|---------------------|--------------|
| Software only       | ~24 tok/min  |
| DMA accelerator     | ~150 tok/min |
| DMA + B-caching     | ~225 tok/min |
| DMA + optimizations | ~255 tok/min |

Tested with TinyStories GPT-2 model (dim=64, 6 layers, 1M parameters).

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          Analogue Pocket FPGA                             │
│                        (Cyclone V 5CEBA4F23C8)                            │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────┐    ┌─────────────────────────────────────────────┐  │
│  │   VexRiscv CPU   │    │              Memory Subsystem               │  │
│  │   RV32IM 50MHz   │    │                                             │  │
│  │                  │    │  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │  │
│  │  ┌────┐  ┌────┐  │    │  │  BRAM   │  │  SDRAM  │  │   PSRAM     │  │  │
│  │  │ I$ │  │ D$ │  │    │  │  64KB   │  │  64MB   │  │   16MB      │  │  │
│  │  └────┘  └────┘  │    │  │Firmware │  │ Weights │  │  KV Cache   │  │  │
│  └────────┬─────────┘    │  └─────────┘  └────┬────┘  └─────────────┘  │  │
│           │              │                    │                        │  │
│           │ Bus          └────────────────────┼────────────────────────┘  │
│           │                                   │                           │
│  ┌────────┴───────────────────────────────────┴───────────────────────┐   │
│  │                         Wishbone Bus                               │   │
│  └────────┬───────────────────┬───────────────────┬───────────────────┘   │
│           │                   │                   │                       │
│  ┌────────┴────────┐ ┌────────┴────────┐ ┌────────┴────────┐              │
│  │  DMA Dot Prod   │ │  8-Element DSP  │ │  Text Terminal  │              │
│  │   Accelerator   │ │   Accelerator   │ │     40x30       │              │
│  │                 │ │                 │ │                 │              │
│  │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │              │
│  │ │Weight Cache │ │ │ │ 8 DSP Mults │ │ │ │  Font ROM   │ │              │
│  │ │    16KB     │ │ │ │  Parallel   │ │ │ │    8x8      │ │              │
│  │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │              │
│  │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │              │
│  │ │ 2x DMA Bufs │ │ │ │ Adder Tree  │ │ │ │    VRAM     │ │              │
│  │ │ 512 x 32bit │ │ │ │  Pipelined  │ │ │ │   1.2KB     │ │              │
│  │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │              │
│  │ ┌─────────────┐ │ │                 │ │                 │              │
│  │ │ 2-way MAC   │ │ │  Q16.16 I/O     │ │  720x480 Video  │              │
│  │ │  Pipeline   │ │ │  ~5 cycles      │ │                 │              │
│  │ └─────────────┘ │ │                 │ │                 │              │
│  │  0x50000000     │ │  0x51000000     │ │  0x20000000     │              │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘              │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘

                            Data Flow During Inference

┌──────────┐           ┌──────────┐           ┌──────────┐
│  SDRAM   │───DMA────>│  Weight  │───MAC────>│  Result  │
│ Weights  │           │  Buffer  │           │  Accum   │
│ (Q16.16) │           └──────────┘           └──────────┘
└──────────┘                ▲
                            │
┌──────────┐           ┌──────────┐
│  SDRAM   │───DMA────>│ Q8 Dequant│──────────┘
│ Weights  │           │ (HW)     │
│ (Q8_0)   │           └──────────┘
└──────────┘                ▲
                            │
┌──────────┐                │
│  SDRAM   │───DMA──────────┘
│  Input   │  (B-cached)
└──────────┘

┌──────────┐           ┌──────────┐           ┌──────────┐
│  PSRAM   │───CPU────>│   Dot8   │──────────>│ Attention│
│ KV Cache │           │  Accel   │           │  Scores  │
└──────────┘           └──────────┘           └──────────┘
```

## Quick Start

1. Copy `release/Cores` and `release/Platforms` folders to your Analogue Pocket SD card
2. Place model files in `Assets/pockettpu/common/`:
   - `tinystories-gpt-1M.gguf` (or your model)
3. Power on and select the core from the menu

## Project Structure

```
.
├── src/
│   ├── fpga/                      # FPGA source code
│   │   ├── core/
│   │   │   ├── core_top.v         # Top-level module
│   │   │   ├── cpu_system.v       # VexRiscv + peripherals
│   │   │   ├── dma_dot_product.v  # DMA dot product accelerator
│   │   │   ├── dot8_accel.v       # 8-element DSP accelerator
│   │   │   ├── text_terminal.v    # Text rendering
│   │   │   └── io_sdram.v         # SDRAM controller
│   │   └── vexriscv/
│   │       └── VexRiscv_Full.v    # RISC-V CPU core
│   │
│   └── firmware/                  # RISC-V firmware
│       ├── main.c                 # Entry point
│       ├── llama_embedded.c       # LLM inference engine
│       ├── gguf.c                 # GGUF file parser
│       ├── dma_dot_accel.h        # DMA accelerator driver
│       ├── dot8_accel.h           # 8-element DSP driver
│       └── terminal.c             # Terminal driver
│
├── dist/assets/                   # Model files
│   └── tinystories-gpt-1M.gguf
│
└── release/                       # SD card files
    ├── Cores/ThinkElastic.PocketTPU/
    └── Platforms/
```

## Hardware Architecture

### Resource Utilization (Cyclone V 5CEBA4F23C8)

| Resource      | Used   | Available | Utilization |
|---------------|--------|-----------|-------------|
| ALMs          | 13,662 | 18,480    | 74%         |
| Registers     | 25,683 | -         | -           |
| Block Memory  | 908 KB | 3,154 KB  | 29%         |
| RAM Blocks    | 116    | 308       | 38%         |
| DSP Blocks    | 34     | 66        | 52%         |

### Memory Map

| Address Range | Size  | Description                 |
|---------------|-------|-----------------------------|
| `0x00000000`  | 64KB  | BRAM (firmware)             |
| `0x10000000`  | 64MB  | SDRAM (model weights)       |
| `0x20000000`  | 1.2KB | VRAM (text terminal)        |
| `0x30000000`  | 16MB  | PSRAM (KV cache)            |
| `0x50000000`  | 256B  | DMA Dot Product Accelerator |
| `0x51000000`  | 256B  | 8-Element DSP Accelerator   |

## DMA Dot Product Accelerator

Hardware accelerator for matrix-vector multiplication with DMA from SDRAM.

### Features

- **Double-buffered DMA** - Overlaps memory transfer with computation
- **B-caching** - Preload input vector, reuse for all weight rows
- **Weight Cache** - 16KB BRAM cache for frequently-used weights
- **2-way parallel MAC** - Process 2 elements per cycle
- **Q16.16 fixed-point** - Weights pre-converted for fast computation
- **Q8 hardware dequantization** - Streams Q8_0 weights directly, converts FP16 scale to Q16.16 on-the-fly
- **512-element vectors** - Large batch support

### Register Map (0x50000000)

| Offset | Register     | Description                          |
|--------|--------------|--------------------------------------|
| 0x00   | CTRL         | [0]=start, [1]=use_cached_b,         |
|                       | [2]=preload_b, [3]=pipeline,         |
|                       | [4]=use_cache, [5]=q8_mode           |
| 0x04   | LENGTH       | Vector length (up to 512)            |
| 0x08   | RESULT_LO    | Result bits [31:0]                   |
| 0x0C   | RESULT_HI    | Result bits [63:32]                  |
| 0x10   | ADDR_A       | SDRAM address for weights            |
| 0x14   | ADDR_B       | SDRAM address for input              |
| 0x18   | ADDR_A_NEXT  | Next address for pipelining          |
| 0x20   | CACHE_CTRL   | [3:0]=slot, [4]=busy, [8]=start_load |
| 0x24   | CACHE_VALID  | [15:0]=slot valid bitmask            |
| 0x28   | CACHE_ADDR   | SDRAM source for cache load          |
| 0x2C   | CACHE_LEN    | Elements to load (up to 4096)        |
| 0x30   | CACHE_OFFSET | Row offset within cache slot         |

### Q8 Mode

When bit 5 of CTRL is set, the accelerator reads Q8_0 quantized weights directly from SDRAM and dequantizes them in hardware:

- **Q8_0 format**: 34 bytes per 32 elements (2-byte FP16 scale + 32 int8 values)
- **Hardware conversion**: FP16 scale → Q16.16 using case-based shifter (efficient synthesis)
- **Pipelined**: Scale captured on word 0, dequantization starts on word 1
- **Memory savings**: Q8 weights use ~3.7x less SDRAM than Q16.16

```
Q8 Block (34 bytes):
┌──────────┬────────────────────────────────────────────┐
│ FP16     │        32 x int8 quantized values          │
│ scale    │                                            │
└──────────┴────────────────────────────────────────────┘
   2 bytes                   32 bytes

Dequantization: value[i] = scale_q16 * int8[i]
```

## 8-Element DSP Accelerator

Dedicated hardware for head_size=8 attention dot products.

### Features

- **8 parallel DSP multipliers** - Single-cycle multiply
- **3-stage pipelined adder tree** - ~5 cycle latency
- **Q16.16 fixed-point** - Matches DMA accelerator format

### Register Map (0x51000000)

| Offset | Register  | Description                           |
|--------|-----------|---------------------------------------|
| 0x00   | A_DATA    | Write A[0-7] (auto-increment)         |
| 0x04   | B_DATA    | Write B[0-7], triggers compute on 8th |
| 0x08   | CTRL      | [0]=busy, [1]=reset_idx               |
| 0x0C   | RESULT_LO | Result bits [31:0]                    |
| 0x10   | RESULT_HI | Result bits [63:32]                   |

## Supported Models

### GGUF Format (Recommended)

- TinyStories GPT-2 models
- TinyLlama models (with SentencePiece tokenizer)
- Custom GGUF models with GPT-2 or LLaMA architecture

### Model Requirements

- Must fit in 64MB SDRAM
- Supported architectures: `gpt2`, `llama`
- Supported quantization: `F16`, `F32`, `Q8_0`
- Recommended: dim ≤ 512, layers ≤ 12
- Q8_0 models use ~3.7x less memory than F32

## Configuration

Default settings in `llama_embedded.c`:

```c
#define DEFAULT_STEPS       64      // Max tokens to generate
#define DEFAULT_TEMPERATURE 0.5f    // Sampling temperature
#define DEFAULT_TOPP        1.0f    // Top-p (1.0 = disabled)
#define DEFAULT_PROMPT      "Once upon a time"
```

## Building

### Prerequisites

```bash
# RISC-V toolchain
sudo pacman -S riscv64-elf-gcc  # Arch Linux

# Intel Quartus Prime 25.1+
```

### Firmware

```bash
cd src/firmware
make clean all
```

### FPGA

```bash
cd src/fpga
make build      # Full compilation (~6 min)
make quick      # Update firmware only (fast)
make program    # Program via JTAG
```

## Development

### Fast Iteration Workflow

```bash
# Edit firmware, rebuild, and program
cd src/fpga && make quick
```

This updates only the firmware MIF and reprograms without full synthesis.

### Capture Output

```bash
# OCR the Pocket screen via HDMI capture
tools/capture_ocr.sh
```

## License

- **VexRiscv**: MIT License (SpinalHDL)
- **llama2.c**: MIT License (Andrej Karpathy)
- **PocketTPU Core**: MIT License

## Acknowledgments

- [karpathy/llama2.c](https://github.com/karpathy/llama2.c) - Original LLaMA inference implementation
- [SpinalHDL/VexRiscv](https://github.com/SpinalHDL/VexRiscv) - RISC-V CPU core
- [Analogue](https://www.analogue.co/developer) - Pocket development framework
