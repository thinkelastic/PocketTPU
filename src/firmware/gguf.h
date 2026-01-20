/*
 * GGUF Format Parser for Embedded Systems
 *
 * Minimal GGUF parser for loading quantized LLM weights.
 * Supports GPT-2 and LLaMA architectures with FP16/Q8_0 weights.
 *
 * Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 */

#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stddef.h>

/* GGUF Magic and Version */
#define GGUF_MAGIC          0x46554747  /* "GGUF" in little-endian */
#define GGUF_VERSION_MIN    2
#define GGUF_VERSION_MAX    3

/* GGUF Value Types */
typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} GGUFType;

/* GGML Tensor Types (quantization formats) */
typedef enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_I8      = 16,
    GGML_TYPE_I16     = 17,
    GGML_TYPE_I32     = 18,
    GGML_TYPE_COUNT,
} GGMLType;

/* Model Architecture Types */
typedef enum {
    GGUF_ARCH_UNKNOWN = 0,
    GGUF_ARCH_LLAMA   = 1,
    GGUF_ARCH_GPT2    = 2,
    GGUF_ARCH_GPTNEOX = 3,
} GGUFArch;

/* GGUF Header (parsed) */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
} GGUFHeader;

/* Model Configuration (extracted from metadata) */
typedef struct {
    GGUFArch arch;
    uint32_t dim;           /* embedding dimension */
    uint32_t hidden_dim;    /* FFN hidden dimension */
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;    /* For MQA/GQA */
    uint32_t vocab_size;
    uint32_t seq_len;       /* max context length */
    uint32_t head_size;     /* dim / n_heads */

    /* GPT-2 specific */
    int has_bias;           /* LayerNorm has bias */

    /* Tokenizer info */
    uint32_t bos_token_id;
    uint32_t eos_token_id;
} GGUFConfig;

/* Tensor Info (parsed from GGUF) */
typedef struct {
    const char* name;       /* Pointer into GGUF data */
    uint32_t n_dims;
    uint64_t dims[4];
    GGMLType type;
    uint64_t offset;        /* Offset from data section start */
    uint64_t size_bytes;    /* Total size in bytes */
} GGUFTensorInfo;

/* GGUF Parser Context */
typedef struct {
    const uint8_t* data;    /* Raw GGUF file data */
    size_t data_size;

    GGUFHeader header;
    GGUFConfig config;

    /* Parsing state */
    const uint8_t* kv_start;        /* Start of KV pairs */
    const uint8_t* tensor_start;    /* Start of tensor infos */
    const uint8_t* data_start;      /* Start of tensor data */

    /* Tokenizer vocabulary (if present) */
    uint64_t vocab_offset;          /* Offset in metadata to tokens array */
    uint64_t vocab_count;           /* Number of tokens */
} GGUFContext;

/* Parse GGUF header and validate */
int gguf_init(GGUFContext* ctx, const uint8_t* data, size_t size);

/* Parse metadata to extract model config */
int gguf_parse_metadata(GGUFContext* ctx);

/* Find tensor by name and get info */
int gguf_find_tensor(GGUFContext* ctx, const char* name, GGUFTensorInfo* info);

/* Get pointer to tensor data (FP16/Q8 format, in SDRAM) */
const void* gguf_get_tensor_data(GGUFContext* ctx, const GGUFTensorInfo* info);

/* Tokenizer functions */
int gguf_get_vocab_string(GGUFContext* ctx, uint32_t token_id, char* out, size_t max_len);

/* FP16 to float conversion */
static inline float fp16_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);

    if (exp == 0) {
        /* Subnormal or zero */
        if (mant == 0) {
            uint32_t result = sign;
            float f;
            __builtin_memcpy(&f, &result, sizeof(f));
            return f;
        }
        /* Subnormal - normalize */
        while ((mant & 0x400) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3FF;
    } else if (exp == 31) {
        /* Inf or NaN */
        uint32_t result = sign | 0x7F800000 | (mant << 13);
        float f;
        __builtin_memcpy(&f, &result, sizeof(f));
        return f;
    }

    uint32_t result = sign | ((exp + 112) << 23) | (mant << 13);
    float f;
    __builtin_memcpy(&f, &result, sizeof(f));
    return f;
}

/* FP16 to Q16.16 fixed-point conversion */
static inline int32_t fp16_to_q16(uint16_t h) {
    float f = fp16_to_float(h);
    return (int32_t)(f * 65536.0f);
}

#endif /* GGUF_H */
