/*
 * GGUF Format Parser Implementation
 */

#include "gguf.h"
#include "libc/libc.h"

/* Debug output - redirect to terminal */
#define printf term_printf
extern int term_printf(const char* fmt, ...);

/* Read helpers for potentially unaligned SDRAM access */
static inline uint32_t read_u32(const uint8_t* ptr) {
    uintptr_t addr = (uintptr_t)ptr;
    uintptr_t aligned = addr & ~3;
    int offset = addr & 3;

    uint32_t w0 = *(const volatile uint32_t*)aligned;
    if (offset == 0) return w0;

    uint32_t w1 = *(const volatile uint32_t*)(aligned + 4);
    return (w0 >> (offset * 8)) | (w1 << ((4 - offset) * 8));
}

static inline uint64_t read_u64(const uint8_t* ptr) {
    uint32_t lo = read_u32(ptr);
    uint32_t hi = read_u32(ptr + 4);
    return ((uint64_t)hi << 32) | lo;
}

static inline uint16_t read_u16(const uint8_t* ptr) {
    return (uint16_t)(read_u32(ptr) & 0xFFFF);
}

/* Read GGUF string: length (u64) followed by bytes (NOT null-terminated in file) */
static const uint8_t* read_gguf_string(const uint8_t* ptr, const char** str_out, uint64_t* len_out) {
    uint64_t len = read_u64(ptr);
    ptr += 8;
    *str_out = (const char*)ptr;
    *len_out = len;
    return ptr + len;
}

/* Skip a GGUF value based on type */
static const uint8_t* skip_gguf_value(const uint8_t* ptr, GGUFType type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return ptr + 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return ptr + 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return ptr + 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return ptr + 8;
        case GGUF_TYPE_STRING: {
            const char* s;
            uint64_t len;
            return read_gguf_string(ptr, &s, &len);
        }
        case GGUF_TYPE_ARRAY: {
            GGUFType arr_type = (GGUFType)read_u32(ptr);
            ptr += 4;
            uint64_t count = read_u64(ptr);
            ptr += 8;
            for (uint64_t i = 0; i < count; i++) {
                ptr = skip_gguf_value(ptr, arr_type);
            }
            return ptr;
        }
        default:
            return ptr;
    }
}

/* Compare string with length to null-terminated string */
static int strncmp_len(const char* s1, uint64_t len1, const char* s2) {
    size_t len2 = strlen(s2);
    if (len1 != len2) return 1;
    return memcmp(s1, s2, len1);
}

/* Initialize GGUF context and parse header */
int gguf_init(GGUFContext* ctx, const uint8_t* data, size_t size) {
    if (!ctx || !data || size < 24) {
        return -1;
    }

    memset(ctx, 0, sizeof(*ctx));
    ctx->data = data;
    ctx->data_size = size;

    /* Parse header */
    ctx->header.magic = read_u32(data);
    ctx->header.version = read_u32(data + 4);
    ctx->header.n_tensors = read_u64(data + 8);
    ctx->header.n_kv = read_u64(data + 16);

    /* Validate */
    if (ctx->header.magic != GGUF_MAGIC) {
        printf("GGUF: Invalid magic 0x%08X\n", ctx->header.magic);
        return -1;
    }

    if (ctx->header.version < GGUF_VERSION_MIN || ctx->header.version > GGUF_VERSION_MAX) {
        printf("GGUF: Unsupported version %d\n", ctx->header.version);
        return -1;
    }

    printf("GGUF: v%d, %d tensors, %d metadata\n",
           (int)ctx->header.version,
           (int)ctx->header.n_tensors,
           (int)ctx->header.n_kv);

    ctx->kv_start = data + 24;
    return 0;
}

/* Parse metadata key-value pairs to extract model config */
int gguf_parse_metadata(GGUFContext* ctx) {
    if (!ctx || !ctx->kv_start) return -1;

    const uint8_t* ptr = ctx->kv_start;

    /* Default config values */
    ctx->config.arch = GGUF_ARCH_UNKNOWN;
    ctx->config.n_kv_heads = 0;  /* Will default to n_heads if not set */
    ctx->config.has_bias = 1;    /* GPT-2 has bias by default */
    ctx->config.bos_token_id = 1;
    ctx->config.eos_token_id = 2;

    /* Iterate through all KV pairs */
    for (uint64_t i = 0; i < ctx->header.n_kv; i++) {
        const char* key;
        uint64_t key_len;
        ptr = read_gguf_string(ptr, &key, &key_len);

        GGUFType type = (GGUFType)read_u32(ptr);
        ptr += 4;

        /* Check for known keys */
        if (strncmp_len(key, key_len, "general.architecture") == 0) {
            if (type == GGUF_TYPE_STRING) {
                const char* arch;
                uint64_t arch_len;
                ptr = read_gguf_string(ptr, &arch, &arch_len);
                if (strncmp_len(arch, arch_len, "llama") == 0) {
                    ctx->config.arch = GGUF_ARCH_LLAMA;
                } else if (strncmp_len(arch, arch_len, "gpt2") == 0) {
                    ctx->config.arch = GGUF_ARCH_GPT2;
                } else if (strncmp_len(arch, arch_len, "gptneox") == 0) {
                    ctx->config.arch = GGUF_ARCH_GPTNEOX;
                }
                continue;
            }
        }
        /* GPT-2 keys */
        else if (strncmp_len(key, key_len, "gpt2.context_length") == 0 ||
                 strncmp_len(key, key_len, "llama.context_length") == 0) {
            if (type == GGUF_TYPE_UINT32) {
                ctx->config.seq_len = read_u32(ptr);
                ptr += 4;
                continue;
            }
        }
        else if (strncmp_len(key, key_len, "gpt2.embedding_length") == 0 ||
                 strncmp_len(key, key_len, "llama.embedding_length") == 0) {
            if (type == GGUF_TYPE_UINT32) {
                ctx->config.dim = read_u32(ptr);
                ptr += 4;
                continue;
            }
        }
        else if (strncmp_len(key, key_len, "gpt2.feed_forward_length") == 0 ||
                 strncmp_len(key, key_len, "llama.feed_forward_length") == 0) {
            if (type == GGUF_TYPE_UINT32) {
                ctx->config.hidden_dim = read_u32(ptr);
                ptr += 4;
                continue;
            }
        }
        else if (strncmp_len(key, key_len, "gpt2.block_count") == 0 ||
                 strncmp_len(key, key_len, "llama.block_count") == 0) {
            if (type == GGUF_TYPE_UINT32) {
                ctx->config.n_layers = read_u32(ptr);
                ptr += 4;
                continue;
            }
        }
        else if (strncmp_len(key, key_len, "gpt2.attention.head_count") == 0 ||
                 strncmp_len(key, key_len, "llama.attention.head_count") == 0) {
            if (type == GGUF_TYPE_UINT32) {
                ctx->config.n_heads = read_u32(ptr);
                ptr += 4;
                continue;
            }
        }
        else if (strncmp_len(key, key_len, "llama.attention.head_count_kv") == 0 ||
                 strncmp_len(key, key_len, "gpt2.attention.head_count_kv") == 0) {
            if (type == GGUF_TYPE_UINT32) {
                ctx->config.n_kv_heads = read_u32(ptr);
                ptr += 4;
                continue;
            }
        }
        /* Tokenizer keys */
        else if (strncmp_len(key, key_len, "tokenizer.ggml.tokens") == 0) {
            if (type == GGUF_TYPE_ARRAY) {
                GGUFType arr_type = (GGUFType)read_u32(ptr);
                ptr += 4;
                uint64_t count = read_u64(ptr);
                ptr += 8;
                ctx->vocab_offset = (uint64_t)(ptr - ctx->data);
                ctx->vocab_count = count;
                ctx->config.vocab_size = (uint32_t)count;
                /* Skip array elements */
                for (uint64_t j = 0; j < count; j++) {
                    ptr = skip_gguf_value(ptr, arr_type);
                }
                continue;
            }
        }
        else if (strncmp_len(key, key_len, "tokenizer.ggml.bos_token_id") == 0) {
            if (type == GGUF_TYPE_UINT32) {
                ctx->config.bos_token_id = read_u32(ptr);
                ptr += 4;
                continue;
            }
        }
        else if (strncmp_len(key, key_len, "tokenizer.ggml.eos_token_id") == 0) {
            if (type == GGUF_TYPE_UINT32) {
                ctx->config.eos_token_id = read_u32(ptr);
                ptr += 4;
                continue;
            }
        }

        /* Skip unknown values */
        ptr = skip_gguf_value(ptr, type);
    }

    /* Store where tensor infos start */
    ctx->tensor_start = ptr;

    /* Defaults */
    if (ctx->config.n_kv_heads == 0) {
        ctx->config.n_kv_heads = ctx->config.n_heads;
    }
    if (ctx->config.dim > 0 && ctx->config.n_heads > 0) {
        ctx->config.head_size = ctx->config.dim / ctx->config.n_heads;
    }

    printf("GGUF Config:\n");
    printf("  arch=%d dim=%d hidden=%d\n",
           ctx->config.arch, ctx->config.dim, ctx->config.hidden_dim);
    printf("  layers=%d heads=%d kv_heads=%d\n",
           ctx->config.n_layers, ctx->config.n_heads, ctx->config.n_kv_heads);
    printf("  vocab=%d seq_len=%d\n",
           ctx->config.vocab_size, ctx->config.seq_len);

    /* Now parse tensor infos to find data section start */
    ptr = ctx->tensor_start;
    for (uint64_t i = 0; i < ctx->header.n_tensors; i++) {
        /* Skip name */
        const char* name;
        uint64_t name_len;
        ptr = read_gguf_string(ptr, &name, &name_len);

        /* n_dims (uint32) */
        uint32_t n_dims = read_u32(ptr);
        ptr += 4;

        /* dims (n_dims x uint64) */
        ptr += n_dims * 8;

        /* type (uint32) */
        ptr += 4;

        /* offset (uint64) */
        ptr += 8;
    }

    /* Align to GGUF_DEFAULT_ALIGNMENT (32 bytes in v3) */
    size_t alignment = (ctx->header.version >= 3) ? 32 : 4;
    size_t offset = ptr - ctx->data;
    offset = (offset + alignment - 1) & ~(alignment - 1);
    ctx->data_start = ctx->data + offset;

    printf("  data_start=0x%08X\n", (uint32_t)(ctx->data_start - ctx->data));

    return 0;
}

/* Find a tensor by name */
int gguf_find_tensor(GGUFContext* ctx, const char* name, GGUFTensorInfo* info) {
    if (!ctx || !ctx->tensor_start || !name || !info) return -1;

    const uint8_t* ptr = ctx->tensor_start;
    size_t name_len = strlen(name);

    for (uint64_t i = 0; i < ctx->header.n_tensors; i++) {
        const char* tname;
        uint64_t tname_len;
        ptr = read_gguf_string(ptr, &tname, &tname_len);

        uint32_t n_dims = read_u32(ptr);
        ptr += 4;

        uint64_t dims[4] = {1, 1, 1, 1};
        for (uint32_t d = 0; d < n_dims && d < 4; d++) {
            dims[d] = read_u64(ptr);
            ptr += 8;
        }

        GGMLType type = (GGMLType)read_u32(ptr);
        ptr += 4;

        uint64_t offset = read_u64(ptr);
        ptr += 8;

        /* Check if this is the tensor we're looking for */
        if (tname_len == name_len && memcmp(tname, name, name_len) == 0) {
            info->name = tname;
            info->n_dims = n_dims;
            for (int d = 0; d < 4; d++) info->dims[d] = dims[d];
            info->type = type;
            info->offset = offset;

            /* Calculate size */
            uint64_t n_elements = dims[0] * dims[1] * dims[2] * dims[3];
            switch (type) {
                case GGML_TYPE_F32:
                    info->size_bytes = n_elements * 4;
                    break;
                case GGML_TYPE_F16:
                    info->size_bytes = n_elements * 2;
                    break;
                case GGML_TYPE_Q8_0:
                    /* Q8_0: blocks of 32 elements, each block is 34 bytes (32 q8 + 2 scale) */
                    info->size_bytes = (n_elements / 32) * 34;
                    break;
                default:
                    info->size_bytes = n_elements * 2;  /* Assume FP16 */
            }
            return 0;
        }
    }

    return -1;  /* Not found */
}

/* Get pointer to tensor data */
const void* gguf_get_tensor_data(GGUFContext* ctx, const GGUFTensorInfo* info) {
    if (!ctx || !ctx->data_start || !info) return NULL;
    return ctx->data_start + info->offset;
}

/* Get vocabulary string for a token ID */
int gguf_get_vocab_string(GGUFContext* ctx, uint32_t token_id, char* out, size_t max_len) {
    if (!ctx || token_id >= ctx->vocab_count || !out || max_len == 0) {
        return -1;
    }

    /* Navigate to vocab array */
    const uint8_t* ptr = ctx->data + ctx->vocab_offset;

    /* Skip tokens until we reach the one we want */
    for (uint32_t i = 0; i < token_id; i++) {
        uint64_t len = read_u64(ptr);
        ptr += 8 + len;
    }

    /* Read the target token */
    uint64_t len = read_u64(ptr);
    ptr += 8;

    if (len >= max_len) len = max_len - 1;

    /* Copy string (token strings in GGUF are NOT null-terminated)
     * Use aligned reads for SDRAM access */
    for (uint64_t i = 0; i < len; i++) {
        /* Read byte using aligned word access */
        uintptr_t addr = (uintptr_t)(ptr + i);
        uintptr_t aligned = addr & ~3;
        int offset = addr & 3;
        uint32_t word = *(const volatile uint32_t*)aligned;
        out[i] = (char)((word >> (offset * 8)) & 0xFF);
    }
    out[len] = '\0';

    return (int)len;
}
