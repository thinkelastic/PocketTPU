/*
 * Llama-2 Inference for VexRiscv on Analogue Pocket
 *
 * This is an embedded adaptation of llama2.c (karpathy/llama2.c)
 * Modified to work with:
 *   - Data slot loading instead of file I/O
 *   - SDRAM for model weights and heap
 *   - Terminal output instead of stdout
 *
 * Original: https://github.com/karpathy/llama2.c
 */

#include "libc/libc.h"
#include "dataslot.h"
#include "terminal.h"
#include "tokenizer_data.h"  /* Embedded tokenizer workaround */
#include "gguf.h"            /* GGUF format parser */

/* Redirect printf to terminal */
#define printf term_printf

/* SDRAM arena for large allocations (RunState) - simple bump allocator */
#define SDRAM_ARENA_ADDR      0x12100000                  /* After tokenizer data */
#define SDRAM_ARENA_END       0x14000000                  /* End of 64MB SDRAM */
static uint8_t* sdram_arena_ptr = (uint8_t*)SDRAM_ARENA_ADDR;

/* Simple bump allocator for SDRAM - no free, just allocate sequentially */
static void* sdram_alloc(size_t size) {
    /* Align to 8 bytes */
    size = (size + 7) & ~7;
    if (sdram_arena_ptr + size > (uint8_t*)SDRAM_ARENA_END) {
        return NULL;
    }
    void* ptr = sdram_arena_ptr;
    sdram_arena_ptr += size;
    return ptr;
}

/* Fast PSRAM arena for KV cache - PSRAM is ~3-5x faster than SDRAM for random access.
 * PSRAM is 16MB total (CRAM0 + CRAM1).
 * Reserve upper 8MB of PSRAM for KV cache, leaving lower 8MB for heap. */
#define PSRAM_CACHE_ADDR      0x30800000                  /* Upper 8MB of 16MB PSRAM */
#define PSRAM_CACHE_END       0x31000000                  /* End of 16MB PSRAM */
static uint8_t* psram_cache_ptr = (uint8_t*)PSRAM_CACHE_ADDR;

/* Bump allocator for PSRAM KV cache region */
static void* psram_cache_alloc(size_t size) {
    /* Align to 8 bytes */
    size = (size + 7) & ~7;
    if (psram_cache_ptr + size > (uint8_t*)PSRAM_CACHE_END) {
        return NULL;  /* Fall back to SDRAM if PSRAM cache region full */
    }
    void* ptr = psram_cache_ptr;
    psram_cache_ptr += size;
    return ptr;
}

/* Configuration - adjust these for your model */
#define DEFAULT_STEPS       64     /* Max tokens to generate */
#define DEFAULT_TEMPERATURE 0.3f    /* 0 = greedy sampling (deterministic) */
#define DEFAULT_TOPP        1.0f    /* 1 = disabled */
#define DEFAULT_PROMPT      "Once upon a time"

/* ============================================
 * Transformer model structures
 * ============================================ */

/* Architecture types */
#define ARCH_LLAMA 1
#define ARCH_GPT2  2

/* Maximum layers supported for per-layer direct GGUF access */
#define MAX_LAYERS 32

typedef struct {
    int dim;         /* Transformer dimension */
    int hidden_dim;  /* FFN hidden dimension */
    int n_layers;    /* Number of layers */
    int n_heads;     /* Number of attention heads */
    int n_kv_heads;  /* Number of KV heads (can be < n_heads for MQA) */
    int vocab_size;  /* Vocabulary size */
    int seq_len;     /* Max sequence length */
    int arch;        /* Architecture type: ARCH_LLAMA or ARCH_GPT2 */
} Config;

typedef struct {
    /* Common weights */
    float* token_embedding_table;
    float* wcls;                    /* Output projection */

    /* LLaMA weights (RMSNorm, SwiGLU) */
    float* rms_att_weight;
    float* rms_ffn_weight;
    float* rms_final_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* w1;                      /* FFN gate (SwiGLU) */
    float* w2;                      /* FFN down */
    float* w3;                      /* FFN up (SwiGLU) */

    /* GPT-2 specific weights (LayerNorm with bias, GELU FFN) */
    float* position_embedding;      /* Learned position embeddings [seq_len, dim] */
    float* ln_att_weight;           /* LayerNorm attention weight */
    float* ln_att_bias;             /* LayerNorm attention bias */
    float* ln_ffn_weight;           /* LayerNorm FFN weight */
    float* ln_ffn_bias;             /* LayerNorm FFN bias */
    float* ln_final_weight;         /* Final LayerNorm weight */
    float* ln_final_bias;           /* Final LayerNorm bias */
    float* wqkv;                    /* Fused QKV projection [3*dim, dim] (contiguous) */
    float* wqkv_bias;               /* Fused QKV bias [3*dim] */
    float* wo_bias;                 /* Output projection bias [dim] */
    float* ffn_up_weight;           /* FFN up projection (GELU) (contiguous) */
    float* ffn_up_bias;
    float* ffn_down_weight;         /* FFN down projection (contiguous) */
    float* ffn_down_bias;

    /* Per-layer direct GGUF pointers (NULL if using contiguous arrays above) */
    float* wqkv_layer[MAX_LAYERS];       /* Per-layer QKV weights */
    float* wo_layer[MAX_LAYERS];         /* Per-layer output projection */
    float* ffn_up_layer[MAX_LAYERS];     /* Per-layer FFN up */
    float* ffn_down_layer[MAX_LAYERS];   /* Per-layer FFN down */
} TransformerWeights;

typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float* key_cache;
    float* value_cache;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    float* data;
    size_t file_size;
} Transformer;

/* ============================================
 * Tokenizer structures
 * ============================================ */

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

/* Global pointer for str_lookup to access vocab without qsort */
Tokenizer* g_tokenizer = NULL;

/* ============================================
 * Sampler structures
 * ============================================ */

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

/* ============================================
 * Memory allocation for run state
 * ============================================ */

static void malloc_run_state(RunState* s, Config* p) {
    /* KV cache dimension: LLaMA uses kv_dim (MQA/GQA), GPT-2 uses full dim */
    int kv_dim;
    if (p->arch == ARCH_GPT2) {
        kv_dim = p->dim;  /* GPT-2: full dimension for KV */
    } else {
        kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;  /* LLaMA: MQA/GQA */
    }
    int kv_cache_size = p->n_layers * p->seq_len * kv_dim * sizeof(float);

    /* Activations go in SDRAM (sequential access pattern, less critical) */
    s->x = sdram_alloc(p->dim * sizeof(float));
    s->xb = sdram_alloc(p->dim * sizeof(float));
    /* xb2 needs to be larger for GPT-2 fused QKV (3*dim output) */
    int xb2_size = (p->arch == ARCH_GPT2) ? 3 * p->dim : p->dim;
    s->xb2 = sdram_alloc(xb2_size * sizeof(float));
    s->hb = sdram_alloc(p->hidden_dim * sizeof(float));
    s->hb2 = sdram_alloc(p->hidden_dim * sizeof(float));
    s->q = sdram_alloc(p->dim * sizeof(float));

    /* KV cache - use PSRAM for faster random access */
    s->key_cache = psram_cache_alloc(kv_cache_size);
    s->value_cache = psram_cache_alloc(kv_cache_size);
    if (!s->key_cache || !s->value_cache) {
        printf("PSRAM cache full, using SDRAM for KV cache\n");
        s->key_cache = sdram_alloc(kv_cache_size);
        s->value_cache = sdram_alloc(kv_cache_size);
    } else {
        printf("KV cache in PSRAM (%d KB x2)\n", kv_cache_size / 1024);
    }

    s->att = sdram_alloc(p->n_heads * p->seq_len * sizeof(float));
    s->logits = sdram_alloc(p->vocab_size * sizeof(float));

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        printf("ERROR: memory allocation failed!\n");
        while(1);
    }
}

static void free_run_state(RunState* s) {
    (void)s;  /* SDRAM bump allocator doesn't free */
}

/* ============================================
 * Weight memory mapping
 * ============================================ */

static void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;

    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; /* skip freq_cis_real */
    ptr += p->seq_len * head_size / 2; /* skip freq_cis_imag */
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

/* ============================================
 * Build transformer from SDRAM data
 * ============================================ */

static void build_transformer_from_memory(Transformer *t, void* data, size_t size) {
    Config* config = (Config*)data;
    t->config = *config;
    t->config.arch = ARCH_LLAMA;  /* model.bin format is always LLaMA */

    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    t->config.vocab_size = abs(config->vocab_size);

    float* weights_ptr = (float*)((char*)data + sizeof(Config));
    memory_map_weights(&t->weights, &t->config, weights_ptr, shared_weights);

    /* For shared weights with DMA acceleration, we need a COPY of token_embedding_table
     * for wcls because:
     * - token_embedding_table must stay as float for token lookups
     * - wcls must be converted to Q16.16 for DMA matmul
     */
    if (shared_weights && t->weights.wcls == t->weights.token_embedding_table) {
        size_t wcls_size = (size_t)t->config.vocab_size * t->config.dim * sizeof(float);
        float* wcls_copy = sdram_alloc(wcls_size);
        if (wcls_copy) {
            memcpy(wcls_copy, t->weights.token_embedding_table, wcls_size);
            t->weights.wcls = wcls_copy;
            printf("  Copied embeddings for wcls (shared weights)\n");
        }
    }

    malloc_run_state(&t->state, &t->config);

    t->data = data;
    t->file_size = size;
}

static void free_transformer(Transformer* t) {
    free_run_state(&t->state);
    /* Don't free t->data - it's in SDRAM and managed elsewhere */
}

/* ============================================
 * Neural network operations
 * ============================================ */

/* Fast inverse square root using Quake/Newton-Raphson method.
 * ~1% error, much faster than 1/sqrtf on soft CPU without FPU. */
static inline float fast_rsqrtf(float x) {
    union { float f; int32_t i; } u = {x};
    u.i = 0x5f3759df - (u.i >> 1);  /* Initial approximation */
    u.f *= 1.5f - (0.5f * x * u.f * u.f);  /* One Newton iteration */
    return u.f;
}

static void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    /* Use multiply by reciprocal and fast inverse sqrt */
    ss = ss * (1.0f / size) + 1e-5f;
    ss = fast_rsqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

/* LayerNorm with bias (for GPT-2) */
static void layernorm(float* o, float* x, float* weight, float* bias, int size) {
    float inv_size = 1.0f / size;

    /* Calculate mean */
    float mean = 0.0f;
    for (int j = 0; j < size; j++) {
        mean += x[j];
    }
    mean *= inv_size;

    /* Calculate variance */
    float var = 0.0f;
    for (int j = 0; j < size; j++) {
        float diff = x[j] - mean;
        var += diff * diff;
    }

    /* Normalize and scale using fast inverse sqrt */
    float inv_std = fast_rsqrtf(var * inv_size + 1e-5f);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * ((x[j] - mean) * inv_std) + bias[j];
    }
}

/* Fast exp approximation using IEEE 754 bit manipulation.
 * ~4% max error but much faster than software expf on soft CPU without FPU.
 * Valid for x in [-87, 88] range (covers typical neural network values). */
static inline float fast_expf(float x) {
    /* Clamp to valid range to avoid overflow/underflow */
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) return 3.4e38f;  /* Large but not inf */
    union { float f; int32_t i; } u;
    u.i = (int32_t)(12102203.0f * x + 1064872507.0f);
    return u.f;
}

/* GELU activation (for GPT-2) - approximation using tanh */
static float gelu(float x) {
    /* GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    float x3 = x * x * x;
    float tanh_arg = 0.7978845608f * (x + 0.044715f * x3);
    /* tanh approximation using fast_exp */
    float exp2x = fast_expf(2.0f * tanh_arg);
    float tanh_val = (exp2x - 1.0f) / (exp2x + 1.0f);
    return 0.5f * x * (1.0f + tanh_val);
}

static void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = fast_expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

/* Option: USE_DMA_ACCEL to use DMA hardware accelerator with pre-converted weights */
#define USE_DMA_ACCEL 1

#if USE_DMA_ACCEL
#include "dma_dot_accel.h"
#include "dot8_accel.h"

/* Flag to track if weights have been converted */
static int weights_converted = 0;

/* Convert model weights from float to Q16.16 in-place.
 * Call once at startup before inference. */
static void convert_weights_to_q16(float* weights, size_t num_floats) {
    int32_t* q16_weights = (int32_t*)weights;
    for (size_t i = 0; i < num_floats; i++) {
        q16_weights[i] = FLOAT_TO_Q16(weights[i]);
    }
}

/* SDRAM buffer for x vector - allocated at startup
 * DMA can only read from SDRAM, so we copy x here before each matmul */
static int32_t* x_sdram_buf = NULL;

/* SDRAM buffer for attention K vectors - copied from PSRAM for DMA access
 * Batch size chosen to balance copy overhead vs DMA efficiency */
#define ATTN_BATCH_SIZE 16
static int32_t* k_sdram_batch = NULL;

/* Minimum head_size to use DMA acceleration
 * For very small vectors, DMA overhead exceeds the benefit */
#define MIN_HEAD_SIZE_FOR_DMA 32

/* Accelerated attention score computation using DMA dot product
 * Copies K vectors from PSRAM to SDRAM in batches, then uses hardware accelerator */
static void accel_attention_scores(float* att, float* q, float* key_cache,
                                   int pos, int head_size, int kv_dim, int kv_head_offset) {
    float scale = fast_rsqrtf((float)head_size);

    /* For small head_size, DMA overhead exceeds benefit - use software */
    if (head_size < MIN_HEAD_SIZE_FOR_DMA) {
        /* Unrolled path for common head_size=8 case */
        if (head_size == 8) {
            /* Precomputed scale for head_size=8: 1/sqrt(8) = 0.35355339f */
            const float scale8 = 0.35355339f;
            float q0=q[0], q1=q[1], q2=q[2], q3=q[3], q4=q[4], q5=q[5], q6=q[6], q7=q[7];
            for (int t = 0; t <= pos; t++) {
                float* k = key_cache + t * kv_dim + kv_head_offset;
                float score = q0*k[0] + q1*k[1] + q2*k[2] + q3*k[3] +
                              q4*k[4] + q5*k[5] + q6*k[6] + q7*k[7];
                att[t] = score * scale8;
            }
        } else {
            for (int t = 0; t <= pos; t++) {
                float* k = key_cache + t * kv_dim + kv_head_offset;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                att[t] = score * scale;
            }
        }
        return;
    }

    /* Convert Q to Q16.16 and preload into accelerator's B-cache */
    for (int i = 0; i < head_size; i++) {
        x_sdram_buf[i] = FLOAT_TO_Q16(q[i]);
    }
    dma_dot_preload_b_vector(x_sdram_buf, head_size);

    /* Process K vectors in batches */
    for (int t = 0; t <= pos; t += ATTN_BATCH_SIZE) {
        int batch_end = (t + ATTN_BATCH_SIZE <= pos + 1) ? t + ATTN_BATCH_SIZE : pos + 1;
        int batch_size = batch_end - t;

        /* Copy batch of K vectors from PSRAM to SDRAM buffer */
        for (int b = 0; b < batch_size; b++) {
            float* k = key_cache + (t + b) * kv_dim + kv_head_offset;
            int32_t* k_dst = k_sdram_batch + b * head_size;
            for (int i = 0; i < head_size; i++) {
                k_dst[i] = FLOAT_TO_Q16(k[i]);
            }
        }

        /* Compute dot products using DMA with cached B (Q vector) */
        for (int b = 0; b < batch_size; b++) {
            int32_t* k_q16 = k_sdram_batch + b * head_size;
            int64_t result = dma_dot_product_q16_cached(k_q16, head_size);
            att[t + b] = Q32_TO_FLOAT(result) * scale;
        }
    }
}

/* Cached input vector length for batch matmul */
static int cached_x_len = 0;

/*
 * Prepare input vector for batch matmul.
 * Call once before multiple matmuls with same x.
 */
static void matmul_batch_begin(float* x, int n) {
    /* Convert and preload x vector */
    for (int j = 0; j < n; j++) {
        x_sdram_buf[j] = FLOAT_TO_Q16(x[j]);
    }
    if (n <= DMA_DOT_MAX_LEN) {
        dma_dot_preload_b_vector(x_sdram_buf, n);
    }
    cached_x_len = n;
}

/*
 * Pipelined matmul using double-buffering to overlap DMA with compute.
 * For d rows, while computing row i, prefetch row i+1.
 * Requires n <= DMA_DOT_MAX_LEN and d >= 2.
 */
static void matmul_batch_pipelined(float* xout, float* w, int n, int d) {
    int32_t* w_q16 = (int32_t*)w;

    if (d < 2 || n > DMA_DOT_MAX_LEN) {
        /* Fall back to simple version */
        for (int i = 0; i < d; i++) {
            int32_t* wi = w_q16 + i * n;
            int64_t result = dma_dot_product_q16_cached(wi, n);
            xout[i] = Q32_TO_FLOAT(result);
        }
        return;
    }

    /* Start first row, prefetch second row */
    dma_dot_pipeline_first(w_q16, w_q16 + n, n);

    /* Process rows 2 to d-1, getting results 0 to d-3 */
    for (int i = 0; i < d - 2; i++) {
        int64_t result = dma_dot_pipeline_next(w_q16 + (i + 2) * n, n);
        xout[i] = Q32_TO_FLOAT(result);
    }

    /* Get result[d-2], start final compute (uses prefetched row d-1) */
    /* Use w_q16 as dummy prefetch address - it won't be used */
    int64_t result_dm2 = dma_dot_pipeline_next(w_q16, n);
    xout[d - 2] = Q32_TO_FLOAT(result_dm2);

    /* Get final result */
    int64_t result_final = dma_dot_pipeline_last();
    xout[d - 1] = Q32_TO_FLOAT(result_final);
}

/*
 * Matmul using previously prepared x vector (from matmul_batch_begin).
 * Skips Q16 conversion of x.
 */
static void matmul_batch(float* xout, float* w, int n, int d) {
    int32_t* w_q16 = (int32_t*)w;

    if (n <= DMA_DOT_MAX_LEN) {
        /* Try pipelined version for better throughput */
        matmul_batch_pipelined(xout, w, n, d);
    } else {
        /* Multiple batches needed */
        for (int i = 0; i < d; i++) {
            int32_t* wi = w_q16 + i * n;
            int64_t result = dma_dot_product_q16(wi, x_sdram_buf, n);
            xout[i] = Q32_TO_FLOAT(result);
        }
    }
}

static void matmul(float* xout, float* x, float* w, int n, int d) {
    /* W (d,n) @ x (n,) -> xout (d,)
     * Weights are pre-converted to Q16.16 and stored in SDRAM.
     * x is converted to Q16.16 and copied to SDRAM buffer for DMA.
     *
     * B-caching optimization: Preload x vector once, reuse for all rows.
     */
    matmul_batch_begin(x, n);
    matmul_batch(xout, w, n, d);
}

#else
static void matmul(float* xout, float* x, float* w, int n, int d) {
    /* W (d,n) @ x (n,) -> xout (d,)
     * Software implementation - simple and correct.
     */
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        float* wi = w + i * n;
        for (int j = 0; j < n; j++) {
            val += wi[j] * x[j];
        }
        xout[i] = val;
    }
}
#endif

/* Precomputed RoPE frequencies and sin/cos caches */
static float* rope_freq_cache = NULL;
static float* rope_sin_cache = NULL;
static float* rope_cos_cache = NULL;
static int rope_freq_head_size = 0;
static int rope_last_pos = -1;

static void ensure_rope_freq(int head_size) {
    if (rope_freq_cache && rope_freq_head_size == head_size) return;
    if (rope_freq_cache) free(rope_freq_cache);
    if (rope_sin_cache) free(rope_sin_cache);
    if (rope_cos_cache) free(rope_cos_cache);
    rope_freq_cache = (float*)malloc((head_size/2) * sizeof(float));
    rope_sin_cache = (float*)malloc((head_size/2) * sizeof(float));
    rope_cos_cache = (float*)malloc((head_size/2) * sizeof(float));
    rope_freq_head_size = head_size;
    rope_last_pos = -1;  /* Force recompute of sin/cos */
    for (int i = 0; i < head_size; i += 2) {
        int head_dim = i;  /* Within one head, i is the head_dim */
        rope_freq_cache[i/2] = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    }
}

/* Precompute sin/cos for current position (once per forward pass) */
static void precompute_rope_sincos(int pos, int head_size) {
    if (pos == rope_last_pos) return;
    rope_last_pos = pos;
    for (int i = 0; i < head_size / 2; i++) {
        float val = pos * rope_freq_cache[i];
        rope_sin_cache[i] = sinf(val);
        rope_cos_cache[i] = cosf(val);
    }
}

/* LLaMA forward pass (RMSNorm, RoPE, SwiGLU) */
static float* forward_llama(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    /* Ensure RoPE frequencies are precomputed */
    ensure_rope_freq(head_size);
    /* Precompute sin/cos for current position (once per forward pass) */
    precompute_rope_sincos(pos, head_size);

    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(*x));

    for (unsigned long long l = 0; l < (unsigned long long)p->n_layers; l++) {
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);


        int loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        /* Batch Q/K/V projections - same input, avoid redundant Q16 conversion */
        matmul_batch_begin(s->xb, dim);
        matmul_batch(s->q, w->wq + l*dim*dim, dim, dim);
        matmul_batch(s->k, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul_batch(s->v, w->wv + l*dim*kv_dim, dim, kv_dim);

        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            int freq_idx = head_dim / 2;
            float fcr = rope_cos_cache[freq_idx];  /* Use precomputed cos */
            float fci = rope_sin_cache[freq_idx];  /* Use precomputed sin */
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            int kv_head_offset = (h / kv_mul) * head_size;

#if USE_DMA_ACCEL
            /* Use hardware accelerator for Q·K dot products */
            accel_attention_scores(att, q, s->key_cache + loff, pos,
                                   head_size, kv_dim, kv_head_offset);
#else
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + kv_head_offset;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }
#endif

            softmax(att, pos + 1);

            float* xb = s->xb + h * head_size;
            int kv_head_off = (h / kv_mul) * head_size;

            /* Unrolled attention value accumulation for head_size=8 */
            if (head_size == 8) {
                float xb0=0, xb1=0, xb2=0, xb3=0, xb4=0, xb5=0, xb6=0, xb7=0;
                for (int t = 0; t <= pos; t++) {
                    float* v = s->value_cache + loff + t * kv_dim + kv_head_off;
                    float a = att[t];
                    xb0 += a * v[0]; xb1 += a * v[1]; xb2 += a * v[2]; xb3 += a * v[3];
                    xb4 += a * v[4]; xb5 += a * v[5]; xb6 += a * v[6]; xb7 += a * v[7];
                }
                xb[0]=xb0; xb[1]=xb1; xb[2]=xb2; xb[3]=xb3;
                xb[4]=xb4; xb[5]=xb5; xb[6]=xb6; xb[7]=xb7;
            } else {
                memset(xb, 0, head_size * sizeof(float));
                for (int t = 0; t <= pos; t++) {
                    float* v = s->value_cache + loff + t * kv_dim + kv_head_off;
                    float a = att[t];
                    for (int i = 0; i < head_size; i++) {
                        xb[i] += a * v[i];
                    }
                }
            }
        }

        /* Output projection */
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        /* Batch FFN up projections - same input, avoid redundant Q16 conversion */
        matmul_batch_begin(s->xb, dim);
        matmul_batch(s->hb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul_batch(s->hb2, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        /* SwiGLU activation: silu(x) * gate, where silu(x) = x * sigmoid(x) */
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + fast_expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

/* GPT-2 forward pass (LayerNorm, learned pos emb, GELU) */
static float* forward_gpt2(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int n_heads = p->n_heads;

    /* Token embedding + position embedding */
    float* tok_emb = w->token_embedding_table + token * dim;
    float* pos_emb = w->position_embedding + pos * dim;
    for (int i = 0; i < dim; i++) {
        x[i] = tok_emb[i] + pos_emb[i];
    }

    for (int l = 0; l < p->n_layers; l++) {
        /* Pre-attention LayerNorm */
        layernorm(s->xb, x, w->ln_att_weight + l*dim, w->ln_att_bias + l*dim, dim);

        /* Fused QKV projection: [dim] -> [3*dim] */
        /* Output: s->q[0..dim-1], s->k[0..dim-1], s->v[0..dim-1] stored in s->xb2 */
        /* Use per-layer pointer if available, otherwise contiguous array */
        float* wqkv_l = w->wqkv_layer[l] ? w->wqkv_layer[l] : (w->wqkv + l*dim*3*dim);
        matmul(s->xb2, s->xb, wqkv_l, dim, 3*dim);
        /* Add bias */
        for (int i = 0; i < 3*dim; i++) {
            s->xb2[i] += w->wqkv_bias[l*3*dim + i];
        }

        /* Split Q, K, V and store in cache */
        int loff = l * p->seq_len * dim;
        float* q_out = s->q;
        float* k_cache_pos = s->key_cache + loff + pos * dim;
        float* v_cache_pos = s->value_cache + loff + pos * dim;
        for (int i = 0; i < dim; i++) {
            q_out[i] = s->xb2[i];
            k_cache_pos[i] = s->xb2[dim + i];
            v_cache_pos[i] = s->xb2[2*dim + i];
        }

        /* Multi-head attention (no RoPE - GPT-2 uses learned positions) */
        for (int h = 0; h < n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;

            /* Compute attention scores: Q @ K^T / sqrt(head_size) - use DMA accelerator */
            accel_attention_scores(att, q, s->key_cache + loff, pos, head_size, dim, h * head_size);

            softmax(att, pos + 1);

            /* Weighted sum of values */
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * dim + h * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        /* Output projection */
        float* wo_l = w->wo_layer[l] ? w->wo_layer[l] : (w->wo + l*dim*dim);
        matmul(s->xb2, s->xb, wo_l, dim, dim);
        /* Add bias */
        for (int i = 0; i < dim; i++) {
            s->xb2[i] += w->wo_bias[l*dim + i];
        }

        /* Residual connection */
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        /* Pre-FFN LayerNorm */
        layernorm(s->xb, x, w->ln_ffn_weight + l*dim, w->ln_ffn_bias + l*dim, dim);

        /* FFN: up projection -> GELU -> down projection */
        float* ffn_up_l = w->ffn_up_layer[l] ? w->ffn_up_layer[l] : (w->ffn_up_weight + l*dim*hidden_dim);
        matmul(s->hb, s->xb, ffn_up_l, dim, hidden_dim);
        /* Add bias and apply GELU */
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = gelu(s->hb[i] + w->ffn_up_bias[l*hidden_dim + i]);
        }

        /* Down projection */
        float* ffn_down_l = w->ffn_down_layer[l] ? w->ffn_down_layer[l] : (w->ffn_down_weight + l*hidden_dim*dim);
        matmul(s->xb, s->hb, ffn_down_l, hidden_dim, dim);
        /* Add bias */
        for (int i = 0; i < dim; i++) {
            s->xb[i] += w->ffn_down_bias[l*dim + i];
        }

        /* Residual connection */
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    /* Final LayerNorm */
    layernorm(x, x, w->ln_final_weight, w->ln_final_bias, dim);

    /* Output projection (logits) */
    matmul(s->logits, x, w->wcls, dim, p->vocab_size);
    return s->logits;
}

/* Dispatch to appropriate forward pass based on architecture */
static float* forward(Transformer* transformer, int token, int pos) {
    if (transformer->config.arch == ARCH_GPT2) {
        return forward_gpt2(transformer, token, pos);
    } else {
        return forward_llama(transformer, token, pos);
    }
}

/* ============================================
 * Tokenizer
 * ============================================ */

/* Read 32-bit value from potentially unaligned address using only word-aligned reads */
static inline uint32_t read_u32(const uint8_t* ptr) {
    uintptr_t addr = (uintptr_t)ptr;
    uintptr_t aligned_addr = addr & ~3;
    int offset = addr & 3;

    /* Read the aligned word(s) */
    uint32_t w0 = *(const volatile uint32_t*)aligned_addr;

    if (offset == 0) {
        return w0;
    }

    /* Need to read next word and combine */
    uint32_t w1 = *(const volatile uint32_t*)(aligned_addr + 4);

    /* Extract bytes from the two words based on offset */
    return (w0 >> (offset * 8)) | (w1 << ((4 - offset) * 8));
}

/* Helper to read potentially unaligned float */
static inline float read_float(const uint8_t* ptr) {
    uint32_t bits = read_u32(ptr);
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

/* Tokenizer buffers:
 * - Pointers and scores allocated from PSRAM heap (word-aligned access)
 * - String pool in BRAM (PSRAM doesn't support byte writes!)
 */
static char** tok_vocab_ptrs = NULL;
static float* tok_scores_ptr = NULL;

/* String pool - allocated from PSRAM heap for 32K tokenizer (~500KB) */
#define TOK_STRING_POOL_SIZE (512 * 1024)  /* 512KB for token strings */
static char* tok_string_pool = NULL;
static char* tok_string_ptr = NULL;

/* BRAM buffer for encode() - SDRAM byte writes don't work! */
#define ENCODE_BUFFER_SIZE 256       /* For BPE string operations */
static char encode_str_buffer[ENCODE_BUFFER_SIZE];

static void build_tokenizer_from_memory(Tokenizer* t, void* data, int vocab_size) {
    t->vocab_size = vocab_size;
    t->sorted_vocab = NULL;

    /* Allocate vocab pointers, scores, and string pool from PSRAM heap */
    tok_vocab_ptrs = (char**)malloc(vocab_size * sizeof(char*));
    tok_scores_ptr = (float*)malloc(vocab_size * sizeof(float));
    tok_string_pool = (char*)malloc(TOK_STRING_POOL_SIZE);

    if (!tok_vocab_ptrs || !tok_scores_ptr || !tok_string_pool) {
        printf("ERROR: Failed to allocate tokenizer buffers\n");
        return;
    }

    /* Reset string pool pointer */
    tok_string_ptr = tok_string_pool;

    t->vocab = tok_vocab_ptrs;
    t->vocab_scores = tok_scores_ptr;

    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    /* Read directly from SDRAM using byte access */
    uint8_t* ptr = (uint8_t*)data;

    printf("  Tok at 0x%08X\n", (uint32_t)data);

    /* Read max_token_length */
    uint32_t raw_max_tok = read_u32(ptr);
    printf("  raw max_token_length=%d (0x%08X)\n", raw_max_tok, raw_max_tok);

    /* WORKAROUND: The first word of tokenizer data gets corrupted by CDC bug.
     * Known corruption: 0x07 -> 0x1B (bits 3,4 set incorrectly)
     * If we detect this pattern, use the correct value. */
    if (raw_max_tok == 0x1B) {
        t->max_token_length = 7;  /* Correct value for tokenizer.bin */
        printf("  WORKAROUND: Corrected to %d\n", t->max_token_length);
    } else {
        t->max_token_length = raw_max_tok;
    }
    printf("  max_token_length=%d\n", t->max_token_length);
    ptr += 4;

    /* Parse tokens - copy from embedded BRAM data to BRAM string pool */
    int error_count = 0;
    for (int i = 0; i < vocab_size; i++) {

        /* Read length to check for corruption */
        int32_t len = (int32_t)read_u32(ptr + 4);


        /* Detect obvious corruption: len should never be huge or negative */
        if (len < 0 || len > 20) {
            if (error_count < 5) {
                printf("\n  ERR tok[%d] @0x%X: suspicious len=%d\n", i, (uint32_t)ptr, len);
            }
            error_count++;
        }

        /* Read score (4 bytes) */
        t->vocab_scores[i] = read_float(ptr);
        ptr += 4;

        /* Skip length (already read above) */
        ptr += 4;

        if (len < 0 || len > 100) {
            printf("\nERROR: bad len %d at tok %d\n", len, i);
            return;
        }

        /* Allocate string from BRAM pool */
        if (tok_string_ptr + len + 1 > tok_string_pool + TOK_STRING_POOL_SIZE) {
            printf("\nERROR: String pool exhausted at tok %d\n", i);
            return;
        }
        char* str = tok_string_ptr;
        /* Align string pool pointer to next word boundary for proper writes */
        tok_string_ptr += (len + 1 + 3) & ~3;

        /* Copy string using word-aligned writes to PSRAM */
        /* First, read all bytes into a temp buffer, then write as words */
        uint32_t word_buf = 0;
        int word_idx = 0;
        volatile uint32_t* dst_word = (volatile uint32_t*)str;

        for (int j = 0; j < len; j++) {
            /* Read byte from SDRAM using word-aligned read */
            uintptr_t byte_addr = (uintptr_t)(ptr + j);
            uintptr_t word_addr = byte_addr & ~3;
            int byte_offset = byte_addr & 3;
            uint32_t src_word = *(const volatile uint32_t*)word_addr;
            uint8_t byte_val = (src_word >> (byte_offset * 8)) & 0xFF;

            /* Pack into word buffer */
            word_buf |= ((uint32_t)byte_val) << (word_idx * 8);
            word_idx++;

            if (word_idx == 4) {
                *dst_word++ = word_buf;
                word_buf = 0;
                word_idx = 0;
            }
        }
        /* Add null terminator and write final word */
        if (word_idx < 4) {
            /* Null terminator goes in current word */
            *dst_word = word_buf;  /* Remaining bytes are already 0 */
        } else {
            /* Need new word for null terminator */
            *dst_word = 0;
        }
        t->vocab[i] = str;

        ptr += len;
    }
    printf("Tokenizer: %d tokens loaded\n", vocab_size);
}

static void free_tokenizer(Tokenizer* t) {
    (void)t;  /* SDRAM bump allocator doesn't free */
}

/* Parse <0xXX> format manually (sscanf doesn't support %hhX) */
static int parse_byte_token(const char *s, unsigned char *out) {
    /* Check length first: need exactly "<0xXX>" = 6 chars */
    if (s[0] != '<') return 0;
    if (s[1] != '0') return 0;
    if (s[2] != 'x') return 0;
    /* s[3] and s[4] are hex digits, s[5] must be '>' */
    if (s[3] == '\0' || s[4] == '\0' || s[5] != '>') return 0;

    int hi = 0, lo = 0;
    char c = s[3];
    if (c >= '0' && c <= '9') hi = c - '0';
    else if (c >= 'A' && c <= 'F') hi = c - 'A' + 10;
    else if (c >= 'a' && c <= 'f') hi = c - 'a' + 10;
    else return 0;

    c = s[4];
    if (c >= '0' && c <= '9') lo = c - '0';
    else if (c >= 'A' && c <= 'F') lo = c - 'A' + 10;
    else if (c >= 'a' && c <= 'f') lo = c - 'a' + 10;
    else return 0;

    *out = (unsigned char)((hi << 4) | lo);
    return 1;
}

static char* decode(Tokenizer* t, int prev_token, int token) {
    static char decoded[256];
    /* Bounds check */
    if (token < 0 || token >= t->vocab_size) {
        return "(BAD)";
    }
    char *piece = t->vocab[token];
    if (piece == NULL) {
        return "(NULL)";
    }
    if (prev_token == 1 && piece[0] == ' ') { piece++; }

    /* Handle <0xXX> format byte tokens */
    unsigned char byte_val;
    if (parse_byte_token(piece, &byte_val)) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }

    /* GPT-2 BPE uses special byte encoding:
     * - Ġ (U+0120, UTF-8: 0xC4 0xA0) represents space
     * - Ċ (U+010A, UTF-8: 0xC4 0x8A) represents newline */
    int j = 0;
    for (int i = 0; piece[i] && j < (int)sizeof(decoded) - 1; i++) {
        unsigned char c = (unsigned char)piece[i];
        if (c == 0xC4 && (unsigned char)piece[i+1] == 0xA0) {
            decoded[j++] = ' ';  /* Ġ -> space */
            i++;
        } else if (c == 0xC4 && (unsigned char)piece[i+1] == 0x8A) {
            decoded[j++] = '\n'; /* Ċ -> newline */
            i++;
        } else {
            decoded[j++] = piece[i];
        }
    }
    decoded[j] = '\0';
    return decoded;
}

static void safe_printf(char *piece) {
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;
        }
    }
    printf("%s", piece);
}

/* Linear search - slower but avoids qsort which is too slow with SDRAM strings */
static int str_lookup_linear(char *str, char **vocab, int vocab_size) {
    for (int i = 0; i < vocab_size; i++) {
        if (strcmp(str, vocab[i]) == 0) {
            return i;
        }
    }
    return -1;
}

static int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    (void)sorted_vocab;  /* Not used with linear search */
    /* Use global tokenizer vocab for linear search */
    extern Tokenizer* g_tokenizer;
    return str_lookup_linear(str, g_tokenizer->vocab, vocab_size);
}

static void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) {
        printf("ERROR: cannot encode NULL text\n");
        while(1);
    }

    /* Skip qsort - use linear search instead (qsort too slow with SDRAM strings) */
    (void)t->sorted_vocab;  /* Not used */

    /* Use static BRAM buffer - SDRAM byte writes don't work! */
    char* str_buffer = encode_str_buffer;
    size_t str_len = 0;

    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        if (dummy_prefix != -1) {
            tokens[(*n_tokens)++] = dummy_prefix;
        }
        /* If space not found, skip the dummy prefix - some tokenizers don't have it */
    }
    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }

        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    /* BPE merge */
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break;
        }

        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;

    /* str_buffer is static BRAM, no free needed */
}

/* ============================================
 * Sampler
 * ============================================ */

static int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];

    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

static int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

static int compare_prob(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

static int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare_prob);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}

static void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    /* Only allocate probindex if using top-p sampling */
    if (temperature > 0.0f && topp > 0.0f && topp < 1.0f) {
        sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
        if (!sampler->probindex) {
            printf("WARNING: probindex malloc failed, using greedy sampling\n");
            sampler->temperature = 0.0f;  /* Force greedy sampling */
        }
    } else {
        sampler->probindex = NULL;
    }
}

static void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

static unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

static float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

static int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q = 0; q < sampler->vocab_size; q++) {
            logits[q] /= sampler->temperature;
        }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

/* ============================================
 * Generation loop
 * ============================================ */

static void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    if (!prompt_tokens) {
        printf("ERROR: malloc failed\n");
        while(1);
    }
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    if (num_prompt_tokens < 1) {
        printf("ERROR: expected at least 1 prompt token\n");
        while(1);
    }

    uint64_t start_cycles = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;

    while (pos < steps) {
        float* logits = forward(transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        if (next == 1) { break; }  /* EOS token */

        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        token = next;

        /* Stop on newline */
        if (piece && strchr(piece, '\n')) { break; }

        if (start_cycles == 0) {
            /* Start timing after first token (prompt processing) */
            start_cycles = ((uint64_t)SYS_CYCLE_HI << 32) | SYS_CYCLE_LO;
        }
    }
    printf("\n");

    if (pos > 1 && start_cycles > 0) {
        uint64_t end_cycles = ((uint64_t)SYS_CYCLE_HI << 32) | SYS_CYCLE_LO;
        uint64_t elapsed_cycles = end_cycles - start_cycles;
        /* CPU runs at 50MHz, so cycles / 50000 = milliseconds */
        uint32_t elapsed_ms = (uint32_t)(elapsed_cycles / 50000);
        int tokens_generated = pos - 1;
        if (elapsed_ms > 0) {
            /* tokens per minute = tokens * 60000 / elapsed_ms */
            uint32_t tok_per_min = (uint32_t)((uint64_t)tokens_generated * 60000 / elapsed_ms);
            printf("Speed: %d tok/min (%d ms total)\n", tok_per_min, elapsed_ms);
        }
    }

    free(prompt_tokens);
}

/* ============================================
 * GGUF Model Loading
 * ============================================ */

/* Global GGUF context for tokenizer access */
static GGUFContext g_gguf_ctx;

/* Read float from potentially unaligned SDRAM address using word-aligned reads */
static inline float read_float_aligned(const uint8_t* ptr) {
    uintptr_t addr = (uintptr_t)ptr;
    uintptr_t aligned = addr & ~3;
    int offset = addr & 3;

    uint32_t w0 = *(const volatile uint32_t*)aligned;
    if (offset == 0) {
        float f;
        memcpy(&f, &w0, sizeof(float));
        return f;
    }

    uint32_t w1 = *(const volatile uint32_t*)(aligned + 4);
    uint32_t bits = (w0 >> (offset * 8)) | (w1 << ((4 - offset) * 8));
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

/* Get direct pointer to F32 tensor data in GGUF (no copy needed if aligned)
 * Returns pointer to tensor data if F32, or NULL if conversion needed */
static float* get_tensor_direct(GGUFContext* ctx, const char* name, GGUFTensorInfo* out_info) {
    if (gguf_find_tensor(ctx, name, out_info) != 0) {
        return NULL;  /* Not found */
    }

    /* Only return direct pointer for F32 tensors that are 4-byte aligned */
    if (out_info->type == GGML_TYPE_F32) {
        const uint8_t* ptr = gguf_get_tensor_data(ctx, out_info);
        if (((uintptr_t)ptr & 3) == 0) {
            return (float*)ptr;  /* Aligned F32 - can use directly */
        }
    }

    return NULL;  /* Need conversion/copy */
}

/* Load tensor data directly into destination buffer (no transpose needed) */
static int load_tensor_to_buffer(GGUFContext* ctx, const char* name, float* dest, int count) {
    GGUFTensorInfo info;
    if (gguf_find_tensor(ctx, name, &info) != 0) {
        return -1;  /* Not found */
    }

    const uint8_t* src = gguf_get_tensor_data(ctx, &info);

    if (info.type == GGML_TYPE_F32) {
        /* Use aligned reads for SDRAM access */
        for (int i = 0; i < count; i++) {
            dest[i] = read_float_aligned(src + i * 4);
        }
    } else if (info.type == GGML_TYPE_F16) {
        /* FP16: read 2 bytes at a time with alignment handling */
        for (int i = 0; i < count; i++) {
            uintptr_t addr = (uintptr_t)(src + i * 2);
            uintptr_t aligned = addr & ~3;
            int offset = addr & 3;
            uint32_t word = *(const volatile uint32_t*)aligned;
            uint16_t h;
            if (offset <= 2) {
                h = (word >> (offset * 8)) & 0xFFFF;
            } else {
                uint32_t w1 = *(const volatile uint32_t*)(aligned + 4);
                h = ((word >> 24) | (w1 << 8)) & 0xFFFF;
            }
            dest[i] = fp16_to_float(h);
        }
    } else {
        return -2;  /* Unsupported type */
    }

    return 0;
}

/* Try to get direct pointer, fall back to allocating and copying */
static float* get_or_load_tensor(GGUFContext* ctx, const char* name, int count, const char* desc) {
    GGUFTensorInfo info;
    float* ptr = get_tensor_direct(ctx, name, &info);
    if (ptr) {
        return ptr;
    }

    /* Need to allocate and convert */
    ptr = sdram_alloc(count * sizeof(float));
    if (!ptr) {
        printf("FAILED: %s alloc\n", desc);
        return NULL;
    }
    if (load_tensor_to_buffer(ctx, name, ptr, count) != 0) {
        printf("FAILED: %s load\n", desc);
        return NULL;
    }
    return ptr;
}

/* Build LLaMA transformer from GGUF file */
static int build_llama_from_gguf(Transformer* t, GGUFContext* ctx) {
    int dim = t->config.dim;
    int n_layers = t->config.n_layers;
    int hidden_dim = t->config.hidden_dim;
    int vocab_size = t->config.vocab_size;
    int kv_dim = (dim * t->config.n_kv_heads) / t->config.n_heads;

    /* Token embeddings */
    t->weights.token_embedding_table = sdram_alloc((size_t)vocab_size * dim * sizeof(float));
    if (!t->weights.token_embedding_table) return -1;
    load_tensor_to_buffer(ctx, "token_embd.weight", t->weights.token_embedding_table, vocab_size * dim);
    printf("  Loaded embeddings [%d x %d]\n", vocab_size, dim);

    /* RMSNorm weights - allocate contiguously for all layers */
    t->weights.rms_att_weight = sdram_alloc(n_layers * dim * sizeof(float));
    t->weights.rms_ffn_weight = sdram_alloc(n_layers * dim * sizeof(float));

    /* Attention weights */
    t->weights.wq = sdram_alloc(n_layers * dim * dim * sizeof(float));
    t->weights.wk = sdram_alloc(n_layers * kv_dim * dim * sizeof(float));
    t->weights.wv = sdram_alloc(n_layers * kv_dim * dim * sizeof(float));
    t->weights.wo = sdram_alloc(n_layers * dim * dim * sizeof(float));

    /* FFN weights */
    t->weights.w1 = sdram_alloc(n_layers * hidden_dim * dim * sizeof(float));
    t->weights.w2 = sdram_alloc(n_layers * dim * hidden_dim * sizeof(float));
    t->weights.w3 = sdram_alloc(n_layers * hidden_dim * dim * sizeof(float));

    /* Load each layer's weights */
    char tensor_name[64];
    for (int l = 0; l < n_layers; l++) {
        sprintf(tensor_name, "blk.%d.attn_norm.weight", l);
        load_tensor_to_buffer(ctx, tensor_name, t->weights.rms_att_weight + l * dim, dim);

        sprintf(tensor_name, "blk.%d.attn_q.weight", l);
        load_tensor_to_buffer(ctx, tensor_name, t->weights.wq + l * dim * dim, dim * dim);
        sprintf(tensor_name, "blk.%d.attn_k.weight", l);
        load_tensor_to_buffer(ctx, tensor_name, t->weights.wk + l * kv_dim * dim, kv_dim * dim);
        sprintf(tensor_name, "blk.%d.attn_v.weight", l);
        load_tensor_to_buffer(ctx, tensor_name, t->weights.wv + l * kv_dim * dim, kv_dim * dim);

        sprintf(tensor_name, "blk.%d.attn_output.weight", l);
        load_tensor_to_buffer(ctx, tensor_name, t->weights.wo + l * dim * dim, dim * dim);

        sprintf(tensor_name, "blk.%d.ffn_norm.weight", l);
        load_tensor_to_buffer(ctx, tensor_name, t->weights.rms_ffn_weight + l * dim, dim);

        sprintf(tensor_name, "blk.%d.ffn_gate.weight", l);
        load_tensor_to_buffer(ctx, tensor_name, t->weights.w1 + l * hidden_dim * dim, hidden_dim * dim);
        sprintf(tensor_name, "blk.%d.ffn_down.weight", l);
        load_tensor_to_buffer(ctx, tensor_name, t->weights.w2 + l * dim * hidden_dim, dim * hidden_dim);
        sprintf(tensor_name, "blk.%d.ffn_up.weight", l);
        load_tensor_to_buffer(ctx, tensor_name, t->weights.w3 + l * hidden_dim * dim, hidden_dim * dim);

        if ((l + 1) % 2 == 0 || l == n_layers - 1) {
            printf("  Loaded layer %d/%d\n", l + 1, n_layers);
        }
    }

    /* Final RMSNorm */
    t->weights.rms_final_weight = sdram_alloc(dim * sizeof(float));
    load_tensor_to_buffer(ctx, "output_norm.weight", t->weights.rms_final_weight, dim);

    /* Output projection */
    GGUFTensorInfo wcls_info;
    if (gguf_find_tensor(ctx, "output.weight", &wcls_info) == 0) {
        t->weights.wcls = sdram_alloc((size_t)vocab_size * dim * sizeof(float));
        load_tensor_to_buffer(ctx, "output.weight", t->weights.wcls, vocab_size * dim);
        printf("  Loaded output projection\n");
    } else {
        t->weights.wcls = sdram_alloc((size_t)vocab_size * dim * sizeof(float));
        if (!t->weights.wcls) return -1;
        memcpy(t->weights.wcls, t->weights.token_embedding_table,
               (size_t)vocab_size * dim * sizeof(float));
        printf("  Copied embeddings for output (tied weights)\n");
    }

    return 0;
}

/* Build GPT-2 transformer from GGUF file */
static int build_gpt2_from_gguf(Transformer* t, GGUFContext* ctx) {
    int dim = t->config.dim;
    int n_layers = t->config.n_layers;
    int hidden_dim = t->config.hidden_dim;
    int vocab_size = t->config.vocab_size;
    int seq_len = t->config.seq_len;

    printf("Loading GPT-2: dim=%d layers=%d vocab=%d\n", dim, n_layers, vocab_size);

    /* Token embeddings - try direct pointer first */
    t->weights.token_embedding_table = get_or_load_tensor(ctx, "token_embd.weight",
                                                          vocab_size * dim, "token_embd");
    if (!t->weights.token_embedding_table) return -1;

    /* Position embeddings */
    t->weights.position_embedding = get_or_load_tensor(ctx, "position_embd.weight",
                                                       seq_len * dim, "position_embd");
    if (!t->weights.position_embedding) return -1;

    if (n_layers > MAX_LAYERS) {
        printf("  FAILED: n_layers %d > MAX_LAYERS %d\n", n_layers, MAX_LAYERS);
        return -1;
    }

    /* LayerNorm weights are small, always copy for simplicity */
    t->weights.ln_att_weight = sdram_alloc(n_layers * dim * sizeof(float));
    t->weights.ln_att_bias = sdram_alloc(n_layers * dim * sizeof(float));
    t->weights.ln_ffn_weight = sdram_alloc(n_layers * dim * sizeof(float));
    t->weights.ln_ffn_bias = sdram_alloc(n_layers * dim * sizeof(float));

    if (!t->weights.ln_att_weight || !t->weights.ln_att_bias ||
        !t->weights.ln_ffn_weight || !t->weights.ln_ffn_bias) {
        printf("  FAILED: LayerNorm alloc\n");
        return -1;
    }

    /* Large weight matrices - use per-layer direct GGUF pointers */
    t->weights.wqkv = NULL;  /* Using per-layer access */
    t->weights.wo = NULL;
    t->weights.ffn_up_weight = NULL;
    t->weights.ffn_down_weight = NULL;

    /* Biases are small, allocate contiguous arrays */
    t->weights.wqkv_bias = sdram_alloc(n_layers * 3 * dim * sizeof(float));
    t->weights.wo_bias = sdram_alloc(n_layers * dim * sizeof(float));
    t->weights.ffn_up_bias = sdram_alloc(n_layers * hidden_dim * sizeof(float));
    t->weights.ffn_down_bias = sdram_alloc(n_layers * dim * sizeof(float));

    if (!t->weights.wqkv_bias || !t->weights.wo_bias ||
        !t->weights.ffn_up_bias || !t->weights.ffn_down_bias) {
        printf("  FAILED: bias alloc\n");
        return -1;
    }

    /* Load each layer's weights - use direct pointers for large weights */
    char tensor_name[64];
    for (int l = 0; l < n_layers; l++) {
        /* LayerNorm (attention) - small, always copy */
        sprintf(tensor_name, "blk.%d.attn_norm.weight", l);
        load_tensor_to_buffer(ctx, tensor_name, t->weights.ln_att_weight + l * dim, dim);
        sprintf(tensor_name, "blk.%d.attn_norm.bias", l);
        load_tensor_to_buffer(ctx, tensor_name, t->weights.ln_att_bias + l * dim, dim);

        /* Large weight matrices - get direct pointers */
        sprintf(tensor_name, "blk.%d.attn_qkv.weight", l);
        t->weights.wqkv_layer[l] = get_or_load_tensor(ctx, tensor_name, dim * 3 * dim, tensor_name);
        if (!t->weights.wqkv_layer[l]) return -1;

        sprintf(tensor_name, "blk.%d.attn_qkv.bias", l);
        load_tensor_to_buffer(ctx, tensor_name, t->weights.wqkv_bias + l * 3 * dim, 3 * dim);

        sprintf(tensor_name, "blk.%d.attn_output.weight", l);
        t->weights.wo_layer[l] = get_or_load_tensor(ctx, tensor_name, dim * dim, tensor_name);
        if (!t->weights.wo_layer[l]) return -1;

        sprintf(tensor_name, "blk.%d.attn_output.bias", l);
        load_tensor_to_buffer(ctx, tensor_name, t->weights.wo_bias + l * dim, dim);

        /* LayerNorm (FFN) */
        sprintf(tensor_name, "blk.%d.ffn_norm.weight", l);
        load_tensor_to_buffer(ctx, tensor_name, t->weights.ln_ffn_weight + l * dim, dim);
        sprintf(tensor_name, "blk.%d.ffn_norm.bias", l);
        load_tensor_to_buffer(ctx, tensor_name, t->weights.ln_ffn_bias + l * dim, dim);

        /* FFN weights - direct pointers */
        sprintf(tensor_name, "blk.%d.ffn_up.weight", l);
        t->weights.ffn_up_layer[l] = get_or_load_tensor(ctx, tensor_name, dim * hidden_dim, tensor_name);
        if (!t->weights.ffn_up_layer[l]) return -1;

        sprintf(tensor_name, "blk.%d.ffn_up.bias", l);
        load_tensor_to_buffer(ctx, tensor_name, t->weights.ffn_up_bias + l * hidden_dim, hidden_dim);

        sprintf(tensor_name, "blk.%d.ffn_down.weight", l);
        t->weights.ffn_down_layer[l] = get_or_load_tensor(ctx, tensor_name, hidden_dim * dim, tensor_name);
        if (!t->weights.ffn_down_layer[l]) return -1;

        sprintf(tensor_name, "blk.%d.ffn_down.bias", l);
        load_tensor_to_buffer(ctx, tensor_name, t->weights.ffn_down_bias + l * dim, dim);
    }

    /* Final LayerNorm - small, always copy */
    t->weights.ln_final_weight = sdram_alloc(dim * sizeof(float));
    t->weights.ln_final_bias = sdram_alloc(dim * sizeof(float));
    if (!t->weights.ln_final_weight || !t->weights.ln_final_bias) {
        printf("  FAILED: final LayerNorm alloc\n");
        return -1;
    }
    load_tensor_to_buffer(ctx, "output_norm.weight", t->weights.ln_final_weight, dim);
    load_tensor_to_buffer(ctx, "output_norm.bias", t->weights.ln_final_bias, dim);

    /* Output projection (lm_head) - use tied weights to save memory
     * GPT-2 models typically share token embeddings with output projection */
    t->weights.wcls = t->weights.token_embedding_table;  /* Tied weights */
    printf("  Using tied weights for output projection\n");

    printf("  GPT-2 weights loaded (direct GGUF access for large tensors)\n");
    return 0;
}

/* Build transformer from GGUF file */
static int build_transformer_from_gguf(Transformer* t, GGUFContext* ctx) {
    GGUFConfig* gc = &ctx->config;

    /* Copy config */
    t->config.dim = gc->dim;
    t->config.hidden_dim = gc->hidden_dim;
    t->config.n_layers = gc->n_layers;
    t->config.n_heads = gc->n_heads;
    t->config.n_kv_heads = gc->n_kv_heads;
    t->config.vocab_size = gc->vocab_size;
    t->config.seq_len = gc->seq_len;

    /* Set architecture based on GGUF metadata */
    if (gc->arch == GGUF_ARCH_GPT2) {
        t->config.arch = ARCH_GPT2;
        printf("Loading GPT-2 model from GGUF...\n");
    } else {
        t->config.arch = ARCH_LLAMA;
        printf("Loading LLaMA model from GGUF...\n");
    }

    int ret;
    if (t->config.arch == ARCH_GPT2) {
        ret = build_gpt2_from_gguf(t, ctx);
    } else {
        ret = build_llama_from_gguf(t, ctx);
    }

    if (ret != 0) {
        printf("ERROR: Failed to load model weights\n");
        return ret;
    }

    /* Allocate run state */
    malloc_run_state(&t->state, &t->config);

    printf("Model loaded successfully!\n");
    return 0;
}

/* GGUF-based tokenizer decode */
static char* decode_gguf(int prev_token, int token) {
    static char token_str[128];
    static char decoded[128];
    if (gguf_get_vocab_string(&g_gguf_ctx, token, token_str, sizeof(token_str)) < 0) {
        return "(ERR)";
    }

    /* Handle different tokenizer encodings:
     * GPT-2 BPE:
     * - Ġ (U+0120, UTF-8: 0xC4 0xA0) represents space
     * - Ċ (U+010A, UTF-8: 0xC4 0x8A) represents newline
     * SentencePiece (LLaMA/TinyLlama):
     * - ▁ (U+2581, UTF-8: 0xE2 0x96 0x81) represents space
     * Convert these back to normal characters */
    int j = 0;
    for (int i = 0; token_str[i] && j < (int)sizeof(decoded) - 1; i++) {
        unsigned char c = (unsigned char)token_str[i];
        if (c == 0xC4 && (unsigned char)token_str[i+1] == 0xA0) {
            decoded[j++] = ' ';  /* Ġ -> space */
            i++;
        } else if (c == 0xC4 && (unsigned char)token_str[i+1] == 0x8A) {
            decoded[j++] = '\n'; /* Ċ -> newline */
            i++;
        } else if (c == 0xE2 && (unsigned char)token_str[i+1] == 0x96 &&
                   (unsigned char)token_str[i+2] == 0x81) {
            decoded[j++] = ' ';  /* ▁ -> space (SentencePiece) */
            i += 2;
        } else {
            decoded[j++] = token_str[i];
        }
    }
    decoded[j] = '\0';

    /* Skip leading space after BOS */
    if (prev_token == (int)g_gguf_ctx.config.bos_token_id && decoded[0] == ' ') {
        return decoded + 1;
    }
    return decoded;
}

/* Generation loop for GGUF models */
static void generate_gguf(Transformer *transformer, GGUFContext* ctx, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+128) * sizeof(int));
    if (!prompt_tokens) {
        printf("ERROR: malloc failed\n");
        while(1);
    }

    /* Use BOS token only - tokenization is too slow on this hardware */
    prompt_tokens[num_prompt_tokens++] = ctx->config.bos_token_id;

    uint64_t start_cycles = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    int tokens_generated = 0;

    /* Start output */
    printf("\n> ");

    while (pos < steps) {
        float* logits = forward(transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
            tokens_generated++;
        }
        pos++;

        /* Check for EOS */
        if (next == (int)ctx->config.eos_token_id) { break; }

        /* Validate token is in vocab range */
        if (next < 0 || next >= (int)ctx->config.vocab_size) {
            printf("\nERROR: invalid token %d\n", next);
            break;
        }

        /* Decode and print token */
        char* piece = decode_gguf(token, next);
        printf("%s", piece);
        token = next;

        if (start_cycles == 0 && pos >= num_prompt_tokens) {
            start_cycles = ((uint64_t)SYS_CYCLE_HI << 32) | SYS_CYCLE_LO;
        }
    }
    printf("\n");

    if (tokens_generated > 1 && start_cycles > 0) {
        uint64_t end_cycles = ((uint64_t)SYS_CYCLE_HI << 32) | SYS_CYCLE_LO;
        uint64_t elapsed_cycles = end_cycles - start_cycles;
        uint32_t elapsed_ms = (uint32_t)(elapsed_cycles / 50000);
        if (elapsed_ms > 0) {
            uint32_t tok_per_min = (uint32_t)((uint64_t)tokens_generated * 60000 / elapsed_ms);
            printf("Speed: %d tok/min (%d ms total)\n", tok_per_min, elapsed_ms);
        }
    }

    free(prompt_tokens);
}

/* ============================================
 * Main entry point
 * ============================================ */

/*
 * Memory Layout:
 *
 * SDRAM (64MB): Model loaded by APF (single GGUF file or model.bin+tokenizer.bin)
 * Bridge 0x00000000 -> CPU 0x10000000: Model/GGUF file
 * Bridge 0x03F00000 -> CPU 0x13F00000: Tokenizer (only for model.bin format)
 *
 * PSRAM (CRAM0, 16MB): Heap for runtime allocations
 * CPU 0x30000000 - 0x30FFFFFF
 */
#define MODEL_SDRAM_ADDR      0x10000000                  /* Slot 0: bridge 0x00000000 */
#define TOKENIZER_SDRAM_ADDR  0x13F00000                  /* Slot 1: bridge 0x03F00000 */
#define HEAP_PSRAM_ADDR       0x30000000                  /* Heap in PSRAM (CRAM0) */
#define HEAP_SIZE             (PSRAM_CACHE_ADDR - HEAP_PSRAM_ADDR)  /* 8MB for heap, upper 8MB for KV cache */

void llama_main(void) {
    printf("LLM Inference Engine\n\n");

    /* Wait for SDRAM and APF automatic data slot loading */
    while (!(SYS_STATUS & SYS_STATUS_SDRAM_READY)) {}
    printf("SDRAM ready, waiting for data...\n");

    /* Wait for APF to finish auto-loading data slots */
    while (!(SYS_STATUS & SYS_STATUS_DATASLOT_COMPLETE)) {}
    printf("Data loaded\n");

    /* Test PSRAM write/read from CPU */
    volatile uint32_t *test_addr = (volatile uint32_t *)HEAP_PSRAM_ADDR;
    *test_addr = 0xDEADBEEF;
    if (*test_addr != 0xDEADBEEF) {
        printf("ERROR: PSRAM write failed!\n");
        while(1);
    }
    heap_init((void*)HEAP_PSRAM_ADDR, HEAP_SIZE);
    printf("PSRAM heap OK\n");

    /* Check if model is GGUF format */
    volatile uint32_t *model_header = (volatile uint32_t *)MODEL_SDRAM_ADDR;
    uint32_t magic = model_header[0];
    int is_gguf = (magic == GGUF_MAGIC);

    Transformer transformer;

    if (is_gguf) {
        printf("Detected GGUF format model\n");

        /* Initialize GGUF parser */
        if (gguf_init(&g_gguf_ctx, (const uint8_t*)MODEL_SDRAM_ADDR, 64 * 1024 * 1024) != 0) {
            printf("ERROR: Failed to parse GGUF header\n");
            while(1);
        }

        /* Parse metadata */
        if (gguf_parse_metadata(&g_gguf_ctx) != 0) {
            printf("ERROR: Failed to parse GGUF metadata\n");
            while(1);
        }

        /* Build transformer from GGUF */
        if (build_transformer_from_gguf(&transformer, &g_gguf_ctx) != 0) {
            printf("ERROR: Failed to build transformer from GGUF\n");
            while(1);
        }
    } else {
        /* Quick model sanity check for model.bin format */
        uint32_t dim = model_header[0];
        if (dim == 0 || dim > 10000) {
            printf("ERROR: Invalid model (magic=0x%08X dim=%d)\n", magic, dim);
            while(1);
        }
        printf("Detected model.bin format (dim=%d)\n", dim);

        /* Build transformer from model.bin */
        build_transformer_from_memory(&transformer, (void*)MODEL_SDRAM_ADDR, 0);
    }
#if USE_DMA_ACCEL
    /* Allocate SDRAM buffer for x vector (used by DMA accelerator) */
    {
        int max_dim = transformer.config.dim > transformer.config.hidden_dim ?
                      transformer.config.dim : transformer.config.hidden_dim;
        x_sdram_buf = sdram_alloc(max_dim * sizeof(int32_t));
        if (!x_sdram_buf) {
            printf("ERROR: failed to allocate x_sdram_buf!\n");
            while(1);
        }

        /* Allocate SDRAM buffer for attention K vectors (batch processing) */
        int head_size = transformer.config.dim / transformer.config.n_heads;
        int k_batch_words = ATTN_BATCH_SIZE * head_size;
        k_sdram_batch = sdram_alloc(k_batch_words * sizeof(int32_t));
        if (!k_sdram_batch) {
            printf("ERROR: failed to allocate k_sdram_batch!\n");
            while(1);
        }
    }

    /* Convert weight matrices from float to Q16.16 for hardware accelerator */
    printf("Converting weights...\n");
    {
        Config* p = &transformer.config;
        TransformerWeights* w = &transformer.weights;
        int head_size = p->dim / p->n_heads;
        size_t n_layers = p->n_layers;

        if (p->arch == ARCH_GPT2) {
            /* GPT-2 weights - convert per-layer if using direct pointers */
            if (w->wqkv) {
                convert_weights_to_q16(w->wqkv, n_layers * p->dim * 3 * p->dim);
            } else {
                for (size_t l = 0; l < n_layers; l++) {
                    convert_weights_to_q16(w->wqkv_layer[l], p->dim * 3 * p->dim);
                }
            }
            if (w->wo) {
                convert_weights_to_q16(w->wo, n_layers * p->dim * p->dim);
            } else {
                for (size_t l = 0; l < n_layers; l++) {
                    convert_weights_to_q16(w->wo_layer[l], p->dim * p->dim);
                }
            }
            if (w->ffn_up_weight) {
                convert_weights_to_q16(w->ffn_up_weight, n_layers * p->dim * p->hidden_dim);
            } else {
                for (size_t l = 0; l < n_layers; l++) {
                    convert_weights_to_q16(w->ffn_up_layer[l], p->dim * p->hidden_dim);
                }
            }
            if (w->ffn_down_weight) {
                convert_weights_to_q16(w->ffn_down_weight, n_layers * p->hidden_dim * p->dim);
            } else {
                for (size_t l = 0; l < n_layers; l++) {
                    convert_weights_to_q16(w->ffn_down_layer[l], p->hidden_dim * p->dim);
                }
            }
        } else {
            /* LLaMA weights */
            /* wq, wk, wv, wo - attention weights */
            convert_weights_to_q16(w->wq, n_layers * p->dim * (p->n_heads * head_size));
            convert_weights_to_q16(w->wk, n_layers * p->dim * (p->n_kv_heads * head_size));
            convert_weights_to_q16(w->wv, n_layers * p->dim * (p->n_kv_heads * head_size));
            convert_weights_to_q16(w->wo, n_layers * (p->n_heads * head_size) * p->dim);

            /* w1, w2, w3 - FFN weights */
            convert_weights_to_q16(w->w1, n_layers * p->dim * p->hidden_dim);
            convert_weights_to_q16(w->w2, n_layers * p->hidden_dim * p->dim);
            convert_weights_to_q16(w->w3, n_layers * p->dim * p->hidden_dim);
        }

        /* wcls - output projection (may be shared with embedding) */
        if (w->wcls != w->token_embedding_table) {
            convert_weights_to_q16(w->wcls, p->vocab_size * p->dim);
        }

        weights_converted = 1;
    }

#endif

    /* Build sampler */
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, DEFAULT_TEMPERATURE, DEFAULT_TOPP, SYS_CYCLE_LO);

    /* Run generation */
    printf("\n--- Generating ---\n");
    printf("Prompt: \"%s\"\n\n", DEFAULT_PROMPT);

    if (is_gguf) {
        /* Generate using GGUF tokenizer */
        generate_gguf(&transformer, &g_gguf_ctx, &sampler, (char*)DEFAULT_PROMPT, DEFAULT_STEPS);
    } else {
        /* Build tokenizer from SDRAM (model.bin format) */
        Tokenizer tokenizer;
        build_tokenizer_from_memory(&tokenizer, (void*)TOKENIZER_SDRAM_ADDR, transformer.config.vocab_size);
        g_tokenizer = &tokenizer;

        generate(&transformer, &tokenizer, &sampler, (char*)DEFAULT_PROMPT, DEFAULT_STEPS);
        free_tokenizer(&tokenizer);
    }

    /* Cleanup */
    printf("\nCleaning up...\n");
    free_sampler(&sampler);
    free_transformer(&transformer);

    printf("Done!\n");

    /* Halt */
    while(1);
}