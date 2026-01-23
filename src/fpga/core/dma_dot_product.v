//
// DMA Dot Product Accelerator with Double-Buffering and Q8 Dequantization
// Streams vectors directly from SDRAM using burst reads
// Memory-mapped interface at 0x50000000
//
// Registers:
//   0x00: CTRL       - Write to start, read for status (bit 0 = busy, bit 4 = ready for next)
//                      Write bits: [0]=start, [1]=use_cached_b, [2]=preload_b_only, [3]=pipeline_mode
//                                  [4]=use_weight_cache, [5]=q8_mode
//   0x04: LENGTH     - Vector length in elements (up to 512)
//   0x08: RESULT_LO  - Low 32 bits of accumulated result
//   0x0C: RESULT_HI  - High 32 bits of accumulated result
//   0x10: ADDR_A     - SDRAM word address for vector A (24-bit)
//   0x14: ADDR_B     - SDRAM word address for vector B (24-bit)
//   0x18: ADDR_A_NEXT - Next A address for pipelined operation
//
// Q8 Mode (bit 5):
//   Q8_0 format: 34 bytes per 32 elements (2-byte FP16 scale + 32 int8 values)
//   Hardware dequantizes on-the-fly during DMA fetch
//   For 512 elements: 16 blocks x 34 bytes = 544 bytes = 272 x 16-bit words
//
// Double-buffering: While computing dot product with buffer 0, DMA fills buffer 1
// This overlaps memory latency with computation for ~2x throughput
//
// Vectors are Q16.16 fixed-point (either pre-converted or Q8-dequantized in hardware)
// Result is 64-bit to handle overflow from accumulation
//

`default_nettype none

module dma_dot_product #(
    parameter MAX_LENGTH = 512   // Maximum vector length
) (
    input wire clk,
    input wire reset_n,

    // CPU register interface
    input wire         reg_valid,
    input wire         reg_write,
    input wire  [7:0]  reg_addr,
    input wire  [31:0] reg_wdata,
    output wire [31:0] reg_rdata,
    output wire        reg_ready,

    // SDRAM burst read interface (directly to io_sdram)
    output reg         burst_rd,
    output reg  [24:0] burst_addr,
    output reg  [10:0] burst_len,
    output wire        burst_32bit,
    input wire  [31:0] burst_data,
    input wire         burst_data_valid,
    input wire         burst_data_done
);

// Control/status
reg busy;
reg [9:0] vec_length;        // Up to 512 elements
reg signed [63:0] accumulator;

// Address registers
reg [23:0] addr_a;
reg [23:0] addr_b;
reg [23:0] addr_a_next;      // Next A address for pipelining

// B-caching control
reg use_cached_b;           // Use cached B instead of fetching
reg preload_b_only;         // Only preload B, no computation
reg pipeline_mode;          // Enable double-buffering pipeline
reg [9:0] cached_b_length;  // Length of cached B vector

// Q8 mode
reg q8_mode;                // Enable Q8 dequantization

// Double-buffer control
reg active_buf;             // Which buffer is being used for compute (0 or 1)
reg prefetch_pending;       // A prefetch is in progress
reg prefetch_done;          // Prefetch completed, ready for next compute
reg ready_for_next;         // Can accept next operation

// State machine - expanded for concurrent operation
localparam STATE_IDLE           = 4'd0;
localparam STATE_FETCH_A        = 4'd1;
localparam STATE_WAIT_A         = 4'd2;
localparam STATE_FETCH_B        = 4'd3;
localparam STATE_WAIT_B         = 4'd4;
localparam STATE_COMPUTE        = 4'd5;
localparam STATE_DONE           = 4'd6;
localparam STATE_COMPUTE_FETCH  = 4'd7;  // Compute while fetching next A
localparam STATE_WAIT_PREFETCH  = 4'd8;  // Wait for prefetch to complete
localparam STATE_CACHE_LOAD     = 4'd9;  // Start DMA for cache load
localparam STATE_CACHE_WAIT     = 4'd10; // Wait for cache load to complete
localparam STATE_CACHE_PRIME    = 4'd11; // Prime cache read pipeline (1 cycle delay)
localparam STATE_FETCH_A_Q8     = 4'd12; // Start Q8 fetch
localparam STATE_WAIT_A_Q8      = 4'd13; // Wait for Q8 data and dequantize

reg [3:0] state;

// Double A buffers
reg signed [31:0] vec_a0 [0:MAX_LENGTH-1];
reg signed [31:0] vec_a1 [0:MAX_LENGTH-1];
reg signed [31:0] vec_b [0:MAX_LENGTH-1];

// ==========================================================================
// BRAM Weight Cache - 1 slot x 4096 elements = 16 KB
// Caches one layer's output projection weights (64x64 = 4096 elements)
// ==========================================================================
parameter CACHE_SLOTS = 1;
parameter CACHE_SLOT_SIZE = 4096;
(* ramstyle = "M10K" *) reg signed [31:0] weight_cache [0:CACHE_SLOT_SIZE-1];

// Cache control registers
reg [3:0] cache_slot;               // Active slot (0-15)
reg cache_load_busy;                // Loading in progress
reg [23:0] cache_sdram_addr;        // SDRAM source address for loading
reg [11:0] cache_load_length;       // Elements to load (up to 4096)
reg [11:0] cache_write_idx;         // Write index during load
reg use_weight_cache;               // Use cache instead of SDRAM for A
reg [15:0] cache_slot_valid;        // Bitmask of valid slots
reg [11:0] cache_row_offset;        // Row offset within cache slot (for matmul)

// Fetch counters
reg [9:0] fetch_idx;

// Computation pipeline
reg [9:0] comp_idx;

// Pipeline registers for 2-way parallel multiply-accumulate
reg signed [31:0] op_a0, op_a1;
reg signed [31:0] op_b0, op_b1;
reg pipe1_valid;
reg signed [63:0] prod0, prod1;
reg pipe2_valid;

// ==========================================================================
// Q8 Dequantization Logic
// Q8_0 format: 2-byte FP16 scale + 32 int8 values = 34 bytes per block
// We read 17 x 16-bit words per block (34 bytes)
// Word 0: FP16 scale
// Words 1-16: 32 x int8 values (2 per word)
// ==========================================================================

// Q8 block state
reg [4:0] q8_block_idx;       // Current block (0-15 for 512 elements)
reg [4:0] q8_word_in_block;   // Word within block (0-16)
reg signed [31:0] q8_scale_q16; // Converted Q16.16 scale
reg [9:0] q8_elem_idx;        // Element index for storing dequantized values

// FP16 to Q16.16 conversion - registered pipeline
// Capture FP16 scale on cycle N, use converted value on cycle N+1
reg [15:0] fp16_scale_reg;

// FP16 conversion: sign(1) | exp(5) | mant(10)
// For Q16.16: value = sign * (1 + mant/1024) * 2^(exp-15) * 65536
//                   = sign * (1024 + mant) * 2^(exp-15+16-10)
//                   = sign * (1024 + mant) * 2^(exp-9)
wire fp16_sign = fp16_scale_reg[15];
wire [4:0] fp16_exp = fp16_scale_reg[14:10];
wire [9:0] fp16_mant = fp16_scale_reg[9:0];
wire [10:0] fp16_mant_full = {1'b1, fp16_mant};  // Add implicit 1

// Simplified shift using case statement (synthesizes more efficiently than barrel shifter)
reg [31:0] fp16_shifted;
always @(*) begin
    case (fp16_exp)
        5'd0:  fp16_shifted = 32'd0;  // Zero/subnormal
        5'd1:  fp16_shifted = {24'b0, fp16_mant_full[10:3]};   // exp-9 = -8
        5'd2:  fp16_shifted = {23'b0, fp16_mant_full[10:2]};   // exp-9 = -7
        5'd3:  fp16_shifted = {22'b0, fp16_mant_full[10:1]};   // exp-9 = -6
        5'd4:  fp16_shifted = {21'b0, fp16_mant_full};         // exp-9 = -5
        5'd5:  fp16_shifted = {20'b0, fp16_mant_full, 1'b0};   // exp-9 = -4
        5'd6:  fp16_shifted = {19'b0, fp16_mant_full, 2'b0};   // exp-9 = -3
        5'd7:  fp16_shifted = {18'b0, fp16_mant_full, 3'b0};   // exp-9 = -2
        5'd8:  fp16_shifted = {17'b0, fp16_mant_full, 4'b0};   // exp-9 = -1
        5'd9:  fp16_shifted = {16'b0, fp16_mant_full, 5'b0};   // exp-9 = 0
        5'd10: fp16_shifted = {15'b0, fp16_mant_full, 6'b0};   // exp-9 = 1
        5'd11: fp16_shifted = {14'b0, fp16_mant_full, 7'b0};   // exp-9 = 2
        5'd12: fp16_shifted = {13'b0, fp16_mant_full, 8'b0};   // exp-9 = 3
        5'd13: fp16_shifted = {12'b0, fp16_mant_full, 9'b0};   // exp-9 = 4
        5'd14: fp16_shifted = {11'b0, fp16_mant_full, 10'b0};  // exp-9 = 5
        5'd15: fp16_shifted = {10'b0, fp16_mant_full, 11'b0};  // exp-9 = 6
        5'd16: fp16_shifted = {9'b0, fp16_mant_full, 12'b0};   // exp-9 = 7
        5'd17: fp16_shifted = {8'b0, fp16_mant_full, 13'b0};   // exp-9 = 8
        5'd18: fp16_shifted = {7'b0, fp16_mant_full, 14'b0};   // exp-9 = 9
        5'd19: fp16_shifted = {6'b0, fp16_mant_full, 15'b0};   // exp-9 = 10
        5'd20: fp16_shifted = {5'b0, fp16_mant_full, 16'b0};   // exp-9 = 11
        5'd21: fp16_shifted = {4'b0, fp16_mant_full, 17'b0};   // exp-9 = 12
        default: fp16_shifted = 32'h7FFFFFFF;  // Saturate for exp >= 22
    endcase
end

// Apply sign
wire [31:0] q16_from_fp16 = (fp16_exp == 5'd0) ? 32'd0 :
                            fp16_sign ? -fp16_shifted : fp16_shifted;

// Int8 dequantization: result = scale_q16 * int8
wire [15:0] q8_data_word = burst_data[15:0];
wire signed [7:0] q8_val_lo = q8_data_word[7:0];
wire signed [7:0] q8_val_hi = q8_data_word[15:8];

// Select scale: use q16_from_fp16 for word 1 (scale just converted), else use registered scale
wire signed [31:0] q8_active_scale = (q8_word_in_block == 5'd1) ? q16_from_fp16 : q8_scale_q16;

// Dequantized values: scale * int8
wire signed [31:0] q8_dequant_lo = q8_active_scale * q8_val_lo;
wire signed [31:0] q8_dequant_hi = q8_active_scale * q8_val_hi;

// Use burst_32bit = 0 for 16-bit transfers in Q8 mode, 1 for 32-bit
assign burst_32bit = ~q8_mode;

// Combinational ready - always respond immediately for reg access
assign reg_ready = reg_valid;

// Combinational read data
reg [31:0] rdata_comb;
always @(*) begin
    case (reg_addr[7:2])
        6'h00: rdata_comb = {26'b0, q8_mode, ready_for_next, 3'b0, busy};  // CTRL/STATUS
        6'h01: rdata_comb = {22'b0, vec_length};    // LENGTH
        6'h02: rdata_comb = accumulator[31:0];      // RESULT_LO
        6'h03: rdata_comb = accumulator[63:32];     // RESULT_HI
        6'h04: rdata_comb = {8'b0, addr_a};         // ADDR_A
        6'h05: rdata_comb = {8'b0, addr_b};         // ADDR_B
        6'h06: rdata_comb = {8'b0, addr_a_next};    // ADDR_A_NEXT
        // Cache registers (0x20-0x30)
        6'h08: rdata_comb = {27'b0, cache_load_busy, cache_slot};  // 0x20 CACHE_CTRL
        6'h09: rdata_comb = cache_slot_valid;       // 0x24 CACHE_VALID
        6'h0A: rdata_comb = {8'b0, cache_sdram_addr}; // 0x28 CACHE_ADDR
        6'h0B: rdata_comb = {20'b0, cache_load_length}; // 0x2C CACHE_LEN
        6'h0C: rdata_comb = {20'b0, cache_row_offset}; // 0x30 CACHE_ROW_OFFSET
        default: rdata_comb = 32'h0;
    endcase
end
assign reg_rdata = rdata_comb;

// Track if we've processed this access
reg access_done;

// 2-way parallel reads from active buffer
wire signed [31:0] vec_a_read0 = active_buf ? vec_a1[comp_idx]   : vec_a0[comp_idx];
wire signed [31:0] vec_a_read1 = active_buf ? vec_a1[comp_idx+1] : vec_a0[comp_idx+1];

// Cache read - synchronous for M10K inference
// Address is combinational (fast), data is registered (1-cycle latency)
wire [11:0] cache_read_addr = cache_row_offset + {2'b0, comp_idx};
reg signed [31:0] cache_read0_reg, cache_read1_reg;

// Select weight source: cache or DMA buffer
wire signed [31:0] weight_val0 = use_weight_cache ? cache_read0_reg : vec_a_read0;
wire signed [31:0] weight_val1 = use_weight_cache ? cache_read1_reg : vec_a_read1;

// Calculate Q8 burst length: (n_elements / 32) blocks * 17 words per block
// For 512 elements: 16 blocks * 17 = 272 words
wire [10:0] q8_burst_len = ((vec_length + 10'd31) >> 5) * 11'd17;

// Main state machine
always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        busy <= 0;
        vec_length <= 0;
        accumulator <= 0;
        addr_a <= 0;
        addr_b <= 0;
        addr_a_next <= 0;
        use_cached_b <= 0;
        preload_b_only <= 0;
        pipeline_mode <= 0;
        q8_mode <= 0;
        cached_b_length <= 0;
        active_buf <= 0;
        prefetch_pending <= 0;
        prefetch_done <= 0;
        ready_for_next <= 0;
        state <= STATE_IDLE;
        burst_rd <= 0;
        burst_addr <= 0;
        burst_len <= 0;
        fetch_idx <= 0;
        comp_idx <= 0;
        access_done <= 0;
        pipe1_valid <= 0;
        pipe2_valid <= 0;
        op_a0 <= 0; op_a1 <= 0;
        op_b0 <= 0; op_b1 <= 0;
        prod0 <= 0; prod1 <= 0;
        // Cache registers
        cache_slot <= 0;
        cache_load_busy <= 0;
        cache_sdram_addr <= 0;
        cache_load_length <= 0;
        cache_write_idx <= 0;
        use_weight_cache <= 0;
        cache_slot_valid <= 0;
        cache_row_offset <= 0;
        cache_read0_reg <= 0;
        cache_read1_reg <= 0;
        // Q8 registers
        q8_block_idx <= 0;
        q8_word_in_block <= 0;
        q8_scale_q16 <= 0;
        q8_elem_idx <= 0;
        fp16_scale_reg <= 0;
    end else begin
        // Default: deassert burst_rd after one cycle
        burst_rd <= 0;

        // Synchronous cache read (required for M10K block RAM inference)
        // Address is combinational, data is registered with 1-cycle latency
        cache_read0_reg <= weight_cache[cache_read_addr];
        cache_read1_reg <= weight_cache[cache_read_addr + 1];

        // Clear access_done when valid goes low
        if (!reg_valid) begin
            access_done <= 0;
        end

        // Handle register writes
        if (reg_valid && reg_write && !access_done) begin
            access_done <= 1;
            case (reg_addr[7:2])
                6'h00: begin  // CTRL
                    if (reg_wdata[0] && !busy) begin
                        // Start new operation
                        busy <= 1;
                        ready_for_next <= 0;
                        use_cached_b <= reg_wdata[1];
                        preload_b_only <= reg_wdata[2];
                        pipeline_mode <= reg_wdata[3];
                        use_weight_cache <= reg_wdata[4];  // Bit 4: use BRAM weight cache
                        q8_mode <= reg_wdata[5];           // Bit 5: Q8 dequantization mode
                        accumulator <= 0;
                        fetch_idx <= 0;
                        comp_idx <= 0;
                        pipe1_valid <= 0;
                        pipe2_valid <= 0;

                        if (reg_wdata[2]) begin
                            // Preload B only
                            state <= STATE_FETCH_B;
                            cached_b_length <= vec_length;
                        end else if (reg_wdata[4] && cache_slot_valid[0]) begin
                            // Use weight cache - skip SDRAM A fetch
                            if (reg_wdata[1]) begin
                                // Use cached B too - go to cache prime then compute
                                state <= STATE_CACHE_PRIME;
                            end else begin
                                // Need to fetch B first
                                state <= STATE_FETCH_B;
                            end
                        end else if (prefetch_done) begin
                            // Have prefetched data - switch buffers and compute
                            active_buf <= ~active_buf;
                            prefetch_done <= 0;
                            if (reg_wdata[3]) begin
                                // Pipeline mode - start fetching next while computing
                                state <= STATE_COMPUTE_FETCH;
                                prefetch_pending <= 1;
                            end else begin
                                state <= STATE_COMPUTE;
                            end
                        end else if (reg_wdata[5]) begin
                            // Q8 mode - use special fetch states
                            state <= STATE_FETCH_A_Q8;
                            q8_block_idx <= 0;
                            q8_word_in_block <= 0;
                            q8_elem_idx <= 0;
                        end else begin
                            // Normal start - fetch A first
                            state <= STATE_FETCH_A;
                        end
                    end
                end
                6'h01: vec_length <= reg_wdata[9:0];
                6'h04: addr_a <= reg_wdata[23:0];
                6'h05: addr_b <= reg_wdata[23:0];
                6'h06: addr_a_next <= reg_wdata[23:0];
                // Cache registers (0x20-0x2C)
                6'h08: begin  // CACHE_CTRL - start load or select slot
                    cache_slot <= reg_wdata[3:0];
                    if (reg_wdata[8] && !cache_load_busy && !busy) begin
                        // Bit 8: start cache load
                        cache_load_busy <= 1;
                        cache_write_idx <= 0;
                        state <= STATE_CACHE_LOAD;
                    end
                end
                6'h0A: cache_sdram_addr <= reg_wdata[23:0];
                6'h0B: cache_load_length <= reg_wdata[11:0];
                6'h0C: cache_row_offset <= reg_wdata[11:0];  // Row offset for matmul
                default: ;
            endcase
        end

        // State machine
        case (state)
            STATE_IDLE: begin
                // Wait for start
            end

            STATE_FETCH_A: begin
                // Start burst read for vector A into active buffer
                burst_rd <= 1;
                burst_addr <= {addr_a, 1'b0};
                burst_len <= {vec_length, 1'b0};
                fetch_idx <= 0;
                state <= STATE_WAIT_A;
            end

            STATE_WAIT_A: begin
                // Store incoming data into active buffer
                if (burst_data_valid) begin
                    if (active_buf)
                        vec_a1[fetch_idx] <= burst_data;
                    else
                        vec_a0[fetch_idx] <= burst_data;
                    fetch_idx <= fetch_idx + 1;
                end
                if (burst_data_done) begin
                    fetch_idx <= 0;
                    if (use_cached_b) begin
                        if (pipeline_mode) begin
                            // Start compute while fetching next
                            state <= STATE_COMPUTE_FETCH;
                            prefetch_pending <= 1;
                            comp_idx <= 0;
                            pipe1_valid <= 0;
                            pipe2_valid <= 0;
                        end else begin
                            state <= STATE_COMPUTE;
                            comp_idx <= 0;
                            pipe1_valid <= 0;
                            pipe2_valid <= 0;
                        end
                    end else begin
                        state <= STATE_FETCH_B;
                    end
                end
            end

            STATE_FETCH_B: begin
                burst_rd <= 1;
                burst_addr <= {addr_b, 1'b0};
                burst_len <= {vec_length, 1'b0};
                fetch_idx <= 0;
                state <= STATE_WAIT_B;
            end

            STATE_WAIT_B: begin
                if (burst_data_valid) begin
                    vec_b[fetch_idx] <= burst_data;
                    fetch_idx <= fetch_idx + 1;
                end
                if (burst_data_done) begin
                    if (preload_b_only) begin
                        state <= STATE_DONE;
                    end else if (use_weight_cache) begin
                        // Go to cache prime to handle 1-cycle read latency
                        state <= STATE_CACHE_PRIME;
                        comp_idx <= 0;
                        pipe1_valid <= 0;
                        pipe2_valid <= 0;
                    end else begin
                        state <= STATE_COMPUTE;
                        comp_idx <= 0;
                        pipe1_valid <= 0;
                        pipe2_valid <= 0;
                    end
                end
            end

            STATE_COMPUTE: begin
                // 2-way parallel computation from vec_a buffers or weight cache
                if (comp_idx < vec_length) begin
                    op_a0 <= weight_val0; op_b0 <= vec_b[comp_idx];
                    op_a1 <= weight_val1; op_b1 <= vec_b[comp_idx+1];
                    pipe1_valid <= 1;
                    comp_idx <= comp_idx + 2;
                end else begin
                    pipe1_valid <= 0;
                end

                if (pipe1_valid) begin
                    // 2 parallel multiplies
                    prod0 <= op_a0 * op_b0;
                    prod1 <= op_a1 * op_b1;
                    pipe2_valid <= 1;
                end else begin
                    pipe2_valid <= 0;
                end

                if (pipe2_valid) begin
                    // Sum both products and accumulate
                    accumulator <= accumulator + prod0 + prod1;
                end

                // Done when all elements processed
                if (comp_idx >= vec_length && !pipe1_valid && !pipe2_valid) begin
                    state <= STATE_DONE;
                end
            end

            STATE_COMPUTE_FETCH: begin
                // Concurrent 2-way parallel compute + prefetch into other buffer

                // 2-way parallel computation pipeline
                if (comp_idx < vec_length) begin
                    // Read 2 elements in parallel from weight cache or active buffer
                    op_a0 <= weight_val0; op_b0 <= vec_b[comp_idx];
                    op_a1 <= weight_val1; op_b1 <= vec_b[comp_idx+1];
                    pipe1_valid <= 1;
                    comp_idx <= comp_idx + 2;  // Process 2 elements per cycle
                end else begin
                    pipe1_valid <= 0;
                end

                if (pipe1_valid) begin
                    // 2 parallel multiplies
                    prod0 <= op_a0 * op_b0;
                    prod1 <= op_a1 * op_b1;
                    pipe2_valid <= 1;
                end else begin
                    pipe2_valid <= 0;
                end

                if (pipe2_valid) begin
                    // Sum both products and accumulate
                    accumulator <= accumulator + prod0 + prod1;
                end

                // Start prefetch on first cycle
                if (prefetch_pending && !burst_rd && fetch_idx == 0) begin
                    burst_rd <= 1;
                    burst_addr <= {addr_a_next, 1'b0};
                    burst_len <= {vec_length, 1'b0};
                end

                // Store prefetch data into OTHER buffer
                if (burst_data_valid) begin
                    if (active_buf)
                        vec_a0[fetch_idx] <= burst_data;  // Active=1, prefetch to 0
                    else
                        vec_a1[fetch_idx] <= burst_data;  // Active=0, prefetch to 1
                    fetch_idx <= fetch_idx + 1;
                end

                if (burst_data_done) begin
                    prefetch_pending <= 0;
                    prefetch_done <= 1;
                    ready_for_next <= 1;  // Signal CPU can queue next
                end

                // Check if compute is done
                if (comp_idx >= vec_length && !pipe1_valid && !pipe2_valid) begin
                    if (prefetch_pending) begin
                        // Compute done but prefetch still going
                        state <= STATE_WAIT_PREFETCH;
                    end else begin
                        state <= STATE_DONE;
                    end
                end
            end

            STATE_WAIT_PREFETCH: begin
                // Compute finished, waiting for prefetch to complete
                if (burst_data_valid) begin
                    if (active_buf)
                        vec_a0[fetch_idx] <= burst_data;
                    else
                        vec_a1[fetch_idx] <= burst_data;
                    fetch_idx <= fetch_idx + 1;
                end

                if (burst_data_done) begin
                    prefetch_pending <= 0;
                    prefetch_done <= 1;
                    ready_for_next <= 1;
                    state <= STATE_DONE;
                end
            end

            // ==========================================================================
            // Cache States
            // ==========================================================================

            // Prime cache read pipeline (1 cycle to fill read registers)
            STATE_CACHE_PRIME: begin
                // Cache read is happening this cycle (combinational address)
                // Next cycle, cache_read0/1_reg will have valid data
                // Transition to compute - first compute cycle will use correct data
                state <= STATE_COMPUTE;
                comp_idx <= 0;
                pipe1_valid <= 0;
                pipe2_valid <= 0;
            end

            STATE_CACHE_LOAD: begin
                // Start DMA burst read from SDRAM into weight cache
                burst_rd <= 1;
                burst_addr <= {cache_sdram_addr, 1'b0};  // Convert to byte address
                burst_len <= {cache_load_length, 1'b0};  // Convert to 16-bit words
                cache_write_idx <= 0;
                state <= STATE_CACHE_WAIT;
            end

            STATE_CACHE_WAIT: begin
                // Store incoming data into weight cache
                if (burst_data_valid) begin
                    weight_cache[cache_write_idx] <= burst_data;
                    cache_write_idx <= cache_write_idx + 1;
                end
                if (burst_data_done) begin
                    // Mark slot as valid
                    cache_slot_valid[cache_slot] <= 1;
                    cache_load_busy <= 0;
                    state <= STATE_IDLE;
                end
            end

            // ==========================================================================
            // Q8 Fetch States - Read Q8 blocks and dequantize to Q16.16
            // ==========================================================================

            STATE_FETCH_A_Q8: begin
                // Start burst read for Q8 data (16-bit words)
                burst_rd <= 1;
                burst_addr <= {addr_a[23:0], 1'b0};  // Byte address
                burst_len <= q8_burst_len;           // 17 words per 32 elements
                fetch_idx <= 0;
                q8_block_idx <= 0;
                q8_word_in_block <= 0;
                q8_elem_idx <= 0;
                state <= STATE_WAIT_A_Q8;
            end

            STATE_WAIT_A_Q8: begin
                // Process incoming Q8 data with pipelined FP16 conversion
                // Word 0 of each block: FP16 scale -> fp16_scale_reg
                // Word 1: capture converted scale + first data
                // Words 2-16: dequantize using captured scale

                if (burst_data_valid) begin
                    if (q8_word_in_block == 5'd0) begin
                        // First word of block: FP16 scale
                        // Register for conversion (combinational logic uses this)
                        fp16_scale_reg <= burst_data[15:0];
                        q8_word_in_block <= 5'd1;
                    end else if (q8_word_in_block == 5'd1) begin
                        // Second word: scale conversion is now valid, capture it
                        q8_scale_q16 <= q16_from_fp16;
                        // Store first data word (using converted scale)
                        if (q8_elem_idx < vec_length) begin
                            if (active_buf) begin
                                vec_a1[q8_elem_idx] <= q8_dequant_lo;
                                if (q8_elem_idx + 1 < vec_length)
                                    vec_a1[q8_elem_idx + 1] <= q8_dequant_hi;
                            end else begin
                                vec_a0[q8_elem_idx] <= q8_dequant_lo;
                                if (q8_elem_idx + 1 < vec_length)
                                    vec_a0[q8_elem_idx + 1] <= q8_dequant_hi;
                            end
                            q8_elem_idx <= q8_elem_idx + 2;
                        end
                        q8_word_in_block <= 5'd2;
                    end else begin
                        // Data words 2-16: scale already captured
                        if (q8_elem_idx < vec_length) begin
                            if (active_buf) begin
                                vec_a1[q8_elem_idx] <= q8_dequant_lo;
                                if (q8_elem_idx + 1 < vec_length)
                                    vec_a1[q8_elem_idx + 1] <= q8_dequant_hi;
                            end else begin
                                vec_a0[q8_elem_idx] <= q8_dequant_lo;
                                if (q8_elem_idx + 1 < vec_length)
                                    vec_a0[q8_elem_idx + 1] <= q8_dequant_hi;
                            end
                            q8_elem_idx <= q8_elem_idx + 2;
                        end

                        // Advance word counter
                        if (q8_word_in_block == 5'd16) begin
                            // End of block
                            q8_word_in_block <= 5'd0;
                            q8_block_idx <= q8_block_idx + 1;
                        end else begin
                            q8_word_in_block <= q8_word_in_block + 1;
                        end
                    end
                end

                if (burst_data_done) begin
                    fetch_idx <= 0;
                    if (use_cached_b) begin
                        state <= STATE_COMPUTE;
                        comp_idx <= 0;
                        pipe1_valid <= 0;
                        pipe2_valid <= 0;
                    end else begin
                        state <= STATE_FETCH_B;
                    end
                end
            end

            STATE_DONE: begin
                busy <= 0;
                state <= STATE_IDLE;
            end
        endcase
    end
end

endmodule
