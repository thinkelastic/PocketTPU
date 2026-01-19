//
// DMA Dot Product Accelerator
// Streams vectors directly from SDRAM using burst reads
// Memory-mapped interface at 0x50000000
//
// Registers:
//   0x00: CTRL       - Write to start, read for status (bit 0 = busy)
//                      Write bits: [0]=start, [1]=use_cached_b, [2]=preload_b_only
//   0x04: LENGTH     - Vector length in elements (up to 256)
//   0x08: RESULT_LO  - Low 32 bits of accumulated result
//   0x0C: RESULT_HI  - High 32 bits of accumulated result
//   0x10: ADDR_A     - SDRAM word address for vector A (24-bit)
//   0x14: ADDR_B     - SDRAM word address for vector B (24-bit)
//
// Operation modes:
//   Normal (CTRL=1): Fetch A and B from SDRAM, compute dot product
//   Preload B (CTRL=5): Fetch B into cache, no computation
//   Use cached B (CTRL=3): Fetch only A, use cached B, compute dot product
//
// Vectors are Q16.16 fixed-point (pre-converted by firmware)
// Result is 64-bit to handle overflow from accumulation
//

`default_nettype none

module dma_dot_product #(
    parameter MAX_LENGTH = 512    // Maximum vector length (increased for B-caching)
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

// B-caching control
reg use_cached_b;           // Use cached B instead of fetching
reg preload_b_only;         // Only preload B, no computation
reg [9:0] cached_b_length;  // Length of cached B vector

// State machine
localparam STATE_IDLE     = 3'd0;
localparam STATE_FETCH_A  = 3'd1;
localparam STATE_WAIT_A   = 3'd2;
localparam STATE_FETCH_B  = 3'd3;
localparam STATE_WAIT_B   = 3'd4;
localparam STATE_COMPUTE  = 3'd5;
localparam STATE_DONE     = 3'd6;

reg [2:0] state;

// Vector buffers - dual-port for simultaneous load and compute
// Use simple arrays - Quartus will infer BRAM
reg signed [31:0] vec_a [0:MAX_LENGTH-1];
reg signed [31:0] vec_b [0:MAX_LENGTH-1];

// Fetch counters
reg [9:0] fetch_idx;

// Computation pipeline
reg [9:0] comp_idx;

// Pipeline registers for multiply-accumulate
// Stage 1: read operands
reg signed [31:0] op_a, op_b;
reg pipe1_valid;

// Stage 2: multiply
reg signed [63:0] product;
reg pipe2_valid;

// Use burst_32bit for 32-bit transfers
assign burst_32bit = 1'b1;

// Combinational ready - always respond immediately for reg access
assign reg_ready = reg_valid;

// Combinational read data
reg [31:0] rdata_comb;
always @(*) begin
    case (reg_addr[7:2])
        6'h00: rdata_comb = {31'b0, busy};           // CTRL/STATUS
        6'h01: rdata_comb = {22'b0, vec_length};    // LENGTH (10 bits, up to 512)
        6'h02: rdata_comb = accumulator[31:0];      // RESULT_LO
        6'h03: rdata_comb = accumulator[63:32];     // RESULT_HI
        6'h04: rdata_comb = {8'b0, addr_a};         // ADDR_A
        6'h05: rdata_comb = {8'b0, addr_b};         // ADDR_B
        default: rdata_comb = 32'h0;
    endcase
end
assign reg_rdata = rdata_comb;

// Track if we've processed this access
reg access_done;

// Main state machine
always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        busy <= 0;
        vec_length <= 0;
        accumulator <= 0;
        addr_a <= 0;
        addr_b <= 0;
        use_cached_b <= 0;
        preload_b_only <= 0;
        cached_b_length <= 0;
        state <= STATE_IDLE;
        burst_rd <= 0;
        burst_addr <= 0;
        burst_len <= 0;
        fetch_idx <= 0;
        comp_idx <= 0;
        access_done <= 0;
        pipe1_valid <= 0;
        pipe2_valid <= 0;
        op_a <= 0;
        op_b <= 0;
        product <= 0;
    end else begin
        // Default: deassert burst_rd after one cycle
        burst_rd <= 0;

        // Clear access_done when valid goes low
        if (!reg_valid) begin
            access_done <= 0;
        end

        // Handle register writes (only once per bus cycle)
        if (reg_valid && reg_write && !access_done && !busy) begin
            access_done <= 1;
            case (reg_addr[7:2])
                6'h00: begin  // CTRL
                    if (reg_wdata[0]) begin
                        // Start operation
                        // Bit 0: start
                        // Bit 1: use cached B (skip B fetch)
                        // Bit 2: preload B only (fetch B, no A, no compute)
                        busy <= 1;
                        use_cached_b <= reg_wdata[1];
                        preload_b_only <= reg_wdata[2];
                        accumulator <= 0;
                        fetch_idx <= 0;
                        comp_idx <= 0;
                        pipe1_valid <= 0;
                        pipe2_valid <= 0;

                        if (reg_wdata[2]) begin
                            // Preload B only mode - go directly to fetch B
                            state <= STATE_FETCH_B;
                            cached_b_length <= vec_length;  // Save length for later use
                        end else begin
                            // Normal or use-cached-B mode - start with fetch A
                            state <= STATE_FETCH_A;
                        end
                    end
                end
                6'h01: vec_length <= reg_wdata[9:0];
                6'h04: addr_a <= reg_wdata[23:0];
                6'h05: addr_b <= reg_wdata[23:0];
                default: ;
            endcase
        end

        // State machine
        case (state)
            STATE_IDLE: begin
                // Wait for start
            end

            STATE_FETCH_A: begin
                // Start burst read for vector A
                // addr_a is word address (32-bit), burst_addr needs half-word address (16-bit)
                // So multiply by 2 (shift left by 1)
                // burst_len is in half-words, so for N 32-bit words we need 2*N half-words
                burst_rd <= 1;
                burst_addr <= {addr_a, 1'b0};  // 25-bit half-word address
                burst_len <= {vec_length, 1'b0};  // vec_length * 2 for 32-bit mode
                fetch_idx <= 0;
                state <= STATE_WAIT_A;
            end

            STATE_WAIT_A: begin
                // Store incoming data
                if (burst_data_valid) begin
                    vec_a[fetch_idx] <= burst_data;
                    fetch_idx <= fetch_idx + 1;
                end
                if (burst_data_done) begin
                    fetch_idx <= 0;
                    if (use_cached_b) begin
                        // Skip B fetch, go directly to compute using cached B
                        state <= STATE_COMPUTE;
                        comp_idx <= 0;
                        pipe1_valid <= 0;
                        pipe2_valid <= 0;
                    end else begin
                        // Normal mode - fetch B
                        state <= STATE_FETCH_B;
                    end
                end
            end

            STATE_FETCH_B: begin
                // Start burst read for vector B
                // addr_b is word address (32-bit), burst_addr needs half-word address (16-bit)
                // burst_len is in half-words, so for N 32-bit words we need 2*N half-words
                burst_rd <= 1;
                burst_addr <= {addr_b, 1'b0};  // 25-bit half-word address
                burst_len <= {vec_length, 1'b0};  // vec_length * 2 for 32-bit mode
                fetch_idx <= 0;
                state <= STATE_WAIT_B;
            end

            STATE_WAIT_B: begin
                // Store incoming data
                if (burst_data_valid) begin
                    vec_b[fetch_idx] <= burst_data;
                    fetch_idx <= fetch_idx + 1;
                end
                if (burst_data_done) begin
                    if (preload_b_only) begin
                        // Preload complete - go directly to done, no compute
                        state <= STATE_DONE;
                    end else begin
                        // Normal mode - proceed to compute
                        state <= STATE_COMPUTE;
                        comp_idx <= 0;
                        pipe1_valid <= 0;
                        pipe2_valid <= 0;
                    end
                end
            end

            STATE_COMPUTE: begin
                // 3-stage pipeline: read -> multiply -> accumulate

                // Stage 1: Read operands from BRAM
                if (comp_idx < vec_length) begin
                    op_a <= vec_a[comp_idx];
                    op_b <= vec_b[comp_idx];
                    pipe1_valid <= 1;
                    comp_idx <= comp_idx + 1;
                end else begin
                    pipe1_valid <= 0;
                end

                // Stage 2: Multiply
                if (pipe1_valid) begin
                    product <= op_a * op_b;
                    pipe2_valid <= 1;
                end else begin
                    pipe2_valid <= 0;
                end

                // Stage 3: Accumulate
                if (pipe2_valid) begin
                    accumulator <= accumulator + product;
                end

                // Check if pipeline is drained
                if (comp_idx >= vec_length && !pipe1_valid && !pipe2_valid) begin
                    state <= STATE_DONE;
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
