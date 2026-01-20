//
// DMA Dot Product Accelerator with Double-Buffering
// Streams vectors directly from SDRAM using burst reads
// Memory-mapped interface at 0x50000000
//
// Registers:
//   0x00: CTRL       - Write to start, read for status (bit 0 = busy, bit 4 = ready for next)
//                      Write bits: [0]=start, [1]=use_cached_b, [2]=preload_b_only, [3]=pipeline_mode
//   0x04: LENGTH     - Vector length in elements (up to 512)
//   0x08: RESULT_LO  - Low 32 bits of accumulated result
//   0x0C: RESULT_HI  - High 32 bits of accumulated result
//   0x10: ADDR_A     - SDRAM word address for vector A (24-bit)
//   0x14: ADDR_B     - SDRAM word address for vector B (24-bit)
//   0x18: ADDR_A_NEXT - Next A address for pipelined operation
//
// Operation modes:
//   Normal (CTRL=1): Fetch A and B from SDRAM, compute dot product
//   Preload B (CTRL=5): Fetch B into cache, no computation
//   Use cached B (CTRL=3): Fetch only A, use cached B, compute dot product
//   Pipeline (CTRL=0xB): Use cached B, start fetching next A while computing current
//
// Double-buffering: While computing dot product with buffer 0, DMA fills buffer 1
// This overlaps memory latency with computation for ~2x throughput
//
// Vectors are Q16.16 fixed-point (pre-converted by firmware)
// Result is 64-bit to handle overflow from accumulation
//

`default_nettype none

module dma_dot_product #(
    parameter MAX_LENGTH = 512    // Maximum vector length
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

reg [3:0] state;

// Double A buffers
reg signed [31:0] vec_a0 [0:MAX_LENGTH-1];
reg signed [31:0] vec_a1 [0:MAX_LENGTH-1];
reg signed [31:0] vec_b [0:MAX_LENGTH-1];

// Fetch counters
reg [9:0] fetch_idx;

// Computation pipeline
reg [9:0] comp_idx;

// Pipeline registers for multiply-accumulate
reg signed [31:0] op_a, op_b;
reg pipe1_valid;
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
        6'h00: rdata_comb = {27'b0, ready_for_next, 3'b0, busy};  // CTRL/STATUS
        6'h01: rdata_comb = {22'b0, vec_length};    // LENGTH
        6'h02: rdata_comb = accumulator[31:0];      // RESULT_LO
        6'h03: rdata_comb = accumulator[63:32];     // RESULT_HI
        6'h04: rdata_comb = {8'b0, addr_a};         // ADDR_A
        6'h05: rdata_comb = {8'b0, addr_b};         // ADDR_B
        6'h06: rdata_comb = {8'b0, addr_a_next};    // ADDR_A_NEXT
        default: rdata_comb = 32'h0;
    endcase
end
assign reg_rdata = rdata_comb;

// Track if we've processed this access
reg access_done;

// Read from active buffer
wire signed [31:0] vec_a_read = active_buf ? vec_a1[comp_idx] : vec_a0[comp_idx];

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
                        accumulator <= 0;
                        fetch_idx <= 0;
                        comp_idx <= 0;
                        pipe1_valid <= 0;
                        pipe2_valid <= 0;

                        if (reg_wdata[2]) begin
                            // Preload B only
                            state <= STATE_FETCH_B;
                            cached_b_length <= vec_length;
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
                    end else begin
                        state <= STATE_COMPUTE;
                        comp_idx <= 0;
                        pipe1_valid <= 0;
                        pipe2_valid <= 0;
                    end
                end
            end

            STATE_COMPUTE: begin
                // Standard compute - no concurrent fetch
                if (comp_idx < vec_length) begin
                    op_a <= vec_a_read;
                    op_b <= vec_b[comp_idx];
                    pipe1_valid <= 1;
                    comp_idx <= comp_idx + 1;
                end else begin
                    pipe1_valid <= 0;
                end

                if (pipe1_valid) begin
                    product <= op_a * op_b;
                    pipe2_valid <= 1;
                end else begin
                    pipe2_valid <= 0;
                end

                if (pipe2_valid) begin
                    accumulator <= accumulator + product;
                end

                if (comp_idx >= vec_length && !pipe1_valid && !pipe2_valid) begin
                    state <= STATE_DONE;
                end
            end

            STATE_COMPUTE_FETCH: begin
                // Concurrent compute + prefetch into other buffer

                // Computation pipeline (same as STATE_COMPUTE)
                if (comp_idx < vec_length) begin
                    op_a <= vec_a_read;
                    op_b <= vec_b[comp_idx];
                    pipe1_valid <= 1;
                    comp_idx <= comp_idx + 1;
                end else begin
                    pipe1_valid <= 0;
                end

                if (pipe1_valid) begin
                    product <= op_a * op_b;
                    pipe2_valid <= 1;
                end else begin
                    pipe2_valid <= 0;
                end

                if (pipe2_valid) begin
                    accumulator <= accumulator + product;
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

            STATE_DONE: begin
                busy <= 0;
                state <= STATE_IDLE;
            end
        endcase
    end
end

endmodule
