//
// Dot Product Accelerator
// Uses DSP blocks for parallel multiply-accumulate operations
// Memory-mapped interface at 0x50000000
//
// Registers:
//   0x00: CONTROL   - Write 1 to start, read for status (bit 0 = busy)
//   0x04: LENGTH    - Vector length (number of elements)
//   0x08: RESULT_LO - Low 32 bits of accumulated result
//   0x0C: RESULT_HI - High 32 bits of accumulated result (for overflow)
//   0x10: VEC_A[0]  - First vector elements (write)
//   0x14: VEC_A[1]
//   ...
//   0x50: VEC_B[0]  - Second vector elements (write)
//   0x54: VEC_B[1]
//   ...
//
// Operation:
//   1. Write vector length to LENGTH register
//   2. Write vector A elements to VEC_A registers
//   3. Write vector B elements to VEC_B registers
//   4. Write 1 to CONTROL to start computation
//   5. Poll CONTROL until bit 0 is 0 (not busy)
//   6. Read RESULT_LO (and RESULT_HI if needed)
//
// Vectors are stored as signed 32-bit integers (Q16.16 fixed-point)
// Result is 64-bit to handle overflow from accumulation
//

`default_nettype none

module dot_product_accel #(
    parameter VEC_SIZE = 16    // Maximum vector length per batch
) (
    input wire clk,
    input wire reset_n,

    // Memory-mapped interface
    input wire         valid,
    input wire         write,
    input wire  [7:0]  addr,      // Byte address within accelerator
    input wire  [31:0] wdata,
    output wire [31:0] rdata,
    output wire        ready
);

// Internal registers
reg busy;
reg [5:0] vec_length;        // Up to 63 elements per batch
reg signed [63:0] accumulator;

// Vector storage - two vectors of VEC_SIZE elements
reg signed [31:0] vec_a [0:VEC_SIZE-1];
reg signed [31:0] vec_b [0:VEC_SIZE-1];

// Processing state
reg [5:0] proc_idx;
reg processing;

// Parallel multiply results (using DSP inference)
// Process 4 elements in parallel for better throughput
wire signed [63:0] prod0 = vec_a[proc_idx] * vec_b[proc_idx];
wire signed [63:0] prod1 = (proc_idx + 1 < vec_length) ? vec_a[proc_idx+1] * vec_b[proc_idx+1] : 64'sd0;
wire signed [63:0] prod2 = (proc_idx + 2 < vec_length) ? vec_a[proc_idx+2] * vec_b[proc_idx+2] : 64'sd0;
wire signed [63:0] prod3 = (proc_idx + 3 < vec_length) ? vec_a[proc_idx+3] * vec_b[proc_idx+3] : 64'sd0;

// Sum of 4 products (pipelined addition)
wire signed [63:0] sum_01 = prod0 + prod1;
wire signed [63:0] sum_23 = prod2 + prod3;
wire signed [63:0] sum_all = sum_01 + sum_23;

// Address decoding
wire [7:0] word_addr = {addr[7:2], 2'b00};  // Word-aligned address

// Combinational ready - always respond immediately
assign ready = valid;

// Combinational read data
reg [31:0] rdata_comb;
always @(*) begin
    case (word_addr)
        8'h00: rdata_comb = {31'b0, busy};         // CONTROL/STATUS
        8'h04: rdata_comb = {26'b0, vec_length};   // LENGTH
        8'h08: rdata_comb = accumulator[31:0];    // RESULT_LO
        8'h0C: rdata_comb = accumulator[63:32];   // RESULT_HI
        default: rdata_comb = 32'h0;
    endcase
end
assign rdata = rdata_comb;

// Track if we've processed this access (to handle writes only once per bus cycle)
reg access_done;

always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        busy <= 0;
        vec_length <= 0;
        accumulator <= 0;
        proc_idx <= 0;
        processing <= 0;
        access_done <= 0;
    end else begin
        // Clear access_done when valid goes low
        if (!valid) begin
            access_done <= 0;
        end

        // Handle write access (only once per bus cycle)
        if (valid && write && !access_done) begin
            access_done <= 1;
            case (word_addr)
                8'h00: begin  // CONTROL
                    if (wdata[0] && !busy) begin
                        // Start computation
                        busy <= 1;
                        processing <= 1;
                        proc_idx <= 0;
                        accumulator <= 0;
                    end
                end
                8'h04: vec_length <= wdata[5:0];  // LENGTH
                // VEC_A: 0x10-0x4F (16 words)
                8'h10: vec_a[0] <= wdata;
                8'h14: vec_a[1] <= wdata;
                8'h18: vec_a[2] <= wdata;
                8'h1C: vec_a[3] <= wdata;
                8'h20: vec_a[4] <= wdata;
                8'h24: vec_a[5] <= wdata;
                8'h28: vec_a[6] <= wdata;
                8'h2C: vec_a[7] <= wdata;
                8'h30: vec_a[8] <= wdata;
                8'h34: vec_a[9] <= wdata;
                8'h38: vec_a[10] <= wdata;
                8'h3C: vec_a[11] <= wdata;
                8'h40: vec_a[12] <= wdata;
                8'h44: vec_a[13] <= wdata;
                8'h48: vec_a[14] <= wdata;
                8'h4C: vec_a[15] <= wdata;
                // VEC_B: 0x50-0x8F (16 words)
                8'h50: vec_b[0] <= wdata;
                8'h54: vec_b[1] <= wdata;
                8'h58: vec_b[2] <= wdata;
                8'h5C: vec_b[3] <= wdata;
                8'h60: vec_b[4] <= wdata;
                8'h64: vec_b[5] <= wdata;
                8'h68: vec_b[6] <= wdata;
                8'h6C: vec_b[7] <= wdata;
                8'h70: vec_b[8] <= wdata;
                8'h74: vec_b[9] <= wdata;
                8'h78: vec_b[10] <= wdata;
                8'h7C: vec_b[11] <= wdata;
                8'h80: vec_b[12] <= wdata;
                8'h84: vec_b[13] <= wdata;
                8'h88: vec_b[14] <= wdata;
                8'h8C: vec_b[15] <= wdata;
                default: ;  // Ignore unknown addresses
            endcase
        end

        // Processing state machine
        if (processing) begin
            if (proc_idx < vec_length) begin
                // Accumulate 4 products per cycle
                accumulator <= accumulator + sum_all;

                // Advance index by 4
                if (proc_idx + 4 >= vec_length) begin
                    processing <= 0;
                    busy <= 0;
                end else begin
                    proc_idx <= proc_idx + 6'd4;
                end
            end else begin
                processing <= 0;
                busy <= 0;
            end
        end
    end
end

endmodule
