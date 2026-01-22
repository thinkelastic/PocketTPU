//
// 8-Element DSP Dot Product Accelerator
// Dedicated hardware for head_size=8 attention computation
// Memory-mapped interface at 0x51000000
//
// Registers:
//   0x00: A_DATA    - Write A elements (auto-increment index 0-7)
//   0x04: B_DATA    - Write B elements (auto-increment index 0-7), triggers compute on 8th write
//   0x08: CTRL      - [0]=busy (read), [1]=reset_idx (write)
//   0x0C: RESULT_LO - Low 32 bits of result
//   0x10: RESULT_HI - High 32 bits of result
//
// Usage:
//   1. Write A[0-7] to A_DATA (8 writes)
//   2. Write B[0-7] to B_DATA (8 writes) - auto-triggers compute on 8th write
//   3. Poll CTRL until not busy (~5 cycles)
//   4. Read RESULT_LO/HI
//   5. For next dot product with same A: repeat from step 2
//
// Uses 8 DSP blocks for parallel 32x32 signed multiply
// 3-stage pipelined adder tree for minimal latency
//

`default_nettype none

module dot8_accel (
    input wire clk,
    input wire reset_n,

    // CPU register interface
    input wire         reg_valid,
    input wire         reg_write,
    input wire  [7:0]  reg_addr,
    input wire  [31:0] reg_wdata,
    output wire [31:0] reg_rdata,
    output wire        reg_ready
);

// A and B vectors (Q16.16 signed)
reg signed [31:0] vec_a [0:7];
reg signed [31:0] vec_b [0:7];

// Write indices
reg [2:0] a_idx;
reg [2:0] b_idx;

// Control
reg busy;
reg [2:0] pipe_stage;

// Pipeline registers for multiply results (64-bit each)
reg signed [63:0] prod [0:7];

// Adder tree stage 1: 8 -> 4
reg signed [63:0] sum_s1 [0:3];

// Adder tree stage 2: 4 -> 2
reg signed [63:0] sum_s2 [0:1];

// Final result
reg signed [63:0] result;

// Always ready for register access
assign reg_ready = reg_valid;

// Register read mux
reg [31:0] rdata_comb;
always @(*) begin
    case (reg_addr[7:2])
        6'h00: rdata_comb = 32'h0;                    // A_DATA (write-only)
        6'h01: rdata_comb = 32'h0;                    // B_DATA (write-only)
        6'h02: rdata_comb = {31'b0, busy};            // CTRL/STATUS
        6'h03: rdata_comb = result[31:0];             // RESULT_LO
        6'h04: rdata_comb = result[63:32];            // RESULT_HI
        default: rdata_comb = 32'h0;
    endcase
end
assign reg_rdata = rdata_comb;

// Track if we've processed this access
reg access_done;

// Main logic
always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        a_idx <= 0;
        b_idx <= 0;
        busy <= 0;
        pipe_stage <= 0;
        access_done <= 0;
        result <= 0;
    end else begin
        // Clear access_done when valid goes low
        if (!reg_valid) begin
            access_done <= 0;
        end

        // Handle register writes
        if (reg_valid && reg_write && !access_done) begin
            access_done <= 1;
            case (reg_addr[7:2])
                6'h00: begin  // A_DATA
                    vec_a[a_idx] <= reg_wdata;
                    a_idx <= a_idx + 1;
                end
                6'h01: begin  // B_DATA
                    vec_b[b_idx] <= reg_wdata;
                    if (b_idx == 7) begin
                        // All B elements written, start compute
                        busy <= 1;
                        pipe_stage <= 1;
                        b_idx <= 0;
                    end else begin
                        b_idx <= b_idx + 1;
                    end
                end
                6'h02: begin  // CTRL
                    if (reg_wdata[1]) begin
                        // Reset indices
                        a_idx <= 0;
                        b_idx <= 0;
                    end
                end
                default: ;
            endcase
        end

        // Computation pipeline
        if (busy) begin
            case (pipe_stage)
                3'd1: begin
                    // Stage 1: 8 parallel multiplies (DSP inference)
                    prod[0] <= vec_a[0] * vec_b[0];
                    prod[1] <= vec_a[1] * vec_b[1];
                    prod[2] <= vec_a[2] * vec_b[2];
                    prod[3] <= vec_a[3] * vec_b[3];
                    prod[4] <= vec_a[4] * vec_b[4];
                    prod[5] <= vec_a[5] * vec_b[5];
                    prod[6] <= vec_a[6] * vec_b[6];
                    prod[7] <= vec_a[7] * vec_b[7];
                    pipe_stage <= 2;
                end
                3'd2: begin
                    // Stage 2: First level of adder tree (8 -> 4)
                    sum_s1[0] <= prod[0] + prod[1];
                    sum_s1[1] <= prod[2] + prod[3];
                    sum_s1[2] <= prod[4] + prod[5];
                    sum_s1[3] <= prod[6] + prod[7];
                    pipe_stage <= 3;
                end
                3'd3: begin
                    // Stage 3: Second level of adder tree (4 -> 2)
                    sum_s2[0] <= sum_s1[0] + sum_s1[1];
                    sum_s2[1] <= sum_s1[2] + sum_s1[3];
                    pipe_stage <= 4;
                end
                3'd4: begin
                    // Stage 4: Final sum (2 -> 1)
                    result <= sum_s2[0] + sum_s2[1];
                    pipe_stage <= 0;
                    busy <= 0;
                end
                default: begin
                    pipe_stage <= 0;
                    busy <= 0;
                end
            endcase
        end
    end
end

endmodule
