//
// User core top-level
//
// Instantiated by the real top-level: apf_top
//

`default_nettype none

module core_top (

//
// physical connections
//

///////////////////////////////////////////////////
// clock inputs 74.25mhz. not phase aligned, so treat these domains as asynchronous

input   wire            clk_74a, // mainclk1
input   wire            clk_74b, // mainclk1 

///////////////////////////////////////////////////
// cartridge interface
// switches between 3.3v and 5v mechanically
// output enable for multibit translators controlled by pic32

// GBA AD[15:8]
inout   wire    [7:0]   cart_tran_bank2,
output  wire            cart_tran_bank2_dir,

// GBA AD[7:0]
inout   wire    [7:0]   cart_tran_bank3,
output  wire            cart_tran_bank3_dir,

// GBA A[23:16]
inout   wire    [7:0]   cart_tran_bank1,
output  wire            cart_tran_bank1_dir,

// GBA [7] PHI#
// GBA [6] WR#
// GBA [5] RD#
// GBA [4] CS1#/CS#
//     [3:0] unwired
inout   wire    [7:4]   cart_tran_bank0,
output  wire            cart_tran_bank0_dir,

// GBA CS2#/RES#
inout   wire            cart_tran_pin30,
output  wire            cart_tran_pin30_dir,
// when GBC cart is inserted, this signal when low or weak will pull GBC /RES low with a special circuit
// the goal is that when unconfigured, the FPGA weak pullups won't interfere.
// thus, if GBC cart is inserted, FPGA must drive this high in order to let the level translators
// and general IO drive this pin.
output  wire            cart_pin30_pwroff_reset,

// GBA IRQ/DRQ
inout   wire            cart_tran_pin31,
output  wire            cart_tran_pin31_dir,

// infrared
input   wire            port_ir_rx,
output  wire            port_ir_tx,
output  wire            port_ir_rx_disable, 

// GBA link port
inout   wire            port_tran_si,
output  wire            port_tran_si_dir,
inout   wire            port_tran_so,
output  wire            port_tran_so_dir,
inout   wire            port_tran_sck,
output  wire            port_tran_sck_dir,
inout   wire            port_tran_sd,
output  wire            port_tran_sd_dir,
 
///////////////////////////////////////////////////
// cellular psram 0 and 1, two chips (64mbit x2 dual die per chip)

output  wire    [21:16] cram0_a,
inout   wire    [15:0]  cram0_dq,
input   wire            cram0_wait,
output  wire            cram0_clk,
output  wire            cram0_adv_n,
output  wire            cram0_cre,
output  wire            cram0_ce0_n,
output  wire            cram0_ce1_n,
output  wire            cram0_oe_n,
output  wire            cram0_we_n,
output  wire            cram0_ub_n,
output  wire            cram0_lb_n,

output  wire    [21:16] cram1_a,
inout   wire    [15:0]  cram1_dq,
input   wire            cram1_wait,
output  wire            cram1_clk,
output  wire            cram1_adv_n,
output  wire            cram1_cre,
output  wire            cram1_ce0_n,
output  wire            cram1_ce1_n,
output  wire            cram1_oe_n,
output  wire            cram1_we_n,
output  wire            cram1_ub_n,
output  wire            cram1_lb_n,

///////////////////////////////////////////////////
// sdram, 512mbit 16bit

output  wire    [12:0]  dram_a,
output  wire    [1:0]   dram_ba,
inout   wire    [15:0]  dram_dq,
output  wire    [1:0]   dram_dqm,
output  wire            dram_clk,
output  wire            dram_cke,
output  wire            dram_ras_n,
output  wire            dram_cas_n,
output  wire            dram_we_n,

///////////////////////////////////////////////////
// sram, 1mbit 16bit

output  wire    [16:0]  sram_a,
inout   wire    [15:0]  sram_dq,
output  wire            sram_oe_n,
output  wire            sram_we_n,
output  wire            sram_ub_n,
output  wire            sram_lb_n,

///////////////////////////////////////////////////
// vblank driven by dock for sync in a certain mode

input   wire            vblank,

///////////////////////////////////////////////////
// i/o to 6515D breakout usb uart

output  wire            dbg_tx,
input   wire            dbg_rx,

///////////////////////////////////////////////////
// i/o pads near jtag connector user can solder to

output  wire            user1,
input   wire            user2,

///////////////////////////////////////////////////
// RFU internal i2c bus 

inout   wire            aux_sda,
output  wire            aux_scl,

///////////////////////////////////////////////////
// RFU, do not use
output  wire            vpll_feed,


//
// logical connections
//

///////////////////////////////////////////////////
// video, audio output to scaler
output  wire    [23:0]  video_rgb,
output  wire            video_rgb_clock,
output  wire            video_rgb_clock_90,
output  wire            video_de,
output  wire            video_skip,
output  wire            video_vs,
output  wire            video_hs,
    
output  wire            audio_mclk,
input   wire            audio_adc,
output  wire            audio_dac,
output  wire            audio_lrck,

///////////////////////////////////////////////////
// bridge bus connection
// synchronous to clk_74a
output  wire            bridge_endian_little,
input   wire    [31:0]  bridge_addr,
input   wire            bridge_rd,
output  reg     [31:0]  bridge_rd_data,
input   wire            bridge_wr,
input   wire    [31:0]  bridge_wr_data,

///////////////////////////////////////////////////
// controller data
// 
// key bitmap:
//   [0]    dpad_up
//   [1]    dpad_down
//   [2]    dpad_left
//   [3]    dpad_right
//   [4]    face_a
//   [5]    face_b
//   [6]    face_x
//   [7]    face_y
//   [8]    trig_l1
//   [9]    trig_r1
//   [10]   trig_l2
//   [11]   trig_r2
//   [12]   trig_l3
//   [13]   trig_r3
//   [14]   face_select
//   [15]   face_start
//   [31:28] type
// joy values - unsigned
//   [ 7: 0] lstick_x
//   [15: 8] lstick_y
//   [23:16] rstick_x
//   [31:24] rstick_y
// trigger values - unsigned
//   [ 7: 0] ltrig
//   [15: 8] rtrig
//
input   wire    [31:0]  cont1_key,
input   wire    [31:0]  cont2_key,
input   wire    [31:0]  cont3_key,
input   wire    [31:0]  cont4_key,
input   wire    [31:0]  cont1_joy,
input   wire    [31:0]  cont2_joy,
input   wire    [31:0]  cont3_joy,
input   wire    [31:0]  cont4_joy,
input   wire    [15:0]  cont1_trig,
input   wire    [15:0]  cont2_trig,
input   wire    [15:0]  cont3_trig,
input   wire    [15:0]  cont4_trig
    
);

// not using the IR port, so turn off both the LED, and
// disable the receive circuit to save power
assign port_ir_tx = 0;
assign port_ir_rx_disable = 1;

// bridge endianness
// Set to 1 for little-endian (RISC-V native format)
assign bridge_endian_little = 1;

// cart is unused, so set all level translators accordingly
// directions are 0:IN, 1:OUT
assign cart_tran_bank3 = 8'hzz;
assign cart_tran_bank3_dir = 1'b0;
assign cart_tran_bank2 = 8'hzz;
assign cart_tran_bank2_dir = 1'b0;
assign cart_tran_bank1 = 8'hzz;
assign cart_tran_bank1_dir = 1'b0;
assign cart_tran_bank0 = 4'hf;
assign cart_tran_bank0_dir = 1'b1;
assign cart_tran_pin30 = 1'b0;      // reset or cs2, we let the hw control it by itself
assign cart_tran_pin30_dir = 1'bz;
assign cart_pin30_pwroff_reset = 1'b0;  // hardware can control this
assign cart_tran_pin31 = 1'bz;      // input
assign cart_tran_pin31_dir = 1'b0;  // input

// link port is unused, set to input only to be safe
// each bit may be bidirectional in some applications
assign port_tran_so = 1'bz;
assign port_tran_so_dir = 1'b0;     // SO is output only
assign port_tran_si = 1'bz;
assign port_tran_si_dir = 1'b0;     // SI is input only
assign port_tran_sck = 1'bz;
assign port_tran_sck_dir = 1'b0;    // clock direction can change
assign port_tran_sd = 1'bz;
assign port_tran_sd_dir = 1'b0;     // SD is input and not used

// PSRAM controller for CRAM0+CRAM1 (16MB total accessible via CPU)
// Memory map: 0x30000000 - 0x30FFFFFF (16MB byte addressable, 4M words)
// Address bit 23 selects chip: CRAM0 (0x30000000-0x307FFFFF) or CRAM1 (0x30800000-0x30FFFFFF)
wire        cpu_psram_rd;
wire        cpu_psram_wr;
wire [21:0] cpu_psram_addr;
wire [31:0] cpu_psram_wdata;
wire [31:0] cpu_psram_rdata;
wire        cpu_psram_busy;

psram_controller #(
    .CLOCK_SPEED(133.333333)  // 133 MHz CPU clock
) psram0 (
    .clk(clk_ram_controller),
    .reset_n(reset_n),

    // CPU interface
    .word_rd(cpu_psram_rd),
    .word_wr(cpu_psram_wr),
    .word_addr(cpu_psram_addr),
    .word_data(cpu_psram_wdata),
    .word_q(cpu_psram_rdata),
    .word_busy(cpu_psram_busy),

    // CRAM0 physical signals
    .cram_a(cram0_a),
    .cram_dq(cram0_dq),
    .cram_wait(cram0_wait),
    .cram_clk(cram0_clk),
    .cram_adv_n(cram0_adv_n),
    .cram_cre(cram0_cre),
    .cram_ce0_n(cram0_ce0_n),
    .cram_ce1_n(cram0_ce1_n),
    .cram_oe_n(cram0_oe_n),
    .cram_we_n(cram0_we_n),
    .cram_ub_n(cram0_ub_n),
    .cram_lb_n(cram0_lb_n)
);

// Tie off CRAM1 (not used)
assign cram1_a = 'h0;
assign cram1_dq = {16{1'bZ}};
assign cram1_clk = 0;
assign cram1_adv_n = 1;
assign cram1_cre = 0;
assign cram1_ce0_n = 1;
assign cram1_ce1_n = 1;
assign cram1_oe_n = 1;
assign cram1_we_n = 1;
assign cram1_ub_n = 1;
assign cram1_lb_n = 1;

// SDRAM word interface signals (directly matching io_sdram interface)
reg             ram1_word_rd;
reg             ram1_word_wr;
reg     [23:0]  ram1_word_addr;
reg     [31:0]  ram1_word_data;
wire    [31:0]  ram1_word_q;
wire            ram1_word_busy;
wire            ram1_word_q_valid;

// DMA accelerator burst interface signals
wire            dma_burst_rd;
wire    [24:0]  dma_burst_addr;
wire    [10:0]  dma_burst_len;
wire            dma_burst_32bit;
wire    [31:0]  dma_burst_data;
wire            dma_burst_data_valid;
wire            dma_burst_data_done;

// CPU runs at same clock as SDRAM controller (133 MHz) - no CDC needed!
// Direct connection to ram1_word_q since same clock domain

// CPU to SDRAM interface (directly exposed for cpu_system to use)
wire        cpu_sdram_rd;
wire        cpu_sdram_wr;
wire [23:0] cpu_sdram_addr;
wire [31:0] cpu_sdram_wdata;
wire [31:0] cpu_sdram_rdata;
wire        cpu_sdram_busy;

assign sram_a = 'h0;
assign sram_dq = {16{1'bZ}};
assign sram_oe_n  = 1;
assign sram_we_n  = 1;
assign sram_ub_n  = 1;
assign sram_lb_n  = 1;

assign dbg_tx = 1'bZ;
assign user1 = 1'bZ;
assign aux_scl = 1'bZ;
assign vpll_feed = 1'bZ;


// Bridge read data mux
reg     [31:0]  ram1_bridge_rd_data;

always @(*) begin
    casex(bridge_addr)
    default: begin
        bridge_rd_data <= 0;
    end
    32'b000000xx_xxxxxxxx_xxxxxxxx_xxxxxxxx: begin
        // SDRAM mapped at 0x00000000 - 0x03FFFFFF (64MB)
        bridge_rd_data <= ram1_bridge_rd_data;
    end
    32'hF8xxxxxx: begin
        bridge_rd_data <= cmd_bridge_rd_data;
    end
    endcase
end

// Synchronize bridge signals from clk_74a to clk_ram_controller
reg [31:0] bridge_addr_captured;
reg [31:0] bridge_wr_data_captured;
reg bridge_sdram_wr, bridge_sdram_rd;
reg [31:0] bridge_addr_ram_clk;
reg [31:0] bridge_wr_data_ram_clk;
reg bridge_wr_done, bridge_rd_done;  // Feedback to 74a domain
reg bridge_wr_done_sync1, bridge_wr_done_sync2;
reg bridge_rd_done_sync1, bridge_rd_done_sync2;

// Capture bridge signals in clk_74a domain
// IMPORTANT: Hold the captured values until the RAM controller has processed them
// This prevents race conditions when multiple writes come quickly
always @(posedge clk_74a) begin
    // Synchronize done signals back from RAM controller clock
    bridge_wr_done_sync1 <= bridge_wr_done;
    bridge_wr_done_sync2 <= bridge_wr_done_sync1;
    bridge_rd_done_sync1 <= bridge_rd_done;
    bridge_rd_done_sync2 <= bridge_rd_done_sync1;

    // Clear the request when done is seen
    if (bridge_wr_done_sync2) bridge_sdram_wr <= 0;
    if (bridge_rd_done_sync2) bridge_sdram_rd <= 0;

    // Only accept new requests when not busy
    if (!bridge_sdram_wr && bridge_wr) begin
        casex(bridge_addr[31:24])
        8'b000000xx: begin
            bridge_sdram_wr <= 1;
            bridge_addr_captured <= bridge_addr;
            bridge_wr_data_captured <= bridge_wr_data;
        end
        endcase
    end
    if (!bridge_sdram_rd && bridge_rd) begin
        casex(bridge_addr[31:24])
        8'b000000xx: begin
            bridge_sdram_rd <= 1;
            bridge_addr_captured <= bridge_addr;
            ram1_bridge_rd_data <= ram1_word_q;  // Capture for bridge read (pipelined)
        end
        endcase
    end
end

// 4-stage synchronizer for control signals + data capture
// sync1 -> sync2 -> sync3 -> sync4 gives data plenty of time to stabilize
reg bridge_wr_sync1, bridge_wr_sync2, bridge_wr_sync3, bridge_wr_sync4;
reg bridge_rd_sync1, bridge_rd_sync2, bridge_rd_sync3, bridge_rd_sync4;

always @(posedge clk_ram_controller) begin
    // 4-stage sync for control signals
    bridge_wr_sync1 <= bridge_sdram_wr;
    bridge_wr_sync2 <= bridge_wr_sync1;
    bridge_wr_sync3 <= bridge_wr_sync2;
    bridge_wr_sync4 <= bridge_wr_sync3;
    bridge_rd_sync1 <= bridge_sdram_rd;
    bridge_rd_sync2 <= bridge_rd_sync1;
    bridge_rd_sync3 <= bridge_rd_sync2;
    bridge_rd_sync4 <= bridge_rd_sync3;

    // Capture address/data on sync3 rising edge (when sync3 goes high but sync4 still low)
    // By this point, data has been stable for 3 cycles in the fast clock domain
    if (bridge_wr_sync3 && !bridge_wr_sync4) begin
        bridge_addr_ram_clk <= bridge_addr_captured;
        bridge_wr_data_ram_clk <= bridge_wr_data_captured;
    end
    if (bridge_rd_sync3 && !bridge_rd_sync4) begin
        bridge_addr_ram_clk <= bridge_addr_captured;
    end

    // Signal done on sync4 rising edge - AFTER data has been captured and used
    if (bridge_wr_sync4 && !bridge_wr_done) begin
        bridge_wr_done <= 1;
    end
    if (bridge_rd_sync4 && !bridge_rd_done) begin
        bridge_rd_done <= 1;
    end

    // Clear done when sync goes low
    if (!bridge_wr_sync1) bridge_wr_done <= 0;
    if (!bridge_rd_sync1) bridge_rd_done <= 0;
end

// Bridge is active from sync3 through done (when we're processing)
wire bridge_wr_active = bridge_wr_sync3 | bridge_wr_sync4 | bridge_wr_done;
wire bridge_rd_active = bridge_rd_sync3 | bridge_rd_sync4 | bridge_rd_done;

// SDRAM access arbiter - runs at SDRAM controller clock (133 MHz)
// Bridge has priority, CPU runs when bridge is idle
always @(posedge clk_ram_controller) begin
    ram1_word_rd <= 0;
    ram1_word_wr <= 0;

    // Issue SDRAM command on sync4 rising edge
    // Data was captured on sync3->sync4 edge, so it's now stable in bridge_*_ram_clk
    if (bridge_wr_sync4 && !bridge_wr_done) begin
        // Bridge write - data is now stable in bridge_addr_ram_clk
        ram1_word_wr <= 1;
        ram1_word_addr <= bridge_addr_ram_clk[25:2];
        ram1_word_data <= bridge_wr_data_ram_clk;
    end else if (bridge_rd_sync4 && !bridge_rd_done) begin
        // Bridge read - data is now stable in bridge_addr_ram_clk
        ram1_word_rd <= 1;
        ram1_word_addr <= bridge_addr_ram_clk[25:2];
    end else if (!bridge_wr_active && !bridge_rd_active) begin
        // CPU SDRAM access - same clock domain, direct connection
        if (cpu_sdram_rd) begin
            ram1_word_rd <= 1;
            ram1_word_addr <= cpu_sdram_addr;
        end
        if (cpu_sdram_wr) begin
            ram1_word_wr <= 1;
            ram1_word_addr <= cpu_sdram_addr;
            ram1_word_data <= cpu_sdram_wdata;
        end
    end
end

// CPU reads SDRAM data - direct connection (same clock domain)
assign cpu_sdram_rdata = ram1_word_q;
assign cpu_sdram_busy = ram1_word_busy;


//
// host/target command handler
//
    wire            reset_n_apf;            // driven by host commands from APF bridge
    wire    [31:0]  cmd_bridge_rd_data;

    // JTAG Debug bypass: Force reset_n after short timeout for development
    // NOTE: Does not work because APF video scaler requires bridge communication
    // `define JTAG_DEBUG_BYPASS  // Uncomment for JTAG debugging (doesn't work fully)
    `ifdef JTAG_DEBUG_BYPASS
    reg [24:0] jtag_reset_counter;
    reg        jtag_reset_force;
    always @(posedge clk_74a) begin
        if (!pll_core_locked_s) begin
            jtag_reset_counter <= 0;
            jtag_reset_force <= 0;
        end else if (!reset_n_apf && !jtag_reset_force) begin
            // Wait ~0.5 seconds for APF to deassert reset (74.25 MHz * 0.5 = 37.125M)
            if (jtag_reset_counter < 25'd37125000) begin
                jtag_reset_counter <= jtag_reset_counter + 1;
            end else begin
                jtag_reset_force <= 1;
            end
        end
    end
    wire reset_n = reset_n_apf | jtag_reset_force;
    `else
    wire reset_n = reset_n_apf;
    `endif

// bridge host commands
// synchronous to clk_74a
    wire            status_boot_done = pll_core_locked_s;
    wire            status_setup_done = pll_core_locked_s; // rising edge triggers a target command
    wire            status_running = reset_n; // we are running as soon as reset_n goes high

    wire            dataslot_requestread;
    wire    [15:0]  dataslot_requestread_id;
    wire            dataslot_requestread_ack = 1;
    wire            dataslot_requestread_ok = 1;

    wire            dataslot_requestwrite;
    wire    [15:0]  dataslot_requestwrite_id;
    wire    [31:0]  dataslot_requestwrite_size;
    wire            dataslot_requestwrite_ack = 1;
    wire            dataslot_requestwrite_ok = 1;

    wire            dataslot_update;
    wire    [15:0]  dataslot_update_id;
    wire    [31:0]  dataslot_update_size;
    
    wire            dataslot_allcomplete_apf;

    // JTAG Debug bypass: Force dataslot_allcomplete after timeout for development
    // When programming via JTAG, APF doesn't run so we force the signal after 2 seconds
    `ifdef JTAG_DEBUG_BYPASS
    reg [27:0] jtag_timeout_counter;
    reg        jtag_dataslot_force;
    always @(posedge clk_74a or negedge reset_n) begin
        if (~reset_n) begin
            jtag_timeout_counter <= 0;
            jtag_dataslot_force <= 0;
        end else begin
            if (!dataslot_allcomplete_apf && !jtag_dataslot_force) begin
                // Count for ~2 seconds at 74.25 MHz (148.5M cycles)
                if (jtag_timeout_counter < 28'd148500000) begin
                    jtag_timeout_counter <= jtag_timeout_counter + 1;
                end else begin
                    jtag_dataslot_force <= 1;
                end
            end
        end
    end
    wire dataslot_allcomplete = dataslot_allcomplete_apf | jtag_dataslot_force;
    `else
    wire dataslot_allcomplete = dataslot_allcomplete_apf;
    `endif

    wire     [31:0] rtc_epoch_seconds;
    wire     [31:0] rtc_date_bcd;
    wire     [31:0] rtc_time_bcd;
    wire            rtc_valid;

    wire            savestate_supported;
    wire    [31:0]  savestate_addr;
    wire    [31:0]  savestate_size;
    wire    [31:0]  savestate_maxloadsize;

    wire            savestate_start;
    wire            savestate_start_ack;
    wire            savestate_start_busy;
    wire            savestate_start_ok;
    wire            savestate_start_err;

    wire            savestate_load;
    wire            savestate_load_ack;
    wire            savestate_load_busy;
    wire            savestate_load_ok;
    wire            savestate_load_err;
    
    wire            osnotify_inmenu;

// bridge target commands
// synchronous to clk_74a
// Not used - APF handles data slot loading automatically via addresses in data.json

    wire            target_dataslot_read       = 0;
    wire            target_dataslot_write      = 0;
    wire            target_dataslot_getfile    = 0;
    wire            target_dataslot_openfile   = 0;

    wire            target_dataslot_ack;
    wire            target_dataslot_done;
    wire    [2:0]   target_dataslot_err;

    wire    [15:0]  target_dataslot_id         = 0;
    wire    [31:0]  target_dataslot_slotoffset = 0;
    wire    [31:0]  target_dataslot_bridgeaddr = 0;
    wire    [31:0]  target_dataslot_length     = 0;
    
    wire    [31:0]  target_buffer_param_struct; // to be mapped/implemented when using some Target commands
    wire    [31:0]  target_buffer_resp_struct;  // to be mapped/implemented when using some Target commands
    
// bridge data slot access
// synchronous to clk_74a
// Not used - APF handles data slot loading automatically

    wire    [9:0]   datatable_addr = 0;
    wire    [31:0]  datatable_q;
    wire            datatable_wren = 0;
    wire    [31:0]  datatable_data = 0;

core_bridge_cmd icb (

    .clk                ( clk_74a ),
    .reset_n            ( reset_n_apf ),

    .bridge_endian_little   ( bridge_endian_little ),
    .bridge_addr            ( bridge_addr ),
    .bridge_rd              ( bridge_rd ),
    .bridge_rd_data         ( cmd_bridge_rd_data ),
    .bridge_wr              ( bridge_wr ),
    .bridge_wr_data         ( bridge_wr_data ),
    
    .status_boot_done       ( status_boot_done ),
    .status_setup_done      ( status_setup_done ),
    .status_running         ( status_running ),

    .dataslot_requestread       ( dataslot_requestread ),
    .dataslot_requestread_id    ( dataslot_requestread_id ),
    .dataslot_requestread_ack   ( dataslot_requestread_ack ),
    .dataslot_requestread_ok    ( dataslot_requestread_ok ),

    .dataslot_requestwrite      ( dataslot_requestwrite ),
    .dataslot_requestwrite_id   ( dataslot_requestwrite_id ),
    .dataslot_requestwrite_size ( dataslot_requestwrite_size ),
    .dataslot_requestwrite_ack  ( dataslot_requestwrite_ack ),
    .dataslot_requestwrite_ok   ( dataslot_requestwrite_ok ),

    .dataslot_update            ( dataslot_update ),
    .dataslot_update_id         ( dataslot_update_id ),
    .dataslot_update_size       ( dataslot_update_size ),
    
    .dataslot_allcomplete   ( dataslot_allcomplete_apf ),

    .rtc_epoch_seconds      ( rtc_epoch_seconds ),
    .rtc_date_bcd           ( rtc_date_bcd ),
    .rtc_time_bcd           ( rtc_time_bcd ),
    .rtc_valid              ( rtc_valid ),
    
    .savestate_supported    ( savestate_supported ),
    .savestate_addr         ( savestate_addr ),
    .savestate_size         ( savestate_size ),
    .savestate_maxloadsize  ( savestate_maxloadsize ),

    .savestate_start        ( savestate_start ),
    .savestate_start_ack    ( savestate_start_ack ),
    .savestate_start_busy   ( savestate_start_busy ),
    .savestate_start_ok     ( savestate_start_ok ),
    .savestate_start_err    ( savestate_start_err ),

    .savestate_load         ( savestate_load ),
    .savestate_load_ack     ( savestate_load_ack ),
    .savestate_load_busy    ( savestate_load_busy ),
    .savestate_load_ok      ( savestate_load_ok ),
    .savestate_load_err     ( savestate_load_err ),

    .osnotify_inmenu        ( osnotify_inmenu ),
    
    .target_dataslot_read       ( target_dataslot_read ),
    .target_dataslot_write      ( target_dataslot_write ),
    .target_dataslot_getfile    ( target_dataslot_getfile ),
    .target_dataslot_openfile   ( target_dataslot_openfile ),
    
    .target_dataslot_ack        ( target_dataslot_ack ),
    .target_dataslot_done       ( target_dataslot_done ),
    .target_dataslot_err        ( target_dataslot_err ),

    .target_dataslot_id         ( target_dataslot_id ),
    .target_dataslot_slotoffset ( target_dataslot_slotoffset ),
    .target_dataslot_bridgeaddr ( target_dataslot_bridgeaddr ),
    .target_dataslot_length     ( target_dataslot_length ),

    .target_buffer_param_struct ( target_buffer_param_struct ),
    .target_buffer_resp_struct  ( target_buffer_resp_struct ),
    
    .datatable_addr         ( datatable_addr ),
    .datatable_wren         ( datatable_wren ),
    .datatable_data         ( datatable_data ),
    .datatable_q            ( datatable_q )

);



////////////////////////////////////////////////////////////////////////////////////////



// video generation
// Using 12.288 MHz pixel clock
//
// For 60 Hz: 12,288,000 / 60 = 204,800 pixels per frame
// Using 320x240 visible with blanking:
// - 400 total horizontal (320 visible + 80 blanking)
// - 262 total vertical (240 visible + 22 blanking)
// - 400 * 262 = 104,800 -> ~117 Hz (too fast)
//
// Let's try 320x200 with more blanking for ~60Hz:
// - 408 total horizontal (320 + 88)
// - 502 total vertical (200 + 302) -> way too much blanking
//
// Better approach: 320x240 @ ~48Hz (close enough for scaler)
// - 400 H total, 256 V total = 102,400 -> 120 Hz
// - 400 H total, 512 V total = 204,800 -> 60 Hz exactly!
//
// 320x240 visible, 400x512 total = 60 Hz at 12.288 MHz

assign video_rgb_clock = clk_core_12288;
assign video_rgb_clock_90 = clk_core_12288_90deg;
assign video_rgb = vidout_rgb;
assign video_de = vidout_de;
assign video_skip = vidout_skip;
assign video_vs = vidout_vs;
assign video_hs = vidout_hs;

    // 320x240 @ 60Hz with 12.288 MHz pixel clock
    // Total: 400 x 512 = 204,800 pixels/frame
    // 12,288,000 / 204,800 = 60 Hz
    localparam  VID_V_BPORCH = 'd16;
    localparam  VID_V_ACTIVE = 'd240;
    localparam  VID_V_TOTAL = 'd512;
    localparam  VID_H_BPORCH = 'd40;
    localparam  VID_H_ACTIVE = 'd320;
    localparam  VID_H_TOTAL = 'd400;

    reg [15:0]  frame_count;
    
    reg [9:0]   x_count;
    reg [9:0]   y_count;
    
    wire [9:0]  visible_x = x_count - VID_H_BPORCH;
    wire [9:0]  visible_y = y_count - VID_V_BPORCH;

    reg [23:0]  vidout_rgb;
    reg         vidout_de, vidout_de_1;
    reg         vidout_skip;
    reg         vidout_vs;
    reg         vidout_hs, vidout_hs_1;
    
    // CPU to terminal interface signals
    wire        term_mem_valid;
    wire [31:0] term_mem_addr;
    wire [31:0] term_mem_wdata;
    wire [3:0]  term_mem_wstrb;
    wire [31:0] term_mem_rdata;
    wire        term_mem_ready;

    // DMA accelerator register interface signals
    wire        accel_reg_valid;
    wire        accel_reg_write;
    wire [7:0]  accel_reg_addr;
    wire [31:0] accel_reg_wdata;
    wire [31:0] accel_reg_rdata;
    wire        accel_reg_ready;

    // VexRiscv CPU system - run at 133 MHz (same as SDRAM controller, no CDC needed)
    cpu_system cpu (
        .clk(clk_ram_controller),  // 133 MHz - same as SDRAM controller
        .clk_74a(clk_74a),
        .reset_n(reset_n),
        .dataslot_allcomplete(dataslot_allcomplete),
        // Terminal interface
        .term_mem_valid(term_mem_valid),
        .term_mem_addr(term_mem_addr),
        .term_mem_wdata(term_mem_wdata),
        .term_mem_wstrb(term_mem_wstrb),
        .term_mem_rdata(term_mem_rdata),
        .term_mem_ready(term_mem_ready),
        // SDRAM interface (directly to io_sdram word interface via core_top)
        .sdram_rd(cpu_sdram_rd),
        .sdram_wr(cpu_sdram_wr),
        .sdram_addr(cpu_sdram_addr),
        .sdram_wdata(cpu_sdram_wdata),
        .sdram_rdata(cpu_sdram_rdata),
        .sdram_busy(cpu_sdram_busy),
        .sdram_rdata_valid(ram1_word_q_valid),
        // PSRAM interface (CRAM0)
        .psram_rd(cpu_psram_rd),
        .psram_wr(cpu_psram_wr),
        .psram_addr(cpu_psram_addr),
        .psram_wdata(cpu_psram_wdata),
        .psram_rdata(cpu_psram_rdata),
        .psram_busy(cpu_psram_busy),
        // DMA accelerator register interface
        .accel_reg_valid(accel_reg_valid),
        .accel_reg_write(accel_reg_write),
        .accel_reg_addr(accel_reg_addr),
        .accel_reg_wdata(accel_reg_wdata),
        .accel_reg_rdata(accel_reg_rdata),
        .accel_reg_ready(accel_reg_ready)
    );

    // DMA Dot Product Accelerator
    // Has direct access to SDRAM burst interface for high-bandwidth data transfer
    dma_dot_product #(
        .MAX_LENGTH(256)
    ) dma_accel (
        .clk(clk_ram_controller),
        .reset_n(reset_n),
        // CPU register interface
        .reg_valid(accel_reg_valid),
        .reg_write(accel_reg_write),
        .reg_addr(accel_reg_addr),
        .reg_wdata(accel_reg_wdata),
        .reg_rdata(accel_reg_rdata),
        .reg_ready(accel_reg_ready),
        // SDRAM burst interface
        .burst_rd(dma_burst_rd),
        .burst_addr(dma_burst_addr),
        .burst_len(dma_burst_len),
        .burst_32bit(dma_burst_32bit),
        .burst_data(dma_burst_data),
        .burst_data_valid(dma_burst_data_valid),
        .burst_data_done(dma_burst_data_done)
    );

    // Terminal display (40x30 characters, 320x240 pixels)
    wire [23:0] terminal_pixel_color;

    text_terminal terminal (
        .clk(clk_core_12288),
        .clk_cpu(clk_ram_controller),  // CPU clock for memory interface (133 MHz)
        .reset_n(reset_n),
        .pixel_x(visible_x),
        .pixel_y(visible_y),
        .pixel_color(terminal_pixel_color),
        .mem_valid(term_mem_valid),
        .mem_addr(term_mem_addr),
        .mem_wdata(term_mem_wdata),
        .mem_wstrb(term_mem_wstrb),
        .mem_rdata(term_mem_rdata),
        .mem_ready(term_mem_ready)
    );

always @(posedge clk_core_12288 or negedge reset_n) begin

    if(~reset_n) begin
    
        x_count <= 0;
        y_count <= 0;
        
    end else begin
        vidout_de <= 0;
        vidout_skip <= 0;
        vidout_vs <= 0;
        vidout_hs <= 0;
        
        vidout_hs_1 <= vidout_hs;
        vidout_de_1 <= vidout_de;
        
        // x and y counters
        x_count <= x_count + 1'b1;
        if(x_count == VID_H_TOTAL-1) begin
            x_count <= 0;
            
            y_count <= y_count + 1'b1;
            if(y_count == VID_V_TOTAL-1) begin
                y_count <= 0;
            end
        end
        
        // generate sync 
        if(x_count == 0 && y_count == 0) begin
            // sync signal in back porch
            // new frame
            vidout_vs <= 1;
            frame_count <= frame_count + 1'b1;
        end
        
        // we want HS to occur a bit after VS, not on the same cycle
        if(x_count == 3) begin
            // sync signal in back porch
            // new line
            vidout_hs <= 1;
        end

        // inactive screen areas are black
        vidout_rgb <= 24'h0;
        // generate active video
        if(x_count >= VID_H_BPORCH && x_count < VID_H_ACTIVE+VID_H_BPORCH) begin

            if(y_count >= VID_V_BPORCH && y_count < VID_V_ACTIVE+VID_V_BPORCH) begin
                // data enable. this is the active region of the line
                vidout_de <= 1;

                // Display terminal output
                vidout_rgb <= terminal_pixel_color;
            end
        end
    end
end




//
// audio i2s silence generator
// see other examples for actual audio generation
//

assign audio_mclk = audgen_mclk;
assign audio_dac = audgen_dac;
assign audio_lrck = audgen_lrck;

// generate MCLK = 12.288mhz with fractional accumulator
    reg         [21:0]  audgen_accum;
    reg                 audgen_mclk;
    parameter   [20:0]  CYCLE_48KHZ = 21'd122880 * 2;
always @(posedge clk_74a) begin
    audgen_accum <= audgen_accum + CYCLE_48KHZ;
    if(audgen_accum >= 21'd742500) begin
        audgen_mclk <= ~audgen_mclk;
        audgen_accum <= audgen_accum - 21'd742500 + CYCLE_48KHZ;
    end
end

// generate SCLK = 3.072mhz by dividing MCLK by 4
    reg [1:0]   aud_mclk_divider;
    wire        audgen_sclk = aud_mclk_divider[1] /* synthesis keep*/;
    reg         audgen_lrck_1;
always @(posedge audgen_mclk) begin
    aud_mclk_divider <= aud_mclk_divider + 1'b1;
end

// shift out audio data as I2S 
// 32 total bits per channel, but only 16 active bits at the start and then 16 dummy bits
//
    reg     [4:0]   audgen_lrck_cnt;    
    reg             audgen_lrck;
    reg             audgen_dac;
always @(negedge audgen_sclk) begin
    audgen_dac <= 1'b0;
    // 48khz * 64
    audgen_lrck_cnt <= audgen_lrck_cnt + 1'b1;
    if(audgen_lrck_cnt == 31) begin
        // switch channels
        audgen_lrck <= ~audgen_lrck;
        
    end 
end


///////////////////////////////////////////////


    wire    clk_core_12288;
    wire    clk_core_12288_90deg;
    wire    clk_ram_controller;
    wire    clk_ram_chip;
    wire    clk_ram_90;
    wire    clk_cpu;  // 36.3 MHz CPU clock (unused - using clk_74a instead)

    wire    pll_core_locked;
    wire    pll_core_locked_s;
synch_3 s01(pll_core_locked, pll_core_locked_s, clk_74a);

mf_pllbase mp1 (
    .refclk         ( clk_74a ),
    .rst            ( 0 ),

    .outclk_0       ( clk_core_12288 ),
    .outclk_1       ( clk_core_12288_90deg ),

    .outclk_2       ( clk_ram_controller ),
    .outclk_3       ( clk_ram_chip ),
    .outclk_4       ( clk_ram_90 ),
    .outclk_5       ( clk_cpu ),

    .locked         ( pll_core_locked )
);


// SDRAM controller using io_sdram from example
// Uses word interface for both bridge writes and CPU access

io_sdram isr0 (
    .controller_clk ( clk_ram_controller ),
    .chip_clk       ( clk_ram_chip ),
    .clk_90         ( clk_ram_90 ),
    .reset_n        ( 1'b1 ), // fsm has its own boot reset

    .phy_cke        ( dram_cke ),
    .phy_clk        ( dram_clk ),
    .phy_cas        ( dram_cas_n ),
    .phy_ras        ( dram_ras_n ),
    .phy_we         ( dram_we_n ),
    .phy_ba         ( dram_ba ),
    .phy_a          ( dram_a ),
    .phy_dq         ( dram_dq ),
    .phy_dqm        ( dram_dqm ),

    // Burst interface - used by DMA accelerator
    .burst_rd           ( dma_burst_rd ),
    .burst_addr         ( dma_burst_addr ),
    .burst_len          ( dma_burst_len ),
    .burst_32bit        ( dma_burst_32bit ),
    .burst_data         ( dma_burst_data ),
    .burst_data_valid   ( dma_burst_data_valid ),
    .burst_data_done    ( dma_burst_data_done ),

    // Burst write interface - not used
    .burstwr        ( 1'b0 ),
    .burstwr_addr   ( 25'b0 ),
    .burstwr_ready  ( ),
    .burstwr_strobe ( 1'b0 ),
    .burstwr_data   ( 16'b0 ),
    .burstwr_done   ( 1'b0 ),

    // Word interface - used for bridge writes and CPU access
    .word_rd    ( ram1_word_rd ),
    .word_wr    ( ram1_word_wr ),
    .word_addr  ( ram1_word_addr ),
    .word_data  ( ram1_word_data ),
    .word_q     ( ram1_word_q ),
    .word_busy  ( ram1_word_busy ),
    .word_q_valid ( ram1_word_q_valid )

);



endmodule
