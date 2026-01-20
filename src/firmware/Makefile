# VexRiscv Firmware Makefile for Analogue Pocket
# Supports llama2.c inference with SDRAM and data slot loading

# Toolchain
CROSS = riscv64-elf-
CC = $(CROSS)gcc
AS = $(CROSS)gcc
LD = $(CROSS)gcc
OBJCOPY = $(CROSS)objcopy
OBJDUMP = $(CROSS)objdump
SIZE = $(CROSS)size

# Target
TARGET = firmware

# Directories
FPGA_CORE_DIR = ../fpga/core
LIBC_DIR = libc

# Source files - Core
SRCS_C = main.c terminal.c dataslot.c llama_embedded.c memtest.c gguf.c

# Source files - libc
LIBC_SRCS = $(LIBC_DIR)/memory.c \
            $(LIBC_DIR)/string.c \
            $(LIBC_DIR)/ctype.c \
            $(LIBC_DIR)/stdlib.c \
            $(LIBC_DIR)/qsort.c \
            $(LIBC_DIR)/time.c \
            $(LIBC_DIR)/math.c \
            $(LIBC_DIR)/file.c

# Assembly sources
SRCS_S = start.S

# All sources
ALL_C_SRCS = $(SRCS_C) $(LIBC_SRCS)
OBJS = $(ALL_C_SRCS:.c=.o) $(SRCS_S:.S=.o)

# Compiler flags for RV32IM (VexRiscv_Full has MUL/DIV support)
ARCH = rv32im
ABI = ilp32
CFLAGS = -march=$(ARCH) -mabi=$(ABI) -O2 -Wall -Wextra
CFLAGS += -ffreestanding -nostdlib -nostartfiles
CFLAGS += -ffunction-sections -fdata-sections
CFLAGS += -fno-builtin
# VexRiscv_Full supports unaligned access - no need for -mstrict-align
CFLAGS += -I. -I$(LIBC_DIR)

# Assembler flags
ASFLAGS = -march=$(ARCH) -mabi=$(ABI)

# Linker flags
LDFLAGS = -march=$(ARCH) -mabi=$(ABI)
LDFLAGS += -T linker.ld -nostdlib -nostartfiles
LDFLAGS += -Wl,--gc-sections
LDFLAGS += -Wl,-Map=$(TARGET).map

# libgcc provides software floating-point routines (__addsf3, __mulsf3, etc.)
LIBS = -lgcc

# RAM size for MIF generation (64KB = 16384 words)
RAM_WORDS = 16384

# Default target
all: $(TARGET).bin $(TARGET).mif $(TARGET).lst
	$(SIZE) $(TARGET).elf

# Link
$(TARGET).elf: $(OBJS) linker.ld
	$(LD) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

# Binary output
$(TARGET).bin: $(TARGET).elf
	$(OBJCOPY) -O binary $< $@

# Quartus MIF output (Memory Initialization File)
# Converts binary to MIF format for altsyncram initialization
$(TARGET).mif: $(TARGET).bin
	@echo "-- Firmware RAM initialization - $(RAM_WORDS) x 32-bit words (16KB)" > $@
	@echo "-- Auto-generated from $(TARGET).bin" >> $@
	@echo "" >> $@
	@echo "WIDTH=32;" >> $@
	@echo "DEPTH=$(RAM_WORDS);" >> $@
	@echo "" >> $@
	@echo "ADDRESS_RADIX=DEC;" >> $@
	@echo "DATA_RADIX=HEX;" >> $@
	@echo "" >> $@
	@echo "CONTENT BEGIN" >> $@
	@hexdump -v -e '1/4 "%08X\n"' $< | awk '{printf "%d : %s;\n", NR-1, $$1}' >> $@
	@LAST=$$(hexdump -v -e '1/4 "%08X\n"' $< | wc -l); \
	if [ $$LAST -lt $(RAM_WORDS) ]; then \
		echo "[$$LAST..$$(($(RAM_WORDS)-1))] : 00000013;" >> $@; \
	fi
	@echo "END;" >> $@
	@echo "Generated $@ with $$(hexdump -v -e '1/4 "%08X\n"' $< | wc -l) words of firmware"

# Disassembly listing
$(TARGET).lst: $(TARGET).elf
	$(OBJDUMP) -d -S $< > $@

# Compile C sources
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Compile libc sources
$(LIBC_DIR)/%.o: $(LIBC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Assemble assembly sources
%.o: %.S
	$(AS) $(ASFLAGS) -c -o $@ $<

# Install MIF to FPGA core directory
install: $(TARGET).bin $(TARGET).mif
	cp $(TARGET).mif $(FPGA_CORE_DIR)/firmware.mif
	@echo "MIF installed to $(FPGA_CORE_DIR)/firmware.mif"

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET).elf $(TARGET).bin $(TARGET).mif $(TARGET).map $(TARGET).lst

# Rebuild everything
rebuild: clean all

# Show memory usage
mem: $(TARGET).elf
	$(SIZE) -A -x $(TARGET).elf

.PHONY: all clean rebuild install mem
