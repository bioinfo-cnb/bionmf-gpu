#########################################################################
# Copyright (C) 2011-2013:
#
#	Edgardo Mejia-Roa(*), Carlos Garcia, Jose Ignacio Gomez,
#	Manuel Prieto, Francisco Tirado and Alberto Pascual-Montano(**).
#
#	(*)  ArTeCS Group, Complutense University of Madrid (UCM), Spain.
#	(**) Functional Bioinformatics Group, Biocomputing Unit,
#		National Center for Biotechnology-CSIC, Madrid, Spain.
#
#	E-mail for E. Mejia-Roa: <edgardomejia@fis.ucm.es>
#	E-mail for A. Pascual-Montano: <pascual@cnb.csic.es>
#
#
# This file is part of bioNMF-mGPU..
#
# BioNMF-mGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BioNMF-mGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BioNMF-mGPU.  If not, see <http://www.gnu.org/licenses/>.
#
#########################################################################

################################################################################
#
# Makefile for bioNMF-GPU
#
# Parameters (set to '1' to enable; e.g., "make SINGLE=1"):
#
#	SINGLE:			If set to 1, uses single-precision data (i.e., 'float').
#				Else, uses double-precision data (i.e., 'double')
#
#	FAST_MATH:		Uses less-precise faster math functions.
#
#	TIME:			Shows total elapsed time.
#
#	TRANSF_TIME:		Shows time elapsed on data transfers.
#
#	KERNEL_TIME:		Shows time elapsed on kernel code.
#
#	SYNC_TRANSF:		Performs SYNCHRONOUS data transfers
#
#	FIXED_INIT:		Makes use of "random" values from a fixed seed (default: random seed).
#
#	DBG:			VERBOSE and DEBUG mode (prints A LOT of information).
#
#	VERBOSE:		Command-line verbosity level. Valid values are 0 (none, default),
#				1 (shows make commands) and 2 (shows, also, internal NVCC commands).
#
#	NVCC_WARN_VERBOSE:	Shows NVCC warnings. Implies PTXAS_WARN_VERBOSE.
#
#	PTXAS_WARN_VERBOSE:	Shows PTX-specific (i.e., DEVICE-code compilation) warnings.
#
#	KEEP_INTERMEDIATE:	Keeps NVCC temporary files.
#
#
################################################################################

# Default Values

# Single precision (i.e., float). Set to 0 for double-precision data
SINGLE := 1

# Use less-precise, faster math functions
FAST_MATH := 1

# Show total elapsed time
TIME := 1

# User compilation flags
USER_FLAGS :=


########################################

# NOTE: PLEASE ADJUST THE FOLLOWING OPTIONS ACCORDING TO YOUR SYSTEM.

# Path to CUDA Toolkit.
CUDA_PATH	:= /usr/local/cuda-5.5
CUDA_INCDIR	:= $(CUDA_PATH)/include
export CPATH	:= $(CPATH):$(CUDA_INCDIR)

# GPU Compute Capability
SM := 13

# Target GPU architectures.
#
# Note that "compute_XX" refers to a PTX version and "sm_XX" refers to a cubin version.
# The 'arch=' clause must always be a PTX version. The 'code=' clause can either be cubin, PTX or both.
# Only the target version(s) specified by the 'code=' clause will be retained in the resulting binary.
# At least one of such targets should be PTX in order to provide compatibility with future architectures.
GENCODE_ARCH := -gencode=arch=compute_$(SM),code=\"sm_$(SM),compute_$(SM)\"


########################################


# C/C++ common flags
CC := gcc
COMMON_FLAGS	  := -pipe -O3 -fstrict-aliasing
COMMON_WARN_FLAGS := -Wall -Wextra -Wpointer-arith -Wcast-align -Wstrict-overflow=5 -Wunsafe-loop-optimizations \
			-Wmissing-declarations -Winline -Wno-unused-parameter -Wno-unused-variable

# Additional flags for FAST_MATH mode.
COMMON_OPTIM_FLAGS := -march=native -ffast-math -fbranch-target-load-optimize2 -fomit-frame-pointer \
			-ftree-loop-distribution -funroll-loops -funsafe-loop-optimizations
# -Ofast -mssse3 -msse4.2 -mfpmath=sse -fbranch-target-load-optimize

# C-only flags.
CFLAGS		:= $(COMMON_FLAGS) -D_GNU_SOURCE -std=c99 -fPIC
CWARN_FLAGS	:= $(COMMON_WARN_FLAGS) -Wnested-externs -Wstrict-prototypes -Wmissing-prototypes
C_OPTIM_FLAGS	:= $(COMMON_OPTIM_FLAGS)


# Linkage flags (must be in C++ mode).
LINK	:= g++


########################################

# CUDA compiler
NVCC := nvcc


# Flags for HOST code (always compiled as C++).
NVCCFLAGS := --restrict $(addprefix --compiler-options ,$(COMMON_FLAGS))

# # Internal compiler for HOST code.
# #	Useful to make NVCC to follow a customized CC variable (e.g., 'CC=icc', 'CC=gcc-4.2', etc).
# #	Otherwise, CC is ignored and the default C/C++ compiler is invoked (GNU CC for UNIX, and Visual for Windows).
# NVCCFLAGS += --compiler-bindir $(CC)

# Additional flags for WARN_VERBOSE mode.
NVCC_WARN_FLAGS := $(addprefix --compiler-options ,$(COMMON_WARN_FLAGS))
# --Werror cross-execution-space-call

# Additional flags for FAST_MATH mode.
NVCC_OPTIM_FLAGS := --use_fast_math $(addprefix --compiler-options ,$(COMMON_OPTIM_FLAGS))


#########


# Flags for nvopencc (i.e., PTX code generation).
export OPENCC_FLAGS := -inline -O3 -fPIC -fstrict-aliasing

# Additional flags for WARN_VERBOSE mode.
OPENCC_WARN_FLAGS := -Wall -W -Wcast-align -Wstrict-aliasing -Wstrict-prototypes -Wmissing-declarations -Winline -Wpointer-arith \
		-Wno-unused-parameter -Wno-unused-variable

# Additional flags for FAST_MATH mode.
OPENCC_OPTIM_FLAGS := -ffast-stdlib -ffast-math -finline-functions -fomit-frame-pointer -funroll-loops -funsafe-math-optimizations
# -msse4a -gcc -gnu42


#########


# Flags for PTX (i.e., GPU-assembler) code compilation.
export PTXAS_FLAGS := --opt-level=4 --allow-expensive-optimizations=true --warning-as-error 0
# --def-load-cache=ca --def-store-cache=wb

# Additional flags for WARN_VERBOSE mode.
PTXAS_WARN_FLAGS := --generate-line-info --verbose


########################################
# Source and target filenames
########################################

# Directory tree
SRCDIR := src
INCDIR := include
BINDIR := bin
OBJDIR := $(BINDIR)/obj

# Matrix Operations directory name
MATRIX_DIR := matrix

# Tools directory name
TOOLS_DIRNAME := tools
TOOLS_DIR := $(MATRIX_DIR)/$(TOOLS_DIRNAME)

########################################

# C files
C_FILES		:= $(MATRIX_DIR)/matrix_io_routines.c $(MATRIX_DIR)/matrix_io.c common.c
C_OBJS		:= $(addprefix $(OBJDIR)/,$(addsuffix .o,$(C_FILES)))
C_INCLUDES	:= -I$(INCDIR) -I$(INCDIR)/$(MATRIX_DIR)
C_LDFLAGS	:= -lm

# CUDA files
CUDA_FILES	:= timing.cu GPU_kernels.cu GPU_setup.cu $(MATRIX_DIR)/matrix_operations.cu NMF_routines.cu
CUDA_OBJS	:= $(addprefix $(OBJDIR)/,$(addsuffix .o,$(CUDA_FILES)))
CUDA_INCLUDES	:= --include-path $(INCDIR) --include-path $(INCDIR)/$(MATRIX_DIR) --include-path $(CUDA_INCDIR)
CUDA_LDFLAGS	:= -lcudart -lcublas -lcurand

.SUFFIXES:
.SUFFIXES: .c .o .cu

########################################

# Tools
TOOLS_FILES	:= $(TOOLS_DIR)/file_converter.c $(TOOLS_DIR)/generate_matrix.c
TOOLS_OBJDIR	:= $(OBJDIR)/$(TOOLS_DIR)
TOOLS_BINDIR	:= $(BINDIR)/$(TOOLS_DIRNAME)
TOOLS_TARGETS	:= $(patsubst $(TOOLS_DIR)/%.c,$(TOOLS_BINDIR)/%,$(TOOLS_FILES))
TOOLS_DEPS	:= $(C_OBJS)
TOOLS_INCLUDES	:= $(C_INCLUDES)
TOOLS_LDFLAGS	:= $(C_LDFLAGS)


# Main Program (single-GPU version)
SINGLE_GPU_FILE		:= NMF_GPU.c
SINGLE_GPU_SRC		:= $(SRCDIR)/$(SINGLE_GPU_FILE)
SINGLE_GPU_OBJ		:= $(OBJDIR)/$(SINGLE_GPU_FILE).o
SINGLE_GPU_TARGET	:= $(BINDIR)/$(basename $(SINGLE_GPU_FILE))
SINGLE_GPU_DEPS		:= $(C_OBJS) $(CUDA_OBJS)
SINGLE_GPU_INCLUDES	:= $(C_INCLUDES) -I$(CUDA_INCDIR)
SINGLE_GPU_LDFLAGS	:= $(CUDA_LDFLAGS) $(C_LDFLAGS)


########################################
# Compilation flags according to Makefile parameters.
########################################

# Use single-precision data.
ifeq ($(SINGLE),1)
	USER_FLAGS += -DNMFGPU_SINGLE_PREC
endif


# Use faster math functions
ifeq ($(FAST_MATH),1)
	CFLAGS += $(C_OPTIM_FLAGS)
	NVCCFLAGS += $(NVCC_OPTIM_FLAGS)
	export OPENCC_FLAGS := $(OPENCC_FLAGS) $(OPENCC_OPTIM_FLAGS)
endif


# Show total elapsed time
ifeq ($(TIME),1)
	USER_FLAGS += -DNMFGPU_PROFILING_GLOBAL
endif


# Show time elapsed on data transfers
ifeq ($(TRANSF_TIME),1)
	USER_FLAGS += -DNMFGPU_PROFILING_TRANF
endif


# Show time elapsed on kernel code.
ifeq ($(KERNEL_TIME),1)
	USER_FLAGS += -DNMFGPU_PROFILING_KERNELS
endif


# Performs SYNCHRONOUS data transfers.
ifeq ($(SYNC_TRANSF),1)
	USER_FLAGS += -DNMFGPU_SYNC_TRANSF
endif


# Fixed initial values for W and H. Useful for debugging.
ifeq ($(FIXED_INIT),1)
	USER_FLAGS += -DNMFGPU_FIXED_INIT
endif


# VERBOSE and DEBUG mode (prints A LOT of information)
ifeq ($(DBG),1)
	USER_FLAGS += -DNMFGPU_VERBOSE -DNMFGPU_DEBUG

	# Other DEBUG flags

	# Verbosity level
	USER_FLAGS += -DNMFGPU_VERBOSE2

	# Data transfers
	USER_FLAGS += -DNMFGPU_DEBUG_TRANSF


	USER_FLAGS += -DNMFGPU_FORCE_BLOCKS
	USER_FLAGS += -DNMFGPU_TEST_BLOCKS
	USER_FLAGS += -DNMFGPU_DEBUG_REDUCT

	# Flags for I/O Debug & testing.
	USER_FLAGS += -DNMFGPU_TESTING
	USER_FLAGS += -DNMFGPU_DEBUG_READ_FILE
	USER_FLAGS += -DNMFGPU_DEBUG_READ_FILE2
	USER_FLAGS += -DNMFGPU_DEBUG_READ_MATRIX
	USER_FLAGS += -DNMFGPU_DEBUG_READ_MATRIX2
endif


# Verbose compiling commands
ifeq ($(VERBOSE),2)
	CMD_PREFIX :=
	NVCCFLAGS += --verbose
else
	ifeq ($(VERBOSE),1)
		CMD_PREFIX :=
	else
		CMD_PREFIX := @
	endif
endif


# Show NVCC warnings.
ifeq ($(NVCC_WARN_VERBOSE),1)
	NVCCFLAGS += $(NVCC_WARN_FLAGS)
	export OPENCC_FLAGS := $(OPENCC_FLAGS) $(OPENCC_WARN_FLAGS)
	PTXAS_WARN_VERBOSE := 1
endif


# Show kernel compilation warnings
ifeq ($(PTXAS_WARN_VERBOSE),1)
	export PTXAS_FLAGS := $(PTXAS_FLAGS) $(PTXAS_WARN_FLAGS)
endif


# Keep intermediate code files.
ifeq ($(KEEP_INTERMEDIATE),1)
	NVCCFLAGS += --keep
	CLEAN :=
else
	CLEAN := clean
endif


########################################
# Main Compilation Rules
########################################

# Always keeps all intermediate files (unless 'clean' target is explicitly requested)
.SECONDARY:
.PRECIOUS:


# Rule to compile all programs.
.PHONY: all
all : single_gpu tools $(CLEAN)


# Main Program (single-GPU version)
.PHONY: single_gpu
single_gpu : $(SINGLE_GPU_TARGET)


# Utility programs
.PHONY: tools
tools : $(TOOLS_TARGETS)


########################################
# Program-specific rules
########################################

# Main Program (single-GPU version, C++ code)
$(SINGLE_GPU_TARGET) : $(SINGLE_GPU_DEPS)
	$(CMD_PREFIX)mkdir -p $(@D) $(<D)
	$(CMD_PREFIX)$(CC) $(CFLAGS) $(CWARN_FLAGS) $(USER_FLAGS) $(SINGLE_GPU_INCLUDES) -o $(SINGLE_GPU_OBJ) -c $(SINGLE_GPU_SRC)
	$(CMD_PREFIX)$(LINK) $(COMMON_WARN_FLAGS) $(SINGLE_GPU_INCLUDES) $(SINGLE_GPU_LDFLAGS) -o $@ $^ $(SINGLE_GPU_OBJ)


# Rule for a single tool program (C code)
$(TOOLS_BINDIR)/% : $(TOOLS_OBJDIR)/%.c.o $(TOOLS_DEPS)
	$(CMD_PREFIX)mkdir -p $(@D)
	$(CMD_PREFIX)$(CC) $(CFLAGS) $(CWARN_FLAGS) $(USER_FLAGS) $(TOOLS_INCLUDES) $(TOOLS_LDFLAGS) -o $@ $^


#########


# CUDA_FILES
$(OBJDIR)/%.cu.o : $(SRCDIR)/%.cu
	$(CMD_PREFIX)mkdir -p $(@D)
	$(CMD_PREFIX)$(NVCC) $(NVCCFLAGS) $(GENCODE_ARCH) $(USER_FLAGS) $(CUDA_INCLUDES) --output-file $@ --compile $<


# C_FILES
$(OBJDIR)/%.c.o : $(SRCDIR)/%.c
	$(CMD_PREFIX)mkdir -p $(@D)
	$(CMD_PREFIX)$(CC) $(CFLAGS) $(CWARN_FLAGS) $(USER_FLAGS) $(C_INCLUDES) -o $@ -c $<


########################################
# Clean-up rules
########################################

# Removes executable and object files, as well as the directory tree.

.PHONY: clobber
clobber:
	$(CMD_PREFIX)rm -rf $(BINDIR)


.PHONY: clobber_single_gpu
clobber_single_gpu: clean_single_gpu
	$(CMD_PREFIX)rm -f $(SINGLE_GPU_TARGET)


.PHONY: clobber_tools
clobber_tools: clean_tools
	$(CMD_PREFIX)rm -f $(TOOLS_TARGETS)


########################################

# Removes object files ONLY.

.PHONY: clean
clean:
	$(CMD_PREFIX)rm -rf $(OBJDIR)


.PHONY: clean_single_gpu
clean_single_gpu: clean_cuda_deps clean_c_deps


.PHONY: clean_tools
clean_tools : clean_c_deps


.PHONY: clean_cuda_deps
clean_cuda_deps:
	$(CMD_PREFIX)rm -f $(CUDA_DEPS_OBJS)


.PHONY: clean_c_deps
clean_c_deps:
	$(CMD_PREFIX)rm -f $(C_OBJS)
