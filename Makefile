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
#				Else, uses double-precision data (i.e., 'double').
#				Please note that double-precision data is NOT supported on
#				Compute Capability 1.2 or lower.
#
#	UNSIGNED:		Uses unsigned integers for matrix dimensions.
#				Note that, however, some CUDA Libraries require SIGNED integer values.
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
#	CPU_RANDOM:		Generates random values from the CPU Host, not the GPU device.
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
#################
#
# You can add compiling options using the following variables:
# Note they might overwrite other default flags.
#
#	CFLAGS: Options for C-only programs (excludes CUDA code).
#		They are also included in the linkage command.
#
#	CXXFLAGS: Options controlling the NVCC's internal compiler for CUDA source files.
#		  They are automatically prefixed with "--compiler-options".
#
#	NVCCFLAGS: Options for the NVCC compiler.
#
################################################################################

# Default Values:

# Single precision (i.e., float). Set to 0 for double-precision data
SINGLE := 1

# Use unsigned integers for matrix dimensions
UNSIGNED := 1

# Use less-precise, faster math functions
FAST_MATH := 1

# Show total elapsed time
TIME := 1

########################################

# NOTE: PLEASE ADJUST THE FOLLOWING OPTIONS ACCORDING TO YOUR SYSTEM.

# Path to CUDA Toolkit.
CUDA_INSTALL_PATH := /usr/local/cuda-5.5
nvcc_BINDIR	  := $(CUDA_INSTALL_PATH)/bin
nvcc_libdir	  := $(CUDA_INSTALL_PATH)/lib
nvcc_incdir	  := $(CUDA_INSTALL_PATH)/include

# Target GPU architectures.
#	Please note these values will be used to set both PTX intermediate code, and the final binary image.
#	Since there is no "compute_21" PTX version, if you want to compile specific code for Compute Capability 2.1, please
#	use the dash-separated form as shown below.
#
# WARNING: The compilation process will fail on SM versions "12" or less if the "SINGLE" option is set to zero, since
# double-precision data is NOT supported on those architectures.
SM_VERSIONS := 10 13 20 30

# To compile specific code for Compute Capability 2.1. Comment to disable.
SM_VERSIONS := $(SM_VERSIONS) 20-21

# This line provides compatibility with future GPU architectures. Comment to disable.
SM_VERSIONS := $(SM_VERSIONS) 35-35


########################################
# Source code directory and filenames
########################################

# Directory tree
srcdir := src
incdir := include
bindir := bin
objdir := $(bindir)/obj

# Matrix Operations directory name
matrix_dir := matrix

# Tools directory name
tools_dirname := tools
tools_dir := $(matrix_dir)/$(tools_dirname)

# Source files
 # Note : "cuda_device_FILES" refers to CUDA files including device code.
single_gpu_FILE	  := NMF_GPU.c
tools_FILES	  := $(tools_dir)/file_converter.c $(tools_dir)/generate_matrix.c
cuda_device_FILES := GPU_kernels.cu
cuda_FILES	  := timing.cu GPU_setup.cu $(matrix_dir)/matrix_operations.cu NMF_routines.cu
c_FILES		  := $(matrix_dir)/matrix_io_routines.c $(matrix_dir)/matrix_io.c common.c


########################################
# Compiler options
########################################

# C/C++ common flags
CC := gcc
optim_level	   := 3
common_CFLAGS	   := -O$(optim_level) -pipe -fPIC
common_warn_CFLAGS := -Wall -Wextra -Wpointer-arith -Wcast-align -Wstrict-overflow=5 -Wunsafe-loop-optimizations \
			-Wmissing-declarations -Winline -Wno-unused-parameter -Wno-unused-variable -Wno-type-limits
common_fast_CFLAGS := -march=native -ffast-math -fbranch-target-load-optimize2 -fomit-frame-pointer \
			-ftree-loop-distribution -funroll-loops -funsafe-loop-optimizations
 # Other system-dependent fast flags: -Ofast -mssse3 -msse4.2 -mfpmath=sse -fbranch-target-load-optimize

# C-only flags.
c_only_CFLAGS	   := -D_GNU_SOURCE -std=c99 $(common_CFLAGS) $(common_warn_CFLAGS) -Wnested-externs -Wstrict-prototypes -Wmissing-prototypes
c_only_fast_CFLAGS := $(common_fast_CFLAGS)

# Linkage program (must be in C++).
LINK	:= gcc

########################################

# CUDA compiler
NVCC := $(nvcc_BINDIR)/nvcc

# Flags for HOST code (always compiled as C++).
nvcc_CFLAGS	 := --restrict --optimize $(optim_level) $(addprefix --compiler-options ,$(common_CFLAGS))
nvcc_warn_CFLAGS := $(addprefix --compiler-options ,$(common_warn_CFLAGS))
nvcc_fast_CFLAGS := --use_fast_math $(addprefix --compiler-options ,$(common_fast_CFLAGS))
nvcc_LDLIBS	 := -lcudart


# Internal compiler for HOST code.
#	Useful to make NVCC to follow a customized CC variable (e.g., 'CC=icc', 'CC=gcc-4.2', etc).
#	Otherwise, CC is ignored and the default C/C++ compiler is invoked (GNU CC for UNIX, and Visual for Windows).
#
# nvcc_CFLAGS += --compiler-bindir $(CC)

# Others useful options for CygWin and Mingw under Windows platforms.
#
# nvcc_CFLAGS += --drive-prefix /cygwin/
# nvcc_CFLAGS += --drive-prefix /


#########


# Flags for nvopencc (i.e., PTX code generation).
 export OPENCC_FLAGS := $(OPENCC_FLAGS) -inline -O3 -fPIC -fstrict-aliasing
opencc_warn_CFLAGS := -Wall -W -Wcast-align -Wstrict-aliasing -Wstrict-prototypes -Wmissing-declarations -Winline -Wpointer-arith \
			-Wno-unused-parameter -Wno-unused-variable
opencc_fast_CFLAGS := -ffast-stdlib -ffast-math -finline-functions -fomit-frame-pointer -funroll-loops -funsafe-math-optimizations
 # Other system-dependent flags: -msse4a -gcc -gnu42


#########


# Flags for PTX (i.e., GPU-assembler) code compilation.
 export PTXAS_FLAGS := $(PTXAS_FLAGS) --opt-level=4 --allow-expensive-optimizations=true --warning-as-error 0
ptxas_warn_CFLAGS := --generate-line-info --verbose

 # # Other flags controlling the cache policy:
 # export PTXAS_FLAGS = $(PTXAS_FLAGS) --def-load-cache=ca --def-store-cache=wb


#########


# Target GPU architectures.
#	In the following rule, "compute_XX" refers to a virtual architecture, and "sm_XX" refers to a "real" device.
#	The 'arch=' clause must always be of type "compute_XX". The 'code=' clause can either be "compute_XX", "sm_XX" or both.
#	Only target versions specified in the 'code=' clause will be retained in the resulting binary. At least one of such
#	targets must be virtual in order to provide compatibility with future ('real') architectures.
#
gencode_arch_template = --generate-code=arch=compute_$(1),code=sm_$(2)

# The argument may be of the form "XX" or "XX-XX".
sm_template = $(call gencode_arch_template,$(firstword $(1)),$(lastword $(1)))

# For each SM version, splits any existing dash-separated sub-arguments and call "sm_template".
sm_CFLAGS := $(foreach sm_ver,$(SM_VERSIONS),$(call sm_template,$(subst -, ,$(sm_ver))))


# Double-precision data is NOT supported on Compute Capability <= 1.2

# Searches for unsupported Compute Capabilities.
unsupported_sm_versions := $(filter 10 11 12,$(subst -, ,$(SM_VERSIONS)))

# Error message to show.
error_msj := Double-precision data is NOT supported on Compute Capability 1.2 or lower. \
		Please, either use single-precision data (SINGLE=1) or remove the unsupported \
		architectures from the SM_VERSIONS variable.


########################################

SHELL := /bin/sh

.SUFFIXES:
.SUFFIXES: .c .o .cu

########################################
# User Makefile parameters.
########################################

# Preprocessor options
CPPFLAGS :=

# Use single-precision data.
ifeq ($(SINGLE),1)
	CPPFLAGS += -DNMFGPU_SINGLE_PREC=1
endif

# Use unsigned integers for matrix dimensions
ifeq ($(UNSIGNED),1)
	CPPFLAGS += -DNMFGPU_UINDEX=1
endif

# Use faster math functions
ifeq ($(FAST_MATH),1)
	c_only_CFLAGS	+= $(c_only_fast_CFLAGS)
	nvcc_CFLAGS	+= $(nvcc_fast_CFLAGS)
	export OPENCC_FLAGS := $(OPENCC_FLAGS) $(opencc_fast_CFLAGS)
endif


# Show total elapsed time
ifeq ($(TIME),1)
	CPPFLAGS += -DNMFGPU_PROFILING_GLOBAL=1
endif


# Show time elapsed on data transfers
ifeq ($(TRANSF_TIME),1)
	CPPFLAGS += -DNMFGPU_PROFILING_TRANSF=1
endif


# Show time elapsed on kernel code.
ifeq ($(KERNEL_TIME),1)
	CPPFLAGS += -DNMFGPU_PROFILING_KERNELS=1
endif


# Performs SYNCHRONOUS data transfers.
ifeq ($(SYNC_TRANSF),1)
	CPPFLAGS += -DNMFGPU_SYNC_TRANSF=1
endif


# Fixed initial values for W and H. Useful for debugging.
ifeq ($(FIXED_INIT),1)
	CPPFLAGS += -DNMFGPU_FIXED_INIT=1
endif


# Generates values from the CPU (host) random generator,
# not from the GPU device.
ifeq ($(CPU_RANDOM),1)
	CPPFLAGS += -DNMFGPU_CPU_RANDOM=1
endif

# VERBOSE and DEBUG mode (prints A LOT of information)
ifeq ($(DBG),1)
	CPPFLAGS += -DNMFGPU_VERBOSE=1 -DNMFGPU_DEBUG=1

	# Other DEBUG flags

	# Verbosity level
	CPPFLAGS += -DNMFGPU_VERBOSE_2=1

#	# Data transfers
#	CPPFLAGS += -DNMFGPU_DEBUG_TRANSF=1

#	CPPFLAGS += -DNMFGPU_FORCE_BLOCKS=1
#	CPPFLAGS += -DNMFGPU_TEST_BLOCKS=1
#	CPPFLAGS += -DNMFGPU_FORCE_DIMENSIONS=1
#	CPPFLAGS += -DNMFGPU_DEBUG_REDUCT=1

#	# Flags for I/O Debug & testing.
#	CPPFLAGS += -DNMFGPU_DEBUG_READ_FILE=1
#	CPPFLAGS += -DNMFGPU_DEBUG_READ_FILE2=1
#	CPPFLAGS += -DNMFGPU_DEBUG_READ_MATRIX=1
#	CPPFLAGS += -DNMFGPU_DEBUG_READ_MATRIX2=1
#	CPPFLAGS += -DNMFGPU_TESTING=1
endif


# Verbose compiling commands
ifeq ($(VERBOSE),2)
	cmd_prefix :=
	nvcc_CFLAGS += --verbose
else
	ifeq ($(VERBOSE),1)
		cmd_prefix :=
	else
		cmd_prefix := @
	endif
endif


# Show NVCC warnings.
ifeq ($(NVCC_WARN_VERBOSE),1)
	nvcc_CFLAGS += $(nvcc_warn_CFLAGS)
	export OPENCC_FLAGS := $(OPENCC_FLAGS) $(opencc_warn_CFLAGS)
	PTXAS_WARN_VERBOSE := 1
endif


# Show kernel compilation warnings
ifeq ($(PTXAS_WARN_VERBOSE),1)
	export PTXAS_FLAGS := $(PTXAS_FLAGS) $(ptxas_warn_CFLAGS)
endif


# Keep intermediate code files.
ifeq ($(KEEP_INTERMEDIATE),1)
	nvcc_CFLAGS += --keep
	clean_FILES :=
else
	clean_FILES := clean
endif


########################################
# Source code compilation flags
########################################

# C files
c_OBJS		:= $(addprefix $(objdir)/,$(addsuffix .o,$(c_FILES)))
c_CFLAGS	:= $(c_only_CFLAGS) $(CPPFLAGS)
c_INCLUDES	:= -I$(incdir) -I$(incdir)/$(matrix_dir)
c_LDLIBS	:= -lm

# CUDA files
cuda_device_OBJS := $(addprefix $(objdir)/,$(addsuffix .o,$(cuda_device_FILES)))
cuda_OBJS	 := $(addprefix $(objdir)/,$(addsuffix .o,$(cuda_FILES)))
cuda_CFLAGS	 := $(nvcc_CFLAGS) $(CPPFLAGS)
cuda_INCLUDES	 := $(addprefix --compiler-options ,$(c_INCLUDES)) $(addprefix -I,$(nvcc_incdir))
cuda_LDLIBS	 := -lcublas -lcurand

# Tools
tools_OBJDIR	:= $(objdir)/$(tools_dir)
tools_BINDIR	:= $(bindir)/$(tools_dirname)
tools_TARGETS	:= $(patsubst $(tools_dir)/%.c,$(tools_BINDIR)/%,$(tools_FILES))
tools_DEPS	:= $(c_OBJS)
tools_CFLAGS	:= $(c_CFLAGS)
tools_INCLUDES	:= $(c_INCLUDES)
tools_LDLIBS	:= $(c_LDLIBS)

# Main Program (single-GPU version)
single_gpu_OBJ		:= $(objdir)/$(single_gpu_FILE).o
single_gpu_TARGET	:= $(bindir)/$(basename $(single_gpu_FILE))
single_gpu_DEPS		:= $(c_OBJS) $(cuda_device_OBJS) $(cuda_OBJS)
single_gpu_CFLAGS	:= $(c_CFLAGS)
single_gpu_INCLUDES	:= $(c_INCLUDES) $(addprefix -I,$(nvcc_incdir))
single_gpu_LDFLAGS	:= $(addprefix -L,$(nvcc_libdir))
single_gpu_LDLIBS	:= $(cuda_LDLIBS) $(nvcc_LDLIBS) $(c_LDLIBS)


########################################
# Main Compilation Rules
########################################

# Always keeps all intermediate files (unless the 'clean' target is explicitly requested)
.SECONDARY:
.PRECIOUS:


# Rule to compile all programs.
.PHONY: all
all : single_gpu tools $(clean_FILES)


# Main Program (single-GPU version)
.PHONY: single_gpu
single_gpu : $(single_gpu_TARGET)


# Utility programs
.PHONY: tools
tools : $(tools_TARGETS)


########################################
# Program-specific rules
########################################

# Main Program (single-GPU version, C++ code)
$(single_gpu_TARGET) : $(single_gpu_DEPS) $(single_gpu_OBJ)
	$(cmd_prefix)mkdir -p $(@D)
	$(cmd_prefix)$(LINK) $(single_gpu_CFLAGS) $(single_gpu_INCLUDES) $(single_gpu_LDFLAGS) $^ $(single_gpu_LDLIBS) $(CFLAGS) -o $@

# Tools (C code)
$(tools_BINDIR)/% : $(tools_DEPS) $(tools_OBJDIR)/%.c.o
	$(cmd_prefix)mkdir -p $(@D)
	$(cmd_prefix)$(CC) $(tools_CFLAGS) $(tools_INCLUDES) $^ $(tools_LDLIBS) $(CFLAGS) -o $@


# CUDA DEVICE code
$(cuda_device_OBJS) : cuda_CFLAGS+=$(sm_CFLAGS)
$(objdir)/%.cu.o : $(srcdir)/%.cu check_sm_versions
	$(cmd_prefix)mkdir -p $(@D)
	$(cmd_prefix)$(NVCC) $(cuda_CFLAGS) $(cuda_INCLUDES) $(addprefix --compiler-options ,$(CXXFLAGS)) $(NVCCFLAGS) --output-file $@ --compile $<

# CUDA HOST code
$(objdir)/%.cu.o : $(srcdir)/%.cu
	$(cmd_prefix)mkdir -p $(@D)
	$(cmd_prefix)$(NVCC) $(cuda_CFLAGS) $(cuda_INCLUDES) $(addprefix --compiler-options ,$(CXXFLAGS)) $(NVCCFLAGS) --output-file $@ --compile $<


# C files
$(single_gpu_OBJ) : c_INCLUDES:=$(single_gpu_INCLUDES)
$(objdir)/%.c.o : $(srcdir)/%.c
	$(cmd_prefix)mkdir -p $(@D)
	$(cmd_prefix)$(CC) $(c_CFLAGS) $(c_INCLUDES) $(CFLAGS) -o $@ -c $<


# Double-precision data is not supported on Compute Capability <= 1.2
.PHONY: check_sm_versions
check_sm_versions :
ifneq ($(SINGLE),1)
ifneq ($(strip $(unsupported_sm_versions)),)
	$(error $(error_msj))
endif
endif

########################################
# Clean-up rules
########################################

# Removes executable and object files, as well as the directory tree.

.PHONY: clobber
clobber:
	$(cmd_prefix)rm -rf $(bindir)


.PHONY: clobber_single_gpu
clobber_single_gpu: clean_single_gpu
	$(cmd_prefix)rm -f $(single_gpu_TARGET)


.PHONY: clobber_tools
clobber_tools: clean_tools
	$(cmd_prefix)rm -f $(tools_TARGETS)


########################################

# Removes object files ONLY.

.PHONY: clean
clean:
	$(cmd_prefix)rm -rf $(objdir)


.PHONY: clean_single_gpu
clean_single_gpu: clean_cuda clean_c


.PHONY: clean_tools
clean_tools : clean_c


.PHONY: clean_cuda
clean_cuda:
	$(cmd_prefix)rm -f $(cuda_OBJS)


.PHONY: clean_c
clean_c:
	$(cmd_prefix)rm -f $(c_OBJS)

########################################

