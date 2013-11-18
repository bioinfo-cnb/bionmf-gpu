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
 ################################################################################

 # Default Values:

# Single precision (i.e., float). Set to 0 for double-precision data
SINGLE := 1
# Use less-precise, faster math functions
FAST_MATH := 1

# Show total elapsed time
TIME := 1

########################################

 # NOTE: PLEASE ADJUST THE FOLLOWING OPTIONS ACCORDING TO YOUR SYSTEM.

# Path to CUDA Toolkit.
CUDA_PATH	:= /usr/local/cuda-5.5
cuda_incdir	:= $(CUDA_PATH)/include
 export CPATH	:= $(CPATH):$(cuda_incdir)

# Target GPU architectures.
#	Please note these values will be used to set both PTX intermediate code, and the final binary image.
#	Since there is no "compute_21" PTX version, if you want to compile specific code for Compute Capability 2.1, please
#	uncomment the "other_archs" sentence below.
SM_VERSIONS := 13 10 13 20 30

# To compile specific code for Compute Capability 2.1. Comment to disable.
# other_archs := --generate-code=arch=compute_20,code=sm_21

# This line provides compatibility with future GPU architectures. Comment to disable.
other_archs += --generate-code=arch=compute_35,code=\"sm_35,compute_35\"


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
single_gpu_file	  := NMF_GPU.c
tools_files	  := $(tools_dir)/file_converter.c $(tools_dir)/generate_matrix.c
cuda_files	  := timing.cu GPU_setup.cu $(matrix_dir)/matrix_operations.cu NMF_routines.cu
cuda_device_files := GPU_kernels.cu
c_files		  := $(matrix_dir)/matrix_io_routines.c $(matrix_dir)/matrix_io.c common.c


########################################
# Compiler options
########################################

# C/C++ common flags
CC := gcc
optim_level	   := 3
common_cflags	   := -O$(optim_level) -pipe
common_warn_cflags := -Wall -Wextra -Wpointer-arith -Wcast-align -Wstrict-overflow=5 -Wunsafe-loop-optimizations \
			-Wmissing-declarations -Winline -Wno-unused-parameter -Wno-unused-variable
common_fast_cflags := -march=native -ffast-math -fbranch-target-load-optimize2 -fomit-frame-pointer \
			-ftree-loop-distribution -funroll-loops -funsafe-loop-optimizations
 # Other system-dependent fast flags: -Ofast -mssse3 -msse4.2 -mfpmath=sse -fbranch-target-load-optimize
common_ldflags	   :=
common_ldlibs	   :=


# C-only flags.
c_only_cflags	   := -D_GNU_SOURCE -std=c99 $(common_cflags) $(common_warn_cflags) -Wnested-externs -Wstrict-prototypes -Wmissing-prototypes
c_only_fast_cflags := $(common_fast_cflags)
c_only_ldflags	   := $(common_ldflags)
c_only_ldlibs	   := $(common_ldlibs)


# Linkage program (must be in C++).
LINK	:= $(CXX)


########################################

# CUDA compiler
NVCC	:= nvcc

# Flags for HOST code (always compiled as C++).
nvcc_cflags	 := --restrict --optimize $(optim_level) $(addprefix --compiler-options ,$(common_cflags)) \
			$(addprefix --compiler-options ,$(CXXFLAGS))
nvcc_warn_cflags := $(addprefix --compiler-options ,$(common_warn_cflags))
nvcc_fast_cflags := --use_fast_math $(addprefix --compiler-options ,$(common_fast_cflags))
nvcc_ldflags	 := $(addprefix --linker-options ,$(common_ldflags))
nvcc_ldlibs	 := -lcudart $(common_ldlibs)

# Internal compiler for HOST code.
#	Useful to make NVCC to follow a customized CC variable (e.g., 'CC=icc', 'CC=gcc-4.2', etc).
#	Otherwise, CC is ignored and the default C/C++ compiler is invoked (GNU CC for UNIX, and Visual for Windows).
#
# nvcc_cflags += --compiler-bindir $(CC)

# Others useful options for CygWin and Mingw under Windows platforms.
#
# nvcc_cflags += --drive-prefix /cygwin/
# nvcc_cflags += --drive-prefix /


#########


# Flags for nvopencc (i.e., PTX code generation).
 export OPENCC_FLAGS := -inline -O3 -fPIC -fstrict-aliasing
opencc_warn_cflags := -Wall -W -Wcast-align -Wstrict-aliasing -Wstrict-prototypes -Wmissing-declarations -Winline -Wpointer-arith \
			-Wno-unused-parameter -Wno-unused-variable
opencc_fast_cflags := -ffast-stdlib -ffast-math -finline-functions -fomit-frame-pointer -funroll-loops -funsafe-math-optimizations
 # Other system-dependent flags: -msse4a -gcc -gnu42


#########


# Flags for PTX (i.e., GPU-assembler) code compilation.
 export PTXAS_FLAGS := --opt-level=4 --allow-expensive-optimizations=true --warning-as-error 0
ptxas_warn_cflags := --generate-line-info --verbose

 # # Other flags controlling the cache policy:
 # export PTXAS_FLAGS = $(PTXAS_FLAGS) --def-load-cache=ca --def-store-cache=wb


#########


# Target GPU architectures.
#	In the following rule, "compute_XX" refers to a virtual architecture, and "sm_XX" refers to a "real" device.
#	The 'arch=' clause must always be of type "compute_XX". The 'code=' clause can either be "compute_XX", "sm_XX" or both.
#	Only target versions specified in the 'code=' clause will be retained in the resulting binary. At least one of such
#	targets must be virtual in order to provide compatibility with future ('real') architectures.
#
gencode_arch_template = --generate-code=arch=compute_$(sm_ver),code=sm_$(sm_ver)

gencode_arch := $(foreach sm_ver,$(SM_VERSIONS),$(gencode_arch_template))
gencode_arch += $(other_archs)


########################################

SHELL := /bin/sh

.SUFFIXES:
.SUFFIXES: .c .o .cu

########################################
# User Makefile parameters.
########################################

# Preprocessor options
cppflags :=

# Use single-precision data.
ifeq ($(SINGLE),1)
	cppflags += -DNMFGPU_SINGLE_PREC
endif


# Use faster math functions
ifeq ($(FAST_MATH),1)
	c_only_cflags	+= $(c_only_fast_cflags)
	nvcc_cflags	+= $(nvcc_fast_cflags)
	export OPENCC_FLAGS := $(OPENCC_FLAGS) $(opencc_fast_cflags)
endif


# Show total elapsed time
ifeq ($(TIME),1)
	cppflags += -DNMFGPU_PROFILING_GLOBAL
endif


# Show time elapsed on data transfers
ifeq ($(TRANSF_TIME),1)
	cppflags += -DNMFGPU_PROFILING_TRANF
endif


# Show time elapsed on kernel code.
ifeq ($(KERNEL_TIME),1)
	cppflags += -DNMFGPU_PROFILING_KERNELS
endif


# Performs SYNCHRONOUS data transfers.
ifeq ($(SYNC_TRANSF),1)
	cppflags += -DNMFGPU_SYNC_TRANSF
endif


# Fixed initial values for W and H. Useful for debugging.
ifeq ($(FIXED_INIT),1)
	cppflags += -DNMFGPU_FIXED_INIT
endif


# Generates values from the CPU (host) random generator,
# not from the GPU device.
ifeq ($(CPU_RANDOM),1)
	cppflags += -DNMFGPU_CPU_RANDOM
endif

# VERBOSE and DEBUG mode (prints A LOT of information)
ifeq ($(DBG),1)
	cppflags += -DNMFGPU_VERBOSE -DNMFGPU_DEBUG

	# Other DEBUG flags

	# Verbosity level
	cppflags += -DNMFGPU_VERBOSE_2

	# Data transfers
	cppflags += -DNMFGPU_DEBUG_TRANSF


#	cppflags += -DNMFGPU_FORCE_BLOCKS
#	cppflags += -DNMFGPU_TEST_BLOCKS
#	cppflags += -DNMFGPU_FORCE_DIMENSIONS
#	cppflags += -DNMFGPU_DEBUG_REDUCT

#	# Flags for I/O Debug & testing.
#	cppflags += -DNMFGPU_TESTING
#	cppflags += -DNMFGPU_DEBUG_READ_FILE
#	cppflags += -DNMFGPU_DEBUG_READ_FILE2
#	cppflags += -DNMFGPU_DEBUG_READ_MATRIX
#	cppflags += -DNMFGPU_DEBUG_READ_MATRIX2
endif


# Verbose compiling commands
ifeq ($(VERBOSE),2)
	cmd_prefix :=
	nvcc_cflags += --verbose
else
	ifeq ($(VERBOSE),1)
		cmd_prefix :=
	else
		cmd_prefix := @
	endif
endif


# Show NVCC warnings.
ifeq ($(NVCC_WARN_VERBOSE),1)
	nvcc_cflags += $(nvcc_warn_cflags)
	export OPENCC_FLAGS := $(OPENCC_FLAGS) $(opencc_warn_cflags)
	PTXAS_WARN_VERBOSE := 1
endif


# Show kernel compilation warnings
ifeq ($(PTXAS_WARN_VERBOSE),1)
	export PTXAS_FLAGS := $(PTXAS_FLAGS) $(ptxas_warn_cflags)
endif


# Keep intermediate code files.
ifeq ($(KEEP_INTERMEDIATE),1)
	nvcc_cflags += --keep
	clean_files :=
else
	clean_files := clean
endif


########################################
# Source code compilation flags
########################################

# C files
c_objs		:= $(addprefix $(objdir)/,$(addsuffix .o,$(c_files)))
c_cflags	:= $(c_only_cflags) $(cppflags)
c_includes	:= -I$(incdir) -I$(incdir)/$(matrix_dir)
c_ldflags	:= $(c_only_ldflags)
c_ldlibs	:= $(c_only_ldlibs) -lm

# CUDA files
cuda_device_objs := $(addprefix $(objdir)/,$(addsuffix .o,$(cuda_device_files)))
cuda_objs	 := $(cuda_device_objs) $(addprefix $(objdir)/,$(addsuffix .o,$(cuda_files)))
cuda_cflags	 := $(patsubst -D%,--define-macro %,$(cppflags) $(CPPFLAGS)) $(nvcc_cflags)
cuda_includes	 := --include-path $(incdir) --include-path $(incdir)/$(matrix_dir)
cuda_ldflags	 := $(nvcc_ldflags)
cuda_ldlibs	 := $(nvcc_ldlibs) -lcublas -lcurand


# Tools
tools_objdir	:= $(objdir)/$(tools_dir)
tools_bindir	:= $(bindir)/$(tools_dirname)
tools_targets	:= $(patsubst $(tools_dir)/%.c,$(tools_bindir)/%,$(tools_files))
tools_deps	:= $(c_objs)
tools_cflags	:= $(c_cflags)
tools_includes	:= $(c_includes)
tools_ldflags	:= $(c_ldflags)
tools_ldlibs	:= $(c_ldlibs)


# Main Program (single-GPU version)
single_gpu_obj		:= $(objdir)/$(single_gpu_file).o
single_gpu_target	:= $(bindir)/$(basename $(single_gpu_file))
single_gpu_deps		:= $(c_objs) $(cuda_objs)
single_gpu_cflags	:= $(c_cflags)
single_gpu_includes	:= $(c_includes)
single_gpu_ldflags	:= $(c_ldflags)
single_gpu_ldlibs	:= $(cuda_ldlibs) $(c_ldlibs)


########################################
# Main Compilation Rules
########################################

# Always keeps all intermediate files (unless the 'clean' target is explicitly requested)
.SECONDARY:
.PRECIOUS:


# Rule to compile all programs.
.PHONY: all
all : single_gpu tools $(clean_files)


# Main Program (single-GPU version)
.PHONY: single_gpu
single_gpu : $(single_gpu_target)


# Utility programs
.PHONY: tools
tools : $(tools_targets)


########################################
# Program-specific rules
########################################

# Main Program (single-GPU version, C++ code)
$(single_gpu_target) : $(single_gpu_deps) $(single_gpu_obj)
	$(cmd_prefix)mkdir -p $(@D)
	$(cmd_prefix)$(LINK) $(single_gpu_includes) $(single_gpu_cflags) $(single_gpu_ldlibs) $(single_gpu_ldflags) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $^


# Tools (C code)
$(tools_bindir)/% : $(tools_deps) $(tools_objdir)/%.c.o
	$(cmd_prefix)mkdir -p $(@D)
	$(cmd_prefix)$(CC) $(tools_cflags) $(tools_includes) $(tools_ldlibs) $(tools_ldflags) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $^


# CUDA host and device code
$(cuda_device_objs) : cuda_cflags+=$(gencode_arch)
$(objdir)/%.cu.o : $(srcdir)/%.cu
	$(cmd_prefix)mkdir -p $(@D)
	$(cmd_prefix)$(NVCC) $(cuda_cflags) $(cuda_includes) $(NVCCFLAGS) --output-file $@ --compile $<


# C files
$(objdir)/%.c.o : $(srcdir)/%.c
	$(cmd_prefix)mkdir -p $(@D)
	$(cmd_prefix)$(CC) $(c_cflags) $(c_includes) $(CPPFLAGS) $(CFLAGS) -o $@ -c $<


########################################
# Clean-up rules
########################################

# Removes executable and object files, as well as the directory tree.

.PHONY: clobber
clobber:
	$(cmd_prefix)rm -rf $(bindir)


.PHONY: clobber_single_gpu
clobber_single_gpu: clean_single_gpu
	$(cmd_prefix)rm -f $(single_gpu_target)


.PHONY: clobber_tools
clobber_tools: clean_tools
	$(cmd_prefix)rm -f $(tools_targets)


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
	$(cmd_prefix)rm -f $(cuda_objs)


.PHONY: clean_c
clean_c:
	$(cmd_prefix)rm -f $(c_objs)

########################################

