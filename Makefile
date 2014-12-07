##########################################################################
#
# NMF-mGPU - Non-negative Matrix Factorization on multi-GPU systems.
#
# Copyright (C) 2011-2014:
#
#	Edgardo Mejia-Roa(*), Carlos Garcia(*), Jose Ignacio Gomez(*),
#	Manuel Prieto(*), Francisco Tirado(*) and Alberto Pascual-Montano(**).
#
#	(*)  ArTeCS Group, Complutense University of Madrid (UCM), Spain.
#	(**) Functional Bioinformatics Group, Biocomputing Unit,
#		National Center for Biotechnology-CSIC, Madrid, Spain.
#
#	E-mail for E. Mejia-Roa: <edgardomejia@fis.ucm.es>
#	E-mail for A. Pascual-Montano: <pascual@cnb.csic.es>
#
#
# This file is part of NMF-mGPU.
#
# NMF-mGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# NMF-mGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NMF-mGPU. If not, see <http://www.gnu.org/licenses/>.
#
##########################################################################

##########################################################################
#
# Makefile for NMF-mGPU on UNIX systems.
#
# Targets:
#
#	all:		DEFAULT. Compiles all programs except the multi-GPU
#			version. It is equivalent to: 'single_gpu tools'.
#
#	single_gpu:	Compiles NMF-mGPU (single-GPU version).
#
#	multi_gpu:	Compiles bioNMF-mGPU (multi-GPU version).
#			Target NOT compiled by default.
#
#	tools:		Compiles some utility programs.
#			Currently, this target does NOT require any CUDA-related
#			configuration or software. In particular, it is NOT
#			necessary to specify 'CUDA_HOME' or 'SM_VERSIONS'
#			parameters.
#
#	help:		Prints a help message with all this information.
#
#	help_sm_versions:
#			Prints a detailed description of valid values for
#			the 'SM_VERSIONS' parameter.
#
#	help_tools:	Prints a short description of available utility programs.
#
#	clobber:	Removes the entire binary directory.
#
#	clobber_single_gpu, clobber_multi_gpu, clobber_tools:
#			Removes the specified executable and object files.
#
#	clean:		Removes all object files (i.e., *.o).
#
#	clean_single_gpu, clean_multi_gpu, clean_tools:
#			Removes the specified object files.
#
#
# Parameters:
#
#	CUDA_HOME:	Path to CUDA Toolkit. It may be an environment variable
#			or an argument. If not specified, it will be derived by
#			looking for <NVCC> in all folders stored in your "PATH"
#			environment variable. Please note that folder names
#			containing whitespace characters are NOT supported. In
#			that case, either use a (soft) link, or rename your CUDA
#			installation directory. This parameter is currently
#			ignored on the 'tools' target.
#
#	SM_VERSIONS:	Target GPU architecture(s). This parameter may be an
#			environment variable or an argument. Device code will be
#			generated for the specified "Compute Capability(-ies)" (CC).
#			For instance, SM_VERSIONS="10-13  30  PTX35":
#			* Generates device-specific executable code for CC 1.3,
#			  using only the basic functionality present on CC 1.0.
#			* Generates device-specific executable code for CC 3.0,
#			  using all available features on such architecture.
#			* Emits PTX code for CC 3.5, which can be later
#			  compiled and executed by any current or future device,
#			  with a similar or higher Compute Capability.
#			To generate device-specific executable code for CC 2.1,
#			please specify it as: '20-21'.
#			See a more detailed description of this parameter by
#			executing: 'make help_sm_versions'.
#			This parameter is currently ignored on the 'tools' target.
#
#	SINGLE:		If set to '1', uses single-precision data (i.e., 'float').
#			Else, uses double-precision data (i.e., 'double'). Note,
#			however, that in Compute Capability 1.2 and lower, all
#			double-precision operations are demoted to single-
#			precision arithmetic.
#
#	UNSIGNED:	Uses UNSIGNED integers for matrix dimensions, which may
#			generate faster code. Nevertheless, please note that
#			CUBLAS library functions use SIGNED-integer parameters.
#			Therefore, matrix dimensions must not overflow such data
#			type. An error message will be shown if this happens.
#
#	FAST_MATH:	Uses less-precise faster math functions.
#
#	TIME:		Shows total elapsed time.
#
#	TRANSF_TIME:	Shows time elapsed on data transfers.
#
#	KERNEL_TIME:	Shows time elapsed on kernel code.
#
#	COMM_TIME:	Shows time elapsed on MPI communications.
#
#	SYNC_TRANSF:	Performs SYNCHRONOUS data transfers.
#
#	FIXED_INIT:	Makes use of "random" values from a fixed seed.
#
#	CPU_RANDOM:	Generates random values from the CPU host, not from the
#			GPU device.
#
#	DBG:		VERBOSE and DEBUG mode (prints A LOT of information).
#			It implies CC_WARN_VERBOSE, NVCC_WARN_VERBOSE, and
#			PTXAS_WARN_VERBOSE set to '1'.
#
#	VERBOSE:	Command-line verbosity level. Valid values are '0'
#			(none), '1' (shows make commands), and '2' (shows make
#			and internal NVCC commands).
#
#	CC_WARN_VERBOSE:
#			Shows extra warning messages on programs compiled with
#			CC. Please note that it may include "false-positive"
#			warnings.
#
#	NVCC_WARN_VERBOSE:
#			Shows extra warning messages on programs compiled with
#			NVCC. Please note that it may include "false-positive"
#			warnings. It implies PTXAS_WARN_VERBOSE set to '1'.
#
#	PTXAS_WARN_VERBOSE:
#			Shows PTX-specific compilation warnings.
#
#	KEEP_INTERMEDIATE:
#			Keeps temporary files generated by NVCC.
#
#
# You can add other compiling options, or overwrite the default flags, using the
# following parameters (they may be environment variables, as well):
#
#	CC:		Compiler for C-only programs and CUDA host code.
#			Supported compilers: 'gcc' and 'clang'.
#			Default value: 'clang' for Darwin. 'gcc' otherwise.
#
#	NVCC:		Compiler for CUDA device code, and CUDA kernel-related
#			host code. Default value: 'nvcc'.
#
#	CFLAGS:		Options for C-only programs (excludes CUDA code).
#			They are also included in the final linking stage.
#
#	CXXFLAGS:	Options controlling the NVCC's internal compiler for
#			CUDA source files. They are automatically prefixed
#			with '--compiler-options' in the command line.
#
#	NVCCFLAGS:	Options for the NVCC compiler.
#
#	LDFLAGS:	Options for the linker.
#
#	OPENCC_FLAGS:	Flags for the 'nvopencc' compiler, which generates PTX
#			(intermediate) code on devices of Compute Capability "1.x".
#
#	PTXAS_FLAGS:	Flags for PTX code compilation, which generates the
#			actual GPU assembler.
#
#	MPICC:		Compiler for MPI code. Default value: 'mpicc'.
#
#	MPICC_FLAGS:	Options for MPI code. They are also included in the
#			final linking stage.
#
###################
#
# Tool programs:
#
# In addition to "NMF-mGPU", there are some utility programs to make easier
# working with input files. It includes a program for binary-text file conversion
# and another to generate input matrices with random data (useful for testing).
#
# List of generated files:
#	bin/tools/file_converter
#	bin/tools/generate_matrix
#
# Please note that such programs do NOT make use of the GPU device. They are
# implemented in pure-C99 language, and all operations are performed on the HOST
# (i.e., the CPU). Therefore, they do NOT require any CUDA-related option,
# configuration or software. In particular, it is NOT necessary to specify the
# 'CUDA_HOME' and/or 'SM_VERSIONS' parameters.
#
# 1) Binary-text file converter:
#    Since "NMF-mGPU" accepts input matrices stored in a binary or ASCII-text
#    file, this program allows file conversion between both formats.
#    For binary files, there are two sub-formats: "native" and non-"native".
#
#	* "Native" mode refers to "raw" I/O. That is, data are stored/loaded
#	  with the precision specified at compilation time: 'float' if 'SINGLE'
#	  was set to '1', or 'double' otherwise. Matrix dimensions are stored/
#	  loaded in a similar way (i.e., 'unsigned int', if 'UNSIGNED' was set
#	  to '1'; '[signed] int', otherwise). This mode is faster because NO
#	  error checking is performed. Therefore, it should NOT be used to read
#	  untrusted input files.
#
#	* In Non-"Native" mode, data are ALWAYS stored/loaded using double
#	  precision (and unsigned integers for matrix dimensions), regardless
#	  the options specified at compilation. This is the recommended mode for
#	  input or final output data, since every datum is checked for invalid
#	  format.
#
# 2) Matrix generator:
#    This program generates a valid data matrix with random values. You can
#    select output matrix dimensions, as well as the maximum value for matrix
#    data. The output matrix can be written as ASCII-text, or in a binary file
#    (in any of the binary modes described above).
#
#################################################################################

# Default Values:

# Single precision (i.e., float). Set to '0' for double-precision data.
SINGLE := 1

# Unsigned integers for matrix dimensions.
UNSIGNED := 1

# Less-precise, faster math functions.
FAST_MATH := 1

# Target GPU architectures.
ifeq ($(SM_VERSIONS),)

	#############
	 #
	 # There are three ways to specify target architecture(s) in the SM_VERSIONS
	 # parameter:
	 #
	 # 1) Device-specific features & code (i.e., PTX and executable code, with SIMILAR
	 #    Compute Capability):
	 #
	 #    Emits PTX assembler code, which is then compiled into executable
	 #    instructions just for the given "Compute Capability(-ies)" (CC). Since the
	 #    former is just an intermediate code (i.e., it is NOT retained in the output
	 #    file), and the latter is generated with a device-specific binary format, the
	 #    program may not be compatible with other GPU architectures. That is, any
	 #    given CC value, "XY", is translated to the following NVCC option:
	 #    "--generate-code=arch=compute_XY,code=sm_XY"
	 #
	 #    For instance, 'SM_VERSIONS="13  35"' generates executable code just for
	 #    devices of Compute Capability 1.3 and 3.5, respectively. GPU devices with
	 #    other CC values, such as 1.1 or 2.0, may not be able to execute the program.
	 #
	 #    NOTE:
	 #    For devices, such as 2.1, that do not have a similar PTX CC number, please
	 #    specify the nearest lower value ("2.0", for the previous example) by using
	 #    the dashed-separated form below.
	 #
	 #
	 # 2) Generic features, device-specific code (i.e., PTX and executable code, with
	 #    DIFFERENT Compute Capabilities):
	 #
	 #    This is a generalization of the previous form. Here, different "Compute
	 #    Capabilities" (CC) can be specified for both, PTX and executable code,
	 #    separated by a dash.
	 #    That is,
	 #
	 #		"XY-WZ"		(with XY <= WZ),
	 #
	 #    emits PTX assembler code for CC "X.Y", which is then compiled into
	 #    executable instructions for a device of CC "W.Z". The former, determines the
	 #    target ARCHITECTURE (i.e., the available hardware/ software features and
	 #    functionality). The latter, specifies the target DEVICE in terms of binary
	 #    code format.
	 #
	 #    Similarly as above, no PTX code is embedded in the output file, so the
	 #    program may not be compatible with other GPU device models. That is, the
	 #    previous expression is translated to the following NVCC option:
	 #    "--generate-code=arch=compute_XY,code=sm_WZ"
	 #
	 #    Note that "XY-XY" is equivalent to just specify "XY" as in the previous form.
	 #    On the other hand, if XY < WZ, the program is still compiled for the target
	 #    device (i.e. CC "W.Z"), but it will only make use of features available on
	 #    CC "X.Y", discarding any functionality introduced since.
	 #
	 #    NOTE:
	 #    As stated above, please use this form to specify target devices, such as
	 #    CC 2.1, that do not have a similar PTX CC number (so, a lower value must
	 #    also be biven). Example: '20-21'.
	 #
	 #    For instance, 'SM_VERSIONS="10-13  20-21"':
	 #
	 #	* Generates executable code for a device of CC 1.3, with the basic
	 #	  features that are available on CC 1.0. In particular, it discards all
	 #	  support for double-precision floating-point data, introduced on CC 1.3.
	 #	* Compiles the algorithm with the features and functionality available
	 #	  on CC 2.0, and generates a binary image for a device of CC 2.1.
	 #	* Since no PTX code is retained in the output file, the program may not
	 #	  compatible with other GPU devices (e.g., CC 3.0).
	 #
	 #
	 # 3) Generic features & "code":
	 #
	 #    Emits PTX assembler code for the given "Compute Capability" (CC), and
	 #    embeds it into the output file. No executable code is generated. Instead,
	 #    the former is dynamically compiled at runtime according to the actual GPU
	 #    device. Such process is known as "Just In Time" (JIT) compilation.
	 #
	 #    To specify a target architecture in a such way, please use the word 'PTX'
	 #    followed by the target Compute Capability.
	 #    That is,
	 #		"PTXwz"
	 #
	 #    generates PTX code for Compute Capability "w.z", and embeds it into the
	 #    output file. Such code can be later compiled and executed on any device,
	 #    with a similar or greater CC value. Similarly as previous forms, the
	 #    expression above is translated to the following NVCC option:
	 #    "--generate-code=arch=compute_wz,code=compute_wz".
	 #
	 #    Note, however, that JIT compilation increases the startup delay. In
	 #    addition, the final executable code will use just those architectural
	 #    features that are available on CC "w.z", discarding any functionality
	 #    introduced since.
	 #
	 #    For instance, 'SM_VERSIONS="PTX10  PTX35"':
	 #
	 #	* Emits PTX code for the first CUDA-capable architecture (i.e., CC 1.0).
	 #	  Therefore, the program can be later dynamically compiled and executed
	 #	  on ANY current or future GPU device. Nevertheless, it will only use
	 #	  the (very) basic features present on such architecture.
	 #	* Generates PTX code that can be later compiled and executed on devices
	 #	  of CC 3.5, or higher.
	 #	* Any device prior to CC 3.5 (e.g., 1.3, 2.1, or 3.0), will execute the
	 #	  basic CC 1.0 version.
	 #
	 #    This parameter is currently ignored on the 'tools' target.
	 #
	#############


	# Compiles device-specific code for the following Compute Capability
	# values.
	#
	# WARNING:
	#	Double-precision floating-point data are NOT supported on Compute
	#	Capability 1.2, or lower. If parameter "SINGLE" is set to '0',
	#	the compilation process will fail.
	#
	SM_VERSIONS := 10 13 20 30 35

	# Compiles specific code for Compute Capability 2.1. Comment to disable.
	SM_VERSIONS += 20-21

	# This line provides compatibility with future GPU architectures. Comment to disable.
	SM_VERSIONS += PTX35
endif

# Initializes unspecified input parameters.
CFLAGS		?=
CXXFLAGS	?=
NVCCFLAGS	?=
LDFLAGS		?=
OPENCC_FLAGS	?=
PTXAS_FLAGS 	?=
MPICC_FLAGS	?=

########################################

# Shell to be used for command executions.
SHELL := /bin/bash

# OS Name
os_lower := $(strip $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]"))

# Flags to detect 32-bits or 64-bits OS platform
os_size := $(strip $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/"))

# Whitespace
empty :=
space := $(empty) $(empty)

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
# Note : "cuda_device_FILES" refers to CUDA files with device code.
single_gpu_FILE	  := NMF_GPU.c
multi_gpu_FILE	  := NMF_mGPU.c
tools_FILES	  := $(tools_dir)/file_converter.c $(tools_dir)/generate_matrix.c
cuda_device_FILES := GPU_kernels.cu
cuda_FILES	  := timing.cu GPU_setup.cu $(matrix_dir)/matrix_operations.cu NMF_routines.cu
c_FILES		  := common.c $(matrix_dir)/matrix_io_routines.c $(matrix_dir)/matrix_io.c


########################################
# Compiler options
########################################

# Default C compiler:
ifeq ("$(os_lower)","darwin")
	default_CC := clang
else
	default_CC := gcc
endif

# Previous selection can be overridden with the parameter or environment variable "CC"
CC ?= $(default_CC)

CC_name := $(or $(findstring gcc,$(CC)),$(findstring clang,$(CC)),$(strip $(notdir $(CC))))

#######

# Common C/C++ flags (i.e., common CC/NVCC flags)
common_CFLAGS = -O3 -pipe -fPIC -fPIE -D_GNU_SOURCE -fpch-preprocess -Wall -Wcast-align -Wstrict-overflow=5 \
		-Winline -Wno-invalid-pch -Wno-unused-parameter -Wno-unused-variable \
		-m$(os_size) $($(CC_name)_$(os_lower)_common_CFLAGS)

# Common C/C++ flags for faster code:
common_fast_CFLAGS = $($(CC_name)_$(os_lower)_common_fast_CFLAGS) -ffast-math -funroll-loops -mfpmath=sse -msse4.2

# Flags for C-only programs (i.e., flags for programs to be compiled with CC)
c_only_CFLAGS	   = -std=c99 $(common_CFLAGS) $($(CC_name)_$(os_lower)_c_only_CFLAGS) -Wnested-externs \
		     -Wstrict-prototypes -Wmissing-prototypes
c_only_fast_CFLAGS = $($(CC_name)_$(os_lower)_c_only_fast_CFLAGS) $(common_fast_CFLAGS)

# Common link flags
common_LDFLAGS	= -pie

#######

# Compiler- and OS-specific flags.

# GCC on Linux:
gcc_linux_common_CFLAGS		:= -Wextra -Wmissing-declarations -Wunsafe-loop-optimizations -Wno-type-limits
gcc_linux_common_fast_CFLAGS	:= -Ofast -march=native -ftree-loop-distribution -fbranch-target-load-optimize -funsafe-loop-optimizations
gcc_linux_c_only_CFLAGS		:= -Wno-unsuffixed-float-constants -Wno-unused-result
gcc_linux_c_only_fast_CFLAGS	:=

# GCC on Darwin:
gcc_darwin_common_CFLAGS	:= -Wunsafe-loop-optimizations -Wmissing-field-initializers -Wempty-body
gcc_darwin_common_fast_CFLAGS	:= -fast -fbranch-target-load-optimize -funsafe-loop-optimizations
gcc_darwin_c_only_CFLAGS	:= -Wmissing-declarations
gcc_darwin_c_only_fast_CFLAGS	:=

# Clang on Linux:
clang_linux_common_CFLAGS	:= -Wextra -Wmissing-declarations -Wno-unknown-warning-option -Qunused-arguments -Wno-tautological-compare \
					-Wno-type-limits
clang_linux_common_fast_CFLAGS	:= -Ofast -march=native
clang_linux_c_only_CFLAGS	:= -Wno-unused-result
clang_linux_c_only_fast_CFLAGS	:=

# Clang on Darwin:
clang_darwin_common_CFLAGS	:= -Wextra -Wmissing-declarations -Qunused-arguments -Wno-tautological-compare -Wno-type-limits
clang_darwin_common_fast_CFLAGS	:= -O4 -march=native -ftree-loop-distribution -fbranch-target-load-optimize -funsafe-loop-optimizations
clang_darwin_c_only_CFLAGS	:= -Wno-unused-result -Wno-unknown-warning-option
clang_darwin_c_only_fast_CFLAGS	:=


########################################

# MPI compiler
default_MPICC := mpicc

# Default MPI compiler can be overridden with "MPICC"
MPICC ?= $(default_MPICC)

########################################

# CUDA compiler
default_NVCC := nvcc

NVCC ?= $(default_NVCC)
NVCC_basename := $(notdir $(NVCC))


# Flags for HOST code (always compiled as C++).
nvcc_CFLAGS	 := --restrict --optimize 3 $(addprefix --compiler-options ,$(common_CFLAGS)) --disable-warnings
nvcc_fast_CFLAGS := --use_fast_math $(addprefix --compiler-options ,$(common_fast_CFLAGS))
nvcc_LDLIBS	 := -lcudart


# Internal compiler for HOST code.
#	Useful to make NVCC to follow a customized CC variable (e.g., CC='gcc-4.2').
#	Otherwise, CC is ignored and the default C/C++ compiler is invoked ('gcc' for Linux, clang for Darwin, and 'cl.exe' for Windows).
nvcc_CFLAGS	 += --compiler-bindir $(CC)

# Others useful options for CygWin and Mingw, respectively, under MS Windows platforms.
#
# nvcc_CFLAGS += --drive-prefix /cygwin/
# nvcc_CFLAGS += --drive-prefix /

# To specify the version of Microsoft Visual Studio installation. Valid values: 2008, 2010, and 2012.
# nvcc_CFLAGS += --use-local-env --cl-version 2008


# Path to CUDA Toolkit:
#	If CUDA_HOME was specified, uses that path, regardless any (other) environment variable.
#	Else, searches for <NVCC_basename> in all folders stored in PATH.
#
# WARNING:
#	Folder names with whitespace characters are NOT supported. Please, either use a (soft) link,
#	or rename your CUDA installation directory.
#
ifeq ($(CUDA_HOME),)

	# There was no CUDA_HOME explicitly given, so tries to determine it from locating <NVCC_basename> in PATH:

	pathsearch = $(firstword $(wildcard $(addsuffix /$(1),$(subst :, ,$(PATH)))))
	# Note: Same effect using the "which" shell command: $(shell which $(1))

	nvcc_path := $(dir $(call pathsearch,$(NVCC_basename)))

	ifneq ($(nvcc_path),)
		CUDA_HOME := $(realpath $(nvcc_path)/..)
	endif
endif


# Updates NVCC
NVCC := $(CUDA_HOME)/bin/$(NVCC_basename)

nvcc_incdir := $(CUDA_HOME)/include
nvcc_libdir := $(firstword $(wildcard $(CUDA_HOME)/lib$(os_size) $(CUDA_HOME)/lib))

# Error message to show if no path to CUDA Toolkit was found.
# NOTE:
#	* Literal tabs and newline characters are ignored. Only '\t' and '\n' are printed.
#	* Only double quotes (") need to be escaped, if any.
error_cuda_home_not_found := "\n\
	ERROR:\n\n\
		'CUDA_HOME' not set, and could not find any path to '$(NVCC_basename)' in your \"PATH\"\n\
		environment variable. Please, either set CUDA_HOME to your CUDA-Toolkit\n\
		installation path, or add to PATH the path to '$(NVCC_basename)'.\n\
		\n\
		Finally, please remember that folders with whitespace characters are NOT\n\
		supported. In that case, please either use a (soft) link or rename your CUDA\n\
		installation directory.\n"


#########

# Flags for nvopencc, which generates PTX (intermediate) code.
# Used on Compute Capability 1.x only.

# Many options are similar to those used on C-only programs, but not all...
not_in_opencc := -Wextra -Wstrict-overflow% -W%type-limits -ftree-loop-distribution -mfpmath=% -msse4.% -W%unsuffixed-float-constants \
		-march=% -f%branch-target-load-optimize -f%branch-target-load-optimize2 -Ofast% -fast -fpch% -W%pch -f%pch \
		-f%unsafe-loop-optimizations -W%unknown-% -Qunused-% -W%tautological-compare -W%missing-field-initializers

export OPENCC_FLAGS := $(filter-out $(not_in_opencc),$(common_CFLAGS)) -W $(OPENCC_FLAGS)
opencc_fast_CFLAGS  := $(filter-out $(not_in_opencc),$(c_only_fast_CFLAGS)) -ffast-stdlib -ffast-math -inline

#########

# Flags for PTX code compilation, which generates the actual GPU assembler.
export PTXAS_FLAGS := --opt-level=4 --allow-expensive-optimizations=true --warning-as-error 0 $(PTXAS_FLAGS)
ptxas_warn_CFLAGS := --generate-line-info --verbose

# # Other flags controlling the cache policy:
# export PTXAS_FLAGS = --def-load-cache=ca --def-store-cache=wb $(PTXAS_FLAGS)


#########

##
 # Target GPU architectures.
 #
 #	Device code is compiled in two stages. First, the compiler emits an assembler
 #	code (named "PTX") for a virtual device that represents a class of GPU models
 #	with similar features and/or architecture. In the second stage, such PTX code
 #	is then compiled in order to generate executable code for a particular (real)
 #	device from the former GPU class.
 #
 #	Virtual architectures are named "compute_XY", while real GPU models are denoted
 #	as "sm_XY". In both terms, the two-digits value "XY" represents the Compute
 #	Capability "X.Y".
 #
 #	Note that both architectures must be compatible. That is, PTX code emitted for a
 #	"compute_XY" GPU class can be compiled and executed on a "sm_WZ" device, if and
 #	only if, XY <= WZ. For instance, 'compute_13' is NOT compatible with 'sm_10',
 #	because the former architecture assumes the availability of features not
 #	present on devices of Compute Capability 1.0, but on 1.3 and beyond.
 #
 #	In the following rules, the 'arch=' clause must always be of type "compute_XY",
 #	while the 'code=' clause can either be "compute_XY", "sm_XY", or both. Note
 #	that only target versions specified in the latter clause will be retained in
 #	the output file.
 #
 #	See "GPU Compilation" (currently, chapter 6) in the "CUDA Compiler Driver NVCC"
 #	reference guide, for a detailed description of concepts above. Such document
 #	can be found at folder <CUDA_HOME>/doc/, or at URL:
 #	http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-compilation
##

# Generates PTX and executable code, for the specified "real" device.
# However, only the latter is kept.
gencode_real_template = --generate-code=arch=compute_$(1),code=sm_$(2)

# Emits PTX code, which is embedded into the executable file.
# Compilation will be performed at runtime according to the actual device.
gencode_virtual_template = --generate-code=arch=compute_$(1),code=compute_$(1)


# Calls gencode_real_template() with similar or different values for PTX and executable code.
sm_real_template = $(call gencode_real_template,$(firstword $(1)),$(lastword $(1)))

# Calls gencode_virtual_template(), previously removing the prefix 'PTX' from the argument.
sm_virtual_template = $(call gencode_virtual_template,$(subst PTX,,$(1)))


# Calls sm_real_template for each value in SM_VERSIONS, previously removing any dash.
# Values prefixed with 'PTX' are also removed.
sm_CFLAGS := $(foreach sm_ver,$(filter-out PTX%,$(SM_VERSIONS)),$(call sm_real_template,$(subst -, ,$(sm_ver))))

# Calls sm_virtual_template for each value in SM_VERSIONS prefixed with 'PTX'.
# Non-prefixed values are previously removed.
sm_CFLAGS += $(foreach sm_ver,$(filter PTX%,$(SM_VERSIONS)),$(call sm_virtual_template,$(sm_ver)))


#####


# Double-precision operations are not natively supported on Compute Capability <= 1.2

# Searches for unsupported Compute Capabilities.
unsupported_sm_versions := $(filter 10 11 12,$(subst -, ,$(SM_VERSIONS)))

# Warning message to show.
# NOTE:
#	* Literal tabs and newline characters are ignored. Only '\t' and '\n' are printed.
#	* Only double quotes (") need to be escaped, if any.
warning_unsupported_sm_versions := "\n\
	Warning:\n\
		On Compute Capability 1.2 and lower, all double-precision operations will be\n\
		demoted to single-precision arithmetic.\n"



########################################
# User Makefile parameters.
########################################

# Preprocessor options
CPPFLAGS :=

# Uses single-precision data.
ifeq ($(SINGLE),1)
	CPPFLAGS += -DNMFGPU_SINGLE_PREC=1
else
	# Just in case it was not defined.
	SINGLE := 0
endif

# Uses unsigned integers for matrix dimensions
ifeq ($(UNSIGNED),1)
	CPPFLAGS += -DNMFGPU_UINDEX=1
else
	# Just in case it was not defined.
	UNSIGNED := 0
endif

# Uses faster math functions
ifeq ($(FAST_MATH),1)
	c_only_CFLAGS	:= $(filter-out -O%,$(c_only_CFLAGS)) $(c_only_fast_CFLAGS)

	# Removes "--compiler-options -O[number]" from nvcc_CFLAGS.
	tmp := $(subst --compiler-options$(space)-O,--compiler-options__-O,$(nvcc_CFLAGS))
	nvcc_CFLAGS	:= $(filter-out --compiler-options__-O%,$(tmp))

	nvcc_CFLAGS	+= $(nvcc_fast_CFLAGS)
	export OPENCC_FLAGS := $(OPENCC_FLAGS) $(opencc_fast_CFLAGS)
else
	# Just in case it was not defined.
	FAST_MATH := 0
endif


# Show total elapsed time
ifeq ($(TIME),1)
	CPPFLAGS += -DNMFGPU_PROFILING_GLOBAL=1
else
	# Just in case it was not defined.
	TIME := 0
endif


# Shows time elapsed on data transfers
ifeq ($(TRANSF_TIME),1)
	CPPFLAGS += -DNMFGPU_PROFILING_TRANSF=1
else
	# Just in case it was not defined.
	TRANSF_TIME := 0
endif


# Shows time elapsed on kernel code.
ifeq ($(KERNEL_TIME),1)
	CPPFLAGS += -DNMFGPU_PROFILING_KERNELS=1
else
	# Just in case it was not defined.
	KERNEL_TIME := 0
endif

# Shows time elapsed on MPI communications.
ifeq ($(COMM_TIME),1)
	CPPFLAGS += -DNMFGPU_PROFILING_COMM=1
else
	# Just in case it was not defined.
	COMM_TIME := 0
endif

# Performs SYNCHRONOUS data transfers.
ifeq ($(SYNC_TRANSF),1)
	CPPFLAGS += -DNMFGPU_SYNC_TRANSF=1
else
	# Just in case it was not defined.
	SYNC_TRANSF := 0
endif


# Fixed initial values for W and H. Useful for debugging.
ifeq ($(FIXED_INIT),1)
	CPPFLAGS += -DNMFGPU_FIXED_INIT=1
else
	# Just in case it was not defined.
	FIXED_INIT := 0
endif


# Generates values from the CPU (host) random generator,
# not from the GPU device.
ifeq ($(CPU_RANDOM),1)
	CPPFLAGS += -DNMFGPU_CPU_RANDOM=1
else
	# Just in case it was not defined.
	CPU_RANDOM := 0
endif


# VERBOSE and DEBUG mode (prints A LOT of information)
ifeq ($(DBG),1)
	CPPFLAGS += -DNMFGPU_VERBOSE=1 -DNMFGPU_DEBUG=1

	# Other DEBUG flags

	# Verbosity level
	CPPFLAGS += -DNMFGPU_VERBOSE_2=1

	# # Data transfers
	# CPPFLAGS += -DNMFGPU_DEBUG_TRANSF=1

	# CPPFLAGS += -DNMFGPU_FORCE_BLOCKS=1
	# CPPFLAGS += -DNMFGPU_TEST_BLOCKS=1
	# CPPFLAGS += -DNMFGPU_DEBUG_REDUCT=1

	# # Flags for I/O Debug & testing.
	# CPPFLAGS += -DNMFGPU_DEBUG_READ_FILE=1
	# CPPFLAGS += -DNMFGPU_DEBUG_READ_FILE2=1
	# CPPFLAGS += -DNMFGPU_DEBUG_READ_MATRIX=1
	# CPPFLAGS += -DNMFGPU_DEBUG_READ_MATRIX2=1
	# CPPFLAGS += -DNMFGPU_DEBUG_WRITE_MATRIX=1
	# CPPFLAGS += -DNMFGPU_TESTING=1

	CC_WARN_VERBOSE := 1
	NVCC_WARN_VERBOSE := 1
	PTXAS_WARN_VERBOSE := 1

else
	# Just in case it was not defined.
	DBG := 0
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

		# Just in case it was not defined.
		VERBOSE := 0
	endif
endif

# Shows extra warnings on code compiled by CC, including several "false-positive" warnings.
ifeq ($(CC_WARN_VERBOSE),1)
	c_only_CFLAGS := $(subst -Wno-,-W,$(c_only_CFLAGS))
else
	# Just in case it was not defined.
	CC_WARN_VERBOSE := 0
endif

# Shows extra warnings on code compiled by NVCC, including many "false-positive" warnings.
ifeq ($(NVCC_WARN_VERBOSE),1)
	nvcc_CFLAGS := $(filter-out --disable-warnings,$(subst -Wno-,-W,$(nvcc_CFLAGS)))
	export OPENCC_FLAGS := $(subst -Wno-,-W,$(OPENCC_FLAGS))
	PTXAS_WARN_VERBOSE := 1
else
	# Just in case it was not defined.
	NVCC_WARN_VERBOSE := 0
endif


# Shows kernel compilation warnings
ifeq ($(PTXAS_WARN_VERBOSE),1)
	export PTXAS_FLAGS := $(PTXAS_FLAGS) $(ptxas_warn_CFLAGS)
else
	# Just in case it was not defined.
	PTXAS_WARN_VERBOSE := 0
endif


# Keep intermediate code files.
ifeq ($(KEEP_INTERMEDIATE),1)
	nvcc_CFLAGS += --keep
else
	# Just in case it was not defined.
	KEEP_INTERMEDIATE := 0
endif


########################################
# Source code compilation flags
########################################

# C files
c_OBJS		:= $(addprefix $(objdir)/,$(addsuffix .o,$(c_FILES)))
c_CFLAGS	:= $(c_only_CFLAGS) $(CPPFLAGS)
c_INCLUDES	:= -I$(incdir)
c_LDLIBS	:= -lm
# -lrt

# CUDA files
cuda_device_OBJS := $(addprefix $(objdir)/,$(addsuffix .o,$(cuda_device_FILES)))
cuda_OBJS	 := $(addprefix $(objdir)/,$(addsuffix .o,$(cuda_FILES)))
cuda_CFLAGS	 := $(nvcc_CFLAGS) $(CPPFLAGS)
cuda_INCLUDES	 := --include-path $(incdir) $(addprefix --compiler-options ,$(c_INCLUDES))
cuda_LDLIBS	 := -lcublas -lcurand

# Tools
tools_OBJDIR	:= $(objdir)/$(tools_dir)
tools_BINDIR	:= $(bindir)/$(tools_dirname)
tools_TARGETS	:= $(patsubst $(tools_dir)/%.c,$(tools_BINDIR)/%,$(tools_FILES))
tools_DEPS	:= $(c_OBJS)
tools_CFLAGS	:= $(c_CFLAGS)
tools_INCLUDES	:= $(c_INCLUDES)
tools_LDFLAGS	:= $(common_LDFLAGS)
tools_LDLIBS	:= $(c_LDLIBS)

# Main Program (single-GPU version)
single_gpu_OBJ		:= $(objdir)/$(single_gpu_FILE).o
single_gpu_TARGET	:= $(bindir)/$(basename $(single_gpu_FILE))
single_gpu_DEPS		:= $(c_OBJS) $(cuda_device_OBJS) $(cuda_OBJS)
single_gpu_CFLAGS	:= $(c_CFLAGS)
single_gpu_INCLUDES	:= $(c_INCLUDES) -I$(nvcc_incdir)
single_gpu_LDFLAGS	:= -L$(nvcc_libdir) $(common_LDFLAGS)
single_gpu_LDLIBS	:= $(nvcc_LDLIBS) $(cuda_LDLIBS) $(c_LDLIBS)

# Main Program (multi-GPU version)
multi_gpu_SRC		:= $(srcdir)/$(multi_gpu_FILE)
multi_gpu_OBJ		:= $(objdir)/$(multi_gpu_FILE).o
multi_gpu_TARGET	:= $(bindir)/$(basename $(multi_gpu_FILE))
multi_gpu_DEPS		:= $(c_OBJS) $(cuda_device_OBJS) $(cuda_OBJS)
multi_gpu_CFLAGS	:= $(c_CFLAGS)
multi_gpu_INCLUDES	:= $(c_INCLUDES) -I$(nvcc_incdir)
multi_gpu_LDFLAGS	:= -L$(nvcc_libdir) $(common_LDFLAGS)
multi_gpu_LDLIBS	:= $(nvcc_LDLIBS) $(cuda_LDLIBS) $(c_LDLIBS)


########################################
# Main Compilation Rules
########################################

.SUFFIXES:
.SUFFIXES: .c .c.o .cu .cu.o


# Always keeps all intermediate files (unless the 'clean' target is explicitly requested)
.SECONDARY:
.PRECIOUS:


# Rule to compile all programs.
.PHONY: all All ALL
all All ALL : single_gpu tools


# Main Program (single-GPU version)
.PHONY: single_gpu Single_GPU SINGLE_GPU
single_gpu Single_GPU SINGLE_GPU : $(single_gpu_TARGET) check_sm_versions check_cuda_path


# Main Program (multi-GPU version)
.PHONY: multi_gpu Multi_GPU MULTI_GPU
multi_gpu Multi_GPU MULTI_GPU : $(multi_gpu_TARGET) check_sm_versions check_cuda_path


# Utility programs
.PHONY: tools Tools TOOLS
tools Tools TOOLS : $(tools_TARGETS)


# Main help
.PHONY: help Help HELP
help Help HELP :
	@echo -e $(help_message)


# Help for 'SM_VERSIONS' parameter
.PHONY: help_sm_versions Help_SM_Versions HELP_SM_VERSIONS
help_sm_versions Help_SM_Versions HELP_SM_VERSIONS :
	@echo -e $(help_sm_versions_message)


# Help for utility programs
.PHONY: help_tools Help_Tools HELP_TOOLS
help_tools Help_Tools HELP_TOOLS :
	@echo -e $(help_tools_message)


########################################
# Program-specific rules
########################################

# Main Program (single-GPU version, C++ code)
$(single_gpu_TARGET) : $(single_gpu_DEPS) $(single_gpu_OBJ)
	$(cmd_prefix)mkdir -p $(@D)
	$(cmd_prefix)$(CC) $(single_gpu_CFLAGS) $(single_gpu_INCLUDES) $(single_gpu_LDFLAGS) $^ $(single_gpu_LDLIBS) $(CFLAGS) $(LDFLAGS) -o $@

# Main Program (multi-GPU version, C++ code)
$(multi_gpu_TARGET) : $(multi_gpu_DEPS) $(multi_gpu_OBJ)
	$(cmd_prefix)mkdir -p $(@D)
	$(cmd_prefix)$(MPICC) $(multi_gpu_CFLAGS) $(multi_gpu_INCLUDES) $(multi_gpu_LDFLAGS) $^ $(multi_gpu_LDLIBS) $(CFLAGS) $(LDFLAGS) -o $@

# Tools (C code)
$(tools_BINDIR)/% : $(tools_DEPS) $(tools_OBJDIR)/%.c.o
	$(cmd_prefix)mkdir -p $(@D)
	$(cmd_prefix)$(CC) $(tools_CFLAGS) $(tools_INCLUDES) $(tools_LDFLAGS) $^ $(tools_LDLIBS) $(CFLAGS) $(LDFLAGS) -o $@


# CUDA (HOST and DEVICE) code
$(cuda_device_OBJS) : cuda_CFLAGS+=$(sm_CFLAGS)
$(objdir)/%.cu.o : $(srcdir)/%.cu
	$(cmd_prefix)mkdir -p $(@D)
	$(cmd_prefix)$(NVCC) $(cuda_CFLAGS) $(cuda_INCLUDES) $(addprefix --compiler-options ,$(CXXFLAGS)) $(NVCCFLAGS) --output-file $@ --compile $<


# MPI/C files
$(multi_gpu_OBJ) : $(multi_gpu_SRC)
	$(cmd_prefix)mkdir -p $(@D)
	$(cmd_prefix)$(MPICC) $(multi_gpu_CFLAGS) $(multi_gpu_INCLUDES) $(CFLAGS) $(MPICC_FLAGS) -o $@ -c $<


# C files
$(single_gpu_OBJ) : c_INCLUDES:=$(single_gpu_INCLUDES)
$(objdir)/%.c.o : $(srcdir)/%.c
	$(cmd_prefix)mkdir -p $(@D)
	$(cmd_prefix)$(CC) $(c_CFLAGS) $(c_INCLUDES) $(CFLAGS) -o $@ -c $<


########################################
# Error-checking rules
########################################

# Some targets require "CUDA_HOME" to be set.
.PHONY: check_cuda_path
check_cuda_path :
ifeq ($(CUDA_HOME),)
	@echo -e $(error_cuda_home_not_found) >&2
	@exit 1
endif


# On Compute Capability <= 1.2, double-precision operations are demoted to
# single-precision arithmetic.
.PHONY: check_sm_versions
check_sm_versions :
ifneq ($(SINGLE),1)
ifneq ($(strip $(unsupported_sm_versions)),)
 	@echo -e $(warning_unsupported_sm_versions) >&2
endif
endif


########################################
# Clean-up rules
########################################

# Removes executable and object files, as well as the directory tree.

.PHONY: clobber Clobber CLOBBER
clobber Clobber CLOBBER : # clobber_tools clobber_single_gpu clobber_multi_gpu
	$(cmd_prefix)rm -rf $(bindir)


.PHONY: clobber_single_gpu Clobber_Single_GPU CLOBBER_SINGLE_GPU
clobber_single_gpu Clobber_Single_GPU CLOBBER_SINGLE_GPU : clean_single_gpu
	$(cmd_prefix)rm -f $(single_gpu_TARGET)


.PHONY: clobber_multi_gpu Clobber_Multi_GPU CLOBBER_MULTI_GPU
clobber_multi_gpu Clobber_Multi_GPU CLOBBER_MULTI_GPU : clean_multi_gpu
	$(cmd_prefix)rm -f $(multi_gpu_TARGET)


.PHONY: clobber_tools Clobber_Tools CLOBBER_TOOLS
clobber_tools Clobber_Tools CLOBBER_TOOLS : clean_tools
	$(cmd_prefix)rm -f $(tools_TARGETS)


####################

# Removes object files ONLY.

.PHONY: clean Clean CLEAN
clean Clean CLEAN : # clean_tools clean_single_gpu clean_multi_gpu
	$(cmd_prefix)rm -rf $(objdir)


.PHONY: clean_single_gpu Clean_Single_GPU CLEAN_SINGLE_GPU
clean_single_gpu Clean_Single_GPU CLEAN_SINGLE_GPU : clean_cuda clean_c


.PHONY: clean_multi_gpu Clean_Multi_GPU CLEAN_MULTI_GPU
clean_multi_gpu Clean_Multi_GPU CLEAN_MULTI_GPU : clean_cuda clean_c


.PHONY: clean_tools Clean_Tools CLEAN_TOOLS
clean_tools Clean_Tools CLEAN_TOOLS : clean_c


.PHONY: clean_cuda Clean_Cuda Clean_CUDA CLEAN_CUDA
clean_cuda Clean_Cuda Clean_CUDA CLEAN_CUDA :
	$(cmd_prefix)rm -f $(cuda_OBJS)


.PHONY: clean_c Clean_C CLEAN_C
clean_c Clean_C CLEAN_C :
	$(cmd_prefix)rm -f $(c_OBJS)


########################################
# Help messages
########################################

# NOTE:
#	* Literal tabs and newline characters are ignored. Only '\t' and '\n' are printed.
#	* Double-quote characters ('"') need to be escaped.

# Main help message

help_message := "\n\
 Makefile for NMF-mGPU on UNIX systems.\n\
 \n\
 Targets:\n\
	\n\
	\tall:\t\tDEFAULT. Compiles all programs except the multi-GPU\n\
			\t\t\tversion. It is equivalent to: 'single_gpu tools'.\n\
	\n\
	\tsingle_gpu:\tCompiles NMF-mGPU (single-GPU version).\n\
	\n\
	\tmulti_gpu:\tCompiles bioNMF-mGPU (multi-GPU version).\n\
			\t\t\tTarget NOT compiled by default.\n\
	\n\
	\ttools:\t\tCompiles some utility programs.\n\
			\t\t\tCurrently, this target does NOT require any CUDA-related\n\
			\t\t\tconfiguration or software. In particular, it is NOT\n\
			\t\t\tnecessary to specify 'CUDA_HOME' and/or 'SM_VERSIONS'\n\
			\t\t\tparameters.\n\
	\n\
	\thelp:\t\tPrints this help message.\n\
	\n\
	\thelp_sm_versions:\n\
			\t\t\tPrints a detailed description of valid values for\n\
			\t\t\tthe 'SM_VERSIONS' parameter.\n\
	\n\
	\thelp_tools:\tPrints a short description of available utility programs\n\
	\n\
	\tclobber:\tRemoves the entire binary directory.\n\
	\n\
	\tclobber_single_gpu, clobber_tools:\n\
			\t\t\tRemoves the specified executable and object files.\n\
	\n\
	\tclean:\t\tRemoves all object files (i.e., *.o).\n\
	\n\
	\tclean_single_gpu, clean_tools:\n\
			\t\t\tRemoves the specified object files.\n\
	\n\
	\n\
 Parameters:\n\
	\n\
	\tCUDA_HOME:\tPath to CUDA Toolkit. It may be an environment variable\n\
		\t\t\tor an argument. If not specified, it will be derived by\n\
		\t\t\tlooking for <NVCC> in all folders stored in your \"PATH\"\n\
		\t\t\tenvironment variable. Please note that folder names\n\
		\t\t\tcontaining whitespace characters are NOT supported. In\n\
		\t\t\tthat case, either use a (soft) link, or rename your CUDA\n\
		\t\t\tinstallation directory. This parameter is currently\n\
		\t\t\tignored on the 'tools' target.\n\
	\n\
	\tSM_VERSIONS:\tTarget GPU architecture. This parameter may be an\n\
		\t\t\tenvironment variable or an argument. Device code will be\n\
		\t\t\tgenerated for the specified \"Compute Capability\" (CC).\n\
		\t\t\tFor instance, 'SM_VERSIONS=\"10-13  30  PTX35\"':\n\
		\t\t\t* Generates device-specific executable code for CC 1.3,\n\
		\t\t\t  using only the basic functionality present on CC 1.0.\n\
		\t\t\t* Generates device-specific executable code for CC 3.0,\n\
		\t\t\t  using all available features on such architecture.\n\
		\t\t\t* Emits PTX code for CC 3.5, which can be later\n\
		\t\t\t  compiled and executed by any actual or future device,\n\
		\t\t\t  with a similar or higher Compute Capability.\n\
		\t\t\tTo generate device-specific executable code for CC 2.1,\n\
		\t\t\tplease specify it as: '20-21'.\n\
		\t\t\tSee a more detailed description of this parameter by\n\
		\t\t\texecuting: 'make help_sm_versions'.\n\
		\t\t\tThis parameter is currently ignored on the 'tools'\n\
		\t\t\ttarget.\n\
		\t\t\tDefault value(s): \"$(SM_VERSIONS)\".\n\
	\n\
	\tSINGLE:\t\tIf set to '1', uses single-precision data (ie, 'float').\n\
		\t\t\tElse, uses double-precision data (ie, 'double'). Note,\n\
		\t\t\thowever, that in Compute Capability 1.2 and lower, all\n\
		\t\t\tdouble-precision operations are demoted to single-\n\
		\t\t\tprecision arithmetic. Default value: '$(SINGLE)'.\n\
	\n\
	\tUNSIGNED:\tUses unsigned integers for matrix dimensions, which may\n\
		\t\t\tgenerate faster code. Nevertheless, please note that\n\
		\t\t\tCUBLAS library functions use SIGNED-integer parameters.\n\
		\t\t\tTherefore, matrix dimensions must not overflow such data\n\
		\t\t\ttype. An error message will be shown if this happens.\n\
		\t\t\tDefault value: '$(UNSIGNED)'.\n\
	\n\
	\tFAST_MATH:\tUses less-precise faster math functions. Default: '$(FAST_MATH)'.\n\
	\n\
	\tTIME:\t\tShows total elapsed time. Default value: '$(TIME)'.\n\
	\n\
	\tTRANSF_TIME:\tShows time elapsed on data transfers. Default: '$(TRANSF_TIME)'.\n\
	\n\
	\tKERNEL_TIME:\tShows time elapsed on kernel code. Default value: '$(KERNEL_TIME)'.\n\
	\n\
	\tCOMM_TIME:\tShows time elapsed on MPI communications. Default: '$(COMM_TIME)'.\n\
	\n\
	\tSYNC_TRANSF:\tPerforms SYNCHRONOUS data transfers. Default value: '$(SYNC_TRANSF)'.\n\
	\n\
	\tFIXED_INIT:\tMakes use of \"random\" values from a fixed seed.\n\
		\t\t\tDefault value: '$(FIXED_INIT)'.\n\
	\n\
	\tCPU_RANDOM:\tGenerates random values from the CPU host, not from the\n\
		\t\t\tGPU device. Default value: '$(CPU_RANDOM)'.\n\
	\n\
	\tDBG:\t\tVERBOSE and DEBUG mode (prints A LOT of information).\n\
		\t\t\tIt implies CC_WARN_VERBOSE, NVCC_WARN_VERBOSE, and\n\
		\t\t\tPTXAS_WARN_VERBOSE set to '1'.\n\
		\t\t\tDefault value: '$(DBG)'.\n\
	\n\
	\tVERBOSE:\tCommand-line verbosity level. Valid values are '0'\n\
		\t\t\t(none), '1' (shows make commands), and '2' (shows make\n\
		\t\t\tand internal NVCC commands). Default value: '$(VERBOSE)'.\n\
	\n\
	\tCC_WARN_VERBOSE:\n\
		\t\t\tShows extra warning messages on programs compiled with\n\
		\t\t\tCC. Please note that it may include \"false-positive\"\n\
		\t\t\twarnings. Default value: '$(CC_WARN_VERBOSE)'.\n\
	\n\
	\tNVCC_WARN_VERBOSE:\n\
		\t\t\tShows extra warning messages on programs compiled with\n\
		\t\t\tCC. Please note that it may include \"false-positive\"\n\
		\t\t\twarnings. It implies PTXAS_WARN_VERBOSE set to '1'.\n\
		\t\t\tDefault value: '$(NVCC_WARN_VERBOSE)'.\n\
	\n\
	\tPTXAS_WARN_VERBOSE:\n\
		\t\t\tShows PTX-specific compilation warnings. Default: '$(PTXAS_WARN_VERBOSE)'.\n\
	\n\
	\tKEEP_INTERMEDIATE:\n\
		\t\t\tKeeps temporary files generated by NVCC. Default: '$(KEEP_INTERMEDIATE)'.\n\
	\n\
	\n\
 You can add other compiling options, or overwrite the default flags, using the\n\
	following parameters (they may be environment variables, as well):\n\
	\n\
	\tCC:\t\tCompiler for C-only programs and CUDA host code.\n\
		\t\t\tSupported compilers: 'gcc' and 'clang'.\n\
		\t\t\tDefault value: '$(default_CC)'.\n\
	\n\
	\tNVCC:\t\tCompiler for CUDA device code, and CUDA kernel-related\n\
		\t\t\thost code. Default value: '$(NVCC_basename)'.\n\
	\n\
	\tCFLAGS:\t\tOptions for C-only programs (excludes CUDA code).\n\
		\t\t\tThey are also included in the final linking stage.\n\
	\n\
	\tCXXFLAGS:\tOptions controlling the NVCC's internal compiler for\n\
		\t\t\tCUDA source files. They are automatically prefixed\n\
		\t\t\twith '--compiler-options' in the command line.\n\
	\n\
	\tNVCCFLAGS:\tOptions for the NVCC compiler.\n\
	\n\
	\tLDFLAGS:\tOptions for the linker.\n\
	\n\
	\tOPENCC_FLAGS:\tFlags for the 'nvopencc' compiler, which generates PTX\n\
		\t\t\t(intermediate) code on devices of Compute Capability 1.x\n\
	\n\
	\tPTXAS_FLAGS:\tFlags for PTX code compilation, which generates the\n\
		\t\t\tactual GPU assembler.\n\
	\n\
	\tMPICC:\t\tCompiler for MPI code. Default value: '$(default_MPICC)'.\n\
	\n\
	\tMPICC_FLAGS:\tOptions for MPI code. They are also included in the\n\
		\t\t\tfinal linking stage.\n"

####################

# Help message for SM_VERSIONS.

help_sm_versions_message := "\n\
 The SM_VERSIONS parameter:\n\
 \n\
 Device code is compiled in two stages. First, the compiler emits an assembler\n\
 code (named \"PTX\") for a virtual device that represents a class of GPU models\n\
 with similar features and/or architecture. In the second stage, such PTX code\n\
 is then compiled in order to generate executable code for a particular (real)\n\
 device from the former GPU class.\n\
 \n\
 Virtual architectures are named \"compute_XY\", while real GPU models are denoted\n\
 as \"sm_XY\". In both terms, the two-digits value \"XY\" represents the Compute\n\
 Capability \"X.Y\".\n\
 \n\
 Note that both architectures must be compatible. That is, PTX code emitted for a\n\
 \"compute_XY\" GPU class can be compiled and executed on a \"sm_WZ\" device, if and\n\
 only if, XY <= WZ. For instance, 'compute_13' is NOT compatible with 'sm_10',\n\
 because the former architecture assumes the availability of features not\n\
 present on devices of Compute Capability 1.0, but on 1.3 and beyond.\n\
 \n\
 \n\
 For a detailed description of concepts above, please see chapter \"GPU\n\
 Compilation\" in the \"CUDA Compiler Driver NVCC\" reference guide, which can be\n\
 found at folder <CUDA_HOME>/doc, or at URL\n\
 http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html\n\
 \n\
 \n\
 There are three ways to specify target architecture(s) in the SM_VERSIONS\n\
 parameter:\n\
 \n\
 1) Device-specific features & code (i.e., PTX and executable code, with SIMILAR\n\
	" "  Compute Capability):\n\
	\n\
	" "  Emits PTX assembler code, which is then compiled into executable\n\
	" "  instructions just for the given \"Compute Capability(-ies)\" (CC). Since the\n\
	" "  former is just an intermediate code (i.e., it is NOT retained in the output\n\
	" "  file), and the latter is generated with a device-specific binary format, the\n\
	" "  program may not be compatible with other GPU architectures. That is, any\n\
	" "  given CC value, \"XY\", is translated to the following NVCC option:\n\
	" "  \t\"--generate-code=arch=compute_XY,code=sm_XY\".\n\
	\n\
	" "  For instance, 'SM_VERSIONS=\"13  35\"' generates executable code just for\n\
	" "  devices of Compute Capability 1.3 and 3.5, respectively. GPU devices with\n\
	" "  other CC values, such as 1.1 or 2.0, may not be able to execute the program.\n\
	\n\
	" "  NOTE:\n\
	" "  For devices, such as 2.1, that do not have a similar PTX CC number, please\n\
	" "  specify the nearest lower value (\"2.0\", for the previous example) by using\n\
	" "  the dashed-separated form below.\n\
	\n\
	\n\
 2) Generic features, device-specific code (i.e., PTX and executable code, with\n\
	" "  DIFFERENT Compute Capabilities):\n\
	\n\
	" "  This is a generalization of the previous form. Here, different \"Compute\n\
	" "  Capabilities\" (CC) can be specified for both, PTX and executable code,\n\
	" "  separated by a dash.\n\
	" "  That is,\n\
			\t\t\"XY-WZ\"   (with XY <= WZ),\n\
	\n\
	" "  Emits PTX assembler code for CC \"X.Y\", which is then compiled into\n\
	" "  executable instructions for a device of CC \"W.Z\". The former, determines the\n\
	" "  target ARCHITECTURE (i.e., the available hardware/ software features and\n\
	" "  functionality). The latter, specifies the target DEVICE in terms of binary\n\
	" "  code format.\n\
	\n\
	" "  Similarly as above, no PTX code is embedded in the output file, so the\n\
	" "  program may not be compatible with other GPU device models. That is, the\n\
	" "  previous expression is translated to the following NVCC option:\n\
	" "  \t\"--generate-code=arch=compute_XY,code=sm_WZ\".\n\
	\n\
	" "  Note that \"XY-XY\" is equivalent to just specify \"XY\" as in the previous form\n\
	" "  On the other hand, if XY < WZ, the program is still compiled for the target\n\
	" "  device (i.e. CC \"W.Z\"), but it will only make use of features available on\n\
	" "  CC \"X.Y\", discarding any functionality introduced since.\n\
	\n\
	" "  NOTE:\n\
	" "  As stated above, please use this form to specify target devices, such as\n\
	" "  CC 2.1, that do not have a similar PTX CC number (so, a lower value must\n\
	" "  also be biven). Example: '20-21'.\n\
	\n\
	" "  For instance, 'SM_VERSIONS=\"10-13  20-21\"':\n\
		\n\
		\t* Generates executable code for a device of CC 1.3, with the basic\n\
		\t  features that are available on CC 1.0. In particular, it discards all\n\
		\t  support for double-precision floating-point data, introduced on CC 1.3\n\
		\t* Compiles the algorithm with the features and functionality available\n\
		\t  on CC 2.0, and generates a binary image for a device of CC 2.1.\n\
		\t* Since no PTX code is retained in the output file, the program may not\n\
		\t  compatible with other GPU devices (e.g., CC 3.0).\n\
	\n\
	\n\
 3) Generic features & \"code\":\n\
	\n\
	" "  Emits PTX assembler code for the given \"Compute Capability\" (CC), and\n\
	" "  embeds it into the output file. No executable code is generated. Instead,\n\
	" "  the former is dynamically compiled at runtime according to the actual GPU\n\
	" "  device. Such process is known as \"Just In Time\" (JIT) compilation.\n\
	\n\
	" "  To specify a target architecture in a such way, please use the word 'PTX'\n\
	" "  followed by the target Compute Capability.\n\
	" "  That is,\n\
			\t\t\"PTXwz\"\n\
	\n\
	" "  generates PTX code for Compute Capability \"w.z\", and embeds it into the\n\
	" "  output file. Such code can be later compiled and executed on any device,\n\
	" "  with a similar or greater CC value. Similarly as previous forms, the\n\
	" "  expression above is translated to the following NVCC option:\n\
	" "  \"--generate-code=arch=compute_wz,code=compute_wz\".\n\
	\n\
	" "  Note, however, that JIT compilation increases the startup delay. In\n\
	" "  addition, the final executable code will use just those architectural\n\
	" "  features that are available on CC \"w.z\", discarding any functionality\n\
	" "  introduced since.\n\
	\n\
	" "  For instance, 'SM_VERSIONS=\"PTX10  PTX35\"':\n\
		\n\
		\t* Emits PTX code for the first CUDA-capable architecture (ie., CC 1.0).\n\
		\t  Therefore, the program can be later dynamically compiled and executed\n\
		\t  on ANY current or future GPU device. Nevertheless, it will only use\n\
		\t  the (very) basic features present on such architecture.\n\
		\t* Generates PTX code that can be later compiled and executed on devices\n\
		\t  of CC 3.5, or higher.\n\
		\t* Any device prior to CC 3.5 (e.g., 1.3, 2.1, or 3.0), will execute the\n\
		\t  basic CC 1.0 version.\n\
	\n\
	\n\
 This parameter is currently ignored on the 'tools' target.\n\
 \n\
 \n\
 Current default value(s):\n\
 \n\
	\t\"$(subst $(space),  ,$(SM_VERSIONS))\",\n\
 \n\
 which will be translated into the following argument(s) for NVCC:\n\
 \n\
	\t$(subst $(space),\n\t,$(sm_CFLAGS))\n"


####################

# Help message for tools.

help_tools_message := "\n\
 Tool programs:\n\
 \n\
 In addition to \"NMF-mGPU\", there are some utility programs to make easier\n\
 working with input files. It includes a program for binary-text file conversion\n\
 and another to generate input matrices with random data (useful for testing).\n\
 \n\
 List of generated files:\n\
	\t$(subst $(space),\n\t,$(tools_TARGETS))\n\
 \n\
 Please note that such programs do NOT make use of the GPU device. They are\n\
 implemented in pure-C99 language, and all operations are performed on the HOST\n\
 (i.e., the CPU). Therefore, they do NOT require any CUDA-related option,\n\
 configuration or software. In particular, it is NOT necessary to specify the\n\
 'CUDA_HOME' and/or 'SM_VERSIONS' parameters.\n\
 \n\
 1) Binary-text file converter:\n\
	" "  Since \"NMF-mGPU\" accepts input matrices stored in a binary or ASCII-text\n\
	" "  file, this program allows file conversion between both formats.\n\
	" "  For binary files, there are two sub-formats: \"native\" and non-\"native\".\n\
		\n\
		\t* \"Native\" mode refers to \"raw\" I/O. That is, data are stored/loaded\n\
		\t  with the precision specified at compilation time: 'float' if 'SINGLE'\n\
		\t  was set to '1', or 'double' otherwise. Matrix dimensions are stored/\n\
		\t  loaded in a similar way (i.e., 'unsigned int', if 'UNSIGNED' was set\n\
		\t  to '1'; '[signed] int', otherwise). This mode is faster because NO\n\
		\t  error checking is performed. Therefore, it should NOT be used to read\n\
		\t  untrusted input files.\n\
		\n\
		\t* In Non-\"Native\" mode, data are ALWAYS stored/loaded using double\n\
		\t  precision (and unsigned integers for matrix dimensions), regardless\n\
		\t  the options specified at compilation. This is the recommended mode for\n\
		\t  input or final output data, since every datum is checked for invalid\n\
		\t  format.\n\
		\n\
 2) Matrix generator:\n\
	" "  This program generates a valid data matrix with random values. You can\n\
	" "  select output matrix dimensions, as well as the maximum value for matrix\n\
	" "  data. The output matrix can be written as ASCII-text, or in a binary file\n\
	" "  (in any of the binary modes described above).\n"


########################################

