<!--
 ************************************************************************
 *
 * BioNMF-GPU 2.0 -- Non-negative Matrix Factorization on (multi-)GPU systems.
 *
 * Copyright (C) 2011-2014:
 *
 *		Edgardo Mejia-Roa(*), Carlos Garcia(*), Jose Ignacio Gomez(*),
 *		Manuel Prieto(*), Francisco Tirado(*) and Alberto Pascual-Montano(**).
 *
 *		(*)  ArTeCS Group, Complutense University of Madrid (UCM), Spain.
 *		(**) Functional Bioinformatics Group, Biocomputing Unit,
 *		     National Center for Biotechnology-CSIC, Madrid, Spain.
 *
 *		E-mail for E. Mejia-Roa: <edgardomejia@fis.ucm.es>
 *		E-mail for A. Pascual-Montano: <pascual@cnb.csic.es>
 *
 *
 * This file is part of bioNMF-GPU.
 *
 * BioNMF-GPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * BioNMF-GPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with BioNMF-GPU. If not, see <http://www.gnu.org/licenses/>.
 *
 ************************************************************************
-->
<!-- ==================================================== -->
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
 <html lang="en" xml:lang="en" xmlns="http://www.w3.org/1999/xhtml">
 <head>
   <meta name="application-name" content="BioNMF-GPU"/>
   <meta name="author" content="Edgardo Mejia-Roa (edgardomejia@fis.ucm.es), Carlos Garcia, Jose Ignacio Gomez, Manuel Prieto, Francisco Tirado, and Alberto Pascual-Montano (pascual@cnb.csic.es)."/>
   <meta name="description" content="Non-negative Matrix Factorization (NMF) on (multi-)GPU systems, for Biology. Installation guide"/>
   <meta name="keywords" content="bioNMF, NMF, Matrix factorization, GPU, multi-GPU, GPGPU, NVIDIA, CUDA, CUBLAS, Bioinformatics"/>
   <meta name="language" content="en"/>
   <meta name="copyright" content="(C) 2011-2014 Edgardo Mejia-Roa (edgardomejia@fis.ucm.es). ArTeCS Group, Complutense University of Madrid (UCM), Spain."/>
   <meta http-equiv="content-Type" content="text/html; charset=UTF-8"/>
   <meta http-equiv="last-modified" content="2014/04/30" scheme="YYYY/MM/DD"/>
   <link rel="stylesheet" type="text/css" href="styles.css"/>
   <title>BioNMF-GPU Installation Guide</title>
 </head>
 <body>

<!-- ==================================================== -->

# *BioNMF-GPU* INSTALLATION GUIDE

 This document shows how to install and compile *bioNMF-GPU*.

 **Index:**

   1. Introduction.
   2. System requirements.
   3. Directory structure.
   4. Compiling *bioNMF-GPU*.
   5. Utility programs.
   6. *BioNMF-GPU* execution setup.
   7. Testing *bioNMF-GPU*.
   8. Issues/troubleshooting.
   9. How cite *bioNMF-GPU*.


*****************************


## 1. Introduction

 ***BioNMF-GPU*** implements the ***Non-negative Matrix Factorization*** (***NMF***) algorithm by making use of a ***Graphics-Processing Unit*** (***GPU***). NMF takes an input matrix (**V**) and returns two matrices, **W** and **H**, whose product is equal to the former (i.e., **V** ~ **W** \* **H**). If **V** has *n* rows and *m* columns, then dimensions for **W** and **H**, will be *n* × *k* and *k* × *m*, respectively. The *factorization rank* (*"k"*) specified by the user, is usually a value much less than *n* and *m*.

 This GPU implementation of the NMF algorithm has been developed using the NVIDIA's ***Compute Unified Device Architecture*** (***CUDA***) programming model. *CUDA* represents the GPU as a programmable *co-processor*, which will be then responsible of computing all required algebraic operations.

 *BioNMF-GPU* is able to process matrices of any size. Even on detached devices with a dedicated (and limited) on-board memory, the NMF algorithm can be executed on large datasets by blockwise transferring and processing the input matrix.

 Finally, this software can make use of multiple GPU devices through ***MPI*** (***Message-Passing Interface***).

 Please, see our *User guide* for implementation details, as well as a description of software usage.


*****************************


## 2. System requirements


### 2.1. Linux / Mac OS X:

* `CUDA Toolkit`, version 4.2 or greater, freely available at <https://developer.nvidia.com/cuda-downloads/>.
* Any of the following compilers:
	 + `GNU C/C++ Compiler` (`gcc`).
	 + `Clang C/C++ front-end for LLVM`.
* `GNU Make`.

On Mac OS X, the `Clang` compiler can be installed from [`Xcode`](https://developer.apple.com/technologies/tools/). It will be used by default, in the compilation process. Please read the [CUDA Release Notes][RN] for any `clang`/`Xcode` -related issue.

An exhaustive list of requirements, as well as detailed installation instructions, can be found on the *"Getting Starting"* guides for [Linux](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/index.html) and [Mac OS X](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-mac-os-x/index.html).


**Important notes:**
   * Folder names containing whitespace characters are *NOT* supported by *BioNMF-GPU*. In that case, either use a (soft) link, or rename your CUDA installation directory.
   * `Clang` compiler is **not** supported on 32-bits systems. Please, read the [CUDA Release Notes][RN] for details.

[RN]: <http://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html> "CUDA Release Notes"

<!-- ==================== -->

### 2.2. Microsoft Windows:

To compile *bioNMF-GPU* on this platform, you will need the following software:

   * `NVIDIA CUDA Toolkit`, version 4.2 or greater, freely available at <https://developer.nvidia.com/cuda-downloads/>.
   * [`Microsoft Visual Studio`](http://www.microsoft.com/visualstudio/), versions 2008, 2010 or 2012.  
	 In order to install all features, the CUDA Toolkit requires the **full version**, not just the *Express edition*.

See detailed installation instructions on the [Getting Starting guide for Windows][GSGW]. After installing the software, you can use the project template files supplied with the Toolkit to create a new project and import the *bioNMF-GPU*'s source files. There are step-by-step instructions on *Chapter 2* of the [CUDA Samples Reference Manual](http://docs.nvidia.com/cuda/cuda-samples/index.html).

[GSGW]: <http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-microsoft-windows/index.html> "Getting Starting guide for Windows"

<!--
#### Alternative install process:

If you prefer a UNIX-like compilation process (i.e., using the `Makefile` we provide, in a command-line environment), or you do not have access to a *full version* of Visual Studio, you can perform the steps below. Note, however, that `nvcc.exe` (the CUDA Compiler) still requires the Microsoft Visual Studio compiler (`cl.exe`), but it can be found on any (old and/or free) `Visual C++ Express edition`.

   1. Download the [`CUDA Toolkit`](https://developer.nvidia.com/cuda-downloads/) and uncompress the file.  
	  It is highly recommended to *previously* move the downloaded file to an empty folder, since it does not contain a *"root"* directory, but multiple files and folders. The installation package can be extracted using a decompression tool which supports the LZMA compression method, such as [`7-zip`](http://www.7-zip.org) or [`WinZip`](http://www.winzip.com).

   2. Execute the file `setup.exe` and install the display driver *only*.

   3. Check your GPU device and the driver by executing any of the CUDA sample programs (e.g., `CUDASamples\Bin\win32\Release\deviceQuery.exe`). They are *statically* linked and do not require any library. You can find some screenshots on the [Getting Starting Guide for Windows][GSGW] (there is a copy of such document in the `CUDADocumentation\` folder).

	  From this point, the only folder to keep is `CUDAToolkit\`, which can be moved to any desired location. Everything else from the extracted file can be safety deleted, unless you want to also keep the documentation and/or the sample programs (folders `CUDADocumentation\` and `CUDASamples\`, respectively).

   4. Download and install [`Cygwin`](http://cygwin.com/index.html), which contains a set of GNU tools. **Please include the shell interface it provides**. It is not necessary to install the GNU compiler (`gcc`), since it will not be used by the CUDA compiler.  
	  **NOTE:** folder names containing whitespace characters are *NOT* supported by *BioNMF-GPU*. In that case, either use a (soft) link, or rename your CUDA installation directory.

   5. Download and install any old and/or free `MS Visual C++ Express edition`, such as the 2005, 2008 or 2010 version. You don't need to setup the graphical environment.

   6. Edit the `Makefile` we provide, go to the *"Compiler options"* section, and uncomment the `nvcc` flag '`--driver-prefix`'. You might need to also uncomment and adjust the flag '`--compiler-bindir`' in order to specify the path to `cl.exe` (the MS Visual C++ compiler).  
	  See a detailed description of these flags on the *"CUDA Compiler Driver NVCC" reference guide*, which can be found at folder `CUDADocumentation/` or at URL <http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>.

Now, you can follow the rest of this installation guide as if you were on a UNIX system.
-->

*****************************


## 3. Directory structure

This section lists the files and folders you should find after downloading and decompressing *BioNMF-GPU*.

### 3.1. Files and folders

After extraction of the compressed file, you should see the following folders:

   * `doc/`		─ Installation and User guides.
   * `src/`		─ Source code.
   * `include/`	─ Header files.
   * `test/`	─ Examples of a valid input files.


In addition, there are some important files that must be customized *before* the compilation process:

   * `Makefile`	─ Compilation directives.
   * `env.sh`	─ Required environment variables (see chapter 6.1 *Execution environment*).

<!-- ==================== -->

### 3.2. Source-code directory structure

The `src/` folder is organized as follow:

   * Main-program files (`C99` code):
	 + `bioNMF-GPU.c`					─ Main program (single-GPU version).

   * CUDA files (compiled as `C++` code):
	 + `NMF_routines.cu`				─ NMF-related operations.
	 + `matrix/matrix_operations.cu`	─ Algebraic operations and data transfers on matrices.
	 + `GPU_kernels.cu`					─ CUDA kernels (i.e., device code).
	 + `GPU_setup.cu`					─ GPU set-up and management routines.
	 + `timing.cu`						─ Timing and profiling routines.

   * `C` files (`C99` code):
	 + `matrix/matrix_io.c`				─ Functions to read/write data matrices.
	 + `matrix/matrix_io_routines.c`	─ Auxiliary I/O routines.

   * Utility programs (`C99` code; see section *5 "Utility programs"*):
	 + `matrix/tools/file_converter.c`	─ ASCII-Binary file conversion.
	 + `matrix/tools/generate_matrix.c`	─ Generates a matrix with random values in text or binary format.


The `include/` folder is similarly organized:

   * `real_type.h`						─ Floating-point data definition.
   * `index_type.h`						─ Data type used for array index.
   * `nmf_routines.cuh`
   * `matrix_operations.cuh`
   * `GPU_kernels.cuh`
   * `GPU_setup.cuh`
   * `timing.cuh`
   * `matrix/matrix_io.h`
   * `matrix/matrix_io_routines.h`


*****************************


## 4. Compiling *bioNMF-GPU*

 This section describes all supported options provided by our `Makefile`, which allow to customize the compilation process.

 Note: All the information contained in the following subsections can be printed on the screen by executing:

		$> cd <BioNMF_PATH>
		$> make help


### 4.1. `Makefile` targets:

 List of (main) available actions:

   * `all`: Compiles all programs (*default* target if none specified).

   * `single_gpu`: Compiles *bioNMF-GPU* (single-GPU version) only.

   * `multi_gpu`: Compiles *bioNMF-mGPU* (multi-GPU version) only.

   * `tools`: Compiles some utility programs (see section *5 "Utility programs"* for details).  
	 Currently, this target does *not* require any CUDA-related configuration or software. In particular, it is *not* necessary to specify  `CUDA_HOME` or `SM_VERSIONS` parameters.

   * `help`: Prints a help message with all this information.

   * `help_sm_versions`: Prints a detailed description of valid values for the `SM_VERSIONS` parameter.

   * `help_tools`: Prints a short description of available utility programs (see section *5 "Utility programs"*).

   * `clobber`: Removes the entire binary directory, with all executable and object files.

   * `clobber_single_gpu`, `clobber_tools`: Removes the specified executable and its associated object files.

   * `clean`: Removes all object files (i.e., `*.o`), keeping any executable program.

   * `clean_single_gpu`, `clean_tools`: Removes all object files associated to the specified program.

<!-- ==================== -->

### 4.2. `Makefile` parameters

The compilation process can be customized with the following parameters:

   * `CUDA_HOME`: Path to your CUDA Toolkit.  
	 It may be an environment variable or an argument.  
	 If not specified, it will be derived by looking for `<NVCC>` in all folders stored in your `PATH` environment variable.  
	 Please note that **folder names containing whitespace characters are NOT supported**. In that case, either use a (soft) link, or rename your CUDA installation directory.  
	 This parameter is currently ignored on the `tools` target.

   * `SM_VERSIONS`: Target GPU architecture(s).  
	 This parameter may be an environment variable or an argument.  
	 Device code will be generated for the specified *Compute Capability(-ies)* (CC).  
	 For instance, '`SM_VERSIONS="10-13  30  PTX35"`':
	   + Generates device-specific executable code for CC **1.3**, using only the basic functionality present on CC **1.0**.
	   + Generates device-specific executable code for CC **3.0**, using all available features on such architecture.
	   + Emits `PTX` code for CC **3.5**, which can be later dynamically compiled and executed by any current or future device, with a similar or higher Compute Capability.

	To generate device-specific executable code for CC **2.1**, please specify it as: '`20-21`'.  
	See a more detailed description of this parameter below, or by executing: '`make help_sm_versions`'.  
	This parameter is currently ignored on the `tools` target.  
	Default value(s): `"10  13  20  30  20-21  PTX35"`.

   * `SINGLE`: If set to '`1`', uses single-precision data (i.e., '`float`'). Else, uses double-precision data (i.e., '`double`').  
	 Note, however, that in Compute Capability 1.2 and lower, all double-precision operations are *demoted* to single-precision arithmetic.  
	 Default value: '`1`'.

   * `UNSIGNED`: Uses *unsigned* integers for matrix dimensions, which may generate faster code. Nevertheless, please note that *CUBLAS* library functions use *signed*-integer parameters. Therefore, matrix dimensions must *not* overflow such data type. An error message will be shown if this happens.  
	 Default value: '`1`'.

   * `FAST_MATH`: Uses less-precise faster math functions.  
	 Default value: '`1`'.

   * `TIME`: Shows total elapsed time.  
	 Default value: '`1`'.

   * `TRANSF_TIME`: Shows time elapsed on data transfers.  
	 Default value: '`0`'.

   * `KERNEL_TIME`: Shows time elapsed on kernel code.  
	 Default value: '`0`'.

   * `COMM_TIME`: Shows time elapsed on MPI communications.  
	 Default value: '`0`'.

   * `SYNC_TRANSF`: Performs *synchronous* data transfers.  
	 Default value: '`0`'.

   * `FIXED_INIT`: Makes use of "random" values from a fixed seed.  
	 Default value: '`0`'.

   * `CPU_RANDOM`: Generates random values from the CPU host, not from the GPU device.  
	 Default value: '`0`'.

   * `DBG`: *Verbose* and *Debug* mode (prints *a lot* of information). It implies `CC_WARN_VERBOSE`, `NVCC_WARN_VERBOSE`, and `PTXAS_WARN_VERBOSE` set to '`1`'.  
	 Default value: '`0`'.

   * `VERBOSE`: Command-line verbosity level. Valid values are '`0`' (none), '`1`' (shows `make` commands), and '`2`' (shows `make` and internal `NVCC` commands).  
	 Default value: '`0`'.

   * `CC_WARN_VERBOSE`: Shows extra warning messages on programs compiled with `CC`. Please note that it may include "*false-positive*" warnings.  
	 Default value: '`0`'.

   * `NVCC_WARN_VERBOSE`: Shows extra warning messages on programs compiled with `NVCC`. Please note that it may include "*false-positive*" warnings. It implies `PTXAS_WARN_VERBOSE` set to '`1`'.  
	 Default value: '`0`'.

   * `PTXAS_WARN_VERBOSE`: Shows `PTX`-specific compilation warnings.  
	 Default value: '`0`'.

   * `KEEP_INTERMEDIATE`: Keeps temporary files generated by `NVCC`.  
	 Default value: '`0`'.


You can add other compiling options or overwrite the default flags, with the following parameters (which may be environment variables, as well):

   * `CC`: Compiler for C-only programs and CUDA host code.
	 Supported compilers: '`gcc`' and '`clang`'.
	 Default value: '`gcc`'

   * `NVCC`: Compiler for CUDA device code, and CUDA kernel-related host code.
	 Default value: '`nvcc`'

   * `CFLAGS`: Options for `C`-only programs (excludes `CUDA` code).  
	 They are also included in the final linking stage.

   * `CXXFLAGS`: Options controlling the `NVCC`'s internal compiler for `CUDA` source files.  
	 They are automatically prefixed with '`--compiler-options`' in the command line.

   * `NVCCFLAGS`: Options for the `NVCC` compiler.

   * `LDFLAGS`: Options for the linker.

   * `OPENCC_FLAGS`: Flags for the `nvopencc` compiler, which generates `PTX` (intermediate) code on devices of Compute Capability *"1.x"*.

   * `PTXAS_FLAGS`: Flags for `PTX` code compilation, which generates the actual GPU assembler.

   * `MPICC`: Compiler for `MPI` code.
	  Default value: '`mpicc`'

   * `MPICC_FLAGS`: Options for `MPI` code.  
	  They are also included in the final linking stage.


#### The `SM_VERSIONS` parameter:

Device code is compiled in two stages. First, the compiler emits an assembler code (named `PTX`) for a *virtual* device that represents a *class* of GPU models with similar features and/or functionality. In the second stage, such `PTX` code is then compiled in order to generate executable code for a particular (*real*) device from the former GPU class.

*Virtual* architectures are named "`compute_`*XY*", while *real* GPU models are denoted as "`sm_`*XY*". In both terms, the value *"XY"* represents the *Compute Capability* (CC) *"X.Y"*.

Note that both architectures must be compatible. That is, `PTX` code emitted for a `compute_`*XY* GPU class can be compiled and executed on a `sm_`*WZ* device, if and only if, *X.Y* <= *W.Z*. For instance, '`compute_13`' is *not* compatible with '`sm_10`', because the former architecture assumes the availability of features that are not present on devices of CC **1.0**, only on **1.3** and beyond.

For a detailed description of concepts above, please see chapter *GPU Compilation* in the *"CUDA Compiler Driver NVCC" reference guide*, which can be found at folder `<CUDA_HOME>/doc/`, or at URL <http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-compilation>.


There are three ways to specify target architecture(s) in the `SM_VERSIONS` parameter:

   * **Device-specific features & code** (i.e., `PTX` and executable code with **similar** Compute Capability):
	 Emits `PTX` assembler code, which is then compiled into executable instructions, just for the given *Compute Capability(-ies)* (CC). Since the former is just an intermediate code (i.e., it is *not* retained in the output file), and the latter is generated with a device-specific binary format, the program may not be compatible with other GPU architectures. That is, any given CC value, *"XY"*, is translated to the following `nvcc` option: "`--generate-code=arch=compute_`*XY*`,code=sm_`*XY*".

	 For instance, '`SM_VERSIONS="13  35"`' generates executable code just for devices of Compute Capability **1.3** and **3.5**, respectively. GPU devices with other CC values, such as 1.1 or 2.0, may not be able to execute the program.

	 *Note:* For devices, such as CC **2.1**, that do not have a similar `PTX` CC number, please specify the nearest lower value (***"2.0"***, for the previous example) by using the dashed-separated form below.


   * **Generic features, device-specific code** (i.e., `PTX` and executable code, with **different** Compute Capabilities):
	 This is a generalization of the previous form. Here, different *Compute Capabilities* (CC) values can be specified for both, `PTX` and executable code, separated by a dash.  
	 That is,
	   > "*XY*`-`*WZ*"  (with *XY* <= *WZ*),

	 emits `PTX` assembler code for CC *"X.Y"*, which is then compiled into executable instructions for a device of CC *"W.Z"*. The former, determines the target *architecture* (i.e., the available hardware/software features and functionality). The latter, specifies the target *device* in terms of binary code format.

	 Similarly as above, no `PTX` code is embedded in the output file, so the program may not be compatible with other GPU device models. That is, the previous expression is translated to the `NVCC` option: "`--generate-code=arch=compute_`*XY*`,code=sm_`*WZ*"

	 Note that "*XY*`-`*XY*" is equivalent to just specify *"XY"* as in the previous form. On the other hand, if *XY* < *WZ*, the program is still compiled for the target device (i.e., CC *"W.Z"*), but it will only make use of features available on CC *"X.Y"*, discarding any functionality introduced since.

	 *Note:* As stated above, please use this form to specify target devices, such as CC **2.1**, that do not have a similar `PTX` CC number (so, a lower value must also be given). Example: '`20-21`'.

	 For instance, '`SM_VERSIONS="10-13  20-21"`':
	   + Generates executable code for a device of CC **1.3**, with the basic features that are available on CC **1.0**. In particular, it *discards* all support for double-precision floating-point data, introduced on CC **1.3**.
	   + Compiles the algorithm with the features and functionality available on CC **2.0**, and generates a binary image for a device of CC **2.1**.
	   + Since no `PTX` code is retained in the output file, the program may not compatible with other GPU devices (e.g., CC 3.0).


   * **Generic features & *"code"***:

	 Emits `PTX` assembler code for the given *Compute Capability* (CC), which is then embedded into the output file. No executable code is generated. Instead, the former is dynamically compiled at runtime according to the actual GPU device. Such process is known as *Just In Time* (JIT) compilation.

	 To specify a target architecture in a such way, please use the word '`PTX`' followed by the target Compute Capability.  
	 That is,
	   > "`PTX`*wz*",

	 generates `PTX` code for Compute Capability *"w.z"*, and embeds it into the output file. Such code can be later compiled and executed on any device, with a similar or greater CC value. Similarly as previous forms, the expression above is translated to the following `nvcc` option: "`--generate-code=arch=compute_`*wz*`,code=compute_`*wz*".

	 Note, however, that JIT compilation increases the startup delay. In addition, the final executable code will use just those architectural features that are available on CC *"w.z"*, discarding any functionality introduced since.

	 For instance, '`SM_VERSIONS="PTX10  PTX35"`':
	   + Emits `PTX` code for the first CUDA-capable architecture (i.e., CC **1.0**). Therefore, the program can be later dynamically compiled and executed on *any* current or future GPU device. Nevertheless, it will only use the (very) basic features present on such architecture.
	   + Generates `PTX` code that can be later compiled and executed on devices of CC **3.5**, or higher.
	   + Any device prior to CC **3.5** (e.g., 1.3, 2.1, or 3.0), will execute the basic CC **1.0** version.


This parameter is currently ignored on the '`tools`' target.


Current default value(s):

		"10  13  20  30  35  20-21  PTX35"

which will be translated into the following argument(s) for `NVCC`:

		--generate-code=arch=compute_10,code=sm_10
		--generate-code=arch=compute_13,code=sm_13
		--generate-code=arch=compute_20,code=sm_20
		--generate-code=arch=compute_30,code=sm_30
		--generate-code=arch=compute_35,code=sm_35
		--generate-code=arch=compute_20,code=sm_21
		--generate-code=arch=compute_35,code=compute_35


<!-- ==================== -->

### 4.3. Other compiling options

 The compilation process can be further customized by editing section *"Compiler Options"* in `Makefile`. There, you can adjust `CC`-specific flags, and/or options for the `NVCC` compiler, as well as its internal tools (e.g., `nvopencc`, `ptx`, etc).

#### `CC` options:
 Since host (i.e., non-device) code in `CUDA` source files is internally compiled by `NVCC` as `C++`, options for `CC` are distributed into two groups: common- `C`/`C++` flags, and options for `C`-only source files. Within each group, some flags may, or may not be applied according to `Makefile` input parameters. For instance, '`common_fast_CFLAGS`' and '`c_only_fast_CFLAGS`', specify which flags are applied when the input parameter `FAST_MATH` is set to '`1`'.

 *Notes:*

   * Further options may apply according to the selected `CC` compiler (`gcc` or `clang`) and the current OS (`darwin` or `linux`).  
   * `C`-only source files require support for `ISO-C99` standard. Therefore, flags for `C`-only code include the option '`-std=c99`'.  
   * Some (optional) useful features for processing single-precision floating-point data are only available when '`_GNU_SOURCE`' is defined.
     Such option is included for `C`-only source files when the input parameter `SINGLE` is set to '`1`'.  


#### `NVCC` options:
 Variables containing flags for `NVCC`, follow a similar naming scheme as for `CC` options. Most of options for `NVCC` are those specified as common `C`/`C++` flags, prefixed with '`--compiler-options`'.

 **Note for Windows CygWin/Mingw users**: It may be necessary to uncomment the '`--drive-prefix`' option in order to correctly handle filenames and paths in Windows native format.

 In this section of `Makefile`, you can also find options for `nvopencc`. This tool is internally used by `nvcc` to generate `PTX` assembler code on devices of Compute Capability *1.x*. For newer devices, such options are ignored. Finally, there are flags to control `PTX` code compilation.

 See a detailed description of these flags on the *"CUDA Compiler Driver NVCC" reference guide*, which can be found at folder `<CUDA_HOME>/doc/`, or at URL <http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>.


<!-- ==================== -->

### 4.4. Constants in the source code:

<!-- ALERT: TODO: -->
TBD
Further tunning for a particular GPU architecture can be performed by customizing some constants in the source code: `GPU_kernels.cuh` ...

<!-- ==================== -->

### 4.5. Compilation process and generated files

To compile *bioNMF-GPU* and utility programs, you just need to execute at the prompt:

		$> make  [ all ]  CUDA_HOME="/path/to/CUDA-Toolkit"


Options for `Makefile` can be specified as arguments,

		$> make SINGLE=1  FAST_MATH=1  CUDA_HOME="/path/to/CUDA-Toolkit"

or as environment variables, for some of them:

		$> export  CUDA_HOME="/path/to/CUDA-Toolkit"
		$> export  SM_VERSIONS="13  20-21  PTX35"
		$> make    FAST_MATH=1  SINGLE=1

*Notes:*

   * Parameters set as environment variables *remain in effect* until you close your terminal session. So, subsequent invocations to `make`, do *not* require to specify them again, unless you want to modify their value.

   * `Makefile` arguments *have priority* over environment variables.  
	 For instance, in:

			$> export CUDA_HOME="/usr/local/cuda-5.5"
			$> make CUDA_HOME="/usr/local/cuda-4.1"

	 *bioNMF-GPU* will be compiled with CUDA Toolkit version *4.1*.

   * Since the `env.sh` script also requires the path to your CUDA Toolkit (see chapter 6.1 *"Execution environment"*), you can perform both actions in a single step by executing such script *before* compiling *bioNMF-GPU*.  
     That is,

			$> .  env.sh  "/path/to/CUDA"
			$> make  SM_VERSIONS=13   FAST_MATH=1
	 In any case, if no path to CUDA is specified, both, `env.sh` and `Makefile`, will try to derive it by looking for `nvcc` in all folders stored in your `PATH` environment variable.


Finally, to show all command executed by `make`, you can use the `VERBOSE` parameter,

		$> make VERBOSE=1

If such variable is set to '`2`', it also prints all `nvcc` internal commands. Please, see section *4.2 "`Makefile` parameters"* for details.


#### Structure of binary folder:

After compilation, you should find the following files and folders (among others) in `bin/` directory:

   * `bin/NMF_GPU`	─ *bioNMF-GPU* executable file (single-GPU version).
   * `bin/tools/`	─ Utility programs (see next section).
   * `bin/obj/`		─ All object files.

The `obj/` folder contains all object files following a directory structure similar to `src/` (shown on section *3.2 "Source-code directory structure"*). This folder can be safety deleted after compilation by executing:

		$> make clean


*****************************


## 5. Utility programs

In addition to *bioNMF-GPU*, there are some utility programs to make easier working with input files. It includes a program for binary-text file conversion, and another to generate input matrices with random data (useful for testing *bioNMF-GPU*).

To compile such programs, just execute:

		$> make utils

which will generate the following files:

   * `bin/tools/file_converter`		─ Binary-text file conversions.
   * `bin/tools/generate_matrix`	─ Program to generate a synthetic-data matrix.

***Note:*** These programs do *not* make use of the GPU device. They are implemented in pure-`C99` language, and all operations are performed on the *host* (i.e., the CPU). Therefore, they do *not* require any CUDA-related option, configuration or software. In particular, it is *not* necessary to specify the `Makefile` parameters "`CUDA_HOME`" and/or "`SM_VERSIONS`".


### 5.1. Binary-text file converter

Since *bioNMF-GPU* accepts input matrices stored in a binary or text file, this program allows file conversion between both formats. For binary files, there are two sub-formats: *"native"* and *non-"native"*.

   * ***"Native"* mode** refers to *raw* I/O. That is, data are stored/loaded with the precision specified at compilation time: '`float`' if the `Makefile` parameter "`SINGLE`" was set to '`1`', or '`double`' otherwise. Matrix dimensions are stored/loaded in a similar way (i.e., '`unsigned int`', if "`UNSIGNED`" was set to '`1`'; '[`signed`] `int`', otherwise). This mode is faster because *no* error checking is performed. Therefore, it should *not* be used to read untrusted input files.  
     **Important note:** In *output* files, this mode also *skips* all data transformation steps (e.g., *matrix transposing*). In particular, matrix **H** (which is computed in transposed mode due to performance reasons) will *not* be "restored" before writing the output file.

   * In **non-*"native"* mode**, data are *always* stored/loaded using *double* precision (and *unsigned* integers for matrix dimensions), regardless the options specified at compilation. This is the recommended mode for input or final output data, since every datum is checked for invalid format.

<!-- ALERT: TODO: -->
All file formats accepted by *bioNMF-GPU* are detailed in section *3 "Data-file format"* of the *User guide*, similarly for program usage (section *6 "Utility programs"*). Finally, there are some examples of valid input files in the `test/` folder.

<!-- ==================== -->

### 5.2. Matrix generator

 This program generates a data matrix with non-negative random values. You can select the output matrix dimensions, as well as the maximum value (i.e, all data will be between '`0.0`' and the specified value, included). In addition, the output matrix can be written as an ASCII text, or in a binary file (in any of the binary modes described above).

 ***Notes:***

   * Output file will not contain any tag (i.e., row labels, column headers or description string), just numeric data (see section *3 "Data-file formats"* in the *User guide*.)
   * The entire operation is performed on the *host* (i.e., on the CPU), *not* on the GPU device.
   * Please, see program usage in section *6 "Utility programs"* in the *User guide*. <!-- TODO -->


*****************************


## 6. *BioNMF-GPU* execution setup

 This section describes how to set up an appropriate environment to execute *bioNMF-GPU*.

### 6.1. Execution environment

 When *bioNMF-GPU* is compiled on a UNIX system, some of the required libraries (e.g, *CUBLAS*) are *dynamically* linked. That is, they are not embedded into the executable file, but the program locates and loads them *at runtime*.

 To set up an appropriate execution environment, you can use the provided script `env.sh`, as follow:

		$> cd <BIONMF_PATH>
		$>  .  ./env.sh  [ <path_to_CUDA_Toolkit> ]

 If the argument is not specified, it will be derived by looking for `nvcc` in all folders stored in your `PATH` environment variable.


 Please note this script should *not* be *"executed"* (in a sub-shell), but *"sourced"* on the current session by the command '`.`'. On some shells, such as `zsh`, you must use the '`source`' command instead. That is,

		$> source  ./env.sh  [ <path_to_CUDA_Toolkit> ]

 In others (e.g., `dash`), the argument must be previously specified with the '`set`' command, as follow:

		$> set  --  <path_to_CUDA_Toolkit>
		$>  .  ./env.sh

 Finally, this script may not work on (`t`)`csh` shells. If you are not able to use another command-line interpreter (e.g., `bash`, `dash`, `*ksh`, or `zsh`), you can manually set up the environment, as follow:

		$> set CUDA_HOME="/path/to/CUDA_Toolkit/"
		$> set PATH="${CUDA_HOME}/bin":${PATH}
		$> set LD_LIBRARY_PATH="${CUDA_HOME}/lib":${LD_LIBRARY_PATH}

 On *Mac OS X*, replace '`LD_LIBRARY_PATH`' by '`DYLD_LIBRARY_PATH`'. That is,

		$> set DYLD_LIBRARY_PATH="${CUDA_HOME}/lib":${DYLD_LIBRARY_PATH}


 **Important note:** folder names containing whitespace characters are *NOT* supported. In that case, either use a (soft) link, or rename your CUDA installation directory.


#### Compilation & execution environment in a single step:
 The script `env.sh` also exports the environment variable '`CUDA_HOME`' containing the path to your CUDA Toolkit (either because it was specified as an argument, or because it was derived from `PATH`). As described in section *4.2 "`Makefile` parameters"*, this variable is required for the compilation process. Therefore, you can set up both, compilation and execution environments, in a single step by executing `env.sh` *before* compiling *bioNMF-GPU*. That is,

		$> .  env.sh  "/path/to/CUDA"			# Sets up the environment
		$> make  SM_VERSIONS=13   FAST_MATH=1	# Compiles bioNMF-GPU
		$> bin/NMF_GPU <input_file> [...]		# Executes the program


<!-- ==================== -->


### 6.2. Testing your execution environment

 In order to make sure the execution environment has been properly set, you can check if the generated executable files are able to locate all required libraries. According to your UNIX system, this can be performed with one of the following commands:

   * *Linux:*

			ldd  <path_to_executable_file>

   * *Mac OS X:*

			dyldinfo  -dylibs  <path_to_executable_file>


For instance, on a 32-bits Ubuntu Linux, with a CUDA Toolkit version 5.5 installed on '`/usr/local/cuda-5.5`', the output for *bioNMF-GPU* (single-GPU version) should be something similar to:

		$> ldd bin/NMF_GPU
				linux-gate.so.1 =>  (0xb7754000)
				libcublas.so.5.5 => /usr/local/cuda-5.5/lib/libcublas.so.5.5 (0xb4752000)
				libcurand.so.5.5 => /usr/local/cuda-5.5/lib/libcurand.so.5.5 (0xb279d000)
				libcudart.so.5.5 => /usr/local/cuda-5.5/lib/libcudart.so.5.5 (0xb275d000)
				libm.so.6 => /lib/i386-linux-gnu/libm.so.6 (0xb2722000)
				libc.so.6 => /lib/i386-linux-gnu/libc.so.6 (0xb2577000)
				libdl.so.2 => /lib/i386-linux-gnu/libdl.so.2 (0xb2572000)
				libstdc++.so.6 => /usr/lib/i386-linux-gnu/libstdc++.so.6 (0xb2489000)
				libpthread.so.0 => /lib/i386-linux-gnu/libpthread.so.0 (0xb246d000)
				librt.so.1 => /lib/i386-linux-gnu/librt.so.1 (0xb2464000)
				libgcc_s.so.1 => /lib/i386-linux-gnu/libgcc_s.so.1 (0xb2446000)
				/lib/ld-linux.so.2 (0xb7755000)

<!-- ==================== -->

### 6.3. Setting the environment permanently

 To permanently set the required compilation and execution environments, just add the following information to your shell initialization file:

		export CUDA_HOME="/path/to/CUDA_Toolkit/"
		export PATH="${CUDA_HOME}/bin":${PATH}
		export LD_LIBRARY_PATH="${CUDA_HOME}/lib":${LD_LIBRARY_PATH}

 On *Mac Os X*, please replace '`LD_LIBRARY_PATH`' by '`DYLD_LIBRARY_PATH`'. That is,

		export DYLD_LIBRARY_PATH="${CUDA_HOME}/lib":${DYLD_LIBRARY_PATH}

 On some shells, such as (`t`)`csh`, you must use the '`set`' command, instead.


*****************************


## 7. Testing *bioNMF-GPU*

TBD

<!-- ALERT: TODO -->


*****************************


## 8. Issues/Troubleshooting

   1. `Dash: env.sh: argument not recognized.`

   2. `Zsh: 'env.sh' not found.`

   3. Invoking '`env.sh`' on a (`t`)`csh` shell.

   4. `Catastrophic error: could not set locale "" to allow processing of multibyte characters.`

   5. `/usr/include/limits.h(125): catastrophic error: could not open source file "limits.h".`

   6. `/usr/include/bits/stdio2.h:96: sorry, unimplemented: inlining failed in call to 'fprintf': redefined extern inline functions are not considered for inlining.`

   7. `gcc: Option <X> not recognized.`

<!-- ==================== -->

#### 1. `Dash: env.sh: argument not recognized.`

 On `Dash`, the argument must be specified as follow:

		$> set -- <path_to_CUDA_Toolkit>
		$> . ./env.sh

<!-- ==================== -->

#### 2. `Zsh: 'env.sh' not found.`

 `Zsh` does *not* recognize the *"dot"* command ('`.`'). You must use the '`source`' command, instead. That is,

		source ./env.sh [ <path_to_CUDA_Toolkit> ]

<!-- ==================== -->

#### 3. Invoking '`env.sh`' on a (`t`)`csh` shell.

 The script '`env.sh`' may not work on a (`t`)`csh` shell. If you receive any error message (e.g., '`Permission denied.`' or '`Command not found.`'), you can manually set up the environment, as follow:

		set CUDA_HOME="/path/to/CUDA_Toolkit/"
		set PATH="${CUDA_HOME}/bin":${PATH}
		set LD_LIBRARY_PATH="${CUDA_HOME}/lib":${LD_LIBRARY_PATH}

 On *Mac Os X*, please replace '`LD_LIBRARY_PATH`' by '`DYLD_LIBRARY_PATH`'. That is,

		set DYLD_LIBRARY_PATH="${CUDA_HOME}/lib":${DYLD_LIBRARY_PATH}

<!-- ==================== -->

#### 4. `Catastrophic error: could not set locale "" to allow processing of multibyte characters.`

 This issue seems to be related with the Intel compiler, `icc`. Please set your `LANG` environment variable to '`en_US.ISO-8859-15`'

<!-- ==================== -->

#### 5. `/usr/include/limits.h(125): catastrophic error: could not open source file "limits.h".`

 This issue seems to be related with the Intel compiler, `icc`, version `10.1.015`. Please comment all '`#include <limits.h>`' lines and define the '`INT_MAX`' constant in the source code.  
 Example:

		// #include <limits.h>
		#ifndef INT_MAX
			#define INT_MAX 2147483647
		#endif

 Currently, the only file affected by this issue, is: '`include/index_type.h`'

<!-- ==================== -->

#### 6. `/usr/include/bits/stdio2.h:96: sorry, unimplemented: inlining failed in call to 'fprintf': redefined extern inline functions are not considered for inlining.`

 This issue seems to be related with `gcc`'s '`-combine`' option. Just remove it from `Makefile`.

<!-- ==================== -->

#### 7. `gcc: Option '-<X>' not recognized.`

<!-- ALERT: TODO: -->
TBD.

*****************************


## 9. How to cite *bioNMF-GPU*.

 If you use this software, please cite the following work:

   > E. Mejía-Roa, C. García, J.I. Gómez, M. Prieto, R. Nogales, F. Tirado and A. Pascual-Montano. **Biclustering and classification analysis in gene expression using Nonnegative Matrix Factorization on multi-GPU systems**. In *Proc. of the ISDA Conf.*, 2011:882─887. [doi:10.1109/ISDA.2011.6121769](http://dx.doi.org/10.1109/ISDA.2011.6121769).  
   > <http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6121769>

<!-- ==================================================== -->
 <br/>
 <br/>
 </body>
 </html>

<!--
// kate: backspace-indents off; folding-markers on; indent-mode normal; indent-width 3; keep-extra-spaces off; newline-at-eof on; remove-trailing-space off; replace-trailing-space-save off; replace-tabs off; replace-tabs-save off; remove-trailing-spaces none; tab-indents on; tab-width 4;
-->
