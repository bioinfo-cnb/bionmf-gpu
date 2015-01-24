<!--
 ************************************************************************
 *
 * NMF-mGPU - Non-negative Matrix Factorization on multi-GPU systems.
 *
 * Copyright (C) 2011-2014:
 *
 *      Edgardo Mejia-Roa(*), Carlos Garcia(*), Jose Ignacio Gomez(*),
 *      Manuel Prieto(*), Francisco Tirado(*) and Alberto Pascual-Montano(**).
 *
 *      (*)  ArTeCS Group, Complutense University of Madrid (UCM), Spain.
 *      (**) Functional Bioinformatics Group, Biocomputing Unit,
 *           National Center for Biotechnology-CSIC, Madrid, Spain.
 *
 *      E-mail for E. Mejia-Roa: <edgardomejia@fis.ucm.es>
 *      E-mail for A. Pascual-Montano: <pascual@cnb.csic.es>
 *
 *
 * This file is part of NMF-mGPU.
 *
 * NMF-mGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * NMF-mGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with NMF-mGPU. If not, see <http://www.gnu.org/licenses/>.
 *
 ************************************************************************
-->
<!-- ==================================================== -->
 <html lang="en" xml:lang="en" xmlns="http://www.w3.org/1999/xhtml">
 <head>
   <meta name="application-name" content="NMF-mGPU"/>
   <meta name="author" content="Edgardo Mejia-Roa (edgardomejia@fis.ucm.es), Carlos Garcia, Jose Ignacio Gomez, Manuel Prieto, Francisco Tirado, and Alberto Pascual-Montano (pascual@cnb.csic.es)."/>
   <meta name="description" content="Non-negative Matrix Factorization (NMF) for Biology on multi-GPU systems. Installation guide"/>
   <meta name="keywords" content="bioNMF, NMF, Matrix factorization, GPU, multi-GPU, GPGPU, NVIDIA, CUDA, CUBLAS, Bioinformatics"/>
   <meta name="language" content="en"/>
   <meta name="copyright" content="(C) 2011-2014 Edgardo Mejia-Roa (edgardomejia@fis.ucm.es). ArTeCS Group, Complutense University of Madrid (UCM), Spain."/>
   <meta http-equiv="content-Type" content="text/html; charset=UTF-8"/>
   <meta http-equiv="last-modified" content="2014/12/07" scheme="YYYY/MM/DD"/>
   <link rel="stylesheet" type="text/css" href="styles.css"/>
   <title>NMF-mGPU Installation Guide</title>
 </head>
 <body>


<!-- ==================================================== -->


# *NMF-mGPU* INSTALLATION GUIDE

This document shows how to install and compile *NMF-mGPU*.

**Index:**

   1. [Introduction](#intro).
   2. [System Requirements](#requirements).
   3. [Directory Structure](#folders).
   4. [Compiling *NMF-mGPU*](#compilation).
   5. [Utility Programs](#tools).
   6. [*NMF-mGPU* Execution Setup](#setup).
   7. [Testing *NMF-mGPU*](#testing).
   8. [Issues/Troubleshooting](#troubleshooting).
   9. [How to Cite *NMF-mGPU*](#citation).


*****************************


## 1. <a id="intro">Introduction</a>

***NMF-mGPU*** implements the ***Non-negative Matrix Factorization*** (***NMF***) algorithm by making use of ***Graphics Processing Units*** (***GPUs***). NMF takes an input matrix (**V**) and returns two matrices, **W** and **H**, whose product is equal to the former (i.e., **V** ~ **W** \* **H**). If **V** has *n* rows and *m* columns, then dimensions for **W** and **H**, will be *n* × *k* and *k* × *m*, respectively. The *factorization rank* ("*k*") specified by the user, is usually a value much less than both, *n* and *m*.

This software has been developed using the NVIDIA's [***CUDA***][CUDA_homepage] ([***Compute Unified Device Architecture***][CUDA_homepage]) framework for GPU Computing. *CUDA* represents a GPU device as a programmable general-purpose *coprocessor* able to perform linear-algebra operations.

On detached devices with low on-board memory available, large datasets can be blockwise transferred from the CPU's main memory to the GPU's memory and processed accordingly. In addition, *NMF-mGPU* has been explicitly optimized for the different CUDA architectures.

Finally, *NMF-mGPU* also provides a *multi-GPU* version that makes use of multiple GPU devices through the [***MPI***][MPI_homepage] ([***Message Passing Interface***][MPI_homepage]) standard.


[CUDA_homepage]: <http://www.nvidia.com/object/cuda_home_new.html> "CUDA Homepage"
[MPI_homepage]: <http://mpi-forum.org/> "MPI Forum"


*****************************


## 2. <a id="requirements">System Requirements</a>

The main system requirements for *NMF-mGPU* are the following:

   * **UNIX System (GNU/Linux or Darwin/Mac OS X)**. *NMF-mGPU* has not been tested on Microsoft Windows yet.

   * One or more **CUDA-capable GPU device(s)**: A detailed list of compatible hardware can be found at <http://developer.nvidia.com/cuda-gpus>  
     Please note that **all** devices must be of the same architecture (i.e., heterogeneous GPU clusters are not supported yet).

   * **CUDA Toolkit and CUDA Driver**: They are freely available at the [CUDA Downloads Page][CUDA-Download]. Nevertheless, for *deprecated* GPU devices and/or OS platforms, you can download a previous CUDA release (e.g., version 5.5) from the [CUDA Archive Page][CUDA-OR-Download]. Please note that *NMF-mGPU* requires, at least, the version 4.2.

   * A C compiler **conforming to the ISO-C99 standard**, such as [GNU GCC](https://gcc.gnu.org) or [LLVM Clang](http://llvm.org/).

   * The ***optional* multi-GPU version** also requires an *MPI-2.0* (or greater) software library, such as [OpenMPI](http://www.open-mpi.org/) or [MPICH](http://www.mpich.org/).


[CUDA-Download]: <http://developer.nvidia.com/cuda-downloads/> "CUDA Download Page"
[CUDA-OR-Download]: <https://developer.nvidia.com/cuda-toolkit-archive/> "CUDA Archive Page"


Please note that *no* CUDA driver, or even a GPU device, is required to *just* compile *NMF-mGPU*. This is useful, for instance, in cluster environments where the program shall be compiled in a front-end not equipped with a GPU device (e.g., a virtual machine).  
&nbsp;


Further system requirements and installation steps for CUDA software, vary according to your operating system:

   * **GNU/Linux**: For instance, on **Ubuntu 14.04**:

      + **NVIDIA proprietary driver**: Open the program *Software & Updates*, then go to *Additional Drivers* section, and check the "*Using NVIDIA binary driver*" option.  
        Alternatively, you can open a terminal and type:

               sudo apt-get install nvidia-current

        You may have to reboot the system in order to use this driver after installing it.

      + **Additional packages**: The following packages are required: `build-essential`, `nvidia-cuda-dev` and `nvidia-cuda-toolkit`.  
         They can be installed through the *Ubuntu Software Center*, or via a terminal by typing: 

               sudo apt-get install build-essential nvidia-cuda-dev nvidia-cuda-toolkit

      + **Multi-GPU version (optional)**: This version also requires any of the following packages: `openmpi` or `mpich`.

     For other GNU/Linux distributions, we recommend to read the [Getting Starting Guide for GNU/Linux](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/index.html#package-manager-installation). Similarly, the [Release Notes][CRN] contain important information about [unsupported][CRN-unsupported] and [deprecated][CRN-deprecated] features, as well as [known issues][CRN-issues].


   * For **Darwin/Mac OS X**:

      + **C/C++ compiler**: Please install the Apple's [Xcode](https://developer.apple.com/xcode/downloads/) toolset. Some versions may require to explicitly add the *Command Line Developer Tools* plug-in in order to make available the required commands on the Terminal.

      + **CUDA Toolkit and Drivers**: Just download and execute the proper `.dmg` file from the [CUDA Download Page][CUDA-Download] (or the [Archive Page][CUDA-OR-Download] for previous releases), and follow the instructions.

      + **Multi-GPU version (optional)**: Most MPI libraries are available on package managers, such as [MacPorts](http://www.macports.org/) or [Homebrew](http://brew.sh/). Otherwise, you can download the source code and compile it.

     We highly recommend to read the [Getting Starting Guide for Darwin/Mac OS X](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-mac-os-x/index.html) for detailed instructions.


Finally, we also recommend to read the [Release Notes][CRN] contain important information about [unsupported][CRN-unsupported] and [deprecated][CRN-deprecated] features, as well as [known issues][CRN-issues].


### Warning:

   * **Folder names containing whitespace characters are *NOT* supported by *NMF-mGPU***. Please avoid them in the path to your *CUDA Toolkit* installation directory. A (soft) link can be used as a workaround. For instance:

            ln  -s   /Developer/Whitespaced\ foldername/cuda     /Developer/cuda

   * **The [LLVM Clang](http://llvm.org/) compiler is *NOT* supported on *32-bits* system** since it does not recognize the `-malign-double` switch. Please read the [CUDA Release Notes on compilers][CRN-compiler] for details.


[CRN]: <http://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html> "CUDA Release Notes"
[CRN-unsupported]: <http://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#unsupported-features>
    "CUDA Release Notes - Unsupported Features"
[CRN-deprecated]: <http://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#deprecated-features>
    "CUDA Release Notes - Deprecated Features"
[CRN-issues]: <http://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#known-issues-title> "CUDA Release Notes - Known Issues"
[CRN-compiler]: <http://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-compiler-known-issues>
    "CUDA Release Notes - Compiler Known Issues"


*****************************


## 3. <a id="folders">Directory Structure</a>

This section lists the files and folders you should find after downloading and decompressing *NMF-mGPU*.

### 3.1. Files and folders

After extraction of the compressed file, you should see the following folders:

   * `doc/`        ─ Installation and User guides.
   * `src/`        ─ Source code.
   * `include/`    ─ Header files.
   * `test/`    ─ Examples of a valid input file.


In addition, there are some important files that should be customized *before* the compilation process:

   * `Makefile`    ─ Compilation directives for UNIX platforms.
   * `env.sh`    ─ Required environment variables (see [*NMF-mGPU Execution Setup*](#setup)).


<!-- ==================== -->


### 3.2. Source-code directory structure

The `src/` folder is organized as follow:

   * Main-program files:
      + `NMF-GPU.c`  ─ Main program (single-GPU version).
      + `NMF-mGPU.c` ─ Main program (multi-GPU version).

   * CUDA files (compiled as ISO-C code):
      + `NMF_routines.c`      ─ NMF-related operations.
      + `matrix_operations.c` ─ Algebraic operations and data transfers on matrices.
      + `GPU_setup.c`         ─ GPU set-up and management routines.
      + `timing.c`            ─ Timing and profiling routines.

   * CUDA device code:
      + `GPU_kernels.cu` ─ CUDA kernels.

   * Other C files:
      + `matrix_io/matrix_io.c`          ─ Functions to read/write data matrices.
      + `matrix_io/matrix_io_routines.c` ─ Auxiliary I/O routines.

   * Utility programs (see [*Utility Programs*](#tools)):
      + `tools/file_converter.c`  ─ ASCII-Binary file conversion.
      + `tools/generate_matrix.c` ─ Generates a matrix with random values in text or binary file format.


The `include/` folder is similarly organized:

   * `real_type.h`                    ─ Floating-point data definition.
   * `index_type.h`                   ─ Data type used for array index.
   * `nmf_routines.h`
   * `matrix_operations.h`
   * `GPU_kernels.cuh`
   * `GPU_setup.h`
   * `timing.h`
   * `matrix_io/matrix_io.h`
   * `matrix_io/matrix_io_routines.h`


*****************************


## 4. <a id="compilation">Compiling *NMF-mGPU*</a>

This section describes all supported options provided by our `Makefile`, which allow to customize the compilation process.

Note: All the information contained in the following subsections can be printed on the screen by executing:

         $> make help


### 4.1. `Makefile` Targets:

 List of (main) available actions:

   * `all`: *DEFAULT target*. Compiles all programs **except** the multi-GPU version. It is equivalent to: `single_gpu tools`.

   * `all_programs`: Compiles all programs **including** the multi-GPU version. It is equivalent to: `all multi_gpu`.

   * `single_gpu`: Compiles *NMF-mGPU* (single-GPU version) only.

   * `multi_gpu`: Compiles *bioNMF-mGPU* (multi-GPU version) only. Target *NOT* compiled by default.

   * `tools`: Compiles some utility programs (see [*Utility Programs*](#tools) for details).  
     Currently, this target does *not* require any CUDA-related configuration or software. In particular, it is *not* necessary to specify  `CUDA_HOME` or `SM_VERSIONS` parameters.

   * `help`: Prints a help message with all this information.

   * `help_sm_versions`: Prints a detailed description of valid values for the `SM_VERSIONS` parameter.

   * `help_tools`: Prints a short description of available [*utility programs*](#tools).

   * `clobber`: Removes code generated by the 'all' target. It is equivalent to `clobber_single_gpu clobber_tools`.

   * `clobber_all`: Removes the entire binary directory, with all executable and object files.

   * `clobber_single_gpu`, `clobber_multi_gpu`, `clobber_tools`: Removes the specified executable and its associated object files.

   * `clean`: Removes object files (i.e., `*.o`) generated by the 'all' target. It is equivalent to `clean_single_gpu clean_tools`

   * `clean_all`: Removes all object files (i.e., `*.o`).

   * `clean_single_gpu`, `clean_multi_gpu`, `clean_tools`: Removes all object files associated to the specified program.


<!-- ==================== -->


### 4.2. <a id="mkparams">`Makefile` Parameters</a>

The compilation process can be customized with the following parameters:

   * `CUDA_HOME`: Path to your CUDA Toolkit.  
     It may be an environment variable or an argument.  
     If not specified, it will be derived by looking for `<NVCC>` in all folders stored in your `PATH` environment variable.  
     Please note that **folder names containing whitespace characters are NOT supported**. In that case, either use a (soft) link, or rename your CUDA installation directory.  
     This parameter is currently ignored on the `tools` target.

   * `SM_VERSIONS`: Target GPU architecture(s).  
     This parameter may be an environment variable or an argument.  
     Device code will be generated for the specified *Compute Capability(-ies)* (CC).  
     For instance, `SM_VERSIONS="10-13  30  PTX35"`:
      + Generates device-specific executable code for CC **1.3**, using only the basic functionality present on CC **1.0**.
      + Generates device-specific executable code for CC **3.0**, using all available features on such architecture.
      + Emits `PTX` code for CC **3.5**, which can be later dynamically compiled and executed by any current or future device, with a similar or higher Compute Capability.

     To generate device-specific executable code for CC **2.1**, please specify it as: `20-21`.  
     See a more detailed description of this parameter below, or by executing: `make help_sm_versions`.  
     This parameter is currently ignored on the `tools` target.  
     Default value(s): `SM_VERSIONS="13  20  30  20-21  PTX35"`

   * `SINGLE`: If set to `1`, uses single-precision data (i.e., `float`). Else, uses double-precision data (i.e., `double`).  
     Note, however, that in Compute Capability 1.2 and lower, all double-precision operations are *demoted* to single-precision arithmetic, or are not supported.  
     Default value: `1`.

   * `UNSIGNED`: Uses *unsigned* integers for matrix dimensions, which may generate faster code. Nevertheless, please note that *CUBLAS* library functions use *signed*-integer parameters. Therefore, matrix dimensions must *not* overflow such data type. An error message will be shown if this happens.  
     Default value: `1`.

   * `FAST_MATH`: Uses less-precise faster math functions.  
     Default value: `1`.

   * `TIME`: Shows total elapsed time.  
     Default value: `1`.

   * `TRANSF_TIME`: Shows time elapsed on data transfers.  
     Default value: `0`.

   * `KERNEL_TIME`: Shows time elapsed on kernel code.  
     Default value: `0`.

   * `COMM_TIME`: Shows time elapsed on MPI communications.  
     Default value: `0`.

   * `SYNC_TRANSF`: Performs *synchronous* data transfers.  
     Default value: `0`.

   * `FIXED_INIT`: Makes use of "random" values from a fixed seed.  
     Default value: `0`.

   * `CPU_RANDOM`: Generates random values from the CPU host, not from the GPU device.  
     Default value: `0`.

   * `DBG`: *Verbose* and *Debug* mode (prints *a lot* of information). It implies `CC_WARN_VERBOSE`, `NVCC_WARN_VERBOSE`, and `PTXAS_WARN_VERBOSE` set to `1`.  
     Default value: `0`.

   * `VERBOSE`: Command-line verbosity level. Valid values are `0` (none), `1` (shows `make` commands), and `2` (shows `make` and internal `NVCC` commands).  
     Default value: `0`.

   * `CC_WARN_VERBOSE`: Shows extra warning messages on programs compiled with `CC`. Please note that it may include *false-positive* warnings.  
     Default value: `0`.

   * `NVCC_WARN_VERBOSE`: Shows extra warning messages on programs compiled with `NVCC`. Please note that it may include *false-positive* warnings. It implies `PTXAS_WARN_VERBOSE` set to `1`.  
     Default value: `0`.

   * `PTXAS_WARN_VERBOSE`: Shows `PTX`-specific compilation warnings.  
     Default value: `0`.

   * `KEEP_INTERMEDIATE`: Keeps temporary files generated by `NVCC`.  
     Default value: `0`.


Default compilers can be *overridden* with following parameters or environment variables:

   * `CC`: Compiler for ISO-C and CUDA-host code. Used also in the linking stage of targets "`single_gpu`" and "`tools`".
     Supported compilers: `gcc` and `clang`.  
     Default value: `clang` for Darwin (i.e., Mac OS X); `gcc`, otherwise.

   * `NVCC`: Compiler for CUDA device code.  
     Default value: `nvcc`.

   * `MPICC`: Compiler for `MPI` code.  
     Default value: `mpicc`


Additional flags, not affected by other input parameters, can be specified through the following parameters or environment variables:

   * `CPPFLAGS`, `CFLAGS`, `INCLUDES`, `LDFLAGS`, `LDLIBS`: Additional flags included in all targets.  
     *Note:* "`CFLAGS`" is **ignored** by `NVCC`. Instead, please make use of `CXXFLAGS` and/or `NVCC_CFLAGS` (see below).

   * `CXXFLAGS`: Additional options controlling the `NVCC`s internal compiler. Each word is automatically prefixed by `--compiler-options` in the command line. Ignored on files not compiled with `NVCC`.

   * `NVCC_CPPFLAGS`, `NVCCFLAGS`, `NVCC_INCLUDES`: Additional options for `NVCC`.

   * `OPENCC_FLAGS`: Additional flags for `nvopencc`, which generates `PTX` (intermediate) code on devices of *Compute Capability 1.x*. Ignored on newer GPU architectures.

   * `PTXAS_FLAGS`: Additional flags for `PTX` code compilation, which generates the actual GPU assembler.

   * `MPICC_CPPFLAGS`, `MPICC_CFLAGS`, `MPICC_INCLUDES`, `MPICC_LDFLAGS`, `MPICC_LDLIBS`: Additional flags for `MPICC`.


#### The `SM_VERSIONS` parameter:

Device code is compiled in two stages. First, the compiler emits an assembler code (named `PTX`) for a *virtual* device that represents a *class* of GPU models with similar features and/or functionality. In the second stage, such `PTX` code is then compiled in order to generate executable code for a particular (*real*) device from the former GPU class.

*Virtual* architectures are named "`compute_`*XY*", while *real* GPU models are denoted as "`sm_`*XY*". In both terms, the value "*XY*" represents the *Compute Capability* (*CC*) "*X.Y*".

Note that both architectures must be compatible. That is, `PTX` code emitted for a `compute_`*XY* GPU class can be compiled and executed on a `sm_`*WZ* device, if and only if, *X.Y* <= *W.Z*. For instance, `compute_13` is *not* compatible with `sm_10`, because the former architecture assumes the availability of features that are not present on devices of CC **1.0**, but only on **1.3** and greater.

For a detailed description of concepts above, please see chapter *GPU Compilation* in the *"CUDA Compiler Driver NVCC" reference guide*, which can be found in the `doc` folder of your CUDA Toolkit, or at the URL <http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-compilation>.


There are three ways to specify target architecture(s) in the `SM_VERSIONS` parameter:

   1. **Device-specific features & code** (i.e., `PTX` and executable code with **similar** Compute Capability):

      Emits `PTX` assembler code, which is then compiled into executable instructions, just for the given *Compute Capability(-ies)* (CC). Since the former is just an intermediate code (i.e., it is *not* retained in the output file), and the latter is generated with a device-specific binary format, the program may not be compatible with other GPU architectures. That is, any given CC value, "*XY*", is translated to the following `NVCC` option: "`--generate-code=arch=compute_`*XY*`,code=sm_`*XY*".

      For instance, `SM_VERSIONS="13  35"` generates executable code just for devices of Compute Capability **1.3** and **3.5**, respectively. GPU devices with other CC values, such as 1.1 or 2.0, may not be able to execute the program.

      *Note:* For devices, such as CC **2.1**, that do not have a similar `PTX` CC number, please specify the nearest lower value ("***2.0***", for the previous example) by using the dashed-separated form below.


   2. **Generic features, device-specific code** (i.e., `PTX` and executable code, with **different** Compute Capabilities):

      This is a generalization of the previous form. Here, different *Compute Capabilities* (CC) values can be specified for both, `PTX` and executable code, separated by a dash. That is, "*XY*-*WZ*" (with *XY* <= *WZ*), emits `PTX` assembler code for CC "*X.Y*", which is then compiled into executable instructions for a device of CC "*W.Z*". The former, determines the target *architecture* (i.e., the available hardware/software features and functionality). The latter, specifies the target *device* in terms of binary code format.

      Similarly as the previous form, no `PTX` code is embedded in the output file, so the program may not be compatible with other GPU device models. That is, the previous expression is translated to the `NVCC` option: "`--generate-code=arch=compute_`*XY*`,code=sm_`*WZ*"

      Note that "*XY*`-`*XY*" is equivalent to just specify "*XY*" as in the previous form. On the other hand, if *XY* < *WZ*, the program is still compiled for the target device (i.e., CC "*W.Z*"), but it will only make use of features available on CC "*X.Y*", discarding any functionality introduced since.

      *Note:* As stated above, please use this form to specify target devices, such as CC **2.1**, that do not have a similar `PTX` CC number (so, a lower value must also be given). Example: `20-21`.

      For instance, `SM_VERSIONS="10-13  20-21"`:
       * Generates executable code for a device of CC **1.3**, with the basic features that are available on CC **1.0**. In particular, it *discards* all support for double-precision floating-point data, introduced on CC **1.3**.
       * Compiles the algorithm with the features and functionality available on CC **2.0**, and generates a binary image for a device of CC **2.1**.
       * Since no `PTX` code is retained in the output file, the program may not compatible with other GPU devices (e.g., CC 3.0).


   3. **Generic features and "*code*"**:

      Emits `PTX` assembler code for the given *Compute Capability* (CC), which is then embedded into the output file. No executable code is generated. Instead, the former is dynamically compiled at runtime according to the actual GPU device. Such process is known as *Just-in-Time compilation* (*JIT*).

      To specify a target architecture in a such way, please use the word `PTX` followed by the target Compute Capability. That is, "`PTX`*wz*", generates `PTX` code for Compute Capability "*w.z*", and embeds it into the output file. Such code can be later compiled and executed on any device, with a similar or greater CC value. Similarly as previous forms, the expression above is translated to the following `nvcc` option: "`--generate-code=arch=compute_`*wz*`,code=compute_`*wz*".

      Note, however, that JIT compilation increases the startup delay. In addition, the final executable code will use just those architectural features that are available on CC "*w.z*", discarding any functionality introduced since.

      For instance, `SM_VERSIONS="PTX10  PTX35"`:
       * Emits `PTX` code for the first CUDA-capable architecture (i.e., CC **1.0**). Therefore, the program can be later dynamically compiled and executed on *any* current or future GPU device. Nevertheless, it will only use the (very) basic features present on such architecture.
       * Generates `PTX` code that can be later compiled and executed on devices of CC **3.5**, or higher.
       * Any device prior to CC **3.5** (e.g., 1.3, 2.1, or 3.0), will execute the basic CC **1.0** version.


*Notes*:

   * Double-precision floating-point data are **NOT** supported on Compute Capability 1.2 and lower. The compilation process may fail if any of them is specified and the parameter `SINGLE` is set to `0`.
   * By default, **no code is compiled for Compute Capabilities 1.2 and lower**, since they are being deprecated on newer versions of the CUDA Toolkit. Code for such architectures must be explicitly requested as shown in the examples above.
   * This parameter is currently ignored on the `tools` target.


Current default value(s):

         "13  20  30  35  20-21  PTX35"

which will be translated into the following argument(s) for `NVCC`:

         --generate-code=arch=compute_13,code=sm_13
         --generate-code=arch=compute_20,code=sm_20
         --generate-code=arch=compute_30,code=sm_30
         --generate-code=arch=compute_35,code=sm_35
         --generate-code=arch=compute_20,code=sm_21
         --generate-code=arch=compute_35,code=compute_35


<!-- ==================== -->


### 4.3. Other compiling options

The compilation process can be further customized by editing section "*Compiler Options*" in `Makefile`. There, you can adjust `CC`-specific flags, and/or options for the `NVCC` compiler, as well as its internal tools (e.g., `nvopencc`, `ptx`, etc).

#### `CC` options:
Since host (i.e., non-device) code in CUDA source files is internally compiled by `NVCC` as C++, options for `CC` are distributed into two groups: common- C/C++ flags, and options for C-only source files. Within each group, some flags may, or may not be applied according to `Makefile` input parameters. For instance, `common_fast_CFLAGS` and `c_only_fast_CFLAGS`, specify which flags are applied when the input parameter `FAST_MATH` is set to `1`.

*Notes:*

   * Further options may apply according to the selected `CC` compiler (`gcc` or `clang`) and the current OS (`darwin` or `linux`).  
   * C-only source files require support for ISO-C99 standard. Therefore, flags for C-only code include the option `-std=c99`.  
   * Some (optional) useful features for processing single-precision floating-point data are only available when `_GNU_SOURCE` is defined.
     Such option is included for C-only source files when the input parameter `SINGLE` is set to `1`.  


#### `NVCC` options:
Variables containing flags for `NVCC`, follow a similar naming scheme as for `CC` options. Most of options for `NVCC` are those specified as common C/C++ flags, prefixed with `--compiler-options`.

<!--**Note for Windows CygWin/Mingw users**: It may be necessary to uncomment the `--drive-prefix` option in order to correctly handle filenames and paths in Windows native format. -->

In this section of the `Makefile`, you can also find options for `nvopencc`, which is a tool internally used by `nvcc` to generate `PTX` assembler code on devices of Compute Capability *1.x* only. You can ignore them for newer architectures.
 
Finally, there are flags to control `PTX` code compilation.

See a detailed description of these flags on the *"CUDA Compiler Driver NVCC" reference guide*, which can be found at folder `<CUDA_HOME>/doc/`, or at URL <http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>.


<!-- ==================== -->


### 4.4. Compilation process and generated files

To compile the default programs, you just need to execute at the prompt:

         $> make  CUDA_HOME="/path/to/CUDA-Toolkit"

Nevertheless, as previously stated, the multi-GPU version is *not* compiled by default, so it must be explicitly requested:

         $> make  multi_gpu

or, to compile all programs:

         $> make  all_programs


Options for `Makefile` can be specified as arguments,

         $> make SINGLE=1  FAST_MATH=1  CUDA_HOME="/path/to/CUDA-Toolkit"

or as environment variables, for some of them:

         $> export  CUDA_HOME="/path/to/CUDA-Toolkit"
         $> export  SM_VERSIONS="13  20-21  PTX35"
         $> make    FAST_MATH=1  SINGLE=1

*Notes:*

   * **Parameters set as environment variables remain in effect until you close your terminal session.** So, subsequent invocations to `make`, do *not* require to specify them again, unless you want to modify their value.

   * **`Makefile` arguments have priority over environment variables.**
     For instance, in:

            $> export CUDA_HOME="/usr/local/cuda-5.5"
            $> make CUDA_HOME="/usr/local/cuda-4.1"

     *NMF-mGPU* will be compiled with CUDA Toolkit version ***4.1***.

   * Since the `env.sh` script also requires the path to your CUDA Toolkit (see [Execution Setup](#setup)), you can perform both actions in a single step by executing such script *before* compiling *NMF-mGPU*.  
     That is,

            $> .  env.sh  "/path/to/CUDA"
            $> make  SM_VERSIONS=13   FAST_MATH=1
     In any case, if no path to CUDA is specified, both, `env.sh` and `Makefile`, will try to derive it by looking for `nvcc` in all folders stored in your `PATH` environment variable.


Finally, to show all command executed by `make`, you can use the `VERBOSE` parameter:

         $> make VERBOSE=1

If such variable is set to `2`, it also prints all `nvcc` internal commands.


#### Structure of binary folder:

After compilation, you should find the following files and folders (among others) in the `bin/` directory:

   * `bin/NMF_GPU`  ─ *NMF-mGPU* executable file (single-GPU version).
   * `bin/NMF_mGPU` ─ *NMF-mGPU* executable file (multi-GPU version), if explicitly requested.
   * `bin/tools/`   ─ [Utility Programs](#tools).
   * `bin/obj/`     ─ All object files.

The `obj/` folder contains all object files following a tree structure similar to `src/` (see [*Directory Structure*](#folders)). This folder can be safety deleted after compilation by executing:

         $> make clean_all


<!-- ==================== -->


### 4.5. Performance Fine-Tuning:

*NMF-mGPU* has been designed to take advantage of different architecture-specific features among existing GPU models. Nevertheless, device code can be further optimized for modern devices by customizing some constants in the source code. For instance, in most GPU kernels, each CUDA thread processes multiple items from global memory in order to increase the thread-level parallelism. For each kernel, the number of such operations is specified by a constant named "*\<kernel_name\>*`__ITEMS_PER_THREAD`", which is defined in `include/GPU_kernels.cuh`. The default value is set to ensure a 100% of occupancy on devices of *Compute Capability 1.x*, so it can be increased if the program will be compiled for newer GPU architectures.


*****************************


## 5. <a id="tools">Utility Programs</a>

In addition to *NMF-mGPU*, there are some utility programs to make easier working with input files. It includes a program for binary-text file conversion, and another to generate input matrices with random data (useful for testing *NMF-mGPU*).

To compile such programs, just execute:

        $> make tools

which will generate the following files:

   * `bin/tools/file_converter`        ─ Binary-text file conversions.
   * `bin/tools/generate_matrix`    ─ Program to generate a synthetic-data matrix.

***Note:*** These programs do *not* make use of any GPU device. They are implemented in pure-ISO-C99 language, and all operations are performed on the *host* (i.e., the CPU). Therefore, they do *not* require any CUDA-related option, configuration or software. In particular, it is *not* necessary to specify the `Makefile` parameters "`CUDA_HOME`" and/or "`SM_VERSIONS`".


### 5.1. Binary-text File Converter

Since *NMF-mGPU* accepts input matrices stored in a binary or text file, this program allows file conversion between both formats. For binary files there are two sub-formats: "*native*" and "*non-native*".

   * "***Non-native***" **mode**: matrix data are stored using *double*-precision values, and 32-bits *unsigned* integers for matrix dimensions. If necessary, all values must be converted to little-endian format before the writing operation. Finally, the file contains a binary "*signature*", which will be checked when reading.

   * "***native***" **mode**: matrix data are stored in *raw* format according to the selected compilation parameters. That is, `float` values if the program was compiled in *single*-precision mode (i.e., the parameter "`SINGLE`" in `Makefile` was set to `1`), and `double` otherwise. Matrix dimensions are stored in a similar way: `unsigned int` if "`UNSIGNED`" was set to `1`, and '[`signed`] `int` otherwise. All data is stored with the native endianness.

<!-- ALERT: TODO: -->
<!-- All file formats accepted by *NMF-mGPU* are detailed in [*Data-file format*](user_guide.txt.md#fileformat) in the [User guide](user_guide.txt.md), similarly for program usage (section *6 "Utility programs"*). Finally, there are some examples of valid input files in the `test/` folder. -->


<!-- ==================== -->


### 5.2. Matrix Generator

This program generates a data matrix with non-negative random values. The output file can be used as a valid input dataset for NMF-(m)GPU.

You can select the output matrix dimensions, as well as the highest possible random number (i.e., all values will be generated in the closed range between 0.0 and the selected value, both inclusive). The output matrix can be written as ASCII text, or in a binary file (in any of the binary modes described above).

**Warning:** Output matrix will ***not*** contain any tag (i.e., row labels, column headers or description string), just numeric data.

<!-- ALERT: TODO: -->
<!--   * Please, see program usage in section *6 "Utility programs"* in the *User guide*. -->


*****************************


## 6. <a id="setup">*NMF-mGPU* Execution Setup</a>

This section describes how to set up an appropriate environment to execute *NMF-mGPU*.

### 6.1. Execution environment

When *NMF-mGPU* is compiled on a UNIX system, some of the required libraries (e.g, *CUBLAS*) are *dynamically* linked. That is, they are not embedded into the executable file, but the program locates and loads them *at runtime*.

To set up an appropriate execution environment, you can use the provided script `env.sh`, as follow:

         $>  .   ./env.sh   [ <path_to_CUDA_Toolkit> ]

If the argument is not specified, it will be derived by looking for `nvcc` in all folders stored in your `PATH` environment variable.


Please note this script should *not* be "*executed*" (in a sub-shell), but "*sourced*" on the current session by using the command `.`. On some shells, such as `zsh`, you must use the `source` command, instead. That is,

         $> source  ./env.sh  [ <path_to_CUDA_Toolkit> ]

In others (e.g., `dash`), the argument must be previously specified with the `set` command, as follow:

         $> set  --  <path_to_CUDA_Toolkit>
         $>  .  ./env.sh

Finally, this script may not work on (`t`)`csh` shells. If you are not able to use another command-line interpreter (e.g., `bash`, `dash`, `*ksh`, or `zsh`), you can manually set up the environment, as follow:

         $> set  CUDA_HOME="/path/to/CUDA_Toolkit/"
         $> set  PATH="${CUDA_HOME}/bin":${PATH}
         $> set  LD_LIBRARY_PATH="${CUDA_HOME}/lib":${LD_LIBRARY_PATH}

On *Mac OS X*, please replace `LD_LIBRARY_PATH` by `DYLD_LIBRARY_PATH`. That is,

         $> set  DYLD_LIBRARY_PATH="${CUDA_HOME}/lib":${DYLD_LIBRARY_PATH}


**Warnings**:

   * **Folder names containing whitespace characters are *NOT* supported**. Please avoid them in the path to your *CUDA Toolkit* installation directory. A (soft) link can be used as a workaround. For instance:

            $>  ln  -s   /Developer/Whitespaced\ foldername/cuda   /Developer/cuda

   * **Multi-GPU version**: This script does **not** setup any MPI-related environment, since it depends on your actual MPI library. Please make sure it is properly set.



#### Compilation & execution environment in a single step:

The script `env.sh` also exports the environment variable `CUDA_HOME` containing the path to your CUDA Toolkit (either because it was specified as an argument, or because it was derived from `PATH`). As described in [*`Makefile` Parameters*](#mkparams), this variable is required for the compilation process. Therefore, you can set up both, compilation and execution environments, in a single step by executing `env.sh` ***before*** compiling *NMF-mGPU*. That is,

         $> .  env.sh  "/path/to/CUDA"        # Sets up the environment
         $> make                                # Compiles NMF-mGPU
         $> bin/NMF_GPU <input_file>  [...]    # Executes the program

As previously stated, to use the multi-GPU version, please remember to properly setup the environment of your MPI-library.


<!-- ==================== -->


### 6.2. Testing your execution environment

In order to make sure the execution environment has been properly set, you can check if the generated executable files are able to locate all required libraries. According to your UNIX system, this can be performed with one of the following commands:

   * *Linux:*

            ldd  <path_to_executable_file>

   * *Mac OS X:*

            dyldinfo  -dylibs  <path_to_executable_file>


For instance, on a 32-bits Ubuntu Linux with a CUDA Toolkit version 5.5 installed on `/usr/local/cuda-5.5`, the output for `NMF_GPU` (single-GPU version) should be something similar to:

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

         export  CUDA_HOME="/path/to/CUDA_Toolkit/"
         export  PATH="${CUDA_HOME}/bin":${PATH}
         export  LD_LIBRARY_PATH="${CUDA_HOME}/lib":${LD_LIBRARY_PATH}

On *Mac Os X*, please replace `LD_LIBRARY_PATH` by `DYLD_LIBRARY_PATH`. That is,

         export  DYLD_LIBRARY_PATH="${CUDA_HOME}/lib":${DYLD_LIBRARY_PATH}

On some shells, such as (`t`)`csh`, you must use the `set` command, instead.


**Warning**: To use the multi-GPU version, please remember to properly setup the environment of your MPI-library.


*****************************


## 7. <a id="testing">Testing *NMF-mGPU*</a>

In the `test/` folder, you can find different examples of a valid input file. They contain a 5000-by-38 gene-expression data matrix, with or without row/column labels.

For example, to process the file "`test/ALL_AML_data.txt`" with a *factorization rank* of *k=2*, you can use the following command:

         $>  bin/NMF_GPU  test/ALL_AML_data.txt  -k 2  -j 10  -t 40  -i 2000
The rest of arguments denote that:

   * `-j 10`: the *test of convergence* will be performed each 10 iterations.
   * `-t 40`: If there are no relative differences in matrix **H** after 40 *consecutive* convergence tests, it is considered that the algorithm has converged.
   * `-i 2000`: If no convergence is detected, the algorithm stops after 2000 iterations.  
&nbsp;


On the screen, you should see something similar to:

               <<< bioNMF-GPU: Non-negative Matrix Factorization on GPU >>>
                                       Single-GPU version
         Loading input file...
               File selected as ASCII text. Loading...
                        Data matrix selected as having numeric column headers: No.
                        Data matrix selected as having numeric row labels: No.
                        Row labels detected.
                        Number of data columns detected (excluding row labels): 38.
                        Name (i.e., description string) detected.
                        Column headers detected.
                        Loaded a 5000 x 38 data matrix (190000 items).
         Starting NMF( K=2 )...
         NMF: Algorithm converged in 430 iterations.
         Distance between V and W*H: 0.566049

         Saving output file...
               File selected as ASCII text.
         Done.
   In this case, the algorithm converged after a total of 430 iterations.  
   &nbsp;


After completion, both output matrices, **W** and **H**, are stored in the same folder and with a similar filename as the input matrix, but suffixed with `_W` and `_H` respectively. In the example above, such output files would be:

   * `test/ALL_AML_data.txt_W.txt`
   * `test/ALL_AML_data.txt_H.txt`


<!-- ==================== -->


### Multi-GPU version:

The *multi-GPU* version works similarly. Nevertheless, the MPI Standard mandates that all programs must be launched through the `mpiexec` or `mpirun` commands. Using similar arguments as the example above, *NMF-mGPU* can be executed as follow:

         mpiexec  -np 2  bin/NMF_mGPU  test/ALL_AML_data.txt  -k 2  -j 10  -t 40  -i 2000
The argument `-np 2` denotes that *two* GPU devices will be used.

#### Warnings:

   * *All* GPU devices must have a **similar** *Compute Capability*.

   * Please remember to properly setup the environment of your MPI-library.


*****************************


## 8. <a id="troubleshooting">Issues/Troubleshooting</a>


   1. `Dash: env.sh: argument not recognized.`

   2. `Zsh: 'env.sh' not found.`

   3. Trouble invoking `env.sh` on a (`t`)`csh` shell.

   4. Clang on 32-bits System: `error: unknown argument: '-malign-double'.`

   5. `gcc` or `clang`: `Option '-<X>' not recognized.`

   6. `Catastrophic error: could not set locale "" to allow processing of multibyte characters.`

<!--    7.  -->


<!-- ==================== -->

#### 8.1. `Dash: env.sh: argument not recognized.`

On `Dash` shell, the argument must be previously specified with the `set` command:

         $>  set -- <path_to_CUDA_Toolkit>
         $>  . ./env.sh

<!-- ==================== -->

#### 8.2. `Zsh: 'env.sh' not found.`

`Zsh` does *not* recognize the "*dot*" command (`.`). The `source` command must be used, instead. That is,

         $>  source ./env.sh [ <path_to_CUDA_Toolkit> ]

<!-- ==================== -->

#### 8.3. Trouble invoking `env.sh` on a (`t`)`csh` shell.

The script `env.sh` may not work on a (`t`)`csh` shell. If you receive any error message (e.g., `Permission denied.` or `Command not found.`), you can manually set up the environment, as follow:

         $>  set  CUDA_HOME="/path/to/CUDA_Toolkit/"
         $>  set  PATH="${CUDA_HOME}/bin":${PATH}
         $>  set  LD_LIBRARY_PATH="${CUDA_HOME}/lib":${LD_LIBRARY_PATH}

On *Mac Os X*, please replace `LD_LIBRARY_PATH` by `DYLD_LIBRARY_PATH`. That is,

         $>  set  DYLD_LIBRARY_PATH="${CUDA_HOME}/lib":${DYLD_LIBRARY_PATH}

<!-- ==================== -->

#### 8.4. Clang on a 32-bits System: `clang: error: unknown argument: '-malign-double'.`

This switch is not recognized by Clang. Therefore, this compiler **cannot** be used to generate 32-bits code. See more information on the [CUDA Release Notes on compilers][CRN-compiler].

<!-- ==================== -->

#### 8.5. `gcc` or `clang`: `Option '-<X>' not recognized`

Compiler options vary among different tools, versions, and OS. We have tested some of these combinations (e.g., `gcc-4.8` and `clang-3.4` for `Linux`, or `gcc-4.2` and `clang-3.1` for `Mac OS X`), but the list is not exhaustive. If any option is not recognized, just remove it from the `Makefile`.

<!-- ==================== -->

#### 8.6. `Catastrophic error: could not set locale "" to allow processing of multibyte characters.`

Try to change your locales to `en_US.UTF-8` or `en_US.ISO-8859-15`. For instance,

         $>  export  LANG=en_US.UTF-8
         $>  export  LC_ALL=en_US.UTF-8

<!-- ==================== -->

<!-- #### 8.7.  -->


*****************************


## 9. <a id="citation">How to Cite *NMF-mGPU*</a>

If you use this software, please cite one of the following works (the first in the list is preferred):

<!--   * E. Mejía-Roa, D. Tabas-Madrid, J. Setoain, C. García, F. Tirado and A. Pascual-Montano. **NMF-mGPU: Non-negative matrix factorization on multi-GPU systems**. *BMC Bioinformatics* 2015, **1X**:. [doi:](http://dx.doi.org/).  
     <http://www.biomedcentral.com/1471-2105/1X/>  
     &nbsp; -->

   * E. Mejía-Roa, C. García, J.I. Gómez, M. Prieto, C. Tenllado, A. Pascual-Montano and F. Tirado. **Parallelism on the Nonnegative Matrix Factorization**. In *Applications, Tools and Techniques on the Road to Exascale Computing*. IOS Press BV; 2012:421-428. \[*Advances in Parallel Computing*, vol. 22]. [doi:10.3233/978-1-61499-041-3-421](http://dx.doi.org/10.3233/978-1-61499-041-3-421).  
     <http://ebooks.iospress.nl/publication/26557>  
     &nbsp;

   * E. Mejía-Roa, C. García, J.I. Gómez, M. Prieto, R. Nogales, F. Tirado and A. Pascual-Montano. **Biclustering and classification analysis in gene expression using Nonnegative Matrix Factorization on multi-GPU systems**. In *Proceedings of the 11th International Conference on Intelligent Systems Design and Applications* (*ISDA*). IEEE Press; 2011:882-887. [doi:10.1109/ISDA.2011.6121769](http://dx.doi.org/10.1109/ISDA.2011.6121769).  
     <http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6121769>  
     &nbsp;


<!-- ==================================================== -->
 <br/>
 <br/>
 </body>
 </html>

<!--
// kate: backspace-indents off; indent-mode normal; indent-width 3; keep-extra-spaces off; newline-at-eof on; replace-trailing-space-save off; replace-tabs on; replace-tabs-save on; remove-trailing-spaces none; tab-indents on; tab-width 4;
-->
