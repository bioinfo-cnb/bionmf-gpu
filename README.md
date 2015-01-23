<!--
 ************************************************************************
 *
 * NMF-mGPU - Non-negative Matrix Factorization on multi-GPU systems.
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
   <meta name="description" content="Non-negative Matrix Factorization (NMF) for Biology on multi-GPU systems."/>
   <meta name="keywords" content="bioNMF, NMF, Matrix factorization, GPU, multi-GPU, GPGPU, NVIDIA, CUDA, CUBLAS, Bioinformatics"/>
   <meta name="language" content="en"/>
   <meta name="copyright" content="(C) 2011-2014 Edgardo Mejia-Roa (edgardomejia@fis.ucm.es). ArTeCS Group, Complutense University of Madrid (UCM), Spain."/>
   <meta http-equiv="content-Type" content="text/html; charset=UTF-8"/>
   <meta http-equiv="last-modified" content="2015/01/03" scheme="YYYY/MM/DD"/>
   <link rel="stylesheet" type="text/css" href="doc/styles.css"/>
   <title>NMF-mGPU: Non-negative Matrix Factorization on multi-GPU systems</title>
 </head>
 <body>

<!-- ==================================================== -->

# *NMF-mGPU*: Non-negative Matrix Factorization on multi-GPU Systems

 ***NMF-mGPU*** implements the ***Non-negative Matrix Factorization*** (***NMF***) algorithm by making use of ***Graphics Processing Units*** (***GPUs***). NMF takes an input matrix (**V**) and returns two matrices, **W** and **H**, whose product is equal to the former (i.e., **V** ~ **W** \* **H**). If **V** has *n* rows and *m* columns, then dimensions for **W** and **H**, will be *n* × *k* and *k* × *m*, respectively. The *factorization rank* ("*k*") specified by the user, is usually a value much less than both, *n* and *m*.

   This software has been developed using the NVIDIA's [***CUDA***][CUDA_homepage] ([***Compute Unified Device Architecture***][CUDA_homepage]) framework for GPU Computing. *CUDA* represents a GPU device as a programmable general-purpose *coprocessor* able to perform linear-algebra operations.

   On detached devices with low on-board memory available, large datasets can be blockwise transferred from the CPU's main memory to the GPU's memory and processed accordingly. In addition, *NMF-mGPU* has been explicitly optimized for the different CUDA architectures.

   Finally, *NMF-mGPU* also provides a *multi-GPU* version that makes use of multiple GPU devices through the [***MPI***][MPI_homepage] ([***Message Passing Interface***][MPI_homepage]) standard.


[CUDA_homepage]: <http://www.nvidia.com/object/cuda_home_new.html> "CUDA Homepage"
[MPI_homepage]: <http://mpi-forum.org/> "MPI Forum"


*****************************  
*****************************


# QUICK START GUIDE

Basic steps:

   1. [Install the NVIDIA CUDA Toolkit and Drivers](#install).
   2. [Download, Decompress and Compile ***NMF-mGPU***](#compile).
   3. [Example of Use](#example).

The full [installation guide](doc/installation_guide.txt.md) can be found in the '`doc`' folder.  
&nbsp;


<!-- ==================== -->


## 1. <a id="install">Install the NVIDIA CUDA Toolkit and Drivers</a>


### 1.1 System Requirements

   * **UNIX System (GNU/Linux or Darwin/Mac OS X)**. *NMF-mGPU* has not been tested on Microsoft Windows yet.

   * One or more **CUDA-capable GPU device(s)**: A detailed list of compatible hardware can be found at <http://developer.nvidia.com/cuda-gpus>  
	 Please note that **all** devices must be of the same architecture (i.e., heterogeneous GPU clusters are not supported yet).

   * **CUDA Toolkit and CUDA Driver**: They are freely available at the [CUDA Downloads Page][CUDA-Download]. Nevertheless, for *deprecated* GPU devices and/or OS platforms, you can download a previous CUDA release (e.g., version `5.5`) from the [CUDA Archive Page][CUDA-OR-Download]. Please note that *NMF-mGPU* requires, at least, the version `4.2`.

   * A `C` compiler **conforming to the `C99` standard**, such as [GNU GCC](https://gcc.gnu.org) or [LLVM Clang](http://llvm.org/).

   * The ***optional* multi-GPU version** also requires an *MPI-2.0* (or greater) software library, such as [OpenMPI](http://www.open-mpi.org/) or [MPICH](http://www.mpich.org/).


[CUDA-Download]: <http://developer.nvidia.com/cuda-downloads/> "CUDA Download Page"
[CUDA-OR-Download]: <https://developer.nvidia.com/cuda-toolkit-archive/> "CUDA Archive Page"


<!-- ==================== -->


### 1.2 Install Instructions

   * For **GNU/Linux**:

     Instructions vary among different distributions. For instance, on **Ubuntu 14.04**:

	   + **NVIDIA proprietary driver**: Open the program *Software & Updates*, then go to *Additional Drivers* section and check the *Using NVIDIA binary driver* option.  
		 Alternatively, you can open a terminal and type:

				sudo  apt-get  install  nvidia-current

		 You may have to reboot the system in order to use this driver after installing it.

	   + **Additional packages**: The following packages are required: `build-essential`, `nvidia-cuda-dev` and `nvidia-cuda-toolkit`.  
		 They can be installed through the *Ubuntu Software Center*, or via a terminal by typing: 

				sudo  apt-get  install  build-essential  nvidia-cuda-dev  nvidia-cuda-toolkit

	   + **Multi-GPU version (optional)**: This version also requires any of the following packages: `openmpi` or `mpich`.

    For other GNU/Linux distribution, we recommend to read the [Getting Starting Guide for GNU/Linux](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/index.html#package-manager-installation).


   * For **Darwin/Mac OS X**:

	   + **`C`/`C++` compiler**: Please install the Apple's [Xcode](https://developer.apple.com/xcode/downloads/) toolset. Some versions may require to explicitly add the *Command Line Developer Tools* plug-in in order to make available the required commands on the Terminal.

	   + **CUDA Toolkit and Drivers**: Just download and execute the proper `.dmg` file from the [CUDA Download Page][CUDA-Download] (or the [Archive Page][CUDA-OR-Download] for previous releases), and follow the instructions.

	   + **Multi-GPU version (optional)**: Most MPI libraries are available on package managers, such as [MacPorts](http://www.macports.org/) or [Homebrew](http://brew.sh/).

	We highly recommend to read the [Getting Starting Guide for Darwin/Mac OS X](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-mac-os-x/index.html) for detailed instructions.


### Warning:

  **Folder names containing whitespace characters are *NOT* supported**. Please avoid them in the path to your *CUDA Toolkit* installation directory. A (soft) link can be used as a workaround. For instance:

		$>  ln  -s   /Developer/Whitespaced\ foldername/cuda   /Developer/cuda



*****************************


## 2. <a id="compile">Download, Decompress, and Compile ***NMF-mGPU***</a>


 ***NMF-mGPU*** can be downloaded from two main sources:

   * From the [*Releases* Page](https://github.com/bioinfo-cnb/bionmf-gpu/releases) for the latest and previous releases.
   * From the [Applications Page](http://bioinfo-cnb.github.io/bionmf-gpu/) to get the latest development code.
&nbsp;


Once the file has been unzipped, please open a Terminal in the generated folder and execute the following command:

		$>  source  env.sh  <CUDA_PATH>
where "`<CUDA_PATH>`" denotes the path to your CUDA Toolkit (e.g., `/usr/local/cuda-5.5` or `/Developer/NVIDIA/cuda-5.5`).


To compile the program, just execute:

		 $>  make 


The compilation process may take some time since different versions of the software (one per GPU model) will be generated into the executable file. To compile code for just a particular GPU architecture (e.g., for *Compute Capability 1.3*), you can use the following command:

		 $>  make  SM_VERSIONS=13
Please note that by default, **no code is compiled for *Compute Capabilities* 1.2 and lower**, since they are being deprecated on newer versions of the CUDA Toolkit. Code for such GPU models must be explicitly requested as shown in the example above.  
&nbsp;



The resulting executable file, '`NMF_GPU`', will be stored in the `bin/` folder.  
&nbsp;


Finally, to compile the *multi-GPU* version (process not performed by default), just execute:

		 $>  make  multi_gpu  MPICC=<path_to_mpi>/bin/mpicc

Please make sure your MPI Library is properly installed (e.g., environment variables, etc). Similarly as above, the resulting executable file, '`NMF_mGPU`' is stored in the `bin/` folder.


*****************************


## 3. <a id="example">Example of use</a>

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


After completion, both output matrices, **W** and **H**, are stored in the same folder and with a similar filename as the input matrix, but suffixed with '`_W`' and '`_H`' respectively. In the example above, such output files would be:

   * `test/ALL_AML_data.txt_W.txt`
   * `test/ALL_AML_data.txt_H.txt`


<!-- ==================== -->


### Multi-GPU version:

The *multi-GPU* version works similarly. Nevertheless, the MPI Standard mandates that all programs must be launched through the `mpiexec` or `mpirun` commands. Using similar arguments as the example above, *NMF-mGPU* can be executed as follow:

		 mpiexec  -np 2  bin/NMF_mGPU  test/ALL_AML_data.txt  -k 2  -j 10  -t 40  -i 2000
The argument '`-np 2`' denotes that *two* GPU devices will be used.

#### Warnings:

   * *All* GPU devices must have a **similar** *Compute Capability*.

   * Please, remember to properly setup the environment of your MPI-library.



<!-- ==================================================== -->
 <br/>
 <br/>
 </body>
 </html>

<!--
// kate: backspace-indents off; indent-mode normal; indent-width 3; keep-extra-spaces off; newline-at-eof on; remove-trailing-space off; replace-trailing-space-save off; replace-tabs off; replace-tabs-save off; remove-trailing-spaces none; tab-indents on; tab-width 4;
-->
