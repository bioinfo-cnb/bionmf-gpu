<!--
 ************************************************************************
 *
 * NMF-mGPU - Non-negative Matrix Factorization on multi-GPU systems.
 *
 * Copyright (C) 2011-2015:
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
   <meta name="description" content="Non-negative Matrix Factorization (NMF) for Biology on multi-GPU systems. User guide"/>
   <meta name="keywords" content="bioNMF, NMF, Matrix factorization, GPU, multi-GPU, GPGPU, NVIDIA, CUDA, CUBLAS, Bioinformatics"/>
   <meta name="language" content="en"/>
   <meta name="copyright" content="(C) 2011-2014 Edgardo Mejia-Roa (edgardomejia@fis.ucm.es). ArTeCS Group, Complutense University of Madrid (UCM), Spain."/>
   <meta http-equiv="content-Type" content="text/html; charset=UTF-8"/>
   <meta http-equiv="last-modified" content="2015/02/20" scheme="YYYY/MM/DD"/>
   <link rel="stylesheet" type="text/css" href="styles.css"/>
   <title>NMF-mGPU user guide</title>
 </head>
 <body>


<!-- ==================================================== -->


# *NMF-mGPU* USER GUIDE

This documents shows how to use *NMF-mGPU*. To install and configure this program, please read first the [installation guide](installation_guide.txt.md) located in this folder.

**Index:**

   1. [Introduction](#intro).
   2. [*NMF-mGPU* usage](#usage).
   3. [Data File Format](#fileformat).
   4. [Analysis process](#analysis).
   5. [*NMF-mGPU* Setup & Execution](#setupexecution)
   6. [Utility Programs](#tools).
   7. [Issues/Troubleshooting](#troubleshooting).
   8. [How to Cite *NMF-mGPU*](#citation).


*****************************


## 1. <a id="intro">Introduction</a>

***NMF-mGPU*** implements the ***Non-negative Matrix Factorization*** (***NMF***) algorithm by making use of ***Graphics Processing Units*** (***GPUs***). NMF takes an input matrix (**V**) and returns two matrices, **W** and **H**, whose product is equal to the former (i.e., **V** ~ **W** \* **H**). If **V** has *n* rows and *m* columns, then dimensions for **W** and **H**, will be *n* × *k* and *k* × *m*, respectively. The *factorization rank* ("*k*") specified by the user, is usually a value much less than both, *n* and *m*.

This software has been developed using the NVIDIA's [***CUDA***][CUDA_homepage] ([***Compute Unified Device Architecture***][CUDA_homepage]) framework for GPU Computing. *CUDA* represents a GPU device as a programmable general-purpose *coprocessor* able to perform linear-algebra operations.

On detached devices with low on-board memory available, large datasets can be blockwise transferred from the CPU's main memory to the GPU's memory and processed accordingly. In addition, *NMF-mGPU* has been explicitly optimized for the different CUDA architectures.

Finally, *NMF-mGPU* also provides a *multi-GPU* version that makes use of multiple GPU devices through the [***MPI***][MPI_homepage] ([***Message Passing Interface***][MPI_homepage]) standard.


[CUDA_homepage]: <http://www.nvidia.com/object/cuda_home_new.html> "CUDA Homepage"
[MPI_homepage]: <http://mpi-forum.org/> "MPI Forum"


*****************************


## 2. <a id="arguments">*NMF-mGPU* Usage</a>

Program usage (valid for both single- and multi-GPU versions):

         $>  bin/NMF_GPU <filename> [ -b <native> ] [ -c ] [ -r ] [ -k <factorization_rank> ]
                                    [ -i <nIters> ] [ -j <niter_test_conv> ]
                                    [ -t <stop_threshold> ] [ -e <native> ] [ -z <GPU_device> ]
or, for a help message:

         $>  bin/NMF_GPU -h
&nbsp;

The multi-GPU version accepts the same arguments. Nevertheless, the MPI standard mandates that every program must be invoked through the commands `mpirun` or `mpiexec`. Although most of their arguments depend on your actual MPI library, a typical invocation command would be the following:

         $>  mpirun  -np 2  bin/NMF_mGPU  <filename>  [ ... ]

In this example, the argument `-np 2` specifies that *two* GPU devices will be used.


<!-- ==================== -->


### 2.1. <a id="argumentsformatrix">Arguments for input and output file formats</a>

   * *`<filename>`*: Input data matrix (mandatory if *help* is not requested).

   * `-B`,`-b` *`<native>`*: Input file is *binary*, in "***native***" (`-b 1`) or "***non-native***" format (`-b 0`).  
     In **non-*native*** format, the file is read assuming it was written using little-endian *double*-precision data, and 32-bits *unsigned* integers for matrix dimensions, regardless how the program was compiled. The file must also contain a *binary signature*, which will be checked (as well as all numeric data) to make sure it is a valid input file.  
     Otherwise, in ***native*** or *raw* format, the file is read using the native endianness and the data types specified at compilation. Please note this mode **skips** most error checking and information messages.  
     See section "[*Data File Format*](#fileformat)" for details.  
     The default (i.e., if `-b` is not specified) is to read input data from an *ASCII-text* file.  

   * `-C`,`-c`: Input file has *numeric* column headers (***disabled*** by default, ignored for *binary* files).

   * `-R`,`-r`: Input file has *numeric* row labels (***disabled*** by default, ignored for *binary* files).

   * `-E`,`-e` *`<native>`*: Writes output files in "***native***" (`-e 1`) or "***non-native***" *binary* format (`-e 0`).  
     In **non-*native*** format, the file is written using little-endian *double*-precision data, and 32-bits *unsigned* integers for matrix dimensions, regardless how the program was compiled. In addition, a file signature is written to the file.  
     Otherwise, in ***native*** or *raw* format, the file is written using the native endianness, and the data types specified at compilation. Please note this mode **skips** error checking, information messages, and data transformation (e.g., matrix transposing).  
     See "[*Data File Format*](#fileformat)" for details.  
     The default (i.e., if `-e` is not specified) is to write output data to an *ASCII-text* file.

#### Output filenames:

After completion, both output matrices, **W** and **H**, are stored in the same folder and with a similar filename as the input matrix, but suffixed with `_W` and `_H` respectively. Finally, a file extension is appended according to the specified arguments. That is,

   * *`<input_filename>_{W,H}.txt`*: If the file was saved as ASCII text (i.e., if no `-e` argument was specified).

   * *`<input_filename>_{W,H}.dat`*: If the file was saved as "*non-native*" binary (i.e., argument `-e 0`).

   * *`<input_filename>_{W,H}.native.dat`*: If the file was saved as "*native*" binary (i.e., argument `-e 1`).

These extensions are defined in `include/common_defaults.h`.


<!-- ==================== -->


### 2.2. Options to control the NMF algorithm

   * `-K`,`-k` *`<factorization_rank>`*: Factorization Rank. It must be at least `2`, but not greater than any of both matrix dimensions. Default value: `2`.

   * `-I`,`-i` *`<nIters>`*: Maximum number of iterations if the algorithm does not converge to a stable solution. Default: `2000`.

   * `-J`,`-j` *`<niter_test_conv>`*: Performs a convergence test every *`<niter_test_conv>`* iterations. Default: `10`.  
     If this value is greater than *`<nIters>`* (`-i` option), *no* test is performed. See "[*Test of Convergence*](#convergence)" for details.

   * `-T`,`-t` *`<stop_threshold>`*: Stopping threshold. Default value: `40`.  
     If matrix **H** has not changed on the last *`<stop_threshold>`* times that the convergence test has been performed, it is considered that the algorithm *has converged* to a solution and stops it. See "[*Test of Convergence*](#convergence)" for details.


<!-- ==================== -->


### 2.3. Other options

   * `-h`,`-H`: Prints a help message with all arguments.

   * `-Z`,`-z` *`<GPU_device>`*: Device ID to attach on (default: `0`).  
     On the multi-GPU version, devices will be selected from this value.  
     For instance,

            $>  mpirun  -np 4  bin/NMF_mGPU  <filename>  [ ... ]  -z 2
     refers to make use of *four* devices, starting from *device ID 2* (i.e., devices 2, 3, 4, and 5).  
     **Warnings:**
       + No modular arithmetic is performed.
       + All devices must have the *same* Compute Capability.



*****************************


## 3. <a id="fileformat">Data File Format</a>

*NMF-mGPU* is able to work with matrices stored in a binary or text file. This section describes both file formats, which can be used for input or output data.

Regardless of the format, the file must contain *non-negative* data only. Optionally, there can be column headers, and/or row labels, as well as a short description string (denoted as *"matrix name"*). Any of such three *tag elements* is independent of each other, and may or may not appear in the file.

***Note:*** it is *not* allowed to have any column or row with *all* entries set to zero, there must be at least one *positive* value on each. *NMF-mGPU* will try to detect and report it as an invalid file format.

Some examples of valid input files, in both formats, can be found in the `test/` folder. In addition, some utility programs (such as a text-binary file converter) are provided in order to make easier working with input and output files. Please, see "[*Utility Programs*](#tools)" for details.


<!-- ==================== -->


### 3.1. ASCII-text files

In this format, matrix data are separated by (single) *tab* characters. Matrix dimensions are then inferred from the text layout (i.e., the number of data per line and the number of lines) in the file. The optional column headers must be placed on the first row, just after the description string (if any). The (also optional) row labels must be placed at the beginning of each line. Finally, each header, label, or the description string, can be composed by several *space-separated* words.

A text layout scheme, for an *n*-by-*m* data matrix, would be the following:

         [ [Description string  <tab>]  column header 1  <tab>  ...  <tab>  col hdr m  <newline> ]
           [ row label 1        <tab>]  value_11         <tab>  ...  <tab>  value_1m   <newline>
           ...
           [ row label n        <tab>]  value_n1         <tab>  ...  <tab>  value_nm   <newline>

Nevertheless, this is a *simplified* scheme, since actually, every item is optional. For instance, two consecutive tab characters (or, a line beginning or ending with a tab) denote an *empty entry*, which is then processed, either as `0.0` or as an *empty string*, as necessary. Note that a such string is still allocated in memory (it stores a null byte), and will be set as a tab character in any output file. To *completely skip* any or all of the three tag elements (i.e., that no memory be allocated), just set directly the next element, without the delimiter tab. For instance, for a numeric-only data file (i.e., with no tags at all), just start the file (and every line) with the corresponding matrix datum. An example of such file can be found in the `test/` folder.

***Important notes:***

   * Please do **not** use double- (`"`) or single-quote (`'`) characters in any tag element.

   * By default, *NMF-mGPU* follows the system-default locale (e.g., `C` or `UTF-8` on UNIX platforms). In particular, it uses *dots* (`.`) as decimal symbols, and reports a comma (`,`) as an invalid character.

   * Only UNIX (`LF`, `\n`) and MS-DOS/Windows (`CR+LF`, `\r\n`), end-of-line styles are accepted.


#### Numeric row labels and/or column headers:

By default, *NMF-mGPU* processes numeric row labels as matrix data. To prevent this, please add `-r` to the command-line arguments. This option forces *NMF-mGPU* to process the first data column in the file, as row labels.  
Similarly, to prevent a misinterpretation of numeric column headers and/or the description string, please use the `-c` option.

Please note that such options are *not* required for *binary files*, since there is no way to accidentally mix matrix data and tag elements.

A list of all available arguments can be found on "[*NMF-mGPU Usage*](#usage)".


#### Whitespace characters instead of tabs:
Under the following conditions, input data may be separated by *single* whitespace characters ('` `'):

   * *All* row labels and column headers must contain only a single word, each. Similarly for the description string.

   * There must *not* be any *tab* character in the file.

   * *Consecutive space characters* will be processed as `0.0` or an empty string, as necessary.

   * Both, first and second rows must *not* begin with an *empty* column header and/or row label (i.e., they must not start with a whitespace character), since they will be misinterpreted as null data (i.e., as `0.0`). However, you can force a header/label detection with the `-r` and/or `-c` options (see "[*NMF-mGPU Usage*](#usage)" for details).


<!-- ==================== -->


### 3.2. Binary files

In addition to ASCII-text files, *NMF-mGPU* also accepts a data matrix stored in a binary format. Such binary files, can be read and/or written in two sub-formats: "*native*" and "*non-native*".

   * "***Native* mode**" refers to *raw* I/O. That is, data are stored/loaded with the native endianness, and the precision specified at compilation time: `float` values if the program was compiled in *single*-precision mode (i.e., if the [`Makefile` parameter](installation_guide.txt.md#mkparams) `SINGLE` was set to `1`), and `double`, otherwise. Matrix dimensions are stored/loaded in a similar way: `unsigned int`, if `UNSIGNED` was set to `1`; '[`signed`] `int`, otherwise. This mode is faster because *no* error checking is performed. Therefore, **it should *not* be used to read untrusted input files**.  
     **Important note:** In *output* files, this mode also *skips* all data transformation steps (e.g., *matrix transposing*). In particular, matrix **H** (which is computed in transposed mode due to performance reasons) will *not* be "restored" before writing the output file.  
     A full list of available [`Makefile` parameters](installation_guide.txt.md#mkparams) can be found in the [user guide](installation_guide.txt.md).


   * "***Non-native***" **mode**: Matrix data are *always* stored using little-endian *double*-precision values, and 32-bits *unsigned* integers for matrix dimensions. This is the recommended mode for input or final output data, since every datum is checked. Finally, the file also contains a *binary signature*, which is also checked when reading to make sure it is a valid input file. The signature values are defined in `src/matrix_io/matrix_io.c`.


In both modes, all *binary* data will be read/written in the following order:

   * **Number of rows** (*n*): integer.

   * **Number of columns** (*m*): integer.

   * **Matrix data:** *n*-by-*m* (double/single)-precision floating-point values stored in *row-major* order (i.e., contiguous elements belong to the same matrix *row*).

The rest of the file contains the optional column headers, row labels and description string. If set, they must be written in *ASCII-text* format, right after the last binary matrix datum, as follows:

   * **Row labels**: *n* tab-separated strings.

   * A newline character (`\n` or `\r\n`).

   * **Column headers**: *m* tab-separated strings.

   * A newline character (`\n` or `\r\n`).

   * ***Matrix name***: the short description string.

Similarly to ASCII-text files, consecutive, initial, or ending tab characters denote an *empty string*. To fully skip any of the three tag elements, directly write the next newline character (or just close the file, to skip all the subsequent items). For instance, to set only a *matrix name*, with no labels or headers at all, just write the two newline characters before the description string. To skip all three tag elements, just close the file after the last binary matrix datum.

***Notes***:

   * It is not necessary to specify the options `-r` or `-c` to indicate numeric labels and/or headers. In fact, such arguments will be *ignored*.

   * Please do *not* use double- (`"`) or single-quote (`'`) characters in any *tag* element.

   * Only UNIX (`LF`, `\n`) and MS-DOS/Windows (`CR+LF`, `\r\n`), end-of-line styles are accepted.

   * Please remember that the [utility programs](#tools) can be used for binary-text file conversions.


<!-- ==================== -->


### 3.3. <a id="matrixlimits">Limits on Matrix Dimensions</a>

#### Upper limits: 

There are different factors that may limit the maximum dimensions for an input matrix:

   * **Data type for matrix dimensions**: *NMF-mGPU* can be configured to make use of *unsigned* integers for matrix dimensions. In contrast to *signed* integers, the former allow to process up to four times larger input matrices (twice larger per dimension). In addition, they may generate faster kernel code (e.g., by avoiding cast conversions with other unsigned types, such as `size_t`). Data-type signedness can be specified at compile time through the [`Makefile` parameter](installation_guide.txt.md#mkparams): *`UNSIGNED`*.


   * **The cuBLAS Library**: *NMF-mGPU* makes use of the functions provided by the [*cuBLAS library*](https://developer.nvidia.com/cublas) to perform all matrix products. Similarly to others [BLAS](http://www.netlib.org/blas/)-like libraries, *cuBLAS* uses *signed* integers for indexing (at least, on its "*regular*" API). Therefore, matrix dimensions are limited to the maximum representable signed value (2<sup>31</sup>-1 on most architectures).


   * **Available GPU memory**: On detached GPU devices equipped with a dedicated on-board memory, *NMF-mGPU* is able to process large input matrices by blockwise transferring them from the CPU's memory to the GPU's memory. However, there must be enough GPU memory available to transfer at least one entire row *and* one full column. In addition, output matrices **W** and **H** (as well as other similar temporary data) that depend on both, matrix dimensions and the *factorization rank* (option `-k`), must be fully loaded into the GPU memory. *NMF-mGPU* will try to reduce the number of rows and columns of the input matrix transferred at a time, in order to satisfy all these constraints.


   * **GPU architecture**: Regardless of the available GPU (*global*) memory, and the data type used for matrix dimensions, the size of matrices that a kernel is able to process, is limited by the architecture of the actual device (e.g., threads per block, grid size, address space, etc). This includes the **factorization rank** (option `-k`), which **must not be greater than the maximum number of threads per block** (512 on *Compute Capability 1.x*, and 1024 otherwise).


   * **Items per thread**: In order to mitigate the limits imposed by the GPU architecture (as well as for performance reasons), all kernels have been designed such that each thread may process multiple items. This is controlled by some constants in the code (named *`<kernel_name>__ITEMS_PER_THREAD`*) defined in `include/GPU_kernels.cuh`. Their default values are set to ensure a 100% of occupancy on devices of *Compute Capability 1.x*, so they can be increased if the program will be compiled for newer GPU architectures. Note, however, this excludes the *factorization rank* (option `-k`), which is still limited to the maximum number of threads per block.


#### Lower limits:

The NMF algorithm requires that any data matrix must have, at least, *two* columns and *two* rows (in addition to row labels and column headers, if any). Similarly, the *factorization rank* (option `-k`) must be greater than or equal to `2`. *NMF-mGPU* will check such limits and notify if they are underflow.


*****************************


## 4. <a id="analysis">Analysis process</a>

This section describes the NMF algorithm implemented on *NMF-mGPU*, as well as all related command-line options that can be used to customize the analysis process.


<!-- ==================== -->


### 4.1 The NMF algorithm:

NMF takes an input matrix (**V**) and returns two matrices, **W** and **H**, whose product is equal to the former (i.e., **V** ~ **W** \* **H**). If **V** has *n* rows and *m* columns, then dimensions for **W** and **H**, will be *n* × *k* and *k* × *m*, respectively. The *factorization rank* (*"k"*) specified by the user, is usually a value much less than *n* and *m*.

NMF iteratively modifies **W** and **H** until their product approximates to **V**. Such modifications, composed by matrix products and other algebraic operations, are derived from minimizing a cost function that quantifies, in some way, the differences between **W** \* **H** and **V**. There are numerous objective functions, each leading to different update rules.

The following pseudo-code shows the NMF algorithm as it was implemented in *NMF-mGPU*:

         NMF_GPU( input matrix V, <n> rows, <m> columns, <k> factors ):

            W  =  random_init( n, k )
            H  =  random_init( k, m )

            REPEAT:

                  accum_W  =  reduce_to_a_row( W )
                  aux      =  V  ./  (W * H)
                  H_tmp    =  Wt * aux
                  H        =  H  .*  H_tmp  ./  (accum_W)t

                  accum_H  =  reduce_to_a_column( H )
                  aux      =  V  ./  (W * H)
                  W_tmp    =  aux * Ht
                  W        =  W  .*  W_tmp  ./  (accum_H)t

            UNTIL  has_converged()

            RETURN  W and H

Symbols `.*` and `./` denote pointwise matrix operations. In addition, `Ht` and `Wt` represent transposed matrices. Similarly for the *k*-length vectors `(accum_H)t` and `(accum_W)t`.

The update rules were taken from the following works:

   * Daniel D. Lee, H. Sebastian Seung. **Algorithms for Non-negative Matrix Factorization**. In *Advances in Neural Information Processing Systems 13 (Proceedings of the NIPS 2000 Conference)*. Cambridge, MA: The MIT Press; 2001. p. 556-562. \[<http://hebb.mit.edu/people/seung/papers/nmfconverge.pdf>]

   * JP Brunet, P Tamayo, TR Golub, JP Mesirov. **Metagenes and molecular pattern discovery using matrix factorization**. In *Proc Natl Acad Sci USA*. 2004; **101**:4164-9.


<!-- ==================== -->


### 4.2 <a id="convergence">Test of Convergence</a>

The test of convergence computes the assignment of samples to each *factor*, which is represented in matrix **H** by the *row index* of the maximum value for each column. This *sample classification* is then compared with the one computed on the previous convergence test. If no differences are found after a certain number of consecutive tests, the algorithm is considered to have converged to a stable solution.

The test of convergence can be controlled with the following command-line arguments:

   * `-J`,`-j` *`<niter_test_conv>`*: Performs a convergence test every *`<niter_test_conv>`* iterations. Note that, if this value is greater than the maximum number of iterations (option `-i`), no test is performed. The default value is to perform a convergence test every **10** iterations.

   * `-T`,`-t` *`<stop_threshold>`*: Stopping threshold. Number of tests with similar results to consider the algorithm as converged. The default value is to stop after **40** convergence tests without differences.


#### Implementation:

The *stability* of the sample classification is measured in terms of the samples that are grouped on the same *factor*, rather than the particular factor to which each sample belongs.

   Conceptually, this is is quantified with a *connectivity matrix* (**C**) of size *m*-by-*m*, where "*m*" is the number of columns of matrix **H**. Each entry *C(i,j)* is set to `1` if columns *i* and *j* in matrix **H** have their maximum value on the same row (i.e., if both samples belong to the same *factor*), and `0` otherwise. Note that the actual factor they belong to, is never specified. A stable sample classification then results in obtaining *exactly* the same connectivity matrix among different convergence tests.

   However, since matrix **C** is symmetric, it is never stored in memory. Instead, the algorithm just keeps the sample assignments (yes, the particular *factors*) in an *m*-length classification vector. This way, two entries, *classification(i)* and *classification(j)*, with the same value, would represent a non-zero in *C(i,j)*. This equivalence is used to partially build the connectivity matrix when "comparing" the classification vector with the one resulting from the previous convergence test.


For more details about the Consensus method, please read the following papers:

   * E. Mejía-Roa, P. Carmona-Saez, R. Nogales, C. Vicente, M. Vázquez, X. Y. Yang, C. García, F. Tirado and A. Pascual-Montano. ***bioNMF:* a web-based tool for non-negative matrix factorization in biology**. In *Nucl Acids Res* 2008, **36**(suppl 2):W523-W528. doi:10.1093/nar/gkn335 \[<http://nar.oxfordjournals.org/cgi/content/full/36/suppl_2/W523>] \[<http://bionmf.dacya.ucm.es>]

   * JP Brunet, P Tamayo, TR Golub, JP Mesirov. **Metagenes and molecular pattern discovery using matrix factorization**. In *Proc Natl Acad Sci USA*. 2004; **101**:4164-9.


*****************************


## 5. <a id="setupexecution">*NMF-mGPU* Setup & Execution</a>


<!-- ==================== -->


### 5.1 Execution Environment

Before executing *NMF-mGPU*, it is necessary to properly set up the shell environment. This can be easily done with the script `env.sh`. Please follow the instructions described on section "[*NMF-mGPU Execution Setup*](installation_guide.txt.md#setup)" in the [*installation guide*](installation_guide.txt.md).


<!-- ==================== -->


### 5.2 Example of use

A detailed example of use can be found on section "[*Testing NMF-mGPU*](installation_guide.txt.md#testing)" in the [*installation guide*](installation_guide.txt.md).


*****************************


## 6. <a id="tools">Utility Programs</a>

In addition to *NMF-mGPU*, there are some utility programs to make easier working with input files. It includes a program for binary-text file conversion, and another to generate input matrices with random data (useful for testing).

To compile such programs, just execute:

        $> make tools

which will generate the following files:

   * `bin/tools/file_converter`  ─ Binary-text file conversions.
   * `bin/tools/generate_matrix` ─ Program to generate a matrix with synthetic data.

***Note:*** Tool programs do *not* make use of any GPU device. They have been implemented in pure ISO-C language, so all operations are performed on the *host* (i.e., the CPU). Therefore, they do *not* require any CUDA-related option, configuration or software.


<!-- ==================== -->


### 6.1. Binary-text file converter

As stated on section "[*Data File Format*](#fileformat)", *NMF-mGPU* accepts input matrices stored in binary or ASCII-text files. This program allows file conversion between both formats.


The program accepts the same arguments as *NMF-mGPU* concerning the file formats. That is,

         $>  bin/tools/file_converter <filename> [ -b <native> ] [ -c ] [ -r ] [ -e <native> ]

or, for a help message:

         $>  bin/tools/file_converter -h

Please see their usage on section "[*Arguments for input and output file formats*](#argumentsformatrix)".


#### Output filenames:

The output filename is composed by the input filename plus an additional extension, which is set according to the specified arguments. That is,

   * *`<input_filename>.txt`*: If the file was saved as ASCII text (i.e., if no `-e` argument was specified).

   * *`<input_filename>.dat`*: If the file was saved as "*non-native*" binary (i.e., argument `-e 0`).

   * *`<input_filename>.native.dat`*: If the file was saved as "*native*" binary (i.e., argument `-e 1`).

The extensions are defined in `include/common_defaults.h`.


<!-- ==================== -->


### 6.2. Matrix Generator

This program generates a data matrix with non-negative random values. The output file can be used as a valid input dataset for *NMF-mGPU*. You can specify the output matrix dimensions, as well as the highest possible random number (i.e., all values will be generated in the closed range between `0.0` and the selected value, both inclusive).

Program usage:

         $>  bin/tools/generate_matrix <output_filename> <rows> <columns>
                                       [ -e <native> ] [ <max_value> ]

or, for a help message:

         $>  bin/tools/generate_matrix -h


Accepted arguments:

   * *`<output_filename>`*: Output file name (mandatory if *help* is not requested).

   * *`<rows> <columns>`*: Output matrix dimensions (both are mandatory if *help* is not requested)

   * *`<max_value>`*: Highest possible random number. That is, random number will be in range `[0.0 ... <max_value>]`.  
     Default value: `1.0`

   * `-E`,`-e` *`<native>`*: Writes output files in "***native***" (`-e 1`) or "***non-native***" *binary* format (`-e 0`).  
     See "[*Data File Format*](#fileformat)" for details.  

**Warning:** Output matrix will ***not*** contain any *matrix tag* (i.e., row labels, column headers or description string), just numeric data. A detailed description of such elements can be found on section "[*Data File Formats*](user_guide.txt.md#fileformat)" in the [user guide](user_guide.txt.md).


*****************************


## 7. <a id="troubleshooting">Issues/Troubleshooting</a>


   1. Input matrix not recognized: `Data in column <X> is not numeric`, or `illegal character`.

   2. A large input matrix could not be processed.

<!-- ==================== -->

#### 7.1 Input matrix not recognized: `Data in column <X> is not numeric`, or `illegal character`

By default, *NMF-mGPU* follows the system-default locale (e.g., `C` or `UTF-8` on UNIX platforms). In particular, it uses *dots* (`.`) as decimal symbols, and reports a comma (`,`) as an invalid character. Changing the system's locale may *not* work. 

<!-- ==================== -->

#### 7.2 A large input matrix could not be processed

There are different factors that may limit the maximum dimensions for an input matrix. Please see a list in section "[*Limits on Matrix Dimensions*](#matrixlimits)".


*****************************


## 8. <a id="citation">How to Cite *NMF-mGPU*</a>

If you use this software, please cite the following work:

   > E. Mejía-Roa, D. Tabas-Madrid, J. Setoain, C. García, F. Tirado and A. Pascual-Montano. **NMF-mGPU: Non-negative matrix factorization on multi-GPU systems**. *BMC Bioinformatics* 2015, **16**:43. doi:10.1186/s12859-015-0485-4 \[<http://www.biomedcentral.com/1471-2105/16/43>]


<!-- ==================================================== -->
 <br/>
 <br/>
 </body>
 </html>

<!--
// kate: backspace-indents off; indent-mode normal; indent-width 3; keep-extra-spaces off; newline-at-eof on; replace-trailing-space-save off; replace-tabs on; replace-tabs-save on; remove-trailing-spaces none; tab-indents on; tab-width 4;
-->
