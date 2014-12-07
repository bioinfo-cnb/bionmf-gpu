<!--
 ************************************************************************
 *
 * NMF-mGPU -- Non-negative Matrix Factorization on multi-GPU systems.
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
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
 <html lang="en" xml:lang="en" xmlns="http://www.w3.org/1999/xhtml">
 <head>
   <meta name="application-name" content="NMF-mGPU"/>
   <meta name="author" content="Edgardo Mejia-Roa (edgardomejia@fis.ucm.es), Carlos Garcia, Jose Ignacio Gomez, Manuel Prieto, Francisco Tirado, and Alberto Pascual-Montano (pascual@cnb.csic.es)."/>
   <meta name="description" content="Non-negative Matrix Factorization (NMF) for Biology on multi-GPU systems. User guide"/>
   <meta name="keywords" content="bioNMF, NMF, Matrix factorization, GPU, multi-GPU, GPGPU, NVIDIA, CUDA, CUBLAS, Bioinformatics"/>
   <meta name="language" content="en"/>
   <meta name="copyright" content="(C) 2011-2014 Edgardo Mejia-Roa (edgardomejia@fis.ucm.es). ArTeCS Group, Complutense University of Madrid (UCM), Spain."/>
   <meta http-equiv="content-Type" content="text/html; charset=UTF-8"/>
   <meta http-equiv="last-modified" content="2014/04/30" scheme="YYYY/MM/DD"/>
   <link rel="stylesheet" type="text/css" href="styles.css"/>
   <title>NMF-mGPU user guide</title>
 </head>
 <body>

<!-- ==================================================== -->

# *NMF-mGPU* USER GUIDE

 This documents shows how to use *NMF-mGPU*.

 **Index:**

   1. Introduction.
   2. *NMF-mGPU* arguments.
   3. Data-file format.
   4. Analysis process.
   5. Examples of use.
   6. Utility programs
   7. Issues/Troubleshooting.
   8. How to cite bioNMF.

*****************************

## 1. Introduction

 ***NMF-mGPU*** implements the ***Non-negative Matrix Factorization*** (***NMF***) by making use of a ***Graphics-Processing Unit*** (***GPU***). NMF is able to represent a data set as the linear combination of a collection of elements named *factors*. The *factorization rank* (i.e, the number of *factors*) is usually specified as a low value, so NMF results in a considerably reduced representation of the input data.
 
 Mathematically, this algorithm can be described as the *decomposition* of an input matrix, **V**, into two other matrices, **W** and **H**, whose product is approximately the former (i.e., **V** ~ **W** \* **H**). If matrix **V** has *n* rows and *m* columns, then dimensions for output matrices, **W** and **H**, will be *n* × *k* and *k* × *m* (respectively), being *"k"* the specified *factorization rank*. In this context, **W** contains the (reduced) set of *k* factors, and **H** stores the coefficients of the linear combination of such factors that rebuilds **V**. Usually, the factorization rank (*k*) provided by the user, is a value much less than *n* and *m*.
 
 In contrast to other similar factorization algorithms (e.g., *PCA*, *SVD*, or *ICA*), NMF is constrained to use *non-negative* values and additive-only combinations on all matrices. This results in a *parts-based* representation of data, where each factor can be *contextually interpreted*.
 
 Driven by the ever-growing demands of the game industry, where numerous algebraic operations are required to draw images on the screen, GPUs have evolved from simple graphics-drawing devices into highly parallel and programmable systems that largely outperforms any conventional processor. This GPU implementation of the NMF algorithm has been developed using the NVIDIA's ***Compute Unified Device Architecture*** (***CUDA***) programming model. *CUDA* represents the GPU as a programmable *coprocessor*, which will be then responsible of computing all required algebraic operations.

 *NMF-mGPU* is able to process matrices of any size. Even on detached devices with a dedicated (and limited) on-board memory, the NMF algorithm can be executed on large datasets by blockwise transferring and processing the input matrix, as necessary.

 <!--Finally, this software can make use of multiple GPU devices through ***MPI*** (***Message-Passing Interface***).-->

 In addition to *NMF-mGPU*, there are some utility programs to make easier working with input files. It includes a program for binary-text file conversion, and another to generate input matrices with random data (useful for testing *NMF-mGPU*). See section 6 *"Utility programs"* for details.


*****************************


## 2. *NMF-mGPU* arguments

Program usage:

		bin/NMF_GPU <filename> [ -b <native> ] [ -cr ] [ -k <factorization_rank> ] [ -i <nIters> ] [ -j <niter_test_conv> ]
						[ -t <stop_threshold> ] [ -e <native> ] [ -z <GPU_device> ]
or, for a help message:

		bin/NMF_GPU -h


### 2.1. Arguments to control the *input* data matrix

   * *`<filename>`*: Input data matrix (mandatory if help is not requested).

   * `-B`,`-b` *`<native>`*: Binary input file in ***"native"*** ('`-b` `1`') or ***non-"native"*** format ('`-b` `0`').  
	 In **non-*native*** format, the file is read assuming it was written using double-precision data, and unsigned integers for matrix dimensions, regardless how the program was compiled.  
	 Otherwise, in ***native*** format, the file is read using the data types specified at compilation (e.g., '`float`' and '[`signed`] `int`').
	 Please note that *native* mode *skips* most error checking and information messages. See section *3 "Data-file format"* for details.  
	 The default (if '`-b`' is not specified) is to read input data from an *ASCII-text* file.  

   * `-C`,`-c`: Input text file has *numeric* column headers (*disabled* by default, ignored for *binary* files).

   * `-R`,`-r`: Input text file has *numeric* row labels (*disabled* by default, ignored for *binary* files).

   * `-E`,`-e` *`<native>`*: Writes output files as ***"native"*** ('`-e` `1`') or ***non-"native"*** binary format ('`-e` `0`').  
	 In **non-*native*** format, the file is written using double-precision data, and unsigned integers for matrix dimensions, regardless how the program was compiled.  
	 Otherwise, in ***native*** or raw format, the file is written using the data types specified at compilation (e.g., '`float`' and '[`signed`] `int`').  
	 Please note that *native* mode *skips* error checking, information messages, and data transformation (e.g., matrix transposing). See section *3 "Data-file format"* for details.  
	 The default (if '`-e`' is not specified) is to write output data to an *ASCII-text* file.

<!-- ==================== -->

### 2.2. Options to control the *NMF* algorithm

   * `-K`,`-k` *`<factorization_rank>`*: Factorization Rank (default value: `2` factors).

   * `-I`,`-i` *`<nIters>`*: Maximum number of iterations if the algorithm does not converge to a stable solution (`2000` iterations by default).

   * `-J`,`-j` *`<niter_test_conv>`*: Perform a convergence test each *`<niter_test_conv>`* iterations (default: each `10` iterations).  
     If this value is greater than *`<nIters>`* (see '`-i`' option), *no* test is ever performed.

   * `-T`,`-t` *`<stop_threshold>`*: Stopping threshold (default value: `40`).  
	 When matrix **H** has not changed on the last *`<stop_threshold>`* times that the convergence test has been performed, it is considered that the algorithm *has converged* to a solution and stops it. 

Please see section *4 "Analysis process"* for a detailed description of the implemented NMF algorithm, as well as the method used to test for convergence.

<!-- ==================== -->

### 2.3. Other options

   * `-Z`,`-z` *`<GPU_device>`*: GPU device ID to attach on (default: `0`).
	 <!-- On multi-GPU version, devices will be selected from this value.-->

   * `-h`,`-H`: Prints a help message with all arguments.


*****************************


## 3. Data-file formats

*NMF-mGPU* is able to work with matrices stored in a binary or text file. This section describes both file formats, which can be used for input or output data.

Regardless of the format, the file must contain *non-negative* data only. Optionally, there can be column headers, and/or row labels, as well as a short description string (denoted as *"matrix name"*). Any of such three ***tag elements*** is independent of each other, and may or may not appear in the file.

***Note:*** it is *not* allowed to have any column or row with *all* entries set to zero, there must be at least one *positive* value on each. *NMF-mGPU* will try to detect and report it as invalid file format.

Some examples of valid input files, in both formats, can be found in the `test/` folder. In addition, some utility programs (such as a text-binary file converter) are provided in order to make easier working with input and output files. Please, see section *6 "Utility programs"* for details.
<!-- ALERT: TODO -->

<!-- ==================== -->

### 3.1. ASCII-text files

In this format, matrix data are separated by (single) *tab* characters. Matrix dimensions are then inferred from the text layout (i.e., the number of data per line and the number of lines) in the file. The optionals column headers must be placed on the first row, just after the description string, if set. The (also optionals) row labels must be placed at the beginning of each line. Finally, each header, label, or the description string, can be composed by several *space-separated* words.

A text layout scheme, for an *n*-by-*m* data matrix, would be the following:

		[ [Description string  <tab>]  column header 1	<tab>  ...  <tab>  column header m	<newline> ]
		[ row label 1		   <tab>]  value_11			<tab>  ...  <tab>  value_1m			<newline>
		...
		[ row label n		   <tab>]  value_n1			<tab>  ...  <tab>  value_nm			<newline>

Nevertheless, this is a *simplified* scheme, since actually, every item is optional. For instance, two consecutive tab characters (or, a line beginning or ending with a tab) denote an *empty entry*, which is then processed, either as '`0.0`' or as an *empty string*, as necessary. Note that a such string is still allocated in memory (it stores a null byte), and will be set as a tab character in any output file. To completely *skip* any or all of the three tag elements (i.e., no memory is allocated), just directly set the next element, without the delimiter tab. For instance, for a numeric-only data file (i.e., with no tags at all), just start the file (and every line) with the corresponding matrix datum. An example of such file can be found in the `test/` folder.

***Important notes:***

   * Please do *not* use double- (`"`) or single-quote (`'`) characters in any tag element.

   * By default, *NMF-mGPU* follows the system-default locale (e.g., '`C`' on UNIX platforms). In particular, it uses *dots* ('`.`') as decimal symbols, and reports a comma ('`,`') as an invalid character.

   * Only UNIX (`LF`, '`\n`') and MS-DOS/Windows (`CR+LF`, '`\r\n`'), end-of-line styles are accepted.

#### Numeric row labels and/or column headers:
By default, *NMF-mGPU* processes numeric row labels as matrix data. To prevent this, please add '`-r`' to the command-line arguments. This option forces *NMF-mGPU* to process the first data column in the file, as row labels.  
Similarly, to prevent a misinterpretation of numeric column headers and/or the description string, please use the '`-c`' option.

Please note that such options are *not* required for *binary files*, since there is no way to accidentally mix matrix data and tag elements.

A list of all available arguments can be found on section *2 "NMF-mGPU arguments"*.

#### Whitespace characters rather than tabs:
Under the following conditions, input data may be separated by *single* whitespace characters ('` `'):

   * *All* row labels and column headers must contain (up to) a single word, each. Similarly for the description string.
   * There must *not* be any *tab* character in the file.
   * *Consecutive space characters* will be processed as '`0.0`' or an empty string, as necessary.
   * Both, first and second rows must *not* begin with an *empty* column header and/or row label (i.e., they must not start with a whitespace character), since they will be misinterpreted as null data (i.e., '`0.0`').  
	 However, you can force a header/label detection with the '`-r`' and/or '`-c`' options (see section *2 "NMF-mGPU arguments"* for details).

<!-- ==================== -->

### 3.2. Binary files

In addition to ASCII-text files, *NMF-mGPU* also accepts a data matrix stored in a binary format. Such binary files, can be read and/or written in two sub-formats: *"native"* and *non-"native"*.

   * ***"Native"* mode** refers to *raw* I/O. That is, data are stored/loaded with the precision specified at compilation time: '`float`' if the `Makefile` parameter "`SINGLE`" was set to '`1`', or '`double`' otherwise. Matrix dimensions are stored/loaded in a similar way (i.e., '`unsigned int`', if "`UNSIGNED`" was set to '`1`'; '[`signed`] `int`', otherwise). This mode is faster because *no* error checking is performed. Therefore, it should *not* be used to read untrusted input files.  
     **Important note:** In *output* files, this mode also *skips* all data transformation steps (e.g., *matrix transposing*). In particular, matrix **H** (which is computed in transposed mode due to performance reasons) will *not* be "restored" before writing the output file.  
     Finally, please see section *4 "Compiling NMF-mGPU"*, in the *Installation guide*, for a list of available `Makefile` parameters.

   * In **non-*"native"* mode**, data are *always* stored/loaded using *double* precision (and *unsigned* integers for matrix dimensions), regardless the options specified at compilation. This is the recommended mode for input or final output data, since every datum is checked for invalid format.

In both modes, all *binary* data will be read/written in the following order:

   * **Number of rows** (*n*)**:** unsigned/signed integer.

   * **Number of columns** (*m*)**:** unsigned/signed integer.

   * **Matrix data:** *n*-by-*m* (double/single)-precision floating-point values stored in *row-major* order (i.e., contiguous elements belong to the same matrix *row*).


The rest of the file contains the optional column headers, row labels and description string. If set, they must be written in *ASCII-text* format, right after the last binary matrix datum, as follows:

   * **Row labels**: *n* tab-separated strings.

   * A newline character ('`\n`' or '`\r\n`').

   * **Column headers**: *m* tab-separated strings.

   * A newline character ('`\n`' or '`\r\n`').

   * ***Matrix name***: the short description string.

Similarly to ASCII-text files, consecutive, initial, or ending tab characters denote an *empty string*. To fully skip any of the three tag elements, directly write the next newline character (or just close the file, to skip all the subsequents). For instance, to set only a *matrix name*, with no labels or headers at all, just write the two newline characters before the description string. To skip all three tag elements, just close the file after the last binary matrix datum.

***Notes***:

   * It is not necessary to specify the options '`-r`' or '`-c`' to indicate numeric labels and/or headers. In fact, such arguments will be *ignored*.

   * Please do *not* use double- (`"`) or single-quote (`'`) characters in any *tag* element.

   * Only UNIX (`LF`, '`\n`') and MS-DOS/Windows (`CR+LF`, '`\r\n`'), end-of-line styles are accepted.

<!-- ==================== -->

### 3.3. Maximum matrix dimensions

<!-- ALERT: TODO:  -->
TBD

*****************************


## 4. Analysis process

This section describes the NMF algorithm implemented on *NMF-mGPU*, as well as all related command-line options that can be used to customize the analysis process.

<!-- ==================== -->

### 4.1 The NMF algorithm:

NMF takes an input matrix (**V**) and returns two matrices, **W** and **H**, whose product is equal to the former (i.e., **V** ~ **W** \* **H**). If **V** has *n* rows and *m* columns, then dimensions for **W** and **H**, will be *n* × *k* and *k* × *m*, respectively. The *factorization rank* (*"k"*) specified by the user, is usually a value much less than *n* and *m*.

NMF iteratively modifies **W** and **H** until their product approximates to **V**. Such modifications, composed by matrix products and other algebraic operations, are derived from minimizing a cost function that quantifies, in some way, the differences between **W** \* **H** and **V**. There are numerous objective functions, each leading to different update rules.

The following pseudo-code shows the NMF algorithm as it was implemented in *NMF-mGPU*:

		NMF_GPU( V, n, m, k ):

			W = random_init( n, k )
			H = random_init( k, m )

			REPEAT:

				accum_W = reduce_to_a_row( W )
				VdWH = V ./ (W*H)
				H_tmp = Wt * VdWH
				H = H .* H_tmp ./ (accum_W)t

				accum_H = reduce_to_a_column( H )
				VdWH = V ./ (W*H)
				W_tmp = VdWH * Ht
				W = W .* W_tmp ./ (accum_H)t

			UNTIL has_converged()

Symbols '`.*`' and '`./`' denote point-wise matrix operations (in contrast to the regular matrix product, '`*`'). In addition, '`Ht`' and '`Wt`' represent transposed matrices. Similarly for the *`k`*-length vectors '`(accum_H)t`' and '`(accum_W)t`'.

Finally, the update rules were taken from the following work:

   * D. D. Lee and H. S. Seung: **Algorithms for non-negative matrix factorization**. In *Adv. in Neur. Inf. Process. Syst.* (*NIPS*), 2001, vol. **13**, 556-562, MIT Press.  
	 <http://hebb.mit.edu/people/seung/papers/nmfconverge.pdf>


<!-- ==================== -->

### 4.2 Convergence test

bioNMF uses the ***Consensus Method*** as test of convergence. It is based on changes on a *connectivity matrix* (**C**) of size *m* × *m*, where *"m"* is the number of columns of both **V** and **H**. Each entry *Cij* is set to '`1`' if column *i* and *j* in **H** have their maximum value for the same row, and '`0`' otherwise. 

 <!-- ALERT: TODO: Explicar. --> (i.e., on the same factor), 

This matrix is then compared with the one computed on the previous iteration and changes are counted.

bioNMF considers algorithm has converged when test of convergence (connectivity matrix or distance) does not change its value after a certain number of times it is performed. This number is controlled by the '`stop_threshold`' input parameter (set with '`-t <stop_threshold>`' option). The test is performed each `<niter_test_conv>` iterations (set with '-j <niter_test_conv>' option). 

Before this test, matrices W and H are adjusted to avoid underflow---small entries are replaced by the epsilon value EPS=2^(-52).

Example of use:

   * `bin/bionmf [...] -m SampleClassification -t 40 -j 10 -i 2000`
	 Test of convergence (based on differences on connectivity matrix) is performed each 10 iterations. If connectivity matrix does not change, an internal counter is incremented. When this counter reaches 40, it is considered that the NMF algorithm has converged. However, a limit of 2000 iterations is set just in case NMF never converges.

   * `bin/bionmf [...] -d -t 40 -j 10 -i 2000 -u 1e-3`
	 The test of convergence (distance between W\*H and input-matrix) is performed each 10 iterations. If its value is less than 1e-3, an internal counter is incremented. When this counter reaches 40, it is considered that the NMF algorithm has converged. However, a limit of 2000 iterations is set just in case NMF never converges.

   * `bin/bionmf [...] -m SampleClassification -t 40 -j 10 -i 2000`
	 Test of convergence (differences on connectivity matrix) is performed each 10 iterations. If connectivity matrix does not change, an internal counter is incremented. When this counter reaches 40, it is considered that the NMF algorithm has converged. However, a limit of 2000 iterations is set just in case NMF never converges.


For more details about the Consensus method and CCC values, please read the following papers:

   * E. Mejía-Roa, P. Carmona-Saez, R. Nogales, C. Vicente, M. Vázquez, X. Y. Yang, C. García, F. Tirado and A. Pascual-Montano. ***bioNMF:* a web-based tool for non-negative matrix factorization in biology**. In *Nucl. Acids Res.* 2008, **36** (suppl 2), W523-W528. [doi:10.1093/nar/gkn335](http://dx.doi.org/10.1093/nar/gkn335)  
	 [http://nar.oxfordjournals.org/cgi/content/full/36/suppl_2/W523](http://nar.oxfordjournals.org/cgi/content/full/36/suppl_2/W523 "Free-access full text")  
	 [http://bionmf.dacya.ucm.es](http://bionmf.dacya.ucm.es "bioNMF web site")

   * Brunet JP, Tamayo P, Golub TR, Mesirov JP.: **Metagenes and molecular pattern discovery using matrix factorization**. In *Proc. Natl Acad. Sci. USA* (2004) **101**:4164–4169.

*****************************

## 9. Output files.

This section describes all output files generated by each processing step.




*****************************

## 9. Running tests and examples of use.


This section shows the compilation process for *NMF-mGPU*, with the provided `Makefile` in a command-line environment. 



First, it describes how to set some optional environment variables (although they may be mandatory for execution). Then, it lists all supported options to customize the final executable code.

### 4.1. Environment setup.


*NMF-mGPU* requires the path to your CUDA Toolkit in order to be compiled. It may be necessary also for execution (e.g., dynamically-linked libraries)

You can specify it through any of the following ways:





*****************************

## 10. Utilities

In addition to *NMF-mGPU*, there are some utility programs to make easier working with input files. It includes a program for binary-text file conversion, and another to generate input matrices with random data (useful for testing *NMF-mGPU*).

***Note:*** These programs do *not* make use of the GPU device. They are implemented in pure-`C99` language, and all operations are performed on the *host* (i.e., the CPU). Therefore, they do *not* require any CUDA-related option, configuration or software.

-----------------------------

### 5.1. Binary-text file converter

Program to perform binary-text file conversions according to the arguments. For binary files, there are two sub-formats: *"native"* and *non-"native"*.

Usage:
	file_converter <filename> <file_is_binary> [<has_numeric_column-headers> [<hasn_umeric_row-labels> [<save_as_native-binary>]]]

See a description of both file formats in section *3 "Data-file format"*.

-----------------------------

### 5.2. Matrix generator

 This program generates a data matrix with non-negative random values. You can select the output matrix dimensions, as well as the maximum value (i.e, all data will be between '`0.0`' and the specified value, included). In addition, the output matrix can be written as an ASCII text, or in a binary file (in any of the binary modes described above).

 Usage:

	<directory_to_save_files> <nrows> <ncols> [ <set_labels> [<kStart> [<kEnd> [<nRuns>]]] ]


 ***Notes:***

   * Output file will not contain any tag (i.e., row labels, column headers or description string), just numeric data (see section *3 "Data-file formats"* in the *User guide*.)
   * The entire operation is performed on the *host* (i.e., on the CPU), *not* on the GPU device.

*****************************

## 11. Issues/Troubleshooting.

A) Input matrix not recognized ('`Data in column` *`<X>`* `is not numeric`', or '`illegal character`')

B) Large input matrix could not be processed

### Input matrix no recognized ('`Data in column` *`<x>`* `is not numeric. Invalid numeric format`', or '`illegal character`')

By default, this programs follows the 'C' locale, i.e.,it uses dots ('.') as decimal symbol. Therefore, bioNMF will report an invalid-file-format error if another character is used (unless another locale is  set with setlocale(3) ).

### Large input matrix could not be processed

<!-- ALERT: TODO: -->
TBD (see Section *3.3. "Maximum matrix dimensions"*)

*****************************


## 12. How to cite *NMF-mGPU*.

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
