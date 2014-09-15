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
 <html lang="en" xml:lang="en" xmlns="http://www.w3.org/1999/xhtml">
 <head>
   <meta name="application-name" content="BioNMF-GPU"/>
   <meta name="author" content="Edgardo Mejia-Roa (edgardomejia@fis.ucm.es), Carlos Garcia, Jose Ignacio Gomez, Manuel Prieto, Francisco Tirado, and Alberto Pascual-Montano (pascual@cnb.csic.es)."/>
   <meta name="description" content="Non-negative Matrix Factorization (NMF) on (multi-)GPU systems, for Biology."/>
   <meta name="keywords" content="bioNMF, NMF, Matrix factorization, GPU, multi-GPU, GPGPU, NVIDIA, CUDA, CUBLAS, Bioinformatics"/>
   <meta name="language" content="en"/>
   <meta name="copyright" content="(C) 2011-2014 Edgardo Mejia-Roa (edgardomejia@fis.ucm.es). ArTeCS Group, Complutense University of Madrid (UCM), Spain."/>
   <meta http-equiv="content-Type" content="text/html; charset=UTF-8"/>
   <meta http-equiv="last-modified" content="2014/04/30" scheme="YYYY/MM/DD"/>
   <link rel="stylesheet" type="text/css" href="styles.css"/>
   <title>BioNMF-GPU: Non-negative Matrix Factorization on (multi-)GPU systems</title>
 </head>
 <body>

<!-- ==================================================== -->

## *BioNMF-GPU*: Non-negative Matrix Factorization on (multi-)GPU systems, for Biology

*****************************

***BioNMF-GPU*** is an efficient implementation of the ***Non-negative Matrix Factorization*** (***NMF***) algorithm, based on a programmable ***Graphics-Processing Unit*** (***GPU***). Since numerous linear-algebra operations are required to draw images on the screen, GPUs are devices specially designed to perform such tasks *much faster* than any conventional processor.

This implementation is based on the NVIDIA's programming model: ***CUDA*** (***Compute Unified Device Architecture***). *CUDA* represents the GPU as a programmable *co-processor*, which will be then responsible of computing all required algebraic operations. GPU devices represent a *cost-effective* alternative to conventional multi-processor clusters, since they are already present on almost any modern PC or laptop, as a *graphics card*.

*BioNMF-GPU* is able to process matrices of any size. Even on detached devices with a dedicated (and limited) on-board memory, the NMF algorithm can be executed on large datasets by blockwise transferring and processing the input matrix, as necessary.

<!--Finally, this software can make use of multiple GPU devices through ***MPI*** (***Message-Passing Interface***).-->


*****************************

## Quick-start guide

<!-- ALERT: TODO: Quick start guide  -->
### Install (Linux):
These installation guidelines have been tested on Ubuntu 14.04. If you want to install *bioNMF-GPU* on a different distribution, some of the steps may vary.

First of all, check if your computer has a CUDA-enabled GPU installed on it. You can find more information [here](https://developer.nvidia.com/cuda-gpus). *bioNMF-GPU* works only on computers with CUDA-enabled GPUs.

####Prerequisites:
+ NVIDIA proprietary driver: Open the program *Software & Updates*, then go to *Additional Drivers* section, and check the "*Using NVIDIA binary driver*" option. You can also do this by going to the terminal and typing: `sudo apt-get install nvidia-current`. You may have to reboot the system in order to use this driver after installing it.
+ Additional packages:

<!-- ==================================================== -->
 </body>
 </html>

<!--
// kate: backspace-indents off; folding-markers on; indent-mode normal; indent-width 3; keep-extra-spaces off; newline-at-eof on; remove-trailing-space off; replace-trailing-space-save off; replace-tabs off; replace-tabs-save off; remove-trailing-spaces none; tab-indents on; tab-width 4;
-->
