/************************************************************************
 *
 * NMF-mGPU - Non-negative Matrix Factorization on multi-GPU systems.
 *
 * Copyright (C) 2011-2014:
 *
 *	Edgardo Mejia-Roa(*), Carlos Garcia(*), Jose Ignacio Gomez(*),
 *	Manuel Prieto(*), Francisco Tirado(*) and Alberto Pascual-Montano(**).
 *
 *	(*)  ArTeCS Group, Complutense University of Madrid (UCM), Spain.
 *	(**) Functional Bioinformatics Group, Biocomputing Unit,
 *		National Center for Biotechnology-CSIC, Madrid, Spain.
 *
 *	E-mail for E. Mejia-Roa: <edgardomejia@fis.ucm.es>
 *	E-mail for A. Pascual-Montano: <pascual@cnb.csic.es>
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
 ***********************************************************************/
/**********************************************************
 * real_type.h
 *	Definition of the 'real' data type.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Data types, functions and constants:
 *		NMFGPU_SINGLE_PREC: Makes use of single-precision data (i.e., 'float').
 *
 *		NMFGPU_CUDA_HOST: Defines some constants and functions related to CUDA Runtime, cuBLAS and cuRAND.
 *
 *		NMFGPU_MPI: Defines some MPI-related constants.
 *
 *		__NVCC__, __CUDACC__, __CUDA_ARCH__: CUDA kernel code.
 *
 *********************************************************/

#if ! NMFGPU_REAL_TYPE_H
#define NMFGPU_REAL_TYPE_H (1)

#if NMFGPU_CUDA_HOST
	#include <cublas_v2.h>
	#include <curand.h>
#endif


#if NMFGPU_CUDA_HOST || defined(__NVCC__) || defined(__CUDACC__) || defined(__CUDA_ARCH__)
	#include <cuda_runtime_api.h>
#endif


#if NMFGPU_MPI
	#include <mpi.h> /* MPI_FLOAT, MPI_DOUBLE */
#endif


#include <math.h>	/* sqrt(f) */
#include <string.h>	/* strtof, strtod, */
#include <float.h>	/* DBL_EPSILON, FLT_MAX, etc */

////////////////////////////////////////////////
////////////////////////////////////////////////

/* Data-type definitions */

// 'Real' data type.
#if NMFGPU_SINGLE_PREC	/* Single precision */
	typedef float real;
#else	/* Double precision */
	typedef double real;
#endif

// Real-type data on MPI.
#if NMFGPU_MPI && (! defined(NMFGPU_MPI_REAL_T))
	#if NMFGPU_SINGLE_PREC		/* Single precision */
		#define NMFGPU_MPI_REAL_T MPI_FLOAT
	#else				/* Double precision */
		#define NMFGPU_MPI_REAL_T MPI_DOUBLE
	#endif
#endif

// ---------------------------------------------
// ---------------------------------------------

/* Constants */

// Machine epsilon (a.k.a. unit roundoff or EPS)
#ifndef R_EPSILON
	#if NMFGPU_SINGLE_PREC		/* Single precision */
		#define R_EPSILON ( FLT_EPSILON )
	#else				/* Double precision */
		#define R_EPSILON ( DBL_EPSILON )
	#endif
#endif

// Maximum FINITE value.
#ifndef R_MAX
	#if NMFGPU_SINGLE_PREC		/* Single precision */
		#define R_MAX ( FLT_MAX )
	#else				/* Double precision */
		#define R_MAX ( DBL_MAX )
	#endif
#endif

// Minimum normalized positive value.
#ifndef R_MIN
	#if NMFGPU_SINGLE_PREC		/* Single precision */
		#define R_MIN ( FLT_MIN )
	#else				/* Double precision */
		#define R_MIN ( DBL_MIN )
	#endif
#endif

/* Conversion format from string used in the scanf(3) family of functions.
 * WARNING: Please, do NOT set to longer than 6 characters.
 */
#ifndef SCNgREAL
	#if NMFGPU_SINGLE_PREC
		#define SCNgREAL "g"
	#else
		#define SCNgREAL "lg"
	#endif
#endif

// ---------------------------------------------
// ---------------------------------------------

/* Macro Functions */

// Converts constant values to the right data type (i.e., appends an 'f' on single-precision mode).
#ifndef REAL_C
	#if NMFGPU_SINGLE_PREC	/* Single precision */
		#define REAL_C(c) (c ## f)
	#else			/* Double precision */
		#define REAL_C
	#endif
#endif

/* Converts constants to the right data type (i.e., appends an 'f' on single-precision mode).
 * Useful when the argument is not a value but a macro constant that must also be expanded.
 */
#ifndef REAL_MC
	#define REAL_MC(m) ( REAL_C( m ) )
#endif

// ---------------------------------------------
// ---------------------------------------------

/* ISO-C Functions */

// Function to convert from string to real data type.
#ifndef STRTOREAL
	#if NMFGPU_SINGLE_PREC
		#if (_XOPEN_SOURCE >= 600) || defined(_ISOC99_SOURCE) || (_POSIX_C_SOURCE >= 200112L)
			#define STRTOREAL strtof
		#else
			#define STRTOREAL(nptr,endptr) ( (float) strtod((nptr),(endptr)) )
		#endif
	#else
		#define STRTOREAL strtod
	#endif
#endif

// Square root function
#ifndef SQRTR
	#if NMFGPU_SINGLE_PREC
		#if (_XOPEN_SOURCE >= 600) || defined(_ISOC99_SOURCE) || (_POSIX_C_SOURCE >= 200112L) || \
			defined(_BSD_SOURCE) || defined(_SVID_SOURCE)
			#define SQRTR sqrtf
		#else
			#define SQRTR(x) ( (float) sqrt( (double) (x) ) )
		#endif
	#else
		#define SQRTR sqrt
	#endif
#endif

// ---------------------------------------------
// ---------------------------------------------

/* cuBLAS Functions and Constants */

#if NMFGPU_CUDA_HOST

	// General matrix-matrix product.
	#undef CUBLAS_R_GEMM
	#if NMFGPU_SINGLE_PREC		/* Single precision */
		#define CUBLAS_R_GEMM cublasSgemm
	#else				/* Double precision */
		#define CUBLAS_R_GEMM cublasDgemm
	#endif

	// Dot product.
	#undef CUBLAS_R_DOT
	#if NMFGPU_SINGLE_PREC
		#define CUBLAS_R_DOT cublasSdot
	#else
		#define CUBLAS_R_DOT cublasDdot
	#endif

	// Sum of absolute values.
	#undef CUBLAS_R_ASUM
	#if NMFGPU_SINGLE_PREC
		#define CUBLAS_R_ASUM cublasSasum
	#else
		#define CUBLAS_R_ASUM cublasDasum
	#endif

#endif

// ---------------------------------------------
// ---------------------------------------------

/* cuRAND Functions and Constants */

#if NMFGPU_CUDA_HOST

	// Uniformly distributed real data
	#undef CURAND_GENERATE_UNIFORM_REAL
	#if NMFGPU_SINGLE_PREC
		#define CURAND_GENERATE_UNIFORM_REAL curandGenerateUniform
	#else
		#define CURAND_GENERATE_UNIFORM_REAL curandGenerateUniformDouble
	#endif

#endif

// ---------------------------------------------
// ---------------------------------------------

/* CUDA Runtime Constant and Functions */

#if NMFGPU_CUDA_HOST

	// Argument for cudaDeviceSetSharedMemConfig() function.
	#undef CUDA_SHARED_MEM_BANK_SIZE
	#if NMFGPU_SINGLE_PREC
		#define CUDA_SHARED_MEM_BANK_SIZE cudaSharedMemBankSizeFourByte
	#else
		#define CUDA_SHARED_MEM_BANK_SIZE cudaSharedMemBankSizeEightByte
	#endif

#endif

// ---------------------------------------------
// ---------------------------------------------

/* CUDA DEVICE Functions and Constants */

#if defined(__NVCC__) || defined(__CUDACC__) || defined(__CUDA_ARCH__)

	/* Maximum of two floating-point values.
	 * If one of the values is NaN, returns the other.
	 */
	#undef CUDA_DEVICE_FMAX
	#if NMFGPU_SINGLE_PREC
		#define CUDA_DEVICE_FMAX fmaxf
	#else
		#define CUDA_DEVICE_FMAX fmax
	#endif

	/* Division of two floating-point values.
	 * NOTE:
	 *	On single-precision data, a faster (but less-precise) function may be employed,
	 *	if the compilation flags '-use_fast_math' and '-prec-div' are specified.
	 * WARNING:
	 *	No error checking is performed (e.g., if the second value is non-zero).
	 */
	#undef CUDA_DEVICE_FDIV
	#if NMFGPU_SINGLE_PREC
		#define CUDA_DEVICE_FDIV fdividef
	#else
		#define CUDA_DEVICE_FDIV( a, b ) ( a / b )
	#endif

#endif

////////////////////////////////////////////////
////////////////////////////////////////////////

#endif /* NMFGPU_REAL_TYPE_H */
