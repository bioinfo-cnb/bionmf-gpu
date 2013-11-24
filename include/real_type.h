/************************************************************************
 * Copyright (C) 2011-2013:
 *
 *	Edgardo Mejia-Roa(*), Carlos Garcia, Jose Ignacio Gomez,
 *	Manuel Prieto, Francisco Tirado and Alberto Pascual-Montano(**).
 *
 *	(*)  ArTeCS Group, Complutense University of Madrid (UCM), Spain.
 *	(**) Functional Bioinformatics Group, Biocomputing Unit,
 *		National Center for Biotechnology-CSIC, Madrid, Spain.
 *
 *	E-mail for E. Mejia-Roa: <edgardomejia@fis.ucm.es>
 *	E-mail for A. Pascual-Montano: <pascual@cnb.csic.es>
 *
 *
 * This file is part of bioNMF-mGPU..
 *
 * BioNMF-mGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * BioNMF-mGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with BioNMF-mGPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 ***********************************************************************/
/**********************************************************
 * real_type.h
 *	Definition of the 'real' data type.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Data type:
 *		NMFGPU_SINGLE_PREC: Makes use of single-precision data (i.e., 'float').
 *
 * WARNING:
 *	- Requires support for ISO-C99 standard. It can be enabled with 'gcc -std=c99'.
 *	- Some (optional) useful features for processing single-precision data are only available when '_GNU_SOURCE' is defined.
 *	  Such features can be enabled with 'gcc -D_GNU_SOURCE'
 *
 *********************************************************/

#if ! NMFGPU_REAL_TYPE_H
#define NMFGPU_REAL_TYPE_H (1)

#include <float.h>	/* DBL_EPSILON, FLT_MAX, etc */

///////////////////////////////////////////////////////

/* Data-type definitions */

// 'Real' data type.
#if NMFGPU_SINGLE_PREC

	typedef float real;	// Single precision.

	/* WARNING:
	 * Some (optional) useful features for processing single-precision data are only available when '_GNU_SOURCE' is defined.
	 * They can be done with 'gcc -D_GNU_SOURCE'.
	 */
	#if ( ! _GNU_SOURCE )
		#warning "'_GNU_SOURCE' NOT defined. Some (optional) useful features for processing single-precision data might not be \
			available. This feature macro can be enabled with '-D_GNU_SOURCE' option."
	#endif

#else	/* Double precision */

	typedef double real;

#endif

// ---------------------------------------------

/* Constants */

// Machine epsilon (a.k.a. unit roundoff or EPS)
#ifndef R_EPSILON
	#if NMFGPU_SINGLE_PREC	/* Single precision */
		#define R_EPSILON ( FLT_EPSILON )
	#else				/* Double precision */
		#define R_EPSILON ( DBL_EPSILON )
	#endif
#endif

// Maximum FINITE value.
#ifndef R_MAX
	#if NMFGPU_SINGLE_PREC	/* Single precision */
		#define R_MAX ( FLT_MAX )
	#else				/* Double precision */
		#define R_MAX ( DBL_MAX )
	#endif
#endif

// Minimum normalized positive value.
#ifndef R_MIN
	#if NMFGPU_SINGLE_PREC	/* Single precision */
		#define R_MIN ( FLT_MIN )
	#else				/* Double precision */
		#define R_MIN ( DBL_MIN )
	#endif
#endif

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

/* C HOST Functions */

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

// Conversion format from string used in the scanf(3) family of functions.
#ifndef SCNgREAL
	#if NMFGPU_SINGLE_PREC
		#define SCNgREAL "g"
	#else
		#define SCNgREAL "lg"
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

/* CUDA HOST Functions */

// CURAND: Generate uniformly-distributed random values.
#ifndef CURAND_GENERATE_UNIFORM_REAL
	#if NMFGPU_SINGLE_PREC
		#define CURAND_GENERATE_UNIFORM_REAL curandGenerateUniform
	#else
		#define CURAND_GENERATE_UNIFORM_REAL curandGenerateUniformDouble
	#endif
#endif

// CUBLAS: General matrix-matrix product.
#ifndef CUBLAS_R_GEMM
	#if NMFGPU_SINGLE_PREC
		#define CUBLAS_R_GEMM cublasSgemm
	#else
		#define CUBLAS_R_GEMM cublasDgemm
	#endif

#endif

// CUBLAS: Dot product.
#ifndef CUBLAS_R_DOT
	#if NMFGPU_SINGLE_PREC
		#define CUBLAS_R_DOT cublasSdot
	#else
		#define CUBLAS_R_DOT cublasDdot
	#endif
#endif

// CUBLAS: Sum of absolute values.
#ifndef CUBLAS_R_ASUM
	#if NMFGPU_SINGLE_PREC
		#define CUBLAS_R_ASUM cublasSasum
	#else
		#define CUBLAS_R_ASUM cublasDasum
	#endif
#endif

// cudaSharedMemConfig values used in cudaDeviceSetSharedMemConfig() function.
#ifndef SHARED_MEM_BANK_SIZE
	#if NMFGPU_SINGLE_PREC
		#define SHARED_MEM_BANK_SIZE ( cudaSharedMemBankSizeFourByte )
	#else
		#define SHARED_MEM_BANK_SIZE ( cudaSharedMemBankSizeEightByte )
	#endif
#endif

// ---------------------------------------------

/* CUDA DEVICE Functions */

#if __CUDA_ARCH__

	// Maximum of two floating-point values.
	#ifndef FMAX
		#if NMFGPU_SINGLE_PREC	/* Single Precision */
			#define FMAX fmaxf
		#else			/* Double precision */
			#define FMAX fmax
		#endif
	#endif

	// Floating-point division.
	#ifndef FDIV
		/* This operation can be replaced by a faster less-precise function according to the specified compiler flags.
		 * In this case, division is affected by flags '-use_fast_math' and '-prec-div'.
		 */
		#if NMFGPU_SINGLE_PREC	/* Single Precision */
			#define FDIV fdividef
		#else
			#define FDIV(x,y) ( (x) / (y) )
		#endif
	#endif

	// Declaration of a qualified restricted pointer to the shared-memory variable with an optional offset.
	#ifndef DECLARE_POINTER_SM_VAR
		#if (__CUDA_ARCH__ < 200) && ( ! NMFGPU_SINGLE_PREC )	/* Double-precision data on Compute Capability 1.x */
			#define DECLARE_POINTER_SM_VAR(ptr,qualf1,qualf2,sm_var,offset,size)		\
				qualf1 int *__restrict__ qualf2 ptr##_lo = (int *) &sm_var[ (offset) ];	\
				qualf1 int *__restrict__ qualf2 ptr##_hi = (int *) &ptr##_lo[ (size) ]

		#else	/* Single-precision data <OR> Compute Capability > 1.x */
			#define DECLARE_POINTER_SM_VAR(ptr,qualf1,qualf2,sm_var,offset,size) \
				qualf1 real *__restrict__ qualf2 ptr = (real *) &sm_var[ (offset) ]
		#endif
	#endif

	// Declaration of a qualified pointer to a shared-memory pointer with an optional offset.
	#ifndef DECLARE_SM_POINTER
		#if (__CUDA_ARCH__ < 200) && ( ! NMFGPU_SINGLE_PREC )	/* Double-precision data on Compute Capability 1.x */
			#define DECLARE_SM_POINTER(ptr,qualf1,qualf2,sm_ptr,offset)		\
					qualf1 int *qualf2 ptr##_lo = &sm_ptr##_lo[ (offset) ];	\
					qualf1 int *qualf2 ptr##_hi = &sm_ptr##_hi[ (offset) ]

		#else	/* Single-precision data <OR> Compute Capability > 1.x */
			#define DECLARE_SM_POINTER(ptr,qualf1,qualf2,sm_ptr,offset) \
					qualf1 real *qualf2 ptr = &sm_ptr[ (offset) ]
		#endif
	#endif

	// Loads from shared memory for double-precision data on Compute Capability 1.x
	#ifndef LOAD_FROM_SM
		#if (__CUDA_ARCH__ < 200) && ( ! NMFGPU_SINGLE_PREC )	/* Double-precision data on Compute Capability 1.x */
			#define LOAD_FROM_SM(sm,idx) ( __hiloint2double( sm##_hi[ (idx) ], sm##_lo[ (idx) ] ) )

		#else	/* Single-precision data <OR> Compute Capability > 1.x */
			#define LOAD_FROM_SM(sm,idx) ( sm[ (idx) ] )
		#endif
	#endif

	// Stores in shared memory for double-precision data on Compute Capability 1.x
	#ifndef STORE_IN_SM
		#if (__CUDA_ARCH__ < 200) && ( ! NMFGPU_SINGLE_PREC )	/* Double-precision data on Compute Capability 1.x */
			#define STORE_IN_SM(sm,idx,val) { \
					sm##_lo[ (idx) ] = __double2loint( (val) );	\
					sm##_hi[ (idx) ] = __double2hiint( (val) );	\
			}

		#else	/* Single-precision data <OR> Compute Capability > 1.x */
			#define STORE_IN_SM(sm,idx,val) { sm[ (idx) ] = (val); }
		#endif
	#endif

#endif /* __CUDA_ARCH__ */

///////////////////////////////////////////////////////

#endif /* NMFGPU_REAL_TYPE_H */
