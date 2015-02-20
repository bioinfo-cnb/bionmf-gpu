/************************************************************************
 *
 * NMF-mGPU - Non-negative Matrix Factorization on multi-GPU systems.
 *
 * Copyright (C) 2011-2015:
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
 * GPU_kernels.cuh
 *	Kernel code to execute on the device (GPU).
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Data type:
 *		NMFGPU_SINGLE_PREC: Makes use of single-precision data (i.e., 'float').
 *
 **********************************************************
 *
 * NOTE: In order to improve performance:
 *
 *	+ All matrices include useless data for padding. It is sometimes denoted as 'pitch'.
 *
 *********************************************************/

#if ! NMFGPU_GPU_KERNELS_CUH
#define NMFGPU_GPU_KERNELS_CUH (1)

#include "index_type.h"
#include "real_type.h"

#include <cuda_runtime_api.h>

#if (! defined(__CUDACC__)) && (! defined(__NVCC__)) && (! defined(__cplusplus))	/* ISO-C header file */
	#include <stdbool.h>
	#include <stddef.h>	/* size_t */
#endif

////////////////////////////////////////////////
////////////////////////////////////////////////

/* Conditional Qualifiers */

// Restrict
#undef NMFGPU_RESTRICT
#if defined(__CUDACC__) || defined(__NVCC__)	/* CUDA header file */
	#define NMFGPU_RESTRICT __restrict__
#else						/* ISO-C header file */
	#define NMFGPU_RESTRICT restrict
#endif

// Host
#undef NMFGPU_HOST
#if defined(__CUDACC__) || defined(__NVCC__)	/* CUDA header file */
	#define NMFGPU_HOST __host__
#else						/* ISO-C header file */
	#define NMFGPU_HOST
#endif

// ---------------------------------------------
// ---------------------------------------------

/* Constants */

// Number of items simultaneously read from global memory by each thread in reduce_to_row()
#if REDUCE_TO_ROW__ITEMS_PER_THREAD <= 0

	#undef REDUCE_TO_ROW__ITEMS_PER_THREAD

	/* Maximum values with a 100% of occupancy:
	 * Compute Capability:
	 *		2.x:	6
	 *	3.0 --	3.5:	13
	 *
	 * ATTENTION:
	 *	+ This constant is also required in HOST code. Therefore, please
	 *	  do NOT use "__CUDA_ARCH__" to select different values, since
	 *	  it will not be defined there.
	 *
	 *	+ Change the default value ONLY if you are compiling this program
	 *	  for a particular SM architecture.
	 */
	#define REDUCE_TO_ROW__ITEMS_PER_THREAD (4)
#endif

// ---------------------------------------------

// Number of items simultaneously read from global memory by each thread in div_sub()
#if DIV_SUB__ITEMS_PER_THREAD <= 0

	#undef DIV_SUB__ITEMS_PER_THREAD

	/* Maximum values with a 100% of occupancy:
	 * Compute Capability:
	 *	1.0 --	1.1:	8
	 *	1.2 --	1.3:	5
	 *		2.x:	6
	 *		3.0:	9
	 *		3.5:	10 (sub), 11 (div)
	 *
	 * ATTENTION:
	 *	+ This constant is also required in HOST code. Therefore, please
	 *	  do NOT use "__CUDA_ARCH__" to select different values, since
	 *	  it will not be defined there.
	 *
	 *	+ Change the default value ONLY if you are compiling this program
	 *	  for a particular SM architecture.
	 */
	#define DIV_SUB__ITEMS_PER_THREAD (5)
#endif

// ---------------------------------------------

// Number of items simultaneously read from global memory by each thread in mul_div()
#if MUL_DIV__ITEMS_PER_THREAD <= 0

	#undef MUL_DIV__ITEMS_PER_THREAD

	/* Maximum values with a 100% of occupancy:
	 * Compute Capability:
	 *	1.0 --	1.1:	7
	 *	1.2 --	1.3:	5
	 *		2.0:	6
	 *		3.0:	8
	 *		3.5:	9
	 *
	 * ATTENTION:
	 *	+ This constant is also required in HOST code. Therefore, please
	 *	  do NOT use "__CUDA_ARCH__" to select different values, since
	 *	  it will not be defined there.
	 *
	 *	+ Change the default value ONLY if you are compiling this program
	 *	  for a particular SM architecture.
	 */
	#define MUL_DIV__ITEMS_PER_THREAD (5)
#endif

// ---------------------------------------------

// Number of items simultaneously read from global memory by each thread in adjust()
#if ADJUST__ITEMS_PER_THREAD <= 0

	#undef ADJUST__ITEMS_PER_THREAD

	/* Maximum values with a 100% of occupancy:
	 * Compute Capability:
	 *		2.0:	6
	 *		3.0:	8
	 *		3.5:	9
	 *
	 * ATTENTION:
	 *	+ This constant is also required in HOST code. Therefore, please
	 *	  do NOT use "__CUDA_ARCH__" to select different values, since
	 *	  it will not be defined there.
	 *
	 *	+ Change the default value ONLY if you are compiling this program
	 *	  for a particular SM architecture.
	 */
	#define ADJUST__ITEMS_PER_THREAD (4)
#endif

// ---------------------------------------------

// Number of items simultaneously read from global memory by each thread in idx_max()
#if IDX_MAX__ITEMS_PER_THREAD < 2

	#undef IDX_MAX__ITEMS_PER_THREAD

	/* Maximum values with a 100% of occupancy:
	 * Compute Capability:
	 *	1.0 --	1.1:	3 (except for block_width > 32)
	 *	1.2 --	1.3:	9
	 *		2.0:	11-12
	 *
	 * ATTENTION:
	 *	+ This constant is also required in HOST code. Therefore, please
	 *	  do NOT use "__CUDA_ARCH__" to select different values, since
	 *	  it will not be defined there.
	 *
	 *	+ Change the default value ONLY if you are compiling this program
	 *	  for a particular SM architecture.
	 *
	 *	+ IDX_MAX__ITEMS_PER_THREAD >= 2
	 */
	#define IDX_MAX__ITEMS_PER_THREAD (3)
#endif

// ---------------------------------------------
// ---------------------------------------------

/* C linkage, not C++ */
#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * d_accum_A[j] = SUM( d_A[...][j] )
 *
 * height <= (grid_extension * grid_length) * REDUCE_TO_ROW__ITEMS_PER_THREAD * block_height
 * d_Tmp: Temporary storage. Ignored if grid_length == 1.
 * size_of( d_Tmp ) >= (grid_extension * grid_length) * pitch
 * length( d_accum_A ) >= pitch
 *
 * block_height <= (maxThreadsPerBlock / pitch), and must be a power of 2.
 * (grid_extension * grid_length) <= UINT_MAX
 * "pitch" must be a multiple of 'memory_alignment'.
 *
 * WARNING:
 *	height > 1
 */
NMFGPU_HOST void reduce_to_row( real const *NMFGPU_RESTRICT d_A, index_t height, index_t pitch, real *NMFGPU_RESTRICT d_Tmp, index_t block_height,
				index_t grid_length, index_t grid_extension, cudaStream_t stream_AccA, real *NMFGPU_RESTRICT d_accum_A );

////////////////////////////////////////////////

/*
 * d_A = d_B <op> d_A
 *
 * <op> is "./" or "-"
 *
 * matrix_size <= (grid_extension * grid_length * DIV_SUB__ITEMS_PER_THREAD * block_size)
 * block_size <= maxThreadsPerBlock
 * grid_length  <= maxGridSizeX
 * grid_extension <= MIN( grid_length, maxGridSizeY )
 *
 * div_operator: 'True' if operation to perform is a floating-point division.
 *		Otherwise, a subtraction is performed.
 */
NMFGPU_HOST void div_sub( real *NMFGPU_RESTRICT d_A, real const *NMFGPU_RESTRICT d_B, size_t matrix_size, index_t block_size, index_t grid_length,
			index_t grid_extension, bool div_operator, cudaStream_t stream_A );

////////////////////////////////////////////////

/*
 * d_A[i][j] = d_A[i][j] .* d_Aux[i][j] ./ d_accum_b[j]
 *
 * height <= (grid_extension * grid_length) * MUL_DIV__ITEMS_PER_THREAD * block_height
 * Size_of(d_accum_b) >= pitch
 * block_height <= (maxThreadsPerBlock / pitch)
 * grid_length  <= maxGridSizeX
 * grid_extension <= MIN( grid_length, maxGridSizeY )
 * "pitch" must be a multiple of 'memory_alignment'.
 */
NMFGPU_HOST void mul_div( real *NMFGPU_RESTRICT d_A, real const *NMFGPU_RESTRICT d_Aux, real const *NMFGPU_RESTRICT d_accum_b, index_t height,
			index_t pitch, index_t block_height, index_t grid_length, index_t grid_extension, cudaStream_t stream_A );

////////////////////////////////////////////////

/*
 * d_A = MAX( d_A , R_MIN )
 *
 * Adjusts d_A[ height ][ pitch ] to avoid underflow.
 *
 * height <= (grid_extension * grid_length) * ADJUST__ITEMS_PER_THREAD * block_height
 * block_height <= (maxThreadsPerBlock / pitch)
 * grid_length  <= maxGridSizeX
 * grid_extension <= MIN( grid_length, maxGridSizeY )
 * "pitch" must be a multiple of 'memory_alignment'.
 */
NMFGPU_HOST void adjust( real *NMFGPU_RESTRICT d_A, index_t height, index_t pitch, index_t block_height, index_t grid_length,
			index_t grid_extension, cudaStream_t stream_A );

////////////////////////////////////////////////

/*
 * Computes the maximum value of each row in d_A[] and stores its column index in d_Idx[].
 * That is, returns d_Idx[i], such that:
 *	d_A[i][ d_Idx[i] ] == max( d_A[i][...] ).
 * where
 *	0 <= max_val_idx <= width <= pitch
 *
 * height <= (grid_extension * grid_length) * block_height <= size_of( d_Idx )
 * block_width must be a power of 2, and <= maxThreadsPerBlock
 * block_height <= (maxThreadsPerBlock / pitch).
 * grid_length  <= maxGridSizeX
 * grid_extension <= MIN( grid_length, maxGridSizeY )
 */
NMFGPU_HOST void idx_max( real const *NMFGPU_RESTRICT d_A, index_t height, index_t width, index_t pitch, index_t block_width, index_t block_height,
			index_t grid_length, index_t grid_extension, cudaStream_t stream_A, index_t *NMFGPU_RESTRICT d_Idx );

////////////////////////////////////////////////
////////////////////////////////////////////////

#ifdef __cplusplus
} /* extern "C" */
#endif

////////////////////////////////////////////////
////////////////////////////////////////////////

#endif /* NMFGPU_GPU_KERNELS_CUH  */
