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
 * GPU_kernels.cuh
 *	Kernel code to be executed on the device (GPU).
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

#include "real_type.h"
#include "index_type.h"

#include <cuda_runtime_api.h>

////////////////////////////////////////////////
////////////////////////////////////////////////

/* Constants */

/* Data type used for array length on the GPU (i.e., the equivalent to "size_t",
 * but in number of items, not in bytes). However, it is recommended to currently
 * keep it as an alias of 'index_t' (actually, it should be 'unsigned int', but
 * the former avoids possible signed-unsigned type-casting operations).
 *
 * NOTE: Keep in sync with the GPUSIZE_MAX constant defined at "GPU_kernels.cuh".
 */

// ---------------------------------------------

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

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * d_accum_A[j] = SUM( d_A[...][j] )
 *
 * height <= (dimGrid.y * dimGrid.x) * REDUCE_TO_ROW__ITEMS_PER_THREAD * block_height
 * d_Tmp: Temporary storage. Ignored if grid_length == 1.
 * size_of( d_Tmp ) >= (dimGrid.y * dimGrid.x) * pitch
 * length( d_accum_A ) >= pitch
 *
 * block_height <= (maxThreadsPerBlock / pitch), and must be a power of 2.
 * (dimGrid.y * dimGrid.x) <= UINT_MAX
 * "pitch" must be a multiple of 'memory_alignment'.
 *
 * WARNING:
 *	height > 1
 */
__host__ void reduce_to_row( real const *__restrict__ d_A, index_t height, index_t pitch, real *__restrict__ d_Tmp, index_t block_height,
				dim3 dimGrid, cudaStream_t stream_AccA, real *__restrict__ d_accum_A );

////////////////////////////////////////////////

/*
 * d_A = d_B <op> d_A
 *
 * <op> is "./" or "-"
 *
 * matrix_size <= (dimGrid.y * dimGrid.x * DIV_SUB__ITEMS_PER_THREAD * block_size)
 * block_size <= maxThreadsPerBlock
 * dimGrid.x  <= maxGridSizeX
 * dimGrid.y <= MIN( dimGrid.x, maxGridSizeY )
 *
 * div_operator: 'True' if operation to perform is a floating-point division.
 *		Otherwise, a subtraction is performed.
 */
__host__ void div_sub( real *__restrict__ d_A, real const *__restrict__ d_B, size_t matrix_size, index_t block_size, dim3 dimGrid,
			bool div_operator, cudaStream_t stream_A );

////////////////////////////////////////////////

/*
 * d_A[i][j] = d_A[i][j] .* d_Aux[i][j] ./ d_accum_b[j]
 *
 * height <= (dimGrid.y * dimGrid.x) * MUL_DIV__ITEMS_PER_THREAD * block_height
 * Size_of(d_accum_b) >= pitch
 * block_height <= (maxThreadsPerBlock / pitch)
 * dimGrid.x  <= maxGridSizeX
 * dimGrid.y <= MIN( dimGrid.x, maxGridSizeY )
 * "pitch" must be a multiple of 'memory_alignment'.
 */
__host__ void mul_div( real *__restrict__ d_A, real const *__restrict__ d_Aux, real const *__restrict__ d_accum_b, index_t height, index_t pitch,
			index_t block_height, dim3 dimGrid, cudaStream_t stream_A );

////////////////////////////////////////////////

/*
 * d_A = MAX( d_A , R_MIN )
 *
 * Adjusts d_A[ height ][ pitch ] to avoid underflow.
 *
 * height <= (dimGrid.y * dimGrid.x) * ADJUST__ITEMS_PER_THREAD * block_height
 * block_height <= (maxThreadsPerBlock / pitch)
 * dimGrid.x  <= maxGridSizeX
 * dimGrid.y <= MIN( dimGrid.x, maxGridSizeY )
 * "pitch" must be a multiple of 'memory_alignment'.
 */
__host__ void adjust( real *__restrict__ d_A, index_t height, index_t pitch, index_t block_height, dim3 dimGrid, cudaStream_t stream_A );

////////////////////////////////////////////////

/*
 * Computes the maximum value of each row in d_A[] and stores its column index in d_Idx[].
 * That is, returns d_Idx[i], such that:
 *	d_A[i][ d_Idx[i] ] == max( d_A[i][...] ).
 * where
 *	0 <= max_val_idx <= width <= pitch
 *
 * height <= (dimGrid.y * dimGrid.x) * dimBlock.y <= size_of( d_Idx )
 * dimBlock.x must be a power of 2, and <= maxThreadsPerBlock
 * dimBlock.y <= (maxThreadsPerBlock / pitch).
 * dimGrid.x  <= maxGridSizeX
 * dimGrid.y <= MIN( dimGrid.x, maxGridSizeY )
 */
__host__ void idx_max( real const *__restrict__ d_A, index_t height, index_t width, index_t pitch, dim3 dimBlock, dim3 dimGrid,
			cudaStream_t stream_A, index_t *__restrict__ d_Idx );

////////////////////////////////////////////////
////////////////////////////////////////////////

#endif /* NMFGPU_GPU_KERNELS_CUH  */
