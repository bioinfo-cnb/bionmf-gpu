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
 * GPU_kernels.cuh
 *	Kernel code to be executed on the device (GPU).
 *
 * NOTE:
 *	- All matrices include useless data for padding. It is sometimes denoted as 'pitch'.
 *
 *********************************************************/

#if ! NMFGPU_GPU_KERNELS_CUH
#define NMFGPU_GPU_KERNELS_CUH (1)

////////////////////////////////////////////////

#include "index_type.h"
#include "real_type.h"

// ---------------------------------------------
// ---------------------------------------------

/* Constants */

// Number of items simultaneously read from global memory by each thread in reduce_to_row()
#if REDUCE_TO_ROW__ITEMS_PER_THREAD <= 0

	#undef REDUCE_TO_ROW__ITEMS_PER_THREAD

	/* Maximum values with a 100% of occupancy:
	 * Compute Capability:
	 *	1.0 --	1.1:	5 (**)
	 *		1.2:	6 (**)
	 *		2.x:	6
	 *	3.0 --	3.5:	13
	 *
	 * (**): However, no more than 4 elements are processed at a time.
	 *	That is, the compiler emits 4 loads and 4 add operations; then, other 4 loads, etc.
	 *
	 * WARNING:
	 *	This constant is also required in host code. Therefore, please, do NOT use "__CUDA_ARCH__"
	 *	to select a different value for each SM architecture, since it will not defined there.
	 *	Change the default value only if you are compiling this program for a particular SM architecture.
	 */
	#define REDUCE_TO_ROW__ITEMS_PER_THREAD (4)
#endif

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
	 * WARNING:
	 *	This constant is also required in host code. Therefore, please, do NOT use "__CUDA_ARCH__"
	 *	to select a different value for each SM architecture, since it will not defined there.
	 *	Change the default value only if you are compiling this program for a particular SM architecture.
	 */
	#define DIV_SUB__ITEMS_PER_THREAD (5)
#endif


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
	 * WARNING:
	 *	This constant is also required in host code. Therefore, please, do NOT use "__CUDA_ARCH__"
	 *	to select a different value for each SM architecture, since it will not defined there.
	 *	Change the default value only if you are compiling this program for a particular SM architecture.
	 */
	#define MUL_DIV__ITEMS_PER_THREAD (5)
#endif


// Number of items simultaneously read from global memory by each thread in adjust()
#if ADJUST__ITEMS_PER_THREAD <= 0

	#undef ADJUST__ITEMS_PER_THREAD

	/* Maximum values with a 100% of occupancy:
	 * Compute Capability:
	 *	1.0 --	1.1:	7 (**)
	 *	1.2 --	1.3:	5 (**)
	 *		2.0:	6
	 *		3.0:	8
	 *		3.5:	9
	 *
	 * (**): However, no more than 4 elements are processed at a time.
	 *	That is: the compiler emits 4 loads, 4 operations, and 4 stores; then, other 4 loads, etc.
	 *
	 * WARNING:
	 *	This constant is also required in host code. Therefore, please, do NOT use "__CUDA_ARCH__"
	 *	to select a different value for each SM architecture, since it will not defined there.
	 *	Change the default value only if you are compiling this program for a particular SM architecture.
	 */
	#define ADJUST__ITEMS_PER_THREAD (4)
#endif

// Number of items simultaneously read from global memory by each thread in idx_max()
#if IDX_MAX__ITEMS_PER_THREAD <= 0

	#undef IDX_MAX__ITEMS_PER_THREAD

	/* Maximum values with a 100% of occupancy:
	 * Compute Capability:
	 *	1.0 --	1.1:	3 (except for block_width > 32)
	 *	1.2 --	1.3:	9
	 *		2.0:	11-12
	 *
	 * WARNING:
	 *	- This constant is also required in host code. Therefore, please, do NOT use "__CUDA_ARCH__"
	 *	  to select a different value for each SM architecture, since it will not defined there.
	 *	  Change the default value only if you are compiling this program for a particular SM architecture.
	 *
	 * 	- Please, set always a value > 1.
	 */
	#define IDX_MAX__ITEMS_PER_THREAD (3)
#endif

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * d_accum_A[j] = SUM( d_A[...][j] )
 *
 * matrix_size <= (grid_extension * grid_length * REDUCE_TO_ROW__ITEMS_PER_THREAD * block_height) * pitch
 * d_Tmp: Temporary storage. Ignored if grid_length == 1.
 * size_of( d_Tmp ) >= grid_extension * grid_length * pitch
 * length( d_accum_A ) >= pitch
 *
 * block_height <= (maxThreadsPerBlock / pitch), and must be a power of 2.
 * grid_length  <= maxGridSizeX
 * grid_extension <= MIN( grid_length, maxGridSizeY )
 * "pitch" must be a multiple of 'memory_alignment'.
 *
 * WARNING:
 *	- height > 1
 *	- On Compute Capability 1.x:
 *		In a non-"extended" grid (i.e., with grid_extension == 1):
 *			matrix_size <= (grid_length * REDUCE_TO_ROW__ITEMS_PER_THREAD * bs) must be < 2**24
 *		In any case, (grid_extension * grid_length) must be < 2**24
 */
__host__ void reduce_to_row( real const *__restrict__ d_A, index_t pitch, real *__restrict__ d_Tmp, index_t block_height, index_t grid_extension,
				index_t grid_length, index_t matrix_size, cudaStream_t stream_AccA, real *__restrict__ d_accum_A );

// ==========================================

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
 * div_operand: 'True' if operation to perform is a floating-point division.
 *		Otherwise, a subtraction is performed.
 * WARNING:
 *	- On Compute Capability 1.x:
 *		In a non-"extended" grid (i.e., with grid_extension == gridDim.y == 1):
 *			matrix_size <= (gridDim.x * DIV_SUB__ITEMS_PER_THREAD * block_size) must be < 2**24
 *		In any case, (gridDim.y * gridDim.x) must be < 2**24
 */
__host__ void div_sub( real *__restrict__ d_A, real const *__restrict__ d_B, index_t matrix_size, index_t block_size,
			index_t grid_extension, index_t grid_length, bool div_operand, cudaStream_t stream_A );

// ==========================================

/*
 * d_A[i][j] = d_A[i][j] .* d_Aux[i][j] ./ d_accum_b[j]
 *
 * matrix_size <= (grid_extension * grid_length * MUL_DIV__ITEMS_PER_THREAD * block_height) * pitch
 * Size_of(d_accum_b) >= pitch
 * block_height <= (maxThreadsPerBlock / pitch)
 * grid_length  <= maxGridSizeX
 * grid_extension <= MIN( grid_length, maxGridSizeY )
 * "pitch" must be a multiple of 'memory_alignment'.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		In a non-"extended" grid (i.e., with grid_extension == gridDim.y == 1):
 *			matrix_size <= (gridDim.x * MUL_DIV__ITEMS_PER_THREAD * block_height * pitch) must be < 2**24
 *		In any case, (gridDim.y * gridDim.x) must be < 2**24
 */
__host__ void mul_div( real *__restrict__ d_A, real const *__restrict__ d_Aux, real const *__restrict__ d_accum_b, index_t pitch,
			index_t matrix_size, index_t block_height, index_t grid_extension, index_t grid_length, cudaStream_t stream_A );

// ==========================================

/*
 * d_A = MAX( d_A , R_MIN )
 *
 * matrix_size <= (grid_extension * grid_length * ADJUST__ITEMS_PER_THREAD * block_height) * pitch
 * block_height <= (maxThreadsPerBlock / pitch)
 * grid_length  <= maxGridSizeX
 * grid_extension <= MIN( grid_length, maxGridSizeY )
 * "pitch" must be a multiple of 'memory_alignment'.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		In a non-"extended" grid (i.e., with grid_extension == gridDim.y == 1):
 *			matrix_size <= (gridDim.x * ADJUST__ITEMS_PER_THREAD * block_height * pitch) must be < 2**24
 *		In any case, (gridDim.y * gridDim.x) must be < 2**24
 */
__host__ void adjust( real *__restrict__ d_A, index_t pitch, index_t matrix_size, index_t block_height, index_t grid_extension,
		      index_t grid_length, cudaStream_t stream_A );

// ==========================================

/*
 * Computes the maximum value of each row in d_A[] and stores its column index in d_Idx[].
 * That is, returns d_Idx[i], such that:
 *	d_A[i][ d_Idx[i] ] == max( d_A[i][...] ).
 * where
 *	0 <= max_val_idx <= width <= pitch
 *
 * matrix_size <= (grid_extension * grid_length * block_height) * pitch
 * size_of( d_Idx ) >= grid_extension * grid_length * block_height
 *
 * block_height <= (maxThreadsPerBlock / pitch), and must be a power of 2.
 * grid_length  <= maxGridSizeX
 * grid_extension <= MIN( grid_length, maxGridSizeY )
 * "pitch" must be a multiple of 'memory_alignment'.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		In a non-"extended" grid (i.e., with grid_extension == 1):
 *			matrix_size <= (grid_length * block_height * pitch) must be < 2**24
 *		In any case, (grid_extension * grid_length) must be < 2**24
 */
__host__ void idx_max( real const *__restrict__ d_A, index_t width, index_t pitch, index_t matrix_size, index_t block_width,
		index_t block_height, index_t grid_extension, index_t grid_length, cudaStream_t stream_A, index_t *__restrict__ d_Idx );

////////////////////////////////////////////////

#endif /* NMFGPU_GPU_KERNELS_CUH  */
