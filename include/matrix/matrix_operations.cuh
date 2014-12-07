/************************************************************************
 *
 * NMF-mGPU -- Non-negative Matrix Factorization on multi-GPU systems.
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
 * matrix_operations.cuh
 *	Routines for matrix algebraic operations and data transfers.
 *	Launches kernels on the GPU.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Data type:
 *		NMFGPU_SINGLE_PREC: Makes use of single-precision data (i.e., 'float').
 *
 *	CPU timing:
 *		NMFGPU_PROFILING_GLOBAL: Compute total elapsed time.
 *
 *	GPU timing (WARNING: They PREVENT asynchronous operations. The CPU thread is blocked on synchronization):
 *		NMFGPU_PROFILING_TRANSF: Compute timing of data transfers. Shows additional information.
 *		NMFGPU_PROFILING_KERNELS: Compute timing of CUDA kernels. Shows additional information.
 *
 *	Additional information:
 *		NMFGPU_VERBOSE_2: Shows the parameters on some routine calls.
 *
 *	Debug / Testing:
 *		NMFGPU_CPU_RANDOM: Uses the CPU (host) random generator (not the CURAND library).
 *		NMFGPU_DEBUG: Shows the result of each matrix operation and data transfer.
 *		NMFGPU_DEBUG_TRANSF: Shows the result of each data transfer.
 *		NMFGPU_DEBUG_REDUCT: Shows partial results of the reduction operation.
 *		NMFGPU_SYNC_TRANSF: Performs synchronous data transfers.
 *
 **********************************************************
 *
 * NOTE: In order to improve performance:
 *
 *	+ The number of columns is rounded up to a multiple of <memory_alignment>.
 *	  The padded dimension is referred as "pitch".
 *
 *	  This leads to the following limits:
 *		- Maximum number of columns (padded or not): matrix_max_pitch.
 *		- Maximum number of rows: matrix_max_non_padded_dim.
 *		- Maximum number of items: matrix_max_num_items.
 *
 *	  All four GLOBAL variables must be initialized with the
 *	  set_matrix_limits() function.
 *
 ****************
 *
 * Mapped Memory on integrated GPUs:
 *
 * On integrated systems, such as notebooks, where device memory and host memory are physically the
 * same (but disjoint regions), any data transfer between host and device memory is superfluous.
 * In such case, host memory is mapped into the address space of the device, and all transfer
 * operations are skipped. Memory for temporary buffers (e.g., d_WH or d_Aux) is also allocated
 * on the HOST and then mapped. This saves device memory, which is typically required for graphics/video
 * operations.
 *
 * This feature is disabled if NMFGPU_FORCE_BLOCKS is non-zero.
 *
 **********************************************************/

#if ! NMFGPU_MATRIX_OPERATIONS_CUH
#define NMFGPU_MATRIX_OPERATIONS_CUH (1)

#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
	#include "timing.cuh"
#endif
#include "matrix/matrix_io_routines.h"	/* matrix_tags_t */
#include "real_type.h"
#include "index_type.h"

#include <cuda_runtime_api.h>

////////////////////////////////////////////////
////////////////////////////////////////////////

/* Selects the appropriate "restrict" keyword. */

#undef RESTRICT

#if __CUDACC__				/* CUDA source code */
	#define RESTRICT __restrict__
#else					/* C99 source code */
	#define RESTRICT restrict
#endif

/* C linkage, not C++. */
#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Returns the maximum height supported by this GPU device,
 * for the given pitch, and regardless of the available memory.
 */
index_t gpu_max_height( index_t pitch );

////////////////////////////////////////////////

/*
 * Returns the maximum number of items in a matrix supported by this GPU device,
 * regardless of the available memory.
 */
size_t gpu_max_nitems( void );

////////////////////////////////////////////////

/*
 * Initializes all kernel parameters.
 */
void init_kernel_params( index_t pitch );

////////////////////////////////////////////////

/*
 * Partially prints device matrix content.
 * SYNCHRONOUSLY downloads a matrix from the GPU and shows its content (data, name, headers and/or labels).
 *
 * If "transpose" is 'true':
 * - Reads from "dMatrix": <nrows> rows and <ncols> columns (padded to <pitch>).
 * - Shows:
 *	<ncols> rows and <nrows> columns.
 *	<ncols> row labels from mt->headers, and <nrows> column headers from mt->labels.
 *
 * ncols <= pitch.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int show_device_matrix( void const *RESTRICT dMatrix, index_t nrows, index_t ncols, index_t pitch, bool real_data, bool transpose,
			bool all_processes, struct matrix_tags_t const *RESTRICT mt );

////////////////////////////////////////////////

/*
 * d_A = random value
 *
 * width <= padding
 *
 * If NMFGPU_DEBUG || NMFGPU_VERBOSE_2:
 *	transpose: 'True' if matrix is matrix is transposed.
 *
 * If 'event_A' is non-NULL, the operation is recorded as an event.
 *
 * WARNING: Requires the CURAND Library properly initialized.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_random( real *RESTRICT d_A, index_t height, index_t width, index_t padding,
			#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2
				bool transpose,
			#endif
			#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 || (! NMFGPU_PROFILING_GLOBAL)
				char const *RESTRICT const matrix_name,
			#endif
			cudaStream_t stream_A, cudaEvent_t *RESTRICT event_A );

////////////////////////////////////////////////

/*
 * d_accum_A[ i ] = SUM( d_A[ i ][...] )
 *
 * Reduces matrix d_A to a row.
 *
 * d_Tmp: Temporary storage. Ignored if height <= 2
 * size_of( d_Tmp ) <= (height/REDUCE_TO_ROW__ITEMS_PER_THREAD) * pitch
 * length( d_accum_A ) >= pitch
 *
 * 'pitch' must be a multiple of 'memory_alignment', and <= maxThreadsPerBlock.
 *
 * The operation is recorded with "event_reduction".
 *
 * WARNING:
 *	(height / (prev_power_2(maxBlockHeight_pitch) * REDUCE_TO_ROW__ITEMS_PER_THREAD)) <= UINT_MAX
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_to_row( real const *RESTRICT d_A, index_t height, index_t pitch,
			#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG
				index_t width,
			#endif
			#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				char const *RESTRICT const matrix_name,
			#endif
			real *RESTRICT d_Tmp, real *RESTRICT d_accum_A, cudaStream_t stream_AccA );

////////////////////////////////////////////////

/*
 * d_A = d_B <op> d_A
 *
 * <op> is "./" or "-"
 *
 * div_operator: 'True' if operation to perform is a floating-point division.
 *		Otherwise, a subtraction is performed.
 *
 * If host memory was NOT mapped, kernel launch is delayed upon event "event_B" completes.
 * Then, the operation is registered using the same event object.
 *
 * 'pitch' must be a multiple of 'memory_alignment'.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_div_sub( real *RESTRICT d_A, real const *RESTRICT d_B, index_t height, index_t pitch,
			#if NMFGPU_DEBUG
				index_t width,
			#endif
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				char const *RESTRICT const matrix_name_A, char const *RESTRICT const matrix_name_B,
			#endif
			bool div_operator,
			#if NMFGPU_PROFILING_KERNELS
				timing_data_t *RESTRICT td,
			#endif
			cudaStream_t stream_A, cudaEvent_t event_B );

////////////////////////////////////////////////

/*
 * d_A[i][j] = d_A[i][j] .* d_Aux[i][j] ./ d_accum_B[j]
 *
 * length(d_accum_B) >= pitch
 *
 * Kernel launch is delayed upon event "event_accB" completes.
 * Then, the operation is registered using the same event object.
 *
 * 'pitch' must be a multiple of 'memory_alignment', and <= maxThreadsPerBlock.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_mul_div( real *RESTRICT d_A, real const *RESTRICT d_Aux, real const *RESTRICT d_accum_B, index_t height, index_t pitch,
			#if NMFGPU_DEBUG
				index_t width, bool transpose,
			#endif
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				char const *RESTRICT const matrix_name_A,
			#endif
			#if NMFGPU_DEBUG
				char const *RESTRICT const matrix_name_Aux, char const *RESTRICT const matrix_name_accB,
			#endif
			cudaStream_t stream_A );

////////////////////////////////////////////////

/*
 * d_A = MAX( d_A , R_MIN )
 *
 * 'pitch' must be a multiple of 'memory_alignment', and <= maxThreadsPerBlock.
 *
 * If 'event_A' is non-NULL, delays the operation until such events completes.
 * Then, the operation is recorded using the same event object.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_adjust( real *RESTRICT d_A, index_t height, index_t pitch,
			#if NMFGPU_DEBUG
				index_t width, bool transpose,
			#endif
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				char const *RESTRICT const matrix_name_A,
			#endif
			cudaStream_t stream_A, cudaEvent_t *RESTRICT event_A );

////////////////////////////////////////////////

/*
 * Computes the maximum value of each row in d_A[] and stores its column index in d_Idx[].
 * That is, returns d_Idx[i], such that:
 *	d_A[i][ d_Idx[i] ] == max( d_A[i][...] ).
 *
 * size_of( d_Idx ) >= height
 * width <= pitch <= maxThreadsPerBlock
 * In addition, "pitch" must be a multiple of 'memory_alignment'.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_idx_max( real const *RESTRICT d_A, index_t width, index_t pitch, index_t height,
			#if NMFGPU_DEBUG
				bool transpose,
			#endif
			#if NMFGPU_DEBUG || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_KERNELS))
				char const *RESTRICT const matrix_name_A,
			#endif
			#if NMFGPU_DEBUG
				char const *RESTRICT const matrix_name_Idx,
			#endif
			cudaStream_t stream_A, index_t *RESTRICT d_Idx );

////////////////////////////////////////////////

/*
 * Transfers a matrix from the HOST (CPU) to the DEVICE (GPU) as a row vector.
 *
 * d_A[1..height][1..pitch] <--- A[1..height][1..pitch],
 *
 * If 'event_A' is non-NULL, the operation is recorded as an event.
 *
 * NOTE: If host memory was mapped, the transfer operation is SKIPPED, but NOT the event record (if provided).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int upload_matrix( real const *RESTRICT A, index_t height, index_t pitch, real *RESTRICT d_A,
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
				index_t width, bool transpose,
			#endif
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 || (! NMFGPU_PROFILING_GLOBAL)
				char const *RESTRICT const matrix_name_A,
			#endif
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
				char const *RESTRICT const matrix_name_dA,
			#endif
			#if NMFGPU_PROFILING_TRANSF
				timing_data_t *RESTRICT const upload_timing,
			#endif
			cudaStream_t stream_A, cudaEvent_t *RESTRICT event_A );

////////////////////////////////////////////////

/*
 * Transfers (a portion of) a matrix from the HOST (CPU) to the DEVICE (GPU).
 *
 * d_A[1..height][1..block_pitch] <--- pA[1..height][1..block_pitch],
 * where:
 *	pA[1..height][1..block_pitch] == &A[strow..(strow+height)][stcol..(stcol+block_pitch)]
 *
 * block_pitch: Matrix block pitch.
 * block_width <= block_pitch
 * strow: Starting row.
 * stcol: Starting column.
 *
 * 0 <= stcol < pitch.
 * Matrix is ROW-wise (i.e., it is NOT transposed).
 *
 * The transfer is delayed until the event 'event_A' has completed all previous operations.
 * Then, the operation is recorded using the same event object.
 *
 * It also checks that (stcol + block_pitch) <= pitch,
 * and adjusts the width of the block to be transferred, if necessary.
 *
 * NOTE: If host memory was mapped, the transfer operation and ALL event action(s) are SKIPPED.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int upload_matrix_partial( real const *RESTRICT pA, index_t height, index_t pitch, index_t strow, index_t stcol,
				#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
					index_t block_width,
				#endif
				#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 || (! NMFGPU_PROFILING_GLOBAL)
					char const *RESTRICT const matrix_name_A,
				#endif
				#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
					char const *RESTRICT const matrix_name_dA,
				#endif
				index_t block_pitch, real *RESTRICT d_A, cudaStream_t stream_A, cudaEvent_t event_A
				#if NMFGPU_PROFILING_TRANSF
					, timing_data_t *RESTRICT const upload_timing
				#endif
			);

////////////////////////////////////////////////

/*
 * Transfers a matrix from the DEVICE (GPU) to HOST (CPU), as a row vector.
 *
 * A[1..height][1..pitch] <--- d_A[1..height][1..pitch],
 *
 * nitems == (height * pitch)
 *
 * NOTE: If host memory was mapped, the transfer operation is SKIPPED.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int download_matrix( void *__restrict__ A, size_t nitems, size_t data_size, void const *RESTRICT d_A,
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
				index_t height, index_t width, index_t pitch, bool real_data, bool transpose,
				char const *RESTRICT const matrix_name_A,
			#endif
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
				|| ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_TRANSF))
				char const *RESTRICT const matrix_name_dA,
			#endif
			#if NMFGPU_PROFILING_TRANSF
				timing_data_t *RESTRICT const download_timing,
			#endif
			cudaStream_t stream_A );

////////////////////////////////////////////////
////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

#undef RESTRICT

////////////////////////////////////////////////
////////////////////////////////////////////////

#endif /* NMFGPU_MATRIX_OPERATIONS_CUH */
