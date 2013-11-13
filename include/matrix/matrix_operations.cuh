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
 * matrix_operations.cuh
 *	Routines for matrix algebraic operations and data transfers.
 *	Launches kernels on the GPU.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Timing (WARNING: They PREVENT asynchronous operations):
 *		NMFGPU_PROFILING_TRANSF: Compute timing of data transfers (should be used with NMFGPU_SYNC_TRANSF).
 *		NMFGPU_PROFILING_KERNELS: Compute timing of CUDA kernels.
 *
 *	Additional information:
 *		NMFGPU_VERBOSE_2: Shows the parameters on some routine calls.
 *
 *	Debug / Testing:
 *		NMFGPU_DEBUG: Shows the result of each matrix operation and data transfer.
 *		NMFGPU_DEBUG_TRANSF: Shows the result of each data transfer.
 *		NMFGPU_DEBUG_REDUCT: Shows partial results of the reduction operation.
 *		NMFGPU_SYNC_TRANSF: Performs synchronous data transfers.
 *
 **********************************************************
 *
 * NOTE: In order to improve performance:
 *
 *	- All matrices include useless data for padding. Padded dimensions
 *	  are denoted with the 'p' character, e.g., 'Mp' (i.e., M + padding)
 *	  or 'Kp' (factorization_rank + padding).
 *
 *	- Padded dimensions are a multiple of memory_alignment
 *	  (a global variable which currently is equal to warpSize or warpSize/2).
 *
 **********************************************************/

#if ! NMFGPU_MATRIX_OPERATIONS_CUH
#define NMFGPU_MATRIX_OPERATIONS_CUH (1)

//////////////////////////////////////////////////////

#include "index_type.h"
#include "real_type.h"
#include "matrix/matrix_io_routines.h"	/* matrix_labels */

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Partially prints device matrix content.
 * SYNCHRONOUSLY downloads a matrix from the GPU and shows its content (data, name, headers and/or labels).
 *
 * If 'transpose' is 'true', transposes matrix as follows:
 * - Matrix dimension in memory: <ncols> rows, <nrows> columns.
 * - Matrix dimension on screen: <nrows> rows, <ncols> columns.
 * - Shows <ncols> ml->headers (as column headers) and <nrows> ml->labels (as row labels).
 *
 * ncols <= pitch, unless matrix transposing is set (in that case, nrows <= padding).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int show_device_matrix( real const *__restrict__ dMatrix, index_t nrows, index_t ncols, index_t pitch, bool transpose,
			struct matrix_labels const *__restrict__ ml );

// -----------------------------------

/*
 * Partially prints device matrix content (INTEGER version).
 * SYNCHRONOUSLY downloads a matrix from the GPU and shows its content (data, name, headers and/or labels).
 *
 * If 'transpose' is 'true', transposes matrix as follows:
 * - Matrix dimension in memory: <ncols> rows, <nrows> columns.
 * - Matrix dimension on screen: <nrows> rows, <ncols> columns.
 * - Shows <ncols> ml->headers (as column headers) and <nrows> ml->labels (as row labels).
 *
 * ncols <= pitch, unless matrix transposing is set (in that case, nrows <= padding).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int show_device_matrix_int( index_t const *__restrict__ dMatrix, index_t nrows, index_t ncols, index_t pitch, bool transpose,
			struct matrix_labels const *__restrict__ ml );

// -----------------------------------

/*
 * d_A = random value
 *
 * width <= padding
 *
 * If NMFGPU_DEBUG || NMFGPU_VERBOSE_2:
 *	transpose: 'True' if matrix is matrix is transposed.
 *
 * Operation performed with stream "matrix_stream".
 *
 * WARNING: Requires the CURAND Library properly initialized.
 */
void matrix_random( real *__restrict__ d_A, index_t height, index_t width,
		#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2
			bool transpose, char const *__restrict__ const matrix_name,
		#endif
		index_t padding );

// -----------------------------------

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
 * WARNING:
 *	- On Compute Capability 1.x:
 *		height < PREV_POWER_2(maxBlockHeight_pitch) * REDUCE_TO_ROW__ITEMS_PER_THREAD * (2**24)
 *		('REDUCE_TO_ROW__ITEMS_PER_THREAD' is a constant defined in "GPU_kernels.h").
 */
void matrix_to_row( real const *__restrict__ d_A, index_t height, index_t pitch,
		#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG
		index_t width, char const *__restrict__ const matrix_name,
		#endif
		real *__restrict__ d_Tmp, real *__restrict__ d_accum_A );

// -----------------------------------------

/*
 * d_A = d_B <op> d_A
 *
 * <op> is "./" or "-"
 *
 * div_operand: 'True' if operation to perform is a floating-point division.
 *		Otherwise, a subtraction is performed.
 *
 * If 'event_B' is non-NULL, kernel launch is delayed upon event completion.
 * Then, the operation is registered using the same event object.
 *
 * 'pitch' must be a multiple of 'memory_alignment'.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		matrix_size < threadsPerBlock * DIV_SUB__ITEMS_PER_THREAD * (2**24)
 *		('DIV_SUB__ITEMS_PER_THREAD' is a constant defined in "GPU_kernels.h")
 */
void matrix_div_sub( real *__restrict__ d_A, real const *__restrict__ d_B, index_t height, index_t pitch,
		#if NMFGPU_DEBUG
			index_t width, char const *__restrict__ const matrix_name_A, char const *__restrict__ const matrix_name_B,
		#endif
		bool div_operand, cudaStream_t stream_A, cudaEvent_t *__restrict__ event_B, timing_data_t *__restrict__ td );

// -----------------------------------------

/*
 * d_A[i][j] = d_A[i][j] .* d_Aux[i][j] ./ d_accum_B[j]
 *
 * length(d_accum_B) >= pitch
 *
 * 'pitch' must be a multiple of 'memory_alignment', and <= maxThreadsPerBlock.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		height < maxBlockHeight_pitch * MUL_DIV__ITEMS_PER_THREAD * (2**24)
 *		('MUL_DIV__ITEMS_PER_THREAD' is a constant defined in "GPU_kernels.h").
 */
void matrix_mul_div( real *__restrict__ d_A, real const *__restrict__ d_Aux, real const *__restrict__ d_accum_B, index_t height, index_t pitch,
			#if NMFGPU_DEBUG
				index_t width, bool transpose, char const *__restrict__ const matrix_name_A,
				char const *__restrict__ const matrix_name_Aux, char const *__restrict__ const matrix_name_accB,
			#endif
			cudaStream_t stream_A );

// -----------------------------------------

/*
 * d_A = MAX( d_A , R_MIN )
 *
 * 'pitch' must be a multiple of 'memory_alignment', and <= maxThreadsPerBlock.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		height < maxBlockHeight_pitch * ADJUST__ITEMS_PER_THREAD * (2**24)
 *		('ADJUST__ITEMS_PER_THREAD' is a constant defined in "GPU_kernels.h").
 */
void matrix_adjust( real *__restrict__ d_A, index_t height, index_t pitch,
			#if NMFGPU_DEBUG
				index_t width, bool transpose, char const *__restrict__ const matrix_name_A,
			#endif
			cudaStream_t stream_A );

// -----------------------------------------

/*
 * Computes the maximum value of each row in d_A[] and stores its column index in d_Idx[].
 * That is, returns d_Idx[i], such that:
 *	d_A[i][ d_Idx[i] ] == max( d_A[i][...] ).
 *
 * size_of( d_Idx ) >= height
 *
 * 'pitch' must be a multiple of 'memory_alignment', and <= maxThreadsPerBlock.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		height < (threadsPerBlock/block_width) * (2**24)
 */
void matrix_idx_max( real const *__restrict__ d_A, index_t width, index_t pitch, index_t height,
			#if NMFGPU_DEBUG
				char const *__restrict__ const matrix_name_A, char const *__restrict__ const matrix_name_Idx,
			#endif
			cudaStream_t stream_A, index_t *__restrict__ d_Idx );

// -----------------------------------------

/*
 * Transfers a matrix from the HOST (CPU) to the DEVICE (GPU) as a row vector.
 *
 * d_A[1..height][1..pitch] <--- A[1..height][1..pitch],
 *
 * The transfer is performed with stream "matrix_stream". No event is recorded.
 */
void upload_matrix( real const *__restrict__ A, index_t height, index_t pitch, real *__restrict__ d_A,
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
				index_t width, bool transpose, char const *__restrict__ const matrix_name_A,
				char const *__restrict__ const matrix_name_dA,
			#endif
			timing_data_t *__restrict__ const upload_timing );

// -----------------------------------------

/*
 * Transfers a matrix from the HOST (CPU) to the DEVICE (GPU) as a row vector.
 *
 * d_A[1..height][1..pitch] <--- A[1..height][1..pitch],
 *
 * The transfer is performed with stream "matrix_stream", and is recorded as event "transfer_event".
 */
void upload_matrix_event( real const *__restrict__ A, index_t height, index_t pitch, real *__restrict__ d_A,
				#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
					index_t width, bool transpose, char const *__restrict__ const matrix_name_A,
					char const *__restrict__ const matrix_name_dA,
				#endif
				timing_data_t *__restrict__ const upload_timing );

// -----------------------------------------

/*
 * Transfers (a portion of) a matrix from the HOST (CPU) to the DEVICE (GPU).
 *
 * d_A[1..height][1..block_pitch] <--- p_A[1..height][1..block_pitch],
 * where:
 *	p_A[1..height][1..block_pitch] == &A[X..(X+height)][offset..(offset+block_pitch)]
 *
 * block_pitch: Matrix block pitch.
 * block_width <= block_pitch
 * offset: Starting COLUMN.
 *
 * 0 <= offset < pitch.
 * Matrix is ROW-wise (i.e., it is NOT transposed).
 *
 * The transfer is delayed until the event 'event_A' has completed all previous operations.
 * Then, the operation is recorded using the same event object.
 *
 * It also checks that (offset + block_pitch) <= pitch,
 * and adjusts the width of the block to be transferred, if necessary.
 */
void upload_matrix_partial( real const *__restrict__ p_A, index_t height, index_t pitch, index_t offset,
				#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
					index_t block_width, char const *__restrict__ const matrix_name_A,
					char const *__restrict__ const matrix_name_dA,
				#endif
				index_t block_pitch, real *__restrict__ d_A, cudaEvent_t event_A, cudaStream_t stream_A,
				timing_data_t *__restrict__ upload_timing );

// -----------------------------------------

/*
 * Transfers a matrix from the DEVICE (GPU) to HOST (CPU), as a row vector.
 *
 * A[1..height][1..pitch] <--- d_A[1..height][1..pitch],
 *
 * The transfer is performed with stream "matrix_stream". No event is recorded.
 */
void download_matrix( real const *__restrict__ A, index_t height, index_t pitch, real *__restrict__ d_A,
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
				index_t width, bool transpose, char const *__restrict__ const matrix_name_A,
				char const *__restrict__ const matrix_name_dA,
			#endif
			timing_data_t *__restrict__ const download_timing );

// -----------------------------------------

/*
 * Transfers an INTEGER matrix from the DEVICE (GPU) to HOST (CPU), as a row vector.
 *
 * A[1..height][1..pitch] <--- d_A[1..height][1..pitch],
 *
 * The transfer is performed with stream "matrix_stream". No event is recorded.
 */
void download_matrix_int( index_t const *__restrict__ A, index_t height, index_t pitch, index_t *__restrict__ d_A,
				#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
					index_t width, bool transpose, char const *__restrict__ const matrix_name_A,
					char const *__restrict__ const matrix_name_dA,
				#endif
				timing_data_t *__restrict__ const download_timing );

////////////////////////////////////////////////

#endif /* NMFGPU_MATRIX_OPERATIONS_CUH */
