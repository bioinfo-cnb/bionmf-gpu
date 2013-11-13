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
 * matrix_operations.cu
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
 *		NMFGPU_VERBOSE_2: Shows the parameters in some routine calls.
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

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>	/* [u]intmax_t */

#include <cuda_runtime_api.h>
#include <curand.h>	/* Random values */

#include "matrix/matrix_operations.cuh"
#include "GPU_kernels.cuh"
#include "GPU_setup.cuh"
#include "matrix/matrix_io.h"
#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_CONV
	#include "timing.cuh"
#endif

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

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
			struct matrix_labels const *__restrict__ ml )
{

	// Downloads the device matrix to a temporary array and shows its content.
	real *__restrict__ const buffer = malloc( nrows * pitch * sizeof(real) );
	if ( ! buffer ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Error in HOST memory allocation (malloc): %s\nError in show_device_matrix()\n",
			device_id, strerror(errno) );
		return EXIT_FAILURE;
	}

	// Synchronous data transfer.
	cudaError_t const cuda_status = cudaMemcpy( buffer, dMatrix, nrows * pitch * sizeof(real), cudaMemcpyDeviceToHost );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Error downloading DEVICE matrix (nrows=%" PRI_IDX ", ncols=%" PRI_IDX ", pitch=%"
				PRI_IDX ", transpose=%i): %s\n", device_id, nrows, ncols, pitch, transpose, cudaGetErrorString(cuda_status) );
		free( buffer );
		return EXIT_FAILURE;
	}
	/* Same code using CUBLAS:
	 *	cublasStatus_t cublas_status = cublasGetVector( nrows * pitch, sizeof(real), dMatrix, 1, buffer, 1 );
	 */

	int const status = matrix_show( buffer, nrows, ncols, pitch, transpose, ml );

	free( buffer );

	return status;

} // show_device_matrix

/////////////////////////////////////////////////////////////////////

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
			struct matrix_labels const *__restrict__ ml )
{

	// Downloads the device matrix to a temporary array and shows its content.
	real *__restrict__ const buffer = malloc( nrows * pitch * sizeof(index_t) );
	if ( ! buffer ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Error in HOST memory allocation (malloc): %s\nError in show_device_matrix_int()\n",
			device_id, strerror(errno) );
		return EXIT_FAILURE;
	}

	// *SYNCHRONOUS* data transfer.
	cudaError_t const cuda_status = cudaMemcpy( buffer, dMatrix, nrows * pitch * sizeof(index_t), cudaMemcpyDeviceToHost );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Error downloading DEVICE matrix of INTEGERS (nrows=%" PRI_IDX ", ncols=%" PRI_IDX
				", pitch=%" PRI_IDX ", transpose=%i): %s\n", device_id, nrows, ncols, pitch, transpose,
				cudaGetErrorString(cuda_status) );
		free( buffer );
		return EXIT_FAILURE;
	}
	/* Same code using CUBLAS:
	 *	cublasStatus_t cublas_status = cublasGetVector( nrows * pitch, sizeof(index_t), dMatrix, 1, buffer, 1 );
	 */

	int const status = matrix_int_show( buffer, nrows, ncols, pitch, transpose, ml );

	free( buffer );

	return status;

} // show_device_matrix_int

///////////////////////////////////////////////////////////////////////////////

/*
 * d_A = random_value
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
			index_t padding )
{

	#if NMFGPU_VERBOSE_2
		if (! device_id)
			printf("\nSetting random values to matrix '%s' (height=%" PRI_IDX ", width=%" PRI_IDX ", padding=%" PRI_IDX
				", transpose=%i)\n", matrix_name, height, width, padding, transpose );
	#endif

	// Stream for this operation
	cudaStream_t stream_A = matrix_stream;

	#if NMFGPU_DEBUG
		curandStatus_t curand_status = CURAND_STATUS_SUCCESS;
	#endif

	// ----------------------------------

	// Sets the stream

	#if NMFGPU_DEBUG
		curand_status =
	#endif

		curandSetStream( curand_generator, stream_A );

	///////////////////////////////
	#if NMFGPU_DEBUG
		if ( curand_status != CURAND_STATUS_SUCCESS ) {
			fflush(stdout);
			fprintf(stderr,"\n[GPU%" PRI_IDX "] Error setting stream for CURAND kernel launches (matrix %s): ",
				device_id, matrix_name );
			printCurandErrorString( curand_status );
		}
	#endif
	///////////////////////////////

	// ----------------------------------

	// Generates random values.

	size_t const size = height * padding;

	#if NMFGPU_DEBUG
		curand_status =
	#endif

		CURAND_GENERATE_UNIFORM_REAL( curand_generator, d_A, size );

	///////////////////////////////
	#if NMFGPU_DEBUG
		if ( curand_status != CURAND_STATUS_SUCCESS ) {
			fflush(stdout);
			fprintf(stderr,"\n[GPU%" PRI_IDX "] Error generating random values for matrix %s: ", device_id, matrix_name );
			printCurandErrorString( curand_status );
		}
		printf( "\n--- [GPU%" PRI_IDX "] Random values on matrix %s (height=%" PRI_IDX ", width=%" PRI_IDX ", padding=%" PRI_IDX
			", transpose=%i): ---\n", device_id, matrix_name, height, width, padding, transpose );
		check_cuda_status_st( cuda_status );
		show_device_matrix( d_A, height, width, padding, transpose, NULL );
	#endif
	/////////////////////////////

	// ----------------------------------

	#if NMFGPU_VERBOSE_2
	if (! device_id)
		printf("\nSetting random values to matrix '%s' (height=%" PRI_IDX ", width=%" PRI_IDX ", padding=%" PRI_IDX
			", transpose=%i)... Done.\n", matrix_name, height, width, padding, transpose );
	#endif


} //matrix_random

///////////////////////////////////////////////////////////////////////////////

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
		real *__restrict__ d_Tmp, real *__restrict__ d_accum_A )
{

	///////////////////////////////
	#if NMFGPU_DEBUG_REDUCT
		if ( ! device_id )
			printf("\n--- Begin of matrix_to_row(computeCapability=%" PRI_IDX ", width=%" PRI_IDX ", pitch=%" PRI_IDX
				", height=%" PRI_IDX ") on %s: ---\n", computeCapability, width, pitch, height, matrix_name);
	#endif
	///////////////////////////////

	// ----------------------------------------

	// Event and Stream for this operation.
	cudaEvent_t event_AccA = matrix_events;
	cudaStream_t stream_AccA = matrix_stream;

	// ----------------------------------------

	if ( height > 1 ) {

		#if NMFGPU_PROFILING_KERNELS
			index_t timing_index = 0;	// Non-extended grid (0), extended grid (1), single block (2), copy (3)
		#endif

		index_t const matrix_size = pitch * height;

		/* Uses a block width equal to <pitch>.
		 *
		 * Each block reduces, at least, <REDUCE_TO_ROW__ITEMS_PER_THREAD> times its height
		 * ('REDUCE_TO_ROW__ITEMS_PER_THREAD' is a constant defined in "GPU_kernels.h").
		 *
		 *	Please, see the reduction example in CUDA SDK for details.
		 */

		index_t block_height = PREV_POWER_2( maxBlockHeight_pitch );		// A power of two.

		index_t const abh = block_height * REDUCE_TO_ROW__ITEMS_PER_THREAD;	// "Active" block height.


		// Grid dimensions
		index_t grid_length = 1;	// <= maxGridSizeX
		index_t grid_extension = 1;	// <= maxGridSizeY, and <= grid_length

		/* "grid_extension" is required only if matrix_size > (maxGridSizeX * abh * pitch).
		 *
		 * On Compute Capability >= 3.0:
		 *	It never happens if (IDX_MAX / maxGridSizeX) < memory_alignment
		 *
		 * On Compute Capability 1.x:
		 *	"Grid extension" is also required if matrix_size >= 2**24
		 */
		uintmax_t max_size = maxGridSizeX * abh * pitch;	// It may be > IDX_MAX on Compute Capability >= 3.0
		if ( ( computeCapability == 1 ) * ( max_size >= (1 << 24) ) )
			max_size = (1 << 24) - 1;


		// No "grid extension" required
		if ( (uintmax_t) matrix_size <= max_size ) {

			grid_length = (height + abh - 1) / abh;

			#if NMFGPU_PROFILING_KERNELS
				timing_index = 0;	// Non-extended grid.
			#endif

			/* If there is not enough work for more than two blocks, uses just a single one.
			 * Such block will iteratively read data from global memory.
			 */
			if ( grid_length <= 2 ) {

				// Tries to use a block height as large as possible.

				index_t const max_block_height1 = height / REDUCE_TO_ROW__ITEMS_PER_THREAD;

				index_t const max_block_height2 = maxThreadsPerBlock / pitch;

				block_length = PREV_POWER_2( MIN( max_block_height1, max_block_height2 ) );	// A power of 2

				grid_length = 1;

				#if NMFGPU_PROFILING_KERNELS
					timing_index = 2;	// Single-block mode.
				#endif

			}

		// "Extended" grid.
		} else {

			/* Grid geometry:
			 *
			 * (grid_extension * grid_length * abh) >= height.
			 *
			 * That is,
			 *	grid_extension * grid_length >= ceil( height / abh ).
			 *
			 * So, we can use:
			 *	grid_extension == grid_length == ceil( sqrt( ceil(height/abh) ) ),
			 *
			 * but this may lead to an excessive number of idle thread blocks. That is,
			 *	(grid_extension * grid_length * abh) >> height.
			 *
			 * Instead,
			 *	grid_extension	= ceil( height / (maxGridSizeX	 * abh) ), which is << maxGridSizeY
			 *	grid_length	= ceil( height / (grid_extension * abh) ), which is <= maxGridSizeX
			 *
			 * is more accurate.
			 *
			 * Note that grid_extension <= grid_length
			 */

			// Grid "extension"
			index_t const gh = maxGridSizeX * abh;
			grid_extension = ( height + gh - 1 ) / gh;	// << maxGridSizeY

			// Grid "length"
			index_t const gw = grid_extension * abh;
			grid_length = ( height + gw - 1 ) / gw;		// <= maxGridSizeX

			#if NMFGPU_PROFILING_KERNELS
				timing_index = 1;	// Extended grid
			#endif

		} // If grid extension is required

		// ---------------------------

		#if NMFGPU_PROFILING_KERNELS
			start_cuda_timer( device_id );
		#endif

			/* d_A[ height ][ pitch ] is reduced using a grid of (grid_extension * grid_length) blocks.
			 *
			 * d_Tmp[ grid_extension*grid_length ][ pitch ] is used as a temporary storage.
			 */
			reduce_to_row( d_A, pitch, d_Tmp, block_height, grid_extension, grid_length, matrix_size, stream_AccA, d_accum_A );

			///////////////////////////////
			#if NMFGPU_DEBUG_REDUCT
				if ( ! device_id )
					check_cuda_status();
			#endif
			///////////////////////////////

		#if NMFGPU_PROFILING_KERNELS
			stop_cuda_timer_cnt( &reduce_timing[ timing_index ], matrix_size, 1, device_id );
		#endif

		// ---------------------------

		/* On Compute Capability < 1.2, a second call is required to finish the sum in
		 * d_Tmp[ grid_extension*grid_length ][ pitch ].
		 * Such call is performed in "single-block" mode.
		 */
		if ( (computeCapability == 1) * (computeCapability_minor < 2) * (grid_length > 1) ) {

			///////////////////////////////
			#if NMFGPU_DEBUG_REDUCT
				// Resulting d_Tmp from previous stage:
				if ( ! device_id ) {
					printf("\n---Resulting d_Tmp (height=%" PRI_IDX ",width=%" PRI_IDX ",pitch=%" PRI_IDX
						",block_height=%" PRI_IDX ",grid_extension=%" PRI_IDX ",grid_length=%" PRI_IDX "):---\n",
						height, width, pitch, block_height, grid_extension, grid_length );
					check_cuda_status();
					show_device_matrix( d_Tmp, (grid_extension * grid_length), width, pitch, false, NULL );
				}
			#endif
			///////////////////////////////

			// ---------------------------

			#if NMFGPU_PROFILING_KERNELS
				start_cuda_timer( device_id );
			#endif

				/* d_Tmp[ grid_extension*grid_length ][ pitch ] is reduced with a single block.
				 * No temporary storage is required.
				 */
				index_t const Tmp_size = grid_extension * grid_length;
				reduce_to_row( d_Tmp, pitch, NULL, block_height, 1, 1, Tmp_size, stream_AccA, d_accum_A );

				///////////////////////////////
				#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG
					if ( ! device_id )
						check_cuda_status();
				#endif
				///////////////////////////////

			#if NMFGPU_PROFILING_KERNELS
				stop_cuda_timer_cnt( &reduce_timing[2], matrix_size, 1, device_id );
			#endif

		} // If a second call is required.

	} else { // (height == 1)

		/* Just copies d_A[] to d_accum_A[]. */

		#if NMFGPU_PROFILING_KERNELS
			start_cuda_timer( device_id );
		#endif

			#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG
				cudaError_t cuda_status =
			#endif

				cudaMemcpyAsync( d_accum_A, d_A, pitch * sizeof(real), cudaMemcpyDeviceToDevice, stream_AccA );

			/* Same code using CUBLAS:
			 *	cublas_status = cublasSetStream( cublas_handle, stream_AccA );
			 *	cublas_status = CUBLAS_R_COPY( cublas_handle, pitch, d_A, 1, d_accum_A, 1 );
			 */

			///////////////////////////////
			#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG
				check_cuda_status_st( cuda_status );
			#endif
			///////////////////////////////

		#if NMFGPU_PROFILING_KERNELS
			stop_cuda_timer_cnt( &reduce_timing[3], pitch, 1, device_id );
		#endif

	} // if ( height > 1 )


	///////////////////////////////
	#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG
		printf("\n--- [GPU%" PRI_IDX "] Resulting accumulator (length=%" PRI_IDX ",pitch=%" PRI_IDX ") for matrix %s: ---\n",
			device_id, width, pitch, matrix_name );
		check_cuda_status();
		show_device_matrix( d_accum_A, 1, width, pitch, false, NULL );
	#endif
	///////////////////////////////

	// ------------------------------------

	// Records the previous operation on stream_AccA as 'event_AccA'

	#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG
		cudaError_t cuda_status =
	#endif

		cudaEventRecord( event_AccA, stream_AccA );

		///////////////////////////////
		#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG
			if ( cuda_status != cudaSuccess ) {
				fflush(stdout);
				fprintf(stderr, "\n[GPU%" PRI_IDX "] Error recording CUDA event: %s\nError in matrix2row(%s, width=%" PRI_IDX
					", pitch=%" PRI_IDX ").\n", device_id, cudaGetErrorString(cuda_status), matrix_name, width, pitch );
			}
		#endif
		///////////////////////////////

} // matrix_to_row

///////////////////////////////////////////////////////////////////////////////

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
		bool div_operand, cudaStream_t stream_A, cudaEvent_t *__restrict__ event_B, timing_data_t *__restrict__ td )
{

	#if NMFGPU_PROFILING_KERNELS
		index_t timing_index = 0;
	#endif

	index_t const matrix_size = pitch * height;

	// ------------------------------------------

	/* Uses 1-D blocks.
	 *
	 * Each block processes up to <DIV_SUB__ITEMS_PER_THREAD> times its size
	 * ('DIV_SUB__ITEMS_PER_THREAD' is a constant defined in "GPU_kernels.h").
	 */

	index_t const block_size = threadsPerBlock;

	index_t const act_bs = block_size * DIV_SUB__ITEMS_PER_THREAD;		// "Active" block size


	// Grid dimensions
	index_t grid_length = 1;	// <= maxGridSizeX
	index_t grid_extension = 1;	// <= maxGridSizeY, and <= grid_length

	/* "grid_extension" is required only if matrix_size > (maxGridSizeX * act_bs).
	 *
	 * On Compute Capability >= 3.0:
	 *	It never happens if (IDX_MAX / maxGridSizeX) < memory_alignment
	 *
	 * On Compute Capability 1.x:
	 *	"Grid extension" is also required if matrix_size >= 2**24
	 */
	uintmax_t max_size = maxGridSizeX * act_bs;	// It may be > IDX_MAX on Compute Capability >= 3.0
	if ( ( computeCapability == 1 ) * ( max_size >= (1 << 24) ) )
		max_size = (1 << 24) - 1;


	// No "grid extension" required
	if ( (uintmax_t) matrix_size <= max_size ) {

		grid_length = (matrix_size + act_bs - 1) / act_bs;

		#if NMFGPU_PROFILING_KERNELS
			timing_index = 0;
		#endif

	// "Extended" grid.
	} else {

		/* Grid geometry:
		 *
		 * (grid_extension * grid_length * act_bs) >= matrix_size.
		 *
		 * That is,
		 *	grid_extension * grid_length >= ceil( matrix_size / act_bs ).
		 *
		 * So, we can use:
		 *	grid_extension == grid_length == ceil( sqrt( ceil(matrix_size/act_bs) ) ),
		 *
		 * but this may lead to an excessive number of idle thread blocks. That is,
		 *	(grid_extension * grid_length * act_bs) >> matrix_size.
		 *
		 * Instead,
		 *	grid_extension	= ceil( matrix_size / (maxGridSizeX   * act_bs) ), which is << maxGridSizeY
		 *	grid_length	= ceil( matrix_size / (grid_extension * act_bs) ), which is <= maxGridSizeX
		 *
		 * is more accurate.
		 *
		 * Note that grid_extension <= grid_length
		 */

		// Grid "extension"
		index_t const gh = maxGridSizeX * abh;
		grid_extension = ( height + gh - 1 ) / gh;	// << maxGridSizeY

		// Grid "length"
		index_t const gw = grid_extension * abh;
		grid_length = ( height + gw - 1 ) / gw;		// <= maxGridSizeX

		#if NMFGPU_PROFILING_KERNELS
			timing_index = 1;
		#endif

	} // If grid extension is required

	// ------------------------------------------

	// Delays kernel launch until d_B[] is ready.

	if ( event_B ) {

		#if NMFGPU_DEBUG
			cudaError_t cuda_status =
		#endif

			cudaStreamWaitEvent( stream_A, *event_B, 0 );

		///////////////////////////////
		#if NMFGPU_DEBUG
			if ( cuda_status != cudaSuccess ) {
				fflush(stdout);
				fprintf(stderr,"\n[GPU%" PRI_IDX "] cudaStreamWaitEvent: %s\nError in matrix_div_sub(%s %s %s, height=%" PRI_IDX
					", pitch=%" PRI_IDX ").\n", device_id, cudaGetErrorString(cuda_status), matrix_name_A,
					( div_operand ? "./" : "-" ), matrix_name_B, height, pitch );
			}
		#endif
		///////////////////////////////
	}

	// ------------------------------------------

	// Launches the kernel.

	#if NMFGPU_PROFILING_KERNELS
		start_cuda_timer( device_id );
	#endif

		host_div( d_A, d_B, matrix_size, block_size, grid_extension, grid_length, div_operand, stream_A );

		///////////////////////////////
		#if NMFGPU_DEBUG
			printf("\n--- [GPU%" PRI_IDX "] Resulting %s = %s %s %s (height=%" PRI_IDX ", width=%" PRI_IDX ", pitch=%" PRI_IDX
				",block_size=%" PRI_IDX ",grid_extension=%" PRI_IDX ",grid_length=%" PRI_IDX "): ---\n", device_id,
				matrix_name_A, matrix_name_B, ( div_operand ? "./" : "-" ), matrix_name_A, height, width, pitch, block_size,
				grid_extension, grid_length );
			check_cuda_status();
			show_device_matrix( d_A, height, width, pitch, false, NULL );
		#endif
		///////////////////////////////

	#if NMFGPU_PROFILING_KERNELS
		stop_cuda_timer_cnt( &td[ timing_index ], matrix_size, 1, device_id );
	#endif

	// ------------------------------------------

	// Records the operations as an event (reuses the event object).
	if ( event_B ) {

		#if NMFGPU_DEBUG
			cudaError_t cuda_status =
		#endif

			cudaEventRecord( *event_B, stream_A );

		///////////////////////////////
		#if NMFGPU_DEBUG
			if ( cuda_status != cudaSuccess ) {
				fflush(stdout);
				fprintf(stderr,"\n[GPU%" PRI_IDX "] Error recording CUDA event: %s\nError in matrix_div(%s %s %s, height=%"
						PRI_IDX ", pitch=%" PRI_IDX ").\n", device_id, cudaGetErrorString(cuda_status), matrix_name_A,
						( div_operand ? "./" : "-" ), matrix_name_B, height, pitch );
			}
		#endif
		///////////////////////////////

	}

} // matrix_div_sub

/////////////////////////////////////////////////////////////////////

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
			cudaStream_t stream_A )
{

	/* CPU code:
	 * for ( index_t i=0 ; i<height ; i++ )
	 *	for ( index_t j=0 ; j<pitch ; j++ )
	 *		d_A[i][j] = d_A[i][j] * d_Aux[i][j] / d_accum_B[j];
	 */

	// ------------------------------------------

	#if NMFGPU_PROFILING_KERNELS
		index_t timing_index = 0;
	#endif

	index_t const matrix_size = pitch * height;

	// ------------------------------------------

	/* Uses a block width equal to <pitch>.
	 *
	 * Each block processes up to <MUL_DIV__ITEMS_PER_THREAD> times its height
	 * ('MUL_DIV__ITEMS_PER_THREAD' is a constant defined in "GPU_kernels.h").
	 */

	index_t const abh = maxBlockHeight_pitch * REDUCE_TO_ROW__ITEMS_PER_THREAD;	// "Active" block height.


	// Grid dimensions
	index_t grid_length = 1;	// <= maxGridSizeX
	index_t grid_extension = 1;	// <= maxGridSizeY, and <= grid_length

	/* "grid_extension" is required only if matrix_size > (maxGridSizeX * abh * pitch).
	 *
	 * On Compute Capability >= 3.0:
	 *	It never happens if (IDX_MAX / maxGridSizeX) < memory_alignment
	 *
	 * On Compute Capability 1.x:
	 *	"Grid extension" is also required if matrix_size >= 2**24
	 */
	uintmax_t max_size = maxGridSizeX * abh * pitch;	// It may be > IDX_MAX on Compute Capability >= 3.0
	if ( ( computeCapability == 1 ) * ( max_size >= (1 << 24) ) )
		max_size = (1 << 24) - 1;


	// No "grid extension" required
	if ( (uintmax_t) matrix_size <= max_size ) {

		grid_length = (height + abh - 1) / abh;

		#if NMFGPU_PROFILING_KERNELS
			timing_index = 0;
		#endif

	// "Extended" grid.
	} else {

		/* Grid geometry:
		 *
		 * (grid_extension * grid_length * abh) >= height.
		 *
		 * That is,
		 *	grid_extension * grid_length >= ceil( height / abh ).
		 *
		 * So, we can use:
		 *	grid_extension == grid_length == ceil( sqrt( ceil(height/abh) ) ),
		 *
		 * but this may lead to an excessive number of idle thread blocks. That is,
		 *	(grid_extension * grid_length * abh) >> height.
		 *
		 * Instead,
		 *	grid_extension	= ceil( height / (maxGridSizeX	 * abh) ), which is << maxGridSizeY
		 *	grid_length	= ceil( height / (grid_extension * abh) ), which is <= maxGridSizeX
		 *
		 * is more accurate.
		 *
		 * Note that grid_extension <= grid_length
		 */

		// Grid "extension"
		index_t const gh = maxGridSizeX * abh;
		grid_extension = ( height + gh - 1 ) / gh;	// << maxGridSizeY

		// Grid "length"
		index_t const gw = grid_extension * abh;
		grid_length = ( height + gw - 1 ) / gw;		// <= maxGridSizeX

		#if NMFGPU_PROFILING_KERNELS
			timing_index = 1;
		#endif

	} // If grid extension is required

	// ---------------------------

	#if NMFGPU_PROFILING_KERNELS
		start_cuda_timer( device_id );
	#endif

		mul_div( d_A, d_Aux, d_accum_B, pitch, matrix_size, maxBlockHeight_pitch, grid_extension, grid_length, stream_AccA );

			///////////////////////////////
			#if NMFGPU_DEBUG
				printf("\n--- [GPU%" PRI_IDX "] Resulting %s = %s .* %s ./ %s (height=%" PRI_IDX ",width=%" PRI_IDX
					", pitch=%" PRI_IDX ",block_height=%" PRI_IDX ",grid_extension=%" PRI_IDX ",grid_length=%" PRI_IDX
					", transpose=%i ): ---\n", device_id, matrix_name_A, matrix_name_Aux, matrix_name_accB,
					height, pitch, maxBlockHeight_pitch, grid_extension, grid_length, transpose );
				check_cuda_status();
				show_device_matrix( d_A, height, width, pitch, transpose, NULL );
			#endif
			///////////////////////////////

	#if NMFGPU_PROFILING_KERNELS
		stop_cuda_timer_cnt( &mul_div_timing[ timing_index ], matrix_size, 1, device_id );
	#endif

} // matrix_mul_div

/////////////////////////////////////////////////////////////////////

/*
 * d_A = MAX( d_A , R_MIN )
 *
 * 'pitch' must be a multiple of 'memory_alignment', and <= maxThreadsPerBlock.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		height < maxBlockHeight_pitch * ADJUST__ITEMS_PER_THREAD * (2**24)
 *		('ADJUST__ITEMS_PER_THREAD' is a constant defined in "GPU_kernels.h")
 */
void matrix_adjust( real *__restrict__ d_A, index_t height, index_t pitch,
			#if NMFGPU_DEBUG
				index_t width, bool transpose, char const *__restrict__ const matrix_name_A,
			#endif
			cudaStream_t stream_A )
{

	#if NMFGPU_PROFILING_KERNELS
		index_t timing_index = 0;
	#endif

	index_t const matrix_size = pitch * height;

	// ------------------------------------------

	/* Uses a block width equal to <pitch>.
	 *
	 * Each block processes up to <ADJUST__ITEMS_PER_THREAD> times its height
	 * ('ADJUST__ITEMS_PER_THREAD' is a constant defined in "GPU_kernels.h").
	 */

	index_t const abh = maxBlockHeight_pitch * ADJUST__ITEMS_PER_THREAD;	// "Active" block height.


	// Grid dimensions
	index_t grid_length = 1;	// <= maxGridSizeX
	index_t grid_extension = 1;	// <= maxGridSizeY, and <= grid_length

	/* "grid_extension" is required only if matrix_size > (maxGridSizeX * abh * pitch).
	 *
	 * On Compute Capability >= 3.0:
	 *	It never happens if (IDX_MAX / maxGridSizeX) < memory_alignment
	 *
	 * On Compute Capability 1.x:
	 *	"Grid extension" is also required if matrix_size >= 2**24
	 */
	uintmax_t max_size = maxGridSizeX * abh * pitch;	// It may be > IDX_MAX on Compute Capability >= 3.0
	if ( ( computeCapability == 1 ) * ( max_size >= (1 << 24) ) )
		max_size = (1 << 24) - 1;


	// No "grid extension" required
	if ( (uintmax_t) matrix_size <= max_size ) {

		grid_length = (height + abh - 1) / abh;

		#if NMFGPU_PROFILING_KERNELS
			timing_index = 0;
		#endif

	// "Extended" grid.
	} else {

		/* Grid geometry:
		 *
		 * (grid_extension * grid_length * abh) >= height.
		 *
		 * That is,
		 *	grid_extension * grid_length >= ceil( height / abh ).
		 *
		 * So, we can use:
		 *	grid_extension == grid_length == ceil( sqrt( ceil(height/abh) ) ),
		 *
		 * but this may lead to an excessive number of idle thread blocks. That is,
		 *	(grid_extension * grid_length * abh) >> height.
		 *
		 * Instead,
		 *	grid_extension	= ceil( height / (maxGridSizeX	 * abh) ), which is << maxGridSizeY
		 *	grid_length	= ceil( height / (grid_extension * abh) ), which is <= maxGridSizeX
		 *
		 * is more accurate.
		 *
		 * Note that grid_extension <= grid_length
		 */

		// Grid "extension"
		index_t const gh = maxGridSizeX * abh;
		grid_extension = ( height + gh - 1 ) / gh;	// << maxGridSizeY

		// Grid "length"
		index_t const gw = grid_extension * abh;
		grid_length = ( height + gw - 1 ) / gw;		// <= maxGridSizeX

		#if NMFGPU_PROFILING_KERNELS
			timing_index = 1;
		#endif

	} // If grid extension is required

	// ---------------------------

	#if NMFGPU_PROFILING_KERNELS
		start_cuda_timer( device_id );
	#endif

		adjust( d_A, pitch, matrix_size, maxBlockHeight_pitch, grid_extension, grid_length, stream_AccA );

			///////////////////////////////
			#if NMFGPU_DEBUG
				printf("\n--- [GPU%" PRI_IDX "] Resulting %s = MAX( %s, R_MIN ), (height=%" PRI_IDX ",width=%" PRI_IDX
					", pitch=%" PRI_IDX ",block_height=%" PRI_IDX ",grid_extension=%" PRI_IDX ",grid_length=%" PRI_IDX
					", transpose=%i ): ---\n", device_id, matrix_name_A, matrix_name_A, height, pitch, maxBlockHeight_pitch,
					grid_extension, grid_length , transpose );
				check_cuda_status();
				show_device_matrix( d_A, height, width, pitch, transpose, NULL );
			#endif
			///////////////////////////////

	#if NMFGPU_PROFILING_KERNELS
		stop_cuda_timer_cnt( &adjust_timing[ timing_index ], matrix_size, 1, device_id );
	#endif

} // matrix_adjust

/////////////////////////////////////////////////////////////////////

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
			cudaStream_t stream_A, index_t *__restrict__ d_Idx )
{

	#if NMFGPU_PROFILING_KERNELS
		index_t timing_index = 0;
	#endif

	index_t const matrix_size = pitch * height;

	// ------------------------------------------

	/* Each thread processes up to <IDX_MAX__ITEMS_PER_THREAD> from the same row.
	 * ('IDX_MAX__ITEMS_PER_THREAD' is a constant defined in "GPU_kernels.h").
	 *
	 * (block_width * IDX_MAX__ITEMS_PER_THREAD) must NOT be < width.
	 * Therefore, block_width >= (width / IDX_MAX__ITEMS_PER_THREAD).
	 * In addition, it must be a power of 2 <= width <= pitch.
	 */

	// block_width = Next_power_2( width / IDX_MAX__ITEMS_PER_THREAD ) >= memory_alignment

	index_t block_width = ( width / IDX_MAX__ITEMS_PER_THREAD );
	block_width = PREV_POWER_2( (block_width << 1) );	// PREV_POWER_2( x*2 ) == next_power_2( x )
	block_width = MIN( block_width, memory_alignment );	// Note that memory_alignment is also a power of 2.

	index_t const block_height = threadsPerBlock_pitch / block_width;


	// Grid dimensions
	index_t grid_length = 1;	// <= maxGridSizeX
	index_t grid_extension = 1;	// <= maxGridSizeY, and <= grid_length


	/* "grid_extension" is required only if matrix_size > (maxGridSizeX * abh * pitch).
	 *
	 * On Compute Capability >= 3.0:
	 *	It never happens if (IDX_MAX / maxGridSizeX) < memory_alignment
	 *
	 * On Compute Capability 1.x:
	 *	"Grid extension" is also required if matrix_size >= 2**24
	 */
	uintmax_t max_size = maxGridSizeX * block_height * pitch;	// It may be > IDX_MAX on Compute Capability >= 3.0
	if ( ( computeCapability == 1 ) * ( max_size >= (1 << 24) ) )
		max_size = (1 << 24) - 1;


	// No "grid extension" required
	if ( (uintmax_t) matrix_size <= max_size ) {

		grid_length = (height + block_height - 1) / block_height;

		#if NMFGPU_PROFILING_KERNELS
			timing_index = 0;
		#endif

	// "Extended" grid.
	} else {

		/* Grid geometry:
		 *
		 * (grid_extension * grid_length * block_height) >= height.
		 *
		 * That is,
		 *	grid_extension * grid_length >= ceil( height / block_height ).
		 *
		 * So, we can use:
		 *	grid_extension == grid_length == ceil( sqrt( ceil(height/block_height) ) ),
		 *
		 * but this may lead to an excessive number of idle thread blocks. That is,
		 *	(grid_extension * grid_length * block_height) >> height.
		 *
		 * Instead,
		 *	grid_extension	= ceil( height / (maxGridSizeX	 * block_height) ), which is << maxGridSizeY
		 *	grid_length	= ceil( height / (grid_extension * block_height) ), which is <= maxGridSizeX
		 *
		 * is more accurate.
		 *
		 * Note that grid_extension <= grid_length
		 */

		// Grid "extension"
		index_t const gh = maxGridSizeX * block_height;
		grid_extension = ( height + gh - 1 ) / gh;	// << maxGridSizeY

		// Grid "length"
		index_t const gw = grid_extension * block_height;
		grid_length = ( height + gw - 1 ) / gw;		// <= maxGridSizeX

		#if NMFGPU_PROFILING_KERNELS
			timing_index = 1;
		#endif

	} // If grid extension is required

	// ---------------------------

	#if NMFGPU_PROFILING_KERNELS
		start_cuda_timer( device_id );
	#endif

		idx_max( d_A, width, pitch, matrix_size, block_width, block_height, grid_extension, grid_length, stream_A, d_Idx );

			///////////////////////////////
			#if NMFGPU_DEBUG
				printf("\n--- [GPU%" PRI_IDX "] Resulting %s[i] = max(%s[i][..]) (height=%" PRI_IDX ",width=%" PRI_IDX
					", pitch=%" PRI_IDX ",block_width=%" PRI_IDX ",block_height=%" PRI_IDX ",grid_extension=%" PRI_IDX
					",grid_length=%" PRI_IDX "): ---\n", device_id, matrix_name_Idx, matrix_name_A, height, width, pitch,
					block_width, block_height, grid_extension, grid_length, transpose );
				check_cuda_status();
				show_device_matrix( d_A, height, width, pitch, true, NULL );
			#endif
			///////////////////////////////

	#if NMFGPU_PROFILING_KERNELS
		stop_cuda_timer_cnt( &idx_max_timing[ timing_index ], matrix_size, 1, device_id );
	#endif

} // matrix_idx_max

/////////////////////////////////////////////////////////////////////

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
			timing_data_t *__restrict__ const upload_timing )
{

	#if NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf("\nUploading Matrix %s to %s (no event, height=%" PRI_IDX ",width=%" PRI_IDX ",pitch=%" PRI_IDX
				",transpose: %i)\n", matrix_name_A, matrix_name_dA, height, width, pitch, transpose );
	#endif

	// ----------------------------------

	// Stream for this operation.
	cudaStream_t stream_A = matrix_stream;

	#if NMFGPU_SYNC_TRANSF
		// Synchronous data transfer: Waits until all previous operations have finished.
		check_cuda_status();
	#endif

	// ----------------------------------

	// Starts the transfer...

	size_t const nitems = height * pitch;

	#if NMFGPU_PROFILING_TRANSF
	start_cuda_timer( device_id );
	#endif

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF
		cudaError_t cuda_status =
		#endif

			cudaMemcpyAsync( d_A, A, nitems * sizeof(real), cudaMemcpyHostToDevice, stream_A );

			/* Same code using CUBLAS:
			 *	cublasStatus cublas_status =
			 *		cublasSetVectorAsync( nitems, sizeof(real), A, 1, d_A, 1, stream_A );
			 */

			///////////////////////////////
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
				printf("\n--- [GPU%" PRI_IDX "] Uploaded matrix %s to %s (no event, height=%" PRI_IDX ",width=%" PRI_IDX
					",pitch=%" PRI_IDX ",transpose: %i): ---\n", matrix_name_A, matrix_name_dA, height, width, pitch,
					transpose);
				check_cuda_status_st( cuda_status );
				show_device_matrix( d_A, height, width, pitch, transpose, NULL );
			#elif NMFGPU_SYNC_TRANSF
				check_cuda_status_st( cuda_status );
			#endif
			/////////////////////////////

	#if NMFGPU_PROFILING_TRANSF
		stop_cuda_timer_cnt( upload_timing, nitems, 1, device_id );
	#endif

	// ----------------------------------

	#if NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf("\nUploading Matrix %s to %s (no event, height=%" PRI_IDX ",width=%" PRI_IDX ",pitch=%" PRI_IDX
				",transpose: %i)... Done\n", matrix_name_A, matrix_name_dA, height, width, pitch, transpose );
	#endif

} // upload_matrix

/////////////////////////////////////////////////////////////////////

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
			timing_data_t *__restrict__ const upload_timing )
{

	#if NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf("\nUploading Matrix %s to %s (with event, height=%" PRI_IDX ",width=%" PRI_IDX ",pitch=%" PRI_IDX
				",transpose: %i)\n", matrix_name_A, matrix_name_dA, height, width, pitch, transpose );
	#endif

	#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF
		cudaError_t cuda_status = cudaSuccess;
	#endif

	// ----------------------------------

	// Event and stream used for this operation.
	cudaEvent_t event_A = transfer_event;
	cudaStream_t stream_A = matrix_stream;

	#if NMFGPU_SYNC_TRANSF
		// Synchronous data transfer: Waits until all previous operations have finished.
		check_cuda_status();
	#endif

	// ----------------------------------

	// Performs the transfer...

	size_t const nitems = height * pitch;

	#if NMFGPU_PROFILING_TRANSF
	start_cuda_timer( device_id );
	#endif

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF
			cuda_status =
		#endif

			cudaMemcpyAsync( d_A, A, nitems * sizeof(real), cudaMemcpyHostToDevice, stream_A );

			/* Same code using CUBLAS:
			 *	cublasStatus cublas_status =
			 *		cublasSetVectorAsync( nitems, sizeof(real), A, 1, d_A, 1, stream_A );
			 */

			///////////////////////////////
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
				printf("\n--- [GPU%" PRI_IDX "] Uploaded matrix %s to %s (with event, height=%" PRI_IDX ",width=%" PRI_IDX
					",pitch=%" PRI_IDX ",transpose: %i): ---\n", matrix_name_A, matrix_name_dA, height, width, pitch,
					transpose);
				check_cuda_status_st( cuda_status );
				show_device_matrix( d_A, height, width, pitch, transpose, NULL );
			#elif NMFGPU_SYNC_TRANSF
				check_cuda_status_st( cuda_status );
			#endif
			/////////////////////////////

	#if NMFGPU_PROFILING_TRANSF
		stop_cuda_timer_cnt( upload_timing, nitems, 1, device_id );
	#endif

	// ----------------------------------

	// Records the previous operation on stream_A as 'event_A'

	#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
		cuda_status =
	#endif

		cudaEventRecord( event_A, stream_A );

		///////////////////////////////
		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
			if ( cuda_status != cudaSuccess ) {
				fflush(stdout);
				fprintf(stderr, "\n[GPU%" PRI_IDX "] Error recording CUDA event: %s\nError in upload_matrix_event(%s to %s, "
						"height=%" PRI_IDX ",width=%" PRI_IDX ",pitch=%" PRI_IDX ",transpose: %i).\n", device_id,
						cudaGetErrorString(cuda_status), matrix_name_A, matrix_name_dA, height, width, pitch,
						transpose );
			}
		#endif
		///////////////////////////////

	// ----------------------------------

	#if NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf("\nUploading Matrix %s to %s (with event, height=%" PRI_IDX ",width=%" PRI_IDX ",pitch=%" PRI_IDX
				",transpose: %i)...Done.\n", matrix_name_A, matrix_name_dA, height, width, pitch, transpose );
	#endif

} // upload_matrix_event

/////////////////////////////////////////////////////////////////////

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
			timing_data_t *__restrict__ upload_timing )
{

	#if NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf("\nUploading Matrix %s to %s (partial, height=%" PRI_IDX ",pitch=%" PRI_IDX ",offset=%" PRI_IDX ",block_width=%"
				PRI_IDX ", block_pitch=%" PRI_IDX ")\n", matrix_name_A, matrix_name_dA, height, pitch, offset, block_width,
				block_pitch);
	#endif


	#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF
		cudaError_t cuda_status = cudaSuccess;
	#endif

	// ----------------------------------

	#if NMFGPU_SYNC_TRANSF
		// Synchronous data transfer: Waits until all previous operations have finished.
		check_cuda_status();
	#endif

	// ----------------------------------

	// Delays the transfer until the event has completed all previous operations.

	#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF
		cuda_status =
	#endif

		cudaStreamWaitEvent( stream_A, event_A, 0 );

		///////////////////////////////
		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
			if ( cuda_status != cudaSuccess ) {
				fflush(stdout);
				fprintf(stderr, "\n[GPU%" PRI_IDX "] Error setting CUDA event to wait for (cudaStreamWaitEvent): %s\nError "
						"in upload_matrix_partial(%s to %s, partial, height=%" PRI_IDX ",pitch=%" PRI_IDX ",offset=%"
						PRI_IDX ",block_width=%" PRI_IDX ", block_pitch=%" PRI_IDX ").\n", device_id,
						cudaGetErrorString(cuda_status), matrix_name_A, matrix_name_dA, height, pitch, offset,
						block_width, block_pitch);
			}
		#endif
		///////////////////////////////

	// ----------------------------------

	// Starts the transfer...

	#if NMFGPU_PROFILING_TRANSF
	start_cuda_timer( device_id );
	#endif

	if ( ( block_pitch < pitch ) + ( offset > 0 ) ) {

		/*
		 * It must be transferred as a 2D matrix.
		 */

		/* If necessary, adjusts the width to avoid an out-of-bound failure in CPU memory,
		 * but then, such width will NOT be a multiple of 'memory_alignment', resulting in a slower transfer.
		 */
		index_t const width = ( ( (offset + block_pitch) <= pitch ) ? block_pitch : (pitch - (offset + block_pitch)) );

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF
			cuda_status =
		#endif

			cudaMemcpy2DAsync( d_Matrix, block_pitch * sizeof(real), p_A, pitch * sizeof(real),
						width * sizeof(real), height, cudaMemcpyHostToDevice, stream_A );

			/* Same code using CUBLAS:
			 *	cublasStatus cublas_status =
			 *		cublasSetMatrixAsync( width, height, sizeof(real), p_A, pitch, d_A, block_pitch, stream_A );
			 */


	// ( block_pitch == pitch ) && ( offset == 0 )
	} else {

		/*
		 * It can be transferred as a row vector.
		 */

		size_t const nitems = height * pitch;

		#if NMFGPU_PROFILING_TRANSF
		start_cuda_timer( device_id );
		#endif

			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF
				cuda_status =
			#endif

				cudaMemcpyAsync( d_A, A, nitems * sizeof(real), cudaMemcpyHostToDevice, stream_A );

				/* Same code using CUBLAS:
				 *	cublasStatus cublas_status =
				 *		cublasSetVectorAsync( nitems, sizeof(real), A, 1, d_A, 1, stream_A );
				 */


	} // if ( ( block_pitch < pitch ) || ( offset > 0 ) )


		///////////////////////////////
		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
			printf("\n--- [GPU%" PRI_IDX "] Uploaded Matrix %s to %s (partial, height=%" PRI_IDX ",pitch=%" PRI_IDX ",offset=%"
				PRI_IDX ",block_width=%" PRI_IDX ", block_pitch=%" PRI_IDX ")\n", device_id,  matrix_name_A, matrix_name_dA,
				height, pitch, offset, block_width, block_pitch );
			check_cuda_status_st( cuda_status );
			show_device_matrix( d_A, height, block_width, block_pitch, false, NULL );
		#elif NMFGPU_SYNC_TRANSF
			check_cuda_status_st( cuda_status );
		#endif
		///////////////////////////////


	#if NMFGPU_PROFILING_TRANSF
	stop_cuda_timer_cnt( upload_timing, nitems, 1, device_id );
	#endif


	// ----------------------------------

	// Records this operation using the same event object.

	#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
		cuda_status =
	#endif

		cudaEventRecord( event_A, stream_A );

		///////////////////////////////
		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
			if ( cuda_status != cudaSuccess ) {
				fflush(stdout);
				fprintf(stderr, "\n[GPU%" PRI_IDX "] Error recording CUDA: %s\nError in upload_matrix_partial(%s to %s, "
						"partial, height=%" PRI_IDX ",pitch=%" PRI_IDX ",offset=%" PRI_IDX ",block_width=%" PRI_IDX
						", block_pitch=%" PRI_IDX ").\n", device_id, cudaGetErrorString(cuda_status), matrix_name_A,
						matrix_name_dA, height, pitch, offset, block_width, block_pitch);
			}
		#endif
		///////////////////////////////


	// ----------------------------------

	#if NMFGPU_VERBOSE_2
	if (! device_id)
		if ( ! device_id )
			printf("\nUploading Matrix %s to %s (partial, height=%" PRI_IDX ",pitch=%" PRI_IDX ",offset=%" PRI_IDX ",block_width=%"
				PRI_IDX ", block_pitch=%" PRI_IDX ")...Done.\n", matrix_name_A, matrix_name_dA, height, pitch, offset,
				block_width, block_pitch);
	#endif

} // upload_matrix_partial

/////////////////////////////////////////////////////////////////////

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
			timing_data_t *__restrict__ const download_timing )
{

	#if NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf("\nDownloading Matrix %s to %s (no event, height=%" PRI_IDX ",width=%" PRI_IDX ",pitch=%" PRI_IDX
				",transpose: %i)\n", matrix_name_dA, matrix_name_A, height, width, pitch, transpose );
	#endif

	// ----------------------------------

	// Stream for this operation.
	cudaStream_t stream_A = matrix_stream;

	#if NMFGPU_SYNC_TRANSF
		// Synchronous data transfer: Waits until all previous operations have finished.
		check_cuda_status();
	#endif

	// ----------------------------------

	// Starts the transfer...

	size_t const nitems = height * pitch;

	#if NMFGPU_PROFILING_TRANSF
	start_cuda_timer( device_id );
	#endif

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF
		cudaError_t cuda_status =
		#endif

			cudaMemcpyAsync( A, d_A, size * sizeof(real), cudaMemcpyDeviceToHost, stream_A );

			/* Same code using CUBLAS:
			 *	cublasStatus cublas_status =
			 *		cublasSetVectorAsync( nitems, sizeof(real), d_A, 1, A, 1, stream_A );
			 */

			///////////////////////////////
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
				printf("\n--- [GPU%" PRI_IDX "] Downloaded matrix %s to %s (no event, height=%" PRI_IDX ",width=%" PRI_IDX
					",pitch=%" PRI_IDX ",transpose: %i): ---\n", matrix_name_dA, matrix_name_A, height, width, pitch,
					transpose);
				check_cuda_status_st( cuda_status );
				matrix_show( A, height, width, pitch, transpose, NULL );
			#elif NMFGPU_SYNC_TRANSF
				check_cuda_status_st( cuda_status );
			#endif
			/////////////////////////////

	#if NMFGPU_PROFILING_TRANSF
	stop_cuda_timer_cnt( download_timing, nitems, 1, device_id );
	#endif

	// ----------------------------------

	#if NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf("\nDownloading Matrix %s to %s (no event, height=%" PRI_IDX ",width=%" PRI_IDX ",pitch=%" PRI_IDX
				",transpose: %i)...Done.\n", matrix_name_dA, matrix_name_A, height, width, pitch, transpose );
	#endif

} // download_matrix

////////////////////////////////////////////////////////////////////////

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
			timing_data_t *__restrict__ const download_timing )
{

	#if NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf("\nDownloading integer Matrix %s to %s (no event, height=%" PRI_IDX ",width=%" PRI_IDX ",pitch=%" PRI_IDX
				",transpose: %i)\n", matrix_name_dA, matrix_name_A, height, width, pitch, transpose );
	#endif

	// ----------------------------------

	// Stream for this operation.
	cudaStream_t stream_A = matrix_stream;

	#if NMFGPU_SYNC_TRANSF
		// Synchronous data transfer: Waits until all previous operations have finished.
		check_cuda_status();
	#endif

	// ----------------------------------

	// Starts the transfer...

	size_t const nitems = height * pitch;

	#if NMFGPU_PROFILING_TRANSF
	start_cuda_timer( device_id );
	#endif

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF
		cudaError_t cuda_status =
		#endif

			cudaMemcpyAsync( A, d_A, size * sizeof(index_t), cudaMemcpyDeviceToHost, stream_A );

			/* Same code using CUBLAS:
			 *	cublasStatus cublas_status =
			 *		cublasSetVectorAsync( nitems, sizeof(real), d_A, 1, A, 1, stream_A );
			 */

			///////////////////////////////
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
				printf("\n--- [GPU%" PRI_IDX "] Downloaded integer matrix %s to %s (no event, height=%" PRI_IDX ",width=%"
					PRI_IDX ",pitch=%" PRI_IDX ",transpose: %i): ---\n", matrix_name_dA, matrix_name_A, height, width,
					pitch, transpose);
				check_cuda_status_st( cuda_status );
				matrix_int_show( A, height, width, pitch, transpose, NULL );
			#elif NMFGPU_SYNC_TRANSF
				check_cuda_status_st( cuda_status );
			#endif
			/////////////////////////////

	#if NMFGPU_PROFILING_TRANSF
	stop_cuda_timer_cnt( download_timing, nitems, 1, device_id );
	#endif

	// ----------------------------------

	#if NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf("\nDownloading integer Matrix %s to %s (no event, height=%" PRI_IDX ",width=%" PRI_IDX ",pitch=%" PRI_IDX
				",transpose: %i)...Done.\n", matrix_name_dA, matrix_name_A, height, width, pitch, transpose );
	#endif

} // download_matrix_int

////////////////////////////////////////////////////////////////////////
