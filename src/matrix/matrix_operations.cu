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
 * matrix_operations.cu
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
 *		NMFGPU_VERBOSE_2: Shows the parameters in some routine calls.
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

// Required by <inttypes.h>
#ifndef __STDC_FORMAT_MACROS
	#define __STDC_FORMAT_MACROS (1)
#endif
#ifndef __STDC_CONSTANT_MACROS
	#define __STDC_CONSTANT_MACROS (1)
#endif

#include "matrix/matrix_operations.cuh"
#include "GPU_setup.cuh"
#include "GPU_kernels.cuh"
#include "matrix/matrix_io.h"
#include "common.h"

#include <curand.h>

#include <limits.h>	/* [U]INT_MAX */
/* WARNING:
 *	There is a known bug on Intel C/C++ Compiler (icc), version 10,
 *	when <limits.h> is included.
 *	A workaround may be to use the following definitions, instead:
 *
 * #ifndef INT_MAX
 *	#define INT_MAX (2147483647)
 * #endif
 * #ifndef UINT_MAX
 *	#define UINT_MAX (4294967295U)
 * #endif
 */
#include <inttypes.h>	/* PRIuFAST64, UINT64_C, uint_fast64_t */
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>

////////////////////////////////////////////////
////////////////////////////////////////////////

/* Data types */

// Kernel parameters.
struct kernel_params {
	index_t block_magnitude;	// Block magnitude (height or size).
	index_t abm;			// "Active" block magnitude (i.e., block_magnitude * <items_per_thread>)
	uint_fast64_t max_magnitude;	// Maximum magnitude value with <number_of_blocks> (it may be > IDX_MAX on CC >= 3.0).
};

// ---------------------------------------------
// ---------------------------------------------

/* Macro Constants */

// Generates uniformly distributed real data
#undef CURAND_GENERATE_UNIFORM_REAL
#if NMFGPU_SINGLE_PREC
	#define CURAND_GENERATE_UNIFORM_REAL curandGenerateUniform
#else
	#define CURAND_GENERATE_UNIFORM_REAL curandGenerateUniformDouble
#endif

// ---------------------------------------------
// ---------------------------------------------

/* "Private" Global variables */

// Kernel parameters.
static struct kernel_params matrix_to_row_kparams;
static struct kernel_params matrix_div_sub_kparams;
static struct kernel_params matrix_mul_div_kparams;
static struct kernel_params matrix_adjust_kparams;
static struct kernel_params matrix_idx_max_kparams;

// ---------------------------------------------

// Information and/or error messages shown by all processes.
#if NMFGPU_DEBUG || NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
	static bool const dbg_shown_by_all = true;	// Information messages in debug mode.
	static bool const verb_shown_by_all = false;	// Information messages in verbose mode.
#endif

static bool const sys_error_shown_by_all = true;	// System error messages.

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Returns a new struct kernel_params, with the given values.
 */
static struct kernel_params new_kernel_params( index_t bmag, index_t items_per_thread, uint_fast64_t num_blocks )
{

	struct kernel_params kparams;

	kparams.block_magnitude = bmag;
	kparams.abm = bmag * items_per_thread;
	kparams.max_magnitude = num_blocks * (uint_fast64_t) kparams.abm; // It may be greater than IDX_MAX on Compute Capability >= 3.0

	return kparams;

} // new_kernel_params

////////////////////////////////////////////////

/*
 * Returns a new width-based struct kernel_params, with the given values.
 */
static struct kernel_params new_kernel_params_width( index_t pitch, index_t items_per_thread, uint_fast64_t num_blocks )
{

	/* Block Dimensions:
	 *	Each thread processes <items_per_thread> from the same row.
	 *
	 * (block_width * items_per_thread) must be >= width
	 *
	 * That is (using pitch, instead),
	 *	block_width >= (pitch / items_per_thread).
	 *
	 * In addition, block_width must be a power of 2, and >= memory_alignment
	 */

	/* If pitch <= (memory_alignment * items_per_thread),
	 * then memory_alignment (which is a power of 2) is the required block_width
	 */
	index_t block_width = memory_alignment;
	if ( pitch > (memory_alignment * items_per_thread) ) {

		block_width = (pitch / items_per_thread) + ((pitch % items_per_thread) != 0);	// <= maxThreadsPerBlock

		block_width = next_power_2( block_width );		// < 2 * ceil(pitch/items_per_thread)

		block_width = MIN( block_width, maxThreadsPerBlock );	// maxThreadsPerBlock is also a power of 2.
	}

	index_t block_height = threadsPerBlock / block_width;
	if ( block_height <= 1 ) {

		/* With maxThreadsPerBlock, the block height may be increased to two or three rows,
		 * but then there will be just one or two blocks per multiprocessor. Instead, the
		 * block size is increased to (maxThreadsPerBlock + threadsPerBlock) / 2, which
		 * allows up to two rows, but two or three blocks per multiprocessor.
		 *
		 * In addition, the block_height must be a multiple of <pitch>.
		 */
		block_height = (maxThreadsPerBlock + threadsPerBlock) / (2 * pitch);	// (possibly truncated)
	}

	// -------------------------------------

	struct kernel_params kparams_width;

	kparams_width.block_magnitude = block_height;

	// HACK: Uses the "abm" field to store "block_width".
	kparams_width.abm = block_width;

	// It may be greater than IDX_MAX on Compute Capability >= 3.0
	kparams_width.max_magnitude = num_blocks * (uint_fast64_t) block_height;

	return kparams_width;

} // new_kernel_params_width

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Returns the maximum height supported by this GPU device,
 * for the given pitch, and regardless of the available memory.
 */
index_t gpu_max_height( index_t pitch )
{

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "gpu_max_height( pitch = %" PRI_IDX ", ComputeCapability: %" PRI_IDX ".%" PRI_IDX " )\n",
				pitch, computeCapability, computeCapability_minor );
	#endif

	// ---------------------------------

	// The "height" in a kernel may be any of both input-matrix dimensions.
	index_t max_height = MIN( matrix_max_non_padded_dim, matrix_max_pitch );

	// ---------------------------------

	// maxtrix_to_row():
	{

		index_t const block_height = prev_power_2( maxBlockHeight_pitch );
		index_t const items_per_thread = REDUCE_TO_ROW__ITEMS_PER_THREAD;
		unsigned int const maxBlocksPerGrid = ( (computeCapability == 1) ? (1 << 24) : UINT_MAX ); // << (maxGridSizeX * maxGridSizeY)

		// Computes the maximum dimension.
		struct kernel_params kparams = new_kernel_params( block_height, items_per_thread, maxBlocksPerGrid );
		uint_fast64_t const l_max_height = kparams.max_magnitude;

		#if NMFGPU_DEBUG
		///////////////////////////////
			print_message( verb_shown_by_all, "\tMatrix to row: %" PRIuFAST64 "\n", l_max_height );
		//////////////////////////////
		#endif

		// Final value
		max_height = MIN( l_max_height, ((uint_fast64_t) max_height) );

	} // matrix_to_row()

	// ---------------------------------

	// matrix_mul_div() and matrix_adjust():
	{

		index_t const block_height = maxBlockHeight_pitch;
		index_t const items_per_thread = MIN( MUL_DIV__ITEMS_PER_THREAD, ADJUST__ITEMS_PER_THREAD );
		uint_fast64_t const maxBlocksPerGrid = (uint_fast64_t) maxGridSizeX * (uint_fast64_t) maxGridSizeY;

		struct kernel_params kparams = new_kernel_params( block_height, items_per_thread, maxBlocksPerGrid );
		uint_fast64_t const l_max_height = kparams.max_magnitude;

		#if NMFGPU_DEBUG
		///////////////////////////////
			print_message( verb_shown_by_all, "\tMatrix mul_div and adjust: %" PRIuFAST64 "\n", l_max_height );
		//////////////////////////////
		#endif

		// Final value
		max_height = MIN( l_max_height, ((uint_fast64_t) max_height) );

	} // matrix_mul_div() and matrix_adjust()

	// ---------------------------------

	// matrix_idx_max(): On this one, block_width might be different than <pitch>.
	{

		index_t const items_per_thread = IDX_MAX__ITEMS_PER_THREAD;
		uint_fast64_t const maxBlocksPerGrid = (uint_fast64_t) maxGridSizeX * (uint_fast64_t) maxGridSizeY;

		struct kernel_params kparams_width = new_kernel_params_width( pitch, items_per_thread, maxBlocksPerGrid );
		uint_fast64_t const l_max_height = kparams_width.max_magnitude;

		#if NMFGPU_DEBUG
		///////////////////////////////
			print_message( verb_shown_by_all, "\tMatrix idx_max: %" PRIuFAST64 "\n", l_max_height );
		//////////////////////////////
		#endif

		// Final value
		max_height = MIN( l_max_height, ((uint_fast64_t) max_height) );

	} // matrix_idx_max()

	// ---------------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "gpu_max_height( pitch = %" PRI_IDX ", ComputeCapability: %" PRI_IDX ".%" PRI_IDX " ): %"
				PRI_IDX "\n", pitch, computeCapability, computeCapability_minor, max_height );
	#endif

	return max_height;

} // gpu_max_height

////////////////////////////////////////////////

/*
 * Returns the maximum number of items in a matrix supported by this GPU device,
 * regardless of the available memory.
 */
size_t gpu_max_nitems( void )
{

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "gpu_max_nitems( ComputeCapability: %" PRI_IDX ".%" PRI_IDX " )\n",
				computeCapability, computeCapability_minor );
	#endif

	// ---------------------------------

	// Starting limit.
	size_t max_items = matrix_max_num_items;

	// ---------------------------------

	// Checks the limit for each related kernel, and updates the previous value if necessary.

	// matrix_div_sub
	{

		index_t const block_size = threadsPerBlock;
		index_t const items_per_thread = DIV_SUB__ITEMS_PER_THREAD;
		uint_fast64_t const maxBlocksPerGrid = (uint_fast64_t) maxGridSizeX * (uint_fast64_t) maxGridSizeY;

		struct kernel_params kparams = new_kernel_params( block_size, items_per_thread, maxBlocksPerGrid );
		uint_fast64_t const l_max_items = kparams.max_magnitude;

		#if NMFGPU_DEBUG
		///////////////////////////////
			print_message( verb_shown_by_all, "\tMatrix div_sub: %" PRIuFAST64 "\n", l_max_items );
		//////////////////////////////
		#endif

		// Final value
		max_items = MIN( l_max_items, ((uint_fast64_t) max_items) );

	} // matrix_div_sub

	// ---------------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "gpu_max_nitems( ComputeCapability: %" PRI_IDX ".%" PRI_IDX " ): %zu\n",
				computeCapability, computeCapability_minor, max_items );
	#endif

	return max_items;

} // gpu_max_nitems

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Initializes all kernel parameters.
 */
void init_kernel_params( index_t pitch )
{

	// NOTE: The following "*__ITEMS_PER_THREAD" constants are defined in "GPU_kernels.cuh"

	/* matrix_to_row: Uses a block width equal to <pitch>. Each block reduces <REDUCE_TO_ROW__ITEMS_PER_THREAD> times its height
	 * (see the reduction example in CUDA Samples for details). Block height must be a power of 2.
	 */
	matrix_to_row_kparams = new_kernel_params( prev_power_2( maxBlockHeight_pitch ), REDUCE_TO_ROW__ITEMS_PER_THREAD, maxGridSizeX );


	// matrix_div_sub: Uses 1-D blocks. Each block processes up to <DIV_SUB__ITEMS_PER_THREAD> times its size.
	matrix_div_sub_kparams = new_kernel_params ( threadsPerBlock, DIV_SUB__ITEMS_PER_THREAD, maxGridSizeX );


	// matrix_mul_div: Uses a block width equal to <pitch>. Each block processes up to <MUL_DIV__ITEMS_PER_THREAD> times its height.
	matrix_mul_div_kparams = new_kernel_params( maxBlockHeight_pitch, REDUCE_TO_ROW__ITEMS_PER_THREAD, maxGridSizeX );


	// matrix_adjust: Uses a block width equal to <pitch>. Each block processes up to <ADJUST__ITEMS_PER_THREAD> times its height.
	matrix_adjust_kparams = new_kernel_params( maxBlockHeight_pitch, ADJUST__ITEMS_PER_THREAD, maxGridSizeX );


	/* matrix_idx_max: Each thread processes <IDX_MAX__ITEMS_PER_THREAD> from the same row.
	 * (block_width * IDX_MAX__ITEMS_PER_THREAD) must be >= width
	 * In addition, block_width must be a power of 2, and >= memory_alignment
	 */
	matrix_idx_max_kparams = new_kernel_params_width( pitch, IDX_MAX__ITEMS_PER_THREAD, maxGridSizeX );

} // init_kernel_params

////////////////////////////////////////////////

/*
 * Setups the given matrix operation.
 *
 * That is, computes grid dimensions, and timing index (on NMFGPU_PROFILING_KERNELS, only).
 */
static void setup_matrix_operation( struct kernel_params *__restrict__ p_kparams, size_t magnitude, dim3 *__restrict__ p_dimGrid,
					#if NMFGPU_PROFILING_KERNELS
						index_t *__restrict__ p_timing_index,
					#endif
					index_t *__restrict__ p_block_magnitude )
{

	#if NMFGPU_PROFILING_KERNELS
		index_t l_timing_index = INDEX_C( 0 );	// Non-extended grid (0), extended grid (1), ...
	#endif

	// Kernel parameters
	index_t const block_magnitude = p_kparams->block_magnitude;
	index_t const abm = p_kparams->abm;
	uint_fast64_t const max_magnitude = p_kparams->max_magnitude;	// It may be greater than IDX_MAX on Compute Capability >= 3.0

	// Grid dimensions
	index_t grid_length = 1;	// <= maxGridSizeX
	index_t grid_extension = 1;	// <= MIN( maxGridSizeY, grid_length )

	// -------------------------------------

	/* "Grid extension" is required only if (magnitude > (maxGridSizeX * abm)).
	 *
	 * Note, however, it never happens on Compute Capability >= 3.0 under certain conditions,
	 * described on each matrix operation method.
	 */

	// No "grid extension" required
	if ( (uint_fast64_t) magnitude <= max_magnitude ) {

		grid_length = (magnitude / abm) + ((magnitude % abm) != 0);

		#if NMFGPU_PROFILING_KERNELS
			l_timing_index = INDEX_C( 0 );	// Non-extended grid.
		#endif

	// "Extended" grid
	} else {

		/* Grid geometry:
		 *
		 * (grid_extension * grid_length * abm) >= magnitude.
		 *
		 * That is,
		 *	grid_extension * grid_length >= ceil( magnitude / abm ).
		 *
		 * So, we can use:
		 *	grid_extension == grid_length == ceil( sqrt( ceil(magnitude/abm) ) ),
		 *
		 * but this may lead to an excessive number of idle blocks. That is,
		 *	(grid_extension * grid_length * abm) >> magnitude.
		 *
		 * Instead,
		 *	grid_extension	= ceil( magnitude / (maxGridSizeX   * abm) ), which is << maxGridSizeY,
		 *	grid_length	= ceil( magnitude / (grid_extension * abm) ), which is <= maxGridSizeX,
		 *
		 * is more accurate.
		 *
		 * Note that grid_extension <= grid_length
		 */

		// Grid "extension"
		size_t const gh = max_magnitude;				// max_magnitude < magnitude
		grid_extension = (magnitude / gh) + ((magnitude % gh) != 0);	// << maxGridSizeY

		// Grid "length"
		size_t const gw = (size_t) grid_extension * (size_t) abm;
		grid_length = (magnitude / gw) + ((magnitude % gw) != 0);	// <= maxGridSizeX

		#if NMFGPU_PROFILING_KERNELS
			l_timing_index = INDEX_C(1);	// Extended grid
		#endif

	} // If grid extension is required

	// -------------------------------------

	// Output values

	// Final grid dimensions
	p_dimGrid->x = grid_length;
	p_dimGrid->y = grid_extension;

	#if NMFGPU_PROFILING_KERNELS
		*p_timing_index = l_timing_index;
	#endif

	*p_block_magnitude = block_magnitude;

} // setup_matrix_operation

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * SYNCHRONOUSLY downloads a matrix from the GPU to a temporary buffer,
 * whose memory is automatically allocated.
 *
 * Returns a pointer to the buffer, or NULL on error.
 */
static void *download_to_buffer( void const *__restrict__ dMatrix, index_t height, index_t pitch, size_t data_size )
{

	size_t const size_bytes = (size_t) height * (size_t) pitch * data_size;

	// Downloads the device matrix to a temporary array and shows its content.
	void const *__restrict__ const buffer = (void *) malloc( size_bytes );
	if ( ! buffer ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in download_to_buffer(): malloc()" );
		return NULL;
	}

	// SYNCHRONOUS data transfer.
	cudaError_t const cuda_status = cudaMemcpy( (void *) buffer, dMatrix, size_bytes, cudaMemcpyDeviceToHost );
	if ( cuda_status != cudaSuccess ) {
		print_error( sys_error_shown_by_all, "Error downloading DEVICE matrix (height=%" PRI_IDX ", pitch=%" PRI_IDX "): %s\n",
				height, pitch, cudaGetErrorString(cuda_status) );
		free( (void *) buffer );
		return NULL;
	}
	/* Same code using cuBLAS:
	 *	cublasStatus_t cublas_status = cublasGetVector( (size_t) height * (size_t) pitch, data_size, dMatrix, 1, buffer, 1 );
	 */

	return (void *) buffer;

} // download_to_buffer

// ---------------------------------------------

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
int show_device_matrix( void const *__restrict__ dMatrix, index_t nrows, index_t ncols, index_t pitch, bool real_data, bool transpose,
			bool all_processes, struct matrix_tags_t const *__restrict__ mt )
{

	size_t const data_size = ( real_data ? sizeof(real) : sizeof(index_t) );

	int status = EXIT_SUCCESS;

	// -----------------------------

	void const *__restrict__ const buffer = (void const *) download_to_buffer( dMatrix, nrows, pitch, data_size );
	if ( ! buffer )
		return EXIT_FAILURE;

	// -----------------------------

	status = matrix_show( buffer, nrows, ncols, pitch, real_data, transpose, all_processes, mt );

	free( (void *) buffer );

	return status;

} // show_device_matrix

////////////////////////////////////////////////

/*
 * d_A = random_value
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
 * Return EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_random( real *__restrict__ d_A, index_t height, index_t width, index_t padding,
			#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2
				bool transpose,
			#endif
			#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 || (! NMFGPU_PROFILING_GLOBAL)
				char const *__restrict__ const matrix_name,
			#endif
			cudaStream_t stream_A, cudaEvent_t *__restrict__ event_A )
{

	#if (! NMFGPU_CPU_RANDOM)

		#if NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "Setting random values to matrix '%s' (height=%" PRI_IDX ", width=%" PRI_IDX
					", padding=%" PRI_IDX ", transpose=%i)\n", matrix_name, height, width, padding, transpose );
		#endif

		// ----------------------------------

		// Sets the stream
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				curandStatus_t const curand_status =
			#endif

				curandSetStream( curand_generator, stream_A );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( curand_status != CURAND_STATUS_SUCCESS ) {
					print_error( sys_error_shown_by_all, "Error setting stream for CURAND kernel on matrix %s: %s\n",
							matrix_name, getCurandErrorString( curand_status ) );
					return EXIT_FAILURE;
				}
			#endif
		}

		// ----------------------------------

		// Generates random values.
		{
			size_t const size = (size_t) height * (size_t) padding;

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				curandStatus_t const curand_status =
			#endif

				CURAND_GENERATE_UNIFORM_REAL( curand_generator, d_A, size );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( curand_status != CURAND_STATUS_SUCCESS ) {
					print_error( sys_error_shown_by_all, "Error generating random values for matrix %s: %s\n",
							matrix_name, getCurandErrorString( curand_status ) );
					return EXIT_FAILURE;
				}
			#endif

			///////////////////////////////
			#if NMFGPU_DEBUG
			{
				print_message( dbg_shown_by_all, "--- Random values on matrix %s (height=%" PRI_IDX ", width=%" PRI_IDX
						", padding=%" PRI_IDX ", transpose=%i): ---\n", matrix_name, height, width, padding, transpose );
				int const status1 = check_cuda_status();
				bool const real_data = true;
				struct matrix_tags_t const *mt = NULL;
				int const status2 = show_device_matrix( d_A, height, width, padding, real_data, transpose, dbg_shown_by_all, mt );
				if ( (status1 != EXIT_SUCCESS) + (status2 != EXIT_SUCCESS) )
					return EXIT_FAILURE;
			}
			#endif
			/////////////////////////////
		}

		// ----------------------------------

		// Records the previous operation on stream_A as 'event_A'
		if ( event_A ) {

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				cudaError_t const cuda_status =
			#endif

				cudaEventRecord( *event_A, stream_A );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( cuda_status != cudaSuccess ) {
					print_error( sys_error_shown_by_all, "matrix_random( %s ): cudaEventRecord(): %s\n", matrix_name,
						cudaGetErrorString( cuda_status ) );
					return EXIT_FAILURE;
				}
			#endif

		}

		// ----------------------------------

		#if NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "Setting random values to matrix '%s' (height=%" PRI_IDX ", width=%" PRI_IDX
					", padding=%" PRI_IDX ", transpose=%i)... Done.\n", matrix_name, height, width, padding, transpose );
		#endif

	#endif /* ! NMFGPU_CPU_RANDOM */

	return EXIT_SUCCESS;

} //matrix_random

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
int matrix_to_row( real const *__restrict__ d_A, index_t height, index_t pitch,
			#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG
				index_t width,
			#endif
			#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				char const *__restrict__ const matrix_name,
			#endif
			real *__restrict__ d_Tmp, real *__restrict__ d_accum_A, cudaStream_t stream_AccA )
{

	///////////////////////////////
	#if NMFGPU_DEBUG_REDUCT
		print_message( verb_shown_by_all, "--- Begin of matrix_to_row(computeCapability=%" PRI_IDX ".%" PRI_IDX ", width=%" PRI_IDX
				", pitch=%" PRI_IDX ", height=%" PRI_IDX ") on %s: ---\n", computeCapability, computeCapability_minor, width,
				pitch, height, matrix_name);
	#endif
	///////////////////////////////

	// ----------------------------------------

	// Event for this operation.
	cudaEvent_t event_AccA = event_reduction;

	// ----------------------------------------

	if ( height > 1 ) {

		/* Kernel setup:
		 *
		 * Block dimensions:
		 *	Uses a block width equal to <pitch>.
		 *	Each block reduces <REDUCE_TO_ROW__ITEMS_PER_THREAD> times its height
		 *	(see the reduction example in CUDA Samples for details).
		 *
		 * Grid Dimensions:
		 *	"Grid extension" is required only if (height > (maxGridSizeX * abh)).
		 *
		 *	Note, however, it never happens on Compute Capability >= 3.0, if
		 *		(sizeof(index_t) == sizeof(int)) && (REDUCE_TO_ROW__ITEMS_PER_THREAD > 2).
		 *
		 *	On such architectures, maxGridSizeX >= INT_MAX, so if sizeof(index_t) == sizeof(int),
		 *	then (IDX_MAX / maxGridSizeX) <= 2.
		 *
		 *	Therefore, if REDUCE_TO_ROW__ITEMS_PER_THREAD > 2, then
		 *		(maxGridSizeX * REDUCE_TO_ROW__ITEMS_PER_THREAD) > IDX_MAX >= height
		 *
		 *	This condition will be checked on kernel compilation.
		 */
		index_t block_height;
		dim3 dimGrid;
		#if NMFGPU_PROFILING_KERNELS
			index_t timing_index;		// Non-extended grid (0), extended grid (1), single block (2), copy (3)
		#endif

		setup_matrix_operation( &matrix_to_row_kparams, height, &dimGrid,
					#if NMFGPU_PROFILING_KERNELS
						&timing_index,
					#endif
					&block_height );


		/* Non-extended grid:
		 * If there is not enough work for more than two blocks, overrides the kernel setup
		 * and uses just one block, which will iteratively read data from global memory.
		 */
		if ( dimGrid.x <= 2 ) {	// dimGrid.y == 1

			// Tries to use a block height as large as possible.

			index_t max_block_height1 = height / REDUCE_TO_ROW__ITEMS_PER_THREAD;	// (max_block_height * ITEMS) <= height
			max_block_height1 = MAX( max_block_height1, 1 );			// but greater than 0

			index_t const max_block_height2 = maxThreadsPerBlock / pitch;	// (max_block_height2 * pitch) <= maxThreadsPerBlock

			block_height = prev_power_2( MIN( max_block_height1, max_block_height2 ) );	// A power of 2 > 0

			dimGrid.x = 1;

			#if NMFGPU_PROFILING_KERNELS
				timing_index = 2;	// Single-block mode.
			#endif

		} // Single-block mode

		///////////////////////////////
		#if NMFGPU_DEBUG_REDUCT
			print_message( verb_shown_by_all, "matrix_to_row(height=%" PRI_IDX "pitch=%" PRI_IDX ",block_height=%" PRI_IDX
					",grid_extension=%" PRI_IDX ",grid_length=%" PRI_IDX ", abh=%" PRI_IDX ")...\n",
					height, pitch, block_height, dimGrid.y, dimGrid.x, matrix_to_row_kparams.abm );
		#endif
		///////////////////////////////

		// ---------------------------

		#if NMFGPU_PROFILING_KERNELS
			start_cuda_timer();
		#endif

			/* d_A[ height ][ pitch ] is reduced using a grid of (grid_extension * grid_length) blocks.
			 *
			 * d_Tmp[ grid_extension*grid_length ][ pitch ] is used as a temporary storage.
			 */
			reduce_to_row( d_A, height, pitch, d_Tmp, block_height, dimGrid, stream_AccA, d_accum_A );

			#if NMFGPU_DEBUG || NMFGPU_DEBUG_REDUCT || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_KERNELS))
			{
				cudaError_t const cuda_status = cudaGetLastError();
				if ( cuda_status != cudaSuccess ) {
					print_error( sys_error_shown_by_all, "CUDA error detected in reduce_to_row( %s ): %s\n",
							matrix_name, cudaGetErrorString( cuda_status ) );
					return EXIT_FAILURE;
				}
			}
			#endif

		#if NMFGPU_PROFILING_KERNELS
			stop_cuda_timer_cnt( &reduce_timing[ timing_index ], (size_t) height * (size_t) pitch, 1 );
		#endif

		// ---------------------------

		/* On Compute Capability < 1.2, a second call is required to finish the sum in
		 * d_Tmp[ grid_extension*grid_length ][ pitch ].
		 * Such call is performed in "single-block" mode.
		 */
		if ( (computeCapability == 1) * (computeCapability_minor < 2) * (dimGrid.x > 1) ) {

			///////////////////////////////
			#if NMFGPU_DEBUG_REDUCT
			{
				// Resulting d_Tmp from previous stage:
				print_message( verb_shown_by_all, "---Resulting d_Tmp (height=%" PRI_IDX ",width=%" PRI_IDX ",pitch=%" PRI_IDX
						",block_height=%" PRI_IDX ",grid_extension=%" PRI_IDX ",grid_length=%" PRI_IDX "):---\n",
						height, width, pitch, block_height, dimGrid.y, dimGrid.x );
				int const status1 = check_cuda_status();
				bool const real_data = true;
				bool const transpose = false;
				struct matrix_tags_t const *mt = NULL;
				int const status2  = show_device_matrix( d_Tmp, (dimGrid.y * dimGrid.x), width, pitch, real_data, transpose,
									verb_shown_by_all, mt );
				if ( (status1 != EXIT_SUCCESS) + (status2 != EXIT_SUCCESS) )
					return EXIT_FAILURE;
			}
			#endif
			///////////////////////////////

			index_t const d_Tmp_height = (dimGrid.y * dimGrid.x);

			// New grid dimensions
			dimGrid.x = dimGrid.y = 1;

			// ---------------------------

			#if NMFGPU_PROFILING_KERNELS
				start_cuda_timer();
			#endif

				/* d_Tmp[ grid_extension*grid_length ][ pitch ] is reduced with a single block.
				 * No temporary storage is required.
				 */
				reduce_to_row( d_Tmp, d_Tmp_height, pitch, NULL, block_height, dimGrid, stream_AccA, d_accum_A );

				#if NMFGPU_DEBUG || NMFGPU_DEBUG_REDUCT || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_KERNELS))
				{
					cudaError_t const cuda_status = cudaGetLastError();
					if ( cuda_status != cudaSuccess ) {
						print_error( sys_error_shown_by_all, "CUDA error detected in reduce_to_row( d_Tmp, %s ): %s\n",
								matrix_name, cudaGetErrorString( cuda_status ) );
						return EXIT_FAILURE;
					}
				}
				#endif

			#if NMFGPU_PROFILING_KERNELS
				stop_cuda_timer_cnt( &reduce_timing[2], (size_t) d_Tmp_height * (size_t) pitch, 1 );
			#endif

		} // If a second call was required.

	} else { // (height == 1)

		/* Just copies d_A[] to d_accum_A[]. */

		#if NMFGPU_PROFILING_KERNELS
			start_cuda_timer();
		#endif

			#if NMFGPU_DEBUG || NMFGPU_DEBUG_REDUCT || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_KERNELS))
				cudaError_t const cuda_status =
			#endif

				cudaMemcpyAsync( d_accum_A, d_A, pitch * sizeof(real), cudaMemcpyDeviceToDevice, stream_AccA );

			/* Same code using cuBLAS:
			 *	cublas_status = cublasSetStream( cublas_handle, stream_AccA );
			 *	cublas_status = CUBLAS_R_COPY( cublas_handle, pitch, d_A, 1, d_accum_A, 1 );
			 */

			#if NMFGPU_DEBUG || NMFGPU_DEBUG_REDUCT || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_KERNELS))
				if ( cuda_status != cudaSuccess ) {
					print_error( sys_error_shown_by_all, "matrix_to_row( %s ): cudaMemcpyAsync(): %s\n", matrix_name,
							cudaGetErrorString( cuda_status ) );
					return EXIT_FAILURE;
				}
			#endif

		#if NMFGPU_PROFILING_KERNELS
			stop_cuda_timer_cnt( &reduce_timing[3], pitch, 1 );
		#endif

	} // if ( height > 1 )


	///////////////////////////////
	#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG
	{
		print_message( verb_shown_by_all, "--- Resulting accumulator (length=%" PRI_IDX ",pitch=%" PRI_IDX
				") for matrix %s: ---\n", width, pitch, matrix_name );
		int const status1 = check_cuda_status();
		bool const real_data = true;
		struct matrix_tags_t const *mt = NULL;
		int const status2 = show_device_matrix( d_accum_A, 1, width, pitch, real_data, false, verb_shown_by_all, mt );
		if ( (status1 != EXIT_SUCCESS) + (status2 != EXIT_SUCCESS) )
			return EXIT_FAILURE;
	}
	#endif
	///////////////////////////////

	// ------------------------------------

	// Records the previous operation on stream_AccA as 'event_AccA'
	{
		#if NMFGPU_DEBUG || NMFGPU_DEBUG_REDUCT || (! NMFGPU_PROFILING_GLOBAL)
			cudaError_t const cuda_status =
		#endif

			cudaEventRecord( event_AccA, stream_AccA );

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_REDUCT || (! NMFGPU_PROFILING_GLOBAL)
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "matrix_to_row( %s ): cudaEventRecord(): %s\n", matrix_name,
						cudaGetErrorString( cuda_status ) );
				return EXIT_FAILURE;
			}
		#endif
	}

	return EXIT_SUCCESS;

} // matrix_to_row

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
int matrix_div_sub( real *__restrict__ d_A, real const *__restrict__ d_B, index_t height, index_t pitch,
			#if NMFGPU_DEBUG
				index_t width,
			#endif
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				char const *__restrict__ const matrix_name_A, char const *__restrict__ const matrix_name_B,
			#endif
			bool div_operator,
			#if NMFGPU_PROFILING_KERNELS
				timing_data_t *__restrict__ td,
			#endif
			cudaStream_t stream_A, cudaEvent_t event_B )
{

	#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
		char const *__restrict__ const operator_str = ( div_operator ? "./" : "-" );
	#endif

	// ------------------------------------------

	/* Kernel setup:
	 *
	 * Block dimensions:
	 *	Uses 1-D blocks.
	 *	Each block processes <DIV_SUB__ITEMS_PER_THREAD> times its size.
	 *
	 * Grid Dimensions:
	 *	"Grid extension" is required only if matrix_size > (maxGridSizeX * act_bs).
	 *
	 *	Note, however, it never happens on Compute Capability >= 3.0, if
	 *		gpu_max_nitems() <= IDX_MAX
	 *
	 *	On such architectures, maxGridSizeX ~ IDX_MAX. That is, (IDX_MAX / maxGridSizeX) <= 2.
	 *	So, if gpu_max_nitems() <= IDX_MAX, then (gpu_max_nitems() / maxGridSizeX) <= 2,
	 *	which is certainly less than <memory_alignment> (i.e., the minimum block size).
	 *
	 *	Therefore, (maxGridSizeX * act_bs) > gpu_max_nitems() for any (active) block size.
	 */

	size_t const matrix_size = (size_t) height * (size_t) pitch;

	index_t block_size;
	dim3 dimGrid;
	#if NMFGPU_PROFILING_KERNELS
		index_t timing_index;		// Non-extended grid (0), extended grid (1).
	#endif

	setup_matrix_operation( &matrix_div_sub_kparams, matrix_size, &dimGrid,
				#if NMFGPU_PROFILING_KERNELS
					&timing_index,
				#endif
				&block_size );

	// ------------------------------------------

	// Delays kernel launch until d_B[] is ready (skipped if host memory was mapped)

	if ( ! mappedHostMemory ) {

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cudaError_t const cuda_status =
		#endif

			cudaStreamWaitEvent( stream_A, event_B, 0 );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "matrix_div_sub( %s = %s %s %s ): cudaStreamWaitEvent( "
						"stream_%s, event_%s ): %s\n", matrix_name_A, matrix_name_B, operator_str,
						matrix_name_A, matrix_name_A, matrix_name_B, cudaGetErrorString( cuda_status ) );
				return EXIT_FAILURE;
			}
		#endif
	}

	// ------------------------------------------

	// Launches the kernel.

	#if NMFGPU_PROFILING_KERNELS
		start_cuda_timer();
	#endif

		div_sub( d_A, d_B, matrix_size, block_size, dimGrid, div_operator, stream_A );

		#if NMFGPU_DEBUG || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_KERNELS))
		{
			cudaError_t const cuda_status = cudaGetLastError();
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "CUDA error detected in div_sub(%s = %s %s %s): %s\n", matrix_name_A,
						matrix_name_B, operator_str, matrix_name_A,  cudaGetErrorString( cuda_status ) );
				return EXIT_FAILURE;
			}
		}
		#endif

		///////////////////////////////
		#if NMFGPU_DEBUG
		{
			print_message( dbg_shown_by_all, "--- Resulting %s = %s %s %s (height=%" PRI_IDX ", width=%" PRI_IDX
					", pitch=%" PRI_IDX ",block_size=%" PRI_IDX ",grid_extension=%" PRI_IDX ",grid_length=%"
					PRI_IDX "): ---\n", matrix_name_A, matrix_name_B, operator_str, matrix_name_A, height,
					width, pitch, block_size, dimGrid.y, dimGrid.x );
			int const status1 = check_cuda_status();
			bool const real_data = true;
			bool const transpose = false;
			struct matrix_tags_t const *mt = NULL;
			int const status2 = show_device_matrix( d_A, height, width, pitch, real_data, transpose, dbg_shown_by_all, mt );
			if ( (status1 != EXIT_SUCCESS) + (status2 != EXIT_SUCCESS) )
				return EXIT_FAILURE;
		}
		#endif
		///////////////////////////////

	#if NMFGPU_PROFILING_KERNELS
		stop_cuda_timer_cnt( &td[ timing_index ], matrix_size, 1 );
	#endif

	// ------------------------------------------

	// Records the operations as an event (reuses the event object; skipped if host memory was mapped).
	if ( ! mappedHostMemory ) {

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cudaError_t const cuda_status =
		#endif

			cudaEventRecord( event_B, stream_A );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "matrix_div_sub( %s = %s %s %s ): cudaEventRecord("
						"event_%s, stream_%s ): %s\n", matrix_name_A, matrix_name_B, operator_str,
						matrix_name_A, matrix_name_B, matrix_name_A, cudaGetErrorString( cuda_status ) );
				return EXIT_FAILURE;
			}
		#endif
	}

	return EXIT_SUCCESS;

} // matrix_div_sub

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
int matrix_mul_div( real *__restrict__ d_A, real const *__restrict__ d_Aux, real const *__restrict__ d_accum_B, index_t height, index_t pitch,
			#if NMFGPU_DEBUG
				index_t width, bool transpose,
			#endif
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				char const *__restrict__ const matrix_name_A,
			#endif
			#if NMFGPU_DEBUG
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

	/* Kernel setup:
	 *
	 * Block dimensions:
	 *	Uses a block width equal to <pitch>.
	 *	Each block processes up to <MUL_DIV__ITEMS_PER_THREAD> times its height.
	 *
	 * Grid dimensions:
	 *	"Grid extension" is required only if (height > (maxGridSizeX * abh)).
	 *
	 *	Note, however, it never happens on Compute Capability >= 3.0, if
	 *		(sizeof(index_t) == sizeof(int)) && (MUL_DIV__ITEMS_PER_THREAD > 2).
	 *
	 *	On such architectures, maxGridSizeX >= INT_MAX, so if sizeof(index_t) == sizeof(int),
	 *	then (IDX_MAX / maxGridSizeX) <= 2.
	 *
	 *	Therefore, if MUL_DIV__ITEMS_PER_THREAD > 2, then
	 *		(maxGridSizeX * MUL_DIV__ITEMS_PER_THREAD) > IDX_MAX >= height
	 *
	 *	This condition will be checked on kernel compilation.
	 */
	index_t block_height;
	dim3 dimGrid;
	#if NMFGPU_PROFILING_KERNELS
		index_t timing_index;		// Non-extended grid (0), extended grid (1).
	#endif

	setup_matrix_operation( &matrix_mul_div_kparams, height, &dimGrid,
				#if NMFGPU_PROFILING_KERNELS
					&timing_index,
				#endif
				&block_height );

	// ------------------------------------------

	// Delays kernel launch until d_accum_B[] is ready.
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cudaError_t const cuda_status =
		#endif

			cudaStreamWaitEvent( stream_A, event_reduction, 0 );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "matrix_mul_div(%s): cudaStreamWaitEvent(): %s\n", matrix_name_A,
						cudaGetErrorString( cuda_status ) );
				return EXIT_FAILURE;
			}
		#endif
	}

	// ------------------------------------------

	#if NMFGPU_PROFILING_KERNELS
		start_cuda_timer();
	#endif

		mul_div( d_A, d_Aux, d_accum_B, height, pitch, block_height, dimGrid, stream_A );

		#if NMFGPU_DEBUG || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_KERNELS))
		{
			cudaError_t const cuda_status = cudaGetLastError();
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "CUDA error detected in mul_div(%s): %s\n", matrix_name_A,
						cudaGetErrorString( cuda_status ) );
				return EXIT_FAILURE;
			}
		}
		#endif

		///////////////////////////////
		#if NMFGPU_DEBUG
		{
			print_message( dbg_shown_by_all, "--- Resulting %s = %s .* %s ./ %s (height=%" PRI_IDX ",width=%" PRI_IDX
					", pitch=%" PRI_IDX ",block_height=%" PRI_IDX ",grid_extension=%" PRI_IDX ",grid_length=%"
					PRI_IDX ", transpose=%i ): ---\n", matrix_name_A, matrix_name_A, matrix_name_Aux,
					matrix_name_accB, height, width, pitch, block_height, dimGrid.y, dimGrid.x, transpose );
			int const status1 = check_cuda_status();
			bool const real_data = true;
			struct matrix_tags_t const *mt = NULL;
			int const status2 = show_device_matrix( d_A, height, width, pitch, real_data, transpose, dbg_shown_by_all, mt );
			if ( (status1 != EXIT_SUCCESS) + (status2 != EXIT_SUCCESS) )
				return EXIT_FAILURE;
		}
		#endif
		///////////////////////////////

	#if NMFGPU_PROFILING_KERNELS
		stop_cuda_timer_cnt( &mul_div_timing[ timing_index ], (size_t) height * (size_t) pitch, 1 );
	#endif

	return EXIT_SUCCESS;

} // matrix_mul_div

////////////////////////////////////////////////

/*
 * d_A = MAX( d_A , R_MIN )
 *
 * 'pitch' must be a multiple of 'memory_alignment', and <= maxThreadsPerBlock.
 *
 * If 'event_A' is non-NULL, delays the kernel launch until such event completes.
 * Then, the operation is recorded using the same event object.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_adjust( real *__restrict__ d_A, index_t height, index_t pitch,
			#if NMFGPU_DEBUG
				index_t width, bool transpose,
			#endif
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				char const *__restrict__ const matrix_name_A,
			#endif
			cudaStream_t stream_A, cudaEvent_t *__restrict__ event_A )
{

	/* Kernel setup:
	 *
	 * Block dimensions:
	 *	Uses a block width equal to <pitch>.
	 *	Each block processes up to <ADJUST__ITEMS_PER_THREAD> times its height.
	 *
	 * Grid dimensions:
	 *	"Grid extension" is required only if (height > (maxGridSizeX * abh)).
	 *
	 *	Note, however, it never happens on Compute Capability >= 3.0, if
	 *		(sizeof(index_t) == sizeof(int)) && (ADJUST__ITEMS_PER_THREAD > 2).
	 *
	 *	On such architectures, maxGridSizeX >= INT_MAX, so if sizeof(index_t) == sizeof(int),
	 *	then (IDX_MAX / maxGridSizeX) <= 2.
	 *
	 *	Therefore, if ADJUST__ITEMS_PER_THREAD > 2, then
	 *		(maxGridSizeX * ADJUST__ITEMS_PER_THREAD) > IDX_MAX >= height
	 *
	 *	This condition will be checked on kernel compilation.
	 */
	index_t block_height;
	dim3 dimGrid;
	#if NMFGPU_PROFILING_KERNELS
		index_t timing_index;		// Non-extended grid (0), extended grid (1).
	#endif

	setup_matrix_operation( &matrix_adjust_kparams, height, &dimGrid,
				#if NMFGPU_PROFILING_KERNELS
					&timing_index,
				#endif
				&block_height );

	// ---------------------------

	// Delays kernel launch until d_A[] is ready.
	if ( event_A ) {

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cudaError_t const cuda_status =
		#endif

			cudaStreamWaitEvent( stream_A, *event_A, 0 );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "matrix_adjust(%s): cudaStreamWaitEvent(): %s\n", matrix_name_A,
						cudaGetErrorString( cuda_status ) );
				return EXIT_FAILURE;
			}
		#endif
	}

	// ---------------------------

	#if NMFGPU_PROFILING_KERNELS
		start_cuda_timer();
	#endif

		adjust( d_A, height, pitch, block_height, dimGrid, stream_A );

		#if NMFGPU_DEBUG || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_KERNELS))
		{
			cudaError_t const cuda_status = cudaGetLastError();
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "CUDA error detected in adjust(%s): %s\n", matrix_name_A,
						cudaGetErrorString( cuda_status ) );
				return EXIT_FAILURE;
			}
		}
		#endif

		///////////////////////////////
		#if NMFGPU_DEBUG
		{
			print_message( dbg_shown_by_all, "--- Resulting %s = MAX( %s, R_MIN ), (height=%" PRI_IDX ",width=%"
					PRI_IDX ", pitch=%" PRI_IDX ",block_height=%" PRI_IDX ",grid_extension=%" PRI_IDX
					",grid_length=%" PRI_IDX ", transpose=%i ): ---\n", matrix_name_A, matrix_name_A,
					height, width, pitch, block_height, dimGrid.y, dimGrid.x, transpose );
			int const status1 = check_cuda_status();
			bool const real_data = true;
			bool const transpose = false;
			struct matrix_tags_t const *mt = NULL;
			int const status2 = show_device_matrix( d_A, height, width, pitch, real_data, transpose, dbg_shown_by_all, mt );
			if ( (status1 != EXIT_SUCCESS) + (status2 != EXIT_SUCCESS) )
				return EXIT_FAILURE;
		}
		#endif
		///////////////////////////////

	#if NMFGPU_PROFILING_KERNELS
		stop_cuda_timer_cnt( &adjust_timing[ timing_index ], (size_t) height * (size_t) pitch, 1 );
	#endif

	// ---------------------------

	// Delays kernel launch until d_A[] is ready.
	if ( event_A ) {

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cudaError_t const cuda_status =
		#endif

			cudaEventRecord( *event_A, stream_A );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "matrix_adjust(%s): cudaEventRecord(): %s\n", matrix_name_A,
						cudaGetErrorString( cuda_status ) );
				return EXIT_FAILURE;
			}
		#endif
	}

	return EXIT_SUCCESS;

} // matrix_adjust

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
int matrix_idx_max( real const *__restrict__ d_A, index_t width, index_t pitch, index_t height,
			#if NMFGPU_DEBUG
				bool transpose,
			#endif
			#if NMFGPU_DEBUG || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_KERNELS))
				char const *__restrict__ const matrix_name_A,
			#endif
			#if NMFGPU_DEBUG
				char const *__restrict__ const matrix_name_Idx,
			#endif
			cudaStream_t stream_A, index_t *__restrict__ d_Idx )
{

	/* Kernel setup:
	 *
	 * Block Dimensions:
	 *	Each thread processes <IDX_MAX__ITEMS_PER_THREAD> from the same row.
	 *	(block_width * IDX_MAX__ITEMS_PER_THREAD) must be >= width
	 *
	 *	That is (using pitch, instead),
	 *		block_width >= (pitch / IDX_MAX__ITEMS_PER_THREAD).
	 *
	 *	In addition, block_width must be a power of 2, and >= memory_alignment
	 *
	 *	If pitch <= (memory_alignment * IDX_MAX__ITEMS_PER_THREAD),
	 *	then memory_alignment (which is a power of 2) is the required block_width
	 *
	 * Grid dimensions:
	 *	"Grid extension" is required only if (height > (maxGridSizeX * abh)).
	 *
	 *	Note, however, it never happens on Compute Capability >= 3.0, if
	 *		(sizeof(index_t) == sizeof(int)) && (IDX_MAX__ITEMS_PER_THREAD > 2).
	 *
	 *	On such architectures, maxGridSizeX >= INT_MAX, so if sizeof(index_t) == sizeof(int),
	 *	then (IDX_MAX / maxGridSizeX) <= 2.
	 *
	 *	Therefore, if IDX_MAX__ITEMS_PER_THREAD > 2, then
	 *		(maxGridSizeX * IDX_MAX__ITEMS_PER_THREAD) > IDX_MAX >= height
	 *
	 *	This condition will be checked on kernel compilation.
	 */
	index_t block_height;
	dim3 dimGrid;
	#if NMFGPU_PROFILING_KERNELS
		index_t timing_index;		// Non-extended grid (0), extended grid (1).
	#endif

	setup_matrix_operation( &matrix_idx_max_kparams, height, &dimGrid,
				#if NMFGPU_PROFILING_KERNELS
					&timing_index,
				#endif
				&block_height );

	// HACK: Extracts block_width from matrix_idx_max_kparams.abm.
	index_t const block_width = matrix_idx_max_kparams.abm;

	// Final block dimensions
	dim3 const dimBlock( block_width, block_height );

	// ---------------------------------------------

	#if NMFGPU_PROFILING_KERNELS
		start_cuda_timer();
	#endif

		idx_max( d_A, height, width, pitch, dimBlock, dimGrid, stream_A, d_Idx );

		#if NMFGPU_DEBUG || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_KERNELS))
		{
			cudaError_t const cuda_status = cudaGetLastError();
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "CUDA error detected in idx_max( %s ): %s\n", matrix_name_A,
						cudaGetErrorString( cuda_status ) );
				return EXIT_FAILURE;
			}
		}
		#endif

		///////////////////////////////
		#if NMFGPU_DEBUG
		{
			print_message( dbg_shown_by_all, "--- Resulting %s[i] = max(%s[i][..]) (height=%" PRI_IDX ",width=%" PRI_IDX
					", pitch=%" PRI_IDX ",block_width=%" PRI_IDX ",block_height=%" PRI_IDX ",grid_extension=%" PRI_IDX
					",grid_length=%" PRI_IDX ",transpose=%i): ---\n", matrix_name_Idx, matrix_name_A, height,
					width, pitch, block_width, block_height, dimGrid.y, dimGrid.x, transpose );
			int const status1 = check_cuda_status();
			bool const real_data = false;
			struct matrix_tags_t const *mt = NULL;
			int const status2 = show_device_matrix( d_Idx, 1, height, height, real_data, transpose, dbg_shown_by_all, mt );
			if ( (status1 != EXIT_SUCCESS) + (status2 != EXIT_SUCCESS) )
				return EXIT_FAILURE;
		}
		#endif
		///////////////////////////////

	#if NMFGPU_PROFILING_KERNELS
		stop_cuda_timer_cnt( &idx_max_timing[ timing_index ], (size_t) height * (size_t) pitch, 1 );
	#endif

	return EXIT_SUCCESS;

} // matrix_idx_max

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
int upload_matrix( real const *__restrict__ A, index_t height, index_t pitch, real *__restrict__ d_A,
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
				index_t width, bool transpose,
			#endif
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 || (! NMFGPU_PROFILING_GLOBAL)
				char const *__restrict__ const matrix_name_A,
			#endif
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
				char const *__restrict__ const matrix_name_dA,
			#endif
			#if NMFGPU_PROFILING_TRANSF
				timing_data_t *__restrict__ const upload_timing,
			#endif
			cudaStream_t stream_A, cudaEvent_t *__restrict__ event_A )
{

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "\nUploading Matrix %s to %s (height=%" PRI_IDX ",width=%" PRI_IDX ",pitch=%" PRI_IDX
				",transpose: %i, event %s, mapped memory: %i)\n", matrix_name_A, matrix_name_dA, height, width,
				pitch, transpose, ( event_A ? "provided" : "NOT provided"), mappedHostMemory );
	#endif

	// ----------------------------------

	#if NMFGPU_SYNC_TRANSF
		// Synchronous data transfer: Waits until all previous operations have finished.
		if ( check_cuda_status() != EXIT_SUCCESS )
			return EXIT_FAILURE;
	#endif

	///////////////////////////////
	#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
		print_message( dbg_shown_by_all, "--- Uploading matrix %s to %s (height=%" PRI_IDX ",width=%" PRI_IDX ",pitch=%" PRI_IDX
				",transpose: %i, event %s, mapped memory: %i): ---\n", matrix_name_A, matrix_name_dA, height,
				width, pitch, transpose, ( event_A ? "provided" : "NOT provided"), mappedHostMemory );
	#endif
	/////////////////////////////

	// ----------------------------------

	if ( ! mappedHostMemory ) {	// If host memory was NOT mapped.

		// Starts the transfer...

		size_t const nitems = (size_t) height * (size_t) pitch;

		#if NMFGPU_PROFILING_TRANSF
			start_cuda_timer();
		#endif

			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF \
				|| ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_TRANSF))
				cudaError_t const cuda_status =
			#endif

				cudaMemcpyAsync( d_A, A, nitems * sizeof(real), cudaMemcpyHostToDevice, stream_A );

				/* Same code using cuBLAS:
				 *	cublasStatus cublas_status =
				 *		cublasSetVectorAsync( nitems, sizeof(real), A, 1, d_A, 1, stream_A );
				 */

			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF \
				|| ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_TRANSF))
				if ( cuda_status != cudaSuccess ) {
					print_error( sys_error_shown_by_all, "Could not upload matrix %s: %s\n", matrix_name_A,
							cudaGetErrorString( cuda_status ) );
					return EXIT_FAILURE;
				}
			#endif

			///////////////////////////////
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF
				if ( check_cuda_status() != EXIT_SUCCESS ) {
					print_error( sys_error_shown_by_all, "Error uploading matrix %s\n", matrix_name_A );
					return EXIT_FAILURE;
				}
			#endif
			/////////////////////////////

		#if NMFGPU_PROFILING_TRANSF
			stop_cuda_timer_cnt( upload_timing, nitems, 1 );
		#endif

	} // If host memory was NOT mapped.

	// ----------------------------------

	///////////////////////////////
	#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
	{
		append_printed_message( dbg_shown_by_all, "\n");
		bool const real_data = true;
		struct matrix_tags_t const *mt = NULL;
		int const status2 = show_device_matrix( d_A, height, width, pitch, real_data, transpose, dbg_shown_by_all, mt );
		if ( status2 != EXIT_SUCCESS )
			return EXIT_FAILURE;
	}
	#endif
	/////////////////////////////

	// ----------------------------------

	// If 'event_A' is non-NULL, the operation is recorded as an event.
	if ( event_A ) {

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || (! NMFGPU_PROFILING_GLOBAL)
			cudaError_t const cuda_status =
		#endif

			cudaEventRecord( *event_A, stream_A );

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || (! NMFGPU_PROFILING_GLOBAL)
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "upload_matrix(%s): cudaEventRecord(): %s\n", matrix_name_A,
						cudaGetErrorString( cuda_status ) );
				return EXIT_FAILURE;
			}
		#endif

	} // Records the operation as an event.

	// ----------------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Uploading Matrix %s to %s (height=%" PRI_IDX ",width=%" PRI_IDX ",pitch=%" PRI_IDX
				",transpose: %i, event %s).. Done\n\n", matrix_name_A, matrix_name_dA, height, width, pitch,
				transpose, ( event_A ? "provided" : "NOT provided") );
	#endif

	return EXIT_SUCCESS;

} // upload_matrix

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
 * NOTE: If host memory was mapped, SKIPS the transfer operation and ALL event actions.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int upload_matrix_partial( real const *__restrict__ A, index_t height, index_t pitch, index_t strow, index_t stcol,
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
				index_t block_width,
			#endif
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 || (! NMFGPU_PROFILING_GLOBAL)
				char const *__restrict__ const matrix_name_A,
			#endif
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
				char const *__restrict__ const matrix_name_dA,
			#endif
			index_t block_pitch, real *__restrict__ d_A, cudaStream_t stream_A, cudaEvent_t event_A
			#if NMFGPU_PROFILING_TRANSF
				, timing_data_t *__restrict__ const upload_timing
			#endif
			)
{

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Uploading Matrix %s to %s (partial, height=%" PRI_IDX ",pitch=%" PRI_IDX ",stcol=%" PRI_IDX
				",strow=%" PRI_IDX ",block_width=%" PRI_IDX ", block_pitch=%" PRI_IDX ")\n", matrix_name_A, matrix_name_dA,
				height, pitch, strow, stcol, block_width, block_pitch );
	#endif

	// ----------------------------------

	#if NMFGPU_SYNC_TRANSF
		// Synchronous data transfer: Waits until all previous operations have finished.
		if ( check_cuda_status() != EXIT_SUCCESS )
			return EXIT_FAILURE;
	#endif

	///////////////////////////////
	#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
		print_message( dbg_shown_by_all, "--- Uploading Matrix %s to %s (partial, height=%" PRI_IDX ",pitch=%" PRI_IDX ",strow=%"
				PRI_IDX ",stcol=%" PRI_IDX ",block_width=%" PRI_IDX ", block_pitch=%" PRI_IDX ", mappedHostMemory=%i)\n",
				matrix_name_A, matrix_name_dA, height, pitch, strow, stcol, block_width, block_pitch, mappedHostMemory );
	#endif
	/////////////////////////////

	// ----------------------------------

	if ( ! mappedHostMemory ) {	// If host memory was NOT mapped.

		// Delays the transfer until the event has completed all previous operations.
		{
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || (! NMFGPU_PROFILING_GLOBAL)
				cudaError_t const cuda_status =
			#endif

				cudaStreamWaitEvent( stream_A, event_A, 0 );

			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || (! NMFGPU_PROFILING_GLOBAL)
				if ( cuda_status != cudaSuccess ) {
					print_error( sys_error_shown_by_all, "upload_matrix_partial(%s): cudaStreamWaitEvent(): %s\n",
							matrix_name_A, cudaGetErrorString( cuda_status ) );
					return EXIT_FAILURE;
				}
			#endif
		}
		// ----------------------------------

		// Starts the transfer...

		size_t const nitems = (size_t) height * (size_t) pitch;

		#if NMFGPU_PROFILING_TRANSF
			start_cuda_timer();
		#endif
		{

			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF \
				|| ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_TRANSF))
				cudaError_t cuda_status = cudaSuccess;
			#endif

			if ( (block_pitch < pitch) + (stcol > 0) ) {

				/*
				 * It must be transferred as a 2D matrix.
				 */

				/* If necessary, adjusts the width to avoid an out-of-bound failure in CPU memory,
				 * but then, such width will NOT be a multiple of 'memory_alignment', resulting in a slower transfer.
				 */
				index_t const width = MIN( block_pitch, (pitch-stcol) );

				#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF \
					|| ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_TRANSF))
					cuda_status =
				#endif

					cudaMemcpy2DAsync( d_A, block_pitch * sizeof(real), &A[ ((size_t) strow * (size_t) pitch) + stcol ],
								pitch * sizeof(real), width * sizeof(real), height, cudaMemcpyHostToDevice,
								stream_A );

				/* Same code using cuBLAS:
				 *	cublasStatus cublas_status =
				 *		cublasSetMatrixAsync( width, height, sizeof(real), &A[ ((size_t)strow * (size_t)pitch) + stcol ],
				 *					pitch, d_A, block_pitch, stream_A );
				 */

			// ( block_pitch == pitch ) && ( stcol == 0 )
			} else {

				/*
				 * It can be transferred as a row vector.
				 */

				#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF \
					|| ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_TRANSF))
					cuda_status =
				#endif

					cudaMemcpyAsync( d_A, &A[ ((size_t) strow * (size_t) pitch) ], nitems * sizeof(real),
							cudaMemcpyHostToDevice, stream_A );

				/* Same code using cuBLAS:
				 *	cublasStatus cublas_status =
				 *		cublasSetVectorAsync( nitems, sizeof(real), &A[ ((size_t) strow * (size_t) pitch) ],
				 *					1, d_A, 1, stream_A );
				 */

			} // if ( ( block_pitch < pitch ) || ( stcol > 0 ) )

			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF \
				|| ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_TRANSF))
				if ( cuda_status != cudaSuccess ) {
					print_error( sys_error_shown_by_all, "Could not upload (partial) matrix %s: %s\n", matrix_name_A,
							cudaGetErrorString( cuda_status ) );
					return EXIT_FAILURE;
				}
			#endif

			///////////////////////////////
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF
				if ( check_cuda_status() != EXIT_SUCCESS ) {
					print_error( sys_error_shown_by_all, "Error uploading (partial) matrix %s\n", matrix_name_A );
					return EXIT_FAILURE;
				}
			#endif
			///////////////////////////////
		}
		#if NMFGPU_PROFILING_TRANSF
			stop_cuda_timer_cnt( upload_timing, nitems, 1 );
		#endif

		// ----------------------------------

		// Records this operation using the same event object.
		{
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || (! NMFGPU_PROFILING_GLOBAL)
				cudaError_t const cuda_status =
			#endif

				cudaEventRecord( event_A, stream_A );

			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || (! NMFGPU_PROFILING_GLOBAL)
				if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "upload_matrix_partial(%s): cudaEventRecord(): %s\n",
						matrix_name_A, cudaGetErrorString( cuda_status ) );
					return EXIT_FAILURE;
				}
			#endif
		}

	} // If host memory was NOT mapped.

	// ----------------------------------

	///////////////////////////////
	#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
	{
		append_printed_message( dbg_shown_by_all, "\n");
		bool const real_data = true;
		bool const transpose = false;
		struct matrix_tags_t const *mt = NULL;
		int const status2 = show_device_matrix( d_A, height, block_width, block_pitch, real_data, transpose, dbg_shown_by_all, mt );
		if ( status2 != EXIT_SUCCESS )
			return EXIT_FAILURE;
	}
	#endif
	///////////////////////////////

	// ----------------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Uploading Matrix %s to %s (partial, height=%" PRI_IDX ",pitch=%" PRI_IDX ",strow=%" PRI_IDX
				",stcol=%" PRI_IDX ",block_width=%" PRI_IDX ", block_pitch=%" PRI_IDX ", mappedHostMemory=%i)...Done.\n\n",
				matrix_name_A, matrix_name_dA, height, pitch, strow, stcol, block_width, block_pitch, mappedHostMemory );
	#endif

	return EXIT_SUCCESS;

} // upload_matrix_partial

////////////////////////////////////////////////

/*
 * Transfers a matrix from the DEVICE (GPU) to HOST (CPU), as a row vector.
 *
 * A[1..height][1..pitch] <--- d_A[1..height][1..pitch]
 *
 * nitems == (height * pitch)
 *
 * NOTE: If host memory was mapped, the transfer operation is SKIPPED.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int download_matrix( void *__restrict__ A, size_t nitems, size_t data_size, void const *__restrict__ d_A,
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
				index_t height, index_t width, index_t pitch, bool real_data, bool transpose,
				char const *__restrict__ const matrix_name_A,
			#endif
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
				|| ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_TRANSF))
				char const *__restrict__ const matrix_name_dA,
			#endif
			#if NMFGPU_PROFILING_TRANSF
				timing_data_t *__restrict__ const download_timing,
			#endif
			cudaStream_t stream_A )
{

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Downloading Matrix %s to %s (no event, height=%" PRI_IDX ",width=%" PRI_IDX ",pitch=%"
				PRI_IDX ",transpose: %i, mappedHostMemory: %i)\n", matrix_name_dA, matrix_name_A, height, width,
				pitch, transpose, mappedHostMemory );
	#endif

	// ----------------------------------

	#if NMFGPU_SYNC_TRANSF
		// Synchronous data transfer: Waits until all previous operations have finished.
		if ( check_cuda_status() != EXIT_SUCCESS )
			return EXIT_FAILURE;
	#endif

	///////////////////////////////
	#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
		print_message( dbg_shown_by_all, "--- Downloading matrix %s to %s (no event, height=%" PRI_IDX ",width=%"
				PRI_IDX ",pitch=%" PRI_IDX ",transpose: %i, mappedHostMemory: %i): ---\n", matrix_name_dA,
				matrix_name_A, height, width, pitch, transpose, mappedHostMemory);
	#endif
	/////////////////////////////

	// ----------------------------------

	if ( ! mappedHostMemory ) {	// If host memory was NOT mapped.

		// Starts the transfer...

		#if NMFGPU_PROFILING_TRANSF
			start_cuda_timer();
		#endif

			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF \
				|| ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_TRANSF))
				cudaError_t const cuda_status =
			#endif

				cudaMemcpyAsync( A, d_A, nitems * data_size, cudaMemcpyDeviceToHost, stream_A );

			/* Same code using cuBLAS:
			 *	cublasStatus cublas_status =
			 *		cublasSetVectorAsync( nitems, data_size, d_A, 1, A, 1, stream_A );
			 */

			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF \
				|| ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_TRANSF))
				if ( cuda_status != cudaSuccess ) {
					print_error( sys_error_shown_by_all, "Could not download matrix %s: %s\n", matrix_name_dA,
							cudaGetErrorString( cuda_status ) );
					return EXIT_FAILURE;
				}
			#endif

			///////////////////////////////
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_SYNC_TRANSF
				if ( check_cuda_status() != EXIT_SUCCESS ) {
					print_error( sys_error_shown_by_all, "Error downloading matrix %s\n", matrix_name_dA );
					return EXIT_FAILURE;
				}
			#endif
			///////////////////////////////


		#if NMFGPU_PROFILING_TRANSF
			stop_cuda_timer_cnt( download_timing, nitems, 1 );
		#endif

	} // If host memory was NOT mapped.

	// ----------------------------------

	///////////////////////////////
	#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
	{
		append_printed_message( dbg_shown_by_all, "\n");
		struct matrix_tags_t const *mt = NULL;
		int const status2 = matrix_show( A, height, width, pitch, real_data, transpose, dbg_shown_by_all, mt );
		if ( status2 != EXIT_SUCCESS )
			return EXIT_FAILURE;
	}
	#endif
	/////////////////////////////

	// ----------------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Downloading Matrix %s to %s (no event, height=%" PRI_IDX ",width=%" PRI_IDX
				",pitch=%" PRI_IDX ",transpose: %i, mappedHostMemory: %i)...Done.\n\n", matrix_name_dA, matrix_name_A,
				height, width, pitch, transpose, mappedHostMemory );
	#endif

	return EXIT_SUCCESS;

} // download_matrix

////////////////////////////////////////////////
////////////////////////////////////////////////
