/************************************************************************
 *
 * BioNMF-GPU 2.0 -- Non-negative Matrix Factorization on (multi-)GPU systems.
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
 ***********************************************************************/
/**********************************************************
 * GPU_setup.cu
 *	Generic definitions and routines for GPU set-up and management.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Data type:
 *		NMFGPU_SINGLE_PREC: Makes use of single-precision data (i.e., 'float').
 *
 *	Additional information:
 *		NMFGPU_VERBOSE: Shows some messages concerning the progress of the program, as well as
 *				some configuration parameters.
 *		NMFGPU_VERBOSE_2: Shows the parameters in some routine calls.
 *
 *	CPU timing:
 *		NMFGPU_PROFILING_GLOBAL: Compute total elapsed time. If GPU time is NOT being computed,
 *					the CPU thread performs active waiting (i.e., spins) on
 *					synchronization calls, such as cudaDeviceSynchronize() or
 *					cudaStreamSynchronize(). Otherwise, the CPU thread is blocked.
 *
 *	GPU timing (WARNING: They PREVENT asynchronous operations. The CPU thread is blocked on synchronization):
 *		NMFGPU_PROFILING_TRANSF: Compute timing of data transfers. Shows additional information.
 *		NMFGPU_PROFILING_KERNELS: Compute timing of CUDA kernels. Shows additional information.
 *
 *	Debug / Testing:
 *		NMFGPU_CPU_RANDOM: Uses the CPU (host) random generator (not the cuRAND library).
 *		NMFGPU_FIXED_INIT: Uses "random" values generated from a fixed seed (defined in common.h).
 *		NMFGPU_DEBUG: Shows the result of each matrix operation and data transfer.
 *		NMFGPU_FORCE_BLOCKS: Forces the processing of the input matrix as four blocks.
 *				     It also disables mapping of host memory into device address space.
 *		NMFGPU_TEST_BLOCKS: Just shows block information structure. No GPU memory is allocated.
 *
 **********************************************************
 **********************************************************
 **********************************************************
 *
 * Data matrices:
 *	V (N rows, M columns): input matrix
 *	W (N,K): output matrix
 *	H (K,M): output matrix,
 * such that: V  ~  W * H.
 *
 * Arguments:
 *	Matrix V (and its dimensions)
 *	K: Factorization Rank
 *
 *
 * NOTE: In order to improve performance:
 *
 *	+ Matrix H is stored in memory as COLUMN-major (i.e., it is transposed).
 *
 *	+ All matrices include useless data for padding. Padded dimensions
 *	  are denoted with the 'p' character. For instance:
 *		Mp, which is equal to <M + padding>
 *		Kp, which is equal to <K + padding>.
 *
 *	  Data alignment is controlled by the global variable: memory_alignment.
 *
 *	  This leads to the following limits:
 *		- Maximum number of columns (padded or not): matrix_max_pitch.
 *		- Maximum number of rows: matrix_max_non_padded_dim.
 *		- Maximum number of items: matrix_max_num_items.
 *
 *	  All four GLOBAL variables must be initialized with the set_matrix_limits()
 *	  function.
 *
 ****************
 *
 * Multi-GPU version:
 *
 * When the input matrix V is distributed among multiple devices each host thread processes
 * the following sets of rows and columns:
 *
 *	Vrow[ 1..NpP ][ 1..M ] <-- V[ bN..(bN+NpP) ][ 1..M ]	(i.e., NpP rows, starting from bN)
 *	Vcol[ 1..N ][ 1..MpP ] <-- V[ 1..N ][ bM..(bM+MpP) ]	(i.e., MpP columns, starting from bM)
 *
 * Such sets allow to update the corresponding rows and columns of W and H, respectively.
 *
 * Note that each host thread has a private full copy of matrices W and H, which must be synchronized
 * after being updated.
 *
 ****************
 *
 * Large input matrix (blockwise processing):
 *
 * If the input matrix (or the portion assigned to this device) is too large for the GPU memory,
 * it must be blockwise processed as follow:
 *
 *	d_Vrow[1..BLN][1..Mp] <-- Vrow[ offset..(offset + BLN) ][1..Mp]			(i.e., BLN <= NpP rows)
 *	d_Vcol[1..N][1..BLMp] <-- Vcol[1..N][ offset_Vcol..(offset_Vcol + BLMp) ]	(i.e., BLM <= MpP columns)
 *
 * Note that padded dimensions are denoted with the suffix 'p' (e.g., Mp, BLMp, etc).
 *
 * In any case, matrices W and H are fully loaded into the GPU memory.
 *
 * Information for blockwise processing is stored in two block_t structures (one for each dimension).
 * Such structures ('block_N' and 'block_M') are initialized in the init_block_conf() routine.
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
 *********************************************************/

// Required by <inttypes.h>
#ifndef __STDC_FORMAT_MACROS
	#define __STDC_FORMAT_MACROS (1)
#endif

#include "GPU_setup.cuh"
#include "common.h"
#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
	#include "timing.cuh"
#endif
#include "matrix/matrix_operations.cuh"		/* check_matrix_dimensions(), max_num_items(), init_kernel_params() */

#include <stdlib.h>
#include <errno.h>
#include <string.h>	/* memset() */
#include <limits.h>	/* [U]INT_MAX */
#include <inttypes.h>	/* uintptr_t, uint_fast64_t, PRIuFAST64 */

////////////////////////////////////////////////
////////////////////////////////////////////////

/* Constants */

// cudaSharedMemConfig values used in cudaDeviceSetSharedMemConfig() function.
#undef CUDA_SHARED_MEM_BANK_SIZE
#if NMFGPU_SINGLE_PREC
	#define CUDA_SHARED_MEM_BANK_SIZE ( cudaSharedMemBankSizeFourByte )
#else
	#define CUDA_SHARED_MEM_BANK_SIZE ( cudaSharedMemBankSizeEightByte )
#endif

// ---------------------------------------------
// ---------------------------------------------

/* Macro Functions */

/* Alignment value for data to be stored in GPU memory:
 *	Typically <warpSize/2> on Compute Capability 1.x,
 *	and <warpSize> on CC 2.x and beyond.
 */
#ifndef SET_MEMORY_ALIGNMENT
	#define SET_MEMORY_ALIGNMENT(cc_major,warpSize) ( ((cc_major) > 1) ? (warpSize) : ((warpSize) >> 1) )
#endif

// ---------------------------------------------
// ---------------------------------------------

/* HOST-ONLY GLOBAL Variables */

index_t device_id = 0;			// Current device ID.

cublasHandle_t cublas_handle;		// cuBLAS library context.

curandGenerator_t curand_generator;	// cuRAND Random values Generator.

index_t computeCapability = 2;		// Compute Capability (major).

index_t computeCapability_minor = 0;	// Compute Capability (minor).

index_t maxThreadsPerBlock = 512;	// Maximum number of threads per block.

index_t multiProcessorCount = 30;	// Number of multiprocessors.

index_t maxThreadsPerMultiProcessor = 1024; // Maximum number of resident threads per multiprocessor (>= maxThreadsPerBlock)

// Maximum number of thread blocks on dimensions X and Y.
index_t maxGridSizeX = 65535, maxGridSizeY = 65535;

// Typical number of threads per block. It should be a divisor of maxThreadsPerMultiProcessor.
index_t threadsPerBlock = 256;		// <= maxThreadsPerBlock

// Threads per block for kernels requiring a value multiple of <Kp> (denoted as 'pitch').
index_t threadsPerBlock_pitch = 256;	// threadsPerBlock <= threadsPerBlock_pitch <= maxThreadsPerBlock

// Maximum block height using <threadsPerBlock_pitch> threads.
index_t maxBlockHeight_pitch = 16;	// (threadsPerBlock_pitch / pitch)

bool mappedHostMemory = false;		// Host memory is mapped into the address space of the device.

// Maximum number of items of input arrays in GPU kernels.
size_t gpu_max_num_items = matrix_max_num_items; // <= matrix_max_num_items

block_t block_N, block_M;		// Information for blockwise processing on dimension N and M.

// CUDA Events for synchronization:
cudaEvent_t event_Vrow;				// d_Vrow
cudaEvent_t event_Vcol;				// d_Vcol
cudaEvent_t event_W;				// d_W
cudaEvent_t event_H;				// d_H
cudaEvent_t event_reduction;			// Event to register matrix reduction operations.

// CUDA Streams for synchronization:
cudaStream_t stream_Vrow;			// d_Vrow
cudaStream_t stream_Vcol;			// d_Vcol
cudaStream_t stream_W;				// d_W
cudaStream_t stream_H;				// d_H
cudaStream_t *__restrict__ streams_NMF = NULL;	// Main-flow streams for blockwise processing.
index_t num_streams_NMF = 1;			// Number of main-flow streams: MAX( SUM(block_N.num_steps[i]), SUM(block_M.num_steps[i]) )

// Host matrices (used only with mapped host memory):
real *__restrict__ h_WH = NULL;			// Temporary matrix: d_WH = d_W * d_H
real *__restrict__ h_Aux = NULL;		// Temporary matrix. Sometimes denoted as d_Waux or d_Haux.
real *__restrict__ h_accum = NULL;		// Accumulator. K-length vector (<Kp> with padding).
real const *__restrict__ h_scalar = NULL;	// Scalars for cuBLAS Library calls: scalar[2] = { 0, 1 };

// ---------------------------------------------

/* "Private" HOST-ONLY Global variables */

/* Default stack size of each GPU thread ( Compute Capability >= 2.0 )
 * See:
 *	https://devtalk.nvidia.com/default/topic/481553/curand-eats-device-memory/
 */
static size_t defaultStackSize = 0;

// Information and/or error messages shown by all processes.
#if NMFGPU_DEBUG || NMFGPU_VERBOSE || NMFGPU_VERBOSE_2 || NMFGPU_FORCE_BLOCKS || NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_TEST_BLOCKS
	static bool const dbg_shown_by_all = true;	// Information or error messages on debug.
	static bool const verb_shown_by_all = false;	// Information messages in verbose mode.
#endif
static bool const sys_error_shown_by_all = true;	// System error messages.
static bool const error_shown_by_all = false;		// Error messages on invalid arguments or I/O data.

// ---------------------------------------------
// ---------------------------------------------

/* DEVICE-ONLY GLOBAL Variables */

// Data matrices (device side):
real *__restrict__ d_Vrow = NULL;		// Block of BLN rows from input matrix V.
real *__restrict__ d_Vcol = NULL;		// Block of BLM columns from input matrix V.
real *__restrict__ d_H = NULL;			// Output matrix. Note that it is transposed.
real *__restrict__ d_W = NULL;			// Output matrix.
real *__restrict__ d_WH = NULL;			// Temporary matrix: d_WH = d_W * d_H
real *__restrict__ d_Aux = NULL;		// Temporary matrix. Sometimes denoted as d_Waux or d_Haux.
real *__restrict__ d_accum = NULL;		// Accumulator. K-length vector (<Kp> with padding).
index_t  *__restrict__ d_classification = NULL;	// Classification vector.

// Previous Classification vector (used only with mapped host memory).
index_t  *__restrict__ d_last_classification = NULL;

real const *__restrict__ d_scalar = NULL;	// Scalars for cuBLAS Library calls. d_scalar[2] = { 0, 1 };
real const *__restrict__ d_zero = NULL;		// Pointer to d_scalar[0] == 0
real const *__restrict__ d_one = NULL;		// Pointer to d_scalar[1] == 1

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Returns an error message according to the provided cuBLAS status.
 */
char const *getCublasErrorString( cublasStatus_t cublas_status )
{

	switch( cublas_status ) {

		case CUBLAS_STATUS_SUCCESS:
			return "Operation completed successfully.";
		// break;

		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "cuBLAS library not initialized.";
		// break;

		case CUBLAS_STATUS_ALLOC_FAILED:
			return "Resource allocation failed.";
		// break;

		case CUBLAS_STATUS_INVALID_VALUE:
			return "An invalid numerical value was used as an argument.";
		// break;

		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "Architecture mismatch, GPU does not support the requested feature.";
		// break;

		case CUBLAS_STATUS_MAPPING_ERROR:
			return "Access to GPU memory space failed.";
		// break;

		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "GPU program failed to execute.";
		// break;

		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "An internal cuBLAS operation failed.";
		// break;

		/* // CUDA Runtime >= 6.0
		 * case CUBLAS_STATUS_NOT_SUPPORTED:
		 *	return "The functionality requested is not supported.";
		 * // break;
		 */

		default:
			return "Unknown cuBLAS error code.";
		// break;
	}

} // getCublasErrorString

////////////////////////////////////////////////

/*
 * Returns an error message according to the provided cuRAND status.
 */
char const *getCurandErrorString( curandStatus_t curand_status )
{

	switch( curand_status ) {

		case CURAND_STATUS_SUCCESS:
			return "Operation completed successfully.";
		// break;

		case CURAND_STATUS_VERSION_MISMATCH:
			return "Header file and linked library version do not match.";
		// break;

		case CURAND_STATUS_NOT_INITIALIZED:
			return "cuRAND generator not initialized.";
		// break;

		case CURAND_STATUS_ALLOCATION_FAILED:
			return "Resource allocation failed.";
		// break;

		case CURAND_STATUS_TYPE_ERROR:
			return "Invalid random number generator type.";
		// break;

		case CURAND_STATUS_OUT_OF_RANGE:
			return "Argument is out of range.";
		// break;

		case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
			return "Length requested is not a multiple of dimension.";
		// break;

		case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
			return "GPU device does not support double-precision data.";
		// break;

		case CURAND_STATUS_LAUNCH_FAILURE:
			return "Kernel launch failure.";
		// break;

		case CURAND_STATUS_PREEXISTING_FAILURE:
			return "Preexisting failure on library entry.";
		// break;

		case CURAND_STATUS_INITIALIZATION_FAILED:
			return "Initialization of CUDA failed.";
		// break;

		case CURAND_STATUS_ARCH_MISMATCH:
			return "Architecture mismatch, GPU does not support the requested feature.";
		// break;

		case CURAND_STATUS_INTERNAL_ERROR:
			return "An internal cuRAND operation failed.";
		// break;

		default:
			return "Unknown cuRAND error code.";
		// break;
	}

} // getCurandErrorString

////////////////////////////////////////////////

/*
 * Checks the provided cuBLAS status.
 * If it is NOT OK, it shows an error message.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int check_cublas_status_st( cublasStatus_t cublas_status )
{

	if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
		print_error( sys_error_shown_by_all, "cuBLAS Error detected: %s\n", getCublasErrorString( cublas_status ) );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // check_cublas_status_st

////////////////////////////////////////////////

/*
 * Waits to finish all GPU operations and checks CUDA status.
 * If it is NOT OK, it shows an error message.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int check_cuda_status_st( cudaError_t cuda_status )
{

	if (	( cuda_status != cudaSuccess ) ||
		( (cuda_status=cudaDeviceSynchronize()) != cudaSuccess ) ||
		( (cuda_status=cudaGetLastError()) != cudaSuccess) ) {
		print_error( sys_error_shown_by_all, "CUDA Error detected: %s\n", cudaGetErrorString(cuda_status) );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // check_cuda_status_st

////////////////////////////////////////////////

/*
 * Waits to finish all GPU operations and checks CUDA status.
 * If it is NOT OK, it shows an error message.
 * It is similar to perform: check_cuda_status_st( cudaSuccess )
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int check_cuda_status( void )
{

	cudaError_t cuda_status = cudaDeviceSynchronize();

	if ( (cuda_status != cudaSuccess) || ( (cuda_status=cudaGetLastError()) != cudaSuccess) ) {
		print_error( sys_error_shown_by_all, "CUDA Error detected: %s\n", cudaGetErrorString(cuda_status) );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // check_cuda_status


////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/* Helper function of setup_gpu_factorization_rank()
 *
 * Sets the following GLOBAL variables:
 *
 * threadsPerBlock_pitch:
 *	Number of threads per block for kernels requiring a value multiple of <pitch>
 *	threadsPerBlock <= threadsPerBlock_pitch <= maxThreadsPerBlock
 *
 * maxBlockHeight_pitch:
 *	Maximum block height using <threadsPerBlock_pitch> threads.
 *	maxBlockHeight_pitch <= (threadsPerBlock_pitch / pitch)
 */
static void set_threadsPerBlock_pitch( index_t pitch )
{

	// Number of threads per block for kernels requiring a value multiple of <Kp> (denoted as 'pitch').
	threadsPerBlock_pitch = threadsPerBlock;

	// Maximum block height (result possibly truncated).
	maxBlockHeight_pitch = threadsPerBlock / pitch;

	/* Some kernels make use of shared memory only if block height > 1.
	 * If necessary, increases the block size to try to ensure this.
	 */
	if ( maxBlockHeight_pitch == 1 ) {

		/* With maxThreadsPerBlock, the block height may be increased
		 * to two or three rows, but then there will be just one or
		 * two blocks per multiprocessor. Instead, the block size is
		 * increased to (maxThreadsPerBlock + threadsPerBlock) / 2,
		 * which allows up to two rows, but two or three blocks per
		 * multiprocessor.
		 */

		index_t const tpb = (maxThreadsPerBlock + threadsPerBlock) / 2;

		maxBlockHeight_pitch = (tpb / pitch);	// (possibly truncated)

		// Final number of threads per block must be a multiple of <pitch>.
		threadsPerBlock_pitch = pitch * maxBlockHeight_pitch;	// <= tpb
	}

} // set_threadsPerBlock_pitch

// ---------------------------------------------

/* Helper function of initialize_GPU().
 *
 * Setups the factorization rank for this GPU device.
 *
 * Returns <factorization_rank + padding>, which is a multiple of <memory_alignment>,
 * or 0 on error.
 */
static index_t setup_gpu_factorization_rank( index_t factorization_rank )
{

	// factorization_rank + padding.
	index_t const pitch = get_padding( factorization_rank );

	if ( pitch > maxThreadsPerBlock ) {	// maxThreadsPerBlock << matrix_max_pitch
		print_error( error_shown_by_all, "\nSorry, but the given factorization rank (K=%" PRI_IDX "), exceeds the limits for\n"
				"this GPU device. Maximum allowed value: %" PRI_IDX ".\n", factorization_rank, maxThreadsPerBlock );
		return INDEX_C( 0 );
	}

	// ------------------

	// It also sets some related global variables.
	set_threadsPerBlock_pitch( pitch );

	// ------------------

	return pitch;

} // setup_gpu_factorization_rank

// ---------------------------------------------

/*
 * Initializes CUDA and cuBLAS on the specified device.
 *
 * Updates "memory_alignment", the limits of matrix dimensions, and setups the
 * padding for the given factorization rank (K).
 *
 * WARNING:
 *	- This function must be called *BEFORE* any other CUDA-related routine.
 *
 * Returns the amount of free global memory, or 0 on failure.
 */
size_t initialize_GPU( index_t dev_id, index_t factorization_rank )
{

	#if NMFGPU_VERBOSE
		print_message( dbg_shown_by_all, "Initializing CUDA/cuBLAS on device %" PRI_IDX " (process %" PRI_IDX ", total: %" PRI_IDX
				"), K=%" PRI_IDX "...\n", dev_id, process_id, num_processes, factorization_rank );
	#endif

	if ( (dev_id < 0) + (factorization_rank < 2) ) {
		errno = EINVAL;
		if ( dev_id < 0 ) print_errnum( error_shown_by_all, errno, "\ninitialize_GPU( device_ID=%" PRI_IDX " )", dev_id );
		if ( factorization_rank < 2 )
			print_errnum( error_shown_by_all, errno, "\ninitialize_GPU( factorization_rank=%" PRI_IDX " )", factorization_rank );
		return 0;
	}

	cudaError_t cuda_status = cudaSuccess;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;

	// -------------------------------------

	// Retrieves the properties of the selected device.

	struct cudaDeviceProp device_prop;
	cuda_status = cudaGetDeviceProperties( &device_prop, (int) dev_id );
	if ( cuda_status != cudaSuccess ) {
		print_error( sys_error_shown_by_all, "Could not retrieve properties of device %" PRI_IDX ": %s\n",
				dev_id, cudaGetErrorString(cuda_status) );
		cudaDeviceReset();
		return 0;
	}

	// -----------------------------------

	// Default flags for any device.

	// CPU thread spins/blocks on synchronization primitives (e.g., cudaDeviceSynchronize(), cudaStreamSynchronize()).
	unsigned int flags =

		#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || (! NMFGPU_PROFILING_GLOBAL)
			cudaDeviceScheduleBlockingSync;

		#else
			cudaDeviceScheduleSpin;
		#endif

	cuda_status = cudaSetDeviceFlags(flags);
	if ( cuda_status != cudaSuccess )
		print_error( sys_error_shown_by_all, "Warning: Could not set default flags for GPU devices: %s\n",
				cudaGetErrorString(cuda_status) );


	bool l_mappedHostMemory = false;
	#if ! NMFGPU_FORCE_BLOCKS
	{
		// If the device is integrated, maps host memory into the address space of the device (if it is able to).

		bool const is_integrated = device_prop.integrated;

		bool const canMapHostMemory = device_prop.canMapHostMemory;

		if ( is_integrated * canMapHostMemory ) {

			flags |= cudaDeviceMapHost;

			cuda_status = cudaSetDeviceFlags( flags );
			if ( cuda_status != cudaSuccess )
				print_error( sys_error_shown_by_all, "Warning: Could not set flag to map host memory into the address "
						"space of devices: %s\nTherefore, all buffers will be stored in device "
						"memory and data will be transferred there.\n",cudaGetErrorString(cuda_status) );
			else
				l_mappedHostMemory = true;
		}
	}
	#endif

	mappedHostMemory = l_mappedHostMemory;	// Global variable.

	// -----------------------------------

	// Attaches to the specified device.

	if ( (cuda_status=cudaSetDevice( dev_id )) != cudaSuccess ) {
		print_error( sys_error_shown_by_all, "Could not attach to device %" PRI_IDX ": %s\n", dev_id, cudaGetErrorString(cuda_status) );
		cudaDeviceReset();
		return 0;
	}

	// Makes sure this process is attached to the selected device.
	{
		int device = 0;
		cuda_status = cudaGetDevice( &device );
		if ( ( cuda_status != cudaSuccess ) + ( (index_t) device != dev_id ) ) {
			if ( cuda_status != cudaSuccess )
				print_error( sys_error_shown_by_all, "Error: Could not access to device %" PRI_IDX ": %s\n",
						dev_id, cudaGetErrorString(cuda_status) );
			else
				print_error( sys_error_shown_by_all, "Error: Process NOT attached to device %" PRI_IDX
						" (returned device ID: %i).\n", dev_id, device );
			cudaDeviceReset();
			return 0;
		}
	}

	// Global variables.
	device_id = dev_id;

	// -----------------------------------

	// Initializes cuBLAS on the selected device.

	cublas_status = cublasCreate( &cublas_handle );
	if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
		print_error( sys_error_shown_by_all, "Error: Could not initialize cuBLAS on device=%" PRI_IDX ": %s\n",
				dev_id, getCublasErrorString(cublas_status) );
		cudaDeviceReset();
		return 0;
	}

	// Sets pointer mode used by the cuBLAS library in scalar arguments.

	cublas_status = cublasSetPointerMode( cublas_handle, CUBLAS_POINTER_MODE_DEVICE );
	if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
		print_error( sys_error_shown_by_all, "Error: Could not set the pointer mode to be used by the cuBLAS library on device %"
				PRI_IDX ": %s\n", dev_id, getCublasErrorString(cublas_status) );
		cublasDestroy( cublas_handle );
		cudaDeviceReset();
		return 0;
	}

	// -----------------------------------

	/* Preferred cache configuration
	 *	FIXME: The "PreferL1" configuration might have negative effects on cuBLAS operations.
	 */
	cuda_status = cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
	if ( cuda_status != cudaSuccess ) {
		print_error( sys_error_shown_by_all, "Warning: Could not set the preferred cache configuration to device %" PRI_IDX ": %s\n",
				dev_id, cudaGetErrorString(cuda_status) );
	}

	// Shared memory configuration
	cuda_status = cudaDeviceSetSharedMemConfig( CUDA_SHARED_MEM_BANK_SIZE );
	if ( cuda_status != cudaSuccess ) {
		print_error( sys_error_shown_by_all, "Warning: Could not set the preferred shared memory configuration to device %"
				PRI_IDX ": %s\n", dev_id, cudaGetErrorString(cuda_status) );
	}

	// -------------------------------------

	// Sets some GPU properties

	// Compute Capability (major and minor)
	computeCapability = device_prop.major;
	computeCapability_minor = device_prop.minor;

	// Maximum number of threads per block.
	maxThreadsPerBlock = device_prop.maxThreadsPerBlock;

	// Number of multiprocessors.
	multiProcessorCount = device_prop.multiProcessorCount;

	// Maximum number of resident threads per multiprocessor.
	maxThreadsPerMultiProcessor = device_prop.maxThreadsPerMultiProcessor;

	// Maximum grid dimensions.
	maxGridSizeX = device_prop.maxGridSize[0];
	maxGridSizeY = device_prop.maxGridSize[1];

	// --------------------------------------

	/* Typical number of threads per block <= maxThreadsPerBlock.
	 *	It should be a divisor of maxThreadsPerMultiProcessor
	 *
	 * FIXME: Empirically set to maxThreadsPerMultiProcessor/3 or maxThreadsPerMultiProcessor/4.
	 *	E.g., 256 for Compute Capability 1.x, 512 for Compute Capability >= 2.0
	 */
	{
		// t1: 0, or (maxThreadsPerMultiProcessor/3)
		index_t const t1 = (device_prop.maxThreadsPerMultiProcessor % device_prop.maxThreadsPerBlock);

		// Makes sure that (maxThreadsPerMultiProcessor/4) <= maxThreadsPerBlock
		index_t const t2 = MIN( (device_prop.maxThreadsPerMultiProcessor/4), device_prop.maxThreadsPerBlock );

		threadsPerBlock = ( t1 ? t1 : t2 );	// (maxThreadsPerMultiProcessor/3), or (maxThreadsPerMultiProcessor/4).
	}

	// --------------------------------------

	// Available Device memory (bytes).

	size_t free_mem = 0, total_mem = 0;
	cuda_status = cudaMemGetInfo( &free_mem, &total_mem );
	if ( cuda_status != cudaSuccess ) {
		print_error( sys_error_shown_by_all, "Could not determine the amount of free memory on device %" PRI_IDX ": %s\n",
				dev_id, cudaGetErrorString(cuda_status) );
		cublasDestroy( cublas_handle );
		cudaDeviceReset();
		return 0;
	}

	// NOTE: Reduces 20% to leave place for any CUDA internal temporary buffer.
	size_t const l_mem_size = (float) free_mem * 0.8f;	// Value is truncated.

		// // Other values for debugging:
		//	l_mem_size = device_prop.totalGlobalMem;
		//	l_mem_size = 1000 * 1024 * 1024;	// 1000MBytes (< 1GBytes)
		//	l_mem_size =  512 * 1024 * 1024;	// 512MB
		//	l_mem_size = 1024 * 1024;		// 1MB

	// -------------------------------------

	// updates "memory_alignment", the limits of matrix dimensions, and the padded factorization rank.

	/* Alignment value for data to be stored in GPU/CPU memory.
	 * It is typically equal to <warpSize/2> on devices of Compute Capability 1.x, and <warpSize> on 2.x and beyond.
	 */
	memory_alignment = SET_MEMORY_ALIGNMENT( device_prop.major, device_prop.warpSize );

	// Updates the padded factorization rank and other related global variables.
	Kp = setup_gpu_factorization_rank( factorization_rank );
	if ( ! Kp ) {
		cublasDestroy( cublas_handle );
		cudaDeviceReset();
		return 0;
	}

	// Maximum "height" on GPU kernels. NOTE: It applies to both input-matrix dimensions.
	index_t l_max_dim = gpu_max_height( Kp );

	// Maximum number of items on GPU kernels (global variable).
	gpu_max_num_items = gpu_max_nitems();	// <= matrix_max_num_items

	/* Similarly to others BLAS-like libraries, cuBLAS makes use of SIGNED
	 * integers as function parameters (at least on its "regular" API).
	 * Therefore, matrix dimensions must be limited to 'INT_MAX', regardless
	 * of the signedness.
	 */
	l_max_dim = MIN( l_max_dim, ((index_t) INT_MAX) );

	/* On single-process systems with mapped host memory (e.g.,
	 * notebooks), there is no blockwise processing. Therefore, matrix size
	 * is limited by GPU kernels.
	 */
	size_t const l_max_nitems = gpu_max_num_items * mappedHostMemory * (num_processes == 1);

	// Finally, updates input-matrix limits for the current GPU device.
	if ( set_matrix_limits( memory_alignment, l_max_dim, l_max_nitems ) != EXIT_SUCCESS ) {
		print_error( error_shown_by_all, "Invalid data alignment (" PRI_IDX ") or matrix dimension limit (" PRI_IDX
				") on device %" PRI_IDX ".\n", memory_alignment, l_max_dim, dev_id );
		cublasDestroy( cublas_handle );
		cudaDeviceReset();
		return 0;
	}

	// -------------------------------------

	// Shows some GPU properties

	#if NMFGPU_VERBOSE || NMFGPU_DEBUG || NMFGPU_FORCE_BLOCKS || NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
	{
		// CUDA version
		int runtime_version = 0;
		cudaRuntimeGetVersion( &runtime_version );

		// cuBLAS version
		int cublas_version = 0;
		cublasGetVersion( cublas_handle, &cublas_version );

		print_message( verb_shown_by_all, "%s (device ID: %" PRI_IDX "): Compute_Capability=%i.%i, CUDA=%i.%i, cuBLAS=%i.%i\n"
				"\tWarp_size=%i, Memory_Alignment=%i, Max_Threads_per_Block=%i, Threads_per_Block=%i, "
				"Max_Grid_Dimensions(X,Y)=(%i,%i).\n"
				"\tMultiprocessors=%i, Threads_per_MultiProcessor=%i.\n"
				"\tGlobal_Memory=%zu bytes (%g MiB), Total_Memory=%g MiB, Free_Memory=%g MiB, Used=%g MiB, "
				"Maximum_Allocatable=%g MiB.\n"
				"\tDevice is integrated: %s, can map host memory: %s, host memory mapped: %s.\n",
				device_prop.name, dev_id, device_prop.major, device_prop.minor,
				runtime_version/1000, runtime_version%100, cublas_version/1000, cublas_version%100,
				device_prop.warpSize, memory_alignment, device_prop.maxThreadsPerBlock, threadsPerBlock,
				device_prop.maxGridSize[0], device_prop.maxGridSize[1],
				device_prop.multiProcessorCount, device_prop.maxThreadsPerMultiProcessor,
				device_prop.totalGlobalMem, (device_prop.totalGlobalMem/1048576.0f), (total_mem/1048576.0f),
				(free_mem/1048576.0f), ((total_mem-free_mem)/1048576.0f), (l_mem_size/1048576.0f),
				(device_prop.integrated ? "yes" : "no"), (device_prop.canMapHostMemory ? "yes" : "no"),
				(l_mappedHostMemory ? "yes" : "no") );
	}
	#endif

	// -------------------------------------

	#if NMFGPU_VERBOSE
		print_message( verb_shown_by_all, "Initializing CUDA/cuBLAS on device %" PRI_IDX " (process %" PRI_IDX ", total: %" PRI_IDX
				"), K=%" PRI_IDX "... done (mem_size=%zu).\n", dev_id, process_id, num_processes, factorization_rank, l_mem_size );
	#endif

	cudaGetLastError();	// Clears any possibly error flag.

	return l_mem_size;

} // initialize_GPU

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Computes the size of data blocks to be transferred to the GPU.
 *
 * Assuming that:
 *	N, M: Total number of rows and columns of input matrix V,
 *	Mp: Total number of columns of V, including useless data for padding,
 *		If num_processes == 1, then NpP==N and MpP==M.
 *	NpP, MpP: Dimensions of the block from V to be processed by this GPU,
 *	Kp: Factorization rank with data for padding.
 *
 * This routine computes BLN <= NpP and BLM <= MpP so it can be loaded into the GPU:
 *	One BLN x Mp block from matrix d_Vrow.
 *	One N x BLMp block from matrix d_Vcol ('BLMp' means 'BLM' with padding) .
 *	One MAX( BLN x Mp , N x BLMp ) block for matrix WH (WH == W*H).
 *	Matrix W (N x Kp).
 *	Matrix H (M x Kp).
 *	One <MAX(N,M) x Kp> block for matrix Aux (i.e., Waux or Haux).
 *	One Kp-length vector for accum (i.e., accum_h or accum_w).
 *	Two constant values (scalar arguments for the cuBLAS Library),
 * where BLMp is the padding for BLM (BLMp <= MpPp). It is a multiple of memory_alignment.
 *
 * IF there is enough GPU memory for all matrices at full size:
 *	Sets BLN=NpP, BLM=MpP, BLMp=MpPp and full_matrix='true'.
 * ELSE,
 *	full_matrix is set to 'false'.
 *	Computes BLN, BLM and BLMp for d_WH, d_Vr and d_Vc.
 *
 * do_classf: Set to 'true' if classification vector will be computed.
 */
static int get_BLs( size_t mem_size, bool do_classf, index_t *__restrict__ const BLN, index_t *__restrict__ const BLM,
		index_t *__restrict__ const BLMp, bool *__restrict__ const full_matrix )
{

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "get_BLs( num_processes=%" PRI_IDX ", mem_size=%zu, do_classf=%i, NpP=%" PRI_IDX ", MpP=%"
				PRI_IDX ", MpPp=%" PRI_IDX ", mappedHostMemory=%i )\n", num_processes, mem_size, do_classf, NpP, MpP, MpPp,
				mappedHostMemory );
	#endif


	// Maximum size of d_Vr and d_Vc that can be processed by kernels, regardless of "mem_size".
	size_t const max_nitems_dV = gpu_max_num_items;

	// Initial output values (i.e., input matrix is small enough to be fully loaded into the GPU memory).
	index_t lBLN = NpP;
	index_t lBLM = MpP;
	index_t lBLMp = MpPp;
	bool l_full_matrix = true;

	if ( mappedHostMemory )
		return EXIT_SUCCESS;

	// -----------------------------

	#if NMFGPU_FORCE_BLOCKS

		// Forces block sizes to be the half of dimensions:

		print_message( verb_shown_by_all, "\nForcing blocks size to the half of dimensions.\n" );

		lBLN = (NpP/2) + (NpP % 2);

		if ( MpPp != memory_alignment ) {	// i.e., (MpPp > memory_alignment)
			// Tries to set lBLM == lBLMp, if possible.
			lBLMp = get_padding( MpPp/2 );
			lBLM = MIN( lBLMp, MpP );
		} else {
			lBLM = (MpP/2) + (MpP % 2);
			lBLMp = MpPp;
		}

		l_full_matrix = false;

	#endif

	// -----------------------------

	/* Reduces BLN and BLM(p) if they are too large for kernels on this GPU.
	 *	MAX( lBLN*Mp, N*lBLMp ) <= max_nitems_dV
	 */
	{
		size_t const maxrows_Vr = max_nitems_dV / Mp;
		if ( (size_t) lBLN > maxrows_Vr ) {
			lBLN = maxrows_Vr;
			l_full_matrix = false;
		}

		size_t const maxcols_Vc = max_nitems_dV / N;
		if ( (size_t) lBLMp > maxcols_Vc ) {
			lBLMp = maxcols_Vc - (maxcols_Vc % memory_alignment);	// Previous multiple of <memory_alignment>.
			if ( lBLM > lBLMp ) {
				lBLM = lBLMp;
				l_full_matrix = false;
			}
		}
	}

	// -----------------------------

	/* Required memory for matrices V and WH, expressed in number of real-type items:
	 *
	 *	IF (! l_full_matrix  ||  num_processes > 1): BLN*Mp + N*BLMp + MAX(BLN*Mp , N*BLMp). // Vrow, Vcol and WH
	 *	ELSE (BLN==NpP==N && BLMp==MpPp==Mp): 2*BLN*BLMp // V and WH.
	 */
	uint_fast64_t required_mem;
	{
		size_t const nitems_Vrow = (size_t) lBLN * (size_t) Mp;		// d_Vrow
		size_t const nitems_Vcol = (size_t) N * (size_t) lBLMp;		// d_Vcol
		size_t const nitems_WH = MAX( nitems_Vrow, nitems_Vcol );	// d_WH (if necessary)
		required_mem = (uint_fast64_t) nitems_Vrow + (uint_fast64_t) nitems_Vcol;

		if ( (! l_full_matrix) + (num_processes > 1) )	// (! l_full_matrix  ||  num_processes > 1)
			required_mem += nitems_WH;			// d_Vcol != d_Vrow. Therefore, we need d_WH.
	}

	/* Required memory, for matrices d_W, d_H, d_Aux, d_accum, and the two
	 * scalar values, expressed in number of real-type items.
	 */
	size_t data_matrices = (size_t) Kp * (size_t)(N + M + MAX(N,M)) + Kp + 2;

	// Memory used by classification vector (<Mp> integers) must be scaled since sizeof(index_t) <= sizeof(real)
	if ( do_classf ) {
		size_t const classf_data = (Mp * sizeof(index_t)) / sizeof(real);
		data_matrices += classf_data;
	}

	#if NMFGPU_FORCE_BLOCKS || NMFGPU_VERBOSE_2 || NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
		print_message( verb_shown_by_all, "Total device memory required (approx.): %g MiB.\n"
				"Required by output and auxiliary data: %g MiB.\n",
				((float) required_mem + data_matrices) * ((float) sizeof(real) / 1048576.0f),
				(float) data_matrices * ((float) sizeof(real) / 1048576.0f) );
	#endif

	// ---------------------------------


	// Adjusts (i.e, reduces) lBLN and lBLM according to the available memory.

	if ( (required_mem + data_matrices) > (uint_fast64_t) mem_size ) {

		// full_matrix is set to 'false'.
		l_full_matrix = false;

		/* Memory for "data_matrices" is always required.
		 * Therefore, subtracts its size from "mem_size".
		 */

		// Fails if data_matrices >= mem_size
		if ( data_matrices >= mem_size ) {
			// Minimum required: (lBLN = 1) and (lBLMp = memory_alignment).
			size_t const nitems_Vcol = (size_t) N * (size_t) memory_alignment;	// N * BLMp
			size_t const nitems_Vrow = Mp;						// lBLN * Mp
			required_mem = MAX( nitems_Vrow, nitems_Vcol ) + (uint_fast64_t) nitems_Vrow + (uint_fast64_t) nitems_Vcol;
			print_error( error_shown_by_all, "Not enough memory. Minimum required: %g MiB.\n",
					(float)(required_mem + data_matrices) * ((float)sizeof(real)/1048576.0f) );
			return EXIT_FAILURE;
		}

		size_t const free_memory = mem_size - data_matrices;

		#if NMFGPU_VERBOSE_2 || NMFGPU_FORCE_BLOCKS
			print_message( verb_shown_by_all, "mem_size=%zu, data_matrices=%zu, required_mem=%" PRIuFAST64 ", free_memory=%zu\n",
					mem_size, data_matrices, required_mem, free_memory );
		#endif

		// ----------------------------

		// //////////////////////////
		//
		// Required Memory: max(BLN*Mp, N*BLMp) + BLN*Mp + N*BLMp,
		//	where  BLN = (NpP/dBLN) and  BLMp = padding(MpP/dBLM)
		//
		// Increments iteratively the denominators 'dBLN' and 'dBLM'
		// (one at a time, according to 'dim') until data_matrices fit into the memory.
		//
		// WARNING: if (MpP > memory_alignment), the resulting BLM must be a multiple of memory_alignment.
		//
		// It also checks that the resulting values don't exceed the maximum matrix size.
		//
		// //////////////////////////

		index_t dBLN = 2;	// Denominator for dimension N
		index_t dBLM = 2;	// Denominator for dimension M
		bool dimN = true;	// True: Adjusts BLN, False: Adjusts BLM

		size_t rows = (size_t) lBLN * (size_t) Mp;	// d_Vrow: lBLN * Mp;

		// Initial values for dBLM == 2
		lBLMp = get_padding( (MpPp / 2) );		// Multiple of memory_alignment
		size_t cols = (size_t) N * (size_t) lBLMp;	// (d_Vcol)
		lBLM = MIN( lBLMp, MpP );

		#if NMFGPU_VERBOSE_2
			index_t step = 1;	// Number of loops.
		#endif

		do {
			if ( dimN ) {	// Adjusts BLN
				lBLN = (NpP / dBLN) + ((NpP % dBLN) != 0);
				rows = (size_t) lBLN * (size_t) Mp;
				dBLN++;		// Denominator in next loop with dim=1.
			} else {	// Adjusts BLM
				lBLMp = (MpPp / dBLM) + ((MpPp % dBLM) != 0);
				lBLMp = get_padding( lBLMp );		// Multiple of memory_alignment
				cols = (size_t) N * (size_t) lBLMp;	// (d_Vcol)
				lBLM = MIN( lBLMp, MpP );
				dBLM++;		// Denominator in next loop with dim=0.
			}

			// Changes dimension for next loop.
			dimN = ( ! dimN );

			// Required memory:
			// size(d_Vr) = BLN * Mp	// (i.e., BLN rows)
			// size(d_Vc) = N * BLMp	// (i.e., BLM padded columns)
			// size(d_WH) = Max(size_Vr, size_Vc)

			size_t const nitems_WH = MAX( rows, cols );
			required_mem = (uint_fast64_t) rows + (uint_fast64_t) cols + (uint_fast64_t) nitems_WH;

			#if NMFGPU_VERBOSE_2
				print_message( verb_shown_by_all, "Step %" PRI_IDX ": BLN=%" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%" PRI_IDX
						" (%g MiB), dBLN=%" PRI_IDX ", dBLM=%" PRI_IDX ", required_mem=%" PRIuFAST64 "\n",
						step, lBLN, lBLM, lBLMp, (float)(required_mem + data_matrices) *
						((float)sizeof(real) / 1048576.0f), dBLN, dBLM, required_mem );
				step++;
			#endif

		} while ( (required_mem > (uint_fast64_t) free_memory) * (lBLMp > memory_alignment) * (lBLN > 0) );

		if ( required_mem > (uint_fast64_t) free_memory ) {

			if ( lBLMp == memory_alignment ) {	// BLN >= 0

				/* Fixes lBLMp to the minimum value (<memory_alignment>).
				 * Then, computes lBLN from:
				 *	rows + cols + max(rows, cols) <= free_mem
				 *
				 * In addition, makes sure that:
				 *	+ max(rows, cols) <= max_nitems_dV
				 *	+ lBLN <= NpP
				 */

				lBLMp = memory_alignment;
				lBLM = MIN( memory_alignment, MpP );	// i.e., MIN( lBLMp, MpP )
				cols = (size_t) N * (size_t) memory_alignment;	//  N * lBLMp
				rows = (size_t) lBLN * (size_t) Mp;

				// Maximum values for BLN
				size_t max_dim = MIN( (max_nitems_dV / Mp), (size_t) NpP );

				if ( rows >= cols )		// required_memory == (2*rows + cols) == (2*BLN*Mp + N*lBLMp) <= free_memory
					lBLN = (free_memory - cols) / (2 * Mp);
				else				// required_memory == (rows + 2*cols) == (BLN*Mp + 2*N*lBLMp) <= free_memory
					lBLN = (free_memory - (2*cols)) / Mp;

				if ( (size_t) lBLN > max_dim )
					lBLN = max_dim;

			} else {	// (lBLMp > memory_alignment) && (lBLN == 0)

				/* Fixes lBLN to '1'.
				 * Then, computes lBLM and lBLMp from:
				 *	rows + cols + max(rows, cols) <= free_mem
				 *
				 * In addition, makes sure that:
				 *	+ max(rows, cols) <= max_MatrixSize
				 *	+ lBLM <= MpP
				 *	+ lBLMp = get_padding( lBLM ) <= MpPp
				 */

				lBLN = 1;
				rows = Mp;	// BLN * Mp
				cols = N * lBLMp;

				// Maximum values for BLMp
				size_t const max_dim = MIN( (max_nitems_dV / N), (size_t) MpPp );

				if ( rows >= cols )	// required_memory == (2*rows + cols) == (2*Mp + N*BLMp) <= free_mem
					lBLMp = (free_memory - (2*Mp)) / N;
				else			// required_memory == (rows + 2*cols) == (Mp + 2*N*BLMp) <= free_mem
					lBLMp = (free_memory - Mp) / (2*N);

				lBLMp = MIN( (size_t) lBLMp, max_dim );

				// Makes sure lBLMp is a multiple of <memory_alignment>
				lBLMp -= (lBLMp % memory_alignment);	// Previous multiple of <memory_alignment>

				// Finally, sets lBLM
				lBLM = MIN( lBLMp, MpP );
			}

			#if NMFGPU_VERBOSE_2
			{
				rows = (size_t) lBLN * (size_t) Mp;
				cols = (size_t) N * (size_t) lBLMp;
				size_t const nitems_WH = MAX( rows, cols );
				required_mem = (uint_fast64_t)rows + (uint_fast64_t)cols + (uint_fast64_t)nitems_WH;

				print_message( verb_shown_by_all, "Resulting values: BLN=%" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%" PRI_IDX
						", required_mem (approx.): %g MiB\n", lBLN, lBLM, lBLMp,
						(float) (required_mem + data_matrices) * (sizeof(real) / 1048576.0f) );
			}
			#endif

			if ( (lBLN < 1) + (lBLM < 1) ) {
				// Minimum required: lBLN=1 && lBLMp=memory_alignment.
				cols = (size_t) N * (size_t) memory_alignment;	// N * BLMp
				rows = Mp;			// lBLN * Mp
				required_mem = (uint_fast64_t) MAX( rows, cols ) + (uint_fast64_t) rows + (uint_fast64_t) cols;
				print_error( error_shown_by_all, "Not enough memory. Minimum required: %g MiB.\n",
						(float) ((required_mem + data_matrices) * sizeof(real))/1048576.0f);
				return EXIT_FAILURE;
			}

		} // if ( required_mem > free_mem )

	} // if ( required_mem > free_mem )


	#if NMFGPU_FORCE_BLOCKS || NMFGPU_VERBOSE || NMFGPU_VERBOSE_2 || NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
		print_message( verb_shown_by_all, "Resulting values: BLN=%" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%" PRI_IDX
				" (approx. %g MiB), full_matrix=%i\n", lBLN, lBLM, lBLMp,
				(float) (required_mem + data_matrices) * (sizeof(real) / 1048576.0f), l_full_matrix );
	#endif

	*BLN = lBLN;
	*BLM = lBLM;
	*BLMp = lBLMp;
	*full_matrix = l_full_matrix;

	return EXIT_SUCCESS;

} // get_BLs

// ---------------------------------------------

/*
 * Initializes a "block_D" data (D == NpP or MpP).
 *
 * Data for blockwise processing is stored in a block_t structure. This method initializes such structure.
 *
 * BLDp <= Dp
 *
 * Divides dimension "D" in <num_steps> blocks of length "BLD" and possibly, one more block of length <D % BLD>.
 * In any case, the length information of the last block (either BLD or <D % BLD>) is always stored in block_D.BL[1],
 * EXCEPT if D == BLD (in this case, it is only stored in block_D.BL[0], and sets block_D.num_steps[0..1] to {1,0}).
 */
static void init_block_conf( index_t D, index_t Dp, index_t BLD, index_t BLDp, block_t *__restrict__ const block_D )
{

	block_D->BL[0] = BLD;
	block_D->BLp[0] = BLDp;

	if ( BLDp != Dp ) { // BlDp < Dp, blockwise processing

		/* Divides dimension "D" in <num_steps> blocks of length <BLD>, and possibly one more block of length <D % BLD>.
		 * In any case, the length information of the last block (either, <BLD> or <D % BLD>), is always stored in BL[1].
		 */

		// Number of blocks in BL[0]
		block_D->num_steps[0] = (Dp / BLDp) - (! (Dp % BLDp));	// "-1" if (Dp % BLDp) == 0.

		// Last block
		block_D->BL[1] = ((D % BLDp) ? (D % BLDp) : BLD);
		block_D->BLp[1] = ((Dp % BLDp) ? (Dp % BLDp) : BLDp);	// Already a multiple of memory_alignment
		block_D->num_steps[1] = 1;

	} else {  // D == BLD : There is only one block (BL[0])

		block_D->BL[1] = 0;
		block_D->BLp[1] = 0;
		block_D->num_steps[0] = 1;
		block_D->num_steps[1] = 0;

	} // if (D > BLD)

	// Prints block configuration
	#if NMFGPU_FORCE_BLOCKS || NMFGPU_VERBOSE || NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
		append_printed_message( dbg_shown_by_all, "\t%" PRI_IDX " block(s) of length %" PRI_IDX " (padding=%" PRI_IDX ")\n",
				block_D->num_steps[0], BLD, BLDp);
		if ( block_D->num_steps[1] )
			append_printed_message( dbg_shown_by_all, "\t1 block of length %" PRI_IDX " (padding=%" PRI_IDX ")\n",
					block_D->BL[1], block_D->BLp[1]);
	#endif

} // init_block_conf

// ---------------------------------------------

/*
 * Allocates memory for data matrices.
 *
 * If mappedHostMemory is 'true', maps the host memory that was previously allocated for data matrices,
 * into the address space of the device. Memory for temporary buffers (e.g., d_WH or d_Aux) is first
 * allocated on the host, and then mapped.
 *
 * single_matrix_V: 'True' if input matrix V is small enough to be fully loaded into the GPU memory,
 *		    <AND> this is a single-GPU system.
 *		    In such case, d_Vrow = d_Vcol.
 *
 * do_classf: Set to 'true' if classification vector will be computed.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int allocate_memory( index_t BLN, index_t BLMp, bool single_matrix_V, bool do_classf )
{

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Allocating memory (BLN=%" PRI_IDX ", BLMp=%" PRI_IDX
				", single_matrix_V: %i, do_classf: %i, mappedHostMemory: %i)...\n",
				BLN, BLMp, single_matrix_V, do_classf, mappedHostMemory );
	#endif

	cudaError_t cuda_status = cudaSuccess;

	size_t const sizeVr = (size_t) BLN * (size_t) Mp;
	size_t const sizeVc = (size_t) N * (size_t) BLMp;
	size_t const sizeWH = MAX( sizeVr, sizeVc );

	size_t const sizeW = (size_t) N * (size_t) Kp;
	size_t const sizeH = (size_t) M * (size_t) Kp;
	size_t const sizeAux = MAX( sizeW, sizeH );

	bool const clear_memory = false;	// Do NOT clear allocated memory.
	bool const write_combined = false;	// No write-combined memory.

	// ---------------------------------

	if ( mappedHostMemory ) {	// Maps host memory into address space of the device.

		// d_Vrow: Maps host memory.
		cuda_status = cudaHostGetDevicePointer( (void **)&d_Vrow, (void *)Vrow, 0 );
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Host-memory mapping error (Vrow -> d_Vrow): %s\n",
					cudaGetErrorString(cuda_status) );
			return EXIT_FAILURE;
		}

		// d_Vcol: Maps host memory
		if ( single_matrix_V )		// Fully allocates matrix V (i.e., 1 Process/GPU and full_matrix).
			d_Vcol = d_Vrow;	// Just shares the allocated memory.

		else { // Multiple processes OR Matrix V is too large for this GPU.
			cuda_status = cudaHostGetDevicePointer( (void **)&d_Vcol, (void *)Vcol, 0 );
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "Host-memory mapping error (Vcol -> d_Vcol): %s\n",
						cudaGetErrorString(cuda_status) );
				return EXIT_FAILURE;
			}
		}


		// d_WH: (BLN x Mp) OR (N x BLMp), Allocates and maps host memory
		h_WH = (real *) getHostMemory( sizeWH * sizeof(real), write_combined, clear_memory );
		if ( ! h_WH ) {
			print_error( sys_error_shown_by_all, "Error allocating memory for HOST matrix h_WH (mapped-memory mode).\n" );
			return EXIT_FAILURE;
		}
		cuda_status = cudaHostGetDevicePointer( (void **)&d_WH, (void *)h_WH, 0 );
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Host-memory mapping error (h_WH -> d_WH): %s\n",
					cudaGetErrorString(cuda_status) );
			freeHostMemory( (void *)h_WH, "h_WH, mapped-memory mode" );
			return EXIT_FAILURE;
		}

		// ---------------------------------

		// d_W: Maps host memory.
		cuda_status = cudaHostGetDevicePointer( (void **)&d_W, (void *)W, 0 );
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Host-memory mapping error (W -> d_W): %s\n",
					cudaGetErrorString(cuda_status) );
			freeHostMemory( (void *)h_WH, "h_WH, mapped-memory mode" );
			return EXIT_FAILURE;
		}


		// d_H: Maps host memory.
		cuda_status = cudaHostGetDevicePointer( (void **)&d_H, (void *)H, 0 );
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Host-memory mapping error (H -> d_H): %s\n",
					cudaGetErrorString(cuda_status) );
			freeHostMemory( (void *)h_WH, "h_WH, mapped-memory mode" );
			return EXIT_FAILURE;
		}


		/* d_Aux: N x Kp (i.e., Waux) <OR> M x Kp (i.e., Haux).
		 * Actually, it just uses <MAX(BLN,BLM) * Kp>, but matrix_to_row() might require up to <MAX(N,M) * Kp>.
		 * Allocates and maps host memory.
		 */
		h_Aux = (real *) getHostMemory( sizeAux * sizeof(real), write_combined, clear_memory );
		if ( ! h_Aux ) {
			print_error( sys_error_shown_by_all, "Error allocating memory for HOST matrix h_Aux (mapped-memory mode).\n" );
			freeHostMemory( (void *)h_WH, "h_WH, mapped-memory mode" );
			return EXIT_FAILURE;
		}
		cuda_status = cudaHostGetDevicePointer( (void **)&d_Aux, (void *)h_Aux, 0 );
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Host-memory mapping error (h_Aux -> d_Aux): %s\n",
					cudaGetErrorString(cuda_status) );
			freeHostMemory( (void *)h_Aux, "h_Aux, mapped-memory mode" );
			freeHostMemory( (void *)h_WH, "h_WH, mapped-memory mode" );
			return EXIT_FAILURE;
		}

		// ---------------------------------

		// Classification vectors (if necessary): Maps host memory
		if ( do_classf ) {
			cuda_status = cudaHostGetDevicePointer( (void **)&d_classification, (void *)classification, 0 );
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "Host-memory mapping error (classification -> d_classification): %s\n",
						cudaGetErrorString(cuda_status) );
				freeHostMemory( (void *)h_Aux, "h_Aux, mapped-memory mode" );
				freeHostMemory( (void *)h_WH, "h_WH, mapped-memory mode" );
				return EXIT_FAILURE;
			}
			cuda_status = cudaHostGetDevicePointer( (void **)&d_last_classification, (void *)last_classification, 0 );
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "Host-memory mapping error (last_classification -> "
						"d_last_classification): %s\n", cudaGetErrorString(cuda_status) );
				freeHostMemory( (void *)h_Aux, "h_Aux, mapped-memory mode" );
				freeHostMemory( (void *)h_WH, "h_WH, mapped-memory mode" );
				return EXIT_FAILURE;
			}
		}

		// ---------------------------------

		// d_accum (Kp-length vector): Allocates and maps host memory
		h_accum = (real *) getHostMemory( Kp * sizeof(real), write_combined, clear_memory );
		if ( ! h_accum ) {
			print_error( sys_error_shown_by_all, "Error allocating memory for HOST matrix h_accum[ Kp = %" PRI_IDX
					"] (mapped-memory mode).\n", Kp );
			freeHostMemory( (void *)h_Aux, "h_Aux, mapped-memory mode" );
			freeHostMemory( (void *)h_WH, "h_WH, mapped-memory mode" );
			return EXIT_FAILURE;
		}
		cuda_status = cudaHostGetDevicePointer( (void **)&d_accum, (void *)h_accum, 0 );
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Host-memory mapping error (h_accum -> d_accum): %s\n",
					cudaGetErrorString(cuda_status) );
			freeHostMemory( (void *)h_accum, "h_accum, mapped-memory mode" );
			freeHostMemory( (void *)h_Aux, "h_Aux, mapped-memory mode" );
			freeHostMemory( (void *)h_WH, "h_WH, mapped-memory mode" );
			return EXIT_FAILURE;
		}

		// ---------------------------------

		// Scalar values for cuBLAS Library: Allocates and maps host memory
		h_scalar = (real const *) getHostMemory( 2 * sizeof(real), write_combined, clear_memory );
		if ( ! h_scalar ) {
			print_error( sys_error_shown_by_all, "Error allocating memory for HOST matrix scalar[2] (mapped-memory mode).\n" );
			freeHostMemory( (void *)h_accum, "h_accum, mapped-memory mode" );
			freeHostMemory( (void *)h_Aux, "h_Aux, mapped-memory mode" );
			freeHostMemory( (void *)h_WH, "h_WH, mapped-memory mode" );
			return EXIT_FAILURE;
		}
		cuda_status = cudaHostGetDevicePointer( (void **)&d_scalar, (void *)h_scalar, 0 );
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Host-memory mapping error (h_scalar -> d_scalar): %s\n",
					cudaGetErrorString(cuda_status) );
			freeHostMemory( (void *)h_scalar, "h_scalar, mapped-memory mode" );
			freeHostMemory( (void *)h_accum, "h_accum, mapped-memory mode" );
			freeHostMemory( (void *)h_Aux, "h_Aux, mapped-memory mode" );
			freeHostMemory( (void *)h_WH, "h_WH, mapped-memory mode" );
			return EXIT_FAILURE;
		}

	} else {	// Allocates DEVICE memory

		// d_Vrow:
		cuda_status = cudaMalloc( (void **)&d_Vrow, sizeVr * sizeof(real));
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Device memory allocation error (d_Vrow): %s\n",
					cudaGetErrorString(cuda_status) );
			return EXIT_FAILURE;
		}

		// d_Vcol:
		if ( single_matrix_V )		// Fully allocates matrix V (i.e., 1 Process/GPU and full_matrix).
			d_Vcol = d_Vrow;	// Just shares the allocated memory.

		else { // Multiple processes OR Matrix V is too large for this GPU.
			cuda_status = cudaMalloc( (void **)&d_Vcol, sizeVc * sizeof(real));
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "Device memory allocation error (d_Vcol): %s\n",
						cudaGetErrorString(cuda_status) );
				cudaFree(d_Vrow);
				return EXIT_FAILURE;
			}
		}


		// d_WH: (BLN x Mp) OR (N x BLMp)
		cuda_status = cudaMalloc( (void **)&d_WH, sizeWH * sizeof(real));
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Device memory allocation error (d_WH): %s\n",
					cudaGetErrorString(cuda_status) );
			if (d_Vcol != d_Vrow){ cudaFree(d_Vcol); } cudaFree(d_Vrow);
			return EXIT_FAILURE;
		}

		// ---------------------------------

		// d_W
		cuda_status = cudaMalloc( (void **)&d_W, sizeW * sizeof(real));
		if( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Device memory allocation error (d_W): %s\n",
					cudaGetErrorString(cuda_status) );
			cudaFree(d_WH); if (d_Vcol != d_Vrow){ cudaFree(d_Vcol); } cudaFree(d_Vrow);
			return EXIT_FAILURE;
		}

		// d_H (transp)
		cuda_status = cudaMalloc( (void **)&d_H, sizeH * sizeof(real));
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Device memory allocation error (d_H): %s\n",
					cudaGetErrorString(cuda_status) );
			cudaFree(d_W); cudaFree(d_WH); if (d_Vcol != d_Vrow){ cudaFree(d_Vcol); } cudaFree(d_Vrow);
			return EXIT_FAILURE;
		}

		/* d_Aux: N x Kp (i.e., Waux) <OR> M x Kp (i.e., Haux).
		 * Actually, it just uses <MAX(BLN,BLM) * Kp>, but matrix_to_row() might require up to <MAX(N,M) * Kp>.
		 */
		cuda_status = cudaMalloc( (void **)&d_Aux, sizeAux * sizeof(real));
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Device memory allocation error (d_Aux): %s\n",
					cudaGetErrorString(cuda_status) );
			cudaFree(d_H); cudaFree(d_W); cudaFree(d_WH); if (d_Vcol != d_Vrow){ cudaFree(d_Vcol); } cudaFree(d_Vrow);
			return EXIT_FAILURE;
		}

		// ---------------------------------

		// Classification vector (if necessary)
		if ( do_classf ) {
			cuda_status = cudaMalloc( (void **)&d_classification, Mp * sizeof(index_t) );
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "Device memory allocation error (classification): %s\n",
						cudaGetErrorString(cuda_status) );
				cudaFree(d_Aux); cudaFree(d_H); cudaFree(d_W); cudaFree(d_WH);
				if (d_Vcol != d_Vrow){ cudaFree(d_Vcol); } cudaFree(d_Vrow);
				return EXIT_FAILURE;
			}
		}

		// ---------------------------------

		// d_accum
		cuda_status = cudaMalloc( (void **)&d_accum, Kp * sizeof(real) );
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Device memory allocation error (d_accum): %s\n",
					cudaGetErrorString(cuda_status) );
			if ( do_classf ){ cudaFree(d_classification); }
			cudaFree(d_Aux); cudaFree(d_H); cudaFree(d_W); cudaFree(d_WH);
			if (d_Vcol != d_Vrow){ cudaFree(d_Vcol); } cudaFree(d_Vrow);
			return EXIT_FAILURE;
		}

		// ---------------------------------

		// Scalar values for cuBLAS Library.
		cuda_status = cudaMalloc( (void **)&d_scalar, 2 * sizeof(real) );
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Device memory allocation error (d_scalar): %s\n",
					cudaGetErrorString(cuda_status) );
			cudaFree(d_accum);
			if ( do_classf ){ cudaFree(d_classification); }
			cudaFree(d_Aux); cudaFree(d_H); cudaFree(d_W); cudaFree(d_WH);
			if (d_Vcol != d_Vrow){ cudaFree(d_Vcol); } cudaFree(d_Vrow);
			return EXIT_FAILURE;
		}

	} // If host memory was mapped.

	// ---------------------------------

	// Sets pointers to d_scalar[]

	d_zero = &d_scalar[0];
	d_one  = &d_scalar[1];

	// ---------------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Allocating memory for the matrices... done\n" );
	#endif

	return EXIT_SUCCESS;

} // allocate_memory

// ---------------------------------------------

/*
 * Initializes associated GPU data, such as CUDA events and streams.
 *
 * If host memory was mapped, skips streams and events related to matrices d_Vrow and d_Vcol.
 *
 * single_matrix_V: Set to 'true' if NO blockwise processing is necessary (i.e., input matrix is small enough to be fully loaded),
 *		so that Vrow == Vcol.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int init_GPU_data( bool single_matrix_V )
{

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Initializing GPU data (single matrix=%i, num_streams_NMF=%" PRI_IDX ")...\n",
				single_matrix_V, num_streams_NMF );
	#endif

	cudaError_t cuda_status = cudaSuccess;

	// ----------------------------------

	/* CUDA Events for synchronization. */

	if ( ! mappedHostMemory ) {	// Host memory is NOT mapped.

		// Event for Vrow (i.e., set of rows from matrix V)
		cuda_status = cudaEventCreateWithFlags( &event_Vrow, cudaEventDisableTiming );
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Error creating event object for Vrow: %s\n",
					cudaGetErrorString(cuda_status) );
			return EXIT_FAILURE;
		}


		// Event for Vcol (i.e., set of columns from matrix V)
		if ( single_matrix_V )
			event_Vcol = event_Vrow;	// Same matrix (Vcol == Vrow), same events

		else {
			cuda_status = cudaEventCreateWithFlags( &event_Vcol, cudaEventDisableTiming );
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "Error creating event object for Vcol: %s\n",
						cudaGetErrorString(cuda_status) );
				cudaEventDestroy( event_Vrow );
				return EXIT_FAILURE;
			}
		} // if ( single_matrix_V )

	} // if ! mappedHostMemory

	// Event for d_W
	cuda_status = cudaEventCreateWithFlags( &event_W, cudaEventDisableTiming );
	if ( cuda_status != cudaSuccess ) {
		print_error( sys_error_shown_by_all, "Error creating event object for d_W: %s\n", cudaGetErrorString(cuda_status) );
		if (! mappedHostMemory){ if(! single_matrix_V){ cudaEventDestroy(event_Vcol); } cudaEventDestroy(event_Vrow); }
		return EXIT_FAILURE;
	}

	// Event for d_H.
	cuda_status = cudaEventCreateWithFlags( &event_H, cudaEventDisableTiming );
	if ( cuda_status != cudaSuccess ) {
		print_error( sys_error_shown_by_all, "Error creating event object for d_H: %s\n", cudaGetErrorString(cuda_status) );
		cudaEventDestroy( event_W );
		if (! mappedHostMemory){ if(! single_matrix_V){ cudaEventDestroy(event_Vcol); } cudaEventDestroy(event_Vrow); }
		return EXIT_FAILURE;
	}

	// Event for matrix reduction.
	cuda_status = cudaEventCreateWithFlags( &event_reduction, cudaEventDisableTiming );
	if ( cuda_status != cudaSuccess ) {
		print_error( sys_error_shown_by_all, "Error creating event object for matrix reduction: %s\n",
				cudaGetErrorString(cuda_status) );
		cudaEventDestroy( event_H ); cudaEventDestroy( event_W );
		if (! mappedHostMemory){ if(! single_matrix_V){ cudaEventDestroy(event_Vcol); } cudaEventDestroy(event_Vrow); }
		return EXIT_FAILURE;
	}

	// ---------------------------------

	/* CUDA Streams for synchronization */

	if ( ! mappedHostMemory ) {	// Host memory is NOT mapped.

		// Stream for Vrow (i.e., set of rows from matrix V)
		cuda_status = cudaStreamCreate( &stream_Vrow );
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Error creating stream object for Vrow: %s\n",
					cudaGetErrorString(cuda_status) );
			cudaEventDestroy( event_reduction ); cudaEventDestroy( event_H ); cudaEventDestroy( event_W );
			if ( ! single_matrix_V ) { cudaEventDestroy( event_Vcol ); } cudaEventDestroy( event_Vrow );
			return EXIT_FAILURE;
		}


		// Streams for Vcol (i.e., set of columns from matrix V)
		if ( single_matrix_V )
			stream_Vcol = stream_Vrow;	// Same matrix (Vcol == Vrow), same streams

		else {
			cuda_status = cudaStreamCreate( &stream_Vcol );
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "Error creating stream object for Vcol: %s\n",
						cudaGetErrorString(cuda_status) );
				cudaStreamDestroy( stream_Vrow );
				cudaEventDestroy( event_reduction ); cudaEventDestroy( event_H ); cudaEventDestroy( event_W );
				cudaEventDestroy( event_Vcol ); cudaEventDestroy( event_Vrow );
				return EXIT_FAILURE;
			}
		} // if ( single_matrix_V )

	} // if ! mappedHostMemory

	// Stream for d_W
	cuda_status = cudaStreamCreate( &stream_W );
	if ( cuda_status != cudaSuccess ) {
		print_error( sys_error_shown_by_all, "Error creating stream object for d_W: %s\n", cudaGetErrorString(cuda_status) );
		if (! mappedHostMemory){ if (! single_matrix_V){ cudaStreamDestroy(stream_Vcol); } cudaStreamDestroy(stream_Vrow); }
		cudaEventDestroy( event_reduction ); cudaEventDestroy( event_H ); cudaEventDestroy( event_W );
		if (! mappedHostMemory){ if(! single_matrix_V){ cudaEventDestroy(event_Vcol); } cudaEventDestroy(event_Vrow); }
		return EXIT_FAILURE;
	}

	// Stream for d_W
	cuda_status = cudaStreamCreate( &stream_H );
	if ( cuda_status != cudaSuccess ) {
		print_error( sys_error_shown_by_all, "Error creating stream object for d_H: %s\n", cudaGetErrorString(cuda_status) );
		cudaStreamDestroy( stream_W );
		if (! mappedHostMemory){ if (! single_matrix_V){ cudaStreamDestroy(stream_Vcol); } cudaStreamDestroy(stream_Vrow); }
		cudaEventDestroy( event_reduction ); cudaEventDestroy( event_H ); cudaEventDestroy( event_W );
		if (! mappedHostMemory){ if(! single_matrix_V){ cudaEventDestroy(event_Vcol); } cudaEventDestroy(event_Vrow); }
		return EXIT_FAILURE;
	}

	// Main-flow streams
	streams_NMF = (cudaStream_t *) malloc( num_streams_NMF * sizeof(cudaStream_t) );
	if( ! streams_NMF ) {
		print_errnum( sys_error_shown_by_all, errno, "Error allocating HOST memory for CUDA streams (Main flow, length=%"
				PRI_IDX ")", num_streams_NMF );
		cudaStreamDestroy( stream_H ); cudaStreamDestroy( stream_W );
		if (! mappedHostMemory){ if (! single_matrix_V){ cudaStreamDestroy(stream_Vcol); } cudaStreamDestroy(stream_Vrow); }
		cudaEventDestroy( event_reduction ); cudaEventDestroy( event_H ); cudaEventDestroy( event_W );
		if (! mappedHostMemory){ if(! single_matrix_V){ cudaEventDestroy(event_Vcol); } cudaEventDestroy(event_Vrow); }
		return EXIT_FAILURE;
	}
	for ( index_t st=0 ; st < num_streams_NMF ; st++ ) {
		cuda_status = cudaStreamCreate( &streams_NMF[ st ] );
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Error creating stream object %" PRI_IDX "/%" PRI_IDX
					" for synchronization on main flow: %s\n", st, num_streams_NMF,
					cudaGetErrorString(cuda_status) );
			for ( index_t i=0 ; i < st ; i++ ) { cudaStreamDestroy( streams_NMF[ i ] ); } free( (void *) streams_NMF );
			cudaStreamDestroy( stream_H ); cudaStreamDestroy( stream_W );
			if (! mappedHostMemory){ if (! single_matrix_V){ cudaStreamDestroy(stream_Vcol); } cudaStreamDestroy(stream_Vrow); }
			cudaEventDestroy( event_reduction ); cudaEventDestroy( event_H ); cudaEventDestroy( event_W );
			if (! mappedHostMemory){ if(! single_matrix_V){ cudaEventDestroy(event_Vcol); } cudaEventDestroy(event_Vrow); }
			return EXIT_FAILURE;
		}
	}

	// -----------------------------------

	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS

		/* Timers for matrix operations */

		init_kernel_timers();

		init_transfer_timers();

		/* CUDA Events for timing */
		if ( init_timing_events() != EXIT_SUCCESS ) {
			for ( index_t st=0; st<num_streams_NMF; st++ ) { cudaStreamDestroy( streams_NMF[ st ] ); } free( (void *) streams_NMF );
			cudaStreamDestroy( stream_H ); cudaStreamDestroy( stream_W );
			if (! mappedHostMemory){ if (! single_matrix_V){ cudaStreamDestroy(stream_Vcol); } cudaStreamDestroy(stream_Vrow); }
			cudaEventDestroy( event_reduction ); cudaEventDestroy( event_H ); cudaEventDestroy( event_W );
			if (! mappedHostMemory){ if(! single_matrix_V){ cudaEventDestroy(event_Vcol); } cudaEventDestroy(event_Vrow); }
			return EXIT_FAILURE;
		}
	#endif

	// -----------------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Initializing GPU data (single matrix=%i, num_streams_NMF=%" PRI_IDX ")... Done.\n",
				single_matrix_V, num_streams_NMF );
	#endif

	return EXIT_SUCCESS;

} // init_GPU_data

// ---------------------------------------------

/* Initializes scalar values for cuBLAS Library.
 *	d_scalar[2] = { 0.0, 1.0 }
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int init_scalars( void )
{

	// Scalars to upload.
	#define NMFGPU_NUM_SCALARS (2)
	real const cublas_scalars[ NMFGPU_NUM_SCALARS ] = { REAL_C(0.0), REAL_C(1.0) };

	// ----------------------------

	cudaError_t const cuda_status = cudaMemcpyAsync( (void *)d_scalar, (void *)cublas_scalars, NMFGPU_NUM_SCALARS * sizeof(real),
								cudaMemcpyHostToDevice, streams_NMF[0] );

	if ( cuda_status != cudaSuccess ) {
		print_error( sys_error_shown_by_all, "Could not initialize scalar values for cuBLAS Library.\ncudaMemcpyAsync( cublas_scalars "
				"-> d_scalar%s): %s\n", ( mappedHostMemory ? ", host memory mapped " : " " ),
				cudaGetErrorString(cuda_status) );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

	#undef NMFGPU_NUM_SCALARS

} // init_scalars

////////////////////////////////////////////////

/*
 * Destroys associated GPU data, such as CUDA Streams and/or CUDA Events.
 * If host memory was mapped, skips events and streams related to d_Vrow and d_Vcol.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int finalize_GPU_data( void )
{

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Finalizing GPU data (num_streams_NMF=%" PRI_IDX ")...\n", num_streams_NMF );
	#endif

	int status = EXIT_SUCCESS;	// Return status.

	cudaError_t cuda_status = cudaSuccess;

	// ------------------------------

	/* CUDA Events for timing */
	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
		status = destroy_timing_events();
	#endif

	// ------------------------------

	/* CUDA Streams for synchronization */

	// Main-flow streams
	for ( index_t st=0; st<num_streams_NMF; st++ ) {
		cuda_status = cudaStreamDestroy( streams_NMF[ st ] );
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Error destroying stream object %" PRI_IDX "/%" PRI_IDX
					" for synchronization on main flow: %s\n", st, num_streams_NMF,
					cudaGetErrorString(cuda_status) );
			status = EXIT_FAILURE;
		}
	}
	free( (void *) streams_NMF );


	// Stream for d_H
	cuda_status = cudaStreamDestroy( stream_H );
	if ( cuda_status != cudaSuccess ) {
		print_error( sys_error_shown_by_all, "Error destroying stream object for d_H: %s\n", cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}

	// Stream for d_W
	cuda_status = cudaStreamDestroy( stream_W );
	if ( cuda_status != cudaSuccess ) {
		print_error( sys_error_shown_by_all, "Error destroying stream object for d_W: %s\n", cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}

	if ( ! mappedHostMemory ) {	// Host memory is NOT mapped.

		// Stream for Vcol (i.e., set of columns from matrix V)
		if ( stream_Vcol != stream_Vrow ) {
			cuda_status = cudaStreamDestroy( stream_Vcol );
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "Error destroying stream object for Vcol: %s\n",
						cudaGetErrorString(cuda_status) );
				status = EXIT_FAILURE;
			}
		} // if ( stream_Vcols != stream_Vrows )

		// Stream for Vrow (i.e., set of rows from matrix V)
		cuda_status = cudaStreamDestroy( stream_Vrow );
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Error destroying stream object for Vrow: %s\n",
					cudaGetErrorString(cuda_status) );
			status = EXIT_FAILURE;
		}

	} // if ! mappedHostMemory

	// ----------------------------------

	/* CUDA Events for synchronization. */

	// Event for matrix reduction
	cuda_status = cudaEventDestroy( event_reduction );
	if ( cuda_status != cudaSuccess ) {
		print_error( sys_error_shown_by_all, "Error destroying event object for matrix reduction: %s\n",
				cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}

	// Event for d_H
	cuda_status = cudaEventDestroy( event_H );
	if ( cuda_status != cudaSuccess ) {
		print_error( sys_error_shown_by_all, "Error destroying event object for d_H: %s\n",
				cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}


	// Event for d_W
	cuda_status = cudaEventDestroy( event_W );
	if ( cuda_status != cudaSuccess ) {
		print_error( sys_error_shown_by_all, "Error destroying event object for d_W: %s\n",
				cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}


	if ( ! mappedHostMemory ) {	// Host memory is NOT mapped.

		// Event for Vcol (i.e., set of columns from matrix V)
		if ( event_Vcol != event_Vrow ) {
			cuda_status = cudaEventDestroy( event_Vcol );
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "Error destroying event object for Vcol: %s\n",
						cudaGetErrorString(cuda_status) );
				status = EXIT_FAILURE;
			}
		}

		// Event for Vrow (i.e., set of rows from matrix V)
		cuda_status = cudaEventDestroy( event_Vrow );
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Error destroying event object for Vrow: %s\n",
					cudaGetErrorString(cuda_status) );
			status = EXIT_FAILURE;
		}

	} // if ! mappedHostMemory

	// ---------------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Finalizing GPU data (num_streams_NMF=%" PRI_IDX ")... Done.\n", num_streams_NMF );
	#endif

	return status;

} // finalize_GPU_data

// ---------------------------------------------

/*
 * Frees all allocated memory.
 * Host memory in mapped-memory mode; device memory, otherwise.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int free_allocated_memory( void )
{

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Memory clean-up (mapped memory: %i)...\n", mappedHostMemory );
	#endif

	int status = EXIT_SUCCESS;	// Return status
	cudaError_t cuda_status = cudaSuccess;

	// -----------------------------------

	if ( mappedHostMemory ) {	// Host memory was mapped.

		if ( freeHostMemory( (void *)h_scalar, "h_scalar, mapped-memory mode" ) != EXIT_SUCCESS )
			status = EXIT_FAILURE;

		if ( freeHostMemory( (void *)h_accum, "h_accum, mapped-memory mode" ) != EXIT_SUCCESS )
			status = EXIT_FAILURE;

		if ( freeHostMemory( (void *)h_Aux, "h_Aux, mapped-memory mode" ) != EXIT_SUCCESS )
			status = EXIT_FAILURE;

		if ( freeHostMemory( (void *)h_WH, "h_WH, mapped-memory mode" ) != EXIT_SUCCESS )
			status = EXIT_FAILURE;

	} else {	// Frees DEVICE memory

		if ( d_scalar && ( (cuda_status=cudaFree((void *) d_scalar)) != cudaSuccess ) ) {
			print_error( sys_error_shown_by_all, "Could not free device memory (d_scalar): %s\n",
					cudaGetErrorString(cuda_status) );
			status = EXIT_FAILURE;
		}

		if ( d_accum && ( (cuda_status=cudaFree((void *) d_accum)) != cudaSuccess ) ) {
			print_error( sys_error_shown_by_all, "Could not free device memory (d_accum): %s\n",
					cudaGetErrorString(cuda_status) );
			status = EXIT_FAILURE;
		}

		if ( d_classification && ( (cuda_status=cudaFree( (void *) d_classification)) != cudaSuccess ) ) {
			print_error( sys_error_shown_by_all, "Could not free device memory (d_classification): %s\n",
					cudaGetErrorString(cuda_status) );
			status = EXIT_FAILURE;
		}

		if ( d_Aux && ( (cuda_status=cudaFree((void *) d_Aux)) != cudaSuccess ) ) {
			print_error( sys_error_shown_by_all, "Could not free device memory (d_Aux): %s\n",
					cudaGetErrorString(cuda_status) );
			status = EXIT_FAILURE;
		}

		if ( d_W && ( (cuda_status=cudaFree((void *) d_W)) != cudaSuccess ) ) {
			print_error( sys_error_shown_by_all, "Could not free device memory (d_W): %s\n",
					cudaGetErrorString(cuda_status) );
			status = EXIT_FAILURE;
		}

		if ( d_H && ( (cuda_status=cudaFree((void *) d_H)) != cudaSuccess ) ) {
			print_error( sys_error_shown_by_all, "Could not free device memory (d_H): %s\n",
					cudaGetErrorString(cuda_status) );
			status = EXIT_FAILURE;
		}

		if ( d_WH && ( (cuda_status=cudaFree((void *) d_WH)) != cudaSuccess ) ) {
			print_error( sys_error_shown_by_all, "Could not free device memory (d_WH): %s\n",
					cudaGetErrorString(cuda_status) );
			status = EXIT_FAILURE;
		}

		if ( ((d_Vrow != d_Vcol)*(uintptr_t)d_Vcol) && ( (cuda_status=cudaFree((void *) d_Vcol)) != cudaSuccess ) ) {
			print_error( sys_error_shown_by_all, "Could not free device memory (d_Vcol): %s\n",
					cudaGetErrorString(cuda_status) );
			status = EXIT_FAILURE;
		}

		if ( d_Vrow && ( (cuda_status=cudaFree((void *) d_Vrow)) != cudaSuccess ) ) {
			print_error( sys_error_shown_by_all, "Could not free device memory (d_Vrow): %s\n",
					cudaGetErrorString(cuda_status) );
			status = EXIT_FAILURE;
		}

	} // If host memory was mapped

	d_scalar = h_scalar = NULL;
	d_accum = h_accum = NULL;
	d_classification = d_last_classification = NULL;
	d_Aux = h_Aux = NULL;
	d_W = NULL;
	d_H = NULL;
	d_WH = h_WH = NULL;
	d_Vrow = d_Vcol = NULL;

	// -----------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Memory clean-up (mapped memory: %i)... done.\n", mappedHostMemory );
	#endif

	return status;

} // free_allocated_memory

////////////////////////////////////////////////

/*
 * Setups the GPU device:
 *	- Checks matrix dimensions.
 *	- Computes the required memory.
 *	- Allocates memory for data matrices.
 *	- Initializes associated GPU data, such as CUDA events/streams, and some kernel parameters.
 *
 * do_classf: Set to 'true' if classification vector will be computed.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int setup_GPU( size_t mem_size, bool do_classf )
{

	#if NMFGPU_VERBOSE_2
		print_message( dbg_shown_by_all, "setup_GPU( num_processes=%" PRI_IDX ",mem_size=%zu,do_classf=%i)\n",
				num_processes, mem_size, do_classf );
	#endif

	// -------------------------------------

	/* Computes the size of data blocks to be transferred to the GPU.
	 *
	 * Assuming that:
	 *	N, M: Total number of rows and columns of input matrix V,
	 *	Mp: Total number of columns of V, including useless data for padding,
	 *	NpP, MpP: Dimensions of the block from V to be processed by this GPU,
	 *		If num_processes == 1, then NpP==N and MpP==M.
	 *	Kp: Factorization rank with data for padding.
	 *
	 * this routine computes BLN <= NpP and BLM <= MpP so it can be loaded into the GPU:
	 *	One BLN x Mp block from matrix d_Vrow.
	 *	One N x BLMp block from matrix d_Vcol ('BLMp' means 'BLM' with padding) .
	 *	One MAX( BLN x Mp , N x BLMp ) block for matrix WH (WH == W*H).
	 *	Matrix W (N x Kp).
	 *	Matrix H (M x Kp).
	 *	One <MAX(N,M) x Kp> block for matrix Aux (i.e., Waux or Haux).
	 *	One Kp-length vector for accum (i.e., accum_h or accum_w).
	 *	Two constant values (scalar arguments for the cuBLAS Library),
	 * where BLMp is the padding for BLM (BLMp <= MpPp). It is a multiple of memory_alignment.
	 *
	 * IF there is enough GPU memory for all matrices at full size:
	 *	Sets BLN=NpP, BLM=MpP, BLMp=MpPp and full_matrix=1.
	 * ELSE,
	 *	full_matrix is set to 0.
	 *	Computes BLN, BLM and BLMp for d_WH, d_Vr and d_Vc.
	 *
	 * do_classf: Set to 'true' if classification vector will be computed.
	 */

	index_t BLN = NpP, BLM = MpP, BLMp = MpPp;
	bool full_matrix = true;

	if ( get_BLs( mem_size, do_classf, &BLN, &BLM, &BLMp, &full_matrix ) != EXIT_SUCCESS )
		return EXIT_FAILURE;

	// -------------------------------------

	// Initializes block information structures.

	// block_N
	#if NMFGPU_FORCE_BLOCKS || NMFGPU_VERBOSE || NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
		append_printed_message( verb_shown_by_all, "\n" );
		print_message( dbg_shown_by_all, "For dimension \"NpP\" = %" PRI_IDX " (N=%" PRI_IDX ", %" PRI_IDX " processes):\n",
				NpP, N, num_processes);
	#endif
	init_block_conf( NpP, NpP, BLN, BLN, &block_N );

	// block_M
	#if NMFGPU_FORCE_BLOCKS || NMFGPU_VERBOSE || NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
		append_printed_message( verb_shown_by_all, "\n" );
		print_message( dbg_shown_by_all, "For dimension \"MpP\" = %" PRI_IDX " (%" PRI_IDX " with padding ; M=%" PRI_IDX
				", %" PRI_IDX " processes):\n", MpP, MpPp, M, num_processes );
	#endif
	init_block_conf( MpP, Mp, BLM, BLMp, &block_M );

	// ----------------------------------

	// Block configuration testing. No memory is allocated on the device.

	#if NMFGPU_TEST_BLOCKS
		flush_output( false );
		return EXIT_FAILURE;
	#endif

	// ----------------------------------

	// Allocates memory for data matrices.

	// Matrix V can be fully loaded into the GPU memory (i.e., d_Vrow == d_Vcol)
	bool const single_matrix_V = full_matrix * (num_processes == 1);

	/* If this is an integrated system <AND> it is possible to map host memory:
	 *	Maps the host memory that was previously allocated for data matrices, into the address space of the device.
	 *	Memory for temporary buffers (e.g., d_WH or d_Aux) is first allocated on the host, and then mapped.
	 * Else,
	 *	Allocates DEVICE memory.
	 */
	if ( allocate_memory( BLN, BLMp, single_matrix_V, do_classf ) != EXIT_SUCCESS )
		return EXIT_FAILURE;

	// -----------------------------------

	// Initializes GPU data (CUDA Streams and CUDA Events).

	// Number of main-flow streams.
	num_streams_NMF = MAX( (block_N.num_steps[0] + block_N.num_steps[1]), (block_M.num_steps[0] + block_M.num_steps[1]) );

	if ( init_GPU_data( single_matrix_V ) != EXIT_SUCCESS ) {
		free_allocated_memory();
		return EXIT_FAILURE;
	}

	// -----------------------------------

	// Initializes the scalar values for cuBLAS Library.

	if ( init_scalars() != EXIT_SUCCESS ) {
		finalize_GPU_data();
		free_allocated_memory();
		return EXIT_FAILURE;
	}

	// -----------------------------------

	// Initializes some kernel parameters.
	init_kernel_params( Kp );

	// -----------------------------------

	// Finally, checks if everything was successfully initialized.

	if ( check_cuda_status() != EXIT_SUCCESS ) {
		print_error( sys_error_shown_by_all, "Error setting-up GPU data.\n" );
		finalize_GPU_data();
		free_allocated_memory();
		return EXIT_FAILURE;
	}

	// --------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "setup_GPU( num_processes=%" PRI_IDX ",mem_size=%zu,do_classf=%i)... Done.\n",
				num_processes, mem_size, do_classf );
	#endif

	return EXIT_SUCCESS;

} // setup_GPU

////////////////////////////////////////////////

/*
 * Shuts down current cuBLAS/CUDA context.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int shutdown_GPU( void )
{

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Shutting down CUDA/cuBLAS...\n" );
	#endif

	int status = EXIT_SUCCESS;	// Return status

	// --------------------------------------

	cublasStatus_t const cublas_status = cublasDestroy( cublas_handle );
	if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
		print_error( sys_error_shown_by_all, "Error shutting-down cuBLAS: %s\n", getCublasErrorString( cublas_status ) );
		status = EXIT_FAILURE;
	}

	cudaError_t const cuda_status = cudaDeviceReset();
	if ( cuda_status != cudaSuccess ) {
		print_error( sys_error_shown_by_all, "Error shutting-down CUDA: %s\n", cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}

	// ---------------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Shutting down CUDA/cuBLAS... done\n");
	#endif

	return status;

} // shutdown_GPU

////////////////////////////////////////////////

/*
 * Finalizes the GPU device.
 *
 * Destroys associated GPU data, such as cudaStreams and/or cudaEvents.
 * Frees all allocated device memory, and shuts-down current CUDA/cuBLAS context.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int finalize_GPU_device( void )
{

	#if NMFGPU_VERBOSE
		print_message( dbg_shown_by_all, "Finalizing GPU Device...\n" );
	#endif

	int status = EXIT_SUCCESS;	// Return status

	// ------------------------------------------

	// First, it checks for previous errors.
	status = check_cuda_status();
	if ( status != EXIT_SUCCESS )
		print_error( sys_error_shown_by_all, "Shutting down device...\n" );

	// ------------------------------------------

	// Finalizes GPU data (CUDA Streams and CUDA events).
	if ( finalize_GPU_data() != EXIT_SUCCESS )
		status = EXIT_FAILURE;


	// Device (or host) memory clean up
	if ( free_allocated_memory() != EXIT_SUCCESS )
		status = EXIT_FAILURE;


	// Finally, shuts down the device.
	if ( shutdown_GPU() != EXIT_SUCCESS )
		status = EXIT_FAILURE;

	// -----------------------------------------

	#if NMFGPU_VERBOSE
		print_message( dbg_shown_by_all, "Finalizing GPU Device... done.\n" );
	#endif

	return status;

} // finalize_GPU_device

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Allocates PINNED HOST memory.
 *
 * Allocates HOST memory that is page-locked and accessible to the device,
 * increasing the speed of HOST-DEVICE transfers.
 *
 * size: Size of data IN BYTES.
 *
 * wc: Set to 'true' to allocate the memory as 'write-combined' (WC).
 *	Useful ONLY for data transferred from the HOST to the DEVICE.
 *
 * clear: Set to 'true' to initialize the memory area with zeros.
 *
 * WARNING:
 *	- The GPU device must has been properly initialized through initialize_GPU().
 *	- Allocating excessive amounts of pinned memory may degrade system performance,
 *	  since it reduces the amount of memory available to the system for paging.
 *	- Memory allocated by this function must be freed with freeHostMemory().
 *
 * Returns a pointer to the allocated memory, or NULL on error.
 */
void *getHostMemory( size_t size, bool wc, bool clear )
{

	unsigned int flags = cudaHostAllocPortable;

	if ( wc )
		flags |= cudaHostAllocWriteCombined;

	// Host memory mapped into the address space of the device.
	if ( mappedHostMemory )
		flags |= cudaHostAllocMapped;

	// -----------------------------------------

	void *__restrict__ pHost = NULL;
	cudaError_t const cuda_status = cudaHostAlloc( (void**) &pHost, size, flags );
	if ( cuda_status != cudaSuccess ) {
		print_error( sys_error_shown_by_all, "Error in getHostMemory: cudaHostAlloc(size=%zu bytes): %s\n",
				size, cudaGetErrorString(cuda_status) );
		return NULL;
	}

	// -----------------------------------------

	// Clears the memory area if requested.
	errno = 0;
	if ( clear && (! memset( pHost, 0, size )) ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in getHostMemory: memset(0, size=%zu bytes)", size );
		cudaFreeHost( (void *) pHost );
		return NULL;
	}

	// -----------------------------------------

	return pHost;

} // getHostMemory

////////////////////////////////////////////////

/*
 * Frees HOST memory previously allocated by getHostMemory().
 *
 * WARNING:
 *	This function must be called *BEFORE* shutdown_GPU() or shutdown_GPUdevice().
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int freeHostMemory( void *__restrict__ pHost, char const *__restrict__ pHost_name )
{

	if ( pHost ) {
		cudaError_t const cuda_status = cudaFreeHost( (void *) pHost );
		if ( cuda_status != cudaSuccess ) {
			print_error( sys_error_shown_by_all, "Error in freeHostMemory( %s ): cudaFreeHost(): %s\n",
					pHost_name, cudaGetErrorString(cuda_status) );
			return EXIT_FAILURE;
		}
	}

	return EXIT_SUCCESS;

} // freeHostMemory

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Creates a Random-number Generator using the cuRAND Library.
 *
 * WARNING:
 *	The seed is NOT initialized. For that, please use set_randomGenerator_seed().
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int init_randomGenerator( void )
{

	#if ! NMFGPU_CPU_RANDOM

		#if NMFGPU_VERBOSE
			print_message( verb_shown_by_all, "Starting Random-number Generator...\n" );
		#endif

		curandRngType_t const rng_type = CURAND_RNG_PSEUDO_DEFAULT;			// Random number generator type.

		curandOrdering_t const curand_ordering = CURAND_ORDERING_PSEUDO_BEST;		// Ordering type.

		curandStatus_t curand_status = CURAND_STATUS_SUCCESS;

		// -------------------------------------

		/* Queries the default stack size of each GPU thread on SM_20 and above.
		 * Please, see:
		 *	https://devtalk.nvidia.com/default/topic/481553/curand-eats-device-memory/
		 */
		if ( computeCapability >= 2 ) {	// Compute Capability >= 2.0

			cudaError_t const cuda_status = cudaDeviceGetLimit( &defaultStackSize, cudaLimitStackSize );
			if ( cuda_status != cudaSuccess )
				print_error( sys_error_shown_by_all, "Warning: Could not determine the default stack size of GPU threads: %s\n"
					     "Depending on the version of the cuRAND Library, some memory leaks might appear.\n",
						cudaGetErrorString(cuda_status) );
			defaultStackSize = 0;	// Global variable.

		} // If Compute Capability >= 2.0

		// -------------------------------------

		// Creates the generator.
		curand_status = curandCreateGenerator( &curand_generator, rng_type );
		if ( curand_status != CURAND_STATUS_SUCCESS ) {
			print_error( sys_error_shown_by_all, "Error: Could not create the random numbers generator: %s\n",
					getCurandErrorString( curand_status ) );
			return EXIT_FAILURE;
		}

		// -------------------------------------

		// Sets the ordering
		curand_status = curandSetGeneratorOrdering( curand_generator, curand_ordering );
		if ( curand_status != CURAND_STATUS_SUCCESS ) {
			print_error( sys_error_shown_by_all, "Error setting-up the random numbers generator (ordering): %s\n",
					getCurandErrorString( curand_status ) );
			curandDestroyGenerator( curand_generator );
			return EXIT_FAILURE;
		}

		#if NMFGPU_VERBOSE
			print_message( verb_shown_by_all, "Starting Random-number Generator... done.\n" );
		#endif

	#endif /* NMFGPU_CPU_RANDOM */

	return EXIT_SUCCESS;

} // init_randomGenerator

////////////////////////////////////////////////

/*
 * Sets the seed value for an existing pseudo-random number generator.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int set_randomGenerator_seed( unsigned long long seed )
{

	#if ! NMFGPU_CPU_RANDOM

		#if NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "Setting seed '%llu' for the Random number Generator...\n",seed);
		#endif

		curandStatus_t curand_status = CURAND_STATUS_SUCCESS;

		// Sets the seed.
		curand_status = curandSetPseudoRandomGeneratorSeed( curand_generator, seed );
		if ( curand_status != CURAND_STATUS_SUCCESS ) {
			print_error( sys_error_shown_by_all, "Error setting-up the seed '%llu' for the random number generator: %s\n",
					seed, getCurandErrorString( curand_status ) );
			return EXIT_FAILURE;
		}

		// Sets up the starting state.
		curand_status = curandGenerateSeeds( curand_generator );
		if ( curand_status != CURAND_STATUS_SUCCESS ) {
			print_error( sys_error_shown_by_all, "Error setting-up starting state of the random generator: %s\n",
					getCurandErrorString( curand_status ) );
			return EXIT_FAILURE;
		}

		// -------------------------

		/* Resets the stack size of each GPU thread on SM_20 and above.
		 * Please, see:
		 *	https://devtalk.nvidia.com/default/topic/481553/curand-eats-device-memory/
		 */
		if ( computeCapability >= 2 ) {	// Compute Capability >= 2.0

			if ( defaultStackSize ) {
				cudaError_t const cuda_status = cudaDeviceSetLimit( cudaLimitStackSize, defaultStackSize );
				if ( cuda_status != cudaSuccess ) {
					print_error( sys_error_shown_by_all, "Warning: Could not reset the stack size of GPU threads "
							"to %zu bytes: %s\nDepending on the version of the cuRAND Library, "
							"some memory leaks might appear.\n", defaultStackSize, cudaGetErrorString(cuda_status) );
				}
			} else
				print_error( sys_error_shown_by_all, "Warning: Could not reset the stack size of GPU threads.\n"
						"Depending on the version of the cuRAND Library, some memory leaks might appear.\n" );

		} // if computeCapability >= 2

		// -------------------------

		#if NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "Setting seed '%llu' for the Random number Generator... done.\n",seed);
		#endif

	#endif /* ! NMFGPU_CPU_RANDOM */

	return EXIT_SUCCESS;

} // set_randomGenerator_seed

////////////////////////////////////////////////

/*
 * Creates a Random-number Generator using the cuRAND Library.
 * Sets the seed to the given parameter.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int init_GPU_random( unsigned long long seed )
{

	#if ! NMFGPU_CPU_RANDOM

		if ( init_randomGenerator() != EXIT_SUCCESS )
			return EXIT_FAILURE;

		if ( set_randomGenerator_seed( seed ) != EXIT_SUCCESS ) {
			finalize_randomGenerator();
			return EXIT_FAILURE;
		}

	#endif /* NMFGPU_CPU_RANDOM */

	return EXIT_SUCCESS;

} // init_GPU_random

////////////////////////////////////////////////

/*
 * Destroys an existing generator and free all memory associated with its state.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int finalize_randomGenerator( void )
{

	#if ! NMFGPU_CPU_RANDOM

		#if NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "Destroying Random Number Generator...\n" );
		#endif

		curandStatus_t const curand_status = curandDestroyGenerator( curand_generator );
		if ( curand_status != CURAND_STATUS_SUCCESS ) {
			print_error( sys_error_shown_by_all, "Error destroying the random number generator: %s\n",
					getCurandErrorString( curand_status ) );
			return EXIT_FAILURE;
		}

		#if NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "Destroying Random Number Generator... done.\n" );
		#endif

	#endif /* NMFGPU_CPU_RANDOM */

	return EXIT_SUCCESS;

} // finalize_randomGenerator

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Blocks until GPU has completed all operations associated to 'stream'.
 */
void sync_GPU( cudaStream_t stream )
{

	#if NMFGPU_VERBOSE_2
		print_message( dbg_shown_by_all, "sync_GPU(): Waiting for results...\n" );
	#endif
	// ----------------------------------

	/* Waits for all operations on 'stream'.
	 * NOTE: The CPU thread will block or spin according to flags
	 *	 specified in initialize_GPU().
	 */

	#if NMFGPU_DEBUG
		cudaError_t const cuda_status =
	#endif
			cudaStreamSynchronize( stream );

			///////////////////////////////
			#if NMFGPU_DEBUG
				if ( cuda_status != cudaSuccess )
					print_error( sys_error_shown_by_all, "Error in sync_GPU(): cudaStreamSynchronize(): %s\n",
							cudaGetErrorString(cuda_status) );
			#endif
			///////////////////////////////

	// ---------------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "sync_GPU(): Waiting for results... Done.\n" );
	#endif

} // sync_GPU

////////////////////////////////////////////////
////////////////////////////////////////////////
