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
 * GPU_setup.cu
 *	Generic definitions and routines for GPU set-up and management.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Additional information:
 *		NMFGPU_VERBOSE: Shows some messages concerning the progress of the program, as well as some configuration parameters.
 *		NMFGPU_VERBOSE_2: Shows the parameters in some routine calls.
 *
 *	Timing:
 *		NMFGPU_PROFILING_TRANSF: Compute timing of data transfers. Shows additional information.
 *		NMFGPU_PROFILING_KERNELS: Compute timing of CUDA kernels. Shows additional information.
 *
 *	Debug / Testing:
 *		NMFGPU_FIXED_INIT: Uses "random" values generated from a fixed seed (define in common.h).
 *		NMFGPU_CPU_RANDOM: Uses the CPU (host) random generator (not the CURAND library).
 *		NMFGPU_DEBUG: Shows the result of each matrix operation and data transfer.
 *		NMFGPU_FORCE_BLOCKS: Forces the processing of the input matrix as four blocks.
 *		NMFGPU_TEST_BLOCKS: Just shows block information structure. No GPU memory is allocated.
 *
 **********************************************************
 **********************************************************
 **********************************************************
 *
 * Data matrices:
 * 	V (N rows, M columns): Input matrix,
 * 	K: Factorization Rank,
 * 	W (N,K): Output matrix,
 * 	H (K,M): Output matrix,
 * such that V ~ W*H
 *
 * NOTE: In order to improve performance:
 *	- Matrix H is stored in memory as COLUMN-major (i.e., it is transposed).
 *
 *	- All matrices include useless data for padding. Padded dimensions
 *	  are denoted with the 'p' character, e.g., 'Mp' (i.e.,, M + padding)
 *	  or 'Kp' (factorization_rank + padding).
 *
 *	- Padded dimensions are a multiple of memory_alignment
 *	  (a global variable which currently is equal to warpSize or warpSize/2).
 *
 ***************
 * Multi-GPU version:
 *
 * When the input matrix V is distributed among multiple devices each host thread processes
 * the following sets of rows and columns:
 *	Vrow[ 1..NnP ][ 1..M ] <-- V[ bN..(bN+NnP) ][ 1..M ]	(i.e., NnP rows, starting from bN)
 *	Vcol[ 1..N ][ 1..MnP ] <-- V[ 1..N ][ bM..(bM+MnP) ]	(i.e., MnP columns, starting from bM)
 *
 * Such sets allow to update the corresponding rows and columns of W and H, respectively.
 *
 * Note that each host thread has a private copy of matrices W and H, which must be synchronized
 * after being updated.
 *
 ****************
 * Large input matrix (blockwise processing):
 *
 * If the input matrix (or the portion assigned to this device) is too large for the GPU memory,
 * it must be blockwise processed as follow:
 *	d_Vrow[1..BLN][1..Mp] <-- Vrow[ offset..(offset + BLN) ][1..Mp]			(i.e., BLN <= NnP rows)
 *	d_Vcol[1..N][1..BLMp] <-- Vcol[1..N][ offset_Vcol..(offset_Vcol + BLMp) ]	(i.e., BLM <= MnP columns)
 *
 * Note that padded dimensions are denoted with the suffix 'p' (e.g., Mp, BLMp, etc).
 *
 * In any case, matrices W and H are fully loaded into the GPU memory.
 *
 * Information for blockwise processing is stored in two block_t structures (one for each dimension).
 * Such structures ('block_N' and 'block_M') are initialized in init_block_conf() routine.
 *
 *********************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
	#include "timing.cuh"
#endif
#include "GPU_kernels.cuh"	/* __ITEMS_PER_THREAD constants */
#include "GPU_setup.cuh"

// ---------------------------------------------
// ---------------------------------------------

/* Macro Functions */

/* Alignment value for data to be stored in GPU memory: Typically 'warpSize/2' on compute capability 1.x, and 'warpSize' on 2.x and beyond.
 * It is similar to the DEVICE-only constant 'MEMORY_ALIGNMENT' defined in "GPU_kernels.h".
 */
#ifndef SET_MEMORY_ALIGNMENT
	#define SET_MEMORY_ALIGNMENT(cc_major,warpSize) ( ((cc_major) > 1) ? (warpSize) : ((warpSize) >> 1) )
#endif

// ---------------------------------------------
// ---------------------------------------------

/* HOST-ONLY GLOBAL Variables */

index_t device_id = 0;			// Device ID number.

index_t num_devices = 1;		// Number of devices.

cublasHandle_t cublas_handle;		// CUBLAS library context.

curandGenerator_t curand_generator;	// CURAND Random values Generator.

index_t computeCapability = 2;		// Compute Capability (major).

index_t computeCapability_minor = 0;	// Compute Capability (minor).

/* Alignment value for data to be stored in GPU memory: Typically 'warpSize/2' on compute capability 1.x, and 'warpSize' on 2.x and beyond.
 * It is similar to the DEVICE-only constant 'MEMORY_ALIGNMENT' defined on "GPU_kernels.h".
 *
 * Value set by the macro SET_MEMORY_ALIGNMENT() defined below.
 */
index_t memory_alignment = 16;

index_t maxThreadsPerBlock = 512;	// Maximum number of threads per block.

index_t multiProcessorCount = 30;	// Number of multiprocessors.

index_t maxThreadsPerMultiProcessor = 1024; // Maximum number of resident threads per multiprocessor (>= maxThreadsPerBlock)

// Maximum number of thread blocks on dimensions X and Y.
index_t maxGridSizeX = 65535, maxGridSizeY = 65535;

size_t defaultStackSize = 1024;		// Default stack size of each GPU thread ( Compute Capability >= 2.0 )

// Typical number of threads per block. It should be a divisor of maxThreadsPerMultiProcessor.
index_t threadsPerBlock = 256;		// <= maxThreadsPerBlock

// Threads per block for kernels requiring a value multiple of <Kp> (denoted as 'pitch').
index_t threadsPerBlock_pitch = 256;	// threadsPerBlock <= threadsPerBlock_pitch <= maxThreadsPerBlock

// Maximum block height using <threadsPerBlock_pitch> threads.
index_t maxBlockHeight_pitch = 16;	// (threadsPerBlock_pitch / pitch)

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

// Matrix dimensions (host side):
index_t N = 0;		// Number of rows of input matrix V.
index_t M = 0;		// Number of columns of input matrix V.
index_t K = 0;		// Factorization rank.

// Dimensions for multi-GPU version:
index_t NnP = 0;	// Number of rows of V assigned to this GPU (NnP <= N).
index_t MnP = 0;	// Number of columns of V assigned to this GPU (MnP <= M).

// Padded dimensions:
index_t Mp = 0;		// 'M' rounded up to the next multiple of <memory_alignment>.
index_t Kp = 0;		// 'K' rounded up to the next multiple of <memory_alignment>.
index_t MnPp = 0;	// 'MnP' rounded up to the next multiple of <memory_alignment> (MnPp <= Mp).

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

real const *__restrict__ d_scalar = NULL;		// Scalars for CUBLAS Library calls. d_scalar[2] = { 0, 1 };
real const *__restrict__ d_zero = NULL;		// Pointer to d_scalar[0] == 0
real const *__restrict__ d_one = NULL;		// Pointer to d_scalar[1] == 1

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

/*
 * Prints an error message according to the provided CUBLAS status.
 */
void printCublasErrorString( cublasStatus_t cublas_status )
{

	switch( cublas_status ) {
		case CUBLAS_STATUS_SUCCESS:
			fprintf(stderr, "Operation completed successfully.\n");
		break;

		case CUBLAS_STATUS_NOT_INITIALIZED:
			fprintf(stderr, "CUBLAS library not initialized.\n");
		break;

		case CUBLAS_STATUS_ALLOC_FAILED:
			fprintf(stderr, "Resource allocation failed.\n");
		break;

		case CUBLAS_STATUS_INVALID_VALUE:
			fprintf(stderr, "An invalid numerical value was used as an argument.\n");
		break;

		case CUBLAS_STATUS_ARCH_MISMATCH:
			fprintf(stderr, "Architecture mismatch, GPU does not support the requested feature.\n");
		break;

		case CUBLAS_STATUS_MAPPING_ERROR:
			fprintf(stderr, "Access to GPU memory space failed.\n");
		break;

		case CUBLAS_STATUS_EXECUTION_FAILED:
			fprintf(stderr, "GPU program failed to execute.\n");
		break;

		case CUBLAS_STATUS_INTERNAL_ERROR:
			fprintf(stderr, "An internal CUBLAS operation failed.\n");
		break;

		default:
			fprintf(stderr, "Unknown CUBLAS error code (value: %x).\n", cublas_status );
		break;
	}

} // printCublasErrorString

/////////////////////////////////////////////////////////////////////

/*
 * Prints an error message according to the provided CURAND status.
 */
void printCurandErrorString( curandStatus_t curand_status )
{

	switch( curand_status ) {
		case CURAND_STATUS_SUCCESS:
			fprintf(stderr, "Operation completed successfully.\n");
		break;

		case CURAND_STATUS_VERSION_MISMATCH:
			fprintf(stderr, "Header file and linked library version do not match.\n");
		break;

		case CURAND_STATUS_NOT_INITIALIZED:
			fprintf(stderr, "CURAND generator not initialized.\n");
		break;

		case CURAND_STATUS_ALLOCATION_FAILED:
			fprintf(stderr, "Resource allocation failed.\n");
		break;

		case CURAND_STATUS_TYPE_ERROR:
			fprintf(stderr, "Invalid random number generator type.\n");
		break;

		case CURAND_STATUS_OUT_OF_RANGE:
			fprintf(stderr, "Argument is out of range.\n");
		break;

		case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
			fprintf(stderr, "Length requested is not a multiple of dimension.\n");
		break;

		case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
			fprintf(stderr, "GPU device does not support double-precision data.\n");
		break;

		case CURAND_STATUS_LAUNCH_FAILURE:
			fprintf(stderr, "Kernel launch failure.\n");
		break;

		case CURAND_STATUS_PREEXISTING_FAILURE:
			fprintf(stderr, "Preexisting failure on library entry.\n");
		break;

		case CURAND_STATUS_INITIALIZATION_FAILED:
			fprintf(stderr, "Initialization of CUDA failed.\n");
		break;

		case CURAND_STATUS_ARCH_MISMATCH:
			fprintf(stderr, "Architecture mismatch, GPU does not support the requested feature.\n");
		break;

		case CURAND_STATUS_INTERNAL_ERROR:
			fprintf(stderr, "An internal CURAND operation failed.\n");
		break;

		default:
			fprintf(stderr, "Unknown CURAND error code (value: %x).\n", curand_status );
		break;
	}

} // printCurandErrorString

/////////////////////////////////////////////////////////////////////

/*
 * Checks the provided CUBLAS status.
 * If it is NOT OK, it shows an error message.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int check_cublas_status_st( cublasStatus_t cublas_status )
{

	if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] CUBLAS Error detected: ", device_id );
		printCublasErrorString( cublas_status );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // check_cublas_status_st

/////////////////////////////////////////////////////////////////////

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
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] CUDA Error detected: %s\n", device_id, cudaGetErrorString(cuda_status) );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // check_cuda_status_st

/////////////////////////////////////////////////////////////////////

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
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] CUDA Error detected: %s\n", device_id, cudaGetErrorString(cuda_status) );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // check_cuda_status

//////////////////////////////////////////////////////////////////////////////

/*
 * Initializes CUDA and CUBLAS on the specified device.
 * Stores in 'mem_size' the amount of free Global Memory.
 *
 * WARNING:
 * 	This function must be called *BEFORE* any other CUDA-related routine (e.g., cudaMallocHost(), cudaHostAlloc()).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int init_GPU( index_t dev_id, index_t num_devs, size_t *__restrict__ const mem_size )
{

	#if NMFGPU_VERBOSE
		printf("\nInitializing CUDA/CUBLAS on device %" PRI_IDX "/%" PRI_IDX "...\n", dev_id, num_devs );
	#endif

	cudaError_t cuda_status = cudaSuccess;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;

	// -----------------------------------

	// Device flags
	unsigned int const flags =
				// // Yields its thread until device completes ops.
				// cudaDeviceScheduleYield	|

				// Blocks CPU thread on sync. primitives (e.g., cudaDeviceSynchronize, cudaStreamSynchronize).
				cudaDeviceScheduleBlockingSync;

	if ( (cuda_status=cudaSetDeviceFlags(flags)) != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\nError setting flags on device %" PRI_IDX " (flags=%x): %s\n", dev_id, flags,
			cudaGetErrorString(cuda_status) );
		return EXIT_FAILURE;
	}

	// -----------------------------------

	// Attaches to the specified device.
	if ( (cuda_status=cudaSetDevice( (int) dev_id )) != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\nCould not attach to device %" PRI_IDX ": %s\n", dev_id, cudaGetErrorString(cuda_status) );
		return EXIT_FAILURE;
	}

	device_id = dev_id;	// Global variables.
	num_devices = num_devs;

	// ----------------------------------

	// Initializes CUDA/CUBLAS on the selected device.

	cublas_status = cublasCreate( &cublas_handle );
	if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
		fflush(stdout);
		fprintf(stderr, "\n[GPU%" PRI_IDX "] Error initializing CUBLAS: ", dev_id);
		printCublasErrorString(cublas_status);
		return EXIT_FAILURE;
	}

	// -----------------------------------

	// Makes sure this process is attached to the selected device.

	int device;
	cuda_status = cudaGetDevice( &device );
	if ( ( cuda_status != cudaSuccess ) + ( (index_t) device != dev_id ) ) {
		fflush(stdout);
		if ( cuda_status != cudaSuccess )
			fprintf( stderr, "\n[GPU%" PRI_IDX "] Error accessing device: %s\n", dev_id, cudaGetErrorString(cuda_status) );
		else
			fprintf( stderr, "\nError: Process NOT attached to device %" PRI_IDX " (returned device ID: %i).\n", dev_id, device );
		cublasDestroy( cublas_handle );
		cudaDeviceReset();
		return EXIT_FAILURE;
	}

	// -------------------------------------

	/* Preferred cache configuration
	 *	FIXME: The "PreferL1" configuration might have negative effects on CUBLAS operation.
	 */
	cuda_status = cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Could not set the preferred cache configuration: %s\n",
			dev_id, cudaGetErrorString(cuda_status) );
		cublasDestroy( cublas_handle );
		cudaDeviceReset();
		return EXIT_FAILURE;
	}

	// Shared memory configuration
	cuda_status = cudaDeviceSetSharedMemConfig( SHARED_MEM_BANK_SIZE );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Could not set the preferred shared memory configuration: %s\n",
			dev_id, cudaGetErrorString(cuda_status) );
		cublasDestroy( cublas_handle );
		cudaDeviceReset();
		return EXIT_FAILURE;
	}

	// -------------------------------------

	// Retrieves GPU properties.

	struct cudaDeviceProp device_prop;
	cuda_status = cudaGetDeviceProperties( &device_prop, (int) dev_id );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Could not retrieve device properties: %s\n", dev_id, cudaGetErrorString(cuda_status) );
		cublasDestroy( cublas_handle );
		cudaDeviceReset();
		return EXIT_FAILURE;
	}

	// Compute Capability (major and minor)
	computeCapability = device_prop.major;
	computeCapability_minor = device_prop.minor;

	/* Alignment value for data to be stored in GPU memory.
	 * It is typically equal to 'WarpSize/2' on devices of compute capability 1.x, and 'WarpSize' on 2.x and beyond.
	 */
	memory_alignment = SET_MEMORY_ALIGNMENT( device_prop.major, device_prop.warpSize );

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
	 *	E.g., 256 for Compute Capability 1.x, 512 for Compute Capability 2.0 - 3.5
	 */
	{
		// t1: 0, or (maxThreadsPerMultiProcessor/3)
		index_t const t1 = device_prop.maxThreadsPerMultiProcessor % device_prop.maxThreadsPerBlock;

		// Makes sure (maxThreadsPerMultiProcessor/4) <= maxThreadsPerBlock
		index_t const t2 = MIN( (device_prop.maxThreadsPerMultiProcessor/4), device_prop.maxThreadsPerBlock );

		threadsPerBlock = ( t1 ? t1 : t2 );	// (maxThreadsPerMultiProcessor/3), or (maxThreadsPerMultiProcessor/4).
	}

	// --------------------------------------

	// Retrieves the default stack size of GPU threads (Compute Capability >= 2.0).
	if ( device_prop.major >= 2 ) {

		cuda_status = cudaDeviceGetLimit( &defaultStackSize, cudaLimitStackSize );
		if ( cuda_status != cudaSuccess ) {
			fflush(stdout);
			fprintf( stderr, "\n[GPU%" PRI_IDX "] Could not determine the default stack size of GPU threads: %s\n",
				dev_id, cudaGetErrorString(cuda_status) );
			cublasDestroy( cublas_handle );
			cudaDeviceReset();
			return EXIT_FAILURE;
		}

	} // If Compute Capability >= 2.0

	// -------------------------------------

	// Available Device memory (bytes).

	size_t free_mem = 0, total_mem = 0;
	cuda_status = cudaMemGetInfo( &free_mem, &total_mem );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Could not determine the amount of free memory: %s\n",
			dev_id, cudaGetErrorString(cuda_status) );
		cublasDestroy( cublas_handle );
		cudaDeviceReset();
		return EXIT_FAILURE;
	}

	// NOTE: Reduces 20% to leave place for CUDA's internal temporary buffers.
	size_t const l_mem_size = (size_t) (free_mem * 0.8f);	// Value is truncated.

		// // Other values for debugging:
		//	l_mem_size = device_prop.totalGlobalMem;
		//	l_mem_size = 1000 * 1024 * 1024;	// 1000MBytes (< 1GBytes)
		//	l_mem_size =  512 * 1024 * 1024;	// 512MB
		//	l_mem_size = 1024 * 1024;		// 1MB

	// -------------------------------------

	// Shows some GPU properties

	#if NMFGPU_VERBOSE || NMFGPU_DEBUG || NMFGPU_FORCE_BLOCKS || NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS

		// CUDA version
		int runtime_version = 0;
		cudaRuntimeGetVersion( &runtime_version );

		// CUBLAS version
		int cublas_version = 0;
		cublasGetVersion( cublas_handle, &cublas_version );

		if ( ( ! dev_id ) + (num_devs == 1) ) {
			printf( "\n[GPU%" PRI_IDX "] %s (total devices=%" PRI_IDX "): Compute_Capability=%i.%i, CUDA=%i.%i, CUBLAS=%i.%i\n"
				"\tWarp_size=%i, Memory_Alignment=%i, Max_Threads_per_Block=%i, Threads_per_Block=%i, "
				"Max_Grid_Dimensions(X,Y)=(%i,%i).\n"
				"\tMultiprocessors=%i, Threads_per_MultiProcessor=%i,\n"
				"\tGlobal_Memory=%zu bytes (%g MiB), Total_Memory=%g MiB.\n"
				"\tFree_Memory=%g MiB, Used=%g MiB, (Maximum_Allocatable=%g MiB).\n",
				dev_id, device_prop.name, num_devs, device_prop.major, device_prop.minor,
				runtime_version/1000, runtime_version%100, cublas_version/1000, cublas_version%100,
				device_prop.warpSize, memory_alignment, device_prop.maxThreadsPerBlock, threadsPerBlock,
				device_prop.maxGridSize[0], device_prop.maxGridSize[1],
				device_prop.multiProcessorCount, device_prop.maxThreadsPerMultiProcessor,
				device_prop.totalGlobalMem, (device_prop.totalGlobalMem/1048576.0f),
				(total_mem/1048576.0f), (free_mem/1048576.0f), ((total_mem-free_mem)/1048576.0f),
				(l_mem_size/1048576.0f) );

			// Specific configuration for Compute Capability >= 2.0
			if ( device_prop.major >= 2 )
				printf("\tDefault_Stack_Size=%zu\n", defaultStackSize );
		}
		fflush(stdout);
	#endif

	// -------------------------------------

	// Sets pointer mode used by the CUBLAS library in scalar arguments.

	cublas_status = cublasSetPointerMode( cublas_handle, CUBLAS_POINTER_MODE_DEVICE );
	if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
		fflush(stdout);
		fprintf(stderr, "\n[GPU%" PRI_IDX "] Error setting pointer mode used by the CUBLAS library: ", dev_id);
		printCublasErrorString(cublas_status);
		cublasDestroy( cublas_handle );
		cudaDeviceReset();
		return EXIT_FAILURE;
	}

	cudaGetLastError();	// Clears any possibly error flag.

	// -------------------------------------

	*mem_size = l_mem_size;

	#if NMFGPU_VERBOSE
		printf("\n[GPU%" PRI_IDX "] Initializing CUDA/CUBLAS... done (mem_size=%zu).\n", dev_id, l_mem_size);
	#endif

	return EXIT_SUCCESS;

} // init_GPU

////////////////////////////////////////////////////////////////////////////////

/*
 * Computes the padded dimension.
 *
 * Returns the next multiple of 'memory_alignment'.
 */
index_t get_padding( index_t dim )
{

	#if NMFGPU_VERBOSE_2
		printf("\n[GPU%" PRI_IDX "] get_padding( dim=%" PRI_IDX " )\n", device_id, dim);
	#endif

	// ------------------

	index_t const dim_mod_ma = ( dim % memory_alignment );

	// If "dim" is NOT a multiple of <memory_alignment>, computes the next multiple.
	index_t const padded_dim = ( dim_mod_ma ? ( dim - dim_mod_ma + memory_alignment ) : dim );

	// ------------------

	#if NMFGPU_VERBOSE_2
		printf("\n[GPU%" PRI_IDX "] get_padding( dim=%" PRI_IDX " )...Done.\n\tpadding=%" PRI_IDX "\n",
			device_id, dim, padded_dim);
	#endif

	return padded_dim;

} // get_padding

////////////////////////////////////////////////////////////////////////////////

/*
 * Computes the highest power of 2 <= x.
 * Returns the same value (x) if it is already a power of 2, or is zero.
 */
index_t prev_power_2( index_t x )
{

	if ( x & (x-1) ) {	// It is not already a power of 2.

		for ( index_t i = 0, b = 1 ; i <= (index_t) sizeof(index_t) ; i++, b <<= 1 )
			x |= ( x >> b );

		x -= (x >> 1);
	}

	return x;

} // prev_power_2

////////////////////////////////////////////////////////////////////////////////

/*
 * Sets the following GLOBAL variables:
 *
 * threadsPerBlock_pitch:
 *	Number of threads per block for kernels requiring a value multiple of <Kp> (denoted as 'pitch').
 *	threadsPerBlock <= threadsPerBlock_pitch <= maxThreadsPerBlock
 *
 * maxBlockHeight_pitch:
 *	Maximum block height using <threadsPerBlock_pitch> threads.
 *	maxBlockHeight_pitch <= (threadsPerBlock_pitch / pitch)
 *
 * Returns EXIT_SUCCESS, or EXIT_FAILURE if Kp > maxThreadsPerBlock.
 */
int static set_threadsPerBlock_pitch( void )
{

	#if NMFGPU_VERBOSE_2
		printf("\n[GPU%" PRI_IDX "] set_threadsPerBlock_pitch( Kp = %" PRI_IDX " )\n", device_id, Kp );
	#endif

	// ------------------

	index_t const pitch = Kp;

	if ( pitch > maxThreadsPerBlock ) {
		fflush(stdout);
		if ( (! device_id) + (num_devices == 1) )
			fprintf( stderr, "\nInvalid factorization rank. On this GPU device, it must be less than or equal to %"
					PRI_IDX ".\n", maxThreadsPerBlock );
		return EXIT_FAILURE;
	}

	// ------------------

	// Number of threads per block for kernels requiring a value multiple of <Kp> (denoted as 'pitch').
	threadsPerBlock_pitch = threadsPerBlock;

	// Maximum block height (result possibly truncated).
	maxBlockHeight_pitch = threadsPerBlock / pitch;

	/* Some kernels make use of shared memory if block height > 1.
	 * If necessary, increases "threadsPerBlock_pitch" to try to ensure this.
	 */
	if ( maxBlockHeight_pitch <= 2 ) {
		maxBlockHeight_pitch = (maxThreadsPerBlock / pitch);	// Result possibly truncated.

		// Final number of threads per block must be a multiple of <pitch>.
		threadsPerBlock_pitch = pitch * maxBlockHeight_pitch;	// <= maxThreadsPerBlock
	}

	// ------------------

	#if NMFGPU_VERBOSE_2
		printf("\n[GPU%" PRI_IDX "] set_threadsPerBlock_pitch( Kp = %" PRI_IDX " ) ... Done.\n\tmaxBlockHeight_pitch=%" PRI_IDX
			", threadsPerBlock_pitch=%" PRI_IDX "\n", device_id, Kp, maxBlockHeight_pitch, threadsPerBlock_pitch );
	#endif

	return EXIT_SUCCESS;

} // set_threadsPerBlock_pitch

// =======================================================================

/*
 * Returns the maximum matrix dimensions for this GPU architecture, using both Kp and memory_alignment
 * (i.e., the selected and the minimum possible rounded-up factorization ranks).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int max_bounds( index_t *__restrict__ maxDimension, index_t *__restrict__ maxDimension_minKp, index_t *__restrict__ max_MatrixSize )
{

	#if NMFGPU_VERBOSE_2
		printf("\n[GPU%" PRI_IDX "] max_bounds( Kp = %" PRI_IDX " )\n", device_id, Kp );
	#endif

	// Checks for NULL pointers
	if ( ! ( (size_t) maxDimension * (size_t) maxDimension_minKp * (size_t) max_MatrixSize ) ) {
		fflush( stdout );
		errno = EFAULT;
		if ( ! maxDimension ) perror("\nmax_bounds( maxDimension )");
		if ( ! maxDimension_minKp ) perror("\nmax_bounds( maxDimension_minKp )");
		if ( ! max_MatrixSize ) perror("\nmax_bounds( max_MatrixSize )");
		return EXIT_FAILURE;
	}

	// ---------------------------------

	/* Generic limit for operations that depend on <Kp>:
	 *	MAX(N, M) * Kp <= IDX_MAX
	 */
	index_t maxDim = (IDX_MAX / Kp);

	// Similar to maxDim, but using <memory_alignment> (i.e., the minimum value for Kp).
	index_t maxDim_minKp = ( IDX_MAX / memory_alignment );

	// Size limit on operations that do not depend on <Kp>.
	index_t maxSize = IDX_MAX;

		#if NMFGPU_DEBUG
		///////////////////////////////
		printf("\t--- [GPU%" PRI_IDX "]: Initial bounds: maxDim=%" PRI_IDX ", maxDim_minKp=%" PRI_IDX ", maxSize=%" PRI_IDX "\n",
			device_id, maxDim, maxDim_minKp, maxSize );
		//////////////////////////////
		#endif

	// ---------------------------------

	// Adjusts previous values for Compute Capability 1.x
	if ( computeCapability == 1 ) {

		index_t const maxBlockHeight_minKp = (threadsPerBlock / memory_alignment);

		// matrix_to_row.

		index_t mbh = prev_power_2( maxBlockHeight_pitch );
		size_t const maxDim_matrix2row = mbh * REDUCE_TO_ROW__ITEMS_PER_THREAD * ((1 << 24) - 1);	// might be > IDX_MAX

		mbh = prev_power_2( maxBlockHeight_minKp );
		size_t const maxDim_minKp_matrix2row = mbh * REDUCE_TO_ROW__ITEMS_PER_THREAD * ((1 << 24) - 1);	// might be > IDX_MAX

			#if NMFGPU_DEBUG
			///////////////////////////////
			printf("\t--- [GPU%" PRI_IDX "] (ComputeCapability:1.x): maxDim_matrix2row=%zu, maxDim_minKp_matrix2row=%zu\n",
				device_id, maxDim_matrix2row, maxDim_minKp_matrix2row);
			//////////////////////////////
			#endif

		// ---------------------------------

		// matrix_mul_div, matrix_adjust (i.e., other operations that depend on Kp).

		index_t const items_per_thread = MIN( MUL_DIV__ITEMS_PER_THREAD, ADJUST__ITEMS_PER_THREAD );
		size_t const maxDim_others = maxBlockHeight_pitch * items_per_thread * ((1 << 24) - 1);
		size_t const maxDim_minKp_others = maxBlockHeight_minKp * items_per_thread * ((1 << 24) - 1);
		// Both values might be > IDX_MAX

			#if NMFGPU_DEBUG
			///////////////////////////////
			printf("\t--- [GPU%" PRI_IDX "] (ComputeCapability:1.x): maxDim_others=%zu, maxDim_minKp_others=%zu\n",
				device_id, maxDim_others, maxDim_minKp_others );
			//////////////////////////////
			#endif

		// ---------------------------------

		// matrix_idx_max
		index_t block_width = memory_alignment;
		if ( K > memory_alignment ) {
			block_width = ( K / IDX_MAX__ITEMS_PER_THREAD );
			block_width = prev_power_2( (block_width << 1) );	// prev_power_2( x*2 ) == next_power_2( x )
			block_width = MIN( block_width, memory_alignment );	// Note that memory_alignment is also a power of 2.
		}
		size_t const maxDim_idxMax = (threadsPerBlock_pitch / block_width) * ((1 << 24) - 1);	// might be > IDX_MAX
		size_t const maxDim_minKp_idxMax = maxBlockHeight_minKp * ((1 << 24) - 1);		// might be > IDX_MAX

			#if NMFGPU_DEBUG
			///////////////////////////////
			printf("\t--- [GPU%" PRI_IDX "] (ComputeCapability:1.x): maxDim_idxMax=%zu, maxDim_minKp_idxMax=%zu\n",
				device_id, maxDim_idxMax, maxDim_minKp_idxMax );
			//////////////////////////////
			#endif

		// ---------------------------------

		// matrix_div_sub (size).

		size_t const max_size = threadsPerBlock * DIV_SUB__ITEMS_PER_THREAD * ((1 << 24) - 1);		// might be > IDX_MAX

			#if NMFGPU_DEBUG
			///////////////////////////////
			printf("\t--- [GPU%" PRI_IDX "] (ComputeCapability:1.x): max_size=%zu\n", device_id, max_size );
			//////////////////////////////
			#endif

		// ---------------------------------

		// Final values

		size_t const max_dim = MIN3( maxDim_matrix2row, maxDim_others, maxDim_idxMax );
		if ( max_dim < (size_t) maxDim )
			maxDim = (index_t) max_dim;		// <= IDX_MAX


		size_t const max_dim_minKp = MIN3( maxDim_minKp_matrix2row, maxDim_minKp_others, maxDim_minKp_idxMax );
		if ( max_dim_minKp < (size_t) maxDim_minKp )
			maxDim_minKp = (index_t) max_dim_minKp;	// <= IDX_MAX


		if ( max_size < (size_t) maxSize )
			maxSize = (index_t) max_size;		// <= IDX_MAX


	} // Compute Capability 1.x

	// ---------------------------------

	*maxDimension = maxDim;
	*maxDimension_minKp = maxDim_minKp;
	*max_MatrixSize = maxSize;

	#if NMFGPU_VERBOSE_2
		printf("\n[GPU%" PRI_IDX "] max_bounds( Kp = %" PRI_IDX " )... Done.\n\tmaxDimension=%" PRI_IDX ", maxDimension_minKp=%"
			PRI_IDX ", max_MatrixSize=%" PRI_IDX "\n", device_id, Kp, maxDim, maxDim_minKp, maxSize );
	#endif

	return EXIT_SUCCESS;

} // max_bounds

// =======================================================================

/*
 * Computes the size of data blocks to be transferred to the GPU.
 *
 * Assuming that:
 *	N, M: Total number of rows and columns of input matrix V,
 *	Mp: Total number of columns of V, including useless data for padding,
 *		If num_devices == 1, then NnP==N and MnP==M.
 *	NnP, MnP: Dimensions of the block from V to be processed by this GPU,
 *	Kp: Factorization rank with data for padding.
 *
 * this routine computes BLN <= NnP and BLM <= MnP so it can be loaded into the GPU:
 *	One BLN x Mp block from matrix d_Vrow.
 *	One N x BLMp block from matrix d_Vcol ('BLMp' means 'BLM' with padding) .
 *	One MAX( BLN x Mp , N x BLMp ) block for matrix WH (WH == W*H).
 *	Matrix W (N x Kp).
 *	Matrix H (M x Kp).
 *	One <MAX(N,M) x Kp> block for matrix Aux (i.e., Waux or Haux).
 *	One Kp-length vector for accum (i.e., accum_h or accum_w).
 *	Two constant values (scalar arguments for the CUBLAS Library),
 * where BLMp is the padding for BLM (BLMp <= MnPp). It is a multiple of memory_alignment.
 *
 * IF there is enough GPU memory for all matrices at full size:
 *	Sets BLN=NnP, BLM=MnP, BLMp=MnPp and full_matrix='true'.
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
		printf("\n[GPU%" PRI_IDX "] get_BLs( num_devices=%" PRI_IDX ", mem_size=%zu, do_classf=%i, NnP=%" PRI_IDX ", MnP=%"
			PRI_IDX ", MnPp=%" PRI_IDX ")\n", device_id, num_devices, mem_size, do_classf, NnP, MnP, MnPp);
	#endif

	// Initial output values (i.e., input matrix is small enough to be fully loaded into GPU memory).
	index_t lBLN = NnP;
	index_t lBLM = MnP;
	index_t lBLMp = MnPp;
	bool l_full_matrix = true;

	/* Forces block sizes to be the half of dimensions: */
	#if NMFGPU_FORCE_BLOCKS

		printf("\n[GPU%" PRI_IDX "] Forcing blocks size to the half of dimensions.\n", device_id);

		lBLN = (NnP/2) + (NnP % 2);

		if ( MnPp > memory_alignment ) {
			lBLMp = get_padding( MnPp/2 );	// MnPp is a multiple of memory_alignment and 2.
			lBLM = MIN( lBLMp, MnP );
		} else {
			lBLM = (MnP + 1) / 2;	// = Ceil( MnP/2 )
			lBLMp = get_padding( lBLM );
		}

		l_full_matrix = false;

	#endif	/* if NMFGPU_FORCE_BLOCKS */


	// -----------------------------


	/* Required Memory, expressed in number of "reals" (i.e., float or double), for matrices V and WH:
	 *
	 *	IF (! l_full_matrix  ||  num_devices > 1): BLN*Mp + N*BLMp + MAX(BLN*Mp , N*BLMp). // Vrow, Vcol and WH
	 *	ELSE (BLN==NnP==N && BLMp==MnPp==Mp): 2*BLN*BLMp // V and WH.
	 */
	size_t required_mem = 0;
	{
		size_t const sizeVr = lBLN * Mp;		// d_Vrow
		size_t const sizeVc = N * lBLMp;		// d_Vcol
		required_mem = (sizeVr + sizeVc);
		if ((! l_full_matrix) + (num_devices > 1))	// (! l_full_matrix  ||  num_devices > 1)
			required_mem += MAX( sizeVr, sizeVc );	// d_Vcol != d_Vrow. Therefore, we need a d_WH.
	}

	/* Required memory, expressed in number of "reals" (i.e., float or double),
	 * for matrices d_W, d_H, d_Aux, d_accum and the two scalar values:
	 */
	size_t data_matrices = Kp * (N + M + MAX(N,M)) + Kp + 2;

	// Memory used by classification vector (<Mp> integers) must be scaled since sizeof(index_t) <= sizeof(real)
	if ( do_classf ) {
		size_t const classf_data = (Mp * sizeof(index_t)) / sizeof(real);
		data_matrices += classf_data;
	}

	#if NMFGPU_FORCE_BLOCKS || NMFGPU_VERBOSE_2 || NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
		printf("\n[GPU%" PRI_IDX "] Total device memory required (approx.): %g MiB. (required by output & auxiliary data: %g MiB)\n",
			device_id, (float) (required_mem + data_matrices) * (sizeof(real) / 1048576.0f),
			(float) data_matrices * (sizeof(real) / 1048576.0f) );
	#endif


	// ---------------------------------


	// Adjusts lBLN and lBLM to the available memory.

	if ( (required_mem + data_matrices) > mem_size ) {

		#if NMFGPU_VERBOSE_2 || NMFGPU_FORCE_BLOCKS
			printf("\n[GPU%" PRI_IDX "] required_mem (%zu) > mem_size (%zu)\n",
				device_id, required_mem + data_matrices, mem_size);
		#endif

		// full_matrix is set to 'false'.
		l_full_matrix = false;

		/* NOTE: Memory for 'data_matrices' is always required.
		 *	Therefore, we subtract their size from 'mem_size'.
		 */
		size_t const free_memory = mem_size - data_matrices;

		#if NMFGPU_VERBOSE_2 || NMFGPU_FORCE_BLOCKS
			printf("\t[GPU%" PRI_IDX "] mem_size=%zu, data_matrices=%zu, required_mem=%zu free_memory=%zu\n",
				device_id, mem_size, data_matrices, required_mem, free_memory );
		#endif

		// ----------------------------

		// //////////////////////////
		//
		// Required Memory: max(BLN*Mp, N*BLMp) + BLN*Mp + N*BLMp,
		//	where  BLN = (NnP/dBLN) and  BLMp = padding(MnP/dBLM)
		//
		// Increments iteratively the denominators 'dBLN' and 'dBLM'
		// (one at a time, according to 'dim') until data_matrices fit into the memory.
		//
		// WARNING: if (MnP > memory_alignment), the resulting BLM must be a multiple of memory_alignment.
		//
		// //////////////////////////

		index_t dBLN = 2;	// Denominator for dimension N
		index_t dBLM = 2;	// Denominator for dimension M
		bool dimN = true;	// True: Adjusts BLN, False: Adjusts BLM

		size_t rows;		// d_Vrow: lBLN * Mp;

		// Initial values for dBLM == 2
		lBLMp = get_padding( (MnPp / 2) );	// Multiple of memory_alignment
		size_t cols = N * lBLMp;		// (d_Vcol)
		lBLM = MIN( lBLMp, MnP );

		#if NMFGPU_VERBOSE_2
			index_t step = 1;	// Number of loops.
		#endif

		do {

			if ( dimN ) {	// Adjusts BLN
				lBLN = ( NnP + dBLN - 1 ) / dBLN;	// (NnP / dBLN) + ( NnP % dBLN )
				rows = lBLN * Mp;
				dBLN++;		// Denominator in next loop with dim=1.
			} else {	// Adjusts BLM
				lBLMp = get_padding( ((MnP + dBLM - 1) / dBLM) );	// Multiple of memory_alignment
				lBLM = MIN( lBLMp, MnP );
				cols = N * lBLMp;
				dBLM++;		// Denominator in next loop with dim=0.
			}

			// Changes dimension for next loop.
			dimN = ( ! dimN );

			// Required memory:
			// size(d_Vr) = BLN * Mp	// (i.e., BLN rows)
			// size(d_Vc) = N * BLMp	// (i.e., BLM padded columns)
			// size(d_WH) = Max(size_Vr, size_Vc)

			required_mem = rows + cols + MAX( rows, cols ); // d_Vr + d_Vc + d_WH

			#if NMFGPU_VERBOSE_2
				printf("[GPU%" PRI_IDX "] Step %" PRI_IDX ": BLN=%" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%" PRI_IDX
					" (%g MiB), dBLN=%" PRI_IDX ", dBLM=%" PRI_IDX ", required_mem=%zu\n", device_id, step, lBLN, lBLM,
					lBLMp, (float) ((required_mem + data_matrices) * sizeof(real)) / 1048576.0f, dBLN, dBLM, required_mem );
				step++;
			#endif

		} while ( (required_mem > free_memory) * (lBLMp > memory_alignment) * (lBLN > 0) );
		// while ( (required_mem > free_memory) && (lBLMp > memory_alignment) && (lBLN > 0) )


		if ( required_mem > free_memory ) {

			if ( lBLMp == memory_alignment ) {	// BLN >= 0

				// Sets lBLM (and lBLMp) to a fixed value, and reduces BLN.
				lBLM = MIN(memory_alignment, MnP);
				cols = N * memory_alignment;

				if ( rows >= cols )	// 2*BLN*Mp + N*memory_alignment == 2*rows + cols <= free_memory
					lBLN = (free_memory - cols) / (2 * Mp);
				else			// BLN*Mp + 2*N*memory_alignment == rows + 2*cols <= free_memory
					lBLN = (free_memory - 2*cols) / Mp;

				lBLN = MIN( lBLN , N );

			} else {	// (lBLMp > memory_alignment) && (lBLN == 0)

				// Sets BLN to 1, and reduces BLM and BLMp
				lBLN = 1;
				rows = Mp;	// == BLN*Mp

				if ( rows >= cols )
					lBLMp = (free_memory - 2*Mp) / N;	// 2*Mp + N*BLMp == 2*rows + cols <= free_mem
				else
					lBLMp = (free_memory - Mp) / (2 * N);	// Mp + 2*N*BLMp == rows + 2*cols <= free_mem

				// BLMp might not be a multiple of memory_alignment.
				index_t rem = lBLMp % memory_alignment;
				lBLMp -= rem;			// Multiple of memory_alignment
				lBLM = MIN( lBLMp, MnP );
			}

			#if NMFGPU_VERBOSE_2
				rows = lBLN * Mp;
				cols = N * lBLMp;
				required_mem = rows + cols + MAX( rows, cols ); // d_Vr + d_Vc + d_WH == MAX( d_Vr , d_Vc )
				printf("[GPU%" PRI_IDX "] Resulting values: BLN=%" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%" PRI_IDX
					" (approx. %g MiB)\n", device_id, lBLN, lBLM, lBLMp,
					(float) (required_mem + data_matrices) * (sizeof(real) / 1048576.0f) );
			#endif

			if ( (lBLN < 1) + (lBLM < 1) ) {	// (lBLN < 1) || (lBLM < 1)
				// Minimum required: lBLN=1 && lBLMp=memory_alignment.
				cols = N * memory_alignment;		// i.e., N*BLMp
				required_mem = MAX( ((size_t) Mp), cols ) + Mp + cols;	// d_WH + d_Vrow + d_Vcol
				fflush(stdout);
				fprintf(stderr,"\n[GPU%" PRI_IDX "] Not enough memory. Minimum required: %g MiB.\n", device_id,
						(float) ((required_mem + data_matrices) * sizeof(real))/1048576.0f);
				return EXIT_FAILURE;
			}
		} // if ( required_mem > free_mem )

	} // if ( required_mem > free_mem )


	#if NMFGPU_FORCE_BLOCKS || NMFGPU_VERBOSE || NMFGPU_VERBOSE_2 || NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
		printf("\n[GPU%" PRI_IDX "] Resulting values: BLN=%" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%" PRI_IDX
			" (approx. %g MiB), full_matrix=%i\n", device_id, lBLN, lBLM, lBLMp,
			(float) (required_mem + data_matrices) * (sizeof(real) / 1048576.0f), l_full_matrix );
	#endif

	*BLN = lBLN;
	*BLM = lBLM;
	*BLMp = lBLMp;
	*full_matrix = l_full_matrix;

	return EXIT_SUCCESS;

} // get_BLs

// =======================================================================

/*
 * Initializes a "block_D" data (D == N or M).
 *
 * Data for blockwise processing is stored in a block_t structure. This method initializes such structure.
 *
 * D: dimension in HOST (i.e., NnP or MnP)
 * BLD: dimension in DEVICE (i.e., BLN or BLM)
 * BLDp: padding for BLD (i.e., 0 or BLMp).
 *
 * Divides dimension "D" in <num_steps> blocks of length "BLD" and possibly, one more block of length <D % BLD>.
 * In any case, the length information of the last block (length == BLD or <D % BLD>) is always stored in block_D.BL[1],
 * EXCEPT if D == BLD (in this case, it is only stored in block_D.BL[0], and sets block_D.num_steps[0..1] to {1,0}).
 */
static void init_block_conf( index_t D, index_t BLD, index_t BLDp, block_t *__restrict__ const block_D )
{

	block_D->BL[0] = BLD;
	block_D->BLp[0] = BLDp;

	if ( D > BLD ) { // Must perform blockwise processing

		div_t const divResult = div( D, BLD );

		if ( divResult.rem > 0 ) {	// Last block with length == <D % BLD>.
			block_D->num_steps[0] = divResult.quot;	// Number of blocks of size BLD
			block_D->BL[1] = divResult.rem;		// Last block information

			// Padding for this block (if any).
			if ( BLDp )
				block_D->BLp[1] = get_padding(divResult.rem);
				/* WARNING:
				 * If (<Total width of previous blocks> + this_padding) > get_padding( D ):
				 *	GPU: It doesn't matter since pitch_Vcol >= this_padding
				 *	CPU: *PLEASE TAKE CARE* to not go out-of-bounds when transfering data.
				 */
			else
				block_D->BLp[1] = 0;

		} else {			// Last block with length == BLD
			block_D->num_steps[0] = divResult.quot - 1;
			block_D->BL[1] = BLD;			// Same configuration as block 0.

			block_D->BLp[1] = BLDp;
				/* WARNING:
				 * If (<Total width of previous blocks> + <this padding>) > get_padding( D ):
				 *	GPU: It doesn't matter since pitch_Vcol >= <this padding>
				 *	CPU: *PLEASE TAKE CARE* to not go out-of-bounds when transferring data.
				 */
		}

		block_D->num_steps[1] = 1;

	} else {  // D == BLD : There is only one block (number 0)

		block_D->BL[1] = 0;
		block_D->BLp[1] = 0;
		block_D->num_steps[0] = 1;
		block_D->num_steps[1] = 0;

	} // if (D > BLD)

	// Prints block configuration
	#if NMFGPU_VERBOSE || NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
		printf("\t[GPU%" PRI_IDX "] %" PRI_IDX " block(s) of length %" PRI_IDX " (padding=%" PRI_IDX ")\n", device_id,
			block_D->num_steps[0], BLD, BLDp);
		if ( block_D->num_steps[1] )
			printf("\t[GPU%" PRI_IDX "] 1 block of length %" PRI_IDX " (padding=%" PRI_IDX ")\n", device_id, block_D->BL[1],
				block_D->BLp[1]);
	#endif

} // init_block_conf

// =======================================================================

/*
 * Allocates device memory for the matrices.
 *
 * single_matrix_V: 'True' if input matrix V is small enough to be fully loaded into the GPU memory.
 *		In this case, d_Vrow = d_Vcol.
 *
 * do_classf: Set to 'true' if classification vector will be computed.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int allocate_dev_mem( index_t BLN, index_t BLMp, bool single_matrix_V, bool do_classf )
{

	#if NMFGPU_VERBOSE_2
		printf("\n[GPU%" PRI_IDX "] Allocating device memory (BLN=%" PRI_IDX ", BLMp=%" PRI_IDX
			", single_matrix_V: %i, do_classf: %i)...\n", device_id, BLN, BLMp, single_matrix_V, do_classf );
	#endif

	cudaError_t cuda_status = cudaSuccess;
	size_t size, size2;	// Size of data matrices.

	// ---------------------------------

	// d_Vrow:
	size = BLN * Mp;
	cuda_status = cudaMalloc((void **)&d_Vrow, size * sizeof(real));
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Device memory allocation error (d_Vrow): %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		return EXIT_FAILURE;
	}

	// d_Vcol:
	size2 = N * BLMp;

	if ( single_matrix_V )		// Fully allocates matrix V (i.e., 1 Process, GPU and full_matrix).
		d_Vcol = d_Vrow;	// Just shares the allocated memory.

	else { // Multiple process OR Matrix V is too large for this GPU.
		cuda_status = cudaMalloc((void **)&d_Vcol, size2 * sizeof(real));
		if ( cuda_status != cudaSuccess ) {
			fflush(stdout);
			fprintf(stderr, "\n[GPU%" PRI_IDX "] Device memory allocation error (d_Vcol): %s\n",
				device_id, cudaGetErrorString(cuda_status) );
			cudaFree(d_Vrow);
			return EXIT_FAILURE;
		}
	}


	// d_WH: (BLN x Mp) OR (N x BLMp)
	size = MAX( size, size2 );
	cuda_status = cudaMalloc((void **)&d_WH, size * sizeof(real));
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf(stderr, "\n[GPU%" PRI_IDX "] Device memory allocation error (d_WH): %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		if (d_Vcol != d_Vrow){ cudaFree(d_Vcol); } cudaFree(d_Vrow);
		return EXIT_FAILURE;
	}

	// ---------------------------------

	// d_W
	size = N * Kp;
	cuda_status = cudaMalloc((void **)&d_W, size * sizeof(real));
	if( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf(stderr, "\n[GPU%" PRI_IDX "] Device memory allocation error (d_W): %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		cudaFree(d_WH); if (d_Vcol != d_Vrow){ cudaFree(d_Vcol); } cudaFree(d_Vrow);
		return EXIT_FAILURE;
	}

	// d_H (transp)
	size2 = M * Kp;
	cuda_status = cudaMalloc((void **)&d_H, size2 * sizeof(real));
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf(stderr, "\n[GPU%" PRI_IDX "] Device memory allocation error (d_H): %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		cudaFree(d_W); cudaFree(d_WH); if (d_Vcol != d_Vrow){ cudaFree(d_Vcol); } cudaFree(d_Vrow);
		return EXIT_FAILURE;
	}

	// d_Aux: N x Kp (i.e., Waux), or M x Kp (i.e., Haux).
	// Actually, it just uses MAX(BLN,BLM)*Kp, but matrix_to_row() might require up to <MAX(N,M) * Kp>.
	if ( size < size2 ) { size = size2; }
	cuda_status = cudaMalloc((void **)&d_Aux, size * sizeof(real));
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf(stderr, "\n[GPU%" PRI_IDX "] Device memory allocation error (d_Aux): %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		cudaFree(d_H); cudaFree(d_W); cudaFree(d_WH); if (d_Vcol != d_Vrow){ cudaFree(d_Vcol); } cudaFree(d_Vrow);
		return EXIT_FAILURE;
	}

	// ---------------------------------

	// Classification vector (if necessary)
	if ( do_classf ) {
		cuda_status = cudaMalloc((void **)&d_classification, Mp * sizeof(index_t) );
		if ( cuda_status != cudaSuccess ) {
			fflush(stdout);
			fprintf( stderr, "\n[GPU%" PRI_IDX "] Device memory allocation error (classification): %s\n",
				device_id, cudaGetErrorString(cuda_status) );
			cudaFree(d_Aux); cudaFree(d_H); cudaFree(d_W); cudaFree(d_WH);
			if (d_Vcol != d_Vrow){ cudaFree(d_Vcol); } cudaFree(d_Vrow);
			return EXIT_FAILURE;
		}
	}

	// ---------------------------------

	// d_accum
	cuda_status = cudaMalloc((void **)&d_accum, Kp * sizeof(real) );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf(stderr, "\n[GPU%" PRI_IDX "] Device memory allocation error (d_accum): %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		if ( do_classf ){ cudaFree(d_classification); }
		cudaFree(d_Aux); cudaFree(d_H); cudaFree(d_W); cudaFree(d_WH);
		if (d_Vcol != d_Vrow){ cudaFree(d_Vcol); } cudaFree(d_Vrow);
		return EXIT_FAILURE;
	}

	// ---------------------------------

	// Scalar values for CUBLAS Library.
	cuda_status = cudaMalloc((void **)&d_scalar, 2 * sizeof(real) );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf(stderr, "\n[GPU%" PRI_IDX "] Device memory allocation error (d_scalar): %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		cudaFree(d_accum);
		if ( do_classf ){ cudaFree(d_classification); }
		cudaFree(d_Aux); cudaFree(d_H); cudaFree(d_W); cudaFree(d_WH);
		if (d_Vcol != d_Vrow){ cudaFree(d_Vcol); } cudaFree(d_Vrow);
		return EXIT_FAILURE;
	}

	// Sets pointers to d_scalar[]
	d_zero = &d_scalar[0];
	d_one  = &d_scalar[1];

	// ---------------------------------

	#if NMFGPU_VERBOSE_2
		printf("\n[GPU%" PRI_IDX "] Allocating device memory for the matrices... done\n", device_id);
	#endif

	return EXIT_SUCCESS;

} // allocate_dev_mem

// =======================================================================

/*
 * Initializes associated GPU data, such as CUDA Streams and/or CUDA Events.
 *
 * single_matrix_V: Set to 'true' if NO blockwise processing is necessary (i.e., input matrix is small enough to be fully loaded),
 *		so that Vrow == Vcol.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int init_GPU_data( bool single_matrix_V )
{

	#if NMFGPU_VERBOSE_2
		printf("\n[GPU%" PRI_IDX "] Initializing GPU data (single matrix=%i, num_streams_NMF=%" PRI_IDX ")...\n",
			device_id, single_matrix_V, num_streams_NMF );
	#endif

	cudaError_t cuda_status = cudaSuccess;

	// ----------------------------------

	/* CUDA Events for synchronization. */

	// Event for Vrow (i.e., set of rows from matrix V)
	cuda_status = cudaEventCreateWithFlags( &event_Vrow, cudaEventDisableTiming );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Error creating event object for Vrow: %s\n", device_id, cudaGetErrorString(cuda_status) );
		cudaEventDestroy( event_Vrow );
		return EXIT_FAILURE;
	}


	// Event for Vcol (i.e., set of columns from matrix V)
	if ( single_matrix_V )
		event_Vcol = event_Vrow;	// Same matrix (Vcol == Vrow), same events

	else {
		cuda_status = cudaEventCreateWithFlags( &event_Vcol, cudaEventDisableTiming );
		if ( cuda_status != cudaSuccess ) {
			fflush(stdout);
			fprintf( stderr, "\n[GPU%" PRI_IDX "] Error creating event object for Vcol: %s\n",
				device_id, cudaGetErrorString(cuda_status) );
			cudaEventDestroy( event_Vrow );
			return EXIT_FAILURE;
		}
	} // if ( single_matrix_V )


	// Event for d_W
	cuda_status = cudaEventCreateWithFlags( &event_W, cudaEventDisableTiming );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr,"\n[GPU%" PRI_IDX "] Error creating event object for d_W: %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		if( ! single_matrix_V ){ cudaEventDestroy( event_Vcol ); } cudaEventDestroy( event_Vrow );
		return EXIT_FAILURE;
	}

	// Event for d_H.
	cuda_status = cudaEventCreateWithFlags( &event_H, cudaEventDisableTiming );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Error creating event object for d_H: %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		cudaEventDestroy( event_W );
		if ( ! single_matrix_V ) { cudaEventDestroy( event_Vcol ); } cudaEventDestroy( event_Vrow );
		return EXIT_FAILURE;
	}

	// Event for matrix reduction.
	cuda_status = cudaEventCreateWithFlags( &event_reduction, cudaEventDisableTiming );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Error creating event object for matrix reduction: %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		cudaEventDestroy( event_H ); cudaEventDestroy( event_W );
		if ( ! single_matrix_V ) { cudaEventDestroy( event_Vcol ); } cudaEventDestroy( event_Vrow );
		return EXIT_FAILURE;
	}



	// ---------------------------------

	/* CUDA Streams for synchronization */

	// Stream for Vrow (i.e., set of rows from matrix V)
	cuda_status = cudaStreamCreate( &stream_Vrow );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Error creating stream object for Vrow: %s\n", device_id, cudaGetErrorString(cuda_status) );
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
			fflush(stdout);
			fprintf( stderr, "\n[GPU%" PRI_IDX "] Error creating stream object for Vcol: %s\n", device_id,
				cudaGetErrorString(cuda_status) );
			cudaStreamDestroy( stream_Vrow );
			cudaEventDestroy( event_reduction ); cudaEventDestroy( event_H ); cudaEventDestroy( event_W );
			cudaEventDestroy( event_Vcol ); cudaEventDestroy( event_Vrow );
			return EXIT_FAILURE;
		}
	} // if ( single_matrix_V )


	// Stream for d_W
	cuda_status = cudaStreamCreate( &stream_W );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Error creating stream object for d_W: %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		if ( ! single_matrix_V ) { cudaStreamDestroy( stream_Vcol ); } cudaStreamDestroy( stream_Vrow );
		cudaEventDestroy( event_reduction ); cudaEventDestroy( event_H ); cudaEventDestroy( event_W );
		if ( ! single_matrix_V ) { cudaEventDestroy( event_Vcol ); } cudaEventDestroy( event_Vrow );
		return EXIT_FAILURE;
	}

	// Stream for d_W
	cuda_status = cudaStreamCreate( &stream_H );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Error creating stream object for d_H: %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		cudaStreamDestroy( stream_W );
		if ( ! single_matrix_V ) { cudaStreamDestroy( stream_Vcol ); } cudaStreamDestroy( stream_Vrow );
		cudaEventDestroy( event_reduction ); cudaEventDestroy( event_H ); cudaEventDestroy( event_W );
		if ( ! single_matrix_V ) { cudaEventDestroy( event_Vcol ); } cudaEventDestroy( event_Vrow );
		return EXIT_FAILURE;
	}

	// Main-flow streams
	streams_NMF = (cudaStream_t *) malloc( num_streams_NMF * sizeof(cudaStream_t) );
	if( ! streams_NMF ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Error allocating HOST memory for CUDA streams (Main flow, length=%" PRI_IDX
			").\nmalloc: %s.\n", device_id, num_streams_NMF, strerror(errno) );
		cudaStreamDestroy( stream_H ); cudaStreamDestroy( stream_W );
		if ( ! single_matrix_V ) { cudaStreamDestroy( stream_Vcol ); } cudaStreamDestroy( stream_Vrow );
		cudaEventDestroy( event_reduction ); cudaEventDestroy( event_H ); cudaEventDestroy( event_W );
		if ( ! single_matrix_V ) { cudaEventDestroy( event_Vcol ); } cudaEventDestroy( event_Vrow );
		return EXIT_FAILURE;
	}
	for ( index_t st=0; st<num_streams_NMF; st++ ) {
		cuda_status = cudaStreamCreate( streams_NMF + st );
		if ( cuda_status != cudaSuccess ) {
			fflush(stdout);
			fprintf( stderr, "\n[GPU%" PRI_IDX "] Error creating stream object %" PRI_IDX "/%" PRI_IDX
					" for synchronization on main flow: %s\n", device_id, st, num_streams_NMF,
					cudaGetErrorString(cuda_status) );
			for ( index_t i=0 ; i < (st-1) ; i++ ) { cudaStreamDestroy( streams_NMF[ st ] ); } free( (void *) streams_NMF );
			cudaStreamDestroy( stream_H ); cudaStreamDestroy( stream_W );
			if ( ! single_matrix_V ) { cudaStreamDestroy( stream_Vcol ); } cudaStreamDestroy( stream_Vrow );
			cudaEventDestroy( event_reduction ); cudaEventDestroy( event_H ); cudaEventDestroy( event_W );
			if ( ! single_matrix_V ) { cudaEventDestroy( event_Vcol ); } cudaEventDestroy( event_Vrow );
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
			if ( ! single_matrix_V ) { cudaStreamDestroy( stream_Vcol ); } cudaStreamDestroy( stream_Vrow );
			cudaEventDestroy( event_reduction ); cudaEventDestroy( event_H ); cudaEventDestroy( event_W );
			if ( ! single_matrix_V ) { cudaEventDestroy( event_Vcol ); } cudaEventDestroy( event_Vrow );
			return EXIT_FAILURE;
		}
	#endif

	// -----------------------------------

	#if NMFGPU_VERBOSE_2
		printf("\n[GPU%" PRI_IDX "] Initializing GPU data (single matrix=%i, num_streams_NMF=%" PRI_IDX ")... Done.\n",
			device_id, single_matrix_V, num_streams_NMF );
	#endif

	return EXIT_SUCCESS;

} // init_GPU_data

// =======================================================================

/* Initializes scalar values for CUBLAS Library.
 *	d_scalar[2] = { 0.0, 1.0 }
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int init_scalars( void )
{

	// Scalars to upload.
	real const scalar[2] = { REAL_C(0.0), REAL_C(1.0) };

	cudaError_t cuda_status =
			cudaMemcpyAsync( (void *)d_scalar, (void *)scalar, 2 * sizeof(real), cudaMemcpyHostToDevice, streams_NMF[0] );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Could not upload scalar values to d_scalar[]: %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // init_scalars

////////////////////////////////////////////////////////////////////////////////

/*
 * Destroys associated GPU data, such as CUDA Streams and/or CUDA Events.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int finalize_GPU_data( void )
{

	#if NMFGPU_VERBOSE_2
		printf( "\n[GPU%" PRI_IDX "] Finalizing GPU data (num_streams_NMF=%" PRI_IDX ")...\n", device_id, num_streams_NMF );
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
			fflush(stdout);
			fprintf( stderr, "\n[GPU%" PRI_IDX "] Error destroying stream object %" PRI_IDX "/%" PRI_IDX
					" for synchronization on main flow: %s\n", device_id, st, num_streams_NMF,
					cudaGetErrorString(cuda_status) );
			status = EXIT_FAILURE;
		}
	}
	free( (void *) streams_NMF );


	// Stream for d_H
	cuda_status = cudaStreamDestroy( stream_H );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr,"\n[GPU%" PRI_IDX "] Error destroying stream object for d_H: %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}

	// Stream for d_W
	cuda_status = cudaStreamDestroy( stream_W );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr,"\n[GPU%" PRI_IDX "] Error destroying stream object for d_W: %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}

	// Stream for Vcol (i.e., set of columns from matrix V)
	if ( stream_Vcol != stream_Vrow ) {
		cuda_status = cudaStreamDestroy( stream_Vcol );
		if ( cuda_status != cudaSuccess ) {
			fflush(stdout);
			fprintf( stderr, "\n[GPU%" PRI_IDX "] Error destroying stream object for Vcol: %s\n", device_id,
				cudaGetErrorString(cuda_status) );
			status = EXIT_FAILURE;
		}
	} // if ( stream_Vcols != stream_Vrows )


	// Stream for Vrow (i.e., set of rows from matrix V)

	cuda_status = cudaStreamDestroy( stream_Vrow );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Error destroying stream object for Vrow: %s\n", device_id,
			cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}

	// ----------------------------------

	/* CUDA Events for synchronization. */

	// Event for matrix reduction
	cuda_status = cudaEventDestroy( event_reduction );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Error destroying event object for matrix reduction: %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}

	// Event for d_H
	cuda_status = cudaEventDestroy( event_H );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Error destroying event object for d_H: %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}


	// Event for d_W
	cuda_status = cudaEventDestroy( event_W );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Error destroying event object for d_W: %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}


	// Event for Vcol (i.e., set of columns from matrix V)
	if ( event_Vcol != event_Vrow ) {
		cuda_status = cudaEventDestroy( event_Vcol );
		if ( cuda_status != cudaSuccess ) {
			fflush(stdout);
			fprintf( stderr, "\n[GPU%" PRI_IDX "] Error destroying event object for Vcol: %s\n", device_id,
				cudaGetErrorString(cuda_status) );
			status = EXIT_FAILURE;
		}
	} // if ( event_Vcols != event_Vrows )


	// Event for Vrow (i.e., set of rows from matrix V)
	cuda_status = cudaEventDestroy( event_Vrow );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Error destroying event object for Vrow: %s\n", device_id,
			cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}

	// ---------------------------------

	#if NMFGPU_VERBOSE_2
		printf( "\n[GPU%" PRI_IDX "] Finalizing GPU data (num_streams_NMF=%" PRI_IDX ")... Done.\n", device_id, num_streams_NMF );
	#endif

	return status;

} // finalize_GPU_data

// =======================================================================

/*
 * Frees all allocated device memory.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int free_dev_mem( void )
{

	#if NMFGPU_VERBOSE_2
		printf("\n[GPU%" PRI_IDX "] Device memory clean-up...\n", device_id );
	#endif

	int status = EXIT_SUCCESS;	// Return status
	cudaError_t cuda_status = cudaSuccess;

	// -----------------------------------

	cuda_status = cudaFree( (void *) d_scalar);
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf(stderr, "\n[GPU%" PRI_IDX "] Error detected while freeing device memory (d_scalar): %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}

	cuda_status = cudaFree( (void *) d_accum);
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf(stderr, "\n[GPU%" PRI_IDX "] Error detected while freeing device memory (d_accum): %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}

	if ( d_classification && ( (cuda_status=cudaFree( (void *) d_classification)) != cudaSuccess ) ) {
		fflush(stdout);
		fprintf(stderr, "\n[GPU%" PRI_IDX "] Error detected while freeing device memory (d_classification): %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}

	cuda_status = cudaFree( (void *) d_Aux);
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf(stderr, "\n[GPU%" PRI_IDX "] Error detected while freeing device memory (d_Aux): %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}

	cuda_status = cudaFree( (void *) d_W);
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf(stderr, "\n[GPU%" PRI_IDX "] Error detected while freeing device memory (d_W): %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}

	cuda_status = cudaFree( (void *) d_H);
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf(stderr, "\n[GPU%" PRI_IDX "] Error detected while freeing device memory (d_H): %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}

	cuda_status = cudaFree( (void *) d_WH);
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf(stderr, "\n[GPU%" PRI_IDX "] Error detected while freeing device memory (d_WH): %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}

	if ( (d_Vrow != d_Vcol) && ( (cuda_status=cudaFree( (void *) d_Vcol)) != cudaSuccess ) ) {
		fflush(stdout);
		fprintf(stderr, "\n[GPU%" PRI_IDX "] Error detected while freeing device memory (d_Vcol): %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}

	cuda_status = cudaFree( (void *) d_Vrow);
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf(stderr, "\n[GPU%" PRI_IDX "] Error detected while freeing device memory (d_Vrow): %s\n",
			device_id, cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}

	// -----------------------------

	#if NMFGPU_VERBOSE_2
		printf("\n[GPU%" PRI_IDX "] Device memory clean-up... done\n", device_id );
	#endif

	return status;

} // free_dev_mem

////////////////////////////////////////////////////////////////////////////////

/*
 * Initializes the GPU device.
 *
 * Initializes CUDA, CUBLAS and data structures.
 * Computes the required amount of memory and allocates memory for data matrices.
 *
 * do_classf: Set to 'true' if classification vector will be computed.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int init_GPUdevice( size_t mem_size, bool do_classf )
{

	#if NMFGPU_VERBOSE_2
		printf("\n[GPU%" PRI_IDX "] init_GPUdevice( num_devices=%" PRI_IDX ",mem_size=%zu,do_classf=%i)\n",
			device_id, num_devices, mem_size, do_classf );
	#endif

	// --------------------------

	/* Sets the following GLOBAL variables:
	 *
	 * threadsPerBlock_pitch:
	 *	Number of threads per block for kernels requiring a value multiple of <Kp> (denoted as 'pitch').
	 *	threadsPerBlock <= threadsPerBlock_pitch <= maxThreadsPerBlock
	 *
	 * maxBlockHeight_pitch:
	 *	Maximum block height using <threadsPerBlock_pitch> threads.
	 *	maxBlockHeight_pitch <= (threadsPerBlock_pitch / pitch)
	 *
	 * Fails if Kp > maxThreadsPerBlock.
	 */
	if ( set_threadsPerBlock_pitch() != EXIT_SUCCESS )
		return EXIT_FAILURE;

	// -------------------------------------

	// Maximum matrix dimensions and size for this GPU architecture.
	index_t maxDim_GPU = IDX_MAX;
	index_t maxSize_GPU = IDX_MAX;

	// Similar to maxDim_GPU, but using <memory_alignment> (i.e., the minimum value for Kp).
	index_t maxDim_minKp = IDX_MAX;

	if ( max_bounds( &maxDim_GPU, &maxDim_minKp, &maxSize_GPU ) == EXIT_FAILURE )
		return EXIT_FAILURE;

	// Fails if any of N or M are out of bounds.
	if ( MAX(N,M) > maxDim_GPU ) {
		fflush(stdout);
		if ( (! device_id) + (num_devices == 1) ) {
			fprintf( stderr, "\nError: matrix dimensions exceed the limits for this GPU device. Both number of rows and columns "
					"must be less than or equal to %" PRI_IDX ".\n", maxDim_GPU );
			// Informs the user if it is possible to reduce the factorization rank
			if ( MAX(N,M) <= maxDim_minKp )	// (Kp > memory_alignment)
				fprintf( stderr, "Note, however, this limit is also affected by the selected (maximum) factorization rank.\n"
						"In particular, this matrix can be processed using %" PRI_IDX " factors (perhaps some more).\n",
					memory_alignment );
		}
		return EXIT_FAILURE;
	}

	// -------------------------------------

	/* Computes the size of data blocks to be transferred to the GPU.
	 *
	 * Assuming that:
	 *	N, M: Total number of rows and columns of input matrix V,
	 *	Mp: Total number of columns of V, including useless data for padding,
	 *	NnP, MnP: Dimensions of the block from V to be processed by this GPU,
	 *		If num_devices == 1, then NnP==N and MnP==M.
	 *	Kp: Factorization rank with data for padding.
	 *
	 * this routine computes BLN <= NnP and BLM <= MnP so it can be loaded into the GPU:
	 *	One BLN x Mp block from matrix d_Vrow.
	 *	One N x BLMp block from matrix d_Vcol ('BLMp' means 'BLM' with padding) .
	 *	One MAX( BLN x Mp , N x BLMp ) block for matrix WH (WH == W*H).
	 *	Matrix W (N x Kp).
	 *	Matrix H (M x Kp).
	 *	One <MAX(N,M) x Kp> block for matrix Aux (i.e., Waux or Haux).
	 *	One Kp-length vector for accum (i.e., accum_h or accum_w).
	 *	Two constant values (scalar arguments for the CUBLAS Library),
	 * where BLMp is the padding for BLM (BLMp <= MnPp). It is a multiple of memory_alignment.
	 *
	 * IF there is enough GPU memory for all matrices at full size:
	 *	Sets BLN=NnP, BLM=MnP, BLMp=MnPp and full_matrix=1.
	 * ELSE,
	 *	full_matrix is set to 0.
	 *	Computes BLN, BLM and BLMp for d_WH, d_Vr and d_Vc.
	 *
	 * do_classf: Set to 'true' if classification vector will be computed.
	 */

	index_t BLN=NnP, BLM=MnP, BLMp=MnPp;
	bool full_matrix = true;

	if ( get_BLs( mem_size, do_classf, &BLN, &BLM, &BLMp, &full_matrix ) == EXIT_FAILURE )
		return EXIT_FAILURE;

	// -------------------------------------

	// Checks the maximum matrix size.
	{
		index_t const sizeVr = BLN * Mp;
		index_t const sizeVc = N * BLMp;

		if ( maxSize_GPU < MIN( sizeVr, sizeVc ) ) {
			fflush(stdout);
			if ( (! device_id) + (num_devices == 1) )
				fprintf( stderr, "\nError: matrix size exceed the limits for this GPU device.\n"
						"Current limit is set to %" PRI_IDX " items (not bytes).\n", maxSize_GPU );
			return EXIT_FAILURE;
		}
	}

	// -------------------------------------

	// Initializes block information structures.

	// block_N
	#if NMFGPU_FORCE_BLOCKS || NMFGPU_VERBOSE || NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
		printf("\n[GPU%" PRI_IDX "] For dimension \"NnP\" = %" PRI_IDX " (N=%" PRI_IDX ", %" PRI_IDX " devices):\n",
			device_id, NnP, N, num_devices);
	#endif
	init_block_conf( NnP, BLN, 0, &block_N );

	// block_M
	#if NMFGPU_FORCE_BLOCKS || NMFGPU_VERBOSE || NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
		printf("\n[GPU%" PRI_IDX "] For dimension \"MnP\" = %" PRI_IDX " (%" PRI_IDX " with padding ; M=%" PRI_IDX ", %" PRI_IDX
			" devices):\n", device_id, MnP, MnPp, M, num_devices );
	#endif
	init_block_conf( MnP, BLM, BLMp, &block_M );

	// ----------------------------------

	// Block configuration testing. No memory is allocated on the device.

	#if NMFGPU_TEST_BLOCKS
		fflush( NULL );
		return EXIT_FAILURE;
	#endif

	// ----------------------------------

	// Allocates device memory.

	// Matrix V can be fully loaded into the GPU memory (i.e., d_Vrow == d_Vcol)
	// if (full_matrix == true) && (num_devices == 1)
	bool const single_matrix_V = ( full_matrix * (num_devices == 1) );

	if ( allocate_dev_mem( BLN, BLMp, single_matrix_V, do_classf ) == EXIT_FAILURE )
		return EXIT_FAILURE;

	// -----------------------------------

	// Initializes GPU data (CUDA Streams and CUDA Events).

	// Number of main-flow streams.
	num_streams_NMF = MAX( (block_N.num_steps[0] + block_N.num_steps[1]), (block_M.num_steps[0] + block_M.num_steps[1]) );

	if ( init_GPU_data( single_matrix_V ) == EXIT_FAILURE ) {
		free_dev_mem();
		return EXIT_FAILURE;
	}

	// -----------------------------------

	// Initializes the scalar values.
	if ( init_scalars() == EXIT_FAILURE ) {
		finalize_GPU_data();
		free_dev_mem();
		return EXIT_FAILURE;
	}

	// -----------------------------------

	// Finally, checks if everything was successfully initialized.

	if ( check_cuda_status() == EXIT_FAILURE ) {
		fprintf( stderr, "\n[GPU%" PRI_IDX "] Error setting-up GPU data.\n", device_id );
		finalize_GPU_data();
		free_dev_mem();
		return EXIT_FAILURE;
	}

	// --------------------------

	#if NMFGPU_VERBOSE_2
		printf("\n[GPU%" PRI_IDX "] init_GPUdevice( num_devices=%" PRI_IDX ",mem_size=%zu,do_classf=%i)... Done.\n",
			device_id, num_devices, mem_size, do_classf );
	#endif

	return EXIT_SUCCESS;

} // init_GPUdevice

////////////////////////////////////////////////////////////////////////////////

/*
 * Shuts-down current CUBLAS/CUDA context.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int finalize_GPU( void )
{

	#if NMFGPU_VERBOSE_2
		printf("\n[GPU%" PRI_IDX "] Shutting down CUDA/CUBLAS...\n", device_id );
	#endif

	int status = EXIT_SUCCESS;	// Return status

	// --------------------------------------

	cublasStatus_t cublas_status = cublasDestroy( cublas_handle );
	if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
		fflush(stdout);
		fprintf(stderr,"\n[GPU%" PRI_IDX "] Error shutting-down CUBLAS: ", device_id);
		printCublasErrorString( cublas_status );
		status = EXIT_FAILURE;
	}

	cudaError_t cuda_status = cudaDeviceReset();
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf(stderr, "\n[GPU%" PRI_IDX "] Error shutting-down CUDA: %s\n", device_id, cudaGetErrorString(cuda_status) );
		status = EXIT_FAILURE;
	}

	// ---------------------------------

	#if NMFGPU_VERBOSE_2
		printf("\n[GPU%" PRI_IDX "] Shutting down CUDA/CUBLAS... done\n", device_id);
	#endif

	return status;

} // finalize_GPU

////////////////////////////////////////////////////////////////////////////////

/*
 * Shuts down the device.
 *
 * Destroys associated GPU data, such as cudaStreams and/or cudaEvents.
 * Frees all allocated device memory, and shuts-down current CUDA/CUBLAS context.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int finalize_GPUdevice( void )
{

	#if NMFGPU_VERBOSE
		printf("\n[GPU%" PRI_IDX "] Finalizing GPU Device...\n",device_id);
	#endif

	int status = EXIT_SUCCESS;	// Return status

	// ------------------------------------------

	// First, it checks for previous errors.
	status = check_cuda_status();
	if ( status == EXIT_FAILURE )
		fprintf(stderr, "[GPU%" PRI_IDX "] Shutting down device...\n", device_id );

	// ------------------------------------------

	// Finalizes GPU data (CUDA Streams and CUDA events).

	if ( finalize_GPU_data() == EXIT_FAILURE )
		status = EXIT_FAILURE;

	// ------------------------------------------

	// Device memory clean up

	if ( free_dev_mem() == EXIT_FAILURE )
		status = EXIT_FAILURE;

	// ------------------------------------------

	if ( finalize_GPU() == EXIT_FAILURE )
		status = EXIT_FAILURE;

	// -----------------------------------------

	#if NMFGPU_VERBOSE
		printf("\n[GPU%" PRI_IDX "] Finalizing GPU Device... done.\n",device_id);
	#endif

	return status;

} // finalize_GPUdevice

////////////////////////////////////////////////////////////////////////////////

/*
 * Allocates PINNED HOST memory.
 *
 * Allocates HOST memory that is page-locked and accessible to the device,
 * increasing the speed of HOST-DEVICE transfers.
 *
 * size: Size of data IN BYTES.
 * wc: Set to 'true' to allocate the memory as 'write-combined' (WC).
 *	Useful ONLY for data to be transferred from the HOST to the DEVICE.
 *
 * WARNING:
 *	- This function must be called AFTER init_GPU().
 * 	- Allocating excessive amounts of pinned memory may degrade system performance,
 * 	  since it reduces the amount of memory available to the system for paging.
 * 	- Memory allocated by this function must be freed with freeHostMemory().
 *
 * Returns a pointer to the allocated memory, or NULL on error.
 */
void *getHostMemory( size_t size, bool wc )
{

	unsigned int const flags = ( wc ? (cudaHostAllocWriteCombined | cudaHostAllocPortable) : cudaHostAllocPortable );

	void *__restrict__ pHost = NULL;
	cudaError_t const cuda_status = cudaHostAlloc( (void**) &pHost, size, flags );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf( stderr, "\n[GPU%" PRI_IDX "] getHostMemory: cudaHostAlloc(size=%zu bytes): %s\n", device_id, size,
			cudaGetErrorString(cuda_status) );
		return NULL;
	}

	return pHost;

} // getHostMemory

////////////////////////////////////////////////////////////////////////////////

/*
 * Frees HOST memory previously allocated by getHostMemory().
 *
 * WARNING: This function must be called *BEFORE* finalize_GPU() or finalize_GPUdevice().
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int freeHostMemory( void *__restrict__ pHost )
{

	cudaError_t const cuda_status = cudaFreeHost( (void *) pHost );
	if ( cuda_status != cudaSuccess ) {
		fflush(stdout);
		fprintf(stderr, "\n[GPU%" PRI_IDX "] freeHostMemory(): cudaFreeHost(): %s\n", device_id, cudaGetErrorString(cuda_status) );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // freeHostMemory

////////////////////////////////////////////////////////////////////////////////

/*
 * Creates a Random-number Generator using the CURAND Library.
 *
 * WARNING: Seed is NOT initialized. For that, please use set_randomGenerator_seed().
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int init_randomGenerator( void )
{

	#if ! NMFGPU_CPU_RANDOM

		#if NMFGPU_VERBOSE
			printf("\n[GPU%" PRI_IDX "] Starting Random-number Generator...\n",device_id);
		#endif

		curandRngType_t rng_type = CURAND_RNG_PSEUDO_DEFAULT;			// Random number generator type.

		curandOrdering_t curand_ordering = CURAND_ORDERING_PSEUDO_BEST;		// Ordering type.

		curandStatus_t curand_status = CURAND_STATUS_SUCCESS;

		// ----------------------

		// Creates the generator.
		curand_status = curandCreateGenerator( &curand_generator, rng_type );
		if ( curand_status != CURAND_STATUS_SUCCESS ) {
			fflush(stdout);
			fprintf(stderr,"\n[GPU%" PRI_IDX "] Error creating the random numbers generator: ", device_id );
			printCurandErrorString( curand_status );
			return EXIT_FAILURE;
		}

		// -----------------------

		// Sets the ordering
		curand_status = curandSetGeneratorOrdering( curand_generator, curand_ordering );
		if ( curand_status != CURAND_STATUS_SUCCESS ) {
			fflush(stdout);
			fprintf(stderr,"\n[GPU%" PRI_IDX "] Error setting-up the random numbers generator (ordering): ", device_id );
			printCurandErrorString( curand_status );
			curandDestroyGenerator( curand_generator );
			return EXIT_FAILURE;
		}

		#if NMFGPU_VERBOSE
			printf("\n[GPU%" PRI_IDX "] Starting Random-number Generator... done.\n",device_id);
		#endif

	#endif /* NMFGPU_CPU_RANDOM */

	return EXIT_SUCCESS;

} // init_randomGenerator

////////////////////////////////////////////////////////////////////////////////

/*
 * Sets the seed value for an existing pseudo-random number generator.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int set_randomGenerator_seed( unsigned long long seed )
{

	#if ! NMFGPU_CPU_RANDOM

		#if NMFGPU_VERBOSE_2
			printf("\n[GPU%" PRI_IDX "] Setting seed '%llu' for the Random number Generator...\n",device_id, seed);
			fflush(stdout);
		#endif

		curandStatus_t curand_status = CURAND_STATUS_SUCCESS;

		// Sets the seed.
		curand_status = curandSetPseudoRandomGeneratorSeed( curand_generator, seed );
		if ( curand_status != CURAND_STATUS_SUCCESS ) {
			fflush(stdout);
			fprintf(stderr,"\n[GPU%" PRI_IDX "] Error setting-up the seed '%llu' for the random number generator: ",
				device_id, seed );
			printCurandErrorString( curand_status );
			return EXIT_FAILURE;
		}

		// Sets up the starting state.
		curand_status = curandGenerateSeeds( curand_generator );
		if ( curand_status != CURAND_STATUS_SUCCESS ) {
			fflush(stdout);
			fprintf(stderr,"\n[GPU%" PRI_IDX "] Error setting-up starting state of the random generator: ", device_id );
			printCurandErrorString( curand_status );
			return EXIT_FAILURE;
		}

		// -------------------------

		/* Resets the stack size of each GPU thread on SM_20 and above.
		* Please, see:
		*	https://devtalk.nvidia.com/default/topic/481553/curand-eats-device-memory/
		*/
		if ( computeCapability >= 2 ) {	// Compute Capability >= 2.0

			cudaError_t cuda_status = cudaDeviceSetLimit( cudaLimitStackSize, defaultStackSize );
			if ( cuda_status != cudaSuccess ) {
				fflush(stdout);
				fprintf(stderr, "\n[GPU%" PRI_IDX "] Warning: Could not reset the stack size of GPU threads to %zu bytes: %s\n",
					device_id, defaultStackSize, cudaGetErrorString(cuda_status) );
			}

		} // if computeCapability >= 2

		// -------------------------

		#if NMFGPU_VERBOSE_2
			printf("\n[GPU%" PRI_IDX "] Setting seed '%llu' for the Random number Generator... done.\n",device_id, seed);
			fflush(stdout);
		#endif

	#endif /* NMFGPU_CPU_RANDOM */

	return EXIT_SUCCESS;

} // set_randomGenerator_seed

////////////////////////////////////////////////////////////////////////////////

/*
 * Creates a Random-number Generator using the CURAND Library.
 * Sets the seed to the given parameter.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int init_GPU_random( unsigned long long seed )
{

	#if ! NMFGPU_CPU_RANDOM

		int status = init_randomGenerator();
		if ( status == EXIT_FAILURE )
			return EXIT_FAILURE;

		status = set_randomGenerator_seed( seed );
		if ( status == EXIT_FAILURE ) {
			finalize_randomGenerator();
			return EXIT_FAILURE;
		}

	#endif /* NMFGPU_CPU_RANDOM */

	return EXIT_SUCCESS;

} // init_GPU_random

////////////////////////////////////////////////////////////////////////////////

/*
 * Destroys an existing generator and free all memory associated with its state.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int finalize_randomGenerator( void )
{

	#if ! NMFGPU_CPU_RANDOM

		#if NMFGPU_VERBOSE_2
			printf("\n[GPU%" PRI_IDX "] Destroying Random Number Generator...\n", device_id );
			fflush(stdout);
		#endif

		curandStatus_t curand_status = curandDestroyGenerator( curand_generator );
		if ( curand_status != CURAND_STATUS_SUCCESS ) {
			fflush(stdout);
			fprintf(stderr,"\n[GPU%" PRI_IDX "] Error destroying the random number generator: ", device_id );
			printCurandErrorString( curand_status );
			return EXIT_FAILURE;
		}

		#if NMFGPU_VERBOSE_2
			printf("\n[GPU%" PRI_IDX "] Destroying Random Number Generator... done.\n", device_id );
			fflush(stdout);
		#endif

	#endif /* NMFGPU_CPU_RANDOM */

	return EXIT_SUCCESS;

} // finalize_randomGenerator

////////////////////////////////////////////////////////////////////////////////

/*
 * Blocks until GPU has completed all operations associated to 'stream'.
 */
void sync_GPU( cudaStream_t stream )
{

	#if NMFGPU_VERBOSE_2
		printf("\n[GPU%" PRI_IDX "] sync_GPU(): Waiting for results...\n", device_id );
	#endif
	// ----------------------------------

	// Waits for all operations on 'stream'.

	#if NMFGPU_DEBUG
		cudaError_t cuda_status =
	#endif
		cudaStreamSynchronize( stream );

			///////////////////////////////
			#if NMFGPU_DEBUG
				if ( cuda_status != cudaSuccess ) {
					fflush(stdout);
					fprintf(stderr, "\n[GPU%" PRI_IDX "] cudaStreamSynchronize: %s\nError in sync_GPU().\n",
						device_id, cudaGetErrorString(cuda_status) );
				}
			#endif
			///////////////////////////////

	// ---------------------------------

	#if NMFGPU_VERBOSE_2
		printf("\n[GPU%" PRI_IDX "] sync_GPU(): Waiting for results... Done.\n", device_id );
	#endif

} // sync_GPU

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
