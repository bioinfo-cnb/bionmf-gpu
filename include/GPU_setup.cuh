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
 * GPU_setup.cuh
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
 *		NMFGPU_FIXED_INIT: Uses "random" values generated from a fixed seed (defined in common.h).
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
 * so that V ~ W*H
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
 *
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
 *
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

#if ! NMFGPU_GPU_SETUP_CUH
#define NMFGPU_GPU_SETUP_CUH (1)

///////////////////////////////////////////////////////

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <curand.h>	/* Random values */

#include "index_type.h"
#include "real_type.h"

///////////////////////////////////////////////////////

/* Selects the appropriate "restrict" keyword. */

#undef RESTRICT

#if __CUDACC__				/* CUDA source code */
	#define RESTRICT __restrict__
#else					/* C99 source code */
	#define RESTRICT restrict
#endif

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////

/* C linkage, not C++. */
#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------
// ---------------------------------------------

/* Data types */

/* Block configuration.
 * NOTE: This refer to a block from input matrix V to be loaded into the GPU memory, NOT a block of CUDA threads.
 * Input matrix V is processed in blocks of <BLN> rows and <BLM> columns.
 * There is a block_t structure for each dimension "BLD" (where "D" is 'N' or 'M').
 */
typedef struct {
	index_t BL[2];		// Number of rows/columns for this dimension in GPU memory ('BLN' or 'BLM')
	index_t BLp[2];		// Padding for <BL[]> (if any).
	index_t num_steps[2];	// Number of blocks of size <BL[]> to be processed.
} block_t;

// ---------------------------------------------

/* HOST-ONLY GLOBAL variables */

extern index_t device_id;			// Device ID number.

extern index_t num_devices;			// Number of devices.

extern cublasHandle_t cublas_handle;		// CUBLAS library context.

extern curandGenerator_t curand_generator;	// CURAND Random values Generator.

extern index_t computeCapability;		// Compute Capability (major).

extern index_t computeCapability_minor;		// Compute Capability (minor).

/* Alignment value for data to be stored in GPU memory: Typically 'warpSize/2' on compute capability 1.x, and 'warpSize' on 2.x and beyond.
 * It is similar to the DEVICE-only constant 'MEMORY_ALIGNMENT' defined on "GPU_kernels.h".
 */
extern index_t memory_alignment;

extern index_t maxThreadsPerBlock;		// Maximum number of threads per block.

extern index_t multiProcessorCount;		// Number of multiprocessors.

extern index_t maxThreadsPerMultiProcessor;	// Maximum number of resident threads per multiprocessor (>= maxThreadsPerBlock)

extern index_t maxGridSizeX, maxGridSizeY;	// Maximum number of thread blocks on dimensions X and Y.

extern size_t defaultStackSize;			// Default stack size of each GPU thread ( Compute Capability >= 2.0 )

// Typical number of threads per block. It should be a divisor of maxThreadsPerMultiProcessor.
extern index_t threadsPerBlock;			// <= maxThreadsPerBlock

// Threads per block for kernels requiring a value multiple of <Kp> (denoted as 'pitch').
extern index_t threadsPerBlock_pitch;		// threadsPerBlock <= threadsPerBlock_pitch <= maxThreadsPerBlock

// Maximum block height using <threadsPerBlock_pitch> threads.
extern index_t maxBlockHeight_pitch;		// <= (threadsPerBlock_pitch / pitch)

extern block_t block_N, block_M;		// Information for blockwise processing on dimension N and M.

// CUDA Events for synchronization:
extern cudaEvent_t event_Vrow;			// d_Vrow
extern cudaEvent_t event_Vcol;			// d_Vcol
extern cudaEvent_t event_W;			// d_W
extern cudaEvent_t event_H;			// d_H
extern cudaEvent_t event_reduction;		// Event to register matrix reduction operations.


// CUDA Streams for synchronization:
extern cudaStream_t stream_Vrow;		// d_Vrow
extern cudaStream_t stream_Vcol;		// d_Vcol
extern cudaStream_t stream_W;			// d_W
extern cudaStream_t stream_H;			// d_H
extern cudaStream_t *RESTRICT streams_NMF;	// Main-flow streams for blockwise processing.
extern index_t num_streams_NMF;			// Number of main-flow streams: MAX( SUM(block_N.num_steps[i]), SUM(block_M.num_steps[i]) )

// Matrix dimensions (host side):
extern index_t N;	// Number of rows of input matrix V.
extern index_t M;	// Number of columns of input matrix V.
extern index_t K;	// Factorization rank.

// Dimensions for multi-GPU version:
extern index_t NnP;	// Number of rows of V assigned to this GPU (NnP <= N).
extern index_t MnP;	// Number of columns of V assigned to this GPU (MnP <= M).

// Padded dimensions:
extern index_t Mp;	// 'M' rounded up to the next multiple of <memory_alignment>.
extern index_t Kp;	// 'K' rounded up to the next multiple of <memory_alignment>.
extern index_t MnPp;	// 'MnP' rounded up to the next multiple of <memory_alignment> (MnPp <= Mp).

// ---------------------------------------------

/* DEVICE-ONLY GLOBAL Variables */

// Data matrices (device side):
extern real *RESTRICT d_Vrow;		// Block of BLN rows from input matrix V.
extern real *RESTRICT d_Vcol;		// Block of BLM columns from input matrix V.
extern real *RESTRICT d_H;			// Output matrix. Note that it is transposed.
extern real *RESTRICT d_W;			// Output matrix.
extern real *RESTRICT d_WH;			// Temporary matrix: d_WH = d_W * d_H
extern real *RESTRICT d_Aux;		// Temporary matrix. Sometimes denoted as d_Waux or d_Haux.
extern real *RESTRICT d_accum;		// Accumulator. K-length vector (<Kp> with padding).
extern index_t  *RESTRICT d_classification;	// Classification vector.

extern real const *RESTRICT d_scalars;	// Scalars for CUBLAS Library calls.
extern real const *RESTRICT d_zero;		// Pointer to d_scalar[0]
extern real const *RESTRICT d_one;		// Pointer to d_scalar[1]

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Prints an error message according to the provided CUBLAS status.
 */
void printCublasErrorString( cublasStatus_t cublas_status );

// ------------------------------------------

/*
 * Prints an error message according to the provided CURAND status.
 */
void printCurandErrorString( curandStatus_t curand_status );

// ------------------------------------------

/*
 * Checks the provided CUBLAS status.
 * If it is NOT OK, it shows an error message.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int check_cublas_status_st( cublasStatus_t cublas_status );

// ------------------------------------------

/*
 * Waits to finish all GPU operations and checks CUDA status.
 * If it is NOT OK, it shows an error message.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int check_cuda_status_st( cudaError_t cuda_status );

// ------------------------------------------

/*
 * Waits to finish all GPU operations and checks CUDA status.
 * If it is NOT OK, it shows an error message.
 * It is similar to perform: check_cuda_status_st( cudaSuccess )
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int check_cuda_status( void );

// ------------------------------------------

/*
 * Initializes CUDA and CUBLAS on the specified device.
 * Stores in 'mem_size' the Global Memory size.
 *
 * WARNING:
 * 	This function must be called *BEFORE* any other CUDA-related routine (e.g., cudaMallocHost(), cudaHostAlloc()).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int init_GPU( index_t dev_id, index_t num_devs, size_t *RESTRICT const mem_size );

// ------------------------------------------

/*
 * Computes the padded dimension.
 *
 * Returns the next multiple of 'memory_alignment'.
 */
index_t get_padding( index_t dim );

// ------------------------------------------

/*
 * Computes the highest power of 2 <= x.
 * Returns the same value (x) if it is already a power of 2, or is zero.
 */
index_t prev_power_2( index_t x );

// ------------------------------------------

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
int init_GPUdevice( size_t mem_size, bool do_classf );

// ------------------------------------------

/*
 * Shuts-down current CUBLAS/CUDA context.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int finalize_GPU( void );

// ------------------------------------------

/*
 * Shuts down the device.
 *
 * Destroys associated GPU data, such as cudaStreams and/or cudaEvents,
 * frees all allocated device memory, and shuts-down current CUDA/CUBLAS context.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int finalize_GPUdevice( void );

// ------------------------------------------

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
void *getHostMemory( size_t size, bool wc );

// ------------------------------------------

/*
 * Frees HOST memory previously allocated by getHostMemory().
 *
 * WARNING: This function must be called *BEFORE* finalize_GPU() or finalize_GPUdevice().
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int freeHostMemory( void *RESTRICT pHost );

// ------------------------------------------

/*
 * Blocks until GPU has completed all operations associated to 'stream'.
 */
void sync_GPU( cudaStream_t stream );

// ------------------------------------------

/*
 * Creates a Random-number Generator using the CURAND Library.
 *
 * WARNING: Seed is NOT initialized. For that, please use set_randomGenerator_seed().
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int init_randomGenerator( void );

// ------------------------------------------

/*
 * Sets the seed value for an existing pseudo-random number generator.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int set_randomGenerator_seed( unsigned long long seed );

// ------------------------------------------

/*
 * Creates a Random-number Generator using the CURAND Library.
 * Sets the seed to the given parameter.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int init_GPU_random( unsigned long long seed );

// ------------------------------------------

/*
 * Destroys an existing generator and free all memory associated with its state.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int finalize_randomGenerator( void );

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////

#undef RESTRICT

///////////////////////////////////////////////////////

#endif /* NMFGPU_GPU_SETUP_CUH */
