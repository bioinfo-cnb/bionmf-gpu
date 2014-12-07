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
 * GPU_setup.cuh
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

#if ! NMFGPU_GPU_SETUP_CUH
#define NMFGPU_GPU_SETUP_CUH (1)

#include "real_type.h"
#include "index_type.h"

#include <cublas_v2.h>
#include <curand.h>	/* Random values */
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
// ---------------------------------------------

/* HOST-ONLY GLOBAL variables */

extern cublasHandle_t cublas_handle;		// cuBLAS library context.

extern curandGenerator_t curand_generator;	// cuRAND Random values Generator.

extern index_t computeCapability;		// Compute Capability (major).

extern index_t computeCapability_minor;		// Compute Capability (minor).

extern index_t maxThreadsPerBlock;		// Maximum number of threads per block.

extern index_t multiProcessorCount;		// Number of multiprocessors.

extern index_t maxThreadsPerMultiProcessor;	// Maximum number of resident threads per multiprocessor (>= maxThreadsPerBlock)

extern index_t maxGridSizeX, maxGridSizeY;	// Maximum number of thread blocks on dimensions X and Y.

// Typical number of threads per block. It should be a divisor of maxThreadsPerMultiProcessor.
extern index_t threadsPerBlock;			// <= maxThreadsPerBlock

// Threads per block for kernels requiring a value multiple of <Kp> (denoted as 'pitch').
extern index_t threadsPerBlock_pitch;		// threadsPerBlock <= threadsPerBlock_pitch <= maxThreadsPerBlock

// Maximum block height using <threadsPerBlock_pitch> threads.
extern index_t maxBlockHeight_pitch;		// <= (threadsPerBlock_pitch / pitch)

extern bool mappedHostMemory;			// Host memory is mapped into the address space of the device.

extern size_t gpu_max_num_items;		// Maximum number of items of input arrays in GPU kernels (<= matrix_max_num_items)

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

// Host matrices (used only with mapped host memory):
extern real *RESTRICT h_accum;			// Accumulator. K-length vector (<Kp> with padding).

// ---------------------------------------------
// ---------------------------------------------

/* DEVICE-ONLY GLOBAL Variables */

// Data matrices (device side):
extern real *RESTRICT d_Vrow;			// Block of BLN rows from input matrix V.
extern real *RESTRICT d_Vcol;			// Block of BLM columns from input matrix V.
extern real *RESTRICT d_H;			// Output matrix. Note that it is transposed.
extern real *RESTRICT d_W;			// Output matrix.
extern real *RESTRICT d_WH;			// Temporary matrix: d_WH = d_W * d_H
extern real *RESTRICT d_Aux;			// Temporary matrix. Sometimes denoted as d_Waux or d_Haux.
extern real *RESTRICT d_accum;			// Accumulator. K-length vector (<Kp> with padding).
extern index_t *RESTRICT d_classification;	// Classification vector.
extern index_t *RESTRICT d_last_classification;	// Previous Classification vector (used only with mapped host memory).
extern real const *RESTRICT d_scalars;		// Scalars for cuBLAS Library calls.

extern real const *RESTRICT d_zero;		// Pointer to d_scalar[0]
extern real const *RESTRICT d_one;		// Pointer to d_scalar[1]

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Returns an error message according to the provided cuBLAS status.
 */
char const *getCublasErrorString( cublasStatus_t cublas_status );

////////////////////////////////////////////////

/*
 * Returns an error message according to the provided cuRAND status.
 */
char const *getCurandErrorString( curandStatus_t curand_status );

////////////////////////////////////////////////

/*
 * Checks the provided cuBLAS status.
 * If it is NOT OK, it shows an error message.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int check_cublas_status_st( cublasStatus_t cublas_status );

////////////////////////////////////////////////

/*
 * Waits to finish all GPU operations and checks CUDA status.
 * If it is NOT OK, it shows an error message.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int check_cuda_status_st( cudaError_t cuda_status );

////////////////////////////////////////////////

/*
 * Waits to finish all GPU operations and checks CUDA status.
 * If it is NOT OK, it shows an error message.
 * It is similar to perform: check_cuda_status_st( cudaSuccess )
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int check_cuda_status( void );

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

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
size_t initialize_GPU( index_t dev_id, index_t factorization_rank );

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
int setup_GPU( size_t mem_size, bool do_classf );

////////////////////////////////////////////////

/*
 * Shuts down current cuBLAS/CUDA context.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int shutdown_GPU( void );

////////////////////////////////////////////////

/*
 * Finalizes the GPU device.
 *
 * Destroys associated GPU data, such as cudaStreams and/or cudaEvents,
 * frees all allocated device memory, and shuts-down current CUDA/cuBLAS context.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int finalize_GPU_device( void );

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
 *	Useful ONLY for data to be transferred from the HOST to the DEVICE.
 *
 * clear: Set to 'true' to initialize the memory area with zeros.
 *
 * WARNING:
 *	- The GPU device must has been properly initialized through init_GPU().
 *	- Allocating excessive amounts of pinned memory may degrade system performance,
 *	  since it reduces the amount of memory available to the system for paging.
 *	- Memory allocated by this function must be freed with freeHostMemory().
 *
 * Returns a pointer to the allocated memory, or NULL on error.
 */
void *getHostMemory( size_t size, bool wc, bool clear );

////////////////////////////////////////////////

/*
 * Frees HOST memory previously allocated by getHostMemory().
 *
 * WARNING:
 *	This function must be called *BEFORE* finalize_GPU() or finalize_GPUdevice().
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int freeHostMemory( void *RESTRICT pHost, char const *RESTRICT pHost_name );

////////////////////////////////////////////////

/*
 * Creates a Random-number Generator using the cuRAND Library.
 *
 * WARNING:
 *	The seed is NOT initialized. For that, please use set_randomGenerator_seed().
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int init_randomGenerator( void );

////////////////////////////////////////////////

/*
 * Sets the seed value for an existing pseudo-random number generator.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int set_randomGenerator_seed( unsigned long long seed );

////////////////////////////////////////////////

/*
 * Creates a Random-number Generator using the cuRAND Library.
 * Sets the seed to the given parameter.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int init_GPU_random( unsigned long long seed );

////////////////////////////////////////////////

/*
 * Destroys an existing generator and free all memory associated with its state.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int finalize_randomGenerator( void );

////////////////////////////////////////////////

/*
 * Blocks until GPU has completed all operations associated to 'stream'.
 *
 * return EXIT_SUCCESS or EXIT_FAILURE.
 */
int sync_GPU( cudaStream_t stream );

////////////////////////////////////////////////
////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

#undef RESTRICT

////////////////////////////////////////////////
////////////////////////////////////////////////

#endif /* NMFGPU_GPU_SETUP_CUH */
