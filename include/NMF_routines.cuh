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
 * nmf_routines.cuh
 *	Routines that implement the NMF algorithm.
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
 *		NMFGPU_VERBOSE: Shows additional information such as the begin or end of a routine call.
 *		NMFGPU_VERBOSE_2: Even more information.
 *
 *	Debug:
 *		NMFGPU_CPU_RANDOM: Uses the CPU (host) random generator (not the CURAND library).
 *		NMFGPU_DEBUG: Shows the result of each matrix operation and data transfer.
 *		NMFGPU_DEBUG_REDUCT: Shows partial results of the reduction operation.
 *		NMFGPU_DEBUG_TRANSF: Shows the result of each data transfer.
 *		NMFGPU_SYNC_TRANSF: Performs synchronous data transfers.
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

#if ! NMFGPU_NMF_ROUTINES_CUH
#define NMFGPU_NMF_ROUTINES_CUH (1)

#include "real_type.h"
#include "index_type.h"

#include <cuda_runtime_api.h>

#include <stdbool.h>

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

/* HOST-ONLY GLOBAL Variables */

// Block Configuration

extern index_t pBLN;		// Current index in block_N.xxx[].
extern index_t pBLM;		// Current index in block_M.xxx[].

extern int stepN;		// Loop directions: +1 (forward) || -1 (backward).
extern int stepM;		// Loop directions: +1 (forward) || -1 (backward).

extern index_t psNMF_N;		// Current index in NMF_streams[].
extern index_t psNMF_M;		// Current index in NMF_streams[].

extern index_t colIdx;		// Current column index in Vcol. It corresponds to <bM + colIdx> in H and d_H.
extern index_t rowIdx;		// Current row index in Vrow. It corresponds to <bN + rowIdx> in W and d_W.

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Initializes the GPU or the CPU random generator,
 * with the given seed value
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int init_random( index_t seed );

////////////////////////////////////////////////

/*
 * Finalizes the selected random generator.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int destroy_random( void );

////////////////////////////////////////////////

/*
 * Sets random values in d_A[] using the selected random generator and seed.
 *
 * If the CPU (host) random generator was selected, it first sets
 * the random values on A[] and uploads its content to d_A[].
 *
 * If 'event_A' is non-NULL, the operation is recorded as an event.
 *
 * WARNING: Requires the random generator properly initialized, with a seed set.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int set_random_values( real *RESTRICT d_A, index_t height, index_t width, index_t padding,
			#if NMFGPU_CPU_RANDOM
				real *RESTRICT A,
			#endif
			#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 || (NMFGPU_CPU_RANDOM && NMFGPU_DEBUG_TRANSF)
				bool transpose,
			#endif
			#if NMFGPU_CPU_RANDOM && (NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 || (! NMFGPU_PROFILING_GLOBAL))
				char const *RESTRICT const matrix_name_A,
			#endif
			#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 || (NMFGPU_CPU_RANDOM && NMFGPU_DEBUG_TRANSF) \
				|| ((! NMFGPU_CPU_RANDOM) && (! NMFGPU_PROFILING_GLOBAL))
				char const *RESTRICT const matrix_name_dA,
			#endif
			#if ( NMFGPU_CPU_RANDOM && NMFGPU_PROFILING_TRANSF )
				timing_data_t *RESTRICT const upload_timing,
			#endif
			cudaStream_t stream_A, cudaEvent_t *RESTRICT event_A );

////////////////////////////////////////////////

/*
 * WH(N,BLMp) = W * pH(BLM,Kp)
 * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
 * Haux(BLM,Kp) = W' * WH(N,BLMp)
 * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W
 *
 * Processes <block_M.num_steps[pBLM]> blocks of size <block_M.BL[pBLM]>
 * Once all these blocks are processed, it updates 'pBLM' to process any
 * remaining block(s).
 * It also updates 'stepM' according to the processing direction (forward or backward).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int update_H( void );

////////////////////////////////////////////////

/*
 * WH(BLN,Mp) = W(BLN,Kp) * H
 * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
 * Waux(BLN,Kp) = WH(BLN,Mp) * H'
 * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
 *
 * Processes <block_N.num_steps[pBLN]> blocks of size <block_N.BL[pBLN]>.
 * Once all these blocks are processed, it updates 'pBLN' to process any
 * remaining block(s).
 * It also updates 'stepN' according to the processing direction (forward or backward).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int update_W( void );

////////////////////////////////////////////////

/*
 * Computes classification vector from matrix d_H (full size), and stores it in "ld_classification[]".
 * Then it is downloaded from the GPU and stored in "lh_classification[]".
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int get_classification( index_t *RESTRICT ld_classification, index_t *RESTRICT lh_classification );

////////////////////////////////////////////////

/*
 * Computes the following dot products:
 *
 *	dot_V	 <-- dot_product( V, V )
 *
 *	dot_VWH  <-- dot_product( V-WH, V-WH )
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int dot_product_VWH( real *RESTRICT dot_V, real *RESTRICT dot_VWH );

////////////////////////////////////////////////
////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

#undef RESTRICT

////////////////////////////////////////////////
////////////////////////////////////////////////

#endif	/* NMFGPU_NMF_ROUTINES_CUH */
