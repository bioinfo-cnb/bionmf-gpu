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
 * nmf_routines.cuh
 *	Routines that implement the NMF algorithm.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Timing (WARNING: They PREVENT asynchronous operations):
 *		NMFGPU_PROFILING_CONV: Compute timing of convergence test.
 *		NMFGPU_PROFILING_TRANSF: Compute timing of data transfers (should be used with NMFGPU_SYNC_TRANSF).
 *		NMFGPU_PROFILING_KERNELS: Compute timing of CUDA kernels.
 *
 *	Additional information:
 *		NMFGPU_VERBOSE: Shows additional information such as the begin or end of a routine call.
 *		NMFGPU_VERBOSE_2: Even more information.
 *
 *	Debug:
 *		NMFGPU_DEBUG: Shows the result of each matrix operation and data transfer.
 *		NMFGPU_DEBUG_TRANSF: Shows the result of each data transfer.
 *		NMFGPU_SYNC_TRANSF: Performs synchronous data transfers.
 *		NMFGPU_DEBUG_REDUCT: Shows partial results of the reduction operation.
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
 *	- Matrix H is stored in memory as COLUMN-major (ie. it is transposed).
 *
 *	- All matrices include unuseful data for padding. Padded dimensions
 *	  are denoted with the 'p' character, eg. 'Mp' (ie., M + padding)
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
 *	Vrow[ 1..NnP ][ 1..M ] <-- V[ bN..(bN+NnP) ][ 1..M ]	(ie. NnP rows, starting from bN)
 *	Vcol[ 1..N ][ 1..MnP ] <-- V[ 1..N ][ bM..(bM+MnP) ]	(ie. MnP columns, starting from bM)
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
 *	d_Vrow[1..BLN][1..Mp] <-- Vrow[ offset..(offset + BLN) ][1..Mp]			(ie. BLN <= NnP rows)
 *	d_Vcol[1..N][1..BLMp] <-- Vcol[1..N][ offset_Vcol..(offset_Vcol + BLMp) ]	(ie. BLM <= MnP columns)
 *
 * Note that padded dimensions are denoted with the suffix 'p' (eg. Mp, BLMp, etc).
 *
 * In any case, matrices W and H are fully loaded into the GPU memory.
 *
 * Information for blockwise processing is stored in two block_t structures (one for each dimension).
 * Such structures (block_N and block_M) are initalized in init_block_conf() routine.
 *
 *********************************************************/

#if ! NMFGPU_NMF_ROUTINES_CUH
#define NMFGPU_NMF_ROUTINES_CUH (1)

///////////////////////////////////////////////////////

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

/* HOST-ONLY GLOBAL Variables */

// Block Configuration

extern index_t pBLN;		// Current index in block_N.xxx[].
extern index_t pBLM;		// Current index in block_M.xxx[].

extern int stepN;		// Loop directions: +1 (forward) || -1 (backward).
extern int stepM;		// Loop directions: +1 (forward) || -1 (backward).

extern index_t psNMF_N;		// Current index in NMF_streams[].
extern index_t psNMF_M;		// Current index in NMF_streams[].

extern index_t offset_Vcol;	// Current column index on first row in Vcol. Used for Vcol -> d_Vcol data transfers.

// Data matrices (host side)
extern real *pVcol;		// Pointers to V
extern real *pVrow;		// Pointers to V

// ---------------------------------------------

/* DEVICE-ONLY GLOBAL Variables */

// Data matrices (device side)
extern real *pd_W;	// Pointer to current row in d_W
extern real *pd_H;	// Pointer to current column in d_H (actually, the current row, since it is transposed).

////////////////////////////////////////////////
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
 */
void update_H( void );

// -----------------------------------

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
 */
void update_W( void );

// -----------------------------------

/*
 * d_A = MAX( d_A , R_MIN )
 *
 * Adjusts 'd_A' to avoid underflows.
 *
 *	- d_A is 'd_H' or 'd_W'.
 *	- BL: Number of rows (ie., BLN or BLM).
 *	- Padding is fixed to 'Kp'.
 *
 * NOTE: This is a portion of the Test of Convergence. Therefore,
 *	the elapsed time is added to the convergence-test time.
 */
void adjust_matrix( real *RESTRICT d_A, index_t height,
			#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2
				bool transpose, char const *RESTRICT const matrix_name_A,
			#endif
			cudaStream_t stream_A, cudaEvent_t event_A );

// -----------------------------------

/*
 * Computes classification vector from matrix d_H (full size).
 * Result is downloaded from the GPU and stored in 'classification[]'.
 */
void get_classification( int *RESTRICT const classification );

// -----------------------------------

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

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////

#undef RESTRICT

///////////////////////////////////////////////////////

#endif	/* NMFGPU_NMF_ROUTINES_CUH */
