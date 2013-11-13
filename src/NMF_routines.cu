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
 * NMF_routines.cu
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

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>	/* strerror() */
#include <math.h>	/* sqrtd() */

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "nmf_routines.cuh"
#include "matrix/matrix_operations.h"
#include "GPU_setup.h"
#if defined(NMFGPU_PROFILING_TRANSF) || defined(NMFGPU_PROFILING_CONV)
	#include "timing.h"
#endif

// ---------------------------------------------
// ---------------------------------------------

/* HOST-ONLY GLOBAL Variables */

// Block Configuration
index_t pBLN = 0, pBLM = 0;		// Current index in block_N.*[] and block_M.*[], respectively.
int stepN = 1, stepM = 1;		// Loop directions: +1 (forward) || -1 (backward).
index_t psNMF_N = 0, psNMF_M = 0;	// Current index in NMF_streams[].

index_t offset_Vcol = 0;		// Current column index on first row in Vcol. Used for Vcol -> d_Vcol data transfers.

// Data matrices (host side)
real *pVcol = NULL, *pVrow = NULL;	// Pointers to V

// ---------------------------------------------

/* DEVICE-ONLY GLOBAL Variables */

// Data matrices (device side)
real *pd_W = NULL, *pd_H = NULL;		// Pointers to d_W and d_H

// Data matrix for dot_product_VWH()
real *d_dot_VWH = NULL, d_dot_V = NULL;
real *pd_dot_VWH = NULL, *pd_dot_V = NULL;	// Pointer to d_dot_VWH


/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

/*
 * WH(N,BLMp) = W * H(BLM,Kp)
 * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
 * Haux(BLM,Kp) = W' * WH(N,BLMp)
 * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W(Kp)
 *
 * WARNING: CUBLAS stream must have been set to NMF_streams[psNMF_M].
 */
static void get_H_BLM( index_t const BLM, index_t const BLMp )
{

	#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t\t\t\t-----get_H_BLM(BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ")-----\n", BLM, BLMp );
	#endif

	// ---------------------------------

	real *const d_Haux = d_Aux;		// Temporary matrix: W' * WH.
	real *const d_accum_w = d_accum;	// Accumulator vector: SUM(W).

	#ifdef NMFGPU_DEBUG
		cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	#endif

	// ----------------------------------

	// WH(N,BLMp) = W * H(BLM,Kp)

	#ifdef NMFGPU_DEBUG
	cublas_status =
	#endif
		// NMF_streams[ psNMF_M ]
		cublasRgemm( cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, BLM, N, K, d_one, pd_H, Kp, d_W, Kp, d_zero, d_WH, BLMp );


		#ifdef NMFGPU_DEBUG
		///////////////////////////////
			printf("\n--- [GPU%" PRI_IDX "] Resulting WHcol=W*H (BLM=%" PRI_IDX ", BLMp=%" PRI_IDX "): ---\n",
				device_id, BLM, BLMp );
			check_cublas_status_st( cublas_status );
			check_cuda_status();
			show_device_matrix( d_WH, N, BLM, BLMp, false, NULL );
		/////////////////////////////
		#endif

	// ---------------------------

        // WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)

        matrix_div_sub( d_WH, d_Vcol, N, BLMp,
			#ifdef NMFGPU_DEBUG
				BLM, "WHcol", "Vcol",
			#endif
			true,		// division
			NMF_streams[ psNMF_M ], Vcol_event, div_timing );

	// ----------------------------

        // Haux(BLM,Kp) = W' * WH(N,BLMp)

	#ifdef NMFGPU_DEBUG
	cublas_status =
	#endif
		// NMF_streams[ psNMF_M ]
		cublasRgemm( cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, K, BLM, N, d_one, d_W, Kp, d_WH, BLMp, d_zero, d_Haux, Kp );

		#ifdef NMFGPU_DEBUG
		///////////////////////////////
			printf("\n--- [GPU%" PRI_IDX "] Resulting d_Haux (BLM=%" PRI_IDX "): ---\n", device_id, BLM );
			check_cublas_status_st( cublas_status );
			check_cuda_status();
			show_device_matrix( d_Haux, BLM, K, Kp, true, NULL );
		/////////////////////////////
		#endif

	// ----------------------------

        // H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W(Kp)

        matrix_mul_div( pd_H, d_Haux, d_accum_w, BLM, Kp,
			#ifdef NMFGPU_DEBUG
				K, true, "H", "Haux", "accum_W",
			#endif
			NMF_streams[ psNMF_M ] );

	// ----------------------------

	#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t\t\t\t-----End of get_H_BLM(BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ")-----\n", BLM, BLMp);
	#endif

} // get_H_BLM

// =======================================================================

/*
 * WH(N,BLMp) = W * pH(BLM,Kp)
 * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
 * Haux(BLM,Kp) = W' * WH(N,BLMp)
 * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W(Kp)
 *
 * Processes <num_steps> blocks of size <BLM>.
 * Executes get_H_BLM() <num_steps> times, where a new portion of matrix 'Vcol' is transferred to the GPU.
 * Pointer to 'd_H' is also updated.
 *
 * WARNING: CUBLAS stream must have been set to NMF_streams[psNMF_M].
 */
static void getH_loop( index_t const num_steps, index_t const BLM, index_t const BLMp )
{

	#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t\t----- getH_loop(BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ", num_steps=%" PRI_IDX ", stepM=%i"
			", offset_Vcol=%" PRI_IDX ") -----\n", BLM, BLMp, num_steps, stepM, offset_Vcol );
	#endif

	// -------------------------------

	/*
	 * Processes <num_steps> blocks of size BLM.
	 */

	// First iteration:

	/* WH(N,BLMp) = W * pH(BLM,Kp)
	 * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
	 * Haux(BLM,Kp) = W' * WH(N,BLMp)
	 * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W
	 */
	get_H_BLM( BLM, BLMp );

	// --------------------

	// Remaining (num_steps-1) iterations:

	for ( index_t st = 1 ; st < num_steps ; st++ ) {

		#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t\t\t-----getH_loop:(step %" PRI_IDX "/%" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ", stepM=%i"
				", offset_Vcol=%" PRI_IDX ")-----\n", st, num_steps, BLM, BLMp, stepM, offset_Vcol );
		#endif

		// ----------------

		// Transfers (asynchronously) a new <N x BLM> block from Vcol to d_Vcol.

		// Sets pointers to the block to be transferred.
		pVcol += (int) (stepM * (int) BLM);
		offset_Vcol += (int) (stepM * (int) BLM);

		#if defined(NMFGPU_DEBUG) || defined(NMFGPU_DEBUG_TRANSF)
			///////////////////////////////
			printf("\n--- [GPU%" PRI_IDX "]: Vcol processed (N=%" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ") (offset=%"
				PRI_IDX ", pitch=%" PRI_IDX "): ---\n", device_id, N, BLM, BLMp, offset_Vcol, MnPp);
			//////////////////////////////
		#endif
		upload_matrix_partial( pVcol, N, MnPp, offset_Vcol, BLM, "Vcol", "d_Vcol", BLMp, d_Vcol, Vcol_event, Vcol_stream,
					&upload_Vcol_timing );

		// ----------------

		// Updates pointer to d_H(BLM,:)
		pd_H += (int)(stepM * (int)(BLM * Kp));

		// Changes main stream.
		psNMF_M += (int) stepM;	// forward or backward

		// Sets new CUBLAS stream.
		cublasSetStream( NMF_streams[ psNMF_M ] );

		// -----------------------

		/* WH(N,BLMp) = W * pH(BLM,Kp)
		 * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
		 * Haux(BLM,Kp) = W' * WH(N,BLMp)
		 * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W
		 */
		get_H_BLM( BLM, BLMp );

		#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t\t\t-----getH_loop: End of loop (step %" PRI_IDX " of %" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%" PRI_IDX
				", stepM=i%, offset_Vcol=%" PRI_IDX ")-----\n", st, num_steps, BLM, BLMp, stepM, offset_Vcol );
		#endif

	} // for ( st=1 ; st<num_steps ; st++ )

	// -------------------------------

	#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t\t-----End of getH_loop(BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ", num_steps=%" PRI_IDX ", stepM=%i"
				", offset_Vcol=%" PRI_IDX ")-----\n", BLM, BLMp, num_steps, stepM, offset_Vcol );
	#endif

} // getH_loop

// =======================================================================

/*
 * WH(N,BLMp) = W * pH(BLM,Kp)
 * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
 * Haux(BLM,Kp) = W' * WH(N,BLMp)
 * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W
 *
 * Processes <block_M.num_steps[pBLM]> blocks of size <block_M.BL[pBLM]>
 * This is performed by getH_loop().
 * Once all these blocks are processed, it updates 'pBLM' to process any
 * remaining block(s) (getH_loop() is called again).
 * It also updates 'stepM' according to the processing direction (forward or backward).
 */
void update_H()
{

	#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n-----update_H(pBLM=%" PRI_IDX ", stepM=%i, offset_Vcol=%" PRI_IDX ")-----\n",
				pBLM, stepM, offset_Vcol );
	#endif

	// ----------------------------------

	/*
	 * Processes <block_M.num_steps[ pBLM ]> blocks of size <block_M.BL[ pBLM ]>
	 */

	// Block configuration.
	index_t num_steps = block_M.num_steps[ pBLM ];	// Number of loops.
	index_t BLM = block_M.BL[ pBLM ];		// Number of columns.
	index_t BLMp = block_M.BLp[ pBLM ];		// Number of columns (with padding).

	// --------------------------------

	// Changes CUBLAS stream.
	cublasSetStream( NMF_streams[ psNMF_M ] );

	// --------------------------------

	/* WH(N,BLMp) = W * pH(BLM,Kp)
	 * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
	 * Haux(BLM,Kp) = W' * WH(N,BLMp)
	 * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W
	 */
	getH_loop( num_steps, BLM, BLMp );

	// --------------------------------

	// Remaining blocks

	if ( block_M.num_steps[1] ) {	// There are more blocks in dimension "M" to process.

		#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t-----update_H(pBLM=%" PRI_IDX ",stepM=%i,offset_Vcol=%" PRI_IDX ",psNMF_M=%" PRI_IDX
				"): New block-----\n", pBLM, stepM, offset_Vcol, psNMF_M );
		#endif

		// Updates pointers to Vcol (HOST matrix) and d_H (DEVICE matrix) and changes block information.

		if ( stepM > 0 ) {	// Going forward

			// First, updates both pointers. Then, changes block information.

			// Rows ALREADY processed:
			pd_H += (BLM * Kp);
			pVcol += BLM;
			offset_Vcol += BLM;

			// Changes block size:
			pBLM = !( pBLM );

			// Updates block information
			BLM = block_M.BL[ pBLM ];		// Number of columns.

		} else {	// Going backward

			// First, changes block information. Then, updates pointers.

			// Changes block size
			pBLM = !( pBLM );

			// Updates block information
			BLM = block_M.BL[ pBLM ];		// Number of columns.

			// Rows TO BE processed (NOTE: offsets are negative).
			pd_H -= (BLM * Kp);
			pVcol -= BLM;
			offset_Vcol -= BLM;

		} // if ( stepM > 0 )

		// Updates other block informations
		num_steps = block_M.num_steps[ pBLM ];	// Number of loops.
		BLMp = block_M.BLp[ pBLM ];		// Number of columns (with padding).

		// Changes main stream.
		psNMF_M += (int) stepM;	// forward or backward

		// Changes CUBLAS stream.
		cublasSetStream( NMF_streams[ psNMF_M ] );

		// ------------------------

		// Transfers (asynchronously) a new <N x BLM> block from Vcol to d_Vcol.

		#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t-----update_H: New block (pBLM=%" PRI_IDX ", stepM=%i, BLM=%" PRI_IDX ", BLMp=%" PRI_IDX
				", num_steps=%" PRI_IDX ", offset_Vcol=%" PRI_IDX ", psNMF_M=%" PRI_IDX ")-----\n", pBLM, stepM, BLM, BLMp,
				num_steps, offset_Vcol, psNMF_M );
		#endif

		#if defined(NMFGPU_DEBUG) || defined(NMFGPU_DEBUG_TRANSF)
			///////////////////////////////
			printf("\n--- [GPU%" PRI_IDX "]: Vcol processed (N=%" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ") (offset=%"
				PRI_IDX ", pitch=%" PRI_IDX "): ---\n", device_id, N, BLM, BLMp, offset_Vcol, MnPp);
			//////////////////////////////
		#endif
		upload_matrix_partial( pVcol, N, MnPp, offset_Vcol, BLM, "Vcol", "d_Vcol", BLMp, d_Vcol, Vcol_event, Vcol_stream,
					&upload_Vcol_timing );

		// -------------------------

		// Processes that block.

		/* WH(N,BLMp) = W * pH(BLM,Kp)
		 * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
		 * Haux(BLM,Kp) = W' * WH(N,BLMp)
		 * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W
		 */
		getH_loop( num_steps, BLM, BLMp, psNMF_M );

		// -------------------------

		// Changes direction (forward -> backward || backward -> forward)
		stepM *= (-1);

		#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf("\n\t-----update_H: End of new block (pBLM=%" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ", num_steps=%" PRI_IDX
				", offset_Vcol=%" PRI_IDX "). New StepM=%i -----\n", pBLM, BLM, BLMp, num_steps, offset_Vcol, stepM );
		#endif

	} // if ( block_M.num_steps[1] > 0 )

	// ------------------------

	#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n-----End of update_H(pBLM=%" PRI_IDX ", stepM=%i, offset_Vcol=%" PRI_IDX ")-----\n",
				pBLM, stepM, offset_Vcol );
	#endif

} // update_H

////////////////////////////////////////////////////////////////

/*
 * WH(BLN,Mp) = W(BLN,Kp) * H
 * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
 * Waux(BLN,Kp) = WH(BLN,Mp) * H'
 * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
 *
 * WARNING: CUBLAS stream must have been set to NMF_streams[psNMF_N].
 */
static void get_W_BLN( index_t const BLN )
{

	#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id ) printf( "\n\t\t\t\t-----get_W_BLN(BLN=%" PRI_IDX ")-----\n", BLN );
	#endif

	// ---------------------------------

	real * const d_Waux = d_Aux;		// Temporary matrix: WH * H'
	real * const d_accum_h = d_accum;	// Accumulator vector: SUM(H).

	#ifdef NMFGPU_DEBUG
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	#endif

	// ----------------------------------

	// WH(BLN,Mp) = W(BLN,Kp) * H

	#ifdef NMFGPU_DEBUG
	cublas_status =
	#endif
		// NMF_streams[ psNMF_N ]
		cublasRgemm( cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, BLN, K, d_one, d_H, Kp, pd_W, Kp, d_zero, d_WH, Mp );

			#ifdef NMFGPU_DEBUG
			///////////////////////////////
				printf("\n--- [GPU%" PRI_IDX "] Resulting WHrow=W*H (BLN=%" PRI_IDX ", Mp=%" PRI_IDX "): ---\n",
					device_id, BLN, Mp );
				check_cublas_status_st( cublas_status );
				check_cuda_status();
				show_device_matrix( d_WH, BLN, M, Mp, false, NULL );
			/////////////////////////////
			#endif

	// ---------------------------

	// WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)

	matrix_div_sub( d_WH, d_Vrow, BLN, Mp,
		#ifdef NMFGPU_DEBUG
			M, "WHrow", "Vrow",
		#endif
		true,	// division
		NMF_streams[ psNMF_N ], Vrow_event, div_timing );

	// ---------------------------

	// Waux(BLN,Kp) = WH(BLN,Mp) * H'

	#ifdef NMFGPU_DEBUG
	cublas_status =
	#endif

	cublasRgemm( cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, K, BLN, M, d_one, d_H, Kp, d_WH, Mp, d_zero, d_Waux, Kp );

		#ifdef NMFGPU_DEBUG
		///////////////////////////////
			printf("\n--- [GPU%" PRI_IDX "] Resulting d_Waux (BLN=%" PRI_IDX "): ---\n", device_id, BLN );
			check_cublas_status_st( cublas_status );
			check_cuda_status();
			show_device_matrix( d_Waux, BLN, K, Kp, false, NULL );
		/////////////////////////////
		#endif


	// ----------------------------


	// W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_H(Kp)

	matrix_mul_div( pd_W, d_Waux, d_accum_h, BLN, Kp,

		#ifdef NMFGPU_DEBUG
			K, false, "W", "Waux", "accum_H",
		#endif
			NMF_streams[ psNMF_N ] );


	// ----------------------------


	#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id ) printf( "\n\t\t\t\t-----End of get_W_BLN(BLN=%" PRI_IDX ")-----\n", BLN );
	#endif

} // getWrow

// =======================================================================

/*
 * WH(BLN,Mp) = W(BLN,Kp) * H
 * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
 * Waux(BLN,Kp) = WH(BLN,Mp) * H'
 * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
 *
 * Processes <num_steps> blocks of size <BLN>.
 * Executes get_W_BLN() <num_steps> times, where a new portion of matrix 'Vrow' is transferred to the GPU.
 * Pointer to 'd_W' is also updated.
 *
 * WARNING: CUBLAS stream must have been set to NMF_streams[psNMF_N].
 */
static void getW_loop( index_t const num_steps, index_t const BLN )
{

	#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t\t-----getW_loop(BLN=%" PRI_IDX ", num_steps=%" PRI_IDX ", stepN=%i)-----\n",
				BLN, num_steps, stepN );
	#endif

	// -------------------------------

	/*
	 * Processes <num_steps> blocks of size BLN.
	 */

	// First iteration:

	/* WH(BLN,Mp) = W(BLN,Kp) * H
	 * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
	 * Waux(BLN,Kp) = WH(BLN,Mp) * H'
	 * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
	 */
	get_W_BLN( BLN );

	// --------------------

	// Remaining (num_steps-1) iterations:

	for ( index_t st = 1 ; st < num_steps ; st++ ) {

		#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t\t\t-----getW_loop:(step %" PRI_IDX "/%" PRI_IDX ", BLN=%" PRI_IDX ", stepN=%i)-----\n",
				st, num_steps, BLN, stepN );
		#endif

		// ----------------

		// Transfers (asynchronously) a new <BLN x M> block from Vrow to d_Vrow

		// Sets pointers to the block to be transferred.
		pVrow += (int)(stepN * (int)(BLN * Mp));

		#if defined(NMFGPU_DEBUG) || defined(NMFGPU_DEBUG_TRANSF)
			///////////////////////////////
			printf("\n--- [GPU%" PRI_IDX "]: Vrow processed (BLN=%" PRI_IDX "): ---\n", device_id, BLN);
			//////////////////////////////
		#endif
		upload_matrix_partial( pVrow, BLN, Mp, 0, M, "Vrow", "d_Vrow", Mp, Vrow_stream, Vrow_event, &upload_Vrow_timing );

		// ----------------

		// Updates pointer to d_W(BLN,:)
		pd_W += (int)(stepN * (int)(BLN * Kp));	// offset_W

		// Changes main stream.
		psNMF_N += (int) stepN;	// forward or backward

		// Changes CUBLAS stream.
		cublasSetStream( NMF_streams[ psNMF_N ] );

		// ----------------

		/* WH(BLN,Mp) = W(BLN,Kp) * H
		 * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
		 * Waux(BLN,Kp) = WH(BLN,Mp) * H'
		 * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
		 */
		get_W_BLN( BLN );

		// ----------------

		#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t\t\t-----getW_loop: End of loop (step %" PRI_IDX " of %" PRI_IDX ", BLN=%" PRI_IDX ", stepN=%i"
				")-----\n", st, num_steps, BLN, stepN );
		#endif

	} // for ( st=1 ; st<num_steps ; st++ )

	// --------------------------

	#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t\t-----End of getW_loop(BLN=%" PRI_IDX ", num_steps=%" PRI_IDX ", stepN=%i)-----\n",
				BLN, num_steps, stepN );
	#endif

} // getW_loop

// =======================================================================

/*
 * WH(BLN,Mp) = W(BLN,Kp) * H
 * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
 * Waux(BLN,Kp) = WH(BLN,Mp) * H'
 * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
 *
 * Processes <block_N.num_steps[pBLN]> blocks of size <block_N.BL[pBLN]>
 * This is performed by getW_loop().
 * Once all these blocks are processed, it updates 'pBLN' to process any
 * remaining block(s) (getW_loop() is called again).
 * It also updates 'stepN' according to the processing direction (forward or backward).
 */
void update_W()
{

	#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n-----update_W(pBLN=%" PRI_IDX ", stepN=%i)-----\n", pBLN, stepN);
	#endif

	// ----------------------------------

	/*
	 * Processes <block_N.num_steps[pBLN]> blocks of size <block_N.BL[pBLN]>
	 */

	// Block configuration.
	index_t num_steps = block_N.num_steps[ pBLN ];	// Number of loops.
	index_t BLN = block_N.BL[ pBLN ];		// Number of rows.

	// --------------------------------

	// Changes CUBLAS stream.
	cublasSetStream( NMF_streams[ psNMF_N ] );

	// --------------------------------

	/* WH(BLN,Mp) = W(BLN,Kp) * H
	 * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
	 * Waux(BLN,Kp) = WH(BLN,Mp) * H'
	 * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
	 */
	getW_loop( num_steps, BLN );

	// --------------------------------

	// Remaining blocks

	if ( block_N.num_steps[1] ) {  // There are more blocks in dimension "N" to process.

		#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t-----update_W(pBLN=%" PRI_IDX ",stepN=%i,psNMF_N=%" PRI_IDX "): New block-----\n",
				pBLN, stepN, psNMF_N );
		#endif

		// Updates pointers to Vrow (HOST matrix) and d_W (DEVICE matrix) and changes block information.

		if ( stepN > 0 ) {	// Going forward

			// First, updates both pointers. Then, changes block information.

			// Rows ALREADY processed:
			pd_W += (BLN * Kp);
			pVrow += (BLN * Mp);

			// Changes block size
			pBLN = !( pBLN );

			// Updates block information
			BLN = block_N.BL[ pBLN ];		// Number of rows.

		} else {	// Going backward

			// First, changes block information. Then, updates pointers.

			// Changes block size
			pBLN = !( pBLN );

			// Updates block information
			BLN = block_N.BL[ pBLN ];		// Number of rows.

			// Rows TO BE processed (NOTE: offsets are negative).
			pd_W -= (BLN * Kp);
			pVrow -= (BLN * Mp);

		} // if ( stepN > 0 )

		// Updates other block informations
		num_steps = block_N.num_steps[ pBLN ];	// Number of loops.

		// Changes main stream.
		psNMF_N += (int) stepN;			// forward or backward

		// Changes CUBLAS stream.
		cublasSetStream( NMF_streams[ psNMF_N ] );

		// ----------------

		// Transfers (asynchronously) a new <BLN x M> block from Vrow to d_Vrow.

		#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t-----update_W: New block (pBLN=%" PRI_IDX ", stepN=%i, BLN=%" PRI_IDX ", num_steps=%" PRI_IDX
				", psNMF_N=%" PRI_IDX ")-----\n", pBLN, stepN, BLN, num_steps, psNMF_N );
		#endif

		#if defined(NMFGPU_DEBUG) || defined(NMFGPU_DEBUG_TRANSF)
			///////////////////////////////
			printf("\n--- [GPU%" PRI_IDX "]: Vrow processed (BLN=%" PRI_IDX ", M=%" PRI_IDX ", Mp=%" PRI_IDX "): ---\n",
				device_id, BLN, M, Mp);
			//////////////////////////////
		#endif
		upload_matrix_partial( pVrow, BLN, Mp, 0, M, "Vrow", "d_Vrow", Mp, Vrow_stream, Vrow_event, &upload_Vrow_timing );

		// ---------------------------

		// Processes that block.

		/* WH(BLN,Mp) = W(BLN,Kp) * H
		 * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
		 * Waux(BLN,Kp) = WH(BLN,Mp) * H'
		 * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
		 */
		getW_loop( num_steps, BLN );

		// -------------------------

		// Changes direction (forward -> backward || backward -> forward)
		stepN *= (-1);

		#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf("\n\t-----update_W: End of new block (pBLN=%" PRI_IDX ", BLN=%" PRI_IDX ", num_steps=%" PRI_IDX
				"). New StepN=%i -----\n", pBLN, BLN, num_steps, stepN );
		#endif

	} // if ( block_N.num_steps[1] > 0 )

	// -----------------------

	#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n-----End of update_W(pBLN=%" PRI_IDX ", stepN=%i)-----\n", pBLN, stepN);
	#endif

} // update_W

/////////////////////////////////////////////////////////////////////

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
void adjust_matrix( real *__restrict__ d_A, index_t height,
			#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2
				bool transpose, char const *__restrict__ const matrix_name_A,
			#endif
			cudaStream_t stream_A, cudaEvent_t *__restrict__ event_A )
{

	#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\nadjust_matrix(height=%" PRI_IDX ")...\n", height );
	#endif

	// ----------------------------

	// Delays the transfer until the event has completed all previous operations.
	if ( event_A ) {

		#if NMFGPU_DEBUG
			cudaErrot_t cuda_status =
		#endif
			cudaStreamWaitEvent( stream_A, event_A, 0 );

			#ifdef NMFGPU_DEBUG
			///////////////////////////////
				fprintf(stderr, "\n[GPU%" PRI_IDX "] Error setting CUDA event to wait for (cudaStreamWaitEvent): %s\nError "
					"in adjust_matrix(%s, height=%" PRI_IDX ").\n", device_id,
					cudaGetErrorString(cuda_status), matrix_name_A, height );
			/////////////////////////////
			#endif
	}

	// ----------------------------

	#ifdef NMFGPU_PROFILING_CONV
		start_cuda_timer_ev( timing_events[START_OUTER_EVENT], device_id );
	#endif

		matrix_adjust( d_A, height, Kp,
			#if NMFGPU_DEBUG
				K, transpose, matrix_name_A,
			#endif
			stream_A );

	#ifdef NMFGPU_PROFILING_CONV
		stop_cuda_timer_cnt_ev( timing_events[START_OUTER_EVENT], adjust_timing, height*Kp, 1, device_id );
	#endif

	// ----------------------------

	#ifdef NMFGPU_DEBUG
	///////////////////////////////
		printf( "\n[GPU%" PRI_IDX "] Matrix adjusted (rows=%" PRI_IDX ").\n", device_id, height );
		check_cuda_status();
	/////////////////////////////
	#endif

	// ----------------------------

	#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\nadjust_matrix(rows=%" PRI_IDX ")...Done\n", height );
	#endif

} // adjust_matrix

/////////////////////////////////////////////////////////////////////

/*
 * Computes classification vector from matrix d_H (full size).
 * Result is downloaded from the GPU and stored in 'classification[]'.
 */
void get_classification( index_t *__restrict__ classification )
{

	#if defined(NMFGPU_DEBUG) || defined(NMFGPU_DEBUG_TRANSF)
		printf("\n[GPU%" PRI_IDX "] get_classification()...\n", device_id );
	#elif defined(NMFGPU_VERBOSE_2)
		if ( ! device_id )
			printf("\nget_classification()...\n");
	#endif

	// ---------------------------------

	// Stream for this operation.
	cudaStream_t stream_A = matrix_stream;

	// ---------------------------------

	#ifdef NMFGPU_PROFILING_CONV
		start_cuda_timer_ev( timing_events[START_OUTER_EVENT], device_id );
	#endif

		// Computes the classification vector: Column index of highest values.

		matrix_idx_max( d_H, K, Kp, M,
				#if NMFGPU_DEBUG
					"d_H", "d_classification"
				#endif
				stream_A, d_classification );

		///////////////////////////////
		#ifdef NMFGPU_DEBUG
			printf( "\n[GPU%" PRI_IDX "] Classification vector computed. Downloading...\n", device_id );
			check_cuda_status();
		#endif
		///////////////////////////////

		// ------------------------------

		// Downloads output vector.

		download_matrix_int( classification, 1, Mp, d_classification,
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
						M, false, "classification", "d_classification",
					#endif
					&download_classf_timing );

	#ifdef NMFGPU_PROFILING_CONV
		stop_cuda_timer_cnt_ev( timing_events[START_OUTER_EVENT], &download_classf_timing, Mp, 1, device_id );
	#endif

	// -----------------------------

	// Waits until complete.

	sync_GPU( stream_A );

	// -----------------------------

	#if defined(NMFGPU_DEBUG) || defined(NMFGPU_DEBUG_TRANSF)
		printf("\n[GPU%" PRI_IDX "] get_classification()... Done.\n",device_id);
	#elif defined(NMFGPU_VERBOSE_2)
		if ( ! device_id )
			printf("\nget_classification()... Done.\n");
	#endif

} // get_classification

/////////////////////////////////////////////////////////////////////

/*
 * WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
 * dot_VWH(BLN) = SUM((V-WH)**2)
 * dot_V(BLN)	= SUM(V**2)
 *
 * WARNING: CUBLAS stream must have been set to NMF_streams[psNMF_N].
 */
static void get_dot_VWH_BLN( index_t const BLN )
{

	#ifdef NMFGPU_DEBUG
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	#endif

	// Uses all the available streams, starting from NMF_streams[ psNMF_N ].
	index_t stream_idx = psNMF_N;

	// ----------------------------------

	// WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)

	#ifdef NMFGPU_DEBUG
	cublas_status =
	#endif
		// NMF_streams[ psNMF_N ]
		cublasRgemm( cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, BLN, K, d_one, d_H, Kp, pd_W, Kp, d_zero, d_WH, Mp );

			#ifdef NMFGPU_DEBUG
			///////////////////////////////
				printf("\n--- [GPU%" PRI_IDX "] Resulting WHrow=W*H (BLN=%" PRI_IDX ", Mp=%" PRI_IDX "): ---\n",
					device_id, BLN, Mp );
				check_cublas_status_st( cublas_status );
				check_cuda_status();
				show_device_matrix( d_WH, BLN, M, Mp, false, NULL );
			/////////////////////////////
			#endif


	// ---------------------------


	// WH(BLN,Mp) = Vcol(BLN,Mp) - WH(BLN,Mp)
	matrix_div_sub( d_WH, d_Vrow, BLN, Mp,
		#ifdef NMFGPU_DEBUG
			M, "WHrow", "Vrow",
		#endif
		false,	// subtraction
		NMF_streams[ psNMF_N ], Vrow_event, sub_timing );


	// ---------------------------

	/*
	 * dot_VWH(BLN) = SUM((V-WH)**2)
	 */

	// TODO: Change this loop for specific kernel(s).

	pd_WH = d_WH;
	{
		#ifdef NMFGPU_DEBUG
		cublasStatus_t cs =
		#endif
			cublasRdot( cublas_handle, M, pd_WH, 1, pd_WH, 1, pd_dot_VWH );

			#ifdef NMFGPU_DEBUG
				if ( cs != CUBLAS_STATUS_SUCCESS );
					cublas_status = cs;
			#endif

		// --------------------

		// Changes CUBLAS stream
		stream_idx = psNMF_N + 1;
		if ( stream_idx == num_NMF_streams )
			stream_idx = 0;

		pd_WH += Mp;
		pd_dot_VWH++;
	}

	for ( index_t i = 1 ; i < BLN ; i++, pd_WH += Mp, pd_dot_VWH++ ) {

		// Sets new CUBLAS stream.
		cublasSetStream( NMF_streams[ stream_idx ] );

		// --------------------

		#ifdef NMFGPU_DEBUG
		cublasStatus_t cs =
		#endif
			cublasRdot( cublas_handle, M, pd_WH, 1, pd_WH, 1, pd_dot_VWH );

			#ifdef NMFGPU_DEBUG
				if ( cs != CUBLAS_STATUS_SUCCESS );
					cublas_status = cs;
			#endif

		// --------------------

		// Changes of CUBLAS stream
		stream_idx++;
		if ( stream_idx == num_NMF_streams )
			stream_idx = 0;

	} // for

	#ifdef NMFGPU_DEBUG
	///////////////////////////////
		printf("\n--- [GPU%" PRI_IDX "] Resulting d_dot_VWH[] in get_dot_VWH_BLN(BLN=%" PRI_IDX ", Mp=%" PRI_IDX "): ---\n",
			device_id, BLN, Mp );
		check_cublas_status_st( cublas_status );
		check_cuda_status();
		show_device_matrix( pd_dot_VWH, 1, BLN, BLN, false, NULL );
	/////////////////////////////
	#endif

	// --------------------------------------

	/*
	 * dot_V(BLN) = SUM(V**2)
	 */

	// TODO: Change this loop for specific kernel(s).

	real *pd_Vrow = d_Vrow;
	for ( index_t i = 0 ; i < BLN ; i++, dp_Vrow += Mp, pd_dot_V++ ) {

		// Sets new CUBLAS stream.
		cublasSetStream( NMF_streams[ stream_idx ] );

		// --------------------

		#ifdef NMFGPU_DEBUG
		cublasStatus_t cs =
		#endif
			cublasRdot( cublas_handle, M, dp_Vrow, 1, dp_Vrow, 1, pd_dot_V );

			#ifdef NMFGPU_DEBUG
				if ( cs != CUBLAS_STATUS_SUCCESS );
					cublas_status = cs;
			#endif

		// --------------------

		// Changes of CUBLAS stream
		stream_idx++;
		if ( stream_idx == num_NMF_streams )
			stream_idx = 0;

	} // for

	#ifdef NMFGPU_DEBUG
	///////////////////////////////
		printf("\n--- [GPU%" PRI_IDX "] Resulting d_dot_V[] in get_dot_VWH_BLN(BLN=%" PRI_IDX ", Mp=%" PRI_IDX "): ---\n",
			device_id, BLN, Mp );
		check_cublas_status_st( cublas_status );
		check_cuda_status();
		show_device_matrix( pd_dot_V, 1, BLN, BLN, false, NULL );
	/////////////////////////////
	#endif

} // get_dot_VWH_BLN

// =======================================================================

/*
 * WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
 * dot_VWH(BLN) = SUM((V-WH)**2)
 * dot_V(BLN)	= SUM(V**2)
 *
 * WARNING: CUBLAS stream must have been set to NMF_streams[psNMF_N].
 */
static void get_dot_VWH_loop( index_t num_steps, index_t BLN )
{

	#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t\t-----get_dot_VWH_loop(BLN=%" PRI_IDX ", num_steps=%" PRI_IDX ", stepN=%i)-----\n",
				BLN, num_steps, stepN );
	#endif

	// -------------------------------

	/*
	 * Processes <num_steps> blocks of size BLN.
	 */

	// First iteration:

	/* WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
	 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
	 * dot_VWH(BLN) = SUM((V-WH)**2)
	 * dot_V(BLN)	= SUM(V**2)
	 */
	get_dot_VWH_BLN BLN );

	// -----------------------

	// Remaining (num_steps-1) iterations:

	for ( index_t st = 1 ; st < num_steps ; st++ ) {

		#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t\t\t-----get_dot_VWH_loop:(step %" PRI_IDX "/%" PRI_IDX ", BLN=%" PRI_IDX ", stepN=%i)-----\n",
				st, num_steps, BLN, stepN );
		#endif

		// ----------------

		// Transfers (asynchronously) a new <BLN x M> block from Vrow to d_Vrow

		// Sets pointers to the block to be transferred.
		pVrow += (int)(stepN * (int)(BLN * Mp));

		#if defined(NMFGPU_DEBUG) || defined(NMFGPU_DEBUG_TRANSF)
			///////////////////////////////
			printf("\n--- [GPU%" PRI_IDX "]: Vrow processed (BLN=%" PRI_IDX "): ---\n", device_id, BLN);
			//////////////////////////////
		#endif
		upload_matrix_partial( pVrow, BLN, Mp, 0, M, "Vrow", "d_Vrow", Mp, Vrow_stream, Vrow_event, &upload_Vrow_timing );

		// ----------------

		// Updates pointer to d_W(BLN,:) and d_dot_VWH(BLN,:)
		pd_W += (int)(stepN * (int)(BLN * Kp));
		pd_dot_VWH += (int)(stepN * (int)(BLN * Kp));

		// Changes main stream.
		psNMF_N += (int) stepN;	// forward or backward

		// Changes CUBLAS stream.
		cublasSetStream( NMF_streams[ psNMF_N ] );

		// ----------------

		/* WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
		 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
		 * dot_VWH(BLN) = SUM((V-WH)**2)
		 * dot_V(BLN)	= SUM(V**2)
		 */
		get_dot_VWH_BLN BLN );

		#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t\t\t-----get_dot_VWH_loop: End of loop (step %" PRI_IDX " of %" PRI_IDX ", BLN=%" PRI_IDX
				", stepN=%i)-----\n", st, num_steps, BLN, stepN );
		#endif

	} // for ( st=1 ; st<num_steps ; st++ )

	// --------------------------

	#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t\t-----End of get_dot_VWH_loop(BLN=%" PRI_IDX ", num_steps=%" PRI_IDX ", stepN=%i)-----\n",
				BLN, num_steps, stepN );
	#endif

} // get_dot_VWH_loop

// =======================================================================

/*
 * WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
 * dot_VWH(BLN) = SUM((V-WH)**2)
 * dot_V(BLN)	= SUM(V**2)
 *
 * Processes <block_N.num_steps[pBLN]> blocks of size <block_N.BL[pBLN]>
 * This is performed by get_dot_VWH_loop().
 * Once all these blocks are processed, it updates 'pBLN' to process any
 * remaining block(s) (get_dot_VWH_loop() is called again).
 * It also updates 'stepN' according to the processing direction (forward or backward).
 */
static void get_dot_VWH()
{

	#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n-----get_dot_VWH(pBLN=%" PRI_IDX ", stepN=%i)-----\n", pBLN, stepN);
	#endif

	// ----------------------------------

	/*
	 * Processes <block_N.num_steps[pBLN]> blocks of size <block_N.BL[pBLN]>
	 */

	// Block configuration.
	index_t num_steps = block_N.num_steps[ pBLN ];	// Number of loops.
	index_t BLN = block_N.BL[ pBLN ];		// Number of rows.

	// --------------------------------

	// Changes CUBLAS stream.
	cublasSetStream( NMF_streams[ psNMF_N ] );

	// --------------------------------

	/* WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
	 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
	 * dot_VWH(BLN) = SUM((V-WH)**2)
	 * dot_V(BLN)	= SUM(V**2)
	 */
	get_dot_VWH_loop( num_steps, BLN );

	// --------------------------------

	// Remaining blocks

	if ( block_N.num_steps[1] ) {  // There are more blocks in dimension "N" to process.

		#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t-----get_dot_VWH(pBLN=%" PRI_IDX ",stepN=%i,psNMF_N=%" PRI_IDX "): New block-----\n",
				pBLN, stepN, psNMF_N );
		#endif

		// Updates pointers to Vrow (HOST matrix) and d_W (DEVICE matrix) and changes block information.

		if ( stepN > 0 ) {	// Going forward

			// First, updates both pointers. Then, changes block information.

			// Rows ALREADY processed:
			pd_W += (BLN * Kp);
			pd_dot_VWH += (BLN * Kp);
			pVrow += (BLN * Mp);

			// Changes block size
			pBLN = !( pBLN );

			// Updates block information
			BLN = block_N.BL[ pBLN ];		// Number of rows.

		} else {	// Going backward

			// First, changes block information. Then, updates pointers.

			// Changes block size
			pBLN = !( pBLN );

			// Updates block information
			BLN = block_N.BL[ pBLN ];		// Number of rows.

			// Rows TO BE processed (NOTE: offsets are negative).
			pd_W -= (BLN * Kp);
			pd_dot_VWH -= (BLN * Kp);
			pVrow -= (BLN * Mp);

		} // if ( stepN > 0 )

		// Updates other block informations
		num_steps = block_N.num_steps[ pBLN ];	// Number of loops.

		// Changes main stream.
		psNMF_N += (int) stepN;			// forward or backward

		// Changes CUBLAS stream.
		cublasSetStream( NMF_streams[ psNMF_N ] );

		// ----------------

		// Transfers (asynchronously) a new <BLN x M> block from Vrow to d_Vrow.

		#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n\t-----get_dot_VWH: New block (pBLN=%" PRI_IDX ", stepN=%i, BLN=%" PRI_IDX
				", num_steps=%" PRI_IDX ", psNMF_N=%" PRI_IDX ")-----\n", pBLN, stepN, BLN, num_steps, psNMF_N );
		#endif

		#if defined(NMFGPU_DEBUG) || defined(NMFGPU_DEBUG_TRANSF)
			///////////////////////////////
			printf("\n--- [GPU%" PRI_IDX "]: Vrow processed (BLN=%" PRI_IDX ", M=%" PRI_IDX ", Mp=%" PRI_IDX "): ---\n",
				device_id, BLN, M, Mp);
			//////////////////////////////
		#endif
		upload_matrix_partial( pVrow, BLN, Mp, 0, M, "Vrow", "d_Vrow", Mp, Vrow_stream, Vrow_event, &upload_Vrow_timing );

		// ---------------------------

		// Processes that block.

		/* WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
		 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
		 * dot_VWH(BLN) = SUM((V-WH)**2)
		 * dot_V(BLN)	= SUM(V**2)
		 */
		get_dot_VWH_loop( num_steps, BLN );

		// -------------------------

		// Changes direction (forward -> backward || backward -> forward)
		stepN *= (-1);

		#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf("\n\t-----get_dot_VWH: End of new block (pBLN=%" PRI_IDX ", BLN=%" PRI_IDX ", num_steps=%" PRI_IDX
				"). New StepN=%i -----\n", pBLN, BLN, num_steps, stepN );
		#endif

	} // if ( block_N.num_steps[1] > 0 )

	// -----------------------

	#ifdef NMFGPU_VERBOSE_2
		if ( ! device_id )
			printf( "\n-----End of get_dot_VWH(pBLN=%" PRI_IDX ", stepN=%i)-----\n", pBLN, stepN);
	#endif

} // get_dot_VWH

// =======================================================================

/*
 * Computes the following dot products:
 *
 *	dot_V	 <-- dot_product( V, V )
 *
 *	dot_VWH  <-- dot_product( V-WH, V-WH )
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int dot_product_VWH( real *__restrict__ dot_V, real *__restrict__ dot_VWH )
{

	// Sets pointers (GLOBAL variables) to be used as data matrix.

	d_dot_VWH = d_Aux;
	pd_dot_VWH = d_Aux;	// Pointer to d_dot_VWH

	d_dot_V = d_Aux + N;
	pd_dot_V = d_dot_V;	// Pointer to d_dot_V

	// ----------------------------------------

	/* WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
	 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
	 * dot_VWH(BLN) = SUM((V-WH)**2)
	 * dot_V(BLN)	= SUM(V**2)
	 */
	get_dot_VWH();

	// ----------------------------------------

	// Waits until finished and checks for errors.

	#if defined(NMFGPU_DEBUG)
	///////////////////////////////
		printf("\n--- [GPU%" PRI_IDX "] dot_product_VWH: Waiting for results...\n", device_id );
	///////////////////////////////
	#endif

	if ( check_cuda_status() == EXIT_FAILURE )
		return EXIT_FAILURE;

	// ----------------------------------------

	// Downloads partial results.

	// Size = 2*N: d_dot_VWH and d_dot_V
	real *__restrict__ const h_dot_VWH = getHostMemory( 2 * N * sizeof(real), false );
	if ( ! h_dot_VWH ) {
		fprintf( stderr, "[GPU%" PRI_IDX "] Error in dot_product_VWH( N=%" PRI_IDX " )\n", device_id, N );
		return EXIT_FAILURE;
	}

	download_matrix( h_dot_VWH, 1, N, d_dot_VWH,
				#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
					N, false, "dot_VWH", "d_dot_VWH",
				#endif
				NULL );

	if ( check_cuda_status() == EXIT_FAILURE ) {
		freeHostMemory( h_dot_VWH );
		return EXIT_FAILURE;
	}

	// ----------------------------------------

	// Computes the final result.

	real l_dot_VWH	= REAL_C( 0.0 );
	real l_dot_V	= REAL_C( 0.0 );

	for ( index_t i = 0 ; i < N ; i++ )
		l_dot_VWH += h_dot_VWH[i];

	for ( index_t i = N ; i < (2*N) ; i++ )
		l_dot_V += h_dot_VWH[i];


	#if defined(NMFGPU_DEBUG) || defined(NMFGPU_VERBOSE)
		///////////////////////////////
		printf("\n[GPU%" PRI_IDX "] dot_product_VWH: dot_V=%g dot_VWH=%g\n", device_id, );
		///////////////////////////////
	#endif

	// ----------------------------------------

	if ( freeHostMemory( h_dot_VWH ) == EXIT_FAILURE )
		return EXIT_FAILURE


	*dot_V = l_dot_V;
	*dot_VWH = l_dot_VWH;

	return EXIT_SUCCESS;

} // dot_product_VWH

////////////////////////////////////////////////////////////////////////
