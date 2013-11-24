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
 *		NMFGPU_PROFILING_TRANSF: Compute timing of data transfers (should be used with NMFGPU_SYNC_TRANSF).
 *		NMFGPU_PROFILING_KERNELS: Compute timing of CUDA kernels.
 *
 *	Additional information:
 *		NMFGPU_VERBOSE: Shows additional information such as the begin or end of a routine call.
 *		NMFGPU_VERBOSE_2: Even more information.
 *
 *	Debug:
 *		NMFGPU_CPU_RANDOM: Uses the CPU (host) random generator (not the CURAND library).
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
 *	d_Vcol[1..N][1..BLMp] <-- Vcol[1..N][ colIdx..(colIdx + BLMp) ]	(ie. BLM <= MnP columns)
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

#include <cublas_v2.h>

#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_CONV
	#include "timing.cuh"
#endif
#include "GPU_setup.cuh"
#include "matrix/matrix_operations.cuh"
#include "NMF_routines.cuh"

// ---------------------------------------------
// ---------------------------------------------

/* HOST-ONLY GLOBAL Variables */

index_t pBLN = 0;	// Current index in block_N.xxx[].
index_t pBLM = 0;	// Current index in block_M.xxx[].

int stepN = 1;		// Loop directions: +1 (forward) || -1 (backward).
int stepM = 1;		// Loop directions: +1 (forward) || -1 (backward).

index_t psNMF_N = 0;	// Current index in streams_NMF[].
index_t psNMF_M = 0;	// Current index in streams_NMF[].

index_t colIdx = 0;	// Current column index in Vcol, H and d_H (actually, row index, since H is transposed).
index_t rowIdx = 0;	// Current row index in Vrow and W.

// Data matrices (host side)
real *Vcol = NULL;		// Pointer to V (actually, it is just an alias).
real *Vrow = NULL;		// Pointer to V (actually, it is just an alias).


/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

/*
 * Initializes the GPU or the CPU random generator,
 * with the given seed value
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int init_random( index_t seed )
{

	int status = EXIT_SUCCESS;

	#if NMFGPU_CPU_RANDOM

		// Initializes random generator on CPU.
		srandom( seed );

	#else
		// Initializes random generator on GPU.
		status = init_GPU_random( seed );

	#endif

	return status;

} // init_random

/////////////////////////////////////////////////////////////////////

/*
 * Finalizes the selected random generator.
 */
void destroy_random( void )
{

	#if ! NMFGPU_CPU_RANDOM

		finalize_randomGenerator();

	#endif

} // destroy_random

/////////////////////////////////////////////////////////////////////

/*
 * Sets random values in d_A[] using the selected random generator and seed.
 *
 * If the CPU (host) random generator was selected, it first sets
 * the random values on A[] and uploads its content to d_A[].
 *
 * If 'event_A' is non-NULL, the operation is recorded as an event.
 *
 * WARNING: Requires the random generator properly initialized, with a seed set.
 */
void set_random_values( real *__restrict__ A, real *__restrict__ d_A, index_t height, index_t width, index_t padding,
				#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2
					bool transpose, char const *__restrict__ const matrix_name_A,
					char const *__restrict__ const matrix_name_dA,
				#endif
				#if NMFGPU_CPU_RANDOM && NMFGPU_PROFILING_TRANSF
					timing_data_t *__restrict__ const upload_timing,
				#endif
				cudaStream_t stream_A, cudaEvent_t *__restrict__ event_A )
{

	#if NMFGPU_CPU_RANDOM

		// CPU Random generator

		real *pA = A;
		for ( index_t i = 0 ; i < height ; i++, pA += padding ) {
			for ( index_t j = 0 ; j < width ; j++ ) {
				real val = ( ((real) random() ) / ((real) RAND_MAX) ) + R_MIN;
				pA[ j ] = val;
			}
			for ( index_t j = width ; j < padding ; j++ )
				pA[ j ] = REAL_C( 1.0 );
		}

		///////////////////////////////
		#if NMFGPU_DEBUG
			printf( "\n--- [GPU%" PRI_IDX "] Random values on matrix %s --> %s (height=%" PRI_IDX ", width=%" PRI_IDX
				", padding=%" PRI_IDX ", transpose=%i): ---\n", device_id, matrix_name_A, matrix_name_dA,
				height, width, padding, transpose );
		#endif
		/////////////////////////////

		// Uploads the new values.
		upload_matrix( A, height, padding, d_A,
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
						width, transpose, matrix_name_A, matrix_name_dA,
					#endif
					#if NMFGPU_PROFILING_TRANSF
						upload_timing,
					#endif
					stream_A, event_A );

	// ---------------------------------

	#else	/* ! NMFGPU_CPU_RANDOM */

		// Device random generator

		matrix_random( d_A, height, width, padding,
				#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2
					transpose, matrix_name_dA,
				#endif
				stream_A, event_A );

	#endif


} // set_random_values

/////////////////////////////////////////////////////////////////////

/*
 * WH(N,BLMp) = W * H(BLM,Kp)
 * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
 * Haux(BLM,Kp) = W' * WH(N,BLMp)
 * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W(Kp)
 *
 * WARNING: CUBLAS stream must have been set to streams_NMF[psNMF_M].
 */
static void get_H_BLM( index_t const BLM, index_t const BLMp )
{

	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n\t\t\t\t-----get_H_BLM(BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ", colIdx=%" PRI_IDX ")-----\n",
				BLM, BLMp, colIdx );
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
		// streams_NMF[ psNMF_M ]
		CUBLAS_R_GEMM( cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, BLM, N, K, d_one, &d_H[ colIdx*Kp ], Kp, d_W, Kp, d_zero, d_WH, BLMp );

			#ifdef NMFGPU_DEBUG
			///////////////////////////////
				printf("\n--- [GPU%" PRI_IDX "] Resulting WHcol=W*H (BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ", colIdx=%" PRI_IDX
					"): ---\n", device_id, BLM, BLMp, colIdx );
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
			true,	// division
			#if NMFGPU_PROFILING_KERNELS
				div_timing,
			#endif
			streams_NMF[ psNMF_M ], event_Vcol );

	// ----------------------------

        // Haux(BLM,Kp) = W' * WH(N,BLMp)

	#ifdef NMFGPU_DEBUG
	cublas_status =
	#endif
		// streams_NMF[ psNMF_M ]
		CUBLAS_R_GEMM( cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, K, BLM, N, d_one, d_W, Kp, d_WH, BLMp, d_zero, d_Haux, Kp );

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

        matrix_mul_div( &d_H[ colIdx*Kp ], d_Haux, d_accum_w, BLM, Kp,
			#ifdef NMFGPU_DEBUG
				K, true, "H", "Haux", "accum_W",
			#endif
			streams_NMF[ psNMF_M ] );

	// ----------------------------

	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n\t\t\t\t-----End of get_H_BLM(BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ", colIdx=%" PRI_IDX ")-----\n",
				BLM, BLMp, colIdx );
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
 * WARNING: CUBLAS stream must have been set to streams_NMF[psNMF_M].
 */
static void getH_loop( index_t const num_steps, index_t const BLM, index_t const BLMp )
{

	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n\t\t----- getH_loop(BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ", num_steps=%" PRI_IDX ", stepM=%i"
			", colIdx=%" PRI_IDX ") -----\n", BLM, BLMp, num_steps, stepM, colIdx );
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
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n\t\t\t-----getH_loop:(step %" PRI_IDX "/%" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ", stepM=%i"
				", colIdx=%" PRI_IDX ")-----\n", st, num_steps, BLM, BLMp, stepM, colIdx );
		#endif

		// ----------------

		// Transfers (asynchronously) a new <N x BLM> block from Vcol to d_Vcol.

		// First, updates the column index.
		colIdx += (stepM * BLM);

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
			///////////////////////////////
			printf("\n--- [GPU%" PRI_IDX "]: Vcol processed (N=%" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ", new colIdx=%"
				PRI_IDX ", pitch=%" PRI_IDX "): ---\n", device_id, N, BLM, BLMp, colIdx, MnPp);
			//////////////////////////////
		#endif
		upload_matrix_partial( Vcol, N, MnPp, 0, colIdx,	// Starting row: 0
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
						BLM, "Vcol", "d_Vcol",
					#endif
					BLMp, d_Vcol, stream_Vcol, event_Vcol
					#if NMFGPU_PROFILING_TRANSF
						, &upload_Vcol_timing
					#endif
				);

		// ----------------

		// Changes the main stream.
		psNMF_M += stepM;	// forward or backward

		// Sets new CUBLAS stream.
		cublasSetStream( cublas_handle, streams_NMF[ psNMF_M ] );

		// -----------------------

		/* WH(N,BLMp) = W * pH(BLM,Kp)
		 * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
		 * Haux(BLM,Kp) = W' * WH(N,BLMp)
		 * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W
		 */
		get_H_BLM( BLM, BLMp );

		#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n\t\t\t-----getH_loop: End of loop (step %" PRI_IDX " of %" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%" PRI_IDX
				", stepM=%i, colIdx=%" PRI_IDX ")-----\n", st, num_steps, BLM, BLMp, stepM, colIdx );
		#endif

	} // for ( st=1 ; st<num_steps ; st++ )

	// -------------------------------

	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n\t\t-----End of getH_loop(BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ", num_steps=%" PRI_IDX ", stepM=%i"
				", colIdx=%" PRI_IDX ")-----\n", BLM, BLMp, num_steps, stepM, colIdx );
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
void update_H( void )
{

	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n-----update_H(pBLM=%" PRI_IDX ", stepM=%i, colIdx=%" PRI_IDX ")-----\n", pBLM, stepM, colIdx );
	#endif

	// ----------------------------------

	// Reduces d_W to a row.
	matrix_to_row( d_W, N, Kp,
			#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG
				K, "d_W",
			#endif
			d_Aux, d_accum, stream_W );

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
	cublasSetStream( cublas_handle, streams_NMF[ psNMF_M ] );

	// Delays further operations until d_W is ready.
	{
		#if NMFGPU_DEBUG
			cudaError_t cuda_status = cudaSuccess;
		#endif

		for ( index_t i = 0, ps = psNMF_M ; i < num_steps ; i++, ps += stepM ) {

			#if NMFGPU_DEBUG
				cudaError_t cs =
			#endif

				cudaStreamWaitEvent( streams_NMF[ ps ], event_W, 0 );

			#if NMFGPU_DEBUG
				if ( cs != cudaSuccess )
					cuda_status = cs;
			#endif
		}

		///////////////////////////////
		#if NMFGPU_DEBUG
			if ( cuda_status != cudaSuccess ) {
				fflush(stdout);
				fprintf(stderr, "\n[GPU%" PRI_IDX "] Error: could not delay operations until d_W is ready: %s\n"
					"Error in update_H(psNMF_M=%" PRI_IDX ", pBLM=%" PRI_IDX ", stepM=%i, colIdx=%" PRI_IDX
					").\n", device_id, cudaGetErrorString(cuda_status), psNMF_M, pBLM, stepM, colIdx );
			}
		#endif
		///////////////////////////////
	}

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
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n\t-----update_H(pBLM=%" PRI_IDX ",stepM=%i,colIdx=%" PRI_IDX ",psNMF_M=%" PRI_IDX
				"): New block-----\n", pBLM, stepM, colIdx, psNMF_M );
		#endif

		// Updates pointers to Vcol (HOST matrix) and d_H (DEVICE matrix) and changes block information.

		if ( stepM > 0 ) {	// Going forward

			// First, updates both pointers. Then, changes block information.

			// Rows ALREADY processed:
			colIdx += BLM;

			// Changes block size:
			pBLM = ! pBLM;

			// Updates block information
			BLM = block_M.BL[ pBLM ];		// Number of columns.

		} else {	// Going backward

			// First, changes block information. Then, updates pointers.

			// Changes block size
			pBLM = ! pBLM;

			// Updates block information
			BLM = block_M.BL[ pBLM ];		// Number of columns.

			// Rows TO BE processed (NOTE: offset is negative).
			colIdx -= BLM;

		} // if ( stepM > 0 )

		// Updates other block informations
		num_steps = block_M.num_steps[ pBLM ];	// Number of loops.
		BLMp = block_M.BLp[ pBLM ];		// Number of columns (with padding).

		// Changes main stream.
		psNMF_M += stepM;	// forward or backward

		// Changes CUBLAS stream.
		cublasSetStream( cublas_handle, streams_NMF[ psNMF_M ] );

		// ------------------------

		// Transfers (asynchronously) a new <N x BLM> block from Vcol to d_Vcol.

		#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n\t-----update_H: New block (pBLM=%" PRI_IDX ", stepM=%i, BLM=%" PRI_IDX ", BLMp=%" PRI_IDX
				", num_steps=%" PRI_IDX ", colIdx=%" PRI_IDX ", psNMF_M=%" PRI_IDX ")-----\n", pBLM, stepM, BLM, BLMp,
				num_steps, colIdx, psNMF_M );
		#endif

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
			///////////////////////////////
			printf("\n--- [GPU%" PRI_IDX "]: Vcol processed (N=%" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ") (offset=%"
				PRI_IDX ", pitch=%" PRI_IDX "): ---\n", device_id, N, BLM, BLMp, colIdx, MnPp);
			//////////////////////////////
		#endif
		upload_matrix_partial( Vcol, N, MnPp, 0, colIdx,	// Starting row: 0
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
						BLM, "Vcol", "d_Vcol",
					#endif
						BLMp, d_Vcol, stream_Vcol, event_Vcol
					#if NMFGPU_PROFILING_TRANSF
						, &upload_Vcol_timing
					#endif
				);

		// -------------------------

		// Processes that block.

		/* WH(N,BLMp) = W * pH(BLM,Kp)
		 * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
		 * Haux(BLM,Kp) = W' * WH(N,BLMp)
		 * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W
		 */
		getH_loop( num_steps, BLM, BLMp );

		// -------------------------

		// Changes direction (forward -> backward || backward -> forward)
		stepM *= (-1);

		#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf("\n\t-----update_H: End of new block (pBLM=%" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ", num_steps=%"
				PRI_IDX ", colIdx=%" PRI_IDX "). New StepM=%i -----\n", pBLM, BLM, BLMp, num_steps, colIdx, stepM );
		#endif

	} // if ( block_M.num_steps[1] > 0 )

	// ------------------------

	// Records as an event all previous operations on matrix H.
	{
		#if NMFGPU_DEBUG
			cudaError_t cuda_status =
		#endif

			cudaEventRecord( event_H, streams_NMF[ psNMF_M ] );

			///////////////////////////////
			#if NMFGPU_DEBUG
				if ( cuda_status != cudaSuccess ) {
					fflush(stdout);
					fprintf(stderr, "\n[GPU%" PRI_IDX "] Error recording CUDA event: %s\nError in update_H(pBLM=%"
						PRI_IDX ", stepM=%i, colIdx=%" PRI_IDX ").\n", device_id, cudaGetErrorString(cuda_status),
						pBLM, stepM, colIdx );
				}
			#endif
			///////////////////////////////
	}

	// ------------------------

	// Delays further operations on "stream_H" until "event_H" completes.
	{
		#if NMFGPU_DEBUG
			cudaError_t cuda_status =
		#endif

			cudaStreamWaitEvent( stream_H, event_H, 0 );

			///////////////////////////////
			#if NMFGPU_DEBUG
				if ( cuda_status != cudaSuccess ) {
					fflush(stdout);
					fprintf(stderr, "\n[GPU%" PRI_IDX "] Error: could not delay operations until event_H completes: %s\n"
						"Error in update_H(pBLM=%" PRI_IDX ", stepM=%i, colIdx=%" PRI_IDX ").\n", device_id,
						cudaGetErrorString(cuda_status), pBLM, stepM, colIdx );
				}
			#endif
			///////////////////////////////
	}

	// ------------------------

	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n-----End of update_H(pBLM=%" PRI_IDX ", stepM=%i, colIdx=%" PRI_IDX ")-----\n",
				pBLM, stepM, colIdx );
	#endif

} // update_H

////////////////////////////////////////////////////////////////

/*
 * WH(BLN,Mp) = W(BLN,Kp) * H
 * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
 * Waux(BLN,Kp) = WH(BLN,Mp) * H'
 * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
 *
 * WARNING: CUBLAS stream must have been set to streams_NMF[psNMF_N].
 */
static void get_W_BLN( index_t const BLN )
{

	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n\t\t\t\t-----get_W_BLN(BLN=%" PRI_IDX ", rowIdx=%" PRI_IDX ")-----\n", BLN, rowIdx );
	#endif

	// ---------------------------------

	real *const d_Waux = d_Aux;		// Temporary matrix: WH * H'
	real *const d_accum_h = d_accum;	// Accumulator vector: SUM(H).

	#ifdef NMFGPU_DEBUG
		cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	#endif

	// ----------------------------------

	// WH(BLN,Mp) = W(BLN,Kp) * H

	#ifdef NMFGPU_DEBUG
	cublas_status =
	#endif
		// streams_NMF[ psNMF_N ]
		CUBLAS_R_GEMM( cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, BLN, K, d_one, d_H, Kp, &d_W[ rowIdx * Kp ], Kp, d_zero, d_WH, Mp );

			#ifdef NMFGPU_DEBUG
			///////////////////////////////
				printf("\n--- [GPU%" PRI_IDX "] Resulting WHrow=W*H (BLN=%" PRI_IDX ", Mp=%" PRI_IDX ", rowIdx=%" PRI_IDX
					"): ---\n", device_id, BLN, Mp, rowIdx );
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
			#if NMFGPU_PROFILING_KERNELS
				div_timing,
			#endif
			streams_NMF[ psNMF_N ], event_Vrow );

	// ---------------------------

	// Waux(BLN,Kp) = WH(BLN,Mp) * H'

	#ifdef NMFGPU_DEBUG
	cublas_status =
	#endif

	CUBLAS_R_GEMM( cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, K, BLN, M, d_one, d_H, Kp, d_WH, Mp, d_zero, d_Waux, Kp );

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

	matrix_mul_div( &d_W[ rowIdx * Kp ], d_Waux, d_accum_h, BLN, Kp,
			#ifdef NMFGPU_DEBUG
				K, false, "W", "Waux", "accum_H",
			#endif
			streams_NMF[ psNMF_N ] );


	// ----------------------------


	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) ) printf( "\n\t\t\t\t-----End of get_W_BLN(BLN=%" PRI_IDX ", rowIdx=%" PRI_IDX ")-----\n", BLN, rowIdx );
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
 * WARNING: CUBLAS stream must have been set to streams_NMF[psNMF_N].
 */
static void getW_loop( index_t const num_steps, index_t const BLN )
{

	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
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
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n\t\t\t-----getW_loop:(step %" PRI_IDX "/%" PRI_IDX ", BLN=%" PRI_IDX ", rowIdx=%" PRI_IDX
				", stepN=%i)-----\n", st, num_steps, BLN, rowIdx, stepN );
		#endif

		// ----------------

		// Transfers (asynchronously) a new <BLN x M> block from Vrow to d_Vrow

		// First, updates the index of current row.
		rowIdx += (stepN * BLN);

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
			///////////////////////////////
			printf("\n--- [GPU%" PRI_IDX "]: Vrow processed (BLN=%" PRI_IDX ", new rowIdx=%" PRI_IDX "): ---\n",
				device_id, BLN,  rowIdx );
			//////////////////////////////
		#endif
		upload_matrix_partial( Vrow, BLN, Mp, rowIdx, 0,	// Starting column: 0
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
						M, "Vrow", "d_Vrow",
					#endif
						Mp, d_Vrow, stream_Vrow, event_Vrow
					#if NMFGPU_PROFILING_TRANSF
						, &upload_Vrow_timing
					#endif
				);

		// ----------------

		// Changes main stream.
		psNMF_N += stepN;	// forward or backward

		// Changes CUBLAS stream.
		cublasSetStream( cublas_handle, streams_NMF[ psNMF_N ] );

		// ----------------

		/* WH(BLN,Mp) = W(BLN,Kp) * H
		 * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
		 * Waux(BLN,Kp) = WH(BLN,Mp) * H'
		 * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
		 */
		get_W_BLN( BLN );

		// ----------------

		#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n\t\t\t-----getW_loop: End of loop (step %" PRI_IDX " of %" PRI_IDX ", BLN=%" PRI_IDX ", stepN=%i"
				")-----\n", st, num_steps, BLN, stepN );
		#endif

	} // for ( st=1 ; st<num_steps ; st++ )

	// --------------------------

	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
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
void update_W( void )
{

	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n-----update_W(pBLN=%" PRI_IDX ", rowIdx=%" PRI_IDX ", stepN=%i)-----\n", pBLN, rowIdx, stepN );
	#endif

	// ----------------------------------

	// Reduces d_H to a row.

	matrix_to_row( d_H, M, Kp,
			#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG
				K, "d_H",
			#endif
			d_Aux, d_accum, stream_H );

	// ----------------------------------

	/*
	 * Processes <block_N.num_steps[pBLN]> blocks of size <block_N.BL[pBLN]>
	 */

	// Block configuration.
	index_t num_steps = block_N.num_steps[ pBLN ];	// Number of loops.
	index_t BLN = block_N.BL[ pBLN ];		// Number of rows.

	// --------------------------------

	// Changes CUBLAS stream.
	cublasSetStream( cublas_handle, streams_NMF[ psNMF_N ] );

	// Delays further operations until d_H is ready.
	{
		#if NMFGPU_DEBUG
			cudaError_t cuda_status = cudaSuccess;
		#endif

		for ( index_t i = 0, ps = psNMF_N ; i < num_steps ; i++, ps += stepN ) {

			#if NMFGPU_DEBUG
				cudaError_t cs =
			#endif

				cudaStreamWaitEvent( streams_NMF[ ps ], event_H, 0 );

			#if NMFGPU_DEBUG
				if ( cs != cudaSuccess )
					cuda_status = cs;
			#endif
		}

		///////////////////////////////
		#if NMFGPU_DEBUG
			if ( cuda_status != cudaSuccess ) {
				fflush(stdout);
				fprintf(stderr, "\n[GPU%" PRI_IDX "] Error: Could not delay operations until d_H is ready: %s\n"
					"Error in update_W(psNMF_N=%" PRI_IDX ", pBLN=%" PRI_IDX ", stepN=%i).\n", device_id,
					cudaGetErrorString(cuda_status), psNMF_N, pBLN, stepN );
			}
		#endif
		///////////////////////////////
	}

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
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n\t-----update_W(pBLN=%" PRI_IDX ",stepN=%i, psNMF_N=%" PRI_IDX ", rowIdx=%" PRI_IDX "): New block-----\n",
				pBLN, stepN, psNMF_N, rowIdx );
		#endif

		// Updates pointers to Vrow (HOST matrix) and d_W (DEVICE matrix) and changes block information.

		if ( stepN > 0 ) {	// Going forward

			// First, updates both pointers. Then, changes block information.

			// Rows ALREADY processed:
			rowIdx += BLN;

			// Changes block size
			pBLN = ! pBLN;

			// Updates block information
			BLN = block_N.BL[ pBLN ];		// Number of rows.

		} else {	// Going backward

			// First, changes block information. Then, updates pointers.

			// Changes block size
			pBLN = ! pBLN;

			// Updates block information
			BLN = block_N.BL[ pBLN ];		// Number of rows.

			// Rows TO BE processed (NOTE: offset is negative).
			rowIdx -= BLN;

		} // if ( stepN > 0 )

		// Updates other block informations
		num_steps = block_N.num_steps[ pBLN ];	// Number of loops.

		// Changes main stream.
		psNMF_N += stepN;			// forward or backward

		// Changes CUBLAS stream.
		cublasSetStream( cublas_handle, streams_NMF[ psNMF_N ] );

		// ----------------

		// Transfers (asynchronously) a new <BLN x M> block from Vrow to d_Vrow.

		#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n\t-----update_W: New block (pBLN=%" PRI_IDX ", stepN=%i, BLN=%" PRI_IDX ", num_steps=%" PRI_IDX
				", psNMF_N=%" PRI_IDX ", rowIdx=%" PRI_IDX ")-----\n", pBLN, stepN, BLN, num_steps, psNMF_N, rowIdx );
		#endif

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
			///////////////////////////////
			printf("\n--- [GPU%" PRI_IDX "]: Vrow processed (BLN=%" PRI_IDX ", M=%" PRI_IDX ", Mp=%" PRI_IDX ", rowIdx=%"
				PRI_IDX "): ---\n", device_id, BLN, M, Mp, rowIdx );
			//////////////////////////////
		#endif
		upload_matrix_partial( Vrow, BLN, Mp, rowIdx, 0,	// Starting column: 0
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
						M, "Vrow", "d_Vrow",
					#endif
					Mp, d_Vrow, stream_Vrow, event_Vrow
					#if NMFGPU_PROFILING_TRANSF
						, &upload_Vrow_timing
					#endif
				);

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
		if ( (! device_id) + (num_devices == 1) )
			printf("\n\t-----update_W: End of new block (pBLN=%" PRI_IDX ", BLN=%" PRI_IDX ", num_steps=%" PRI_IDX
				"). New StepN=%i -----\n", pBLN, BLN, num_steps, stepN );
		#endif

	} // if ( block_N.num_steps[1] > 0 )

	// -----------------------

	// Records as an event all previous operations on matrix W.
	{
		#if NMFGPU_DEBUG
			cudaError_t cuda_status =
		#endif

			cudaEventRecord( event_W, streams_NMF[ psNMF_N ] );

			///////////////////////////////
			#if NMFGPU_DEBUG
				if ( cuda_status != cudaSuccess ) {
					fflush(stdout);
					fprintf( stderr, "\n[GPU%" PRI_IDX "] Error recording CUDA event: %s\nError in update_W(pBLN=%"
						PRI_IDX ", stepN=%i).\n", device_id, cudaGetErrorString(cuda_status), pBLN, stepN );
				}
			#endif
			///////////////////////////////
	}

	// -----------------------

	// Delays further operations on "stream_W" until "event_W" completes.
	{
		#if NMFGPU_DEBUG
			cudaError_t cuda_status =
		#endif

			cudaStreamWaitEvent( stream_W, event_W, 0 );

			///////////////////////////////
			#if NMFGPU_DEBUG
				if ( cuda_status != cudaSuccess ) {
					fflush(stdout);
					fprintf(stderr, "\n[GPU%" PRI_IDX "] Error: could not delay operations until event_H completes: %s\n"
						"Error in update_W(pBLN=%" PRI_IDX ", stepN=%i).\n", device_id, cudaGetErrorString(cuda_status),
						pBLN, stepN );
				}
			#endif
			///////////////////////////////
	}

	// -----------------------

	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n-----End of update_W(pBLN=%" PRI_IDX ", rowIdx=%" PRI_IDX ", stepN=%i)-----\n", pBLN, rowIdx, stepN );
	#endif

} // update_W

/////////////////////////////////////////////////////////////////////

/*
 * Computes classification vector from matrix d_H (full size).
 * Result is downloaded from the GPU and stored in 'classification[]'.
 */
void get_classification( index_t *__restrict__ classification )
{

	#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
		printf("\n[GPU%" PRI_IDX "] get_classification()...\n", device_id );
	#elif defined(NMFGPU_VERBOSE_2)
		if ( (! device_id) + (num_devices == 1) )
			printf("\nget_classification()...\n");
	#endif

	// ---------------------------------

	// Stream for this operation.
	cudaStream_t stream_A = stream_H;

	// ---------------------------------

	// Computes the classification vector: Column index of highest values.

	matrix_idx_max( d_H, K, Kp, M,
			#if NMFGPU_DEBUG
				true, "d_H", "d_classification",
			#endif
			stream_A, d_classification );

	// ------------------------------

	// Downloads output vector.

	download_matrix_int( classification, 1, Mp, d_classification,
				#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
					M, false, "classification", "d_classification",
				#endif
				#if NMFGPU_PROFILING_TRANSF
					&download_classf_timing,
				#endif
				stream_A );

	// -----------------------------

	// Waits until complete.

	sync_GPU( stream_A );

	// -----------------------------

	#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
		printf("\n[GPU%" PRI_IDX "] get_classification()... Done.\n",device_id);
	#elif defined(NMFGPU_VERBOSE_2)
		if ( (! device_id) + (num_devices == 1) )
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
 * WARNING: CUBLAS stream must have been set to streams_NMF[psNMF_N].
 */
static void get_dot_VWH_BLN( index_t const BLN, real *__restrict__ d_dot_VWH, real *__restrict__ d_dot_V )
{

	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n\t\t\t\t-----get_dot_VWH_BLN(BLN=%" PRI_IDX ", rowIdx=%" PRI_IDX ")-----\n", BLN, rowIdx );
	#endif

	// ---------------------------------

	#ifdef NMFGPU_DEBUG
		cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
		cudaError_t cuda_status = cudaSuccess;
	#endif

	// Uses all the available streams, starting from streams_NMF[ psNMF_N ].
	index_t stream_idx = psNMF_N;

	// ----------------------------------

	// WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)

	#ifdef NMFGPU_DEBUG
		cublas_status =
	#endif
		// streams_NMF[ psNMF_N ]
		CUBLAS_R_GEMM( cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, BLN, K, d_one, d_H, Kp, &d_W[ rowIdx * Kp ], Kp, d_zero, d_WH, Mp );

			#ifdef NMFGPU_DEBUG
			///////////////////////////////
				printf("\n--- [GPU%" PRI_IDX "] Resulting WHrow=W*H (BLN=%" PRI_IDX ", Mp=%" PRI_IDX ", rowIdx=%" PRI_IDX
					"): ---\n", device_id, BLN, Mp, rowIdx );
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
			#if NMFGPU_PROFILING_KERNELS
				sub_timing,
			#endif
			streams_NMF[ psNMF_N ], event_Vrow );


	// ---------------------------

	/*
	 * dot_VWH(BLN) = SUM((V-WH)**2)
	 */

	// TODO: Change the loop below for a specific kernel, and just use streams_NMF[ psNMF_N ].

	real *pd_WH = d_WH;
	real *pd_dot_VWH = &d_dot_VWH[ rowIdx ];
	{
		#ifdef NMFGPU_DEBUG
			cublasStatus_t cs =
		#endif
			CUBLAS_R_DOT( cublas_handle, M, pd_WH, 1, pd_WH, 1, pd_dot_VWH );

			#ifdef NMFGPU_DEBUG
				if ( cs != CUBLAS_STATUS_SUCCESS );
					cublas_status = cs;
			#endif

		// --------------------

		// Changes CUBLAS stream
		stream_idx = psNMF_N + 1;
		if ( stream_idx == num_streams_NMF )
			stream_idx = 0;

		pd_WH += Mp;
		pd_dot_VWH++;
	}

	for ( index_t i = 1 ; i < BLN ; i++, pd_WH += Mp, pd_dot_VWH++ ) {

		// Sets new CUBLAS stream.
		cublasSetStream( cublas_handle, streams_NMF[ stream_idx ] );

		// Delays the operation until 'event_Vrow' completes.
		#if NMFGPU_DEBUG
			cudaError_t ce =
		#endif

			cudaStreamWaitEvent( streams_NMF[ stream_idx ], event_Vrow, 0 );

			#ifdef NMFGPU_DEBUG
				if ( ce != cudaSuccess );
					cuda_status = ce;
			#endif

		// --------------------

		#ifdef NMFGPU_DEBUG
		cublasStatus_t cs =
		#endif
			CUBLAS_R_DOT( cublas_handle, M, pd_WH, 1, pd_WH, 1, pd_dot_VWH );

			#ifdef NMFGPU_DEBUG
				if ( cs != CUBLAS_STATUS_SUCCESS );
					cublas_status = cs;
			#endif

		// --------------------

		// Changes of CUBLAS stream
		stream_idx++;
		if ( stream_idx == num_streams_NMF )
			stream_idx = 0;

	} // for

	#ifdef NMFGPU_DEBUG
	///////////////////////////////
		printf("\n--- [GPU%" PRI_IDX "] Resulting d_dot_VWH[] in get_dot_VWH_BLN(BLN=%" PRI_IDX ", Mp=%" PRI_IDX
			", rowIdx=%" PRI_IDX "): ---\n", device_id, BLN, Mp, rowIdx );
		check_cublas_status_st( cublas_status );
		check_cuda_status_st( cuda_status );
		show_device_matrix( &d_dot_VWH[ rowIdx ], 1, BLN, BLN, false, NULL );
		cublas_status = CUBLAS_STATUS_SUCCESS;	// Resets status values.
		cuda_status = cudaSuccess;
	/////////////////////////////
	#endif

	// --------------------------------------

	/*
	 * dot_V(BLN) = SUM(V**2)
	 */

	// TODO: Change the loop below for a specific kernel, and just use streams_NMF[ psNMF_N ].

	real *pd_Vrow = d_Vrow;
	real *pd_dot_V = &d_dot_V[ rowIdx ];
	for ( index_t i = 0 ; i < BLN ; i++, pd_Vrow += Mp, pd_dot_V++ ) {

		// Sets new CUBLAS stream.
		cublasSetStream( cublas_handle, streams_NMF[ stream_idx ] );

		// --------------------

		// Delays the operation until 'event_Vrow' completes.
		#if NMFGPU_DEBUG
			cudaError_t ce =
		#endif

			cudaStreamWaitEvent( streams_NMF[ stream_idx ], event_Vrow, 0 );

			#ifdef NMFGPU_DEBUG
				if ( ce != cudaSuccess );
					cuda_status = ce;
			#endif

		// --------------------

		#ifdef NMFGPU_DEBUG
			cublasStatus_t cs =
		#endif
			CUBLAS_R_DOT( cublas_handle, M, pd_Vrow, 1, pd_Vrow, 1, pd_dot_V );

			#ifdef NMFGPU_DEBUG
				if ( cs != CUBLAS_STATUS_SUCCESS );
					cublas_status = cs;
			#endif

		// --------------------

		// Changes of CUBLAS stream
		stream_idx++;
		if ( stream_idx == num_streams_NMF )
			stream_idx = 0;

	} // for


	#ifdef NMFGPU_DEBUG
	///////////////////////////////
		printf("\n--- [GPU%" PRI_IDX "] Resulting d_dot_V[] in get_dot_VWH_BLN(BLN=%" PRI_IDX ", Mp=%" PRI_IDX
			", rowIdx=%" PRI_IDX "): ---\n", device_id, BLN, Mp, rowIdx );
		check_cublas_status_st( cublas_status );
		check_cuda_status_st( cuda_status );
		show_device_matrix( &d_dot_V[ rowIdx ], 1, BLN, BLN, false, NULL );
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
 * WARNING: CUBLAS stream must have been set to streams_NMF[psNMF_N].
 */
static void get_dot_VWH_loop( index_t num_steps, index_t BLN, real *__restrict__ d_dot_VWH, real *__restrict__ d_dot_V )
{

	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
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
	get_dot_VWH_BLN( BLN, d_dot_VWH, d_dot_V );

	// -----------------------

	// Remaining (num_steps-1) iterations:

	for ( index_t st = 1 ; st < num_steps ; st++ ) {

		#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n\t\t\t-----get_dot_VWH_loop:(step %" PRI_IDX "/%" PRI_IDX ", BLN=%" PRI_IDX ", stepN=%i)-----\n",
				st, num_steps, BLN, stepN );
		#endif

		// ----------------

		// Transfers (asynchronously) a new <BLN x M> block from Vrow to d_Vrow

		// First, updates the index of current row.
		rowIdx += (stepN * BLN);

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
			///////////////////////////////
			printf("\n--- [GPU%" PRI_IDX "]: Vrow processed (BLN=%" PRI_IDX ", new rowIdx=%" PRI_IDX "): ---\n",
				device_id, BLN, rowIdx);
			//////////////////////////////
		#endif

		upload_matrix_partial( Vrow, BLN, Mp, rowIdx, 0,	// Starting column: 0
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
						M, "Vrow", "d_Vrow",
					#endif
					Mp, d_Vrow, stream_Vrow, event_Vrow
					#if NMFGPU_PROFILING_TRANSF
						, &upload_Vrow_timing
					#endif
				);

		// ----------------

		// Changes main stream.
		psNMF_N += stepN;	// forward or backward

		// Changes CUBLAS stream.
		cublasSetStream( cublas_handle, streams_NMF[ psNMF_N ] );

		// ----------------

		/* WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
		 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
		 * dot_VWH(BLN) = SUM((V-WH)**2)
		 * dot_V(BLN)	= SUM(V**2)
		 */
		get_dot_VWH_BLN( BLN, d_dot_VWH, d_dot_V );

		#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n\t\t\t-----get_dot_VWH_loop: End of loop (step %" PRI_IDX " of %" PRI_IDX ", BLN=%" PRI_IDX
				", stepN=%i)-----\n", st, num_steps, BLN, stepN );
		#endif

	} // for ( st=1 ; st<num_steps ; st++ )

	// --------------------------

	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
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
static void get_dot_VWH_N( real *__restrict__ d_dot_VWH, real *__restrict__ d_dot_V )
{

	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n-----get_dot_VWH_N(pBLN=%" PRI_IDX ", rowIdx=%" PRI_IDX ", stepN=%i)-----\n", pBLN, rowIdx, stepN);
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
	cublasSetStream( cublas_handle, streams_NMF[ psNMF_N ] );

	// Delays further operations until d_H is ready.
	{
		#if NMFGPU_DEBUG
			cudaError_t cuda_status = cudaSuccess;
		#endif

		for ( index_t i = 0, ps = psNMF_N ; i < num_steps ; i++, ps += stepN ) {

			#if NMFGPU_DEBUG
				cudaError_t cs =
			#endif

				cudaStreamWaitEvent( streams_NMF[ ps ], event_H, 0 );

			#if NMFGPU_DEBUG
				if ( cs != cudaSuccess )
					cuda_status = cs;
			#endif
		}

		///////////////////////////////
		#if NMFGPU_DEBUG
			if ( cuda_status != cudaSuccess ) {
				fflush(stdout);
				fprintf(stderr, "\n[GPU%" PRI_IDX "] Error: Could not delay operations until d_H is ready: %s\n"
					"Error in get_dot_VWH_N(psNMF_N=%" PRI_IDX ", pBLN=%" PRI_IDX ", stepN=%i).\n", device_id,
					cudaGetErrorString(cuda_status), psNMF_N, pBLN, stepN );
			}
		#endif
		///////////////////////////////
	}

	// --------------------------------

	/* WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
	 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
	 * dot_VWH(BLN) = SUM((V-WH)**2)
	 * dot_V(BLN)	= SUM(V**2)
	 */
	get_dot_VWH_loop( num_steps, BLN, d_dot_VWH, d_dot_V );

	// --------------------------------

	// Remaining blocks

	if ( block_N.num_steps[1] ) {  // There are more blocks in dimension "N" to process.

		#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n\t-----get_dot_VWH_N(pBLN=%" PRI_IDX ",stepN=%i,psNMF_N=%" PRI_IDX ", rowIdx=%" PRI_IDX
				"): New block-----\n", pBLN, stepN, psNMF_N, rowIdx );
		#endif

		// Updates pointers to Vrow (HOST matrix) and d_W (DEVICE matrix) and changes block information.

		if ( stepN > 0 ) {	// Going forward

			// First, updates both pointers. Then, changes block information.

			// Rows ALREADY processed:
			rowIdx += BLN;

			// Changes block size
			pBLN = ! pBLN;

			// Updates block information
			BLN = block_N.BL[ pBLN ];		// Number of rows.

		} else {	// Going backward

			// First, changes block information. Then, updates pointers.

			// Changes block size
			pBLN = ! pBLN;

			// Updates block information
			BLN = block_N.BL[ pBLN ];		// Number of rows.

			// Rows TO BE processed (NOTE: offset is negative).
			rowIdx -= BLN;

		} // if ( stepN > 0 )

		// Updates other block informations
		num_steps = block_N.num_steps[ pBLN ];	// Number of loops.

		// Changes main stream.
		psNMF_N += stepN;			// forward or backward

		// Changes CUBLAS stream.
		cublasSetStream( cublas_handle, streams_NMF[ psNMF_N ] );

		// ----------------

		// Transfers (asynchronously) a new <BLN x M> block from Vrow to d_Vrow.

		#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n\t-----get_dot_VWH_N: New block (pBLN=%" PRI_IDX ", stepN=%i, BLN=%" PRI_IDX ", num_steps=%" PRI_IDX
				", psNMF_N=%" PRI_IDX ", rowIdx=%" PRI_IDX ")-----\n", pBLN, stepN, BLN, num_steps, psNMF_N, rowIdx );
		#endif

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
			///////////////////////////////
			printf("\n--- [GPU%" PRI_IDX "]: Vrow processed (BLN=%" PRI_IDX ", M=%" PRI_IDX ", Mp=%" PRI_IDX
				", rowIdx=%" PRI_IDX "): ---\n", device_id, BLN, M, Mp, rowIdx );
			//////////////////////////////
		#endif
		upload_matrix_partial( Vrow, BLN, Mp, rowIdx, 0,	// Starting column: 0
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
						M, "Vrow", "d_Vrow",
					#endif
					Mp, d_Vrow, stream_Vrow, event_Vrow
					#if NMFGPU_PROFILING_TRANSF
						, &upload_Vrow_timing
					#endif
					);

		// ---------------------------

		// Processes that block.

		/* WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
		 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
		 * dot_VWH(BLN) = SUM((V-WH)**2)
		 * dot_V(BLN)	= SUM(V**2)
		 */
		get_dot_VWH_loop( num_steps, BLN, d_dot_VWH, d_dot_V );

		// -------------------------

		// Changes direction (forward -> backward || backward -> forward)
		stepN *= (-1);

		#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf("\n\t-----get_dot_VWH_N: End of new block (pBLN=%" PRI_IDX ", BLN=%" PRI_IDX ", num_steps=%" PRI_IDX
				"). New StepN=%i -----\n", pBLN, BLN, num_steps, stepN );
		#endif

	} // if ( block_N.num_steps[1] > 0 )

	// -----------------------

	// Records as an event all previous operations.
	{
		#if NMFGPU_DEBUG
			cudaError_t cuda_status =
		#endif

			cudaEventRecord( event_W, 0 );

			///////////////////////////////
			#if NMFGPU_DEBUG
				if ( cuda_status != cudaSuccess ) {
					fflush(stdout);
					fprintf( stderr, "\n[GPU%" PRI_IDX "] Error recording CUDA event: %s\nError in get_dot_VWH_N(pBLN=%"
						PRI_IDX ", stepN=%i).\n", device_id, cudaGetErrorString(cuda_status), pBLN, stepN );
				}
			#endif
			///////////////////////////////
	}

	// -----------------------

	// Delays further operations on "stream_W" until "event_W" completes.
	{
		#if NMFGPU_DEBUG
			cudaError_t cuda_status =
		#endif

			cudaStreamWaitEvent( stream_W, event_W, 0 );

			///////////////////////////////
			#if NMFGPU_DEBUG
				if ( cuda_status != cudaSuccess ) {
					fflush(stdout);
					fprintf(stderr, "\n[GPU%" PRI_IDX "] Error: could not delay operations until event_H completes: %s\n"
							"Error in get_dot_VWH_N(pBLN=%" PRI_IDX ", stepN=%i).\n", device_id,
						cudaGetErrorString(cuda_status), pBLN, stepN );
				}
			#endif
			///////////////////////////////
	}

	// -----------------------

	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n-----End of get_dot_VWH_N(pBLN=%" PRI_IDX ", rowIdx=%" PRI_IDX ", stepN=%i)-----\n",
				pBLN, rowIdx, stepN);
	#endif

} // get_dot_VWH_N

// =======================================================================

/*
 * WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
 * dot_VWH(BLN) = SUM((V-WH)**2)
 * dot_V(BLN)	= SUM(V**2)
 * d_scalars_VWH[0] = SUM(dot_VWH[...])
 * d_scalars_VWH[1] = SUM(dot_V[...])
 *
 * Computes vectors d_dot_VWH[N] and d_dot_V[N], and reduces each to a single scalar.
 * Resulting values are returned in d_scalars_VWH[0] and d_scalars_VWH[1], respectively.
 */
static void get_dot_VWH( real *__restrict__ d_dot_VWH, real *__restrict__ d_dot_V, real *__restrict__ d_scalars_VWH )
{

	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n-----get_dot_VWH(pBLN=%" PRI_IDX ", rowIdx=%" PRI_IDX ", stepN=%i)-----\n", pBLN, rowIdx, stepN);
	#endif

	#ifdef NMFGPU_DEBUG
		cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	#endif

	// -----------------------

	/* WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
	 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
	 * dot_VWH(BLN) = SUM((V-WH)**2)
	 * dot_V(BLN)	= SUM(V**2)
	 */
	get_dot_VWH_N( d_dot_VWH, d_dot_V );

	// -----------------------

	// d_scalars_VWH[0] = SUM(dot_VWH[...])

	// Since all values are positive, we can use the sums of absolute values.
	{
		#ifdef NMFGPU_DEBUG
			cublasStatus_t cs =
		#endif
			CUBLAS_R_ASUM( cublas_handle, N, d_dot_VWH, 1, d_scalars_VWH );	// streams_NMF[ psNMF_N ]

			#ifdef NMFGPU_DEBUG
				if ( cs != CUBLAS_STATUS_SUCCESS );
					cublas_status = cs;
			#endif
	}

	// -----------------------

	// Changes the CUBLAS stream
	index_t stream_idx = psNMF_N + 1;
	if ( stream_idx == num_streams_NMF )
		stream_idx = 0;

	cublasSetStream( cublas_handle, streams_NMF[ stream_idx ] );

	// -----------------------

	// d_scalars_VWH[1] = SUM(dot_VWH[...])
	{
		#ifdef NMFGPU_DEBUG
			cublasStatus_t cs =
		#endif
			// streams_NMF[ (psNMF_N + 1) % num_streams_NMF ]
			CUBLAS_R_ASUM( cublas_handle, N, d_dot_V, 1, &d_scalars_VWH[1] );

			#ifdef NMFGPU_DEBUG
				if ( cs != CUBLAS_STATUS_SUCCESS );
					cublas_status = cs;
			#endif
	}

	// -----------------------

	#ifdef NMFGPU_DEBUG
	///////////////////////////////
		printf("\n--- [GPU%" PRI_IDX "] Resulting scalars in get_dot_VWH(): ---\n", device_id );
		check_cublas_status_st( cublas_status );
		check_cuda_status();
		show_device_matrix( d_scalars_VWH, 1, 2, 2, false, NULL );
	/////////////////////////////
	#endif

	// -----------------------

	// Records as an event all previous operations.
	{
		#if NMFGPU_DEBUG
			cudaError_t cuda_status =
		#endif

			cudaEventRecord( event_W, 0 );

			///////////////////////////////
			#if NMFGPU_DEBUG
				if ( cuda_status != cudaSuccess ) {
					fflush(stdout);
					fprintf( stderr, "\n[GPU%" PRI_IDX "] Error recording CUDA event: %s\nError in get_dot_VWH(pBLN=%"
						PRI_IDX ", stepN=%i).\n", device_id, cudaGetErrorString(cuda_status), pBLN, stepN );
				}
			#endif
			///////////////////////////////
	}

	// -----------------------

	// Delays further operations on "stream_W" until "event_W" completes.
	{
		#if NMFGPU_DEBUG
			cudaError_t cuda_status =
		#endif

			cudaStreamWaitEvent( stream_W, event_W, 0 );

			///////////////////////////////
			#if NMFGPU_DEBUG
				if ( cuda_status != cudaSuccess ) {
					fflush(stdout);
					fprintf(stderr, "\n[GPU%" PRI_IDX "] Error: could not delay operations until event_H completes: %s\n"
							"Error in get_dot_VWH(pBLN=%" PRI_IDX ", stepN=%i).\n", device_id,
						cudaGetErrorString(cuda_status), pBLN, stepN );
				}
			#endif
			///////////////////////////////
	}

	// -----------------------

	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n-----End of get_dot_VWH(pBLN=%" PRI_IDX ", rowIdx=%" PRI_IDX ", stepN=%i)-----\n",
				pBLN, rowIdx, stepN);
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

	#ifdef NMFGPU_VERBOSE_2
		if ( (! device_id) + (num_devices == 1) )
			printf( "\n-----Starting dot_product_VWH( rowIdx=%" PRI_IDX " )-----\n", rowIdx);
	#endif

	/* Sets pointers to be used as data matrix.
	 * We can use "d_Aux", since we need memory for two N-length vectors,
	 * and size(d_Aux) == (MAX(N,M) * Kp), (with Kp >= memory_alignment).
	 */
	real *d_dot_VWH = d_Aux;
	real *d_dot_V = &d_Aux[ N ];

	/* Such vectors will be later reduced. Resulting scalars will be
	 * stored in d_scalars_VWH[2]. We can use d_accum[2] for that.
	 */
	real *d_scalars_VWH = d_accum;

	// ----------------------------------------

	/* WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
	 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
	 * dot_VWH(BLN) = SUM((V-WH)**2)
	 * dot_V(BLN)	= SUM(V**2)
	 * d_scalars_VWH[0] = SUM(dot_VWH[...])
	 * d_scalars_VWH[1] = SUM(dot_V[...])
	 */
	get_dot_VWH( d_dot_VWH, d_dot_V, d_scalars_VWH );

	// ----------------------------------------

	// Downloads partial results.

	// Size = 2: d_dot_VWH and d_dot_V
	real h_dot_VWH[ 2 ];

	download_matrix( h_dot_VWH, 1, 2, d_scalars_VWH,
			#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
				2, false, "h_dot_VWH", "d_scalars_VWH",
			#endif
			#if NMFGPU_PROFILING_TRANSF
			NULL,
			#endif
			stream_W );

	// Waits for the results...
	sync_GPU( stream_W );

	// ----------------------------------------

	#if NMFGPU_DEBUG || NMFGPU_VERBOSE
		///////////////////////////////
		printf("\n[GPU%" PRI_IDX "] dot_product_VWH: dot_VWH=%g dot_V=%g\n", device_id, h_dot_VWH[0], h_dot_VWH[1] );
		///////////////////////////////
	#endif

	*dot_VWH = h_dot_VWH[0];
	*dot_V = h_dot_VWH[1];

	return EXIT_SUCCESS;

} // dot_product_VWH

////////////////////////////////////////////////////////////////////////
