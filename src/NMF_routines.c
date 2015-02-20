/************************************************************************
 *
 * NMF-mGPU - Non-negative Matrix Factorization on multi-GPU systems.
 *
 * Copyright (C) 2011-2015:
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
 * NMF_routines.c
 *	Routines that implement the NMF algorithm.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Data types, functions and constants:
 *		NMFGPU_SINGLE_PREC: Makes use of single-precision data (i.e., 'float').
 *		NMFGPU_CUDA_HOST: Defines some constants and functions related to CUDA Runtime, cuBLAS and cuRAND.
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

#if ! NMFGPU_CUDA_HOST
	#define NMFGPU_CUDA_HOST (1)	/* CUDA runtime, cuBLAS and cuRAND, constants and functions on real_type.h */
#endif

#include "NMF_routines.h"
#include "matrix_operations.h"
#include "GPU_setup.h"
#if NMFGPU_PROFILING_TRANSF
	#include "timing.h"
#endif
#include "common.h"
#include "index_type.h"
#include "real_type.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <string.h>	/* strerror() */
#include <errno.h>
#include <stdbool.h>
#include <stdlib.h>

////////////////////////////////////////////////
////////////////////////////////////////////////

/* HOST-ONLY GLOBAL Variables */

index_t pBLN = 0;	// Current index in block_N.xxx[].
index_t pBLM = 0;	// Current index in block_M.xxx[].

int stepN = 1;		// Loop directions: +1 (forward) || -1 (backward).
int stepM = 1;		// Loop directions: +1 (forward) || -1 (backward).

index_t psNMF_N = 0;	// Current index in streams_NMF[].
index_t psNMF_M = 0;	// Current index in streams_NMF[].

index_t colIdx = 0;	// Current column index in Vcol. It corresponds to <bM + colIdx> in H and d_H.
index_t rowIdx = 0;	// Current row index in Vrow. It corresponds to <bN + rowIdx> in W and d_W.

// ---------------------------------------------

/* "Private" Global variables */

#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE || NMFGPU_VERBOSE_2
	static bool const dbg_shown_by_all = true;		// Information messages in debug mode.
	static bool const verb_shown_by_all = false;		// Information messages in verbose mode.
#endif

#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
	static bool const sys_error_shown_by_all = true;	// System error messages
#endif

////////////////////////////////////////////////
////////////////////////////////////////////////

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

		#if NMFGPU_DEBUG || NMFGPU_VERBOSE || NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "\nInitializing the HOST (i.e., CPU) random-numbers generator...\n" );
		#endif

		// Initializes random generator on CPU.
		srandom( seed );

	#else

		#if NMFGPU_DEBUG || NMFGPU_VERBOSE || NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "\nInitializing the DEVICE (i.e., GPU) random-numbers generator...\n" );
		#endif

		// Initializes random generator on GPU.
		status = init_GPU_random( seed );

	#endif

	return status;

} // init_random

////////////////////////////////////////////////

/*
 * Finalizes the selected random generator.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int destroy_random( void )
{

	int status = EXIT_SUCCESS;

	#if ! NMFGPU_CPU_RANDOM

		status = finalize_randomGenerator();

	#endif

	return status;

} // destroy_random

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
int set_random_values( real *restrict d_A, index_t height, index_t width, index_t padding,
			#if NMFGPU_CPU_RANDOM
				real *restrict A,
			#endif
			#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 || (NMFGPU_CPU_RANDOM && NMFGPU_DEBUG_TRANSF)
				bool transpose,
			#endif
			#if NMFGPU_CPU_RANDOM && (NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 || (! NMFGPU_PROFILING_GLOBAL))
				char const *restrict const matrix_name_A,
			#endif
			#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 || (NMFGPU_CPU_RANDOM && NMFGPU_DEBUG_TRANSF) \
				|| ((! NMFGPU_CPU_RANDOM) && (! NMFGPU_PROFILING_GLOBAL))
				char const *restrict const matrix_name_dA,
			#endif
			#if ( NMFGPU_CPU_RANDOM && NMFGPU_PROFILING_TRANSF )
				timing_data_t *restrict const upload_timing,
			#endif
			cudaStream_t stream_A, cudaEvent_t *restrict event_A )
{

	int status = EXIT_SUCCESS;

	// ---------------------------------

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
			print_message( verb_shown_by_all, "\n--- Random values on matrix %s --> %s (height=%" PRI_IDX ", width=%" PRI_IDX
					", padding=%" PRI_IDX ", transpose=%i): ---\n", matrix_name_A, matrix_name_dA,
					height, width, padding, transpose );
		#endif
		/////////////////////////////

		// Uploads the new values.
		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			upload_matrix( A, height, padding, d_A,
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
						width, transpose,
					#endif
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 || (! NMFGPU_PROFILING_GLOBAL)
						matrix_name_A,
					#endif
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
						matrix_name_dA,
					#endif
					#if NMFGPU_PROFILING_TRANSF
						upload_timing,
					#endif
					stream_A, event_A );

	// ---------------------------------

	#else	/* ! NMFGPU_CPU_RANDOM */

		// Device random generator

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			matrix_random( d_A, height, width, padding,
					#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2
						transpose,
					#endif
					#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 || (! NMFGPU_PROFILING_GLOBAL)
						matrix_name_dA,
					#endif
					stream_A, event_A );

	#endif

	// ---------------------------------

	return status;

} // set_random_values

////////////////////////////////////////////////

/*
 * WH(N,BLMp) = W * H(BLM,Kp)
 * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
 * Haux(BLM,Kp) = W' * WH(N,BLMp)
 * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W(Kp)
 *
 * WARNING: CUBLAS stream must have been set to streams_NMF[psNMF_M].
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int get_H_BLM( index_t BLM, index_t BLMp )
{

	#ifdef NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "\t\t\t\t-----get_H_BLM(BLM=%" PRI_IDX ", BLMp=%" PRI_IDX
				", colIdx=%" PRI_IDX ")-----\n", BLM, BLMp, colIdx );
	#endif

	// ---------------------------------

	real *const d_Haux = d_Aux;		// Temporary matrix: W' * WH.
	real *const d_accum_w = d_accum;	// Accumulator vector: SUM(W).

	size_t const offset_dH = (size_t) (bM + colIdx) * (size_t) Kp;

	// ----------------------------------

	// WH(N,BLMp) = W * H(BLM,Kp)
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cublasStatus_t const cublas_status =
		#endif

			// streams_NMF[ psNMF_M ]
			CUBLAS_R_GEMM( cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, BLM, N, K, d_one, &d_H[ offset_dH ], Kp, d_W, Kp,
					d_zero, d_WH, BLMp );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
				print_error( sys_error_shown_by_all, "get_H_BLM(): cublas_gemm( W * H ): %s\n",
						getCublasErrorString( cublas_status ) );
				return EXIT_FAILURE;
			}
		#endif

		#ifdef NMFGPU_DEBUG
		///////////////////////////////
		{
			print_message( dbg_shown_by_all, "--- Resulting WHcol=W*H (BLM=%" PRI_IDX ", BLMp=%" PRI_IDX
					", colIdx=%" PRI_IDX "): ---\n", BLM, BLMp, colIdx );
			int const status1 = check_cuda_status();
			bool const real_data = true;
			bool const transpose = false;
			struct matrix_tags_t const *restrict mt = NULL;
			int const status2 = show_device_matrix( d_WH, N, BLM, BLMp, real_data, transpose, dbg_shown_by_all, mt );
			if ( (status1 != EXIT_SUCCESS) + (status2 != EXIT_SUCCESS) )
				return EXIT_FAILURE;
		}
		/////////////////////////////
		#endif
	}

	// ---------------------------

        // WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

		matrix_div_sub( d_WH, d_Vcol, N, BLMp,
				#ifdef NMFGPU_DEBUG
					BLM,
				#endif
				#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
					"WHcol", "Vcol",
				#endif
				true,	// division
				#if NMFGPU_PROFILING_KERNELS
					div_timing,
				#endif
				streams_NMF[ psNMF_M ], event_Vcol );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// ----------------------------

        // Haux(BLM,Kp) = W' * WH(N,BLMp)
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cublasStatus_t const cublas_status =
		#endif

			// streams_NMF[ psNMF_M ]
			CUBLAS_R_GEMM( cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, K, BLM, N, d_one, d_W, Kp, d_WH, BLMp, d_zero, d_Haux, Kp );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
				print_error( sys_error_shown_by_all, "get_H_BLM(): cublas_gemm( W' * WH ): %s\n",
						getCublasErrorString( cublas_status ) );
				return EXIT_FAILURE;
			}
		#endif

		#ifdef NMFGPU_DEBUG
		///////////////////////////////
		{
			print_message( dbg_shown_by_all, "--- Resulting d_Haux (BLM=%" PRI_IDX "): ---\n", BLM );
			int const status1 = check_cuda_status();
			bool const real_data = true;
			bool const transpose = true;
			struct matrix_tags_t const *restrict mt = NULL;
			int const status2 = show_device_matrix( d_Haux, BLM, K, Kp, real_data, transpose, dbg_shown_by_all, mt );
			if ( (status1 != EXIT_SUCCESS) + (status2 != EXIT_SUCCESS) )
				return EXIT_FAILURE;
		}
		/////////////////////////////
		#endif
	}

	// ----------------------------

        // H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W(Kp)
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

		matrix_mul_div( &d_H[ offset_dH ], d_Haux, d_accum_w, BLM, Kp,
				#ifdef NMFGPU_DEBUG
					K, true,	// transpose
				#endif
				#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
					"H",
				#endif
				#if NMFGPU_DEBUG
					"Haux", "accum_W",
				#endif
				streams_NMF[ psNMF_M ] );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// ----------------------------

	#ifdef NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "\t\t\t\t-----End of get_H_BLM(BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ", colIdx=%" PRI_IDX
				")-----\n", BLM, BLMp, colIdx );
	#endif

	return EXIT_SUCCESS;

} // get_H_BLM

// ---------------------------------------------

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
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int getH_loop( index_t const num_steps, index_t const BLM, index_t const BLMp )
{

	#ifdef NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "\t\t----- getH_loop(BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ", num_steps=%" PRI_IDX
				", stepM=%i, colIdx=%" PRI_IDX ") -----\n", BLM, BLMp, num_steps, stepM, colIdx );
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
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

		get_H_BLM( BLM, BLMp );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// --------------------

	// Remaining (num_steps-1) iterations:

	for ( index_t st = 1 ; st < num_steps ; st++ ) {

		#ifdef NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "\t\t\t-----getH_loop:(step %" PRI_IDX "/%" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%"
					PRI_IDX ", stepM=%i, colIdx=%" PRI_IDX ")-----\n", st, num_steps, BLM, BLMp, stepM, colIdx );
		#endif

		// ----------------

		// Transfers (asynchronously) a new <N x BLM> block from Vcol to d_Vcol.

		// First, updates the column index.
		colIdx += (stepM * BLM);

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
		///////////////////////////////
			print_message( dbg_shown_by_all, "--- Vcol processed (N=%" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ", new colIdx=%"
					PRI_IDX ", pitch=%" PRI_IDX "): ---\n", N, BLM, BLMp, colIdx, MpPp);
		//////////////////////////////
		#endif
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				int const status =
			#endif

			upload_matrix_partial( Vcol, N, MpPp, 0, colIdx,	// Starting row: 0
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
							BLM,
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 || (! NMFGPU_PROFILING_GLOBAL)
							"Vcol",
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
							"d_Vcol",
						#endif
						BLMp, d_Vcol, stream_Vcol, event_Vcol
						#if NMFGPU_PROFILING_TRANSF
							, &upload_Vcol_timing
						#endif
						);

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					return EXIT_FAILURE;
			#endif
		}

		// ----------------

		// Changes the main stream.
		psNMF_M += stepM;	// forward or backward

		// Sets new CUBLAS stream.
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				cublasStatus_t const cublas_status =
			#endif

				cublasSetStream( cublas_handle, streams_NMF[ psNMF_M ] );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
					print_error( sys_error_shown_by_all, "getH_loop(): cublasSetStream( streams_NMF[" PRI_IDX
							"] ): %s\n", psNMF_M, getCublasErrorString( cublas_status ) );
					return EXIT_FAILURE;
				}
			#endif
		}

		// -----------------------

		/* WH(N,BLMp) = W * pH(BLM,Kp)
		 * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
		 * Haux(BLM,Kp) = W' * WH(N,BLMp)
		 * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W
		 */
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				int const status =
			#endif

			get_H_BLM( BLM, BLMp );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					return EXIT_FAILURE;
			#endif
		}

		#ifdef NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "\t\t\t-----getH_loop: End of loop (step %" PRI_IDX " of %" PRI_IDX ", BLM=%" PRI_IDX
					", BLMp=%" PRI_IDX ", stepM=%i, colIdx=%" PRI_IDX ")-----\n", st, num_steps, BLM, BLMp, stepM, colIdx );
		#endif

	} // for ( st=1 ; st<num_steps ; st++ )

	// -------------------------------

	#ifdef NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "\t\t-----End of getH_loop(BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ", num_steps=%" PRI_IDX
				", stepM=%i, colIdx=%" PRI_IDX ")-----\n", BLM, BLMp, num_steps, stepM, colIdx );
	#endif

	return EXIT_SUCCESS;

} // getH_loop

// ---------------------------------------------

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
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int update_H( void )
{

	#ifdef NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "-----update_H(pBLM=%" PRI_IDX ", stepM=%i, colIdx=%" PRI_IDX ")-----\n",
				pBLM, stepM, colIdx );
	#endif

	// ----------------------------------

	// Reduces d_W to a row.
	{
		#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

		matrix_to_row( d_W, N, Kp,
				#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG
					K,
				#endif
				#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
					"d_W",
				#endif
				d_Aux, d_accum, stream_W );

		#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

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
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cublasStatus_t const cublas_status =
		#endif

		cublasSetStream( cublas_handle, streams_NMF[ psNMF_M ] );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
				print_error( sys_error_shown_by_all, "update_H(): cublasSetStream( streams_NMF[" PRI_IDX
						"] ): %s\n", psNMF_M, getCublasErrorString( cublas_status ) );
				return EXIT_FAILURE;
			}
		#endif
	}

	// --------------------------------

	// Delays further operations until d_W is ready.
	for ( index_t i = 0, ps = psNMF_M ; i < num_steps ; i++, ps += stepM ) {

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cudaError_t const cuda_status =
		#endif

			cudaStreamWaitEvent( streams_NMF[ ps ], event_W, 0 );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "update_H(psNMF_M=%" PRI_IDX ", pBLM=%" PRI_IDX
						", stepM=%i, colIdx=%" PRI_IDX "): cudaStreamWaitEvent( streams_NMF[%"
						PRI_IDX "], event_W ): %s\n", psNMF_M, pBLM, stepM, colIdx, ps,
						cudaGetErrorString(cuda_status) );
				return EXIT_FAILURE;
			}
		#endif

	} // for

	// --------------------------------

	/* WH(N,BLMp) = W * pH(BLM,Kp)
	 * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
	 * Haux(BLM,Kp) = W' * WH(N,BLMp)
	 * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W
	 */
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

		getH_loop( num_steps, BLM, BLMp );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// --------------------------------

	// Remaining blocks

	if ( block_M.num_steps[1] ) {	// There are more blocks in dimension "M" to process.

		#ifdef NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "\t-----update_H(pBLM=%" PRI_IDX ",stepM=%i,colIdx=%" PRI_IDX ",psNMF_M=%" PRI_IDX
					"): New block-----\n", pBLM, stepM, colIdx, psNMF_M );
		#endif

		// Updates pointers to Vcol (HOST matrix) and d_H (DEVICE matrix), and changes block information.

		if ( stepM > 0 ) {	// Going forward

			// First, updates both pointers. Then, changes block information.

			// Columns ALREADY processed:
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

			// Columns TO BE processed (NOTE: offset is negative).
			colIdx -= BLM;

		} // if ( stepM > 0 )

		// Updates other block informations
		num_steps = block_M.num_steps[ pBLM ];	// Number of loops.
		BLMp = block_M.BLp[ pBLM ];		// Number of columns (with padding).

		// Changes main stream.
		psNMF_M += stepM;	// forward or backward

		// Changes CUBLAS stream.
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				cublasStatus_t const cublas_status =
			#endif

				cublasSetStream( cublas_handle, streams_NMF[ psNMF_M ] );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
					print_error( sys_error_shown_by_all, "update_H(new block): cublasSetStream( streams_NMF[" PRI_IDX
							"] ): %s\n", psNMF_M, getCublasErrorString( cublas_status ) );
					return EXIT_FAILURE;
				}
			#endif
		}

		// ------------------------

		// Transfers (asynchronously) a new <N x BLM> block from Vcol to d_Vcol.

		#ifdef NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "\t-----update_H: New block (pBLM=%" PRI_IDX ", stepM=%i, BLM=%" PRI_IDX ", BLMp=%"
					PRI_IDX ", num_steps=%" PRI_IDX ", colIdx=%" PRI_IDX ", psNMF_M=%" PRI_IDX ")-----\n", pBLM,
					stepM, BLM, BLMp, num_steps, colIdx, psNMF_M );
		#endif

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
		///////////////////////////////
			print_message( dbg_shown_by_all, "--- Vcol processed (N=%" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%" PRI_IDX ") (offset=%"
					PRI_IDX ", pitch=%" PRI_IDX "): ---\n", N, BLM, BLMp, colIdx, MpPp);
		//////////////////////////////
		#endif
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				int const status =
			#endif

			upload_matrix_partial( Vcol, N, MpPp, 0, colIdx,	// Starting row: 0
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
							BLM,
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 || (! NMFGPU_PROFILING_GLOBAL)
							"Vcol",
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
							"d_Vcol",
						#endif
						BLMp, d_Vcol, stream_Vcol, event_Vcol
						#if NMFGPU_PROFILING_TRANSF
							, &upload_Vcol_timing
						#endif
						);

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					return EXIT_FAILURE;
			#endif
		}

		// -------------------------

		// Processes that block.

		/* WH(N,BLMp) = W * pH(BLM,Kp)
		 * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
		 * Haux(BLM,Kp) = W' * WH(N,BLMp)
		 * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W
		 */
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				int const status =
			#endif

			getH_loop( num_steps, BLM, BLMp );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					return EXIT_FAILURE;
			#endif
		}

		// -------------------------

		// Changes direction (forward -> backward || backward -> forward)
		stepM *= (-1);

		#ifdef NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "\t-----update_H: End of new block (pBLM=%" PRI_IDX ", BLM=%" PRI_IDX ", BLMp=%"
					PRI_IDX ", num_steps=%" PRI_IDX ", colIdx=%" PRI_IDX "). New StepM=%i -----\n", pBLM, BLM, BLMp,
					num_steps, colIdx, stepM );
		#endif

	} // if ( block_M.num_steps[1] > 0 )

	// ------------------------

	// Records as an event all previous operations on matrix H.
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cudaError_t const cuda_status =
		#endif

			cudaEventRecord( event_H, streams_NMF[ psNMF_M ] );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "Error in update_H( pBLM=%" PRI_IDX ", stepM=%i, colIdx=%" PRI_IDX
						"): cudaEventRecord( event_H, streams_NMF[%" PRI_IDX "] ): %s\n", pBLM, stepM, colIdx,
						psNMF_M, cudaGetErrorString(cuda_status) );
				return EXIT_FAILURE;
			}
		#endif
	}

	// ------------------------

	// Delays further operations on "stream_H" until "event_H" completes.
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cudaError_t const cuda_status =
		#endif

			cudaStreamWaitEvent( stream_H, event_H, 0 );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "Error in update_H( pBLM=%" PRI_IDX ", stepM=%i, colIdx=%"
						PRI_IDX "): cudaStreamWaitEvent( stream_H, event_H ): %s\n", pBLM, stepM, colIdx,
						cudaGetErrorString(cuda_status) );
				return EXIT_FAILURE;
			}
		#endif
	}

	// ------------------------

	#ifdef NMFGPU_VERBOSE_2
		print_message( dbg_shown_by_all, "-----End of update_H(pBLM=%" PRI_IDX ", stepM=%i, colIdx=%" PRI_IDX ")-----\n",
				pBLM, stepM, colIdx );
	#endif

	return EXIT_SUCCESS;

} // update_H

/////////////////////////////////////////////////////////////

/*
 * WH(BLN,Mp) = W(BLN,Kp) * H
 * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
 * Waux(BLN,Kp) = WH(BLN,Mp) * H'
 * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
 *
 * WARNING: CUBLAS stream must have been set to streams_NMF[psNMF_N].
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int get_W_BLN( index_t const BLN )
{

	#ifdef NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "\t\t\t\t-----get_W_BLN(BLN=%" PRI_IDX ", rowIdx=%" PRI_IDX ")-----\n", BLN, rowIdx );
	#endif

	// ---------------------------------

	real *const d_Waux = d_Aux;		// Temporary matrix: WH * H'
	real *const d_accum_h = d_accum;	// Accumulator vector: SUM(H).

	size_t const offset_dW = (size_t) (bN + rowIdx) * (size_t) Kp;

	// ----------------------------------

	// WH(BLN,Mp) = W(BLN,Kp) * H
	{

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cublasStatus_t const cublas_status =
		#endif

			// streams_NMF[ psNMF_N ]
			CUBLAS_R_GEMM( cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, BLN, K, d_one, d_H, Kp, &d_W[ offset_dW ], Kp,
					d_zero, d_WH, Mp );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
				print_error( sys_error_shown_by_all, "get_W_BLN(): cublas_gemm( W * H ): %s\n",
						getCublasErrorString( cublas_status ) );
				return EXIT_FAILURE;
			}
		#endif

		#ifdef NMFGPU_DEBUG
		///////////////////////////////
		{
			print_message( dbg_shown_by_all, "--- Resulting WHrow=W*H (BLN=%" PRI_IDX ", Mp=%" PRI_IDX ", rowIdx=%"
					PRI_IDX "): ---\n", BLN, Mp, rowIdx );
			int const status1 = check_cuda_status();
			bool const real_data = true;
			bool const transpose = false;
			struct matrix_tags_t const *restrict mt = NULL;
			int const status2 = show_device_matrix( d_WH, BLN, M, Mp, real_data, transpose, dbg_shown_by_all, mt );
			if ( (status1 != EXIT_SUCCESS) + (status2 != EXIT_SUCCESS) )
				return EXIT_FAILURE;
		}
		/////////////////////////////
		#endif
	}

	// ---------------------------

	// WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

		matrix_div_sub( d_WH, d_Vrow, BLN, Mp,
				#ifdef NMFGPU_DEBUG
					M,
				#endif
				#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
					"WHrow", "Vrow",
				#endif
				true,	// division
				#if NMFGPU_PROFILING_KERNELS
					div_timing,
				#endif
				streams_NMF[ psNMF_N ], event_Vrow );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// ---------------------------

	// Waux(BLN,Kp) = WH(BLN,Mp) * H'
	{

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cublasStatus_t const cublas_status =
		#endif

			CUBLAS_R_GEMM( cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, K, BLN, M, d_one, d_H, Kp, d_WH, Mp, d_zero, d_Waux, Kp );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
				print_error( sys_error_shown_by_all, "get_W_BLN(): cublas_gemm( WH * H' ): %s\n",
						getCublasErrorString( cublas_status ) );
				return EXIT_FAILURE;
			}
		#endif

		#ifdef NMFGPU_DEBUG
		///////////////////////////////
		{
			print_message( dbg_shown_by_all, "--- Resulting d_Waux (BLN=%" PRI_IDX "): ---\n", BLN );
			int const status1 = check_cuda_status();
			bool const real_data = true;
			bool const transpose = false;
			struct matrix_tags_t const *restrict mt = NULL;
			int const status2 = show_device_matrix( d_Waux, BLN, K, Kp, real_data, transpose, dbg_shown_by_all, mt );
			if ( (status1 != EXIT_SUCCESS) + (status2 != EXIT_SUCCESS) )
				return EXIT_FAILURE;
		}
		/////////////////////////////
		#endif
	}

	// ----------------------------


	// W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_H(Kp)
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

		matrix_mul_div( &d_W[ offset_dW ], d_Waux, d_accum_h, BLN, Kp,
				#ifdef NMFGPU_DEBUG
					K, false,	// No matrix transposing.
				#endif
				#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
					"W",
				#endif
				#if NMFGPU_DEBUG
					"Waux", "accum_H",
				#endif
				streams_NMF[ psNMF_N ] );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// ----------------------------

	#ifdef NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "\t\t\t\t-----End of get_W_BLN(BLN=%" PRI_IDX ", rowIdx=%" PRI_IDX ")-----\n",
				BLN, rowIdx );
	#endif

	return EXIT_SUCCESS;

} // getWrow

// ---------------------------------------------

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
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int getW_loop( index_t const num_steps, index_t const BLN )
{

	#ifdef NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "\t\t-----getW_loop(BLN=%" PRI_IDX ", num_steps=%" PRI_IDX ", stepN=%i)-----\n",
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
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

		get_W_BLN( BLN );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// --------------------

	// Remaining (num_steps-1) iterations:

	for ( index_t st = 1 ; st < num_steps ; st++ ) {

		#ifdef NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "\t\t\t-----getW_loop:(step %" PRI_IDX "/%" PRI_IDX ", BLN=%"
					PRI_IDX ", rowIdx=%" PRI_IDX ", stepN=%i)-----\n", st, num_steps, BLN, rowIdx, stepN );
		#endif

		// ----------------

		// Transfers (asynchronously) a new <BLN x M> block from Vrow to d_Vrow

		// First, updates the index of current row.
		rowIdx += (stepN * BLN);

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
		///////////////////////////////
			print_message( dbg_shown_by_all, "--- Vrow processed (BLN=%" PRI_IDX ", new rowIdx=%" PRI_IDX "): ---\n", BLN, rowIdx );
		//////////////////////////////
		#endif
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				int const status =
			#endif

			upload_matrix_partial( Vrow, BLN, Mp, rowIdx, 0,	// Starting column: 0
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
							M,
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 || (! NMFGPU_PROFILING_GLOBAL)
							"Vrow",
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
							"d_Vrow",
						#endif
						Mp, d_Vrow, stream_Vrow, event_Vrow
						#if NMFGPU_PROFILING_TRANSF
							, &upload_Vrow_timing
						#endif
						);

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					return EXIT_FAILURE;
			#endif
		}

		// ----------------

		// Changes main stream.
		psNMF_N += stepN;	// forward or backward

		// Changes CUBLAS stream.
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				cublasStatus_t const cublas_status =
			#endif

			cublasSetStream( cublas_handle, streams_NMF[ psNMF_N ] );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
					print_error( sys_error_shown_by_all, "getW_loop(): cublasSetStream( streams_NMF[" PRI_IDX
							"] ): %s\n", psNMF_N, getCublasErrorString( cublas_status ) );
					return EXIT_FAILURE;
				}
			#endif
		}

		// ----------------

		/* WH(BLN,Mp) = W(BLN,Kp) * H
		 * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
		 * Waux(BLN,Kp) = WH(BLN,Mp) * H'
		 * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
		 */
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				int const status =
			#endif

			get_W_BLN( BLN );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					return EXIT_FAILURE;
			#endif
		}

		// ----------------

		#ifdef NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "\t\t\t-----getW_loop: End of loop (step %" PRI_IDX " of %" PRI_IDX ", BLN=%"
					PRI_IDX ", stepN=%i)-----\n", st, num_steps, BLN, stepN );
		#endif

	} // for ( st=1 ; st<num_steps ; st++ )

	// --------------------------

	#ifdef NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "\t\t-----End of getW_loop(BLN=%" PRI_IDX ", num_steps=%" PRI_IDX ", stepN=%i)-----\n",
				BLN, num_steps, stepN );
	#endif

	return EXIT_SUCCESS;

} // getW_loop

// ---------------------------------------------

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
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int update_W( void )
{

	#ifdef NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "-----update_W(pBLN=%" PRI_IDX ", rowIdx=%" PRI_IDX ", stepN=%i)-----\n",
				pBLN, rowIdx, stepN );
	#endif

	// ----------------------------------

	// Reduces d_H to a row.
	{
		#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

		matrix_to_row( d_H, M, Kp,
				#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG
					K,
				#endif
				#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
					"d_H",
				#endif
				d_Aux, d_accum, stream_H );

		#if NMFGPU_DEBUG_REDUCT || NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// ----------------------------------

	/*
	 * Processes <block_N.num_steps[pBLN]> blocks of size <block_N.BL[pBLN]>
	 */

	// Block configuration.
	index_t num_steps = block_N.num_steps[ pBLN ];	// Number of loops.
	index_t BLN = block_N.BL[ pBLN ];		// Number of rows.

	// --------------------------------

	// Changes CUBLAS stream.
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cublasStatus_t const cublas_status =
		#endif

			cublasSetStream( cublas_handle, streams_NMF[ psNMF_N ] );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
				print_error( sys_error_shown_by_all, "update_W(): cublasSetStream( streams_NMF[" PRI_IDX
						"] ): %s\n", psNMF_N, getCublasErrorString( cublas_status ) );
				return EXIT_FAILURE;
			}
		#endif
	}

	// --------------------------------

	// Delays further operations until d_H is ready.

	for ( index_t i = 0, ps = psNMF_N ; i < num_steps ; i++, ps += stepN ) {

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cudaError_t const cuda_status =
		#endif

			cudaStreamWaitEvent( streams_NMF[ ps ], event_H, 0 );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "Error in update_W( psNMF_N=%" PRI_IDX ", pBLN=%" PRI_IDX
						", stepN=%i): cudaStreamWaitEvent( streams_NMF[%" PRI_IDX "], event_H ): %s\n",
						psNMF_N, pBLN, stepN, ps, cudaGetErrorString(cuda_status) );
				return EXIT_FAILURE;
			}
		#endif
	}

	// --------------------------------

	/* WH(BLN,Mp) = W(BLN,Kp) * H
	 * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
	 * Waux(BLN,Kp) = WH(BLN,Mp) * H'
	 * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
	 */
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

		getW_loop( num_steps, BLN );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// --------------------------------

	// Remaining blocks

	if ( block_N.num_steps[1] ) {  // There are more blocks in dimension "N" to process.

		#ifdef NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "\t-----update_W(pBLN=%" PRI_IDX ",stepN=%i, psNMF_N=%" PRI_IDX
					", rowIdx=%" PRI_IDX "): New block-----\n", pBLN, stepN, psNMF_N, rowIdx );
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
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				cublasStatus_t const cublas_status =
			#endif

				cublasSetStream( cublas_handle, streams_NMF[ psNMF_N ] );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
					print_error( sys_error_shown_by_all, "update_W(new block): cublasSetStream( streams_NMF["
							PRI_IDX "] ): %s\n", psNMF_N, getCublasErrorString( cublas_status ) );
					return EXIT_FAILURE;
				}
			#endif
		}

		// ----------------

		// Transfers (asynchronously) a new <BLN x M> block from Vrow to d_Vrow.

		#ifdef NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "\t-----update_W: New block (pBLN=%" PRI_IDX ", stepN=%i, BLN=%" PRI_IDX
					", num_steps=%" PRI_IDX ", psNMF_N=%" PRI_IDX ", rowIdx=%" PRI_IDX ")-----\n", pBLN, stepN,
					BLN, num_steps, psNMF_N, rowIdx );
		#endif

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
		///////////////////////////////
			print_message( dbg_shown_by_all, "--- Vrow processed (BLN=%" PRI_IDX ", M=%" PRI_IDX ", Mp=%" PRI_IDX
					", rowIdx=%" PRI_IDX "): ---\n", BLN, M, Mp, rowIdx );
		//////////////////////////////
		#endif
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				int const status =
			#endif

			upload_matrix_partial( Vrow, BLN, Mp, rowIdx, 0,	// Starting column: 0
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
							M,
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 || (! NMFGPU_PROFILING_GLOBAL)
							"Vrow",
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
							"d_Vrow",
						#endif
						Mp, d_Vrow, stream_Vrow, event_Vrow
						#if NMFGPU_PROFILING_TRANSF
							, &upload_Vrow_timing
						#endif
						);

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					return EXIT_FAILURE;
			#endif
		}

		// ---------------------------

		// Processes that block.

		/* WH(BLN,Mp) = W(BLN,Kp) * H
		 * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
		 * Waux(BLN,Kp) = WH(BLN,Mp) * H'
		 * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
		 */
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				int const status =
			#endif

			getW_loop( num_steps, BLN );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					return EXIT_FAILURE;
			#endif
		}

		// -------------------------

		// Changes direction (forward -> backward || backward -> forward)
		stepN *= (-1);

		#ifdef NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "\t-----update_W: End of new block (pBLN=%" PRI_IDX ", BLN=%"
					PRI_IDX ", num_steps=%" PRI_IDX "). New StepN=%i -----\n", pBLN, BLN, num_steps, stepN );
		#endif

	} // if ( block_N.num_steps[1] > 0 )

	// -----------------------

	// Records as an event all previous operations on matrix W.
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cudaError_t const cuda_status =
		#endif

			cudaEventRecord( event_W, streams_NMF[ psNMF_N ] );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "Error in update_W( pBLN=%" PRI_IDX ", stepN=%i): "
						"cudaEventRecord( event_W, streams_NMF[%" PRI_IDX "] ): %s\n", pBLN, stepN,
						psNMF_N, cudaGetErrorString(cuda_status) );
				return EXIT_FAILURE;
			}
		#endif
	}

	// -----------------------

	// Delays further operations on "stream_W" until "event_W" completes.
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cudaError_t const cuda_status =
		#endif

			cudaStreamWaitEvent( stream_W, event_W, 0 );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "Error in update_W( pBLN=%" PRI_IDX", stepN=%i): "
						"cudaStreamWaitEvent( stream_W, event_W ): %s\n", pBLN, stepN,
						cudaGetErrorString(cuda_status) );
				return EXIT_FAILURE;
			}
		#endif
	}

	// -----------------------

	#ifdef NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "-----End of update_W(pBLN=%" PRI_IDX ", rowIdx=%" PRI_IDX ", stepN=%i)-----\n",
				pBLN, rowIdx, stepN );
	#endif

	return EXIT_SUCCESS;

} // update_W

////////////////////////////////////////////////

/*
 * Computes classification vector from matrix d_H (full size), and stores it in "ld_classification[]".
 * Then it is downloaded from the GPU and stored in "lh_classification[]".
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int get_classification( index_t *restrict ld_classification, index_t *restrict lh_classification )
{

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "get_classification()...\n" );
	#endif

	// ---------------------------------

	// Stream for this operation.
	cudaStream_t stream_A = stream_H;

	// ---------------------------------

	// Computes the classification vector: Column index of highest values.
	{
		#if NMFGPU_DEBUG
			bool transpose = true;		// Matrix transposing
		#endif

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

		matrix_idx_max( d_H, K, Kp, M,
				#if NMFGPU_DEBUG
					transpose,
				#endif
				#if NMFGPU_DEBUG || (! (NMFGPU_PROFILING_GLOBAL || NMFGPU_PROFILING_KERNELS) )
					"d_H",
				#endif
				#if NMFGPU_DEBUG
					"d_classification",
				#endif
				stream_A, ld_classification );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// ------------------------------

	// Downloads output vector.
	{
		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
			bool const real_data = false;	// Index-type data
			bool const transpose = false;	// No matrix transposing.
		#endif

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

			download_matrix( (void *) lh_classification, Mp, sizeof(index_t), (void const *) ld_classification,
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
						1, M, Mp, real_data, transpose, "classification",
					#endif
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
						|| ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_TRANSF))
						"d_classification",
					#endif
					#if NMFGPU_PROFILING_TRANSF
						&download_classf_timing,
					#endif
					stream_A );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// -----------------------------

	// Waits until complete.
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

			sync_GPU( stream_A );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// -----------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "get_classification()... Done.\n" );
	#endif

	return EXIT_SUCCESS;

} // get_classification

////////////////////////////////////////////////

/*
 * WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
 * dot_VWH(BLN) = SUM((V-WH)**2)
 * dot_V(BLN)	= SUM(V**2)
 *
 * WARNING: CUBLAS stream must have been set to streams_NMF[psNMF_N].
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int get_dot_VWH_BLN( index_t const BLN, real *restrict d_dot_VWH, real *restrict d_dot_V )
{

	#ifdef NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "\t\t\t\t-----get_dot_VWH_BLN(BLN=%" PRI_IDX
				", rowIdx=%" PRI_IDX ")-----\n", BLN, rowIdx );
	#endif

	// ---------------------------------

	// Uses all available streams, starting from streams_NMF[ psNMF_N ].
	index_t stream_idx = psNMF_N;

	size_t const offset_dW = (size_t) (bN + rowIdx) * (size_t) Kp;

	// ----------------------------------

	// WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cublasStatus_t const cublas_status =
		#endif

		// streams_NMF[ psNMF_N ]
		CUBLAS_R_GEMM( cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, BLN, K, d_one, d_H, Kp, &d_W[ offset_dW ], Kp,
				d_zero, d_WH, Mp );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
				print_error( sys_error_shown_by_all, "get_dot_VWH_BLN(): cublas_gemm( W * H ): %s\n",
						getCublasErrorString( cublas_status ) );
				return EXIT_FAILURE;
			}
		#endif

		#ifdef NMFGPU_DEBUG
		///////////////////////////////
		{
			print_message( dbg_shown_by_all, "--- Resulting WHrow=W*H (BLN=%" PRI_IDX ", Mp=%" PRI_IDX
					", rowIdx=%" PRI_IDX "): ---\n", BLN, Mp, rowIdx );
			int const status1 = check_cuda_status();
			bool const real_data = true;
			bool const transpose = false;
			struct matrix_tags_t const *restrict mt = NULL;
			int const status2 = show_device_matrix( d_WH, BLN, M, Mp, real_data, transpose, dbg_shown_by_all, mt );
			if ( (status1 != EXIT_SUCCESS) + (status2 != EXIT_SUCCESS) )
				return EXIT_FAILURE;
		}
		/////////////////////////////
		#endif
	}

	// ---------------------------

	// WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

		matrix_div_sub( d_WH, d_Vrow, BLN, Mp,
				#ifdef NMFGPU_DEBUG
					M,
				#endif
				#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
					"WHrow", "Vrow",
				#endif
				false,	// subtraction
				#if NMFGPU_PROFILING_KERNELS
					sub_timing,
				#endif
				streams_NMF[ psNMF_N ], event_Vrow );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// ---------------------------

	/*
	 * dot_VWH(BLN) = SUM((V-WH)**2)
	 */

	// TODO: Change the loop below for a specific kernel, and just use streams_NMF[ psNMF_N ].

	real *pd_WH = d_WH;
	real *pd_dot_VWH = &d_dot_VWH[ rowIdx ];
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cublasStatus_t const cublas_status =
		#endif

			CUBLAS_R_DOT( cublas_handle, M, pd_WH, 1, pd_WH, 1, pd_dot_VWH );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
					print_error( sys_error_shown_by_all, "get_dot_VWH_BLN(): cublas_dot( V-WH, 0/BLN=%"
							PRI_IDX " ): %s\n", BLN, getCublasErrorString( cublas_status ) );
				return EXIT_FAILURE;
			}
		#endif

		// --------------------

		// Changes CUBLAS stream
		stream_idx = psNMF_N + 1;
		if ( stream_idx >= num_streams_NMF )
			stream_idx = 0;

		pd_WH += Mp;
		pd_dot_VWH++;
	}

	for ( index_t i = 1 ; i < BLN ; i++, pd_WH += Mp, pd_dot_VWH++ ) {

		// Sets new CUBLAS stream.
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				cublasStatus_t const cublas_status =
			#endif

				cublasSetStream( cublas_handle, streams_NMF[ stream_idx ] );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
					print_error( sys_error_shown_by_all, "get_dot_VWH_BLN(): cublasSetStream( V-WH, %"
							PRI_IDX "/BLN=%" PRI_IDX ", streams_NMF[" PRI_IDX "] ): %s\n", i, BLN,
							stream_idx, getCublasErrorString( cublas_status ) );
					return EXIT_FAILURE;
				}
			#endif
		}

		// --------------------

		// Delays the operation until 'event_Vrow' completes.
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				cudaError_t const cuda_status =
			#endif

			cudaStreamWaitEvent( streams_NMF[ stream_idx ], event_Vrow, 0 );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( cuda_status != cudaSuccess ) {
					print_error( sys_error_shown_by_all, "get_dot_VWH_BLN( V-WH, %" PRI_IDX "/BLN=%" PRI_IDX
							", stepN=%i): cudaStreamWaitEvent( streams_NMF[%" PRI_IDX "], event_Vrow ): %s\n",
							pBLM, stepM, colIdx, cudaGetErrorString(cuda_status) );
					return EXIT_FAILURE;
				}
			#endif
		}

		// --------------------

		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				cublasStatus_t const cublas_status =
			#endif

			CUBLAS_R_DOT( cublas_handle, M, pd_WH, 1, pd_WH, 1, pd_dot_VWH );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
					print_error( sys_error_shown_by_all, "get_dot_VWH_BLN(): cublas_dot( V-WH, %" PRI_IDX "/BLN=%"
							PRI_IDX " ): %s\n", i, BLN, getCublasErrorString( cublas_status ) );
					return EXIT_FAILURE;
				}
			#endif
		}

		// --------------------

		// Changes of CUBLAS stream
		stream_idx++;
		if ( stream_idx >= num_streams_NMF )
			stream_idx = 0;

	} // for

	#ifdef NMFGPU_DEBUG
	///////////////////////////////
	{
		print_message( dbg_shown_by_all, "--- Resulting d_dot_VWH[] in get_dot_VWH_BLN(BLN=%" PRI_IDX
				", Mp=%" PRI_IDX ", rowIdx=%" PRI_IDX "): ---\n", BLN, Mp, rowIdx );
		int const status1 = check_cuda_status();
		bool const real_data = true;
		bool const transpose = false;
		struct matrix_tags_t const *restrict mt = NULL;
		int const status2 = show_device_matrix( &d_dot_VWH[ rowIdx ], 1, BLN, BLN, real_data, transpose, dbg_shown_by_all, mt );
		if ( (status1 != EXIT_SUCCESS) + (status2 != EXIT_SUCCESS) )
			return EXIT_FAILURE;
	}
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
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				cublasStatus_t const cublas_status =
			#endif

				cublasSetStream( cublas_handle, streams_NMF[ stream_idx ] );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
					print_error( sys_error_shown_by_all, "get_dot_VWH_BLN(): cublasSetStream( V, %"
							PRI_IDX "/BLN=%" PRI_IDX ", streams_NMF[" PRI_IDX "] ): %s\n", i, BLN,
							stream_idx, getCublasErrorString( cublas_status ) );
					return EXIT_FAILURE;
				}
			#endif
		}

		// --------------------

		// Delays the operation until 'event_Vrow' completes.
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				cudaError_t const cuda_status =
			#endif

				cudaStreamWaitEvent( streams_NMF[ stream_idx ], event_Vrow, 0 );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( cuda_status != cudaSuccess ) {
					print_error( sys_error_shown_by_all, "get_dot_VWH_BLN(): cudaStreamWaitEvent( V, %"
							PRI_IDX "/BLN=%" PRI_IDX ", streams_NMF[%" PRI_IDX "] ): %s\n", i, BLN,
							stream_idx, cudaGetErrorString(cuda_status) );
					return EXIT_FAILURE;
				}
			#endif
		}

		// --------------------

		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				cublasStatus_t const cublas_status =
			#endif

			CUBLAS_R_DOT( cublas_handle, M, pd_Vrow, 1, pd_Vrow, 1, pd_dot_V );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
					print_error( sys_error_shown_by_all, "get_dot_VWH_BLN(): cublas_dot( V, %" PRI_IDX "/BLN=%"
							PRI_IDX " ): %s\n", i, BLN, getCublasErrorString( cublas_status ) );
					return EXIT_FAILURE;
				}
			#endif
		}

		// --------------------

		// Changes of CUBLAS stream
		stream_idx++;
		if ( stream_idx >= num_streams_NMF )
			stream_idx = 0;

	} // for

	#ifdef NMFGPU_DEBUG
	///////////////////////////////
	{
		print_message( dbg_shown_by_all, "--- Resulting d_dot_V[] in get_dot_VWH_BLN(BLN=%"
				PRI_IDX ", Mp=%" PRI_IDX ", rowIdx=%" PRI_IDX "): ---\n", BLN, Mp, rowIdx );
		int const status1 = check_cuda_status();
		bool const real_data = true;
		bool const transpose = false;
		struct matrix_tags_t const *restrict mt = NULL;
		int const status2 = show_device_matrix( &d_dot_V[ rowIdx ], 1, BLN, BLN, real_data, transpose, dbg_shown_by_all, mt );
		if ( (status1 != EXIT_SUCCESS) + (status2 != EXIT_SUCCESS) )
			return EXIT_FAILURE;
	}
	/////////////////////////////
	#endif

	return EXIT_SUCCESS;

} // get_dot_VWH_BLN

// ---------------------------------------------

/*
 * WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
 * dot_VWH(BLN) = SUM((V-WH)**2)
 * dot_V(BLN)	= SUM(V**2)
 *
 * WARNING: CUBLAS stream must have been set to streams_NMF[psNMF_N].
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int get_dot_VWH_loop( index_t num_steps, index_t BLN, real *restrict d_dot_VWH, real *restrict d_dot_V )
{

	#ifdef NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "\t\t-----get_dot_VWH_loop(BLN=%" PRI_IDX ", num_steps=%" PRI_IDX ", stepN=%i)-----\n",
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
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

		get_dot_VWH_BLN( BLN, d_dot_VWH, d_dot_V );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// -----------------------

	// Remaining (num_steps-1) iterations:

	for ( index_t st = 1 ; st < num_steps ; st++ ) {

		#ifdef NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "\t\t\t-----get_dot_VWH_loop:(step %" PRI_IDX "/%" PRI_IDX ", BLN=%" PRI_IDX
					", stepN=%i)-----\n", st, num_steps, BLN, stepN );
		#endif

		// ----------------

		// Transfers (asynchronously) a new <BLN x M> block from Vrow to d_Vrow

		// First, updates the index of current row.
		rowIdx += (stepN * BLN);

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
		///////////////////////////////
			print_message( dbg_shown_by_all, "--- Vrow processed (BLN=%" PRI_IDX ", new rowIdx=%" PRI_IDX
					"): ---\n", BLN, rowIdx );
		//////////////////////////////
		#endif
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				int const status =
			#endif

			upload_matrix_partial( Vrow, BLN, Mp, rowIdx, 0,	// Starting column: 0
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
							M,
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 || (! NMFGPU_PROFILING_GLOBAL)
							"Vrow",
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
							"d_Vrow",
						#endif
						Mp, d_Vrow, stream_Vrow, event_Vrow
						#if NMFGPU_PROFILING_TRANSF
							, &upload_Vrow_timing
						#endif
						);

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					return EXIT_FAILURE;
			#endif
		}

		// ----------------

		// Changes main stream.
		psNMF_N += stepN;	// forward or backward

		// Changes CUBLAS stream.
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				cublasStatus_t const cublas_status =
			#endif

			cublasSetStream( cublas_handle, streams_NMF[ psNMF_N ] );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
					print_error( sys_error_shown_by_all, "get_dot_VWH_loop(): cublasSetStream( streams_NMF["
							PRI_IDX "] ): %s\n", psNMF_N, getCublasErrorString( cublas_status ) );
					return EXIT_FAILURE;
				}
			#endif
		}

		// ----------------

		/* WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
		 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
		 * dot_VWH(BLN) = SUM((V-WH)**2)
		 * dot_V(BLN)	= SUM(V**2)
		 */
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				int const status =
			#endif

			get_dot_VWH_BLN( BLN, d_dot_VWH, d_dot_V );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					return EXIT_FAILURE;
			#endif
		}

		#ifdef NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "\t\t\t-----get_dot_VWH_loop: End of loop (step %" PRI_IDX
					" of %" PRI_IDX ", BLN=%" PRI_IDX ", stepN=%i)-----\n", st, num_steps, BLN, stepN );
		#endif

	} // for ( st=1 ; st<num_steps ; st++ )

	// --------------------------

	#ifdef NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "\t\t-----End of get_dot_VWH_loop(BLN=%" PRI_IDX ", num_steps=%"
				PRI_IDX ", stepN=%i)-----\n", BLN, num_steps, stepN );
	#endif

	return EXIT_SUCCESS;

} // get_dot_VWH_loop

// ---------------------------------------------

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
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int get_dot_VWH_N( real *restrict d_dot_VWH, real *restrict d_dot_V )
{

	#ifdef NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "-----get_dot_VWH_N(pBLN=%" PRI_IDX ", rowIdx=%" PRI_IDX ", stepN=%i)-----\n",
				pBLN, rowIdx, stepN);
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
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cublasStatus_t const cublas_status =
		#endif

			cublasSetStream( cublas_handle, streams_NMF[ psNMF_N ] );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
				print_error( sys_error_shown_by_all, "get_dot_VWH_N(): cublasSetStream( streams_NMF["
						PRI_IDX "] ): %s\n", psNMF_N, getCublasErrorString( cublas_status ) );
				return EXIT_FAILURE;
			}
		#endif
	}

	// --------------------------------

	// Delays further operations until d_H is ready.

	for ( index_t i = 0, ps = psNMF_N ; i < num_steps ; i++, ps += stepN ) {

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cudaError_t const cuda_status =
		#endif

			cudaStreamWaitEvent( streams_NMF[ ps ], event_H, 0 );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "get_dot_VWH_N( psNMF_N=%" PRI_IDX ", pBLN=%" PRI_IDX
						", stepN=%i): cudaStreamWaitEvent( streams_NMF[%" PRI_IDX "], event_H ): %s\n",
						psNMF_N, pBLN, stepN, ps, cudaGetErrorString(cuda_status) );
				return EXIT_FAILURE;
			}
		#endif
	}

	// --------------------------------

	/* WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
	 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
	 * dot_VWH(BLN) = SUM((V-WH)**2)
	 * dot_V(BLN)	= SUM(V**2)
	 */
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

		get_dot_VWH_loop( num_steps, BLN, d_dot_VWH, d_dot_V );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// --------------------------------

	// Remaining blocks

	if ( block_N.num_steps[1] ) {  // There are more blocks in dimension "N" to process.

		#ifdef NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "\t-----get_dot_VWH_N(pBLN=%" PRI_IDX ",stepN=%i,psNMF_N=%"
					PRI_IDX ", rowIdx=%" PRI_IDX "): New block-----\n", pBLN, stepN, psNMF_N, rowIdx );
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
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				cublasStatus_t const cublas_status =
			#endif

				cublasSetStream( cublas_handle, streams_NMF[ psNMF_N ] );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
					print_error( sys_error_shown_by_all, "get_dot_VWH_N(new block): cublasSetStream( streams_NMF["
							PRI_IDX "] ): %s\n", psNMF_N, getCublasErrorString( cublas_status ) );
					return EXIT_FAILURE;
				}
			#endif
		}

		// ----------------

		// Transfers (asynchronously) a new <BLN x M> block from Vrow to d_Vrow.

		#ifdef NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "\t-----get_dot_VWH_N: New block (pBLN=%" PRI_IDX ", stepN=%i, BLN=%"
					PRI_IDX ", num_steps=%" PRI_IDX ", psNMF_N=%" PRI_IDX ", rowIdx=%" PRI_IDX ")-----\n",
					pBLN, stepN, BLN, num_steps, psNMF_N, rowIdx );
		#endif

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF
		///////////////////////////////
			print_message( dbg_shown_by_all, "--- Vrow processed (BLN=%" PRI_IDX ", M=%" PRI_IDX ", Mp=%" PRI_IDX
					", rowIdx=%" PRI_IDX "): ---\n", BLN, M, Mp, rowIdx );
		//////////////////////////////
		#endif

		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				int const status =
			#endif

			upload_matrix_partial( Vrow, BLN, Mp, rowIdx, 0,	// Starting column: 0
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
							M,
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 || (! NMFGPU_PROFILING_GLOBAL)
							"Vrow",
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
							"d_Vrow",
						#endif
						Mp, d_Vrow, stream_Vrow, event_Vrow
						#if NMFGPU_PROFILING_TRANSF
							, &upload_Vrow_timing
						#endif
						);

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					return EXIT_FAILURE;
			#endif
		}

		// ---------------------------

		// Processes that block.

		/* WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
		 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
		 * dot_VWH(BLN) = SUM((V-WH)**2)
		 * dot_V(BLN)	= SUM(V**2)
		 */
		{
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				int const status =
			#endif

			get_dot_VWH_loop( num_steps, BLN, d_dot_VWH, d_dot_V );

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					return EXIT_FAILURE;
			#endif
		}

		// -------------------------

		// Changes direction (forward -> backward || backward -> forward)
		stepN *= (-1);

		#ifdef NMFGPU_VERBOSE_2
			print_message( verb_shown_by_all, "\t-----get_dot_VWH_N: End of new block (pBLN=%" PRI_IDX ", BLN=%" PRI_IDX
					", num_steps=%" PRI_IDX "). New StepN=%i -----\n", pBLN, BLN, num_steps, stepN );
		#endif

	} // if ( block_N.num_steps[1] > 0 )

	// -----------------------

	// Records as an event all previous operations.
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cudaError_t const cuda_status =
		#endif

			cudaEventRecord( event_W, 0 );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "get_dot_VWH_N( pBLN=%" PRI_IDX ", stepN=%i): "
						"cudaEventRecord( event_W ): %s\n", pBLN, stepN, cudaGetErrorString(cuda_status) );
				return EXIT_FAILURE;
			}
		#endif
	}

	// -----------------------

	// Delays further operations on "stream_W" until "event_W" completes.
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cudaError_t const cuda_status =
		#endif

			cudaStreamWaitEvent( stream_W, event_W, 0 );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "get_dot_VWH_N( pBLN=%" PRI_IDX ", stepN=%i): "
						"cudaStreamWaitEvent( stream_W, event_H ): %s\n", pBLN, stepN,
						cudaGetErrorString(cuda_status) );
				return EXIT_FAILURE;
			}
		#endif
	}

	// -----------------------

	#ifdef NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "-----End of get_dot_VWH_N(pBLN=%" PRI_IDX ", rowIdx=%"
				PRI_IDX ", stepN=%i)-----\n", pBLN, rowIdx, stepN);
	#endif

	return EXIT_SUCCESS;

} // get_dot_VWH_N

// ---------------------------------------------

/*
 * WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
 * dot_VWH(BLN) = SUM((V-WH)**2)
 * dot_V(BLN)	= SUM(V**2)
 * d_scalars_VWH[0] = SUM(dot_VWH[...])
 * d_scalars_VWH[1] = SUM(dot_V[...])
 *
 * Computes vectors d_dot_VWH[NpP] and d_dot_V[NpP], and reduces each to a single scalar.
 * Resulting values are returned in d_scalars_VWH[0] and d_scalars_VWH[1], respectively.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int get_dot_VWH( real *restrict d_dot_VWH, real *restrict d_dot_V, real *restrict d_scalars_VWH )
{

	#ifdef NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "-----get_dot_VWH(pBLN=%" PRI_IDX ", rowIdx=%" PRI_IDX ", stepN=%i)-----\n",
				pBLN, rowIdx, stepN);
	#endif

	// -----------------------

	/* WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
	 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
	 * dot_VWH(BLN) = SUM((V-WH)**2)
	 * dot_V(BLN)	= SUM(V**2)
	 */
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

		get_dot_VWH_N( d_dot_VWH, d_dot_V );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// -----------------------

	// d_scalars_VWH[0] = SUM(dot_VWH[...])

	// Since all values are positive, we can use the sums of absolute values.
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cublasStatus_t const cublas_status =
		#endif

			CUBLAS_R_ASUM( cublas_handle, NpP, d_dot_VWH, 1, d_scalars_VWH );	// streams_NMF[ psNMF_N ]

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
				print_error( sys_error_shown_by_all, "get_dot_VWH(): cublas_asum( d_scalars_VWH[0] ): %s\n",
						getCublasErrorString( cublas_status ) );
				return EXIT_FAILURE;
			}
		#endif
	}

	// -----------------------

	// Changes the CUBLAS stream
	{
		index_t stream_idx = psNMF_N + 1;
		if ( stream_idx >= num_streams_NMF )
			stream_idx = 0;

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cublasStatus_t const cublas_status =
		#endif

			cublasSetStream( cublas_handle, streams_NMF[ stream_idx ] );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
				print_error( sys_error_shown_by_all, "get_dot_VWH(): cublasSetStream( streams_NMF[" PRI_IDX
						"] ): %s\n", stream_idx, getCublasErrorString( cublas_status ) );
				return EXIT_FAILURE;
			}
		#endif
	}

	// -----------------------

	// d_scalars_VWH[1] = SUM(dot_VWH[...])
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cublasStatus_t const cublas_status =
		#endif

			// streams_NMF[ (psNMF_N + 1) % num_streams_NMF ]
			CUBLAS_R_ASUM( cublas_handle, NpP, d_dot_V, 1, &d_scalars_VWH[1] );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cublas_status != CUBLAS_STATUS_SUCCESS ) {
				print_error( sys_error_shown_by_all, "update_W(): cublas_asum( d_scalars_VWH[1] ): %s\n",
						getCublasErrorString( cublas_status ) );
				return EXIT_FAILURE;
			}
		#endif
	}

	#ifdef NMFGPU_DEBUG
	///////////////////////////////
	{
		print_message( dbg_shown_by_all, "--- Resulting scalars in get_dot_VWH(): ---\n" );
		int const status1 = check_cuda_status();
		bool const real_data = true;
		bool const transpose = false;
		struct matrix_tags_t const *restrict mt = NULL;
		int const status2 = show_device_matrix( d_scalars_VWH, 1, 2, 2, real_data, transpose, dbg_shown_by_all, mt );
		if ( (status1 != EXIT_SUCCESS) + (status2 != EXIT_SUCCESS) )
			return EXIT_FAILURE;
	}
	/////////////////////////////
	#endif

	// -----------------------

	// Records as an event all previous operations.
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cudaError_t const cuda_status =
		#endif

			cudaEventRecord( event_W, 0 );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "Error recording CUDA event: %s\nError in get_dot_VWH(pBLN=%"
						PRI_IDX ", stepN=%i).\n", cudaGetErrorString(cuda_status), pBLN, stepN );
				return EXIT_FAILURE;
			}
		#endif
	}

	// -----------------------

	// Delays further operations on "stream_W" until "event_W" completes.
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			cudaError_t const cuda_status =
		#endif

			cudaStreamWaitEvent( stream_W, event_W, 0 );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( cuda_status != cudaSuccess ) {
				print_error( sys_error_shown_by_all, "get_dot_VWH( pBLN=%" PRI_IDX ", stepN=%i): "
						"cudaStreamWaitEvent( stream_W, event_W ): %s\n", pBLN, stepN,
						cudaGetErrorString(cuda_status) );
				return EXIT_FAILURE;
			}
		#endif
	}

	// -----------------------

	#ifdef NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "-----End of get_dot_VWH(pBLN=%" PRI_IDX ", rowIdx=%" PRI_IDX ", stepN=%i)-----\n",
				pBLN, rowIdx, stepN);
	#endif

	return EXIT_SUCCESS;

} // get_dot_VWH

// ---------------------------------------------

/*
 * Computes the following dot products:
 *
 *	dot_V	 <-- dot_product( V, V )
 *
 *	dot_VWH  <-- dot_product( V-WH, V-WH )
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int dot_product_VWH( real *restrict dot_V, real *restrict dot_VWH )
{

	#ifdef NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "-----Starting dot_product_VWH( rowIdx=%" PRI_IDX " )-----\n", rowIdx);
	#endif

	/* Sets pointers to use as data matrix. We can use "d_Aux", since we need memory for two
	 * NpP-length vectors, and size(d_Aux) == MAX(N,M) * Kp.
	 */
	real *d_dot_VWH = d_Aux;
	real *d_dot_V = &d_Aux[ NpP ];

	/* Such vectors will be later reduced. Resulting scalars will be stored in d_scalars_VWH[2]
	 * (we can reuse d_accum[Kp] for that). Then, the scalars will be downloaded to h_scalars_VWH[2].
	 *
	 * HACK:
	 *	When host memory is mapped into the address space of the device, d_scalars_VWH[] is actually
	 *	a pointer to a vector stored in host memory. So, we just need to set h_scalars_VWH to point
	 *	to such array.
	 *	On the other hand, if memory was NOT mapped, we need a local vector to store the values
	 *	downloaded from d_scalars_VWH[2].
	 */

	real *d_scalars_VWH = d_accum;		// Results in DEVICE memory.
	real *h_scalars_VWH = h_accum;		// Results in HOST memory (if it WAS mapped).

	real scalars_VWH[ 2 ];			// Results in HOST memory (if it was NOT mapped).
	if ( ! mappedHostMemory )
		h_scalars_VWH = &scalars_VWH[0];

	// ----------------------------------------

	/* WH(BLN,Mp) = W(BLN,Kp) * H(M,Kp)
	 * WH(BLN,Mp) = Vrow(BLN,Mp) - WH(BLN,Mp)
	 * dot_VWH(BLN) = SUM((V-WH)**2)
	 * dot_V(BLN)	= SUM(V**2)
	 * d_scalars_VWH[0] = SUM(dot_VWH[...])
	 * d_scalars_VWH[1] = SUM(dot_V[...])
	 */
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

		get_dot_VWH( d_dot_VWH, d_dot_V, d_scalars_VWH );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// ----------------------------------------

	// Downloads partial results.
	{
		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
			bool const real_data = true;	// Real-type data
			bool const transpose = false;	// No matrix transposing.
		#endif

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

			download_matrix( (void *) h_scalars_VWH, 2, sizeof(real), (void const *) d_scalars_VWH,
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
						1, 2, 2, real_data, transpose, "h_dot_VWH",
					#endif
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
						|| ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_TRANSF))
						"d_scalars_VWH",
					#endif
					#if NMFGPU_PROFILING_TRANSF
						NULL,
					#endif
					stream_W );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// ----------------------------------------

	// Waits for the results...
	{
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			int const status =
		#endif

		sync_GPU( stream_W );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// ----------------------------------------

	#if NMFGPU_DEBUG || NMFGPU_VERBOSE || NMFGPU_VERBOSE_2
	///////////////////////////////
		print_message( dbg_shown_by_all, "dot_product_VWH: dot_VWH=%g dot_V=%g\n", h_scalars_VWH[0], h_scalars_VWH[1] );
	///////////////////////////////
	#endif

	*dot_VWH = h_scalars_VWH[0];
	*dot_V = h_scalars_VWH[1];

	return EXIT_SUCCESS;

} // dot_product_VWH

////////////////////////////////////////////////
////////////////////////////////////////////////
