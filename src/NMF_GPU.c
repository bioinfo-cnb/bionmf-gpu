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
 *
 * NMF_GPU.c
 *	Main program for single-GPU systems.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
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
 *		NMFGPU_CPU_RANDOM: Uses the CPU (host) random generator (not the CURAND library).
 *		NMFGPU_FIXED_INIT: Initializes W and H with "random" values generated from a fixed seed (defined in common.h).
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
 * Matrix tags:
 *
 * Any matrix may include the following "tag" elements:
 *
 *	+ A short description string, referred as "name".
 *	+ A list of column headers.
 *	+ A list of row labels.
 *
 * Each list is stored in a "struct tag_t" structure, which is composed by:
 *	+ All tokens stored as a (large) single string.
 *	+ An array of pointers to such tokens.
 *
 * All three elements (the "name" string, and the two tag_t structures) are
 * then stored in a "struct matrix_tags_t" structure.
 *
 * Both types of structure are defined in "matrix_io_routines.h".
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
 ****************
 *
 * WARNING:
 *	+ This code requires support for ISO-C99 standard. It can be enabled with 'gcc -std=c99'.
 *
 *********************************************************/

#include "NMF_routines.cuh"
#include "matrix/matrix_operations.cuh"
#include "GPU_setup.cuh"
#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
	#include "timing.cuh"
#endif
#include "matrix/matrix_io.h"
#include "matrix/matrix_io_routines.h"
#include "common.h"
#include "real_type.h"
#include "index_type.h"

#include <cuda_runtime_api.h>

#if NMFGPU_PROFILING_GLOBAL
	#include <sys/time.h>
#endif

#include <stdlib.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>	// uintptr_t

////////////////////////////////////////////////
////////////////////////////////////////////////

/* "Private" global variables */

#if NMFGPU_DEBUG || NMFGPU_VERBOSE || NMFGPU_VERBOSE_2
	static bool const dbg_shown_by_all = false;	// Information or error messages on debug.
	static bool const verb_shown_by_all = false;	// Information messages in verbose mode.
#endif
static bool const shown_by_all = false;			// Information messages.
static bool const sys_error_shown_by_all = false;	// System error messages.
static bool const error_shown_by_all = false;		// Error messages on invalid arguments or I/O data.

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Read matrix from file
 *
 * numeric_hdrs, numeric_lbls: Has <filename> numeric column headers / row headers ?
 * isBinary: Is <filename> a binary file?
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int init_V( const char *restrict filename, bool numeric_hdrs, bool numeric_lbls, index_t isBinary, struct matrix_tags_t *restrict mt )
{

	#if NMFGPU_VERBOSE
		print_message( verb_shown_by_all, "Initializing input matrix from file %s...\n", filename );
	#endif

	int status = EXIT_SUCCESS;

	// ----------------------------

	index_t nrows = 0, ncols = 0, pitch = 0;

	real *restrict matrix = NULL;

	status = matrix_load( filename, numeric_hdrs, numeric_lbls, isBinary, &matrix, &nrows, &ncols, &pitch, mt );
	if ( status != EXIT_SUCCESS ) {
		print_error( error_shown_by_all, "Error reading input file.\n" );
		return EXIT_FAILURE;
	}

	// --------------------------------

	// Sets global variables.
	NpP = N = nrows;
	MpP = M = ncols;
	MpPp = Mp = pitch;

	// --------------------------------

	/* Changes matrix to a PINNED HOST memory.
	 * NOTE:
	 *	As of CUDA 6.0, it is possible to register (i.e., to page-lock)
	 *	a memory area returned by malloc(3), but NOT in write-combined
	 *	mode. Therefore, we still allocating a new memory area and
	 *	copying the input matrix.
	 */

	size_t const nitems = (size_t) N * (size_t) Mp;
	bool const wc = true;				// Write-Combined mode
	bool const clear_memory = false;		// Do NOT initialize the allocated memory

	real *restrict const V = (real *restrict) getHostMemory( nitems * sizeof(real), wc, clear_memory );
	if ( ! V ) {
		print_error( error_shown_by_all, "Error allocating HOST memory for input matrix.\n" );
		matrix_clean( matrix, *mt );
		return EXIT_FAILURE;
	}

	// Copies input matrix to the new memory.
	if ( ! memcpy( V, matrix, nitems * sizeof(real) ) )  {
		print_errnum( sys_error_shown_by_all, errno, "Error initializing input matrix on HOST memory.\n" );
		freeHostMemory( V, "V" );
		matrix_clean( matrix, *mt );
		return EXIT_FAILURE;
	}

	// --------------------------------

	free( matrix );

	// In single-process mode, Vrow and Vcol are just aliases.
	Vcol = Vrow = V;

	return EXIT_SUCCESS;

} // init_V

////////////////////////////////////////////////

/*
 * NMF algorithm
 *
 * Return EXIT_SUCCESS or EXIT_FAILURE.
 */
static int nmf( index_t nIters, index_t niter_test_conv, index_t stop_threshold )
{

	int status = EXIT_SUCCESS;

	#if NMFGPU_PROFILING_GLOBAL
		// GPU time
		struct timeval gpu_tv;
		gettimeofday( &gpu_tv, NULL );
	#endif

	// ----------------------------

	// Initializes matrices W and H with random values.
	{

		index_t const seed = get_seed();

		status = init_random( seed );
		if ( status != EXIT_SUCCESS )
			return EXIT_FAILURE;

		// H
		set_random_values( H, d_H, M, K, Kp,
					#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2
						true, "H", "d_H",	// Matrix transposing
					#endif
					#if NMFGPU_CPU_RANDOM && NMFGPU_PROFILING_TRANSF
						&upload_H_timing,
					#endif
					streams_NMF[ psNMF_N ], NULL );

		// W
		set_random_values( W, d_W, N, K, Kp,
					#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2
						false, "W", "d_W",	// NO matrix transposing
					#endif
					#if NMFGPU_CPU_RANDOM && NMFGPU_PROFILING_TRANSF
						&upload_W_timing,
					#endif
					stream_W, &event_W );

		destroy_random();
	}

	// ----------------------------

	// Uploads matrix V
	{
		// Block configuration.
		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
			index_t const BLM = block_M.BL[ pBLM ];		// Number of columns.
		#endif
		index_t const BLMp = block_M.BLp[ pBLM ];		// Number of columns (with padding).
		index_t const BLN  = block_N.BL[ pBLN ];		// Number of rows.

		// d_Vcol
		if ( d_Vcol != d_Vrow )
			upload_matrix_partial( Vcol, N, MpPp, 0, colIdx,	// Starting row: 0
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
							BLM, "Vcol", "d_Vcol",
						#endif
						BLMp, d_Vcol, stream_Vcol, event_Vcol
						#if NMFGPU_PROFILING_TRANSF
							, &upload_Vcol_timing
						#endif
						);

		// d_Vrow
		upload_matrix_partial( Vrow, BLN, Mp, rowIdx, 0,	// Starting column: 0
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
						M, "Vrow", "d_Vrow",
					#endif
					Mp, d_Vrow, stream_Vrow, event_Vrow
					#if NMFGPU_PROFILING_TRANSF
						, &upload_Vrow_timing
					#endif
					);
	}

	// ----------------------------

	// Number of iterations

	div_t const niter_div = div( nIters , niter_test_conv );
	index_t const niter_conv = (index_t) niter_div.quot;		// Number of times to perform test of convergence.
	index_t const niter_rem = (index_t) niter_div.rem;		// Remaining iterations.
	bool converged = false;

	#if NMFGPU_VERBOSE
		print_message( verb_shown_by_all, "\nniter_test_conv=%" PRI_IDX ", niter_conv=%" PRI_IDX ", niter_rem=%" PRI_IDX ".\n",
				niter_test_conv, niter_div.quot, niter_div.rem );
	#endif


	print_message( shown_by_all, "Starting NMF( K=%"PRI_IDX" )...\n", K );
	flush_output( false );

	// ------------------------

	index_t inc = 0;	// Number of it. w/o changes.

	/* Performs all <nIters> iterations in <niter_conv> groups
	 * of <niter_test_conv> iterations each.
	 */

	index_t iter = 0;	// Required outside this loop.

	for ( ; iter<niter_conv ; iter++ ) {

		// Runs NMF for niter_test_conv iterations...
		for ( index_t i=0 ; i<niter_test_conv ; i++ ) {

			#if NMFGPU_DEBUG
			///////////////////////////////
				print_message( verb_shown_by_all, "\n============ iter=%" PRI_IDX ", Loop %" PRI_IDX
						" (niter_test_conv): ============\n------------ Matrix H: ------------\n", iter,i);
			/////////////////////////////
			#endif

			/*
			 * WH(N,BLMp) = W * pH(BLM,Kp)
			 * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
			 * Haux(BLM,Kp) = W' * WH(N,BLMp)
			 * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W
			 */
			update_H();

			#ifdef NMFGPU_DEBUG
			///////////////////////////////
				print_message( verb_shown_by_all, "------------ iter=%i, loop %i (niter_test_conv) Matrix W: ------------\n",
						iter,i);
			/////////////////////////////
			#endif

			/*
			 * WH(BLN,Mp) = W(BLN,Kp) * H
			 * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
			 * Waux(BLN,Kp) = WH(BLN,Mp) * H'
			 * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
			 */
			update_W();

		} // for niter_test_conv times.

		// -------------------------------------

		// Adjusts matrices W and H.

		matrix_adjust( d_H, block_M.BL[ pBLM ], Kp,
				#if NMFGPU_DEBUG
					K, true, "d_H",		// Matrix transposing
				#endif
				stream_H, NULL );

		matrix_adjust( d_W, block_N.BL[ pBLN ], Kp,
				#if NMFGPU_DEBUG
					K, false, "d_W",	// No matrix transposing
				#endif
				stream_W, &event_W );

		// -------------------------------------

		// Test of convergence

		// Computes classification vector
		get_classification( d_classification, classification );

		// Computes differences
		size_t const diff = get_difference( classification, last_classification, M );

			#ifdef NMFGPU_DEBUG
			///////////////////////////////
				print_message( dbg_shown_by_all, "\nReturned difference between classification vectors: %zu\n", diff );
			/////////////////////////////
			#endif

		// -------------------------------------

		// Saves the new classification.
		{
			// It just swaps the pointers.
			index_t *const h_tmp = classification;
			classification = last_classification;
			last_classification = h_tmp;

			/* If host memory was mapped into the address space of the device,
			 * pointers in device memory must also be swapped.
			 */
			if ( mappedHostMemory ) {
				index_t *const d_tmp = d_classification;
				d_classification = d_last_classification;
				d_last_classification = d_tmp;
			}
		}

		// Stops if Connectivity matrix (actually, the classification vector) has not changed over last <stop_threshold> iterations.

		if ( diff )
			inc = 0;	// Restarts counter.

		// Increments the counter.
		else if ( inc < stop_threshold )
			inc++;

		#if ! NMFGPU_DEBUG
		// Algorithm has converged.
		else {
			iter++; // Adds to counter the last <niter_test_conv> iterations performed
			converged = true;
			break;
		}
		#endif

	} // for  ( nIters / niter_test_conv ) times

	// ---------------------------------------------------------

	// Remaining iterations (if NMF has not converged yet).

	if ( (!converged) * niter_rem ) { // (converged == false) && (niter_rem > 0)

		#if NMFGPU_VERBOSE
			print_message( verb_shown_by_all, "\nPerforming remaining iterations (%" PRI_IDX ")...\n", niter_rem);
		#endif

		// Runs NMF for niter_rem iterations...
		for ( index_t i=0 ; i<niter_rem ; i++ ) {

			#ifdef NMFGPU_DEBUG
			///////////////////////////////
				print_message( verb_shown_by_all, "\n============ Loop %" PRI_IDX " (remaining) ============\n"
						"------------ Matrix H: ------------\n",i);
			/////////////////////////////
			#endif

			/*
			 * WH(N,BLMp) = W * pH(BLM,Kp)
			 * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
			 * Haux(BLM,Kp) = W' * WH(N,BLMp)
			 * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W
			 */
			update_H();

			#ifdef NMFGPU_DEBUG
			///////////////////////////////
				print_message(verb_shown_by_all, "\n------------ Matrix W (loop=%" PRI_IDX ",remaining): ------------\n",i);
			/////////////////////////////
			#endif

			/*
			 * WH(BLN,Mp) = W(BLN,Kp) * H
			 * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
			 * Waux(BLN,Kp) = WH(BLN,Mp) * H'
			 * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
			 */
			update_W();

		} // for niter_rem times.

	} // if has not yet converged.

	#if NMFGPU_VERBOSE
		print_message( verb_shown_by_all, "Done.\n" );
	#endif

	// --------------------------------

	// Number of iterations performed.

	index_t num_iter_performed = nIters;
	if ( converged ) {
		num_iter_performed = iter * niter_test_conv;
		print_message( shown_by_all, "NMF: Algorithm converged in %" PRI_IDX " iterations.\n", num_iter_performed );
	}
	else
		print_message( shown_by_all, "NMF: %" PRI_IDX " iterations performed.\n", num_iter_performed );

	// --------------------------------

	// Downloads output matrices
	{
		bool const real_data = true;
		size_t const data_size = sizeof(real);

		// d_H
		download_matrix( H, M, Kp, data_size, d_H,
				#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
					K, real_data, true, "H", "d_H",		// Matrix transposing
				#endif
				#if NMFGPU_PROFILING_TRANSF
					&download_H_timing,
				#endif
				stream_H );

		// d_W
		download_matrix( W, N, Kp, data_size, d_W,
				#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
					K, real_data, false, "W", "d_W",	// NO matrix transposing
				#endif
				#if NMFGPU_PROFILING_TRANSF
					&download_W_timing,
				#endif
				stream_W );
	}
	// --------------------------------

	/* Checks results:
	 *
	 * Computes the "distance" between V and W*H as follow:
	 *
	 *	distance = norm( V - (W*H) ) / norm( V ),
	 * where
	 *	norm( X ) = sqrt( dot_X )
	 *	dot_V	 <-- dot_product( V, V )
	 *	dot_VWH  <-- dot_product( V-(W*H), V-(W*H) )
	 */

	real dot_V = REAL_C( 0.0 ), dot_VWH = REAL_C( 0.0 );

	status = dot_product_VWH( &dot_V, &dot_VWH );
	if ( status != EXIT_SUCCESS )
		return EXIT_FAILURE;

	#if NMFGPU_DEBUG || NMFGPU_VERBOSE
	///////////////////////////////
		print_message( dbg_shown_by_all, "\tnorm(V)=%g, norm(V-WH)=%g\n", SQRTR( dot_V ), SQRTR( dot_VWH ) );
	///////////////////////////////
	#endif

	print_message( shown_by_all, "\nDistance between V and W*H: %g\n", SQRTR( dot_VWH ) / SQRTR( dot_V ) );

	// --------------------------------

	status = check_cuda_status();

	#if NMFGPU_PROFILING_GLOBAL
	// GPU time
	{
		struct timeval gpu_ftv, gpu_etv;
		gettimeofday( &gpu_ftv, NULL );
		timersub( &gpu_ftv, &gpu_tv, &gpu_etv );	// etv = ftv - tv
		float const total_gpu_time = gpu_etv.tv_sec + ( gpu_etv.tv_usec * 1e-06f );
		print_message( shown_by_all, "\nGPU + classification + check_result time: %g seconds.\n", total_gpu_time );
	}
	#endif

	return status;

} // nmf

////////////////////////////////////////////////

/*
 * Writes output matrices
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int write_matrices( const char *restrict filename, index_t save_bin, struct matrix_tags_t mt )
{

	int status = EXIT_SUCCESS;

	// There are column headers.
	bool const hasheaders = (uintptr_t) mt.name + (uintptr_t) mt.headers.tokens;	// (mt.name != NULL) || (mt.headers.tokens != NULL)

	// There are matrix tag elements.
	bool const is_tagged = (uintptr_t) hasheaders + (uintptr_t) mt.labels.tokens;

	struct matrix_tags_t mt_H, mt_W;	// Labels for output matrices.

	bool transpose = false;
	bool verbose = false;

	struct tag_t tag_factors = new_empty_tag();

	// -----------------------------

	// Initializes labels for output matrices.

	#if NMFGPU_VERBOSE || NMFGPU_DEBUG_NMF
		print_message( verb_shown_by_all, "\tInitializing labels for output matrices...\n");
	#endif

	if ( is_tagged && ( generate_tag( "Factor_", NULL, 0, K, &tag_factors ) != EXIT_SUCCESS ) ) {
		print_error( error_shown_by_all, "Error initializing temporary data (tag_factors).\n");
		return EXIT_FAILURE;
	}

	mt_H = new_matrix_tags( (char *restrict)mt.name, tag_factors, mt.headers );
	mt_W = new_matrix_tags( (char *restrict)mt.name, tag_factors, mt.labels  );

	// -----------------------------

	// Output filenames

	char *restrict const filename_out = (char *restrict) malloc( (strlen(filename) + 6)*sizeof(char) );
	if ( ! filename_out ) {
		print_errnum( sys_error_shown_by_all, errno, "Error allocating memory for output filename" );
		clean_tag( tag_factors );
		return EXIT_FAILURE;
	}

	// -----------------------------

	// Matrix W

	errno = 0;
	if ( sprintf( filename_out, "%s_W.txt", filename ) <= 0 ) {
		print_errnum( sys_error_shown_by_all, errno, "Error setting output filename for matrix W" );
		free( filename_out );
		clean_tag( tag_factors );
		return EXIT_FAILURE;
	}

	transpose = false;
	verbose = true;
	status = matrix_save( filename_out, save_bin, W, N, K, Kp, transpose, &mt_W, verbose );
	if ( status != EXIT_SUCCESS ) {
		print_error( error_shown_by_all, "Error writing matrix W.\n" );
		free( filename_out );
		clean_tag( tag_factors );
		return EXIT_FAILURE;
	}

	// -----------------------------

	// Matrix H

	errno = 0;
	if ( sprintf( filename_out, "%s_H.txt", filename ) <= 0 ) {
		print_errnum( sys_error_shown_by_all, errno, "Error setting output filename for matrix H" );
		free( filename_out );
		clean_tag( tag_factors );
		return EXIT_FAILURE;
	}

	transpose = true;
	verbose = false;
	status = matrix_save( filename_out, save_bin, H, M, K, Kp, transpose, &mt_H, verbose );
	if ( status != EXIT_SUCCESS )
		print_error( error_shown_by_all, "Error writing matrix H.\n" );

	// -----------------------------

	free( filename_out );

	clean_tag( tag_factors );

	return status;

} // write_matrices

////////////////////////////////////////////////
////////////////////////////////////////////////

int main( int argc, char const *restrict *restrict argv )
{

	#if NMFGPU_PROFILING_GLOBAL
		// Elapsed time
		struct timeval t_tv;
	#endif

	process_id = 0;		// Global variables.
	num_processes = 1;

	// Default limits for matrix dimensions. They may be later adjusted, at device initialization.
	set_default_matrix_limits();

	int status = EXIT_SUCCESS;

	// ----------------------------------------

	#if NMFGPU_DEBUG || NMFGPU_DEBUG_READ_MATRIX || NMFGPU_DEBUG_READ_MATRIX2 \
		|| NMFGPU_DEBUG_READ_FILE || NMFGPU_DEBUG_READ_FILE2 || NMFGPU_VERBOSE_2

		// Permanently flushes the output stream in order to prevent losing messages if the program crashes.
		flush_output( true );

	#endif

	// ----------------------------------------

	// Reads all parameters and performs error-checking.

	bool help = false;			// Help message requested

	struct input_arguments arguments;	// Input arguments

	// Checks all arguments (shows error messages).
	if ( check_arguments( argc, argv, &help, &arguments ) != EXIT_SUCCESS )
		return EXIT_FAILURE;

	// If help was requested, just prints a help message and returns.
	if ( help )
		return print_nmf_gpu_help( *argv );


	char const *restrict const filename = arguments.filename;	// Input filename
	bool const numeric_hdrs = arguments.numeric_hdrs;		// Has numeric columns headers.
	bool const numeric_lbls = arguments.numeric_lbls;		// Has numeric row labels.
	index_t const is_bin = arguments.is_bin;			// Input  file is binary (native or non-native format).
	index_t const save_bin = arguments.save_bin;			// Output file is binary (native or non-native format).
	K = arguments.k;						// Factorization rank.
	Kp = arguments.kp;						// Padded factorization rank.
	index_t const nIters = arguments.nIters;			// Maximum number of iterations per run.
	index_t const niter_test_conv = arguments.niter_test_conv;	// Number of iterations before testing convergence.
	index_t const stop_threshold = arguments.stop_threshold;	// Stopping criterion.
	index_t const gpu_device = arguments.gpu_device;		// Device ID.

	// Compute classification vector?
	bool const do_classf = ( nIters >= niter_test_conv );

	// ----------------------------------------

	print_message( shown_by_all, "\t<<< bioNMF-GPU: Non-negative Matrix Factorization on GPU >>>\n"
					"\t\t\t\tSingle-GPU version\n" );

	#if NMFGPU_PROFILING_GLOBAL
		// Total elapsed time
		gettimeofday( &t_tv, NULL );
	#endif

	// ----------------------------------------

	/* Initializes the GPU device.
	 *
	 * In addition:
	 *	- Updates memory_alignment according to the selected GPU device.
	 *	- Updates Kp (i.e., the padded factorization rank).
	 *	- Updates the limits of matrix dimensions.
	 */
	size_t const mem_size = initialize_GPU( gpu_device, K );
	if ( ! mem_size )
		return EXIT_FAILURE;

	// ----------------------------------------

	// Reads input matrix

	struct matrix_tags_t mt = new_empty_matrix_tags();

	status = init_V( filename, numeric_hdrs, numeric_lbls, is_bin, &mt );
	if ( status != EXIT_SUCCESS ) {
		shutdown_GPU();
		return EXIT_FAILURE;
	}

	// ----------------------------------------

	// Fails if the factorization rank is too large.
	if ( K > MIN( N, M ) ) {
		print_error( error_shown_by_all, "\nError: invalid factorization rank: K=%" PRI_IDX ".\nIt cannot be greater "
				"than any of matrix dimensions.\n", K );
		freeHostMemory( Vrow, "V" ); clean_matrix_tags( mt );
		shutdown_GPU();
		return EXIT_FAILURE;
	}

	// ----------------------------------------

	// Setups the GPU device

	status = setup_GPU( mem_size, do_classf );
	if ( status != EXIT_SUCCESS ) {
		freeHostMemory( Vrow, "V" ); clean_matrix_tags( mt );
		shutdown_GPU();
		return EXIT_FAILURE;
	}

	// ----------------------------------------

	// Allocates HOST memory for matrices W and H
	{
		size_t nitems = (size_t) N * (size_t) Kp;
		W = (real *restrict) getHostMemory( nitems * sizeof(real), false, false );
		if ( ! W ) {
			print_error( sys_error_shown_by_all, "Error allocating memory for HOST matrix W (N=%" PRI_IDX
					", Kp=%" PRI_IDX ").\n", N, Kp );
			freeHostMemory( Vrow, "V" ); clean_matrix_tags( mt );
			finalize_GPU_device();
			return EXIT_FAILURE;
		}

		nitems = (size_t) M * (size_t) Kp;
		H = (real *restrict) getHostMemory( nitems * sizeof(real), false, false );
		if ( ! H ) {
			print_error( sys_error_shown_by_all, "Error allocating memory for HOST matrix H (M=%" PRI_IDX
					", Kp=%" PRI_IDX ").\n", M, Kp );
			freeHostMemory( W, "W" ); freeHostMemory( Vrow, "V" ); clean_matrix_tags( mt );
			finalize_GPU_device();
			return EXIT_FAILURE;
		}
	}

	// ----------------------------------------

	// Allocates HOST memory for classification vectors.

	if ( do_classf ) {
		classification = (index_t *restrict) getHostMemory( Mp * sizeof(index_t), false, false );
		if ( ! classification ) {
			print_error( sys_error_shown_by_all, "Error allocating memory for HOST classification vector (M=%" PRI_IDX ", Mp=%"
					PRI_IDX ").\n", M, Mp );
			freeHostMemory( H, "H" ); freeHostMemory( W, "W" ); freeHostMemory( Vrow, "V" ); clean_matrix_tags( mt );
			finalize_GPU_device();
			return EXIT_FAILURE;
		}

		last_classification = (index_t *restrict) getHostMemory( Mp * sizeof(index_t), false, true );	// Initializes with zeros
		if ( ! last_classification ) {
			print_error( sys_error_shown_by_all, "Error allocating memory for HOST classification vector (last, M=%" PRI_IDX
					", Mp=%" PRI_IDX ").\n", M, Mp );
			freeHostMemory(classification, "classification vector"); freeHostMemory( H, "H" ); freeHostMemory( W, "W" );
			freeHostMemory( Vrow, "V" );
			clean_matrix_tags( mt );
			finalize_GPU_device();
			return EXIT_FAILURE;
		}

	} // do_classf

	// ----------------------------------------

	// Executes the NMF Algorithm

	status = nmf( nIters, niter_test_conv, stop_threshold );
	if ( status != EXIT_SUCCESS ) {
		if ( do_classf ) {
			freeHostMemory( last_classification, "previous classification vector" );
			freeHostMemory( classification, "classification vector" );
		}
		freeHostMemory( H, "H" ); freeHostMemory( W, "W" ); freeHostMemory( Vrow, "V" ); clean_matrix_tags( mt );
		finalize_GPU_device();
		return EXIT_FAILURE;
	}

	if ( do_classf ) {
		freeHostMemory( last_classification, "previous classification vector" );
		freeHostMemory( classification, "classification vector" );
	}

	// ----------------------------------------

	// Writes output matrices.

	status = write_matrices( filename, save_bin, mt );
	if ( status != EXIT_SUCCESS ) {
		freeHostMemory( H, "H" ); freeHostMemory( W, "W" ); freeHostMemory( Vrow, "V" ); clean_matrix_tags( mt );
		finalize_GPU_device();
		return EXIT_FAILURE;
	}

	// ----------------------------------------

	// Show elapsed time:

	#if NMFGPU_PROFILING_GLOBAL
		// Total elapsed time
		{
			struct timeval t_ftv, t_etv;
			gettimeofday( &t_ftv, NULL );
			timersub( &t_ftv, &t_tv, &t_etv );	// etv = ftv - tv
			float const total_nmf_time = t_etv.tv_sec + ( t_etv.tv_usec * 1e-06f );
			print_message( shown_by_all, "\nTotal elapsed time: %g seconds.\n", total_nmf_time );
		}
	#endif

	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
		print_message( shown_by_all, "\nTime elapsed on GPU operations:\n" );

		show_kernel_times();

		show_transfer_times();
	#endif

	// ----------------------------------------

	freeHostMemory( H, "H" ); freeHostMemory( W, "W" ); freeHostMemory( Vrow, "V" ); clean_matrix_tags( mt );

	if ( finalize_GPU_device() != EXIT_SUCCESS )
		return EXIT_FAILURE;

	// ----------------------------------------

	print_message( shown_by_all, "Done.\n" );

	return EXIT_SUCCESS;

} // main

////////////////////////////////////////////////
////////////////////////////////////////////////
