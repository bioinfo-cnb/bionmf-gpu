/******************************************************************************
 * Copyright (C) 2011-2013:
 *	Edgardo Mejia-Roa(*), Carlos Garcia, Jose Ignacio Gomez, Manuel Prieto,
 *	Francisco Tirado and Alberto Pascual-Montano(**).
 *
 *	(*)  ArTeCS Group, Complutense University of Madrid (UCM), Spain.
 *	(**) Functional Bioinformatics Group, Biocomputing Unit,
 *		National Center for Biotechnology-CSIC, Madrid, Spain.
 *
 *	E-mail for E. Mejia-Roa: <edgardomejia@fis.ucm.es>
 *	E-mail for A. Pascual-Montano: <pascual@cnb.csic.es>
 *
 *
 * This file is part of BioNMF-GPU.
 *
 * BioNMF-GPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * BioNMF-GPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with BioNMF-GPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 *****************************************************************************/

/**********************************************************
 * Data matrices:
 *	V (N rows, M columns): Input matrix
 *	W (N,K): Output matrix
 *	H (K,M): Output matrix
 * where,
 *	K:  Factorization Rank
 *	V ~ W*H
 *
 * NOTE: In order to improve performance:
 *	- Matrix H is stored in memory as COLUMN-major (i.e., it is transposed).
 *
 *	- All matrices have padded dimensions with useless data to meet memory
 *	  alignment requirements. Such dimensions are denoted with a suffixed
 *	  'p' character, e.g., 'Mp' (i.e., M + padding) or 'Kp'
 *	  (i.e., factorization_rank + padding).
 *
 *	- Padded dimensions are a multiple of 'memory_alignment' (a global
 *	  variable equal to warpSize or warpSize/2).
 *
 ***************
 * Multi-GPU version:
 *
 * When the input matrix V is distributed among multiple MPI processes, each of them has in memory
 * the following sets of rows and columns:
 *	Vrow[ 1..NnP ][ 1..M ] <-- V[ bN..(bN+NnP) ][ 1..M ]	(i.e., NnP rows, starting from bN)
 *	(it is sometimes denoted as 'Vfil' instead).
 *
 *	Vcol[ 1..N ][ 1..MnP ] <-- V[ 1..N ][ bM..(bM+MnP) ]	(i.e., MnP columns, starting from bM)
 *
 * where,
 *	bN == (my_rank*N)/nProcss		(Starting row)
 *	bM == (my_rank*M)/nProcss		(Starting column)
 *
 *	NnP == ((my_rank+1)*N/nProcss) - bN	(Number of rows for this process).
 *	MnP == ((my_rank+1)*M/nProcss) - bM	(Number of columns for this process).
 *
 *	my_rank: This MPI process ID.
 *	nProcss: total number of MPI processes.
 *
 *
 * Note that each MPI process has a private (full) copy of matrices W and H, which must be synchronized
 * (with collective communication such as all_gather()) after being updated.
 *
 * bNv, NnPv: Array with all bN and NnP*Kp values for all processes. Used only if (N % nProcss != 0)
 * bMv, MnPv: Array with all bM and MnP*Kp values for all processes. Used only if (M % nProcss != 0)<W
 *
 *
 ****************
 * Large input matrix (blockwise processing):
 *
 * If (the portion of) the input matrix assigned to this MPI process is too large for the GPU memory,
 * it must be blockwise processed as follow:
 *	d_Vrow[1..BLN][1..Mp] <-- Vrow[ offset..(offset + BLN) ][1..Mp]			(i.e., BLN <= NnP rows)
 *	d_Vcol[1..N][1..BLMp] <-- Vcol[1..N][ offset_Vcol..(offset_Vcol + BLMp) ]	(i.e., BLM <= MnP columns)
 *
 * Note that padded dimensions are denoted with the suffix 'p' (e.g., Mp, BLMp, etc).
 *
 * In any case, matrices W and H are fully loaded into the GPU memory.
 *
 * Information for blockwise processing (i.e., BLN, offsets, etc.) is stored in two
 * 'block_t' structures (one for each dimension): 'block_N' and 'block_M'
 *
 *********************************************************/

#ifndef _NMF_MPI_CUDA_H_

#define _NMF_MPI_CUDA_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <math.h>	/* sqrt() */

// Sched affinity
#ifndef _GNU_SOURCE
	#define GNOTDEF
	#define _GNU_SOURCE
#endif
#ifndef __USE_GNU
	#define UNOTDEF
	#define __USE_GNU
#endif
	#include <sched.h>
#ifdef UNOTDEF
	#undef __USE_GNU
	#undef UNOTDEF
#endif
#ifdef GNOTDEF
	#undef _GNU_SOURCE
	#undef GNOTDEF
#endif

#include <mpi.h>
// 	typedef int MPI_Comm;
// 	typedef int MPI_Status;
// 	#define MPI_Init(...); {{ }}
// 	#define MPI_Comm_rank(...); {{ }}
// 	#define MPI_Comm_size(...); {{ }}
// 	#define MPI_Wtime() 0
// 	#define MPI_Finalize() {{ }}
// 	#define MPI_COMM_WORLD 0
// 	#define MPI_IN_PLACE 0
// 	#define MPI_FLOAT float
// 	#define MPI_SUM 0



// For random init
#ifndef _MY_INIT
	#include <time.h>
#endif

// --------------------------------------------
// --------------------------------------------

/* Includes */

#include "common.h"
#include "kernels_NMF.h"

#ifdef GPU
	#include "rutinesGPU.h"
#endif

// --------------------------------------------
// --------------------------------------------

/* Constants */

#ifdef REAL_FLOAT
	#define MPI_TREAL MPI_FLOAT
#else
	#define MPI_TREAL MPI_DOUBLE
#endif

// TODO: AL MAIN_MPI
#if NMFGPU_PROFILING_COMM
	/* Elapsed time already measured in seconds (NOT in ms):
	 *	Throughput (GB/sec) = ( (td.nitems * sizeof_data) bytes / (2**30 bytes/GB) )  / (td.time sec)
	 * Note that sizeof_data / 2**30 is calculated at compile time.
	 */
	#define PRINT_TIME_SEC(op, td, sizeof_data) { \
		printf( "%s: %.3Lg sec (%" PRIuMAX " time(s), avg: %.6g ms), %.3g GB/s\n", \
			(op), (td).time, (td).counter, ((td).time/(td).counter), \
			( ( (td).nitems * ((double)(sizeof_data)/(1<<30)) ) / (td).time ) ); \
	}
#endif

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus /* If this is a C++ compiler, use C linkage */
extern "C" {
#endif

// -----------------------------------------

int *get_memory1D_int( int nx );

int destroy_memory1D_int( int *restrict buffer );

real *get_memory1D_V( int nx, int is_V );

real *get_memory1D( int nx );

real **get_memory2D_V( int nx, int ny, int is_V );

real **get_memory2D( int nx, int ny );

real **get_memory2D_renew( int nx, int ny, real *restrict mem );

int destroy_memory1D( real *restrict buffer );

int destroy_memory2D( real *restrict *buffer );

void dimension_per_CPU( int N, int M, int my_rank, int nProcs, int *restrict NnP, int *restrict MnP, int *restrict bN, int *restrict bM);

real **read_matrix( FILE *restrict fp, int M, int bN, int bM, int NnP, int MnP, int MnPp, int my_rank );

int init_V( const char *restrict filename, real ***Vfil, real ***Vcol, int *restrict N, int *restrict M,
		int *restrict NnP, int *restrict MnP, int *restrict bN, int *restrict bM, int nProcs, int my_rank);

void init_matrix( real *restrict matrix, int nrows, int ncols, int padding, real min_value );

void show_matrix( real *restrict *matrix, int nrows, int ncols, char row_major );

//////////////////////////////////////////////////

int init_classf_data( int Mp, int my_rank, int *restrict *classification, int *restrict *last_classification );

//////////////////////////////////////////////////

/*
 * Allocates memory for vectors with number of matrix entries, and starting offset,
 * to be processed by each task.
 *
 * R can be set to N or M.
 *
 * bRv: offset to starting row for each process (ie. all bR*C values).
 * RnPv: Number of elements for each process (ie. all RnP*C values).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int allocate_dim_vectors( int nProcs, int *restrict *RnPv, int *restrict *bRv );

// -------------------------------------------

/*
 * Initializes vectors with number of matrix entries, and starting offset, to be processed by each task.
 * R can be set to N or M.
 * C can be set to K (or Kp if using padding).
 *
 * bRv: offset to starting row for each process (ie. all bR*C values).
 * RnPv: Number of elements for each process (ie. all RnP*C values).
 */
void init_dim_vectors( int R, int C, int nProcs, int *restrict RnPv, int *restrict bRv );

// -------------------------------------------

/*
 * All-to-All synchronization.
 * Gathers data from all tasks and distributes the combined data to all tasks.
 *	p_matrix: Starting address of data to be sent by slaves.
 *	size: Size of data to be sent by this process.
 */
void sync_matrix( int my_rank, real *restrict *matrix, int bR, int size, int *restrict RnPv, int *restrict bRv,
		timing_data_t *restrict comm_time, cudaStream_t *restrict pstream, cudaEvent_t *restrict events );

// -------------------------------------------

/*
 * Gathers data from slave processes.
 * All slave processes send their portion of data to master.
 *	p_matrix: Starting address of data to be sent by slaves.
 *	size: Size of data to be sent by this process.
 */
void collect_from_slaves( int my_rank, real *restrict *matrix, int bR, int size, int *restrict RnPv, int *restrict bRv,
			timing_data_t *restrict comm_time, cudaStream_t *restrict pstream, cudaEvent_t *restrict events );

// -------------------------------------------
// -------------------------------------------

/*
 * Performs NMF for niters iterations.
 *
 * bN == (my_rank*N)/nProcs	(ie. Starting row)
 * bM == (my_rank*M)/nProcs	(ie. Starting column)
 *
 * NnP == ((my_rank+1)*N/nProcs) - bN	(ie. Number of rows for this process).
 * MnP == ((my_rank+1)*M/nProcs) - bM	(ie. Number of columns for this process).
 *
 * MnPp == padding( MnP )
 * Kp == padding( Kp )
 *
 * adjust_matrices: Set to 1 to adjust matrices W and H on last iteration.
 */
void nmf_loop( int niters, real *restrict *pVfil, real *restrict *pVcol, real *restrict *WHcol, real *restrict *WHfil, real *restrict *W,
	real *restrict *Htras, real *restrict *Waux, real *restrict *Haux, real *restrict acumm_W, real *restrict acumm_H, int N, int M,
	int K, int Kp, int NnP, int MnP, int bN, int bM, int nProcs, int my_rank, int *restrict NnPv, int *restrict bNv, int *restrict MnPv,
	int *restrict bMv, timing_data_t *restrict comm_time, real *restrict d_Vfil, real *restrict d_Vcol, real *restrict d_WH,
	real *restrict d_W, real *restrict *pd_W, real *restrict d_H, real *restrict *pd_H, real *restrict d_Aux, real *restrict d_accum,
	int Mp, int MnPp, block_t *restrict block_N, block_t *restrict block_M, int *restrict iBlockN, int *restrict iBlockM,
	int *restrict stepN, int *restrict stepM, int *restrict offset_Vcol, cudaStream_t *restrict streams,
	cudaStream_t *restrict *pstream_NMF_N, cudaStream_t *restrict *pstream_NMF_M, cudaEvent_t *restrict events,
	timing_data_t *restrict reduce_timing, timing_data_t *restrict div_timing, timing_data_t *restrict mul_div_timing,
	timing_data_t *restrict upload_Vfil_timing, timing_data_t *restrict upload_Vcol_timing, timing_data_t *restrict upload_H_timing,
	timing_data_t *restrict upload_W_timing, timing_data_t *restrict download_H_timing, timing_data_t *restrict download_W_timing,
	timing_data_t *restrict adjust_timing, timing_data_t *restrict classf_timing, int adjust_matrices );

// -------------------------------------------

/*
 * Performs up to niter iterations. Performs test of convergence each niter_test_conv iterations (niter > niter_test_conv).
 *
 * bN == (my_rank*N)/nProcs	(ie. Starting row)
 * bM == (my_rank*M)/nProcs	(ie. Starting column)
 *
 * NnP == ((my_rank+1)*N/nProcs) - bN	(ie. Number of rows for this process).
 * MnP == ((my_rank+1)*M/nProcs) - bM	(ie. Number of columns for this process).
 *
 * MnPp == padding( MnP )
 * Kp == padding( Kp )
 *
 * Returns number of iterations performed.
 */
int nmf( int niter, int niter_test_conv, int stop_threshold, real *restrict *pVfil, real *restrict *pVcol, real *restrict *WHcol,
	real *restrict *WHfil, real *restrict *W, real *restrict *Htras, real *restrict *Waux, real *restrict *Haux, real *restrict acumm_W,
	real *restrict acumm_H, int *restrict classification, int *restrict last_classification, int N, int M, int K, int Kp, int NnP,
	int MnP, int bN, int bM, int nProcs, int my_rank, int *restrict NnPv, int *restrict bNv, int *restrict MnPv, int *restrict bMv,
	timing_data_t *restrict comm_time, real *restrict d_Vfil, real *restrict d_Vcol, real *restrict d_WH, real *restrict d_W,
	real *restrict *pd_W, real *restrict d_H, real *restrict *pd_H, real *restrict d_Aux, real *restrict d_accum,
	int *restrict d_classification, int Mp, int MnPp, block_t *restrict block_N, block_t *restrict block_M, int *restrict iBlockN,
	int *restrict iBlockM, int *restrict stepN, int *restrict stepM, int *restrict offset_Vcol, cudaStream_t *restrict streams,
	cudaStream_t *restrict *pstream_NMF_N, cudaStream_t *restrict *pstream_NMF_M, cudaEvent_t *restrict events,
	timing_data_t *restrict reduce_timing, timing_data_t *restrict div_timing, timing_data_t *restrict mul_div_timing,
	timing_data_t *restrict upload_Vfil_timing, timing_data_t *restrict upload_Vcol_timing, timing_data_t *restrict upload_H_timing,
	timing_data_t *restrict upload_W_timing, timing_data_t *restrict download_H_timing, timing_data_t *restrict download_W_timing,
	timing_data_t *restrict adjust_timing, timing_data_t *restrict idx_max_timing, timing_data_t *restrict download_classf_timing,
	timing_data_t *restrict classf_timing );

// -------------------------------------------
// -------------------------------------------

/*
 * Returns the reference value for distance:
 *	ref = norm( Vrow, 'Frobenius' )**2
 * where:
 *	norm( A , 'Frobenius' ) = sqrt( sum( diag( At * A ) ) ), where
 *	diag( X ) returns the main diagonal of X.
 *
 * WARNING: Assumes that padding is set to 0.0
 *
 * Returns <= 0 on error.
 */
real get_ref_distance_MPI( real *restrict *Vrow, int NnP, int Mp, int my_rank, size_t maxSize, timing_data_t *restrict comm_time );

// -------------------------------------------

/*
 * Returns the 'distance' between V and W*H.
 *	distance = ( norm(V-WH,'fro')**2 ) / ref
 * where
 *	norm( A , 'Frobenius' ) = sqrt( sum( diag(At * A) ) )
 *	diag( X ) is the main diagonal of X.
 *
 * WARNING: WH is used as a temporary matrix for W * H (GPU or CPU).
 *
 * Returns < 0 on error.
 */
real get_distance_MPI( real ref_distance, real *restrict *Vfil, real *restrict *WHfil, real *restrict *W, real *restrict *Htras, int N, int M,
			int K, int Kp, int NnP, int bN, int nProcs, int my_rank, timing_data_t *restrict comm_time, real *restrict d_Vfil,
			real *restrict d_WH, real *restrict *pd_W, real *restrict d_H, int Mp, block_t *restrict block_N, int *restrict iBlockN,
			int *restrict stepN, cudaStream_t *restrict streams, cudaStream_t *restrict *pstream_NMF_N,
			cudaEvent_t *restrict events, timing_data_t *restrict sub_timing, timing_data_t *restrict upload_Vfil_timing );

// -------------------------------------------

int write_matrix( real *restrict *matrix, int nrows, int ncols, const char *restrict filename, char row_major );

// -------------------------------------------

int write_matrix_binary( real *restrict *matrix, int nrows, int ncols, const char *restrict filename );

// -------------------------------------------

int write_binary_int( int *restrict array, int nitems, const char *restrict filename );

// --------------------------------

/*
 * Sets CPU affinity.
 * Process p will be executed in CPU p*(num_cpus/nProcs).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int set_affinity( int my_rank, int nProcs, int num_cpus );

// --------------------------------

#ifdef __cplusplus /* If this is a C++ compiler, end C linkage */
}
#endif


////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

#endif /* _NMF_MPI_CUDA_H_ */
