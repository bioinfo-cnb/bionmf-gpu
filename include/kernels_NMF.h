/***********************
 * MAYOR CHANGES FROM PREVIOUS VERSION:
 *	- d_Waux, d_Haux  ---->  d_Aux
 *	- d_accum_h, d_accum_w ----> d_accum
 *	- BLN, BLM ----> N, M (or NnP, MnP)
 *	- K_padding ----> Kp
 *	- BLMnP_padding -----> MnPp
 *	- Added cudaStreams and cudaEvents
 *	- Added argument "my_rank" to all functions.
 *	- Asynchronous transfers.
 *	- NOTE: W and Htras (CPU matrices) have padding.
 *	- Block configuration.
 *
 * bN == (my_rank*N)/nProcs	(i.e.,Starting row)
 * bM == (my_rank*M)/nProcs	(i.e.,Starting column)
 *
 * NnP == ((my_rank+1)*N/nProcs) - bN	(i.e.,Number of rows for this process).
 * MnP == ((my_rank+1)*M/nProcs) - bM	(i.e.,Number of columns for this process).
 *
 * MnPp == padding( MnP )
 * Kp == padding( Kp )
 *
 * NOTE: Block configuration might replace NnP, MnP and MnPp by current block values.
 *	GPU modifies pointers to Vcol and Vfil
 *
 **********************/

#ifndef _KERNELS_NMF_H_

#define _KERNELS_NMF_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// --------------------------------------------
// --------------------------------------------

/* Includes */

#include "common.h"

#ifdef GPU
	#include <cublas.h>
	#include "rutinesGPU.h"

#elif ATLAS
	#include <cblas.h>

#endif

// --------------------------------------------
// --------------------------------------------

/* Other macros */

#ifdef ATLAS
	#ifdef REAL_FLOAT
		#define cblas_rgemm cblas_sgemm
		#define cblas_rdot cblas_sdot
		#define catlas_raxpby catlas_saxpby
	#else
		#define cblas_rgemm cblas_dgemm
		#define cblas_rdot cblas_ddot
		#define catlas_raxpby catlas_daxpby
	#endif
#endif

/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////


#ifdef __cplusplus /* If this is a C++ compiler, use C linkage */
extern "C" {
#endif

// -----------------------------

void reduce_matrix( real *restrict *matrix, real *restrict acumm, int R, int K, int my_rank, real *restrict d_Matrix, real *restrict d_Aux,
		int Kp, real *restrict d_accum, cudaStream_t *restrict streams, cudaEvent_t *restrict events,
		timing_data_t *restrict reduce_timing );

// --------------------------------

/*
 * matrix(r,c) = MAX( matrix(r,c) , EPS )
 */
void adjust_matrix( real *restrict *matrix, int R, int K, int my_rank, real *restrict pd_Matrix, int Kp, cudaStream_t *restrict pstream,
			cudaEvent_t *restrict events, timing_data_t *restrict adjust_timing, timing_data_t *restrict classf_timing );

// --------------------------------

/*
 * Computes classification vector.
 */
void get_classification( real *restrict *Htras, int *restrict classification, int M, int K, int Mp, int my_rank,
			real *restrict d_H, int *restrict d_classification, real *restrict d_Aux, int Kp, cudaStream_t *restrict streams,
			cudaEvent_t *restrict events, timing_data_t *restrict idx_max_timing, timing_data_t *restrict download_classf_timing );

// --------------------------------

/*
 * Gets the difference between classification and last_classification vectors.
 */
unsigned int get_difference( const int *restrict classification, const int *restrict last_classification, int M );

// --------------------------------
// --------------------------------

/* WH(N,MnPp) = W * H(MnP,Kp)
 * WH(N,MnPp) = Vcol(N,MnPp) ./ WH(N,MnPp)
 * Haux(MnP,Kp) = W' * WH(N,MnPp)
 * H(MnP,Kp) = H(MnP,Kp) .* Haux(MnP,Kp) ./ accum_W
 *
 * pd_H == d_H + bM*Kp
 *   bM == (my_rank*M)/nProcs
 * MnPp == get_padding( MnP )
 */
void getH( real *restrict *W, real *restrict *Htras, real *restrict *WHcol, real *restrict *Vcol, real *restrict *Haux, real *restrict acumm_W,
	int N, int M, int MnP, int MnPp, int K, int my_rank, int nProcs, block_t *restrict block_M, int *restrict iBlockM, int *restrict stepM,
	int *restrict offset_Vcol, int Kp, real *restrict d_W, real *restrict *pd_H, real *restrict d_WH, real *restrict d_Vcol,
	real *restrict d_Aux, real *restrict d_accum, cudaStream_t *restrict streams, cudaStream_t *restrict *pstream_NMF,
	cudaEvent_t *restrict events, timing_data_t *restrict div_timing, timing_data_t *restrict mul_div_timing,
	timing_data_t *restrict upload_Vcol_timing );

// --------------------------------
// --------------------------------

/* WH(NnP,Mp) = W(NnP,Kp) * H
 * WH(NnP,Mp) = Vfil(NnP,Mp) ./ WH(NnP,Mp)
 * Waux(NnP,Kp) = WH(NnP,Mp) * H'
 * W(NnP,Kp) = W(NnP,Kp) .* Waux(NnP,Kp) ./ accum_h
 *
 * pd_W == d_W + bN*Kp
 *   bN == (my_rank*N)/nProcs
 */
void getW( real *restrict *W, real *restrict *Htras, real *restrict *WHfil, real *restrict *Vfil, real *restrict *Waux, real *restrict acumm_H,
	int N, int NnP, int M, int K, int my_rank, int nProcs, block_t *restrict block_N, int *restrict iBlockN, int *restrict stepN,
	real *restrict *pd_W, real *restrict d_H, real *restrict d_WH, real *restrict d_Vfil, real *restrict d_Aux, real *restrict d_accum,
	int Kp, int Mp, cudaStream_t *restrict streams, cudaStream_t *restrict *pstream_NMF, cudaEvent_t *restrict events,
	timing_data_t *restrict div_timing, timing_data_t *restrict mul_div_timing, timing_data_t *restrict upload_Vfil_timing );

// --------------------------------
// --------------------------------

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
real get_ref_distance( real *restrict *Vrow, int NnP, int Mp, int my_rank, size_t maxSize );

// --------------------------------

/*
 * Returns the 'distance' between V and W*H.
 *	distance = ( norm(V-WH,'fro')**2 ) / ref
 * where
 *	norm( A , 'Frobenius' ) = sqrt( sum( diag(At * A) ) )
 *	diag( X ) is the main diagonal of X.
 *
 * pd_W == d_W + bN*Kp
 *   bN == (my_rank*N)/nProcs
 *
 * WARNING: Modifies iBlockN and stepN same way as getW()
 *
 * Returns < 0 on error.
*/
real get_distance_VWH(  real *restrict *W, real *restrict *Htras, real *restrict *WHfil, real *restrict *Vfil, int N, int NnP, int M, int K,
			int my_rank, int nProcs, block_t *restrict block_N, int *restrict iBlockN, int *restrict stepN,
			real *restrict *pd_W, real *restrict d_H, real *restrict d_WH, real *restrict d_Vfil, int Kp, int Mp,
			cudaStream_t *restrict streams, cudaStream_t *restrict *pstream_NMF, cudaEvent_t *restrict events,
			timing_data_t *restrict sub_timing, timing_data_t *restrict upload_Vfil_timing );

// --------------------------------

#ifdef __cplusplus /* If this is a C++ compiler, end C linkage */
}
#endif

///////////////////////////////////////////////////////

#endif /* _KERNELS_NMF_H_ */
