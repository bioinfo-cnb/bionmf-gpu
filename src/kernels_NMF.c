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

#include "kernels_NMF.h"

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

void reduce_matrix( real *restrict *matrix, real *restrict acumm, int R, int K, int my_rank, real *restrict d_Matrix, real *restrict d_Aux,
		int Kp, real *restrict d_accum, cudaStream_t *restrict streams, cudaEvent_t *restrict events,
		timing_data_t *restrict reduce_timing )
{

	#ifdef GPU

		/* WARNING:
		 * If size( d_Aux ) < N*Kp
		 *	it must reduce only its portion of d_W (NnP*Kp) and combine it with the other processes.
		 */
		reduce_matrix_GPU( d_Matrix, R, Kp, my_rank, d_Aux, d_accum, streams, events, reduce_timing );

	#else
		int i,j;

		for (j=0 ; j<K ; j++)
			acumm[j] = matrix[0][j];

		for (i=1 ; i<R ; i++)
			for (j=0 ; j<K ; j++)
				acumm[j] += matrix[i][j];
	#endif

} // reduce_matrix

///////////////////////////////////////////////////////////

/*
 * matrix(r,c) = MAX( matrix(r,c) , EPS )
 */
void adjust_matrix( real *restrict *matrix, int R, int K, int my_rank, real *restrict pd_Matrix, int Kp, cudaStream_t *restrict pstream,
			cudaEvent_t *restrict events, timing_data_t *restrict adjust_timing, timing_data_t *restrict classf_timing )
{

	#ifdef GPU

		adjust_matrix_GPU( pd_Matrix, R, Kp, my_rank, pstream, events, adjust_timing, classf_timing );

	#else

		int i;
		for( i=0 ; i<R ; i++ ) {
			int j;
			for( j=0 ; j<Kp ; j++ ) {
				real val = matrix[i][j];
				val = MAX_I( val, EPS )
				matrix[i][j] = val;
			}
		}

	#endif

} // adjust_matrix

///////////////////////////////////////////////////////////

/*
 * Computes classification vector.
 */
void get_classification( real *restrict *Htras, int *restrict classification, int M, int K, int Mp, int my_rank,
			real *restrict d_H, int *restrict d_classification, real *restrict d_Aux, int Kp, cudaStream_t *restrict streams,
			cudaEvent_t *restrict events, timing_data_t *restrict idx_max_timing, timing_data_t *restrict download_classf_timing )
{

#ifdef GPU

	get_classification_GPU( classification, d_H, d_classification, d_Aux, M, K, Mp, Kp, my_rank, streams, events,
				idx_max_timing, download_classf_timing );

#else

	int i;
	for ( i=0; i<M; i++ ) {
		real max = Htras[i][0];
		classification[i] = 0;
		int j;
		for ( j=1 ; j<K ; j++ ) {
			real val = Htras[i][j];
			if (val > max) {
				classification[i] = j;
				max = val;
			}
		}
	}

#endif


} // get_classification

///////////////////////////////////////////////////////////

/*
 * Gets the difference between classification and last_classification vectors.
 */
unsigned int get_difference( const int *restrict classification, const int *restrict last_classification, int M )
{
	unsigned int diff = 0;

	int i;
	for( i=0 ; i<(M-1) ; i++ ) {

		int classf_i = classification[i];
		int last_classf_i = last_classification[i];

		int j;
		for( j=(i+1) ; j<M ; j++ ) {

			int classf_j = classification[j];
			int last_classf_j = last_classification[j];

			int conn = ( classf_j == classf_i );
			int conn_last = ( last_classf_j == last_classf_i );

			diff += (unsigned int) ( conn != conn_last );

		}

	}

	return diff;

} // get_difference

///////////////////////////////////////////////////////////

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
	timing_data_t *restrict upload_Vcol_timing )
{

#ifdef GPU

	// WARNING: Modifies pointers Vcol (pitch==MnPp) and pd_H

	getH_GPU( Vcol, block_M, iBlockM, stepM, offset_Vcol, d_W, pd_H, d_WH, d_Vcol, d_Aux, d_accum, N, K, Kp, MnPp, my_rank,
		streams, pstream_NMF, events, div_timing, mul_div_timing, upload_Vcol_timing );

#elif ATLAS
        int i,j,jj;
        cblas_rgemm( CblasRowMajor, CblasNoTrans, CblasTrans,
            N,              /* [m] */
            MnP,                /* [n] */
            K,              /* [k] */
            1,              /* alfa */
            &W[0][0], Kp,            /* A[m][k], num columnas (lda) */
            &Htras[my_rank*M/nProcs][0], Kp, /* B[k][n], num columnas (ldb) */
            0,              /* beta */
            &WHcol[0][0], MnP       /* C[m][n], num columnas (ldc) */
        );
        for (i=0; i<N; i++)
            for (j=0; j<MnP; j++)
            {
                WHcol[i][j] = Vcol[i][j]/WHcol[i][j]; /* V./(W*H) */
            }
        cblas_rgemm( CblasColMajor, CblasNoTrans, CblasTrans,
            K,              /* [m] */
            MnP,                /* [n] */
            N,              /* [k] */
            1,              /* alfa */
            &W[0][0], Kp,            /* A[m][k], num columnas (lda) */
            &WHcol[0][0], MnP,      /* B[k][n], num columnas (ldb) */
            0,                          /* beta */
            &Haux[my_rank*M/nProcs][0], Kp   /* C[m][n], num columnas (ldc) */
        );
        for (j=0; j<MnP; j++){
            jj = j+my_rank*M/nProcs;
            for (i=0; i<K; i++)
                Htras[jj][i] = Htras[jj][i]*Haux[jj][i]/acumm_W[i]; /* H = H .* (Haux) ./ accum_W */
        }

#else
        int i,j,k,jj;
        for (i=0; i<N; i++)
            for (j=0; j<MnP; j++)
            {
                WHcol[i][j] = 0.0;
                jj = j+my_rank*M/nProcs;
                for(k=0; k<K; k++)
                {
                    WHcol[i][j] +=W[i][k]*Htras[jj][k];
                }
                WHcol[i][j] = Vcol[i][j]/WHcol[i][j]; /* V./(W*H) */
            }
        for (j=0; j<MnP; j++){
            jj = j+my_rank*M/nProcs;
            for (i=0; i<K; i++)
                Haux[jj][i] = 0.0;
            }
        for (k=0; k<N; k++)
            for (i=0; i<K; i++)
                for (j=0; j<MnP; j++)
                {
                    jj = j+my_rank*M/nProcs;
                    Haux[jj][i] += W[k][i]*WHcol[k][j];
                }
        for (j=0; j<MnP; j++){
            jj = j+my_rank*M/nProcs;
            for (i=0; i<K; i++)
                Htras[jj][i] = Htras[jj][i]*Haux[jj][i]/acumm_W[i]; /* H = H .* (Haux) ./ accum_W */
        }
#endif
} // getH

//////////////////////////////////////////////////////
//////////////////////////////////////////////////////

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
	timing_data_t *restrict div_timing, timing_data_t *restrict mul_div_timing, timing_data_t *restrict upload_Vfil_timing )
{

        /* WH = W*H */
        /* V./(W*H) */
        /* Waux =  {V./(W*H)} *H' */
        /* W = W .* Waux ./ accum_H */

#ifdef GPU

	// WARNING: Modifies pointers Vfil and pd_W.

	getW_GPU( Vfil, block_N, iBlockN, stepN, pd_W, d_H, d_WH, d_Vfil, d_Aux, d_accum, M, Mp, K, Kp, my_rank,
		streams, pstream_NMF, events, div_timing, mul_div_timing, upload_Vfil_timing );

#elif ATLAS
    int i,j,k,ii;
        cblas_rgemm( CblasRowMajor, CblasNoTrans, CblasTrans,
            NnP,                /* [m] */
            M,              /* [n] */
            K,              /* [k] */
            1,              /* alfa */
            &W[my_rank*N/nProcs][0], Kp,     /* A[m][k], num columnas (lda) */
            &Htras[0][0], Kp,        /* B[k][n], num columnas (ldb) */
            0,              /* beta */
            &WHfil[0][0], M         /* C[m][n], num columnas (ldc) */
        );

            for (i=0; i<NnP; i++)
                    for (j=0; j<M; j++)
                    {
                        WHfil[i][j] = Vfil[i][j]/WHfil[i][j]; /* V./(W*H) */
                    }
    cblas_rgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
        NnP,                /* [m] */
        K,              /* [n] */
        M,              /* [k] */
        1,              /* alfa */
        &WHfil[0][0], M,        /* A[m][k], num columnas (lda) */
        &Htras[0][0], Kp,        /* B[k][n], num columnas (ldb) */
        0,              /* beta */
        &Waux[my_rank*N/nProcs][0], Kp   /* C[m][n], num columnas (ldc) */
    );

    for (i=0; i<NnP; i++)
    {
        ii = i+my_rank*N/nProcs;
        for (j=0; j<K; j++)
        {
            Waux[ii][j] = 0.0;
            for(k=0; k<M; k++)
            {
                Waux[ii][j] += WHfil[i][k]*Htras[k][j];
            }
            W[ii][j] = W[ii][j]*Waux[ii][j]/acumm_H[j]; /* W = W .* Waux ./ accum_H */
        }
    }
#else
    int i,j,k,ii;
        for (i=0; i<NnP; i++)
        {
            ii = i+my_rank*N/nProcs;
            for (j=0; j<M; j++)
            {
                WHfil[i][j] = 0.0;
                for(k=0; k<K; k++)
                {
                    WHfil[i][j] +=W[ii][k]*Htras[j][k];
                }
                WHfil[i][j] = Vfil[i][j]/WHfil[i][j]; /* V./(W*H) */
            }
        }
    for (i=0; i<NnP; i++)
    {
        ii = i+my_rank*N/nProcs;
        for (j=0; j<K; j++)
        {
            Waux[ii][j] = 0.0;
            for(k=0; k<M; k++)
            {
                Waux[ii][j] += WHfil[i][k]*Htras[k][j];
            }
            W[ii][j] = W[ii][j]*Waux[ii][j]/acumm_H[j]; /* W = W .* Waux ./ accum_H */
        }
    }
#endif
}  // getW

//////////////////////////////////////////////////////
//////////////////////////////////////////////////////


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
real get_ref_distance( real *restrict *Vrow, int NnP, int Mp, int my_rank, size_t maxSize )
{

	real dot_V = 0;

#ifdef GPU
	dot_V = dot_product_V_GPU( &Vrow[0][0], NnP, Mp, my_rank, maxSize );
	if ( dot_V == 0 ) {
		fflush(NULL);
		fprintf(stderr, "Process %i: Failed to get dot product from V (GPU).\n", my_rank);
	}


#elif ATLAS
	dot_V = cblas_rdot( NnP * Mp, &Vrow[0][0], 1, &Vrow[0][0], 1 );


#else /* C */
	real *pVrow = &Vrow[0][0];
	int i;
	for ( i=0 ; i<(NnP * Mp) ; i++ ) {
		real val_V = pVrow[i];
		dot_V += (val_V * val_V);
	}


#endif

	return dot_V;

} // get_ref_distance

//////////////////////////////////////////////////////

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
			timing_data_t *restrict sub_timing, timing_data_t *restrict upload_Vfil_timing )
{

	real dot_VWH = 0;

	#ifdef GPU

		// WARNING: Modifies pointers Vfil and pd_W.

		dot_VWH = get_dot_products_GPU( Vfil, block_N, iBlockN, stepN, pd_W, d_H, d_WH, d_Vfil, M, Mp, K, Kp, my_rank,
						streams, pstream_NMF, events, sub_timing, upload_Vfil_timing );

		if ( dot_VWH < 0 ) {
			fprintf(stderr, "Process %i: Failed to get dot products (GPU).\n", my_rank);
			return -1;
		}


	#elif ATLAS

		int i;

		// WH(NnP,Mp) = W(NnP,Kp) * H
		cblas_rgemm( CblasRowMajor, CblasNoTrans, CblasTrans, NnP, M, K, 1,
				&W[my_rank*N/nProcs][0], Kp, &Htras[0][0], Kp, 0, &WHfil[0][0], M );

		// ----------------------------

		// WH = (V - WH) // WARNING: ATLAS only!
		for ( i=0 ; i<NnP ; i++ )
			catlas_raxpby( M, 1, &Vfil[i][0], 1, -1, &WHfil[i][0], 1 );	// WH = V + (-1)WH

		// ----------------------------

		/* Dot product of (its portion of) WH-V
		 * dot_product( V-WH, V-WH ) == SUM( (V-WH)**2 )
		 */

		for ( i=0 ; i<NnP ; i++ )
			dot_VWH += cblas_rdot( M, &WHfil[i][0], 1, &WHfil[i][0], 1 );

		// ----------------------------

	#else /* C */


		// WH(NnP,Mp) = W(NnP,Kp) * H
		int i;
		for ( i=0 ; i<NnP ; i++ ) {
			int ii = i+my_rank*N/nProcs;
			int j;
			for ( j=0 ; j<M ; j++ ) {
				real valWH = 0.0;
				int k;
				for( k=0 ; k<K ; k++ ) {
					real valW = W[ii][k];
					real valH = Htras[j][k];
					valWH += valW * valH;
				}
				WHfil[i][j] = valWH;
			}
		}

		// ----------------------------

		/* Dot product of (its portion of) WH-V
		 * dot_product( V-WH, V-WH ) = SUM( (V-WH)**2 )
		 */

		for ( i=0 ; i<NnP ; i++ ) {
			int j;
			for ( j=0 ; j<M ; j++ ) {
				real val_WH = WHfil[i][j];
				real val_V = Vfil[i][j];
				val_WH = val_V - val_WH;	// V - WH
				val_WH *= val_WH;		// (V-WH)**2
				dot_VWH += val_WH;		// dot_product(V-WH,V-WH)
			}
		}

	#endif /* C */

	// ---------------------------------

	return dot_VWH;

} // get_distance_VWH

//////////////////////////////////////////////////////
