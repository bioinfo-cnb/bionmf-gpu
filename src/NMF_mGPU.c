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
 *	Main program for multi-GPU systems.
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
 *		NMFGPU_FORCE_DIMENSIONS: Overrides matrix dimensions.
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
 ***************
 *
 * Multi-GPU version:
 *
 * When the input matrix V is distributed among multiple devices each host thread processes
 * the following sets of rows and columns:
 *	Vrow[ 1..NnP ][ 1..M ] <-- V[ bN..(bN+NnP) ][ 1..M ]	(i.e., NnP rows, starting from bN)
 *	Vcol[ 1..N ][ 1..MnP ] <-- V[ 1..N ][ bM..(bM+MnP) ]	(i.e., MnP columns, starting from bM)
 *
 * Such sets allow to update the corresponding rows and columns of W and H, respectively.
 *
 * Note that each host thread has a private copy of matrices W and H, which must be synchronized
 * after being updated.
 *
 ****************
 *
 * Large input matrix (blockwise processing):
 *
 * If the input matrix (or the portion assigned to this device) is too large for the GPU memory,
 * it must be blockwise processed as follow:
 *	d_Vrow[1..BLN][1..Mp] <-- Vrow[ offset..(offset + BLN) ][1..Mp]			(i.e., BLN <= NnP rows)
 *	d_Vcol[1..N][1..BLMp] <-- Vcol[1..N][ offset_Vcol..(offset_Vcol + BLMp) ]	(i.e., BLM <= MnP columns)
 *
 * Note that padded dimensions are denoted with the suffix 'p' (e.g., Mp, BLMp, etc).
 *
 * In any case, matrices W and H are fully loaded into the GPU memory.
 *
 * Information for blockwise processing is stored in two block_t structures (one for each dimension).
 * Such structures ('block_N' and 'block_M') are initialized in init_block_conf() routine.
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

#include "nmf_MPI_CUDA.h"

int *get_memory1D_int( int nx )
{
	int *buffer = NULL;

	#ifdef GPU
		buffer = get_memory1D_int_GPU( nx );

	#else
		buffer = (int *) malloc( nx * sizeof(int) );
		if( buffer == NULL ) {
			int err = errno; fflush(NULL); errno = err;
			perror( "malloc (int): ERROR in memory allocation" );
			return( NULL );
		}
	#endif

	return buffer;
}

// ---------------------------------------

int destroy_memory1D_int( int *restrict buffer )
{
	int status = EXIT_SUCCESS;

	if ( buffer ) {
		#ifdef GPU
			status = destroy_memory1D_int_GPU( buffer );
		#else
			free(buffer);
		#endif
	}

	return status;
}

// -------------------------------

real *get_memory1D_V( int nx, int is_V )
{
	real *buffer = NULL;

#ifdef GPU
	buffer = get_memory1D_GPU( nx, is_V );

#else
	if( (buffer=(real *)malloc(nx*sizeof(real)))== NULL )
	{
		int err = errno; fflush(NULL); errno = err;
		perror( "malloc: ERROR in memory allocation" );
		return( NULL );
	}
#endif

	// int i;
	// for( i=0; i<nx; i++ )
	//{
	//	buffer[i] = (real)(i*10);
	//}

	return( buffer );

} // get_memory1D_V


real *get_memory1D( int nx )
{

	return get_memory1D_V( nx, 0 );

} // get_memory1D

// ------------------------------

real **get_memory2D_V( int nx, int ny, int is_V )
{

	real **buffer = NULL;

	if( (buffer=(real **)malloc(nx*sizeof(real *)))== NULL )
	{
		int err = errno; fflush(NULL); errno = err;
		perror( "malloc: ERROR in memory allocation" );
		return( NULL );
	}

	buffer[ 0 ] = get_memory1D_V( nx * ny, is_V );
	if ( buffer[ 0 ] == NULL ) {
		free( buffer );
		return NULL;
	}

	int i;
	for( i=1; i<nx; i++ )
	{
		buffer[i] = buffer[i-1] + ny;
	}

	// for( i=0; i<nx; i++ ) {
	//	int j;
	//	for( j=0; j<ny; j++ )
	//	{
	//		buffer[i][j] = (real)(i*100+j);
	//	}
	//}

	return( buffer );

} // get_memory2D_V

real **get_memory2D( int nx, int ny )
{

	return get_memory2D_V( nx, ny, 0 );

} // get_memory2D


real **get_memory2D_renew( int nx, int ny, real *restrict mem )
{
	int i;
	real **buffer = NULL;

	if( (buffer=(real **)malloc(nx*sizeof(real *)))== NULL )
	{
		int err = errno; fflush(NULL); errno = err;
		perror( "malloc: ERROR in memory allocation" );
		return( NULL );
	}


	for( i=0; i<nx; i++ )
	{
		buffer[i] = &(mem[i*ny]);
	}


	return( buffer );
}

// ---------------------------------------

int destroy_memory1D( real *restrict buffer )
{
	int status = EXIT_SUCCESS;

	if ( buffer ) {
		#ifdef GPU
			status = destroy_memory1D_GPU( buffer );
		#else
			free(buffer);
		#endif
	}

	return status;
}


int destroy_memory2D( real *restrict *buffer )
{
	int status = EXIT_SUCCESS;

	if ( buffer ) {
		status = destroy_memory1D( *buffer );
		free( (void *) buffer );
	}

	return status;
}

//////////////////////////////////////////////////////////////

void dimension_per_CPU( int N, int M, int my_rank, int nProcs, int *restrict NnP, int *restrict MnP, int *restrict bN, int *restrict bM)
{

	int bn = my_rank*N/nProcs;	// Starting row
	int bm = my_rank*M/nProcs;	// Starting column

	int nnp = ((my_rank+1)*N/nProcs) - bn;
	int mnp = ((my_rank+1)*M/nProcs) - bm;

	*NnP = nnp;
	*MnP = mnp;
	*bN = bn;
	*bM = bm;

}

//////////////////////////////////////////////////////////////

real **read_matrix( FILE *restrict fp, int M, int bN, int bM, int NnP, int MnP, int MnPp, int my_rank )
{

	if ( bN + bM ) {	// Musts skip multiple rows/columns

		long offset = 0;

		if ( bN )
			offset = (bN * M * sizeof(real));

		if ( bM )
			offset += (bM * sizeof(real));

		// Moves the file indicator to the selected position.
		if ( fseek(fp, offset , SEEK_CUR ) < 0 ) {
			int err = errno; fflush(NULL); errno = err;
			fprintf(stderr, "\nProcess %i: Error setting file position indicator: %s\n",my_rank, strerror(errno));
			return NULL;
		}
	}

	// ----------------------------

	real **V = get_memory2D_V( NnP, MnPp, 1 );	// Special mode for matrix V.
	if ( V == NULL ) {
		fprintf(stderr,"\nProcess %i: Memory allocation error (V: %ix%i)\n",my_rank,NnP,MnPp);
		return NULL;
	}

	int i;
	for ( i=0 ; i<NnP ; i++ ) {
		size_t nread = fread(&V[i][0],sizeof(real),MnP,fp);
		if ( nread != (size_t) MnP ) {
			fflush(NULL);
			if ( feof(fp) )
				fprintf(stderr, "\nProcess %i: Error reading input file.\nPremature end-of-file detected.\n", my_rank );
			else // error
				fprintf(stderr,"Process %i: Error reading input file\n.",my_rank);
			fprintf(stderr, "Process %i: Items read: %zu, expected: %i. Row=%i", my_rank, nread, MnP, i );
			destroy_memory2D(V);
			return NULL;
		}

		int j;
		for( j=MnP ; j<MnPp ; j++ )	// Padding
			V[i][j] = 0;

		// If necessary, moves to the next row.
		if ( ( (M-MnP) > 0 ) && ( fseek(fp, (M-MnP) * sizeof(real), SEEK_CUR ) < 0 ) ) {
			int err = errno; fflush(NULL); errno = err;
			fprintf(stderr, "\nProcess %i: Error setting file position indicator: %s\n",my_rank, strerror(errno));
			destroy_memory2D(V);
			return NULL;
		}
	}

	return V;

} // read_matrix

//////////////////////////////////////////////////////////////

int init_V( const char *restrict filename, real ***Vfil, real ***Vcol, int *restrict N, int *restrict M,
		int *restrict NnP, int *restrict MnP, int *restrict bN, int *restrict bM, int nProcs, int my_rank )
{

	// TODO:	REVISAR DATOS DE LA MATRIZ (NaN, >= 0, no todos 0's).


#ifdef _VERBOSE
if ( !my_rank ) printf("Reading input matrix from file %s...\n", filename);
#endif

	/* Read input file */
	FILE *fp = NULL;
	if ( (fp = fopen(filename,"r")) == NULL ) {
		int err = errno; fflush(NULL); errno = err;
		fprintf(stderr,"Process %i: Error opening input file\n.",my_rank);
		perror(filename);
		return EXIT_FAILURE;
	}

	int n,m;
//	int conv = fscanf(fp,"N %i M %i\n",&n,&m);
	int dim[2] ; int conv = fread( dim, sizeof(int), 2, fp ); n = dim[0] ; m = dim[1];
	if ( conv != 2 ) {
		int err = errno; fflush(NULL); errno = err;
		if ( ferror(fp) ) {	// Error.
			fprintf(stderr,"Process %i: Error reading input file\n.",my_rank);
			perror("\nfscanf()");
		}
		else if ( conv == EOF || feof(fp) ) { // Premature EOF.
			fprintf(stderr, "\nProcess %i: Error reading input file.\nPremature end-of-file detected.\n", my_rank );
		}
		else
			fprintf(stderr,"Process %i: Error reading input file\n.",my_rank);
		fclose(fp);
		return EXIT_FAILURE;
	}

	// -------------------------------

	#ifdef _FORCE_DIMENSIONS
		// Ignores dimensions in file and uses the ones provided.
		if ( ((*N) * (*M)) >= 1 ) {	// N>=1 && M>=1
			if ( ((*N) * (*M)) <= n * m ) {
				n = *N;
				m = *M;
			} else {
				n = MIN_I( *N, n );
				m = MIN_I( *M, m );
			}
			if (!my_rank) printf("\nForcing to N=%i M=%i\n",n,m);
		}
	#endif

	// -------------------------------

	int mp = get_padding( m );

	int bn, bm;	// Starting row/column
	int nnp, mnp;	// Rows / Columns for this process.
	int mnpp;	// Padding for mnp.

	real **l_Vfil = NULL;
	real **l_Vcol = NULL;

	if ( nProcs > 1 ) {

		#ifdef _VERBOSE
			if ( !my_rank ) printf("\tSetting Vfil and Vcol...\n");
		#endif

		dimension_per_CPU( n, m, my_rank, nProcs, &nnp, &mnp, &bn, &bm );

		mnpp = get_padding( mnp );

		// ------------------------

		// Saves current file position indicator.
		fpos_t fpos;
		if ( fgetpos(fp, &fpos) < 0 ) {
			int err = errno; fflush(NULL); errno = err;
			fprintf(stderr, "\nProcess %i: Error saving file position indicator: %s\n",my_rank, strerror(errno));
			fclose(fp);
			return EXIT_FAILURE;
		}

		// Vfil: NnP x Mp
		l_Vfil = read_matrix( fp, m, bn, 0, nnp, m, mp, my_rank );
		if ( l_Vfil == NULL ) {
			fprintf(stderr,"Process %i: Error setting Vfil.\n", my_rank);
			fclose(fp);
			return EXIT_FAILURE;
		}

		// ------------------------

		// Restores file position indicator.
		if ( fsetpos(fp, &fpos) < 0 ) {
			int err = errno; fflush(NULL); errno = err;
			fprintf(stderr, "\nProcess %i: Error setting file position indicator: %s\n",my_rank, strerror(errno));
			destroy_memory2D( l_Vfil );
			fclose(fp);
			return EXIT_FAILURE;
		}

		// Vcol: N x MnPp
		l_Vcol = read_matrix( fp, m, 0, bm, n, mnp, mnpp, my_rank );
		if ( l_Vcol == NULL ) {
			fprintf(stderr,"Process %i: Error setting Vcol.\n", my_rank);
			destroy_memory2D( l_Vfil );
			fclose(fp);
			return EXIT_FAILURE;
		}

	} else {	// Just one processor.

		bn = bm = 0;
		nnp = n;
		mnp = m;
		mnpp = mp;
		l_Vcol = l_Vfil = read_matrix( fp, m, bn, bm, nnp, mnp, mnpp, my_rank );
		if ( l_Vfil == NULL ) {
			fclose(fp);
			return EXIT_FAILURE;
		}
	} // if (nProcs > 1)

	fclose(fp);

#ifdef _VERBOSE
if ( !my_rank ) printf("\tDone.\n");
#endif

	*N = n;
	*M = m;
	*NnP = nnp;
	*MnP = mnp;
	*bN = bn;
	*bM = bm;
	*Vfil = l_Vfil;
	*Vcol = l_Vcol;

	return EXIT_SUCCESS;

} // init_V

///////////////////////////////////////

void init_matrix( real *restrict matrix, int nrows, int ncols, int padding, real min_value )
{

	real *pmatrix = matrix;

	int i;
	for( i=0 ; i<nrows ; i++ ) {
		int j;
		for ( j=0 ; j<ncols ; j++,pmatrix++ ) {
			real value =  ( ((real) rand()) / ((real) RAND_MAX) ) + min_value  ;
			*pmatrix = value;
		}
		for ( j=ncols ; j<padding ; j++,pmatrix++ )	// Padding
			*pmatrix = 1.0;
	}

} // init_matrix

////////////////////////////////////////

// row_major: 1 if matrix is row-wise.
void show_matrix( real *restrict *matrix, int nrows, int ncols, char row_major )
{

	if ( row_major ) { // matrix is in ROW-major format

		int numrows, numcols;
		if ( nrows == 1 ) {
			numcols = MIN_I( ncols, 225 ) ;
			numrows = 1;
		} else {
			numcols = MIN_I( ncols, 15 ) ;
			numrows = MIN_I( nrows, 9 ) ;
		}

		int i;
		for ( i=0 ; i<numrows ; i++ ) {
			printf("Line %i:", i);
			int j;
			for( j=0 ; j<numcols ; j++ )
				printf(" %g", matrix[i][j] );
			if ( numcols < ncols )	// Last column
				printf( " ... %g", matrix[i][ncols-1] );
			printf("\n");
		}
		if ( numrows < nrows ) {	// Shows last row.
			i = nrows - 1;
			printf("...\nLine %i:",i);
			int j;
			for( j=0 ; j<numcols ; j++ )
				printf(" %g ",matrix[i][j]);
			if ( numcols < ncols )	// Last column
				printf(" ... %g", matrix[i][ncols-1]);
			printf("\n");
		}

	} else { // COLUMN-major (file is shown in ROW-major format).

		int numrows = MIN_I( nrows, 9 ) ;
		int numcols = MIN_I( ncols, 15 ) ;

		int i;
		for ( i=0 ; i<numcols ; i++ ) {
			printf("Line %i:", i);
			int j;
			for( j=0 ; j<numrows ; j++ )
				printf(" %g", matrix[j][i]);
			if ( numrows < nrows )
				printf(" ... %g", matrix[nrows-1][i] );	// Last row
			printf("\n");
		}
		if ( numcols < ncols ) {
			i = ncols - 1;
			printf("...\nLine %i:", i);
			int j;
			for( j=0 ; j<numrows ; j++ )
				printf(" %g", matrix[j][i]);
			if ( numrows < nrows )	// Last row
				printf(" ... %g", matrix[nrows-1][i] );
			printf("\n");
		}
	} // if row_major

} // show_matrix

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

int init_classf_data( int Mp, int my_rank, int *restrict *classification, int *restrict *last_classification )
{

	int *l_classification = get_memory1D_int( Mp );
	if ( l_classification == NULL ) {
		fprintf(stderr,"Process %i: Error allocating memory for classification data.\n",my_rank);
		fflush(NULL);
		return EXIT_FAILURE;
	}

	int *l_last_classification = get_memory1D_int( Mp );
	if ( l_classification == NULL ) {
		fprintf(stderr,"Process %i: Error allocating memory for classification data (last_classification).\n",my_rank);
		fflush(NULL);
		destroy_memory1D_int(l_classification);
		return EXIT_FAILURE;
	}

	if ( memset( (void *)l_last_classification, 0, Mp*sizeof(int) ) == NULL ) {
		int err = errno; fflush(NULL); errno = err;
		perror("\nERROR in last_classification memset:" );
		destroy_memory1D_int(l_last_classification); destroy_memory1D_int(l_classification);
		return EXIT_FAILURE;
	}

	*classification = l_classification;
	*last_classification = l_last_classification;

	return EXIT_SUCCESS;

} // init_classf_data

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

/*
 * Allocates memory for vectors with number of matrix entries, and starting offset,
 * to be processed by each task.
 *
 * R can be set to N or M.
 *
 * bRv: offset to starting row for each process (i.e.,all bR*C values).
 * RnPv: Number of elements for each process (i.e.,all RnP*C values).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int allocate_dim_vectors( int nProcs, int *restrict *RnPv, int *restrict *bRv )
{

	int *l_RnPv = (int *) malloc( nProcs * sizeof(int) );
	if ( l_RnPv == NULL ) {
		int err = errno; fflush(NULL); errno = err;
		perror("\nmalloc(l_RnPv)");
		fprintf(stderr,"Error in allocate_dim_vectors().\n");
		return EXIT_FAILURE;
	}

	int *l_bRv = (int *) malloc( nProcs * sizeof(int) );
	if ( l_bRv == NULL )  {
		int err = errno; fflush(NULL); errno = err;
		perror("\nmalloc(l_bRv)");
		fprintf(stderr,"Error in allocate_dim_vectors().\n");
		free(l_RnPv);
		return EXIT_FAILURE;
	}

	// Sets output pointers
	*RnPv = l_RnPv;
	*bRv = l_bRv;

	return EXIT_SUCCESS;

} // allocate_dim_vectors

//////////////////////////////////////////////////

/*
 * Initializes vectors with number of matrix entries, and starting offset, to be processed by each task.
 * R (rows) can be set to N or M.
 * C (columns) can be set to K (or Kp if using padding).
 *
 * bRv: offset to starting row for each process (i.e.,all bR*C values).
 * RnPv: Number of elements for each process (i.e.,all RnP*C values).
 */
void init_dim_vectors( int R, int C, int nProcs, int *restrict RnPv, int *restrict bRv )
{

	/*
	 * bR == (rank*R) / nProcs	(Starting row)
	 *
	 * nR1 == (rank+1) * R		(Temporary value)
	 *
	 * RnP == (nR1/nProcs) - bR	(Number of rows for this process).
	 *
	 * C columns
	 *
	 * bRv: All bR*C values for all processes.
	 * RnPv: All RnP*C values for all processes.
	 */

	int nR1 = R;	// rank == 0
	int bR  = 0;	// rank == 0

	int rank;
	for ( rank=0 ; rank<nProcs ; rank++ ) {

		// Starting row for process rank+1 (result may be truncated).
		int bR_rank1 = nR1 / nProcs;	// (rank+1)*R / nProcs

		// Number of rows for current process (rank).
		int RnP = bR_rank1 - bR;	// (rank+1)*R/nProcs - rank*R/nProcs

		// Starting offset and number of elements.
		int offset = bR * C;
		int num_elems = RnP * C;

		// Stores current values.
		bRv[ rank ] = offset;
		RnPv[ rank ] = num_elems;

		// Next process...
		nR1 += R;		// (rank+1) * R
		bR = bR_rank1;		// bR <--- (rank+1)*R / nProcs

	} // for nProcs times

} // init_dim_vectors_offset

//////////////////////////////////////////////////

/*
 * All-to-All synchronization.
 * Gathers data from all tasks and distributes the combined data to all tasks.
 *	p_matrix: Starting address of data to be sent by slaves.
 *	size: Size of data to be sent by this process.
 */
void sync_matrix( int my_rank, real *restrict *matrix, int bR, int size, int *restrict RnPv, int *restrict bRv,
		timing_data_t *restrict comm_time, cudaStream_t *restrict pstream, cudaEvent_t *restrict events )
{

	/* The block of data sent from the jth process is received by every process
	 * and placed in the jth block of matrix.
	 */

	#ifdef GPU
		sync_GPU( my_rank, pstream, events );	// Waits until GPU completes all ops. on <stream>.
	#endif

	#ifdef _PROFILING_1
		MPI_Barrier( MPI_COMM_WORLD );
		double time_0 = MPI_Wtime();
	#endif

	if ( RnPv )	// RnP is different on each process.
		MPI_Allgatherv( &matrix[bR][0], size, MPI_TREAL, &matrix[0][0], RnPv, bRv, MPI_TREAL, MPI_COMM_WORLD );

	else		// RnP is equal on all processes.
		MPI_Allgather( &matrix[bR][0], size, MPI_TREAL, &matrix[0][0], size, MPI_TREAL, MPI_COMM_WORLD );

	#ifdef _PROFILING_1
		MPI_Barrier( MPI_COMM_WORLD );
		double time_1 = MPI_Wtime();
		comm_time->time += (time_1 - time_0);
		comm_time->counter++;
	#endif

} // sync_matrix

//////////////////////////////////////////////////

/*
 * Gathers data from slave processes.
 * All slave processes send their portion of data to master.
 *	p_matrix: Starting address of data to be sent by slaves.
 *	size: Size of data to be sent by this process.
 */
void collect_from_slaves( int my_rank, real *restrict *matrix, int bR, int size, int *restrict RnPv, int *restrict bRv,
			timing_data_t *restrict comm_time, cudaStream_t *restrict pstream, cudaEvent_t *restrict events )
{

	/* The block of data sent from the jth process is received by the master
	 * and placed in the jth block of matrix.
	 */

	#ifdef GPU
		sync_GPU( my_rank, pstream, events );	// Waits until GPU completes all ops. on <stream>.
	#endif

	#ifdef _PROFILING_1
		MPI_Barrier( MPI_COMM_WORLD );
		double time_0 = MPI_Wtime();
	#endif

	if ( RnPv )	// RnP is different on each process.
		MPI_Gatherv( &matrix[bR][0], size, MPI_TREAL, &matrix[0][0], RnPv, bRv, MPI_TREAL, 0, MPI_COMM_WORLD );

	else		// RnP is equal on all processes.
		MPI_Gather( &matrix[bR][0], size, MPI_TREAL, &matrix[0][0], size, MPI_TREAL, 0, MPI_COMM_WORLD );

	#ifdef _PROFILING_1
		MPI_Barrier( MPI_COMM_WORLD );
		double time_1 = MPI_Wtime();
		comm_time->time += (time_1 - time_0);
		comm_time->counter++;
	#endif

} // collect_from_slaves

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

/*
 * Performs NMF for <niters> iterations.
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
	timing_data_t *restrict adjust_timing, timing_data_t *restrict classf_timing, int adjust_matrices )
{

	// Full size of matrices W and H.
	int size_W = N * Kp;
	int size_Htras = M * Kp;

	// Portions of matrices W and H to be processed by this thread.
	real *pW = &W[bN][0];
	real *pHtras = &Htras[bM][0];

	// Such portions in GPU.
	real *pd_W0 = (d_W + bN * Kp);
	real *pd_H0 = (d_H + bM * Kp);

	// Size of such portions (GPU and CPU)
	int size_pW = NnP * Kp;
	int size_pHtras = MnP * Kp;


	// -------------------------------


	// Performs <niters> - 1 iterations

	int i;
	for( i=1 ; i<niters ; i++ ) {

		// Synchronizes matrix W among all processes.
		sync_matrix( my_rank, W, bN, size_pW, NnPv, bNv, comm_time, *pstream_NMF_N, events );

		// Uploads matrix W (full size)
		#ifdef GPU
			upload_matrix( W, size_W, my_rank, d_W, streams, *pstream_NMF_M, events, upload_W_timing );
		#endif

		// Reduces to a vector (full size)
		reduce_matrix( W, acumm_W, N, K, my_rank, d_W, d_Aux, Kp, d_accum, streams, events, reduce_timing );

		// Note: Uploading and reducing full matrix W avoids a new synchronization point.


		/*******************************************/
		/*** H = H .* (W'*(V./(W*H))) ./ accum_W ***/
		/*******************************************/

		/* WH = W*H
		 * WH = V ./ WH
		 * Haux = (W'* {V./(WH)}
		 * H = H .* (Haux) ./ accum_W
		 */
		getH( W, Htras, WHcol, pVcol, Haux, acumm_W, N, M, MnP, MnPp, K, my_rank, nProcs,
			block_M, iBlockM, stepM, offset_Vcol, Kp, d_W, pd_H, d_WH, d_Vcol, d_Aux, d_accum,
			streams, pstream_NMF_M, events, div_timing, mul_div_timing, upload_Vcol_timing );


		// GPU: Downloads (asynchronously) the updated block from d_H.
		#ifdef GPU
			download_matrix( pHtras, size_pHtras, my_rank, pd_H0, *pstream_NMF_M, events, download_H_timing );
		#endif


		// Synchronizes matrix H among all processes.
		sync_matrix( my_rank, Htras, bM, size_pHtras, MnPv, bMv, comm_time, *pstream_NMF_M, events );


		// --------------------------------


		// NOTE: From here, uses *pstream_NMF_N:


		// Uploads synchronized matrix H (full size).
		#ifdef GPU
			upload_matrix( Htras, size_Htras, my_rank, d_H, streams, *pstream_NMF_N, events, upload_H_timing );
		#endif


		// Reduces matrix H (full size) to a vector
		reduce_matrix( Htras, acumm_H, M, K, my_rank, d_H, d_Aux, Kp, d_accum, streams, events, reduce_timing );

		// Note: Uploading and reducing full matrix H avoids a new synchronization point.


		//////////////////////////////////////////////////////////////


		/*******************************************/
		/*** W = W .* ((V./(W*H))*H') ./ accum_H ***/
		/*******************************************/

		/* WH = W*H
		 * WH = V ./ WH
		 * Waux = (W'*(V./(WH)))
		 * W = W .* Waux ./ accum_H
		 */
		getW( W, Htras, WHfil, pVfil, Waux, acumm_H, N, NnP, M, K, my_rank, nProcs,
			block_N, iBlockN, stepN, pd_W, d_H, d_WH, d_Vfil, d_Aux, d_accum, Kp, Mp, streams, pstream_NMF_N, events,
			div_timing, mul_div_timing, upload_Vfil_timing );


		// Downloads (asynchronously) the updated block from d_W.
		#ifdef GPU
			download_matrix( pW, size_pW, my_rank, pd_W0, *pstream_NMF_N, events, download_W_timing );
		#endif


	} // for( i=0 ; i<niters-1 ; i++ )


	// ------------------------


	// Last iteration
	{

		// Synchronizes matrix W among all processes.
		sync_matrix( my_rank, W, bN, size_pW, NnPv, bNv, comm_time, *pstream_NMF_N, events );

		// Uploads matrix W (full size)
		#ifdef GPU
			upload_matrix( W, size_W, my_rank, d_W, streams, *pstream_NMF_M, events, upload_W_timing );
		#endif

		// Reduces to a vector (full size)
		reduce_matrix( W, acumm_W, N, K, my_rank, d_W, d_Aux, Kp, d_accum, streams, events, reduce_timing );

		// Note: Uploading and reducing full matrix W avoids a new synchronization point.


		/*******************************************/
		/*** H = H .* (W'*(V./(W*H))) ./ accum_W ***/
		/*******************************************/

		/* WH = W*H
		 * WH = V ./ WH
		 * Haux = (W'* {V./(WH)}
		 * H = H .* (Haux) ./ accum_W
		 */
		getH( W, Htras, WHcol, pVcol, Haux, acumm_W, N, M, MnP, MnPp, K, my_rank, nProcs,
			block_M, iBlockM, stepM, offset_Vcol, Kp, d_W, pd_H, d_WH, d_Vcol, d_Aux, d_accum,
			streams, pstream_NMF_M, events, div_timing, mul_div_timing, upload_Vcol_timing );


		// GPU: Downloads (asynchronously) the updated block from d_H.
		#ifdef GPU
			download_matrix( pHtras, size_pHtras, my_rank, pd_H0, *pstream_NMF_M, events, download_H_timing );
		#endif


		// Synchronizes matrix H among all processes.
		sync_matrix( my_rank, Htras, bM, size_pHtras, MnPv, bMv, comm_time, *pstream_NMF_M, events );


		// --------------------------------


		// NOTE: From here, uses *pstream_NMF_N:


		// Uploads synchronized matrix H (full size).
		#ifdef GPU
			upload_matrix( Htras, size_Htras, my_rank, d_H, streams, *pstream_NMF_N, events, upload_H_timing );
		#endif


		// Reduces matrix H (full size) to a vector
		reduce_matrix( Htras, acumm_H, M, K, my_rank, d_H, d_Aux, Kp, d_accum, streams, events, reduce_timing );

		// Note: Uploading and reducing full matrix H avoids a new synchronization point.


		//////////////////////////////////////////////////////////////


		/*******************************************/
		/*** W = W .* ((V./(W*H))*H') ./ accum_H ***/
		/*******************************************/

		/* WH = W*H
		 * WH = V ./ WH
		 * Waux = (W'*(V./(WH)))
		 * W = W .* Waux ./ accum_H
		 */
		getW( W, Htras, WHfil, pVfil, Waux, acumm_H, N, NnP, M, K, my_rank, nProcs,
			block_N, iBlockN, stepN, pd_W, d_H, d_WH, d_Vfil, d_Aux, d_accum, Kp, Mp, streams, pstream_NMF_N, events,
			div_timing, mul_div_timing, upload_Vfil_timing );


		// Adjusts matrices W and H to avoid underflow (if selected).
		if ( adjust_matrices ) {

			// W: Updated portion.
			adjust_matrix( W, NnP, K, my_rank, pd_W0, Kp, *pstream_NMF_N, events, adjust_timing, classf_timing );

			// H: Full matrix.
			adjust_matrix( Htras, M, K, my_rank, d_H, Kp, streams + STREAM_MATRIX, events, adjust_timing, classf_timing );

		} // if adjust matrices


		// Downloads (asynchronously) the updated block from d_W.
		#ifdef GPU
			download_matrix( pW, size_pW, my_rank, pd_W0, *pstream_NMF_N, events, download_W_timing );
		#endif

	} // Last iteration


} // nmf_loop

///////////////////////////////////////////////////////////////////////

/*
 * Performs up to niter iterations. Performs test of convergence each niter_test_conv iterations (niter > niter_test_conv).
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
	timing_data_t *restrict classf_timing )
{

	int inc = 0;	// Number of iterations without changes.

	#ifdef _PROFILING_CONV
		double time0 = 0;
		double time1 = 0;
	#endif


	// -------------------------------


	// Number of iterations to perform.

	div_t niter_div = div( niter , niter_test_conv );
	int niter_conv = niter_div.quot;	// Number of times to perform test of convergence.
	int niter_rem = niter_div.rem;		// Remaining iterations.

	#ifdef _VERBOSE
		if (!my_rank)
			printf("NITER_TEST_CONV=%i, niter_conv=%i, niter_rem=%i\n",niter_test_conv, niter_div.quot, niter_div.rem);
	#endif


	// ------------------------


	// Performs all <niter> iterations in <niter_conv> groups of <niter_test_conv> iterations each.
	int iter=0;
	for ( iter=0 ; iter<niter_conv ; iter++ ) {

		// Runs NMF for <niter_test_conv> iterations...

		nmf_loop( niter_test_conv, pVfil, pVcol, WHcol, WHfil, W, Htras, Waux, Haux, acumm_W, acumm_H, N, M, K, Kp, NnP, MnP, bN, bM,
			nProcs, my_rank, NnPv, bNv, MnPv, bMv, comm_time, d_Vfil, d_Vcol, d_WH, d_W, pd_W, d_H, pd_H, d_Aux, d_accum,
			Mp, MnPp, block_N, block_M, iBlockN, iBlockM, stepN, stepM, offset_Vcol, streams, pstream_NMF_N, pstream_NMF_M,
			events, reduce_timing, div_timing, mul_div_timing, upload_Vfil_timing, upload_Vcol_timing, upload_H_timing,
			upload_W_timing, download_H_timing, download_W_timing, adjust_timing, classf_timing, 1 );	// Adjusts W and H.


		// -------------------------------


		/*********************
		 * Test of Convergence.
		 *********************/

		#ifdef _PROFILING_CONV
			MPI_Barrier(MPI_COMM_WORLD);
			time0 = MPI_Wtime();
		#endif

			// Computes a new classification vector from matrix H.
			get_classification( Htras, classification, M, K, Mp, my_rank, d_H, d_classification, d_Aux, Kp,
					streams, events, idx_max_timing, download_classf_timing );


			// Computes differences between classification and last_classification.
			unsigned int diff = get_difference( classification, last_classification, M );


			// Saves the new classification (it just swaps their pointers).
			{
				int *tmp = classification;
				classification = last_classification;
				last_classification = tmp;
			}

		#ifdef _PROFILING_CONV
			MPI_Barrier(MPI_COMM_WORLD);
			classf_timing->time += ( MPI_Wtime() - time1 );
			classf_timing->counter++;
		#endif

		/* Stops if Connectivity matrix (actually, classification vector)
		 * has not changed over last <stop_threshold> iterations.
		 */

		if ( diff )
			inc=0;	// Has changed. Restarts counter.

		// Increments the counter.
		else if ( inc < stop_threshold )
			inc++;

		// Algorithm has converged.
		else
			break;

	} // for  ( niter / niter_test_conv ) times


	// ---------------------------------------------------------


	// Remaining iterations (if NMF has not converged yet).

	if ( (iter == niter_conv ) && niter_rem ) {

		#ifdef _VERBOSE
			if (!my_rank) printf("Performing remaining iterations (%i)...\n",niter_rem);
		#endif

		// Runs NMF for <niter_rem> iterations...
		nmf_loop( niter_rem, pVfil, pVcol, WHcol, WHfil, W, Htras, Waux, Haux, acumm_W, acumm_H, N, M, K, Kp, NnP, MnP, bN, bM,
			nProcs, my_rank, NnPv, bNv, MnPv, bMv, comm_time, d_Vfil, d_Vcol, d_WH, d_W, pd_W, d_H, pd_H, d_Aux,
			d_accum, Mp, MnPp, block_N, block_M, iBlockN, iBlockM, stepN, stepM, offset_Vcol, streams, pstream_NMF_N,
			pstream_NMF_M, events, reduce_timing, div_timing, mul_div_timing, upload_Vfil_timing, upload_Vcol_timing,
			upload_H_timing, upload_W_timing, download_H_timing, download_W_timing, NULL, NULL, 0 ); // Do NOT adjusts W and H.

		// ---------------------------

		// Collects matrix W from all processes.
		collect_from_slaves( my_rank, W, bN, NnP*Kp, NnPv, bNv, comm_time, *pstream_NMF_N, events );

	} else { // No remaining iterations.

		if ( my_rank ) // Slaves: Send last matrix W to master.
			collect_from_slaves( my_rank, W, bN, NnP*Kp, NnPv, bNv, comm_time, *pstream_NMF_N, events );

		else {	// Master:

			#ifdef GPU
				// First, downloads last modified (i.e.,adjusted) matrix H (full size).
				download_matrix( &Htras[0][0], M*Kp, 0, d_H, streams + STREAM_MATRIX, events, download_H_timing );
			#endif

			// Meanwhile, collects last matrix W from all processes.
			collect_from_slaves( my_rank, W, bN, NnP*Kp, NnPv, bNv, comm_time, *pstream_NMF_N, events );

			#ifdef GPU
				// Finally, waits until matrix Htras has been downloaded.
				sync_GPU( 0, streams + STREAM_MATRIX, events );
			#endif
		}

	} // if ( (iter == niter_conv ) && niter_rem )

	// --------------------------------

	// Number of iterations elapsed.
	if ( iter < niter_conv )	// Converged.
		iter = ( iter + 1 ) * niter_test_conv;
	else				// Limit was reached without convergence.
		iter = niter;

	#if defined(_PROFILING_CONV) || defined(_VERBOSE) || defined(_VERBOSE_2) || defined(_DEBUG_NMF)
		if (! my_rank) { fflush(NULL); printf("Finished on %i iterations.\n",iter); }
	#endif


	// -------------------------

	// Returns the number of iterations performed.

	return iter;

} // nmf

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////


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
real get_ref_distance_MPI( real *restrict *Vrow, int NnP, int Mp, int my_rank, size_t maxSize, timing_data_t *restrict comm_time )
{

	real ref = 0;

	// Each process computes a portion of ref_distance()
	real l_ref = get_ref_distance( Vrow, NnP, Mp, my_rank, maxSize );	// Local dot_V

	#if defined(_CHECK_RESULT) || defined(_VERBOSE) || defined(_VERBOSE_2) || defined(_DEBUG_NMF) || defined(_DEBUG_NMF_REDUCT)
		fflush(NULL);
		printf("P[%i]: get_ref_distance_MPI: local ref_distance=%.10g\n",my_rank,l_ref);
	#endif

	// -------------------------

	//  Combines values from all processes and distributes the result back to all processes.

	#ifdef _PROFILING_1
		MPI_Barrier( MPI_COMM_WORLD );
		double time_0 = MPI_Wtime();
	#endif

		MPI_Allreduce( &l_ref, &ref, 1, MPI_TREAL, MPI_SUM, MPI_COMM_WORLD );		// MPICH1

	#ifdef _PROFILING_1
		MPI_Barrier( MPI_COMM_WORLD );
		double time_1 = MPI_Wtime();
		comm_time->time += (time_1 - time_0);
		comm_time->counter++;
	#endif

	// --------------------------------------

	return ref;

} // get_ref_distance_MPI

///////////////////////////////////////////////////////////////////////////////////////////

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
			cudaEvent_t *restrict events, timing_data_t *restrict sub_timing, timing_data_t *restrict upload_Vfil_timing )
{

#ifdef _VERBOSE
printf("Process %i: computing distance...\n",my_rank);
#endif

		real distance = HUGE_VALR;

		// Each processor computes: dot_VWH <-- dot_product( V-W*H, V-W*H )
		real l_distance_VWH = get_distance_VWH( W, Htras, WHfil, Vfil, N, NnP, M, K, my_rank, nProcs,
							block_N, iBlockN, stepN, pd_W, d_H, d_WH, d_Vfil, Kp, Mp,
							streams, pstream_NMF_N, events, sub_timing, upload_Vfil_timing );

		#if defined(_CHECK_RESULT) || defined(_VERBOSE) || defined(_VERBOSE_2) || defined(_DEBUG_NMF) || defined(_DEBUG_NMF_REDUCT)
			printf("\tP[%i]: local distance_VWH: %.10g\n",my_rank,l_distance_VWH);
		#endif

		// -------------------------------

		//  Combines values from all processes and distributes the result back to all processes.

		#ifdef _PROFILING_1
			MPI_Barrier( MPI_COMM_WORLD );
			double time_0 = MPI_Wtime();
		#endif

			real distance_VWH = HUGE_VALR;
			MPI_Allreduce( &l_distance_VWH, &distance_VWH, 1, MPI_TREAL, MPI_SUM, MPI_COMM_WORLD );	// MPICH1

		#ifdef _PROFILING_1
			MPI_Barrier( MPI_COMM_WORLD );
			double time_1 = MPI_Wtime();
			comm_time->time += (time_1 - time_0);
			comm_time->counter++;
		#endif

		#if defined(_CHECK_RESULT) || defined(_VERBOSE) || defined(_VERBOSE_2) || defined(_DEBUG_NMF) || defined(_DEBUG_NMF_REDUCT)
			if ( ! my_rank ) printf("\tglobal distance_VWH: %.10g\n",distance_VWH);
		#endif

		// --------------------------

		// Final distance:
		distance = distance_VWH / ref_distance;

#ifdef _VERBOSE
	printf("Process %i: computing distance... Done (distance=%.10g).\n",my_rank,distance);
#endif

	return distance;

} // get_distance_MPI


///////////////////////////////////////////////////////////////////////////////////////////

// Writes matrix to filename
// filename is written in ROW-major format
// row_major: 1 if matrix is row-wise.
int write_matrix( real *restrict *matrix, int nrows, int ncols, const char *restrict filename, char row_major )
{

	if ( ! strlen( filename ) )
		return EXIT_SUCCESS;

	#ifdef _VERBOSE
	printf("Writting output matrix to file %s... ", filename);
	fflush(NULL);
	#endif

	FILE *fp = NULL;
	if ( (fp=fopen(filename,"w")) == NULL) {
		int err = errno; fflush(NULL); errno = err;
		perror(filename);
		return EXIT_FAILURE;
	}

	if ( row_major ) { // matrix is in ROW-major format
		int i;
		for ( i=0 ; i<nrows ; i++ ) {
			fprintf(fp,"%.8g",matrix[i][0]);
			int j;
			for( j=1 ; j<ncols ; j++ )
				fprintf(fp,"\t%.8g",matrix[i][j]);
			fprintf(fp,"\n");
		}
	} else { // COLUMN-major (file is written in ROW-major format).
		int j;
		for ( j=0 ; j<ncols ; j++ ) {
			fprintf(fp,"%.8g",matrix[0][j]);
			int i;
			for( i=1 ; i<nrows ; i++ )
				fprintf(fp,"\t%.8g",matrix[i][j]);
			fprintf(fp,"\n");
		}
	} // if row_major

	fclose(fp);

	#ifdef _VERBOSE
	printf("done\n");
	fflush(NULL);
	#endif

	return EXIT_SUCCESS;

} // write_matrix

///////////////////////////////////////////////////////////////////////////////////////////

/* Writes matrix to filename
 * filename is written in ROW-major format
 * row_major: 1 if matrix is row-wise.
 */
int write_matrix_binary( real *restrict *matrix, int nrows, int ncols, const char *restrict filename )
{

	if ( ! strlen( filename ) )
		return EXIT_SUCCESS;

	#ifdef _VERBOSE
	printf("Writting output matrix to file %s... ", filename);
	fflush(NULL);
	#endif

	FILE *restrict file = fopen( filename, "wb" );
	if( file == NULL) {
		int err = errno; fflush(NULL); errno = err;
		fprintf( stderr, "\nfopen '%s' :\n\t%s.\n", filename, strerror(errno) );
		fprintf(stderr,"Error in write_matrix_binary().\n");
		return EXIT_FAILURE;
	}

	// ----------------------------------

	// Matrix size.
	const int dim[2] = { nrows, ncols };
	size_t nwritten = fwrite( dim, sizeof(int), 2, file );
	if ( nwritten != 2 ) {
		fflush(NULL);
		fprintf(stderr,"\nInternal error in fwrite writing matrix dimensions.\n"
			"Error in write_matrix_binary().\n");
		fclose(file);
		return EXIT_FAILURE;
	}


	// Data matrix
	int nitems = nrows * ncols;
	nwritten = fwrite( &matrix[0][0], sizeof(real), nitems, file );
	if ( nwritten != nitems ) {
		fflush(NULL);
		fprintf(stderr,"\nInternal error in function fwrite: %zu items read, %i expected\n"
			"Error in write_matrix_binary().\n",nwritten,nitems);
		fclose(file);
		return EXIT_FAILURE;
	}

	fclose(file);

	#ifdef _VERBOSE
		printf("done\n");
		fflush(NULL);
	#endif

	return EXIT_SUCCESS;

} // write_matrix_binary

///////////////////////////////////////////////////////////////////////////////////////////

// Writes an int file.
int write_binary_int( int *restrict array, int nitems, const char *restrict filename )
{

	if ( ! strlen( filename ) )
		return EXIT_SUCCESS;

	#ifdef _VERBOSE
		printf("Writting output matrix to file %s... ", filename);
		fflush(NULL);
	#endif

	FILE *file = NULL;
	if ( (file=fopen(filename,"w")) == NULL) {
		int err = errno; fflush(NULL); errno = err;
		perror(filename);
		return EXIT_FAILURE;
	}

	// Data matrix
	size_t nwritten = fwrite( array, sizeof(int), nitems, file );
	if ( nwritten != nitems ) {
		fflush(NULL);
		fprintf(stderr,"\nInternal error in fwrite().\nError in write_binary_int().\n");
		fclose(file);
		return EXIT_FAILURE;
	}

	fclose(file);

	#ifdef _VERBOSE
		printf("done\n");
		fflush(NULL);
	#endif

	return EXIT_SUCCESS;

} // write_matrix_binary_int

///////////////////////////////////////////////////////////////////////////////////////////

/*
 * Sets CPU affinity.
 * Process p will be executed in CPU p*(num_cpus/nProcs).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int set_affinity( int my_rank, int nProcs, int num_cpus )
{

#ifdef _VERBOSE
fflush(NULL);
printf("Process %i: set_affinity(%i,%i,%i)\n",my_rank,my_rank,nProcs,num_cpus);
#endif

	cpu_set_t *cpusetp = CPU_ALLOC(num_cpus);
	if ( cpusetp == NULL ) {
		int err=errno; fflush(NULL); errno=err;
		fprintf( stderr, "Process %i: Error in set_affinity.\nCPU_ALLOC(%i): %s\n",
			my_rank, num_cpus, strerror(errno));
		return EXIT_FAILURE;
	}
	size_t size = CPU_ALLOC_SIZE(num_cpus);

	// --------------

	// Sets CPU affinity
	int cpu = my_rank * ( num_cpus / nProcs );	// nProcs <= num_cpus

	CPU_ZERO_S(size, cpusetp);
	CPU_SET_S(cpu, size, cpusetp);

	if ( sched_setaffinity( 0, size, cpusetp ) ) {
		int err=errno; fflush(NULL); errno=err;
		fprintf( stderr, "Process %i: Error in set_affinity. sched_setaffinity: %s\n",
			my_rank, strerror(errno));
		CPU_FREE(cpusetp);
		return EXIT_FAILURE;
	}

	CPU_FREE(cpusetp);

#if defined(_PROFILING) || defined(_PROFILING_1) || defined(_PROFILING_2) || defined(_PROFILING_3) || defined(_VERBOSE)
printf("Process %i: Affinity set to %i\n",my_rank,cpu);
#endif

#ifdef _VERBOSE
fflush(NULL);
#endif

	return EXIT_SUCCESS;

} // set_affinity

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

int main( int argc, char *argv[] )
{
	int my_rank = 0, nProcs = 1;

	real **W = NULL, **Htras = NULL, **Vfil = NULL, **Vcol = NULL;

	/* Auxiliares para el computo */
	real **WHfil = NULL, **WHcol = NULL, **Haux = NULL, **Waux = NULL, *acumm_W = NULL, *acumm_H = NULL;
	int *classification = NULL, *last_classification = NULL;

    /* Auxiliares para GPU */
	real *d_Vfil = NULL, *d_Vcol = NULL, *d_WH = NULL, *d_W = NULL, *d_H = NULL, *d_Aux = NULL, *d_accum = NULL;
	real *pd_W = NULL, *pd_H = NULL;	// GPU pointers
	int *d_classification = NULL;

	// Pointers to matrices Vfil, Vcol. WARNING: Modified by GPU.
	real **pVcol = NULL, **pVfil = NULL;
	int offset_Vcol = 0;		// Offset from pVcol (GPU only).


	int N = 0, M = 0, K = 0, NnP = 0, MnP = 0, bN = 0, bM = 0, Kp = 0, Mp = 0, MnPp = 0;


	// Block configuration for GPU.
	block_t block_N, block_M;		// Block configuration.
	int stepN=1, stepM=1;			// +1 (forward), -1 (backward)
	int iBlockN=0, iBlockM=0;		// Index in block_N and block_M arrays

	// Pointers to main stream. One for each dimension (GPU only).
	cudaStream_t *pstream_NMF_N = NULL;
	cudaStream_t *pstream_NMF_M = NULL;


	#ifdef _PROFILING
		double elapsed_time = 0.0;
		double time0 = 0.0, time1 = 0.0;
	#endif

	timing_data_t comm_time;
	#ifdef _PROFILING_1
		comm_time.time = 0.0;
		comm_time.counter = 0;
	#endif

	#ifdef _PROFILING_DIST
		double ref_dist_time = 0.0;
		timing_data_t dist_time;
		dist_time.time = 0.0;
		dist_time.counter = 0;
		double time_d = 0.0;
	#endif


	// Number of iterations performed.
	int iter_performed = 0;

	// Distance for best W and H.
	real min_distance = HUGE_VALR;

	// Best run
	int best_run = 0;


	/* If NnP and MnP have not the same value on all processes,
	 * stores in these vectors all NnP, bN, MnP, and bM values for all processes.
	 * (i.e.,all processes will have all NnP, bN,... values for all processes).
	 */
	int *NnPv = NULL, *bNv = NULL;
	int *MnPv = NULL, *bMv = NULL;


	int status = EXIT_SUCCESS;

	/* Init MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

#ifndef MANDATORY_ARGS
	#define MANDATORY_ARGS 8
#endif
	if ( argc < MANDATORY_ARGS ) {
		fflush(NULL);
		fprintf(stderr,"%s num_cpus filename K maxiter nRuns niter_conv stop_threshold [ write_outputs [rows columns] ]\n",argv[0]);
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	// Sets CPU affinity
	int num_cpus = atoi(argv[1]);
	if ( set_affinity( my_rank, nProcs, num_cpus ) == EXIT_FAILURE ) {
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	const char *filename = argv[2];
	K = atoi(argv[3]);
	int numIter = atoi(argv[4]);
	int nRuns = atoi(argv[5]);
	int niter_test_conv = atoi(argv[6]);
	int stop_threshold = atoi(argv[7]);
	int write_outputs = 0;

	if ( argc > MANDATORY_ARGS ) {
		write_outputs = atoi(argv[MANDATORY_ARGS]);
		// Uses optional arguments for matrix dimensions.
		if ( argc > MANDATORY_ARGS + 2 ) {
			N = atoi(argv[MANDATORY_ARGS+1]);
			M = atoi(argv[MANDATORY_ARGS+2]);
		}
	}


	// Filename for output files.
	char filename_out[ FILENAME_MAX ];


	#if defined(_VERBOSE) || defined(_VERBOSE_2) || defined(_DEBUG_NMF) || defined(_DEBUG_NMF_REDUCT)
		setbuf( stdout, NULL );
	#endif

	// Needs to test convergence
	int need_conv = ( numIter >= niter_test_conv );

	// ----------------------------------

	// Initializes GPU

	size_t free_mem = 0;	// Available memory.

	#ifdef GPU
		#ifdef _TEST_BLOCKS_2 /* Testing blocks */
			free_mem = 4294770688;	// GPU Memory in bytes
			fflush(NULL);
			if ( ! my_rank ) {
				printf("\"Available\" memory: %0.2f Mbytes.\n", free_mem / 1048576.0 );
				fflush(NULL);
			}

		#else
			if ( init_GPU( my_rank, &free_mem ) == EXIT_FAILURE ) {
				fflush(NULL);
				fprintf(stderr, "\nProcess %i: Failed to initialize GPU Device.\n", my_rank);
				MPI_Finalize();
				return EXIT_FAILURE;
			}
		#endif

		// NOTE: Expresses available memory in number of sizeof(<data-type>) (real or double).
		free_mem /= sizeof(real);

	#endif

	// ----------------------------------

	// Reads input matrix

	#ifdef _TEST_BLOCKS_2 /* Testing blocks */

		dimension_per_CPU( N, M, my_rank, nProcs, &NnP, &MnP, &bN, &bM );

	#else /* Default case. */

		if ( init_V( filename, &Vfil, &Vcol, &N, &M, &NnP, &MnP, &bN, &bM, nProcs, my_rank ) == EXIT_FAILURE ) {
			fflush(NULL);
			#ifdef GPU
				finalize_GPU( my_rank );
			#endif
			MPI_Finalize();
			return EXIT_FAILURE;
		}
	#endif


	// -----------------------------

	// Data dimensions

	Kp = get_padding( K );
	Mp = get_padding( M );
	MnPp = get_padding( MnP );

	#if defined(_PROFILING) || defined(_PROFILING_1) || defined(_PROFILING_2) || defined(_PROFILING_3) \
		|| defined(_PROFILING_CONV) || defined(_VERBOSE)
		if ( ! my_rank )
			printf("\nN=%i , M=%i (Mp=%i), K=%i (Kp=%i), It=%i, niter_test_conv=%i, nRuns=%i, nProcs=%i\n",
				N,M,Mp,K,Kp,numIter,niter_test_conv,nRuns,nProcs);
		fflush( NULL );
		printf( "Process %i: NnP=%i MnP=%i (MnPp=%i) bN=%i, bM=%i\n", my_rank, NnP, MnP, MnPp, bN, bM );
		fflush( NULL );
	#endif

	// -----------------------------

	// Reference for matrix 'distance'

	#ifndef _TEST_BLOCKS_2

		#ifdef _PROFILING_DIST
			MPI_Barrier(MPI_COMM_WORLD);
			time_d = MPI_Wtime();
		#endif
			real ref_distance = get_ref_distance_MPI( Vfil, NnP, Mp, my_rank, free_mem, &comm_time );
		#ifdef _PROFILING_DIST
			MPI_Barrier(MPI_COMM_WORLD);
			ref_dist_time = (MPI_Wtime() - time_d);
		#endif

		if ( ref_distance == 0 ) {
			fflush(NULL); fprintf(stderr, "[P%i]: Failed to compute ref_distance\n\n\n", my_rank); fflush(NULL);
			destroy_memory2D(Vfil); if(Vfil!=Vcol){destroy_memory2D(Vcol);}
			#ifdef GPU
				finalize_GPU( my_rank );
			#endif
			MPI_Finalize();
			return EXIT_FAILURE;
		}
		#if defined(_CHECK_RESULT) || defined(_VERBOSE) || defined(_VERBOSE_2) || defined(_DEBUG_NMF)
			if ( ! my_rank ) {
				fflush(NULL);
				printf("\nGlobal Reference Distance: %.10g (old_style=%g)\n\n",ref_distance,sqrt((double)ref_distance));
			}
		#endif
	#endif

	// -----------------------------

#ifdef GPU
	cudaStream_t *streams = NULL;
	int num_streams = 0;			// Number of streams 'stream_NMF' (in addition to STREAM_NMF).
	cudaEvent_t events[ NUM_EVENTS ];

	/* Init GPU device */
	status = init_GPUdevice( N, M, NnP, MnP, Mp, Kp, MnPp, my_rank, nProcs, free_mem, need_conv, &block_N, &block_M,
				&d_Vfil, &d_Vcol, &d_WH, &d_H, &d_W, &d_Aux, &d_accum, &d_classification, &streams, &num_streams, events );
	if ( status == EXIT_FAILURE ) {
		#ifndef _TEST_BLOCKS_2
			fprintf(stderr, "GPU init FAILURE of thread=%i!!!\n\n\n", my_rank);
			fflush(NULL);
			destroy_memory2D(Vfil); if(Vfil!=Vcol){destroy_memory2D(Vcol);}
			finalize_GPU( my_rank );
		#endif
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	// Sets initial pointers.
	pd_W = (d_W + bN * Kp);
	pd_H = (d_H + bM * Kp);

	// Trick to fit pointer types
	real *pVc = &Vcol[0][0];
	pVcol = &pVc;
	real *pVr = &Vfil[0][0];
	pVfil = &pVr;

	// Pointers to main stream. One for each dimension.
	pstream_NMF_N = streams + STREAM_NMF;
	pstream_NMF_M = streams + STREAM_NMF;


#else /* NOT GPU */

	#ifdef _TEST_BLOCKS_2
		MPI_Finalize();
		return EXIT_FAILURE;
	#endif


	// Don't wastes memory.
	cudaStream_t streams;
	cudaEvent_t events;

	// Matrix WH. NOTE: Use destroy_memory1D() with WHcol and destroy_memory2D() with WHfil in both cases.
	if (NnP*M>MnP*N) {
		WHfil = get_memory2D(NnP, M);
		WHcol = get_memory2D_renew(N, MnP, WHfil[0]);
	} else {
		WHcol = get_memory2D(N, MnP);
		WHfil = get_memory2D_renew(NnP, M, WHcol[0]);
	}

	Haux  = get_memory2D(M, K);
	Waux  = get_memory2D(N, K);

	acumm_W  = get_memory1D(K);
	acumm_H  = get_memory1D(K);

	if ( !( WHfil && WHcol && Haux && Waux && accum_W && accum_H ) ) {
		fprintf(stderr,"Process %i: Error allocating memory (WH || Haux || Waux || accums ).\n",my_rank);
		fflush(NULL);
		destroy_memory2D( Vfil ); if ( Vfil != Vcol ){destroy_memory2D( Vcol );}
		destroy_memory2D( WHfil ); destroy_memory1D( WHcol ); // always shared
		destroy_memory2D( Haux ); destroy_memory2D( Waux ); destroy_memory1D( accum_H ); destroy_memory1D( accum_W );
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	// Sets initial pointers. Pointers used on GPU.
	pVcol = Vcol;
	pVfil = Vfil;

#endif

	// WARNING: W and Htras with *padding*.

	W   = get_memory2D(N, Kp);
	Htras  = get_memory2D(M, Kp);

	if( !( W && Htras ) ) {
		fprintf(stderr,"Process %i: Error allocating memory (W || Htras).\n",my_rank);
		fflush(NULL);
		destroy_memory2D( Htras ); destroy_memory2D( W ); destroy_memory2D( Vfil );
		if ( Vfil != Vcol ){destroy_memory2D( Vcol );}
		#ifdef GPU
			finalize_GPUdevice( d_Vfil, d_Vcol, d_WH, d_H, d_W, d_Aux, d_accum, d_classification,
						my_rank, streams, num_streams, events );
		#else
			destroy_memory2D( WHfil ); destroy_memory1D( WHcol ); // always shared
			destroy_memory2D( Haux ); destroy_memory2D( Waux );
			destroy_memory1D( accum_H ); destroy_memory1D( accum_W );
		#endif
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	// --------------------------------------

	// Vectors for test of convergence.
	if ( need_conv ) {

		if ( init_classf_data( Mp, my_rank, &classification, &last_classification ) == EXIT_FAILURE ) {
			fflush(NULL);
			destroy_memory2D( Htras ); destroy_memory2D( W ); destroy_memory2D( Vfil );
			if ( Vfil != Vcol ){destroy_memory2D( Vcol );}
			#ifdef GPU
				finalize_GPUdevice( d_Vfil, d_Vcol, d_WH, d_H, d_W, d_Aux, d_accum, d_classification,
						my_rank, streams, num_streams, events );
			#else
				destroy_memory2D( WHfil ); destroy_memory1D( WHcol ); // always shared
				destroy_memory2D( Haux ); destroy_memory2D( Waux );
				destroy_memory1D( accum_H ); destroy_memory1D( accum_W );
			#endif
			MPI_Finalize();
			return EXIT_FAILURE;
		}

	} // if need to perform a convergence test.

	// --------------------------------------


	// Initializes vectors for MPI communications

	if ( N % nProcs ) { // NnP has a different value on each process in the group.

		#ifdef _VERBOSE
		if ( ! my_rank )
			printf("\tNot same NnP. Initializing vectors for dimension N...\n");
		#endif

		// Allocates memory for dimension vector.
		if ( allocate_dim_vectors( nProcs, &NnPv, &bNv ) == EXIT_FAILURE ) {
			fprintf(stderr,"Process %i: Error initializing dimension vectors (group N). "
				"Aborting...\n",my_rank);
			fflush(NULL);
			destroy_memory1D_int(last_classification); destroy_memory1D_int(classification);
			destroy_memory2D( Htras ); destroy_memory2D( W ); destroy_memory2D( Vfil );
			if ( Vfil != Vcol ){destroy_memory2D( Vcol );}
			#ifdef GPU
				finalize_GPUdevice( d_Vfil, d_Vcol, d_WH, d_H, d_W, d_Aux, d_accum, d_classification,
							my_rank, streams, num_streams, events );
			#else
				destroy_memory2D( WHfil ); destroy_memory1D( WHcol ); // always shared
				destroy_memory2D( Haux ); destroy_memory2D( Waux );
				destroy_memory1D( accum_H ); destroy_memory1D( accum_W );
			#endif
			MPI_Finalize();
			return EXIT_FAILURE;
		}

		// Initializes vectors
		init_dim_vectors( N, Kp, nProcs, NnPv, bNv );	// Kp (padding)

	} // if NnP has a different value on each process in the group.

	if ( M % nProcs ) { // MnP has a different value on each process in the group.

		#ifdef _VERBOSE
		if ( ! my_rank )
			printf("\tNot same MnP. Initializing vectors for dimension M...\n");
		#endif

		// Allocates memory for dimension vector.
		if ( allocate_dim_vectors( nProcs, &MnPv, &bMv ) == EXIT_FAILURE ) {
			fprintf(stderr,"Process %i: Error initializing dimension vectors (group M). "
				"Aborting...\n",my_rank);
			fflush(NULL);
			if (bNv != NULL) { free(bNv); free(NnPv); }
			destroy_memory1D_int(last_classification); destroy_memory1D_int(classification);
			destroy_memory2D( Htras ); destroy_memory2D( W ); destroy_memory2D( Vfil );
			if ( Vfil != Vcol ){destroy_memory2D( Vcol );}
			#ifdef GPU
				finalize_GPUdevice( d_Vfil, d_Vcol, d_WH, d_H, d_W, d_Aux, d_accum, d_classification,
							my_rank, streams, num_streams, events );
			#else
				destroy_memory2D( WHfil ); destroy_memory1D( WHcol ); // always shared
				destroy_memory2D( Haux ); destroy_memory2D( Waux );
				destroy_memory1D( accum_H ); destroy_memory1D( accum_W );
			#endif
			MPI_Finalize();
			return EXIT_FAILURE;
		}

		// Initializes vectors
		init_dim_vectors( M, Kp, nProcs, MnPv, bMv );	// Kp (padding)

	} // if MnP has a different value on each process in the group.


	// --------------------------------------


	// Timing data

	timing_data_t reduce_timing[3], div_timing[2], sub_timing[2], idx_max_timing, mul_div_timing, adjust_timing;
	timing_data_t upload_Vfil_timing, upload_Vcol_timing, upload_H_timing, upload_W_timing, download_H_timing, download_W_timing,
			download_classf_timing;
	#ifdef GPU
		#ifdef _PROFILING_3
		{
			int i;
			for( i=0 ; i<3 ; i++ ) {
				reduce_timing[i].time = 0.0;
				reduce_timing[i].counter = 0;
			}
			// for( i=0 ; i<3 ; i++ ) {
				idx_max_timing.time = 0.0;
				idx_max_timing.counter = 0;
			// }
			for( i=0 ; i<2 ; i++ ) {
				div_timing[i].time = 0.0;
				div_timing[i].counter = 0;
			}
			for( i=0 ; i<2 ; i++ ) {
				sub_timing[i].time = 0.0;
				sub_timing[i].counter = 0;
			}
			mul_div_timing.time = 0.0;
			mul_div_timing.counter = 0;
			adjust_timing.time = 0.0;
			adjust_timing.counter = 0;
		}
		#endif
		#ifdef _PROFILING_2
			upload_Vfil_timing.time = 0.0;
			upload_Vfil_timing.counter = 0;
			upload_Vcol_timing.time = 0.0;
			upload_Vcol_timing.counter = 0;
			upload_H_timing.time = 0.0;
			upload_H_timing.counter = 0;
			upload_W_timing.time = 0.0;
			upload_W_timing.counter = 0;
			download_H_timing.time = 0.0;
			download_H_timing.counter = 0;
			download_W_timing.time = 0.0;
			download_W_timing.counter = 0;
			download_classf_timing.time = 0.0;
			download_classf_timing.counter = 0.0;
		#endif
	#endif

	timing_data_t classf_timing;
	#ifdef _PROFILING_CONV
		classf_timing.time = 0.0;
		classf_timing.counter = 0;
	#endif


	///////////////////////////////////////////////////////

	// Starts NMF
	if (! my_rank)
		printf("\nAll: niter=%i, niter_test_conv=%i, stop_threshold=%i, nRuns=%i\n",
			numIter,niter_test_conv,stop_threshold, nRuns);
	fflush(NULL);

	// -----------------------------------------------


	#ifdef _PROFILING
		MPI_Barrier(MPI_COMM_WORLD);
		time0 = MPI_Wtime();
	#endif

	// GPU: Uploads initial matrices.
	#ifdef GPU
		// Uploads initial Vcol if it is NOT shared (shared iff 1 processor <AND> V and WH fit into GPU memory).
		if ( d_Vcol != d_Vfil )
			CPU2GPU_Vcol( pVc, N, block_M.BL[0], block_M.BLp[0], MnPp, 0, my_rank, d_Vcol, streams, events, &upload_Vcol_timing );

		// Initial Vfil.
		CPU2GPU_Vfil( pVr, block_N.BL[0], M, Mp, my_rank, d_Vfil, streams, events, &upload_Vfil_timing );
	#endif

	// -------------------------------

	int run;
	for ( run=0 ; run<nRuns ; run++ ) {

		#ifdef _VERBOSE
			if (!my_rank) { printf("\tRun %i\n", run); fflush(NULL); }
		#endif


		// Initializes matrices W and H.
		#ifdef _MY_INIT
			srand(4);
			init_matrix(&Htras[0][0],M,K,Kp,0);
			srand(3);
			init_matrix(&W[0][0],N,K,Kp,0);
		#else
			// TODO: Master select seed and broadcastes to slaves. Each slave initializes full W and H.

			srand(time(NULL));   // Starts the random seed
			init_matrix(&Htras[bM][0],MnP,K,Kp,EPS);
			init_matrix(&W[bN][0],NnP,K,Kp,EPS);
		#endif

		#if defined(_DEBUG_NMF) || defined(_VERBOSE_2)
			if (! my_rank) {
				printf( "\n\nInitial W:\n" );
				show_matrix( W, N, K, 1 );	// Row major
				printf( "\n\nInitial H:\n" );
				show_matrix( Htras, M, K, 0 );	// Col major
			}
		#endif

		// -----------------------

		// GPU: Uploads initial matrix H (portion).
		#ifdef GPU
			upload_matrix_H( &Htras[bM][0], MnP, K, Kp, my_rank, pd_H, streams, events, &upload_H_timing );
		#endif

		// -----------------------

		iter_performed += nmf( numIter, niter_test_conv, stop_threshold, pVfil, pVcol, WHcol, WHfil, W, Htras, Waux, Haux, acumm_W,
				acumm_H, classification, last_classification, N, M, K, Kp, NnP, MnP, bN, bM, nProcs, my_rank, NnPv, bNv, MnPv,
				bMv, &comm_time, d_Vfil, d_Vcol, d_WH, d_W, &pd_W, d_H, &pd_H, d_Aux, d_accum, d_classification,
				Mp, MnPp, &block_N, &block_M, &iBlockN, &iBlockM, &stepN, &stepM, &offset_Vcol, streams, &pstream_NMF_N,
				&pstream_NMF_M, events, reduce_timing, div_timing, &mul_div_timing, &upload_Vfil_timing, &upload_Vcol_timing,
				&upload_H_timing, &upload_W_timing, &download_H_timing, &download_W_timing, &adjust_timing, &idx_max_timing,
				&download_classf_timing, &classf_timing );

		// -----------------------

		#if defined(GPU) && (defined(_VERBOSE)|| defined(_VERBOSE_2) || defined(_DEBUG_NMF))
			// Checks status:
			if ( check_cuda_status( my_rank, "in NMF()" ) == EXIT_FAILURE ) {
				if (bNv != NULL) { free(bNv); free(NnPv); bNv = NULL ; NnPv = NULL; }
				if (bMv != NULL) { free(bMv); free(MnPv); bMv = NULL ; MnPv = NULL; }
				destroy_memory1D_int(last_classification);
				destroy_memory1D_int(classification);
				if(Vfil!=Vcol){destroy_memory2D(Vcol);} destroy_memory2D( Vfil );
				destroy_memory2D( Htras ); destroy_memory2D( W );
				finalize_GPUdevice( d_Vfil, d_Vcol, d_WH, d_H, d_W, d_Aux, d_accum, d_classification,
							my_rank, streams, num_streams, events );
				MPI_Finalize();
				return EXIT_FAILURE;
			}
		#endif

		// -----------------------

		// Computes Distance

		#ifdef _PROFILING_DIST
			MPI_Barrier(MPI_COMM_WORLD);
			time_d = MPI_Wtime();
		#endif

		real distance = get_distance_MPI( ref_distance, pVfil, WHfil, W, Htras, N, M, K, Kp, NnP, bN, nProcs, my_rank, &comm_time,
						d_Vfil, d_WH, &pd_W, d_H, Mp, &block_N, &iBlockN, &stepN, streams,
						&pstream_NMF_N, events, sub_timing, &upload_Vfil_timing );

		#ifdef _PROFILING_DIST
			MPI_Barrier(MPI_COMM_WORLD);
			dist_time.time += (MPI_Wtime()-time_d);
			dist_time.counter++;
		#endif

		#if defined(_CHECK_RESULT) || defined(_VERBOSE) || defined(_VERBOSE_2) || defined(_DEBUG_NMF)
			if ( ! my_rank ) {
				fflush(NULL);
				printf("\t\tGlobal Distance (run %i/%i): %.10g (old_style=%g)\n",run,nRuns,distance,sqrt((double)distance));
			}
			#if defined(_VERBOSE) || defined(_VERBOSE_2) || defined(_DEBUG_NMF)
				if ( distance < 0 ) {	// Error.
					fflush(NULL);
					fprintf(stderr, "P[%i]: Error in distance. Aborting...\n", my_rank);
					fflush(NULL);
					if (bNv != NULL) { free(bNv); free(NnPv); bNv = NULL ; NnPv = NULL; }
					if (bMv != NULL) { free(bMv); free(MnPv); bMv = NULL ; MnPv = NULL; }
					destroy_memory1D_int(last_classification);
					destroy_memory1D_int(classification);
					if(Vfil!=Vcol){destroy_memory2D(Vcol);} destroy_memory2D( Vfil );
					destroy_memory2D( Htras ); destroy_memory2D( W );
					finalize_GPUdevice( d_Vfil, d_Vcol, d_WH, d_H, d_W, d_Aux, d_accum, d_classification,
								my_rank, streams, num_streams, events );
					MPI_Finalize();
					return EXIT_FAILURE;
				}
			#endif
		#endif

		// Selects best W and H.
		if ( distance < min_distance ) {

			min_distance = distance;
			best_run = run;
		}

		// ----------------------------------

		// Writes output matrices
		if ( ! my_rank ) {

			#if defined(_VERBOSE_2) || defined(_DEBUG_NMF)
				printf( "\n---- Resulting W: ---- \n" );
				show_matrix( W, N, K, 1 );		// Row major
				printf( "\n---- Resulting H: ---- \n" );
				show_matrix( Htras, M, K, 0 );	// Col major
			#endif

			if ( write_outputs ) {
				sprintf( filename_out, "%s_W_k_%i_run_%i.dat", filename, K, run );
				write_matrix_binary( W, N, K, filename_out );

				// // TODO: CHECK!

				//sprintf( filename_out, "%s_H_k_%i_run_%i.dat", filename, K, run );
				//write_matrix_binary( Htras, M, K, filename_out );

				//sprintf( filename_out, "%s_classf_k_%i_run_%i.dat", filename, K, run );
				//write_binary_int( classification, K, filename_out );
			}

		} // Master writes output matrices

	} // for nRuns times

	#ifdef _PROFILING
		MPI_Barrier(MPI_COMM_WORLD);
		time1 = MPI_Wtime();
		elapsed_time = (time1 - time0);
	#endif


	#ifdef _VERBOSE
		printf("Process %i: Done NMF.\n", my_rank); fflush(NULL);
	#endif

	///////////////////////////////////////////////////////

	// Frees all resources...

	if (bNv != NULL) { free(bNv); free(NnPv); bNv = NULL ; NnPv = NULL; }
	if (bMv != NULL) { free(bMv); free(MnPv); bMv = NULL ; MnPv = NULL; }
	destroy_memory1D_int(last_classification); destroy_memory1D_int(classification);
	destroy_memory2D( Htras ); destroy_memory2D( W );
	if(Vfil!=Vcol){destroy_memory2D(Vcol);} destroy_memory2D( Vfil );


	#ifdef GPU
		if ( check_cuda_status( my_rank, "in NMF()" ) == EXIT_FAILURE )
			status = EXIT_FAILURE;

		if ( finalize_GPUdevice( d_Vfil, d_Vcol, d_WH, d_H, d_W, d_Aux, d_accum, d_classification,
					my_rank, streams, num_streams, events ) == EXIT_FAILURE )
			status = EXIT_FAILURE;
	#else
		destroy_memory2D( WHfil );
		destroy_memory1D( WHcol ); // Always shared
		destroy_memory2D( Haux ); destroy_memory2D( Waux ); destroy_memory1D( accum_H ); destroy_memory1D( accum_W );
	#endif

	///////////////////////////////////////////////////////////

	if ( ! my_rank ) {

		fflush(NULL);

		printf("\nBest run: %i (distance=%.10g, old_style=%g)\n",best_run,min_distance,sqrt((double)min_distance));

		// -------------------------------------

		// Saves information of best run

		//// Makes a copy of best W and H....	// TODO
		//if ( write_outputs ) {
		//	sprintf( filename_out, "%s_W_k_%i_run_%i.dat", filename, K, run );
		//	sprintf( filename_out, "%s_H_k_%i_run_%i.dat", filename, K, run );
		//}

		// ----------------------------------------

		// Shows timings

		#ifdef GPU
			#ifdef _PROFILING_3
				show_kernel_times( reduce_timing, div_timing, &mul_div_timing, &adjust_timing, &idx_max_timing, sub_timing );
			#endif
			#ifdef _PROFILING_2
				show_transfers_times( N, M, Kp, NnP, MnP, Mp, MnPp, iter_performed, &upload_Vfil_timing, &upload_Vcol_timing,
						&upload_H_timing, &upload_W_timing, &download_H_timing, &download_W_timing,
						&download_classf_timing );
			#endif
		#endif

		// Communication time
		#ifdef _PROFILING_1
			_PRINT_IT_SEC("\n\tComunication time", &comm_time );
		#endif

		// Test of convergence time
		#ifdef _PROFILING_CONV
			if ( numIter >= niter_test_conv )
				_PRINT_IT("\n\tTest-of-convergence time", &classf_timing );
		#endif

		// Distance time
		#ifdef _PROFILING_DIST
		{
			printf("\n\tDistance between V and W*H time:\n\t\tReference-distance time: %.10g ms\n",ref_dist_time*1000);
			_PRINT_IT_SEC("\t\tDistance computing time", &dist_time );
			printf("\t\tTotal distance-computing time: %.10g sec\n", dist_time.time*1000 + ref_dist_time*1000 );

			elapsed_time += ref_dist_time;	// NOTE: Ref-distance time (in seconds) is computed outside the main loop.
		}
		#endif

		#ifdef _PROFILING
			#ifdef GPU
				printf("\n\tEXEC TIME-GPU %.10g sec. nProcs=%i N=%i M=%i K=%i nRuns=%i iter=%i (total it.: %i)"
					" sizeof(real)=%zu bytes\n",
					elapsed_time, nProcs, N, M, K, nRuns, numIter, iter_performed, sizeof(real) );
			#elif ATLAS
				printf("\n\tEXEC TIME-ATLAS %g (s). nProcs=%i N=%i M=%i K=%i nRuns=%i iter=%i (sizeof(real)=%zu bytes)\n",
					elapsed_time, nProcs, N, M, K, nRuns, numIter, sizeof(real) );
			#else
				printf("\n\tEXEC TIME %g (s). nProcs=%i N=%i M=%i K=%i nRuns=%i iter=%i (sizeof(real)=%zu bytes)\n",
					elapsed_time, nProcs, N, M, K, nRuns, numIter, sizeof(real));
			#endif
		#endif

	} // Process 0

    /***********************/

    /* Finalize MPI */

	if ( status == EXIT_FAILURE ) {
		fflush(NULL);
		fprintf(stderr, "P[%i]: Exiting with status %i...\n",my_rank,status);
	}

	fflush(NULL);

	MPI_Finalize();

	return status;
}
