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
 *
 * NMF_mGPU.c
 *	Main program for multi-GPU systems.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Data types:
 *		NMFGPU_MPI: Defines NMFGPU_MPI_REAL_T and NMFGPU_MPI_INDEX_T.
 *
 *	Additional information:
 *		NMFGPU_VERBOSE: Shows some messages concerning the progress of the program, as well as
 *				some configuration parameters.
 *		NMFGPU_VERBOSE_2: Shows the parameters in some routine calls.
 *
 *	CPU timing:
 *		NMFGPU_PROFILING_GLOBAL: Computes total elapsed time. If GPU time is NOT being computed,
 *					the CPU thread performs active waiting (i.e., it spins) on
 *					synchronization calls such as cudaDeviceSynchronize() or
 *					cudaStreamSynchronize(). Otherwise, the CPU thread is blocked.
 *
 *		NMFGPU_PROFILING_COMM:	Computes time elapsed on communication options among processes
 *					(MPI-version only).
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
 *********************************************************/

#if (! NMFGPU_MPI)
	#define NMFGPU_MPI (1)	/* NMFGPU_MPI_REAL_T, NMFGPU_MPI_INDEX_T on real_type.h */
#endif

#include "NMF_routines.h"
#include "matrix_operations.h"
#include "GPU_setup.h"
#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_COMM
	#include "timing.h"
#endif
#include "matrix_io/matrix_io.h"
#include "matrix_io/matrix_io_routines.h"
#include "common.h"
#include "index_type.h"
#include "real_type.h"

#include <cuda_runtime_api.h>
#include <mpi.h>

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>	/* uintptr_t */
#include <stdbool.h>
#include <stdlib.h>

////////////////////////////////////////////////
////////////////////////////////////////////////

/* Global variables */

// Number of processes for each dimension
static index_t num_processes_N = 0;	// <= MIN( num_processes, N )
static index_t num_processes_M = 0;	// <= MIN( num_processes, (Mp/memory_alignment) )

// Groups to which belongs the current process.
static bool in_group_N = false;		// process_id < num_processes_N
static bool in_group_M = false;		// process_id < num_processes_M
static bool in_group_active = false;	// in_group_N || in_group_M

// Group of processes for each dimension
static MPI_Comm comm_N = MPI_COMM_NULL;
static MPI_Comm comm_M = MPI_COMM_NULL;
static MPI_Comm comm_diff = MPI_COMM_NULL;	// abs( num_processes_N - num_processes_M ) + 1 ('+1' to include the process "0")
static MPI_Comm comm_active = MPI_COMM_NULL;	// MAX( num_processes_N, num_processes_M ).
static MPI_Comm comm_idle = MPI_COMM_NULL;	// num_processes - MAX( num_processes_N, num_processes_M ) + 1 ('+1' to include the process "0")

// <RpP> values for, at least, the first <num_processes_R - 1> processes.
static index_t NpP_base = 0;
static index_t MpP_base = 0;

// If <RpP> for the "last" process in group (i.e., process_id == (num_processes_R - 1)) is DIFFERENT to <RpP_base>.
static bool variable_size_N = false;
static bool variable_size_M = false;

/* Portion of matrices W and H corresponding to all processes.
 * They are similar to NpP, MpP, bN, and bM ; all of them multiplied by <Kp>.
 * Used only for synchronization.
 *
 * Since:
 *	bR == (RpP_base * process_id)
 *
 *	For processes 0..<num_processes_R - 2>:
 *		RpP == RpP_base == RpPp
 *
 *	For the "last" process (i.e., process_id == (num_processes_R - 1)):
 *		RpP  == R  - bR		(sometimes denoted as "RpP_last")
 *		RpPp == Rp - bR
 *
 * Then:
 *	If variable_size_R == true (i.e, "RpP_last" != RpP_base):
 *		length( offsets_pP[] ) == length( nitems_pP[] ) == num_processes_R
 *		offsets_pP[] == bR[] * Kp
 *		nitems_pP[ 0..<num_processes_R - 2> ] ==  RpP_base  * Kp
 *		nitems_pP[   num_processes_R - 1    ] == "RpP_last" * Kp
 *
 *	Otherwise (i.e., <RpP> is SIMILAR for all processes):
 *		length( offsets_pP[] ) == length( nitems_pP[] ) == 1
 *		offsets_pP[0] == (bR[process_id] * Kp). That is, the offset for JUST the current process.
 *		nitems_pP[0]  == (RpP * Kp), which is similar for all processes.
 *
 * NOTE: Type of both vectors is 'index_t' in order to match the MPI standard.
 */
static index_t const *restrict offsets_WpP = NULL;	// Offset to portion of matrix W (i.e., <bN * Kp>) for this, or all processes.
static index_t const *restrict offsets_HpP = NULL;	// Offset to portion of matrix H (i.e., <bM * Kp>) for this, or all processes.
static index_t const *restrict nitems_WpP = NULL;	// Number of items of matrix W (i.e., NpP * Kp) for all processes.
static index_t const *restrict nitems_HpP = NULL;	// Number of items of matrix H (i.e., MpP * Kp) for all processes.


#if NMFGPU_PROFILING_COMM
	// Communication timing
	static timing_data_t comm_W_timing;
	static timing_data_t comm_H_timing;
	static timing_data_t comm_others_timing;
#endif

#if NMFGPU_DEBUG || NMFGPU_VERBOSE || NMFGPU_VERBOSE_2 || NMFGPU_PROFILING_COMM || NMFGPU_PROFILING_GLOBAL
	static bool const dbg_shown_by_all = true;	// Information or error messages on debug.
	static bool const verb_shown_by_all = false;	// Information messages in verbose mode.
#endif
static bool const shown_by_all = false;			// Information messages.
static bool const sys_error_shown_by_all = true;	// System error messages.
static bool const error_shown_by_all = false;		// Error messages on invalid arguments or I/O data.

static char error_string[ MPI_MAX_ERROR_STRING ];		// Array for MPI error messages.

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Sets in the global variable "error_string", the error message according to
 * the provided MPI status.
 */
static void setMPIErrorString( int status )
{

	int len_err = 0;

	// Sets the error message in error_string[]
	MPI_Error_string( status, error_string, &len_err );

	error_string[ len_err ] = '\0';		// Makes sure that the string is null-terminated.

} // getMPIErrorString

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Computes the portion of dimension "R" (N or M) corresponding to this process.
 *
 * On padding mode, tries to use portions of a size that is a multiple of
 * <memory_alignment>.
 *
 * Rp:	Padded dimension for the current process (R <= Rp).
 *	It is IGNORED in "non-padding" mode.
 *
 * bR:	Starting index for the current process (0 <= bR < R).
 *
 * RpP:	 Number of items in dimension "R" for the current process (RpP <= R).
 *
 * RpPp: Padded number of items for the current process (RpP <= RpPp <= Rp).
 *	 It is IGNORED (and not even referenced) in "non-padding" mode.
 *
 * num_processes_R: Effective number of processes for dimension "R"
 *		    (num_processes_R <= num_processes).
 *
 * RpP_base:	Number of items in dimension "R" (i.e., <RpP>) for, at least,
 *		the first <num_processes_R - 1> processes (RpP == RpP_base).
 *		In padding mode, RpP == RpPp == RpP_base
 *		It must be non-NULL regardless of <num_processes>.
 *
 * variable_size_R:
 *		Set to 'true' if (RpP != RpP_base) for the LAST process.
 *		Set to 'false' if (RpP == RpP_base) for ALL processes.
 *
 * in_group_R:	Set to 'true'  if (process_id < num_processes_R)
 *
 * NOTE:
 *	bR == (RpP_base * process_id)
 *
 *	For processes 0..<num_processes_R - 2>:
 *		RpP == RpP_base == RpPp
 *
 *	For the last process (num_processes_R - 1):
 *		RpP  == R  - bR
 *		RpPp == Rp - bR
 *
 *	variable_size_R == ( RpP[num_processes_R - 1] != RpP_base )
 *
 * Return EXIT_SUCCESS or EXIT_FAILURE.
 */
static int get_perProcess_dimension( index_t R, index_t Rp, char const *restrict R_str, bool padding_mode, index_t *restrict bR,
					index_t *restrict RpP, index_t *restrict RpPp, index_t *restrict num_processes_R,
					index_t *restrict RpP_base, bool *restrict variable_size_R, bool *restrict in_group_R )
{

	// Dimensions for the current process
	index_t l_bR = 0, l_RpP = 0, l_RpPp = 0;

	// Effective number of processes
	index_t l_num_processes_R = 0;

	// Dimensions for, at least, the first <num_processes_R - 1> processes.
	index_t l_RpP_base = 0, l_RpPp_base = 0;

	// Dimensions for process ID <num_processes_R - 1>.
	index_t l_RpP_last = 0, l_RpPp_last = 0;

	// If the last process have a different RpP value (i.e., if l_RpP_last != RpP_base)
	bool l_variable_size_R = false;

	// If this process will be included in group R.
	bool l_in_group_R = false;	// process_id < num_processes_R

	// -------------------------------------

	// Checks for NULL pointers
	if ( ! ( (uintptr_t) R_str * (uintptr_t) bR * (uintptr_t) RpP * (uintptr_t) num_processes_R * (uintptr_t) RpP_base *
		(uintptr_t) variable_size_R * ((! padding_mode) + (uintptr_t) RpPp) ) ) {
		int const errnum = EFAULT;
		if ( ! R_str ) print_errnum( error_shown_by_all, errnum, "get_perProcess_dimension( R_str )" );
		if ( ! bR ) print_errnum( error_shown_by_all, errnum, "get_perProcess_dimension( bR )" );
		if ( ! RpP ) print_errnum( error_shown_by_all, errnum, "get_perProcess_dimension( RpP )" );
		if ( padding_mode * (! RpPp) ) print_errnum( error_shown_by_all, errnum, "get_perProcess_dimension( padding_mode, RpPp )" );
		if ( ! num_processes_R ) print_errnum( error_shown_by_all, errnum, "get_perProcess_dimension( num_processes_R )" );
		if ( ! RpP_base ) print_errnum( error_shown_by_all, errnum, "get_perProcess_dimension( RpP_base )" );
		if ( ! variable_size_R ) print_errnum( error_shown_by_all, errnum, "get_perProcess_dimension( variable_size_R )" );
		return EXIT_FAILURE;
	}

	// Checks for invalid values
	if ( (R < num_processes) + (padding_mode * (Rp < MIN(R,memory_alignment))) ) {
		int const errnum = EINVAL;
		if ( R < num_processes )
			print_errnum( error_shown_by_all, errnum, "get_perProcess_dimension( %s = %" PRI_IDX ")", R_str, R );
		if ( padding_mode * (Rp < MIN( R, memory_alignment )) )
			print_errnum( error_shown_by_all, errnum, "get_perProcess_dimension( %sp = %" PRI_IDX ")", R_str, Rp );
		return EXIT_FAILURE;
	}

	// On non-padding mode, sets Rp to <R>.
	if ( ! padding_mode )
		Rp = R;

	// -------------------------------------

	#if NMFGPU_VERBOSE_2 || NMFGPU_VERBOSE || NMFGPU_DEBUG
		print_message( verb_shown_by_all, "get_perProcess_dimension( %s=%" PRI_IDX ", %sp=%" PRI_IDX ", padding_mode: %i)\n",
				R_str, R, R_str, Rp, padding_mode );
	#endif

	// -------------------------------------

	// First, computes the number of processes for this dimensions

	/* On padding mode, portions will be of a size that is a multiple of <memory_alignment>.
	 * Otherwise, the number of processes is limited to <R>.
	 */
	{
		index_t const max_procs = ( padding_mode ? (Rp / memory_alignment) : R ); // > 0
		l_num_processes_R = MIN( max_procs, num_processes );	  // 0 < l_num_processes_R <= MIN( num_processes, R or Rp )
	}


	// -------------------------------------

	// Dimensions for the first <num_processes_R - 1> processes

	// (Possibly padded) Dimension for the first <l_num_processes_R - 1> processes:
	l_RpPp_base = ( Rp / l_num_processes_R );	// May be truncated

	/* In padding mode, tries to set RpP_base <= RpP[last process]
	 * Otherwise, RpP_last <= l_RpP_base
	 */
	if ( padding_mode ) {
		l_RpPp_base -= ( l_RpPp_base % memory_alignment );	// Previous multiple of <memory_alignment>
		// l_RpPp_base = get_padding( (l_RpPp_base + ( Rp % l_num_processes_R )) );	// Testing only.
	} else
		l_RpPp_base += ( Rp % l_num_processes_R );

	// "Actual" dimension
	l_RpP_base = MIN( l_RpPp_base, R );


	// -------------------------------------

	// Dimension for process ID <num_processes_R - 1>:
	{
		index_t const l_last_bR = l_RpPp_base * ( l_num_processes_R - 1 );
		l_RpPp_last = Rp - l_last_bR;		// Already a multiple of <memory_alignment> (as long as "Rp" and "l_RpPp_base" are)
		l_RpP_last  = R  - l_last_bR;
	}

	// -------------------------------------

	/* (Final) Dimensions for the current process:
	 *
	 * NOTE: if (num_processes_R == 1), then (RpP(p)_base == RpP(p)_last)
	 */

	l_bR = process_id * l_RpPp_base;	// l_bR == last_bR when process_id == (num_processes_R - 1)

	l_RpP  = ( (process_id < (l_num_processes_R-1)) ? l_RpP_base  : l_RpP_last  );
	l_RpPp = ( (process_id < (l_num_processes_R-1)) ? l_RpPp_base : l_RpPp_last );

	l_variable_size_R = ( l_RpP_last != l_RpP_base );	// && (num_processes_R > 1)

	l_in_group_R = ( process_id < l_num_processes_R );

	// -------------------------------------

	#if NMFGPU_VERBOSE_2 || NMFGPU_VERBOSE || NMFGPU_DEBUG
	{
		print_message( verb_shown_by_all, "get_perProcess_dimension:\n\tFor dimension %s (%" PRI_IDX " process(es) of %"
				PRI_IDX "):\n\t\t%" PRI_IDX " process(es) with %spP=%" PRI_IDX " (%spPp=%" PRI_IDX ").\n", R_str,
				l_num_processes_R, num_processes, (l_num_processes_R - l_variable_size_R), R_str, l_RpP_base,
				R_str, l_RpPp_base );

		if ( l_variable_size_R )
			append_printed_message( verb_shown_by_all, "\t\t1 process with %spP=%" PRI_IDX " (%spPp=%" PRI_IDX ")\n",
						R_str, l_RpP_last, R_str, l_RpPp_last );
	}
	#endif

	// -------------------------------------

	// Sets output values

	*bR = l_bR;
	*RpP = l_RpP;

	if ( (uintptr_t) padding_mode * (uintptr_t) RpPp )
		*RpPp = l_RpPp;

	*num_processes_R = l_num_processes_R;

	*RpP_base = l_RpP_base;

	*variable_size_R = l_variable_size_R;

	*in_group_R = l_in_group_R;

	return EXIT_SUCCESS;

} // get_perProcess_dimension

////////////////////////////////////////////////

/*
 * Computes the portion of the given matrix corresponding to each process.
 *
 * R: Number of rows IN MEMORY (regardless of matrix transposing).
 *
 * RpP_base:	<RpP> for, at least, the first <num_processes_R - 1> processes
 *		(RpP_base <= R)
 *
 * offsets_pP[]: Offset to portion for this or all processes: <bR * pitch>
 * nitems_pP[] : Number of items for all processes: <RpP * pitch>
 *
 * Since:
 *	bR == (RpP_base * process_id)
 *
 *	For processes 0..<num_processes_R - 2>:
 *		RpP == RpP_base == RpPp
 *
 *	For the "last" process (i.e., process_id == (num_processes_R - 1)):
 *		RpP  == R  - bR		(sometimes denoted as "RpP_last")
 *		RpPp == Rp - bR
 *
 * Then:
 *	If variable_size_R == true (i.e, "RpP_last" != RpP_base):
 *		length( offsets_pP[] ) == length( nitems_pP[] ) == num_processes_R
 *		offsets_pP[] == bR[] * Kp
 *		nitems_pP[ 0..<num_processes_R - 2> ] ==  RpP_base  * Kp
 *		nitems_pP[   num_processes_R - 1    ] == "RpP_last" * Kp
 *
 *	Otherwise (i.e., <RpP> is SIMILAR for all processes):
 *		length( offsets_pP[] ) == length( nitems_pP[] ) == 1
 *		offsets_pP[0] == (bR[process_id] * Kp). That is, the offset for JUST the current process.
 *		nitems_pP[0]  == (RpP_base * Kp), which is similar for all processes.
 *
 * NOTE: Type of both vectors is 'index_t' in order to match the MPI standard.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int get_dim_vectors( char const *restrict matrix_name, index_t R, index_t pitch, bool variable_size_R, index_t RpP_base,
				index_t num_processes_R, index_t const *restrict *restrict offsets_pP,
				index_t const *restrict *restrict nitems_pP )
{

	// Output vectors:
	index_t *restrict l_offsets_pP = NULL;
	index_t *restrict l_nitems_pP = NULL;

	// Length of output vectors:
	index_t const length = ( variable_size_R ? num_processes_R : 1 );

	// Variable or similar number of items.
	char nitems_status_str[ 16 ];
	sprintf( nitems_status_str, ( variable_size_R ? "different" : "similar" ) );

	// ---------------------------

	if ( ! ( (uintptr_t) matrix_name * (uintptr_t) nitems_pP * (uintptr_t) offsets_pP ) ) {
		int const errnum = EFAULT;
		if ( ! matrix_name ) print_errnum( error_shown_by_all, errnum, "get_dim_vectors( matrix_name )" );
		if ( ! nitems_pP ) print_errnum( error_shown_by_all, errnum, "get_dim_vectors( nitems_pP )" );
		if ( ! offsets_pP ) print_errnum( error_shown_by_all, errnum, "get_dim_vectors( offsets_pP )" );
		return EXIT_FAILURE;
	}

	// Checks matrix dimensions (NO matrix transposing ; quiet mode)
	if ( matrix_check_dimensions( "get_dim_vectors", R, pitch, pitch, false, false ) != EXIT_SUCCESS ) {
		print_error( error_shown_by_all, "Invalid dimensions for matrix %s.\n", matrix_name );
		return EXIT_FAILURE;
	}

	if ( (num_processes_R <= 0) + (num_processes_R > MIN(num_processes,R)) + (RpP_base <= 0) + (RpP_base > R) ) {
		int const errnum = EINVAL;
		if ( (num_processes_R <= 0) + (num_processes_R > MIN(num_processes,R)) )
			print_errnum( error_shown_by_all, errnum, "get_dim_vectors( num_processes = %" PRI_IDX " for matrix %s)",
					num_processes_R, matrix_name );
		if ( (RpP_base <= 0) + (RpP_base > R) )
			print_errnum( error_shown_by_all, errnum, "get_dim_vectors( RpP_base = %" PRI_IDX " for matrix %s, R = %"
					PRI_IDX " )", RpP_base, matrix_name, R );
		return EXIT_FAILURE;
	}

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Initializing arrays of portion sizes for matrix %s (%" PRI_IDX " process(es), "
				"%s size)...\n", matrix_name, num_processes_R, nitems_status_str );
	#endif

	// ---------------------------

	// Allocates memory for vectors.

	errno = 0;
	l_offsets_pP = (index_t *restrict) malloc( length * sizeof(index_t) );
	if ( ! l_offsets_pP )  {
		print_errnum( sys_error_shown_by_all, errno, "Error in get_dim_vectors(matrix %s, %" PRI_IDX " process(es), %s size): "
				"malloc( offsets_%spP[%" PRI_IDX "] )", matrix_name, num_processes_R, nitems_status_str, length );
		return EXIT_FAILURE;
	}

	l_nitems_pP = (index_t *restrict) malloc( length * sizeof(index_t) );
	if ( ! l_nitems_pP ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in get_dim_vectors(matrix %s, %" PRI_IDX " process(es), %s size): "
				"malloc( nitems_%spP[%" PRI_IDX "] )", matrix_name, num_processes_R, nitems_status_str, length );
		free( (void *) l_offsets_pP );
		return EXIT_FAILURE;
	}

	// ---------------------------

	if ( variable_size_R ) {

		// Offsets
		l_offsets_pP[ 0 ] = 0;
		for ( index_t p=0 ; p < length ; p++ ) {
			index_t const bR = RpP_base * p;
			size_t const offset = (size_t) bR * (size_t) pitch;
			l_offsets_pP[ p ] = offset;
		}

		// Number of items per process.
		for ( index_t p=0 ; p < (length - 1) ; p++ ) {
			index_t const RpP = RpP_base;
			size_t const nitems = (size_t) RpP * (size_t) pitch;
			l_nitems_pP[ p ] = nitems;
		}

		// Items in last process
		index_t const bR = RpP_base * (length - 1);
		index_t const RpP = R - bR;
		size_t const nitems = (size_t) RpP * (size_t) pitch;
		l_nitems_pP[ (length - 1) ] = nitems;

	} else {	// Same number of items for all processes.

		// Offset for just the current process.
		index_t const bR = RpP_base * process_id;
		size_t const offset = (size_t) bR * (size_t) pitch;
		l_offsets_pP[ 0 ] = offset;

		// Number of items for the current process (which is similar to the rest).
		size_t const nitems = (size_t) RpP_base * (size_t) pitch;
		l_nitems_pP[ 0 ] = nitems;
	}

	// ---------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Initializing arrays of portion sizes for matrix %s (%" PRI_IDX " process(es), "
				"%s size)... Done.\n", matrix_name, num_processes_R, nitems_status_str );
	#endif

	// Sets output pointers
	*offsets_pP = (index_t const *restrict) l_offsets_pP;
	*nitems_pP = (index_t const *restrict) l_nitems_pP;

	return EXIT_SUCCESS;

} // get_dim_vectors

////////////////////////////////////////////////

/*
 * Read matrix from file
 *
 * numeric_hdrs, numeric_lbls: Has <filename> numeric column headers / row headers ?
 * isBinary: Is <filename> a binary file?
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int init_V( const char *restrict filename, bool numeric_hdrs, bool numeric_lbls, file_fmt_t input_file_fmt,
			struct matrix_tags_t *restrict mt )
{

	// Checks for NULL parameters

	if ( ! ( (uintptr_t) filename * (uintptr_t) mt ) ) {
		int const errnum = EFAULT;
		if ( ! filename ) print_errnum( error_shown_by_all, errnum, "init_V( filename )" );
		if ( ! mt )	print_errnum( error_shown_by_all, errnum, "init_V( mt )" );
		return EXIT_FAILURE;
	}

	/////////////////////////////////

	#if NMFGPU_VERBOSE || NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Initializing input matrix from file %s...\n", filename );
	#endif

	int status = EXIT_SUCCESS;

	// ----------------------------

	index_t nrows = 0, ncols = 0, pitch = 0;

	real *restrict matrix = NULL;
	struct matrix_tags_t l_mt = new_empty_matrix_tags();

	status = matrix_load( filename, numeric_hdrs, numeric_lbls, input_file_fmt, &matrix, &nrows, &ncols, &pitch, &l_mt );
	if ( status != EXIT_SUCCESS ) {
		print_error( error_shown_by_all, "Error reading input file.\n" );
		return EXIT_FAILURE;
	}

	// --------------------------------

	/* Per-process matrix dimensions:
	 * NOTE: MpP == MpPp == MpP_base for the first <num_processes_M - 1> processes.
	 */

	index_t l_bN = 0, l_bM = 0;
	index_t l_NpP = 0, l_MpP = 0, l_MpPp = 0;
	index_t l_NpP_base = 0, l_MpP_base = 0;
	index_t l_num_processes_N = 0, l_num_processes_M = 0;
	bool l_variable_size_N = false, l_variable_size_M = false;
	bool l_in_group_N = false, l_in_group_M = false;

	// Dimension N (NON-padding mode):
	bool padding_mode = false;
	status = get_perProcess_dimension( nrows, nrows, "N", padding_mode, &l_bN, &l_NpP, NULL, &l_num_processes_N,
						&l_NpP_base, &l_variable_size_N, &l_in_group_N );
	if ( status != EXIT_SUCCESS ) {
		matrix_clean( matrix, l_mt );
		return EXIT_FAILURE;
	}

	// Dimension M (padding mode):
	padding_mode = true;
	status = get_perProcess_dimension( ncols, pitch, "M", padding_mode, &l_bM, &l_MpP, &l_MpPp, &l_num_processes_M,
						&l_MpP_base, &l_variable_size_M, &l_in_group_M );
	if ( status != EXIT_SUCCESS ) {
		matrix_clean( matrix, l_mt );
		return EXIT_FAILURE;
	}

	/* If any of num_processes_R is '1' forces to work in "single-process" mode.
	 * That is, all other processes become "non-active" (idle).
	 */
	if ( ((l_num_processes_N == 1) + (l_num_processes_M == 1)) * (num_processes > 1) ) {

		// Resets all per-process dimensions as in single-process mode.

		#if NMFGPU_DEBUG || NMFGPU_PROFILING_COMM || NMFGPU_PROFILING_GLOBAL || NMFGPU_VERBOSE || NMFGPU_VERBOSE_2
			print_error( verb_shown_by_all, "WARNING: Due to a low number of columns or rows, "
					"switches to single-process mode.\n" );
		#endif

		l_NpP_base = nrows;
		l_MpP_base = ncols;
		l_num_processes_N = l_num_processes_M = 1;
		l_variable_size_N = l_variable_size_M = false;

		if ( process_id ) {	// Idle processes
			l_bN = nrows;
			l_bM = pitch;
			l_NpP = l_MpP = l_MpPp = 0;
			l_in_group_N = l_in_group_M = false;

		} else { // Master process
			l_bN = l_bM = 0;
			l_NpP = nrows;
			l_MpP = ncols;
			l_MpPp = pitch;
			l_in_group_N = l_in_group_M = true;
		}

	} // If must force the single-process mode

	// --------------------------------

	// Number of "active" processes

	index_t const l_num_act_processes = MAX( l_num_processes_N, l_num_processes_M );	// <= num_processes

	bool const l_in_group_active = ( l_in_group_N + l_in_group_M );

	// --------------------------------

	/* Allocates PINNED (i.e., page-locked) memory for both portions of
	 * the input matrix assigned to this process.
	 */

	real *restrict l_Vrow = NULL;
	real *l_Vcol = NULL;

	// Vrow[ NpP ][ Mp ] <-- V[ bN..(bN+NpP-1) ][ Mp ]
	if ( l_in_group_N ) {		// Group "N" only

		/* NOTE for "single-process" mode:
		 *
		 *	As of CUDA 6.0, it is possible to register (i.e., to page-lock)
		 *	a memory area returned by malloc(3), but NOT in write-combined
		 *	mode. Therefore, we still allocating a new memory area and
		 *	copying the input matrix.
		 */

		size_t const offset = (size_t) l_bN  * (size_t) pitch;	// Offset to Vrow
		size_t const nitems = (size_t) l_NpP * (size_t) pitch;	// Number of items in Vrow
		real const *const pMatrix = (real const *restrict) &matrix[ offset ];		// Pointer to data for Vrow

		bool const wc = true;				// Write-Combined mode
		bool const clear_memory = false;		// Do NOT initialize the allocated memory
		real *restrict const V = (real *restrict) getHostMemory( nitems * sizeof(real), wc, clear_memory );
		if ( ! V ) {
			print_error( sys_error_shown_by_all, "Error allocating HOST memory for input matrix (Vrow).\n" );
			matrix_clean( matrix, l_mt );
			return EXIT_FAILURE;
		}

		// Copies input matrix to the new memory.
		errno = 0;
		if ( ! memcpy( V, pMatrix, nitems * sizeof(real) ) )  {
			print_errnum( sys_error_shown_by_all, errno, "Error initializing input matrix on HOST memory (Vrow)" );
			freeHostMemory( V, "Vrow" );
			matrix_clean( matrix, l_mt );
			return EXIT_FAILURE;
		}

		l_Vrow = V;

	} // If in group "N"

	// --------------------------------

	// Vcol[ N ][ MpPp ] <-- V[ N ][ bM..(bM+MpPp-1) ]

	if ( l_in_group_M ) {	// Group "M" only

		if ( l_num_processes_M > 1 ) {

			size_t const offset = l_bM;				// Offset to Vcol
			size_t const nitems = (size_t) nrows * (size_t) l_MpPp;	// Number of items in Vcol
			real const *pMatrix = (real const *restrict) &matrix[ offset ];		// Pointer to data for Vcol

			bool const wc = true;					// Write-Combined mode
			bool const clear_memory = false;			// Do NOT initialize the allocated memory
			real *restrict const V = (real *restrict) getHostMemory( nitems * sizeof(real), wc, clear_memory );
			if ( ! V ) {
				print_error( sys_error_shown_by_all, "Error allocating HOST memory for input matrix (Vcol).\n" );
				freeHostMemory( l_Vrow, "Vrow" );
				matrix_clean( matrix, l_mt );
				return EXIT_FAILURE;
			}

			// Copies input matrix to the new memory.
			real *pV = V;
			size_t const width = ( (l_bM + l_MpPp <= pitch) ? l_MpPp : l_MpP ) * sizeof(real);
			for ( index_t i = 0 ; i < nrows ; i++, pMatrix += pitch, pV += l_MpPp ) {
				errno = 0;
				if ( ! memcpy( pV, pMatrix, width ) )  {
					print_errnum( sys_error_shown_by_all, errno, "Error initializing input matrix on HOST memory (Vrow)" );
					freeHostMemory( V, "Vcol" );
					freeHostMemory( l_Vrow, "Vrow" );
					matrix_clean( matrix, l_mt );
					return EXIT_FAILURE;
				}
			}

			l_Vcol = V;

		} else	// In single-process mode, Vrow and Vcol are just aliases.
			l_Vcol = l_Vrow;

	} // If in group "M"

	// --------------------------------

	free( matrix );

	#if NMFGPU_VERBOSE_2 || NMFGPU_DEBUG
		print_message( verb_shown_by_all, "\tNumber of active processors: %" PRI_IDX "\n", l_num_act_processes );
		#if NMFGPU_DEBUG
			print_message( dbg_shown_by_all, "In groups N: %i, M: %i, and active: %i.\n",
					l_in_group_N, l_in_group_N, l_in_group_active );
		#endif
	#endif

	// --------------------------------

	// Sets GLOBAL variables.

	N = nrows;	M = ncols;	Mp = pitch;
	bN = l_bN;	bM = l_bM;
	NpP = l_NpP;	MpP = l_MpP;	MpPp = l_MpPp;

	NpP_base = l_NpP_base;			MpP_base = l_MpP_base;
	num_processes_N = l_num_processes_N;	num_processes_M = l_num_processes_M;
	variable_size_N = l_variable_size_N;	variable_size_M = l_variable_size_M;
	in_group_N = l_in_group_N;		in_group_M = l_in_group_M;

	in_group_active = l_in_group_active;	num_act_processes = l_num_act_processes;

	Vrow = l_Vrow;		Vcol = l_Vcol;

	// --------------------------------

	// Output values.

	*mt = l_mt;

	// --------------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Initializing input matrix from file %s... Done.\n", filename );
	#endif

	return EXIT_SUCCESS;

} // init_V

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Creates a new communicator from MPI_COMM_WORLD with processes:
 *	[0 <base_id> ... <max_id>]
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int new_comm( index_t base_id, index_t max_id, MPI_Comm *restrict comm )
{

	int status = MPI_SUCCESS;

	MPI_Comm l_comm = MPI_COMM_NULL;		// New communicator
	MPI_Group group_comm_world = MPI_GROUP_NULL;	// Group of all processes
	MPI_Group group_new_comm = MPI_GROUP_NULL;	// Group of processes in new communicator (i.e., [0, <base_id>...<max_id>])

	// Vector of triplets of the form [first ID, last ID, stride].
	int ranges[2][3] = { {0, 0, 1}, {base_id, max_id, 1} }; // [ [0 0 1]  [<base_id> <max_id> 1] ]

	// If (base_id == 0), skips the first triplet.
	index_t const len_ranges = ((bool) base_id) + 1;	// len = ( (base_id > 0) ? 2 : 1 )

	// -------------------------------------

	// Gets the group underlying MPI_COMM_WORLD.

	status = MPI_Comm_group( MPI_COMM_WORLD, &group_comm_world );

	if ( status != MPI_SUCCESS ) {
		setMPIErrorString( status );
		print_error( sys_error_shown_by_all, "new_comm( processes: [%s%" PRI_IDX "... %" PRI_IDX " ] ): "
				"MPI_Comm_group(): %s\n", ( base_id ? "0 " : " " ), base_id, max_id, error_string );
		return EXIT_FAILURE;
	}

	// -------------------------------------

	// Creates a new group with the given range of processes.

	status = MPI_Group_range_incl( group_comm_world, len_ranges, &ranges[ (! base_id) ], &group_new_comm );

	if ( status != MPI_SUCCESS ) {
		setMPIErrorString( status );
		print_error( sys_error_shown_by_all, "new_comm( processes: [%s%" PRI_IDX "... %" PRI_IDX " ] ): "
				"MPI_Group_range_incl( %" PRI_IDX " range(s) ): %s\n", ( base_id ? "0, " : " " ),
				base_id, max_id, len_ranges, error_string );
		MPI_Group_free( &group_comm_world );
		return EXIT_FAILURE;
	}

	// -------------------------------------

	// Creates the new communicator.

	status = MPI_Comm_create( MPI_COMM_WORLD, group_new_comm, &l_comm );

	if ( status != MPI_SUCCESS ) {
		setMPIErrorString( status );
		print_error( sys_error_shown_by_all, "new_comm( processes: [%s%" PRI_IDX "... %" PRI_IDX " ] ): "
				"MPI_Comm_create(): %s\n", ( base_id ? "0 " : " " ), base_id, max_id, error_string );
		MPI_Group_free( &group_new_comm );
		MPI_Group_free( &group_comm_world );
		return EXIT_FAILURE;
	}

	// -------------------------------------

	MPI_Group_free( &group_new_comm );
	MPI_Group_free( &group_comm_world );

	*comm = l_comm;

	return EXIT_SUCCESS;

} // new_comm

// ---------------------------------------------

/*
 * Creates several communicators:
 *	- For each dimension: num_processes_N and num_processes_M.
 *	- 'diff' group (i.e., abs(num_processes_N - num_processes_M) + 1 ).
 *	- 'active' group (i.e., MAX(num_processes_N,num_processes_M)).
 *	- 'idle' group (i.e., num_processes - MAX(num_processes_N,num_processes_M) + 1).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int setup_communicators( void )
{

	int status = EXIT_SUCCESS;

	index_t const min_num_procs = MIN( num_processes_N, num_processes_M );
	index_t const max_num_procs = MAX( num_processes_N, num_processes_M );

	// Group of processes for each dimension
	MPI_Comm l_comm_N = MPI_COMM_NULL;		// in_group_N
	MPI_Comm l_comm_M = MPI_COMM_NULL;		// in_group_M
	MPI_Comm l_comm_diff = MPI_COMM_NULL;		// in_group_N != in_group_M
	MPI_Comm l_comm_active = MPI_COMM_NULL;		// in_group_N || in_group_M
	MPI_Comm l_comm_idle = MPI_COMM_NULL;		// (! in_group_N) && (! in_group_M)

	// ---------------------------

	// New communicator for dimension N

	status = new_comm( 0, (num_processes_N - 1), &l_comm_N );

	if ( status != EXIT_SUCCESS ) {
		print_error( sys_error_shown_by_all, "Error creating communicator for dimension N.\n" );
		return EXIT_FAILURE;
	}

	// ---------------------------

	// New communicator for dimension M

	status = new_comm( 0, (num_processes_M - 1), &l_comm_M );

	if ( status != EXIT_SUCCESS ) {
		print_error( sys_error_shown_by_all, "Error creating communicator for dimension M.\n" );
		MPI_Comm_free( &l_comm_N );
		return EXIT_FAILURE;
	}

	// ---------------------------

	/* Communicator for group "diff": abs( num_processes_N - num_processes_M ) + 1
	 * ('+1' to include the process "0")
	 *
	 * Processes included: [ 0, min_num_procs, ..., (max_num_procs-1) ]
	 */
	if ( num_processes_N != num_processes_M ) {

		#if NMFGPU_DEBUG
			print_message( dbg_shown_by_all, "Creating communicator for group 'diff'..\n" );
		#endif

		status = new_comm( min_num_procs, (max_num_procs - 1), &l_comm_diff );

		if ( status != EXIT_SUCCESS ) {
			print_error( sys_error_shown_by_all, "Error creating communicator for group 'diff': [ 0, %"
					PRI_IDX " ... %" PRI_IDX "]\n", min_num_procs, (max_num_procs - 1) );
			MPI_Comm_free( &l_comm_M ); MPI_Comm_free( &l_comm_N );
			return EXIT_FAILURE;
		}

	} // if (num_processes_N != num_processes_M)

	// ---------------------------

	/* Communicator for "active" (i.e., not idle) processes: max_num_procs
	 *
	 * Processes included: [ 0 ... (max_num_procs -1) ]
	 *
	 * NOTE:
	 *	Rather than create a new communicator with
	 *		 new_comm( 0, (max_num_procs - 1), &l_comm_active ),
	 *	just duplicates the largest group (N or M).
	 */
	{
		MPI_Comm max_comm = ( (num_processes_N >= num_processes_M) ? l_comm_N : l_comm_M );

		status = MPI_Comm_dup( max_comm, &l_comm_active );

		if ( status != MPI_SUCCESS ) {
			setMPIErrorString( status );
			print_error( sys_error_shown_by_all, "Error creating communicator for group 'active': [ 0 ... %" PRI_IDX "]: %s\n",
					max_num_procs - 1, error_string );
			MPI_Comm_free( &l_comm_diff ); MPI_Comm_free( &l_comm_M ); MPI_Comm_free( &l_comm_N );
			return EXIT_FAILURE;
		}
	}

	// ---------------------------

	/* Communicator for "idle" (i.e., non-active) processes: num_processes - max_num_procs + 1
	 * ('+1' to include the process "0")
	 *
	 * Processes included: [ 0, max_num_procs, ..., (num_processes-1) ]
	 */
	if ( max_num_procs < num_processes ) {

		#if NMFGPU_DEBUG
			print_message( dbg_shown_by_all, "Creating communicator for group 'idle'..\n" );
		#endif

		status = new_comm( max_num_procs, (num_processes - 1), &l_comm_idle );

		if ( status != EXIT_SUCCESS ) {
			print_error( sys_error_shown_by_all, "Error creating communicator for group 'idle': [ 0, %"
					PRI_IDX " ... %" PRI_IDX "]\n", max_num_procs, (num_processes - 1) );
			MPI_Comm_free( &l_comm_active ); MPI_Comm_free( &l_comm_diff );
			MPI_Comm_free( &l_comm_M ); MPI_Comm_free( &l_comm_N );
			return EXIT_FAILURE;
		}
	}

	// ---------------------------

	comm_N = l_comm_N;
	comm_M = l_comm_M;
	comm_diff = l_comm_diff;
	comm_active = l_comm_active;
	comm_idle = l_comm_idle;

	return EXIT_SUCCESS;

} // setup_communicators

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Gather-to-all synchronization on the given group of process.
 * Gathers data from all process and distributes back the combined data.
 *
 * If "variable_size_R" is set to 'true', the length of both vectors is equals
 * the number of process in the group "comm_R" (i.e., <num_processes_R>).
 * Otherwise, the length of both vectors is '1'.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int sync_with_slaves( real *restrict matrix, bool variable_size_R, index_t const *restrict offsets_pP,
				index_t const *restrict nitems_pP, MPI_Comm comm_R
				#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_COMM))
					, char const *restrict const matrix_name
				#endif
				#if NMFGPU_PROFILING_COMM
					, timing_data_t *restrict const comm_timing
				#endif
				)
{

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Synchronizing matrix %s (all_gather, variable sizes: %i)...\n",
				matrix_name, variable_size_R );
	#endif

	// Number of items sent by this process.
	size_t const nitems = nitems_pP[ (variable_size_R * process_id) ];	// ( variable_size_R ? [process_id] : [0] )

	#if NMFGPU_DEBUG || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_COMM))
		int status = EXIT_SUCCESS;
	#endif

	// ---------------------------

	/* The data block sent from the <i>-th process is received by all other
	 * processes and placed in the <i>-th block of matrix.
	 */

	#ifdef NMFGPU_PROFILING_COMM
		MPI_Barrier( comm_R );
		double const time_0 = MPI_Wtime();
	#endif

		if ( variable_size_R ) { // nitems_pP differs on each process.

			#if NMFGPU_DEBUG || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_COMM))
				status =
			#endif

				MPI_Allgatherv( MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, (void *) matrix,
						(int *) nitems_pP, (int *) offsets_pP, NMFGPU_MPI_REAL_T, comm_R );

		} else { // nitems_pP is similar on all processes.

			#if NMFGPU_DEBUG || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_COMM))
				status =
			#endif

				MPI_Allgather( MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, (void *) matrix, nitems, NMFGPU_MPI_REAL_T, comm_R );
		}

		#if NMFGPU_DEBUG || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_COMM))
			if ( status != MPI_SUCCESS ) {
				setMPIErrorString( status );
				print_error( sys_error_shown_by_all, "MPI error: Could not synchronize matrix %s "
						"(variable sizes: %i): %s.\n", matrix_name, variable_size_R, error_string );
				return EXIT_FAILURE;
			}
		#endif


	#ifdef NMFGPU_PROFILING_COMM
	{
		MPI_Barrier( comm_R );
		double const time_1 = MPI_Wtime();
		comm_timing->time += (time_1 - time_0);
		comm_timing->counter++;
	}
	#endif

	// ---------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Synchronizing matrix %s (all_gather, variable size: %i)... Done.\n",
			       matrix_name, variable_size_R );
	#endif

	return EXIT_SUCCESS;

} // sync_with_slaves

////////////////////////////////////////////////

/*
 * Performs a simple global reduction operation.
 *
 * Each position in array[0..(length-1)] is reduced by all processes in "comm".
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int reduce_from_slaves( real *restrict array, index_t length,
				#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_COMM))
					char const *restrict const name,
				#endif
				#if NMFGPU_PROFILING_COMM
					timing_data_t *restrict const comm_timing,
				#endif
				MPI_Comm comm )
{

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Reducing %s...\n", name );
	#endif

	// ---------------------------

	// All processes reduce each position in array[] and return the result to master.

	void *input_arg = (void *) ( process_id ? array : MPI_IN_PLACE );	// ( NOT_the_master ? array : MPI_IN_PLACE )

	void *output_arg = (void *) ((uintptr_t) (! process_id) * (uintptr_t) array);	// ( NOT_the_master ? NULL : array )


	#ifdef NMFGPU_PROFILING_COMM
		MPI_Barrier( comm );
		double const time_0 = MPI_Wtime();
	#endif

		#if NMFGPU_DEBUG || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_COMM))
			int const status =
		#endif

			MPI_Reduce( input_arg, output_arg, length, NMFGPU_MPI_REAL_T, MPI_SUM, 0, comm );

		#if NMFGPU_DEBUG || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_COMM))
			if ( status != MPI_SUCCESS ) {
				setMPIErrorString( status );
				print_error( sys_error_shown_by_all, "MPI error: Could not reduce '%s': %s.\n",
						name, error_string );
				return EXIT_FAILURE;
			}
		#endif

	#ifdef NMFGPU_PROFILING_COMM
	{
		MPI_Barrier( comm );
		double const time_1 = MPI_Wtime();
		comm_timing->time += (time_1 - time_0);
		comm_timing->counter++;
	}
	#endif

	// ---------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Reducing %s... Done.\n", name );
	#endif

	return EXIT_SUCCESS;

} // collect_from_slaves

////////////////////////////////////////////////

/*
 * Broadcast matrix to the given group of processes
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int broadcast_to_slaves( void *restrict matrix, index_t nrows, index_t pitch, bool real_data, MPI_Comm comm_R
				#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_COMM))
					, char const *restrict const matrix_name
				#endif
				#if NMFGPU_PROFILING_COMM
					, timing_data_t *restrict const comm_timing
				#endif
				)
{

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Broadcasting '%s' (real_data: %i) to slaves...\n", matrix_name, real_data );
	#endif

	// Type of MPI data to transfer
	MPI_Datatype const datatype = ( real_data ? NMFGPU_MPI_REAL_T : NMFGPU_MPI_INDEX_T );

	size_t const nitems = (size_t) nrows * (size_t) pitch;

	// ---------------------------

	/* The data block sent from the <i>-th process is received by process "0"
	 * and placed in the <i>-th block of matrix.
	 */

	#ifdef NMFGPU_PROFILING_COMM
		MPI_Barrier( comm_R );
		double const time_0 = MPI_Wtime();
	#endif

		#if NMFGPU_DEBUG || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_COMM))
			int const status =
		#endif

			MPI_Bcast( (void *)matrix, nitems, datatype, 0, comm_R );

		#if NMFGPU_DEBUG || ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_COMM))
			if ( status != MPI_SUCCESS ) {
				setMPIErrorString( status );
				print_error( sys_error_shown_by_all, "MPI error: Could not broadcast %s "
						"to slave processes: %s.\n", matrix_name, error_string );
				return EXIT_FAILURE;
			}
		#endif

	#ifdef NMFGPU_PROFILING_COMM
	{
		MPI_Barrier( comm_R );
		double const time_1 = MPI_Wtime();
		comm_timing->time += (time_1 - time_0);
		comm_timing->counter++;
	}
	#endif

	// ---------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Broadcasting '%s' to slaves...\n", matrix_name );
	#endif

	return EXIT_SUCCESS;

} // broadcast_to_slaves

////////////////////////////////////////////////

/*
 * Synchronizes a matrix on the GPU device with other processes:
 *	- Downloads (its portion of) the matrix from the device.
 *	- Performs a gather-to-all synchronization with processes in "comm_R".
 *	- Process "0" broadcasts the updated matrix to processes in group "diff", if necessary.
 *	- All ("active") processes upload the updated matrix.
 *
 * If "variable_size_R" is set to 'true', the length of both vectors is equals
 * the number of process in the group "comm_R" (i.e., <num_processes_R>).
 * Otherwise, the length of both vectors is '1'.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int sync_GPU_matrix( real *restrict A, real *restrict d_A, index_t nrows, index_t pitch, bool variable_size_R,
				index_t const *restrict offsets_pP, index_t const *restrict nitems_pP, MPI_Comm comm_R,
				bool in_group_R, index_t num_processes_R,
				#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
					index_t RpP,
				#endif
				#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 || (! NMFGPU_PROFILING_GLOBAL)
					index_t ncols, bool transpose, char const *restrict const matrix_name_A,
				#endif
				#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
					|| (! (NMFGPU_PROFILING_GLOBAL || NMFGPU_PROFILING_TRANSF) )
					char const *restrict const matrix_name_dA,
				#endif
				#if NMFGPU_PROFILING_TRANSF
					timing_data_t *restrict const download_timing, timing_data_t *restrict const upload_timing,
				#endif
				cudaStream_t stream_A, cudaEvent_t *restrict event_A
				#if NMFGPU_PROFILING_COMM
					, timing_data_t *restrict const comm_timing
				#endif
				)
{

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Synchronizing matrix %s...\n", matrix_name_A );
	#endif

	int status = EXIT_SUCCESS;

	// ---------------------------

	/* Processes in group "R":
	 *	- Download matrix from the device (just the portion updated by this process).
	 *	- Waits until "matrix" is ready.
	 *	- Synchronize matrix within them.
	 */

	if ( in_group_R ) {

		// Downloads "matrix" (just the portion updated by this process).

		// Offset to portion of data updated by this process.
		size_t const offset = offsets_pP[ (variable_size_R * process_id) ];	// ( variable_size_R ? [process_id] : [0] )

		// Items in such portion.
		size_t const nitems = nitems_pP[ (variable_size_R * process_id) ];	// ( variable_size_R ? [process_id] : [0] )

		// Pointers to the portion
		real *pA = &A[ offset ];
		real *p_dA = &d_A[ offset ];

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
			bool const real_data = true;
		#endif

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			download_matrix( pA, nitems, sizeof(real), p_dA,
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
						RpP, ncols, pitch, real_data, transpose, matrix_name_A,
					#endif
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
						|| ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_TRANSF))
						matrix_name_dA,
					#endif
					#if NMFGPU_PROFILING_TRANSF
						download_timing,
					#endif
					stream_A );

		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif

		// ---------------------------

		// Waits until GPU completes all ops. on <stream>.

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			sync_GPU( stream_A );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif

		// ---------------------------

		// Synchronizes matrix with processes in "comm_R"

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			sync_with_slaves( A, variable_size_R, offsets_pP, nitems_pP, comm_R
					#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 \
						|| (! (NMFGPU_PROFILING_GLOBAL || NMFGPU_PROFILING_COMM) )
						, matrix_name_A
					#endif
					#if NMFGPU_PROFILING_COMM
						, comm_timing
					#endif
					);

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif

	} // If in group "R"

	// ---------------------------

	/* Master process broadcasts to group "diff", if:
	 *	- Current process does NOT belong to group "R", so it is in "diff"
	 *	- Current process is the master, and group "R" is a subset of the "active" group.
	 */
	bool const broadcast_to_diff = (! in_group_R) || ((! process_id) * (num_processes_R < num_act_processes));

	if ( broadcast_to_diff ) {

		bool const real_data = true;

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			broadcast_to_slaves( A, nrows, pitch, real_data, comm_diff
						#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 \
							|| (! (NMFGPU_PROFILING_GLOBAL || NMFGPU_PROFILING_COMM) )
							, matrix_name_A
						#endif
						#if NMFGPU_PROFILING_COMM
							, comm_timing
						#endif
						);

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif

	} // if broadcast_to_diff

	// ---------------------------

	// All ("active") processes upload the updated matrix.

	#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || (! NMFGPU_PROFILING_GLOBAL)
		status =
	#endif

		upload_matrix( A, nrows, pitch, d_A,
				#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
					ncols, transpose,
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

	#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || (! NMFGPU_PROFILING_GLOBAL)
		if ( status != EXIT_SUCCESS )
			return EXIT_FAILURE;
	#endif

	// ---------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "Synchronizing matrix %s... Done.\n", matrix_name_A );
	#endif

	return EXIT_SUCCESS;

} // sync_GPU_matrix

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * NMF algorithm
 *
 * WARNING: This method is intended for "active" processes only
 * (i.e., process_id < num_act_processes == MAX(num_processes_N,num_processes_M) <= num_processes).
 *
 * Return EXIT_SUCCESS or EXIT_FAILURE.
 */
static int nmf( index_t nIters, index_t niter_test_conv, index_t stop_threshold )
{

	#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
		int status = EXIT_SUCCESS;
	#endif

	pBLN = 0;	// Current index in block_N.xxx[].
	pBLM = 0;	// Current index in block_M.xxx[].

	stepN = 1;	// Loop directions: +1 (forward) || -1 (backward).
	stepM = 1;	// Loop directions: +1 (forward) || -1 (backward).

	psNMF_N = 0;	// Current index in streams_NMF[].
	psNMF_M = 0;	// Current index in streams_NMF[].

	colIdx = 0;	// Current column index in Vcol. It corresponds to <bM + colIdx> in H and d_H.
	rowIdx = 0;	// Current row index in Vrow. It corresponds to <bN + rowIdx> in W and d_W.


	#if NMFGPU_PROFILING_GLOBAL
		// GPU time
		MPI_Barrier( comm_active );
		double const time_0 = MPI_Wtime();
	#endif

	// ----------------------------

	// Initializes the random-number generator.
	{
		index_t const seed = get_seed();

		/* The master process broadcasts the seed in order to have
		 * coherent initial values within all (active) processes.
		 */
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			broadcast_to_slaves( (void *)&seed, 1, 1, false, comm_active
						#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 || \
							((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_COMM))
							, "seed"
						#endif
						#if NMFGPU_PROFILING_COMM
							, &comm_others_timing
						#endif
						);

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			init_random( seed );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif
	}

	// ----------------------------

	// Initializes matrix W

	#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
		status =
	#endif

		set_random_values( d_W, N, K, Kp,
				#if NMFGPU_CPU_RANDOM
					W,
				#endif
				#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2
					false,		// NO matrix transposing
				#endif
				#if NMFGPU_CPU_RANDOM && (NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
					|| (! NMFGPU_PROFILING_GLOBAL))
					"W",
				#endif
				#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 || (NMFGPU_CPU_RANDOM && NMFGPU_DEBUG_TRANSF) \
					|| ((! NMFGPU_CPU_RANDOM) && (! NMFGPU_PROFILING_GLOBAL))
					"d_W",
				#endif
				#if ( NMFGPU_CPU_RANDOM && NMFGPU_PROFILING_TRANSF )
					&upload_W_timing,
				#endif
				stream_W, &event_W );

	#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
		if ( status != EXIT_SUCCESS ) {
			destroy_random();
			return EXIT_FAILURE;
		}
	#endif

	// ----------------------------

	// Initializes matrix H, group "M" only.

	if ( in_group_M ) {

		/* In fixed-seed or debug mode, initializes the entire matrix H.
		 * Otherwise, it just initializes the corresponding portion.
		 */

		// Offset to portion of data initialized by this process.
		#if NMFGPU_DEBUG || NMFGPU_FIXED_INIT	// The entire matrix
			size_t const offset = 0;
		#else	// Just the corresponding portion (i.e., bM * pitch)
			size_t const offset = offsets_HpP[ (variable_size_M * process_id) ];	// ( variable_size_M ? [process_id] : [0] )
		#endif

		real *const p_dH = &d_H[ offset ];
		#if NMFGPU_CPU_RANDOM
			real *const pH = &H[ offset ];
		#endif

		// Number of rows
		#if NMFGPU_DEBUG || NMFGPU_FIXED_INIT
			index_t const nrows = M;	// The entire matrix
		#else
			index_t const nrows = MpP;	// Just the corresponding portion
		#endif


		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			set_random_values( p_dH, nrows, K, Kp,
					#if NMFGPU_CPU_RANDOM
						pH,
					#endif
					#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 || (NMFGPU_CPU_RANDOM && NMFGPU_DEBUG_TRANSF)
						true,		// Matrix transposing
					#endif
					#if NMFGPU_CPU_RANDOM && (NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
						|| (! NMFGPU_PROFILING_GLOBAL))
						"H",
					#endif
					#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 || (NMFGPU_CPU_RANDOM && NMFGPU_DEBUG_TRANSF) \
						|| ((! NMFGPU_CPU_RANDOM) && (! NMFGPU_PROFILING_GLOBAL))
						"d_H",
					#endif
					#if ( NMFGPU_CPU_RANDOM && NMFGPU_PROFILING_TRANSF )
						&upload_H_timing,
					#endif
					streams_NMF[ psNMF_M ], NULL );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS ) {
				destroy_random();
				return EXIT_FAILURE;
			}
		#endif
	}

	// ----------------------------

	// Finalizes the random-number generator.

	destroy_random();

	// ----------------------------

	// Uploads matrix V
	{
		// Block configuration.
		#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
			index_t const BLM = block_M.BL[ pBLM ];		// Number of columns.
		#endif
		index_t const BLMp = block_M.BLp[ pBLM ];		// Number of columns (with padding).
		index_t const BLN  = block_N.BL[ pBLN ];		// Number of rows.

		// d_Vcol (group "M" only)
		if ( (d_Vcol != d_Vrow) * in_group_M ) {

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				status =
			#endif

				upload_matrix_partial( Vcol, N, MpPp, 0, colIdx,	// Starting row: 0
							#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
								BLM,
							#endif
							#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
								|| (! NMFGPU_PROFILING_GLOBAL)
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

		} // if (d_Vcol != d_Vrow) && in_group_M

		// ----------------------------

		// d_Vrow (group "N" only)

		if ( in_group_N ) {

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				status =
			#endif

				upload_matrix_partial( Vrow, BLN, Mp, rowIdx, 0,	// Starting column: 0
							#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
								M,
							#endif
							#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
								|| (! NMFGPU_PROFILING_GLOBAL)
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

	} // uploads input matrix.

	// ----------------------------

	// Number of iterations

	index_t const niter_conv = (nIters / niter_test_conv);		// Number of times to perform test of convergence.
	index_t const niter_rem  = (nIters % niter_test_conv);		// Remaining iterations.
	bool converged = false;

	#if NMFGPU_VERBOSE
		print_message( verb_shown_by_all, "niter_test_conv=%" PRI_IDX ", niter_conv=%" PRI_IDX ", niter_rem=%" PRI_IDX ".\n",
				niter_test_conv, niter_conv, niter_rem );
	#endif

	print_message( shown_by_all, "\nStarting NMF( K=%"PRI_IDX" )...\n", K );
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
			if ( in_group_M ) {
				#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
					status =
				#endif

					update_H();

				#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
					if ( status != EXIT_SUCCESS )
						return EXIT_FAILURE;
				#endif
			}

			// ----------------------------

			/* Synchronizes the updated matrix with all other
			 * (active) processes.
			 *
			 * That is:
			 *	- Downloads (its portion of) the matrix from the device.
			 *	- Performs a gather-to-all synchronization with processes in "comm_M".
			 *	- Process "0" broadcasts the updated matrix to processes in group "diff", if necessary.
			 *	- All ("active") processes upload the updated matrix.
			 */
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				status =
			#endif

				sync_GPU_matrix( H, d_H, M, Kp, variable_size_M, offsets_HpP, nitems_HpP, comm_M, in_group_M,
						num_processes_M,
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
							MpP,
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
							|| (! NMFGPU_PROFILING_GLOBAL)
							K, true, "H",		// Matrix transposing
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
							|| (! (NMFGPU_PROFILING_GLOBAL || NMFGPU_PROFILING_TRANSF) )
							"d_H",
						#endif
						#if NMFGPU_PROFILING_TRANSF
							&download_H_timing, &upload_H_timing,
						#endif
						stream_H, &event_H
						#if NMFGPU_PROFILING_COMM
							, &comm_H_timing
						#endif
						);

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					return EXIT_FAILURE;
			#endif

			// ----------------------------


			#ifdef NMFGPU_DEBUG
			///////////////////////////////
				print_message( verb_shown_by_all, "\n------------ iter=%i, loop %i (niter_test_conv) Matrix W: ------------\n",
						iter,i);
			/////////////////////////////
			#endif

			/*
			 * WH(BLN,Mp) = W(BLN,Kp) * H
			 * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
			 * Waux(BLN,Kp) = WH(BLN,Mp) * H'
			 * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
			 */
			if ( in_group_N ) {
				#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
					status =
				#endif

					update_W();

				#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
					if ( status != EXIT_SUCCESS )
						return EXIT_FAILURE;
				#endif
			}

			// ----------------------------

			/* Synchronizes the updated matrix with all other
			 * (active) processes.
			 *
			 * That is:
			 *	- Downloads (its portion of) the matrix from the device.
			 *	- Performs a gather-to-all synchronization with processes in "comm_N".
			 *	- Process "0" broadcasts the updated matrix to processes in group "diff", if necessary.
			 *	- All ("active") processes upload the updated matrix.
			 */
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				status =
			#endif

				sync_GPU_matrix( W, d_W, N, Kp, variable_size_N, offsets_WpP, nitems_WpP, comm_N, in_group_N,
						num_processes_N,
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
							NpP,
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
							|| (! NMFGPU_PROFILING_GLOBAL)
							K, false, "W",		// No matrix transposing
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
							|| (! (NMFGPU_PROFILING_GLOBAL || NMFGPU_PROFILING_TRANSF) )
							"d_W",
						#endif
						#if NMFGPU_PROFILING_TRANSF
							&download_W_timing, &upload_W_timing,
						#endif
						stream_W, &event_W
						#if NMFGPU_PROFILING_COMM
							, &comm_W_timing
						#endif
						);

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					return EXIT_FAILURE;
			#endif


		} // for niter_test_conv times.

		// -------------------------------------

		// Adjusts matrices W and H.

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			matrix_adjust( d_H, M, Kp,
					#if NMFGPU_DEBUG
						K, true, 	// Matrix transposing
					#endif
					#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
						"d_H",
					#endif
					stream_H, NULL );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif

		// ----------------------------

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			matrix_adjust( d_W, N, Kp,
					#if NMFGPU_DEBUG
						K, false,	// No matrix transposing
					#endif
					#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
						"d_W",
					#endif
					stream_W, &event_W );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif

		// -------------------------------------

		// Test of convergence

		// Computes classification vector

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			get_classification( d_classification, classification );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif

		// ----------------------------

		// Computes differences

		size_t const diff = get_difference( classification, last_classification, M );

			#ifdef NMFGPU_DEBUG
			///////////////////////////////
				print_message( dbg_shown_by_all, "Returned difference between classification vectors: %zu\n", diff );
			/////////////////////////////
			#endif

		// -------------------------------------

		// Saves the new classification.
		{
			// It just swaps the pointers.
			index_t *restrict const h_tmp = classification;
			classification = last_classification;
			last_classification = h_tmp;

			/* If host memory was mapped into the address space of the device,
			 * pointers in device memory must also be swapped.
			 */
			if ( mappedHostMemory ) {
				index_t *restrict const d_tmp = d_classification;
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
			if ( in_group_M ) {
				#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
					status =
				#endif

					update_H();

				#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
					if ( status != EXIT_SUCCESS )
						return EXIT_FAILURE;
				#endif
			}

			// ----------------------------

			/* Synchronizes the updated matrix with all other
			 * (active) processes.
			 *
			 * That is:
			 *	- Downloads (its portion of) the matrix from the device.
			 *	- Performs a gather-to-all synchronization with processes in "comm_M".
			 *	- Process "0" broadcasts the updated matrix to processes in group "diff", if necessary.
			 *	- All ("active") processes upload the updated matrix.
			 */
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				status =
			#endif

				sync_GPU_matrix( H, d_H, M, Kp, variable_size_M, offsets_HpP, nitems_HpP, comm_M, in_group_M,
						num_processes_M,
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
							MpP,
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
							|| (! NMFGPU_PROFILING_GLOBAL)
							K, true, "H",		// Matrix transposing
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
							|| (! (NMFGPU_PROFILING_GLOBAL || NMFGPU_PROFILING_TRANSF) )
							"d_H",
						#endif
						#if NMFGPU_PROFILING_TRANSF
							&download_H_timing, &upload_H_timing,
						#endif
						stream_H, &event_H
						#if NMFGPU_PROFILING_COMM
							, &comm_H_timing
						#endif
						);

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					return EXIT_FAILURE;
			#endif

			// ----------------------------

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
			if ( in_group_N ) {
				#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
					status =
				#endif

					update_W();

				#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
					if ( status != EXIT_SUCCESS )
						return EXIT_FAILURE;
				#endif
			}

			// ----------------------------

			/* Synchronizes the updated matrix with all other
			 * (active) processes.
			 *
			 * That is:
			 *	- Downloads (its portion of) the matrix from the device.
			 *	- Performs a gather-to-all synchronization with processes in "comm_N".
			 *	- Process "0" broadcasts the updated matrix to processes in group "diff", if necessary.
			 *	- All ("active") processes upload the updated matrix.
			 */
			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				status =
			#endif

				sync_GPU_matrix( W, d_W, N, Kp, variable_size_N, offsets_WpP, nitems_WpP, comm_N, in_group_N,
						num_processes_N,
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
							NpP,
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
							|| (! NMFGPU_PROFILING_GLOBAL)
							K, false, "W",		// No matrix transposing
						#endif
						#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
							|| (! (NMFGPU_PROFILING_GLOBAL || NMFGPU_PROFILING_TRANSF) )
							"d_W",
						#endif
						#if NMFGPU_PROFILING_TRANSF
							&download_W_timing, &upload_W_timing,
						#endif
						stream_W, &event_W
						#if NMFGPU_PROFILING_COMM
							, &comm_W_timing
						#endif
						);

			#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
				if ( status != EXIT_SUCCESS )
					return EXIT_FAILURE;
			#endif

		} // for niter_rem times.

	} // if has not yet converged.

	#if NMFGPU_VERBOSE
		print_message( verb_shown_by_all, "Done.\n" );
	#endif

	// --------------------------------

	// Number of iterations performed.

	if ( converged ) {
		index_t const num_iter_performed = iter * niter_test_conv;
		print_message( shown_by_all, "\nNMF: Algorithm converged in %" PRI_IDX " iterations.\n", num_iter_performed );
	} else
		print_message( shown_by_all, "\nNMF: %" PRI_IDX " iterations performed.\n", nIters );


	// Master process downloads output matrices, only if no "remaining" iterations were performed...
	if ( (! process_id) * (converged + (! niter_rem)) ) {

		bool const real_data = true;
		size_t const data_size = sizeof(real);
		size_t nitems = (size_t) M * (size_t) Kp;

		// d_H
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			download_matrix( H, nitems, data_size, d_H,
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
						M, K, Kp, real_data, true, "H",		// Matrix transposing
					#endif
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
						|| ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_TRANSF))
						"d_H",
					#endif
					#if NMFGPU_PROFILING_TRANSF
						&download_H_timing,
					#endif
					stream_H );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif

		// ----------------------------

		// d_W
		nitems = (size_t) N * (size_t) Kp;
		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			download_matrix( W, nitems, data_size, d_W,
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2
						N, K, Kp, real_data, false, "W",	// NO matrix transposing
					#endif
					#if NMFGPU_DEBUG || NMFGPU_DEBUG_TRANSF || NMFGPU_VERBOSE_2 \
						|| ((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_TRANSF))
						"d_W",
					#endif
					#if NMFGPU_PROFILING_TRANSF
						&download_W_timing,
					#endif
					stream_W );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif

	} // if master

	// --------------------------------

	/* Checks results: (group "N" only)
	 *
	 * Computes the "distance" between V and W*H as follow:
	 *
	 *	distance = norm( V - (W*H) ) / norm( V ),
	 * where
	 *	norm( X ) = sqrt( dot_X )
	 *	dot_V	 <-- dot_product( V, V )
	 *	dot_VWH  <-- dot_product( V-(W*H), V-(W*H) )
	 */
	if ( in_group_N ) {

		real dot_V = REAL_C( 0.0 );
		real dot_VWH = REAL_C( 0.0 );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

			dot_product_VWH( &dot_V, &dot_VWH );	// Per-process partial results

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif

		// ----------------------------

		// Final results

		real dotsVWH[ 2 ] = { dot_V, dot_VWH };

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			status =
		#endif

		reduce_from_slaves( dotsVWH, 2,
					#if NMFGPU_DEBUG || NMFGPU_VERBOSE_2 || \
						((! NMFGPU_PROFILING_GLOBAL) && (! NMFGPU_PROFILING_COMM))
						"dots(V and WH)",
					#endif
					#if NMFGPU_PROFILING_COMM
						&comm_others_timing,
					#endif
					comm_N );

		#if NMFGPU_DEBUG || (! NMFGPU_PROFILING_GLOBAL)
			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		#endif

		dot_V = dotsVWH[ 0 ];
		dot_VWH = dotsVWH[ 1 ];

		// ----------------------------

		// Shows results

		#if NMFGPU_DEBUG || NMFGPU_VERBOSE
		///////////////////////////////
			print_message( verb_shown_by_all, "Result: dot_V=%g (norm: %g), dot_VWH=%g (norm: %g)\n",
					dot_V, SQRTR( dot_V ), dot_VWH, SQRTR( dot_VWH ) );
		///////////////////////////////
		#endif

		print_message( shown_by_all, "Distance between V and W*H: %g\n", SQRTR( dot_VWH ) / SQRTR( dot_V ) );

	} // group "N"

	// --------------------------------

	#ifdef NMFGPU_PROFILING_GLOBAL
	{
		// GPU time
		MPI_Barrier( comm_active );
		double const time_1 = MPI_Wtime();
		double const total_gpu_time = (time_1 - time_0);
		print_message( shown_by_all, "\nGPU + classification + check_result time: %g seconds.\n", total_gpu_time );
	}
	#endif

	// ----------------------------

	return EXIT_SUCCESS;

} // nmf

////////////////////////////////////////////////

/*
 * Writes output matrices
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int write_matrices( const char *restrict filename, file_fmt_t output_file_fmt, struct matrix_tags_t mt )
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

	// File extension:
	char const *restrict const p_file_ext = file_extension[ output_file_fmt ];

	char *restrict const filename_out = (char *restrict) malloc( (strlen(filename) + strlen(p_file_ext))*sizeof(char) );
	if ( ! filename_out ) {
		print_errnum( sys_error_shown_by_all, errno, "Error allocating memory for output filename" );
		clean_tag( tag_factors );
		return EXIT_FAILURE;
	}

	// -----------------------------

	// Matrix W

	errno = 0;
	if ( sprintf( filename_out, "%s_W%s", filename, p_file_ext ) <= 0 ) {
		print_errnum( sys_error_shown_by_all, errno, "Error setting output filename for matrix W" );
		free( filename_out );
		clean_tag( tag_factors );
		return EXIT_FAILURE;
	}

	transpose = false;
	verbose = true;
	status = matrix_save( filename_out, output_file_fmt, W, N, K, Kp, transpose, &mt_W, verbose );
	if ( status != EXIT_SUCCESS ) {
		print_error( error_shown_by_all, "Error writing matrix W.\n" );
		free( filename_out );
		clean_tag( tag_factors );
		return EXIT_FAILURE;
	}

	// -----------------------------

	// Matrix H

	errno = 0;
	if ( sprintf( filename_out, "%s_H%s", filename, p_file_ext ) <= 0 ) {
		print_errnum( sys_error_shown_by_all, errno, "Error setting output filename for matrix H" );
		free( filename_out );
		clean_tag( tag_factors );
		return EXIT_FAILURE;
	}

	transpose = true;
	verbose = false;
	status = matrix_save( filename_out, output_file_fmt, H, M, K, Kp, transpose, &mt_H, verbose );
	if ( status != EXIT_SUCCESS )
		print_error( error_shown_by_all, "Error writing matrix H.\n" );

	// -----------------------------

	free( filename_out );

	clean_tag( tag_factors );

	return status;

} // write_matrices

////////////////////////////////////////////////

/*
 * Shows time elapsed on MPI operations.
 */
static void show_comm_times( void )
{

	#if NMFGPU_PROFILING_COMM

		bool const show_secs = true;	// Shows elapsed time in seconds.

		// --------------------

		if ( comm_W_timing.counter )
			print_elapsed_time( "\tSynchronize matrix W", &comm_W_timing, sizeof(real), show_secs, shown_by_all );

		if ( comm_H_timing.counter )
			print_elapsed_time( "\tSynchronize matrix H", &comm_H_timing, sizeof(real), show_secs, shown_by_all );

		if ( comm_others_timing.counter )
			print_elapsed_time( "\tSynchronize other data", &comm_others_timing, sizeof(real), show_secs, shown_by_all );

		float const total_data_comm = comm_W_timing.time + comm_H_timing.time + comm_others_timing.time;

		append_printed_message( shown_by_all, "Total communications time: %g ms\n\n", total_data_comm );

	#endif	/* NMFGPU_PROFILING_COMM */

} // show_comm_times

////////////////////////////////////////////////

/*
 * Finalizes MPI data.
 */
static void finalize_mpi_data( void )
{

	if ( nitems_HpP )  { free( (void *) nitems_HpP ); nitems_HpP = NULL; }
	if ( offsets_HpP ) { free( (void *) offsets_HpP ); offsets_HpP = NULL; }

	if ( nitems_WpP )  { free( (void *) nitems_WpP ); nitems_WpP = NULL; }
	if ( offsets_WpP ) { free( (void *) offsets_WpP ); offsets_WpP = NULL; }

	MPI_Comm_free( &comm_idle );
	MPI_Comm_free( &comm_active );
	MPI_Comm_free( &comm_diff );
	MPI_Comm_free( &comm_M );
	MPI_Comm_free( &comm_N );

} // finalize_mpi_data

////////////////////////////////////////////////

/*
 * Frees all NON-MPI data, and shuts down the GPU device.
 */
static void finalize_NMF( void )
{

	#if NMFGPU_VERBOSE_2
		bool const all_processes = verb_shown_by_all;
	#elif NMFGPU_DEBUG
		bool const all_processes = dbg_shown_by_all;
	#endif

	#if NMFGPU_VERBOSE_2 || NMFGPU_DEBUG
		print_message( all_processes, "Shutting down NMF...\n" );
	#endif

	// ----------------------------

	freeHostMemory( last_classification, "previous classification vector" );
	freeHostMemory( classification, "classification vector" );
	freeHostMemory( H, "H" ); freeHostMemory( W, "W" );

	freeHostMemory( Vrow, "Vrow" );
	if ( Vrow != Vcol ) { freeHostMemory( Vcol, "Vcol" ); }

	last_classification =  classification = NULL;
	H = W = Vcol = Vrow = NULL;

	finalize_GPU_device();

	// ----------------------------

	#if NMFGPU_VERBOSE_2 || NMFGPU_DEBUG
		print_message( all_processes, "Shutting down NMF... Done.\n" );
	#endif

} // finalize_NMF

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

int main( int argc, char *argv[] )
{

	#if NMFGPU_PROFILING_COMM
		// Communication timing
		comm_W_timing = new_empty_timing_data();
		comm_H_timing = new_empty_timing_data();
		comm_others_timing = new_empty_timing_data();
	#endif

	int status = EXIT_SUCCESS;

	process_id = 0;		// Global variables.
	num_act_processes = num_processes = 1;

	// Initializes MPI.
	status = MPI_Init( &argc, &argv );
	if ( status != MPI_SUCCESS ) {
		setMPIErrorString( status );
		print_error( sys_error_shown_by_all, "Failed to initialize MPI: %s\n", error_string );
		return EXIT_FAILURE;
	}

	MPI_Comm_rank( MPI_COMM_WORLD, (int *)&process_id );
	MPI_Comm_size( MPI_COMM_WORLD, (int *)&num_processes );
	num_act_processes = num_processes;

	MPI_Comm_set_errhandler( MPI_COMM_WORLD, MPI_ERRORS_RETURN );

	/* Default limits for matrix dimensions.
	 * NOTE: The MPI standard uses 'int' for almost all parameters, including arrays size
	 *	 They may be later adjusted (i.e., reduced), at device initialization.
	 */
	set_matrix_limits( INDEX_C(0), INT_MAX, INT_MAX );

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
	if ( check_arguments( argc, (char const *restrict *restrict) argv, &help, &arguments ) != EXIT_SUCCESS ) {
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	// If help was requested, just prints a help message and returns.
	if ( help ) {
		status = print_nmf_gpu_help( argv[0] );
		MPI_Finalize();
		return status;
	}

	char const *restrict const filename = arguments.filename;	// Input filename
	bool const numeric_hdrs = arguments.numeric_hdrs;		// Has numeric columns headers.
	bool const numeric_lbls = arguments.numeric_lbls;		// Has numeric row labels.
	file_fmt_t const input_file_fmt = arguments.input_file_fmt;	// Input  file format.
	file_fmt_t const output_file_fmt = arguments.output_file_fmt;	// Output file format.
	K = arguments.k;						// Factorization rank.
	Kp = arguments.kp;						// Padded factorization rank.
	index_t const nIters = arguments.nIters;			// Maximum number of iterations per run.
	index_t const niter_test_conv = arguments.niter_test_conv;	// Number of iterations before testing convergence.
	index_t const stop_threshold = arguments.stop_threshold;	// Stopping criterion.
	index_t const gpu_device = arguments.gpu_device;		// Device ID.

	// Compute classification vector?
	bool const do_classf = ( nIters >= niter_test_conv );

	// ----------------------------------------

	print_message( shown_by_all, "\t<<< NMF-mGPU: Non-negative Matrix Factorization on GPU >>>\n"
					"\t\t\t\tMulti-GPU version\n" );

	#if NMFGPU_PROFILING_GLOBAL
		// Global time
		MPI_Barrier( MPI_COMM_WORLD );
		double const global_starting_time = MPI_Wtime();
	#endif

	// ----------------------------------------

	/* Initializes the GPU device.
	 *
	 * In addition:
	 *	- Updates memory_alignment according to the selected GPU device.
	 *	- Updates Kp (i.e., the padded factorization rank).
	 *	- Updates the limits of matrix dimensions.
	 */
	size_t const mem_size = initialize_GPU( (process_id + gpu_device), K );
	if ( ! mem_size ) {
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
		return EXIT_FAILURE;
	}

	// ----------------------------------------

	// Reads input matrix

	struct matrix_tags_t mt = new_empty_matrix_tags();

	status = init_V( filename, numeric_hdrs, numeric_lbls, input_file_fmt, &mt );
	if ( status != EXIT_SUCCESS ) {
		shutdown_GPU();
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
		return EXIT_FAILURE;
	}

	// ----------------------------------------

	// Fails if the factorization rank is too large.
	if ( K > MIN( N, M ) ) {
		print_error( error_shown_by_all, "Error: invalid factorization rank: K=%" PRI_IDX ".\nIt must not be greater "
				"than any of matrix dimensions.\n", K );
		clean_matrix_tags( mt );
		finalize_NMF();
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	// ----------------------------------------

	// Initializes other MPI data

	// MPI communicators
	status = setup_communicators();

	if ( status != EXIT_SUCCESS ) {
		clean_matrix_tags( mt );
		finalize_NMF();
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
		return EXIT_FAILURE;
	}

	// Non-"active" (i.e., idle) processes do not need to go further...
	if ( ! in_group_active ) {
		#if NMFGPU_DEBUG
			print_message( dbg_shown_by_all, "Idle process. Stopping...\n" );
		#endif
		clean_matrix_tags( mt );
		finalize_NMF();
		MPI_Barrier( comm_idle );	// Waits until master finishes.
		finalize_mpi_data();
		MPI_Finalize();
		return EXIT_SUCCESS;
	}

	// dimension vectors
	if (	in_group_N &&
		(get_dim_vectors( "W", N, Kp, variable_size_N, NpP_base, num_processes_N, &offsets_WpP, &nitems_WpP ) != EXIT_SUCCESS) ) {
		clean_matrix_tags( mt );
		finalize_NMF();
		if ( (num_act_processes < num_processes) * (! process_id) )
			MPI_Barrier( comm_idle ); // Master process "releases" the idle processes.
		finalize_mpi_data();
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
		return EXIT_FAILURE;
	}

	if (	in_group_M &&
		(get_dim_vectors( "H", M, Kp, variable_size_M, MpP_base, num_processes_M, &offsets_HpP, &nitems_HpP ) != EXIT_SUCCESS) ) {
		clean_matrix_tags( mt );
		finalize_NMF();
		if ( (num_act_processes < num_processes) * (! process_id) )
			MPI_Barrier( comm_idle ); // Master process "releases" the idle processes.
		finalize_mpi_data();
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
		return EXIT_FAILURE;
	}

	// ----------------------------------------

	// Setups the GPU device

	status = setup_GPU( mem_size, do_classf );

	if ( status != EXIT_SUCCESS ) {
		freeHostMemory( Vrow, "Vrow" ); if ( Vrow != Vcol ) { freeHostMemory( Vcol, "Vcol" ); }
		clean_matrix_tags( mt );
		finalize_NMF();
		if ( (num_act_processes < num_processes) * (! process_id) )
			MPI_Barrier( comm_idle ); // Master process "releases" the idle processes.
		finalize_mpi_data();
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
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
			clean_matrix_tags( mt );
			finalize_NMF();
			if ( (num_act_processes < num_processes) * (! process_id) )
				MPI_Barrier( comm_idle ); // Master process "releases" the idle processes.
			finalize_mpi_data();
			MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
			return EXIT_FAILURE;
		}

		nitems = (size_t) M * (size_t) Kp;
		H = (real *restrict) getHostMemory( nitems * sizeof(real), false, false );
		if ( ! H ) {
			print_error( sys_error_shown_by_all, "Error allocating memory for HOST matrix H (M=%" PRI_IDX
					", Kp=%" PRI_IDX ").\n", M, Kp );
			clean_matrix_tags( mt );
			finalize_NMF();
			if ( (num_act_processes < num_processes) * (! process_id) )
				MPI_Barrier( comm_idle ); // Master process "releases" the idle processes.
			finalize_mpi_data();
			MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
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
			clean_matrix_tags( mt );
			finalize_NMF();
			if ( (num_act_processes < num_processes) * (! process_id) )
				MPI_Barrier( comm_idle ); // Master process "releases" the idle processes.
			finalize_mpi_data();
			MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
			return EXIT_FAILURE;
		}

		last_classification = (index_t *restrict) getHostMemory( Mp * sizeof(index_t), false, true );	// Initializes with zeros
		if ( ! last_classification ) {
			print_error( sys_error_shown_by_all, "Error allocating memory for HOST classification vector (last, M=%" PRI_IDX
					", Mp=%" PRI_IDX ").\n", M, Mp );
			clean_matrix_tags( mt );
			finalize_NMF();
			if ( (num_act_processes < num_processes) * (! process_id) )
				MPI_Barrier( comm_idle ); // Master process "releases" the idle processes.
			finalize_mpi_data();
			MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
			return EXIT_FAILURE;
		}

	} // do_classf

	// ----------------------------------------

	// Executes the NMF Algorithm

	status = nmf( nIters, niter_test_conv, stop_threshold );
	if ( status != EXIT_SUCCESS ) {
		clean_matrix_tags( mt );
		finalize_NMF();
		if ( (num_act_processes < num_processes) * (! process_id) )
			MPI_Barrier( comm_idle ); // Master process "releases" the idle processes.
		finalize_mpi_data();
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
		return EXIT_FAILURE;
	}

	// ----------------------------------------

	// Writes output matrices.

	if ( ! process_id ) {
		status = write_matrices( filename, output_file_fmt, mt );

		if ( status != EXIT_SUCCESS ) {
			clean_matrix_tags( mt );
			finalize_NMF();
			if ( num_act_processes < num_processes )
				MPI_Barrier( comm_idle ); // Master process "releases" the idle processes.
			finalize_mpi_data();
			MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
			return EXIT_FAILURE;
		}
	}

	// ----------------------------------------

	// Show elapsed time:

	#if NMFGPU_PROFILING_GLOBAL
	{
		// Total elapsed time
		MPI_Barrier( comm_active );
		double const global_finalizing_time = MPI_Wtime();
		double const total_nmf_time = (global_finalizing_time - global_starting_time);
		print_message( shown_by_all, "\nTotal elapsed time: %g seconds.\n", total_nmf_time );
	}
	#endif

	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
		print_message( shown_by_all, "Time elapsed on GPU operations:\n" );
		show_kernel_times();
		show_transfer_times();
	#endif

	#if NMFGPU_PROFILING_COMM
		print_message( shown_by_all, "Time elapsed on MPI operations:\n" );
	#endif
		show_comm_times();

	// ----------------------------------------

	clean_matrix_tags( mt );

	finalize_NMF();

	if ( (num_act_processes < num_processes) * (! process_id) )
		MPI_Barrier( comm_idle ); // Master process "releases" the idle processes.

	finalize_mpi_data();

	MPI_Finalize();

	return EXIT_SUCCESS;

} // main

////////////////////////////////////////////////
////////////////////////////////////////////////
