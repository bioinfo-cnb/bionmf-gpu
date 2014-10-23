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
 * common.c
 *	Some generic definitions, constants, macros and functions used by bioNMF-GPU.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Additional information:
 *		NMFGPU_VERBOSE_2: Shows the parameters in some routine calls.
 *
 *	Debug / Testing:
 *		NMFGPU_FIXED_INIT: Uses "random" values generated from a fixed seed (defined in common.h).
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

// Required by <stdint.h>
#ifndef __STDC_CONSTANT_MACROS
	#define __STDC_CONSTANT_MACROS (1)
#endif

#include "common.h"

#include <stdlib.h>
#include <unistd.h>		/* getopt */
#include <errno.h>
#include <string.h>
#include <stdarg.h>		/* vfprintf */
#include <stdio.h>
#include <ctype.h>		/* isprint */
#include <inttypes.h>		/* strtoimax, INTMAX_C, uintptr_t */
#if ! NMFGPU_FIXED_INIT
	#include <sys/time.h>	/* gettimeofday */
#endif


////////////////////////////////////////////////
////////////////////////////////////////////////

/* Constants */

// Fixed seed for the random-values generator.
#ifndef FIXED_SEED
	#define FIXED_SEED ( INDEX_C(3) )
#endif

// Default alignment for data on memory.
#ifndef DEFAULT_MEMORY_ALIGNMENT		/* 64 bytes, expressed in real-type items. */
	#define DEFAULT_MEMORY_ALIGNMENT ( (index_t) ( 64 / sizeof(real) ) )
#endif

// Default values of some input parameters.
#ifndef DEFAULT_K
	#define DEFAULT_K ( INDEX_C(2) )
#endif

#ifndef DEFAULT_NITERS
	#define DEFAULT_NITERS ( INDEX_C(2000) )
#endif

#ifndef DEFAULT_NITER_CONV
	#define DEFAULT_NITER_CONV ( INDEX_C(10) )
#endif

#ifndef DEFAULT_STOP_THRESHOLD
	#define DEFAULT_STOP_THRESHOLD ( INDEX_C(40) )
#endif

#ifndef DEFAULT_GPU_DEVICE
	#define DEFAULT_GPU_DEVICE ( INDEX_C(0) )
#endif


// ---------------------------------------------
// ---------------------------------------------

/* Global variables */

index_t process_id = 0;		// Current process ID.
index_t num_processes = 1;	// (Maximum) Number of processes on the system.
index_t num_act_processes = 1;	//  Number of "active" (i.e., not-idle) processes (<= num_processes).

// Matrix dimension limits (NOTE: they may be modified if the program is executed in a GPU device).
index_t memory_alignment = 1;										// Data alignment on memory.
size_t matrix_max_num_items = SIZE_MAX / ( 2 * sizeof(real) );						// Maximum number of items in a matrix.
index_t matrix_max_pitch = (index_t) MIN( (SIZE_MAX / (4* sizeof(real))), (size_t) IDX_MAX );		// Maximum multiple of <memory_alignment>.
index_t matrix_max_non_padded_dim = (index_t) MIN( (SIZE_MAX / (4* sizeof(real))), (size_t) IDX_MAX );	// Maximum non-padded dimension.

// Matrix dimensions:
index_t N = 0;		// Number of rows of input matrix V.
index_t M = 0;		// Number of columns of input matrix V.
index_t K = 0;		// Factorization rank.

// Dimensions for multi-process version:
index_t NpP = 0;	// Number of rows of V assigned to this process (NpP <= N).
index_t MpP = 0;	// Number of columns of V assigned to this process (MpP <= M).
index_t bN = 0;		// Starting row ((bN + NpP) <= N).
index_t bM = 0;		// Starting column ((bM + MpPp) <= Mp).

// Padded dimensions:
index_t Mp = 0;		// <M> rounded up to the next multiple of <memory_alignment>.
index_t Kp = 0;		// <K> rounded up to the next multiple of <memory_alignment>.
index_t MpPp = 0;	// <MpP> rounded up to the next multiple of <memory_alignment> (MpPp <= Mp).

// Classification vectors.
index_t *restrict classification = NULL;
index_t *restrict last_classification = NULL;

// Data matrices (host side)
real *restrict W = NULL;
real *restrict H = NULL;
real *restrict Vcol = NULL;	// Block of NpP rows from input matrix V.
real *Vrow = NULL;		// Block of MpP columns from input matrix V.

// ---------------------------------------------

/* "Private" global variables. */

// Information and/or error messages shown by all processes.
#if NMFGPU_VERBOSE_2
	static bool const verb_shown_by_all = false;	// Information messages in verbose mode.
#endif
static bool const shown_by_all = false;			// Information messages in non-debug mode.
static bool const sys_error_shown_by_all = true;	// System error messages.
static bool const error_shown_by_all = false;		// Error messages on invalid arguments or I/O data.

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Prints the given message composed by the format string "fmt" and the
 * arguments list "args", if any.
 *
 * If "show_id" is 'true', the string is prefixed by a newline ('\n') character
 * and the process ID.
 *
 * Output mode (err_mode == false):
 *	- The string is printed to the standard output stream ('stdout').
 *	- "errnum" is IGNORED.
 *
 * Error mode (err_mode == true):
 *	- The standard output stream ('stdout') is flushed for all processes.
 *	- The message is printed to the standard error stream ('stderr').
 *	- If "errnum" is non-zero, this function behaves similar to perror(3).
 *	  That is, it appends to the message a colon, the string given by
 *	  strerror(errnum), and a newline character. In contrast, if "errnum"
 *	  is zero, just prints the newline.
 *
 * Note that error messages from this function (i.e., if it fails) are always
 * prefixed and suffixed with a newline character (as well as the process ID
 * on multi-processes system), REGARDLESS of the arguments.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int print_string( bool all_processes, bool show_id, bool err_mode, int errnum, char const *restrict const fmt, va_list args )
{

	if ( ! fmt ) {
		fflush( stdout );
		errno = EFAULT;
		if ( ! process_id )
			perror("\nprint_string( fmt )");
		return EXIT_FAILURE;
	}

	errno = 0;
	int error = 0;

	// Output stream
	FILE *const file = ( err_mode ? stderr : stdout );

	// --------------------------

	// Error mode: flushes the standard output stream

	if ( err_mode && fflush( stdout ) ) {
		int const err = errno;
		fprintf( stderr, "\n" );
		if ( num_processes > 1 )
			fprintf( stderr, "[P%" PRI_IDX "] ", process_id );
		fprintf( stderr, "Warning: Could not flush the standard output stream (stdout), so previous messages "
			"not printed yet can be delayed or lost.\n" );
		if ( err )
			fprintf( stderr, "%s\n", strerror(err) );
		errno = 0;	// Resets errno.
		error = 1;
	}

	// --------------------------

	if ( all_processes + ( ! process_id ) ) {

		if ( show_id )
			error += ( fprintf( file, "\n[P%" PRI_IDX "] ", process_id ) <= 0 );

		error += ( vfprintf( file, fmt, args ) <= 0 );

		if ( err_mode ) {
			if ( errnum )
				error += ( fprintf( file, ": %s", strerror(errnum) ) <= 0 );
			error += ( fprintf( file, "\n" ) <= 0 );
		}
	}

	if ( error ) {
		int const err = errno; fflush(stdout);
		fprintf( stderr, "\n" );
		if ( num_processes > 1 )
			fprintf( stderr, "[P%" PRI_IDX "] ", process_id );
		fprintf( stderr, "Error in print_string()\n" );
		if ( err )
			fprintf( stderr, "%s\n", strerror(err) );
		return EXIT_FAILURE;
	}

	errno = 0;

	return EXIT_SUCCESS;

} // print_string

////////////////////////////////////////////////

/*
 * Prints the given message composed by the format string "fmt" and the subsequent
 * arguments, if any.
 *
 * If "all_processes" is 'true', the message is printed by all existing processes.
 * In addition, if (num_processes > 1), the process ID is also printed.
 *
 * The string is always printed to the standard output stream ('stdout').
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int print_message( bool all_processes, char const *restrict const fmt, ... )
{

	bool const show_id = all_processes * (num_processes > 1);
	bool const error_mode = false;
	int const error_num = 0;

	va_list args;

	// --------------------------

	va_start( args, fmt );

	int const status = print_string( all_processes, show_id, error_mode, error_num, fmt, args );

	va_end( args );

	return status;

} // print_message

////////////////////////////////////////////////

/*
 * Prints the given message, composed by the format string "fmt" and the subsequent
 * arguments, if any.
 *
 * This method is intended for successive portions of a message that was previously
 * printed, so it will never be prefixed by a newline nor the process ID.
 *
 * If "all_processes" is 'true', the message is printed by all existing processes.
 *
 * The string is always printed to the standard output stream ('stdout').
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int append_printed_message( bool all_processes, char const *restrict const fmt, ... )
{

	bool const show_id = false;
	bool const error_mode = false;
	int const error_num = 0;

	va_list args;

	// --------------------------

	va_start( args, fmt );

	int const status = print_string( all_processes, show_id, error_mode, error_num, fmt, args );

	va_end( args );

	return status;


} // append_printed_message

////////////////////////////////////////////////

/*
 * Prints the given error message, composed by the format string "fmt" and the
 * subsequent arguments, if any.
 *
 * If "all_processes" is 'true', the message is printed by all existing processes.
 * In addition, if (num_processes > 1), the process ID is also printed.
 *
 * The string is always printed to the standard error stream ('stderr'). The
 * standard output stream ('stdout') is previously flushed for all processes.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int print_error( bool all_processes, char const *restrict const fmt, ... )
{

	bool const show_id = all_processes * (num_processes > 1);
	bool const error_mode = true;
	int const error_num = 0;

	va_list args;

	// --------------------------

	va_start( args, fmt );

	int const status = print_string( all_processes, show_id, error_mode, error_num, fmt, args );

	va_end( args );

	return status;

} // print_error

////////////////////////////////////////////////

/*
 * Prints the given error message, composed by the format string "fmt" and the
 * subsequent arguments, if any.
 *
 * This method is intended for successive portions of a message that was previously
 * printed, so it will never be prefixed by a newline nor the process ID.
 *
 * If "all_processes" is 'true', the message is printed by all existing processes.
 *
 * The string is always printed to the standard error stream ('stderr'). The
 * standard output stream ('stdout') is previously flushed for all processes.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int append_printed_error( bool all_processes, char const *restrict const fmt, ... )
{

	bool const show_id = false;
	bool const error_mode = true;
	int const error_num = 0;

	va_list args;

	// --------------------------

	va_start( args, fmt );

	int const status = print_string( all_processes, show_id, error_mode, error_num, fmt, args );

	va_end( args );

	return status;

} // append_printed_error

////////////////////////////////////////////////

/*
 * Prints the given error message, composed by the format string "fmt" and the
 * subsequent arguments, if any.
 *
 * If "all_processes" is 'true', the message is printed by all existing processes.
 * In addition, if (num_processes > 1), the process ID is also printed.
 *
 * The string is always printed to the standard error stream ('stderr'). The
 * standard output stream ('stdout') is previously flushed for all processes.
 *
 * Finally, if errnum is non-zero, this function behaves similar to perror(3).
 * That is, it appends to the message a colon, the string given by strerror(errnum)
 * and a newline character. Otherwise, it just prints a newline.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int print_errnum( bool all_processes, int errnum, char const *restrict const fmt, ... )
{

	bool const show_id = all_processes * (num_processes > 1);
	bool const error_mode = true;

	va_list args;

	// --------------------------

	va_start( args, fmt );

	int const status = print_string( all_processes, show_id, error_mode, errnum, fmt, args );

	va_end( args );

	return status;

} // print_errnum

////////////////////////////////////////////////

/*
 * Prints the given error message, composed by the format string "fmt" and the
 * subsequent arguments, if any.
 *
 * This method is intended for successive portions of a message that was previously
 * printed, so it will never be prefixed by a newline nor the process ID.
 *
 * If "all_processes" is 'true', the message is printed by all existing processes.
 *
 * The string is always printed to the standard error stream ('stderr'). The
 * standard output stream ('stdout') is previously flushed for all processes.
 *
 * Finally, if errnum is non-zero, this function behaves similar to perror(3).
 * That is, it appends to the message a colon, the string given by strerror(errnum)
 * and a newline character. Otherwise, it just prints a newline.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int append_printed_errnum( bool all_processes, int errnum, char const *restrict const fmt, ... )
{

	bool const show_id = false;
	bool const error_mode = true;

	va_list args;

	// --------------------------

	va_start( args, fmt );

	int const status = print_string( all_processes, show_id, error_mode, errnum, fmt, args );

	va_end( args );

	return status;

} // append_printed_errnum

////////////////////////////////////////////////

/*
 * Flushes the buffer associated to the standard output stream (stdout).
 *
 * If "permanently" is 'true', the buffer is also disabled.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int flush_output( bool permanently )
{

	int status = EXIT_SUCCESS;

	if ( fflush( stdout ) ) {
		print_errnum( sys_error_shown_by_all, errno, "Warning: Could not flush the standard output stream, so previous messages not "
				"printed yet can be lost" );
		errno = 0; // Resets errno.
		status = EXIT_FAILURE;
	}

	if ( permanently && setvbuf( stdout, NULL, _IONBF, 0 ) ) {
		print_errnum( sys_error_shown_by_all, errno, "Warning: could not permanently flush the standard output stream, so "
				"not all messages might be shown on program error" );
		status = EXIT_FAILURE;
	}

	return status;

} // flush_output

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Prints all arguments regarding the input matrix (e.g., matrix dimensions and format).
 *
 * This message is printed by process 0 only.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int help_matrix( void )
{

	int const status = append_printed_message( shown_by_all,

		"\n<filename>\n\tInput data matrix (mandatory if 'help' is not requested).\n\n"

		"-B,-b <native>\n\tBinary input file in \"native\" ('-b 1') or \"non-native\" format ('-b 0').\n"
			"\tIn NON-native format, the file is read assuming it was written using double-precision data, and unsigned "
			"integers for matrix\n\tdimensions, regardless how the program was compiled.\n"
			"\tOtherwise, in \"native\" format, the file is read using the data types specified at compilation "
			"(e.g., floats and signed int).\n\tPlease note that \"native\" mode skips most error checking and information messages.\n"
			"\tThe default (if '-b' is not specified) is to read input data from an ASCII-text file.\n\n"

		"-C,-c\tInput text file has numeric column headers (disabled by default, ignored for binary files).\n\n"

		"-R,-r\tInput text file has numeric row labels (disabled by default, ignored for binary files).\n\n"

		"-E,-e <native>\n\tWrites output files as \"native\" ('-e 1') or \"non-native\" binary format ('-e 0').\n"
			"\tIn NON-native format, the file is written using double-precision data, and unsigned integers for matrix dimensions, "
			"regardless\n\thow the program was compiled.\n"
			"\tOtherwise, in \"native\" or raw format, the file is written using the data types specified at compilation "
			"(e.g., floats and signed int).\n\tPlease note that native mode skips error checking, information messages, and "
			"data transformation (e.g., matrix transposing).\n"
			"\tThe default (if '-e' is not specified) is to write output data to an ASCII-text file.\n\n" );

	return status;

} // help_matrix

////////////////////////////////////////////////

/*
 * Prints all arguments regarding the NMF algorithm (e.g., factorization rank, number of iterations, etc).
 *
 * This message is printed by process 0 only.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int help_nmf( void )
{

	int const status = append_printed_message( shown_by_all,

		"\n-K,-k <factorization_rank>\n\tFactorization Rank (default: K=%" PRI_IDX ").\n\n"

		"-I,-i <nIters>\n\tMaximum number of iterations (%" PRI_IDX " by default).\n\n"

		"-J,-j <niter_test_conv>\n\tPerform a convergence test each <niter_test_conv> iterations (default: %" PRI_IDX ").\n"
			"\tIf this value is greater than <nIters> (see '-i' option), no test is performed\n\n"

		"-T,-t <stop_threshold>\n\tStopping threshold (default: %" PRI_IDX ").\n"
			"\tWhen matrix H has not changed on the last <stop_threshold> times that the convergence test\n\thas been performed, "
			"it is considered that the algorithm has converged to a solution and stops it.\n\n",

		DEFAULT_K, DEFAULT_NITERS, DEFAULT_NITER_CONV, DEFAULT_STOP_THRESHOLD );

	return status;

} // help_nmf

////////////////////////////////////////////////

/*
 * Prints all arguments regarding the main program <execname>.
 *
 * This message is printed by process 0 only.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int print_nmf_gpu_help( char const *restrict const execname )
{

	// Checks for NULL parameters
	if ( ! execname ) {
		print_errnum( error_shown_by_all, EFAULT, "print_nmf_gpu_help( execname )" );
		return EXIT_FAILURE;
	}

	int status = EXIT_SUCCESS;

	// ---------------------------

	status = print_message( shown_by_all,
				"\n\t<< bioNMF-GPU: Non-negative Matrix Factorization on GPU >>\n\n"
				"Usage:\n\t%s <filename> [ -b <native> ] [ -cr ] [ -k <factorization_rank> ] [ -i <nIters> ] "
				"[ -j <niter_test_conv> ]\n\t\t[ -t <stop_threshold> ] [ -e <native> ] [ -z <GPU_device> ]\n\t%s -h\n\n"
				"---------------\n\nData matrix options:\n", execname, execname );

	if ( help_matrix() != EXIT_SUCCESS )
		status = EXIT_FAILURE;

	if ( append_printed_message( shown_by_all, "\n---------------\n\nNMF options:\n" ) != EXIT_SUCCESS )
		status = EXIT_FAILURE;

	if ( help_nmf() != EXIT_SUCCESS )
		status = EXIT_FAILURE;

	if ( append_printed_message( shown_by_all, "\n---------------\n\nOther options:\n\n"
				"-Z,-z <GPU_device>\n\tGPU device ID to attach on (default: %i).\n"
				"\tOn multi-GPU version, devices will be selected from this value.\n\n"
				"-h,-H\tPrints this help message.\n\n", DEFAULT_GPU_DEVICE ) != EXIT_SUCCESS )
		status = EXIT_FAILURE;

	return status;

} // print_nmf_gpu_help

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Sets the maximum dimensions and number of items, for the given data
 * alignment and dimension limit.
 *
 * The resulting values are stored in the global variables "matrix_max_pitch",
 * "matrix_max_non_padded_dim" and "matrix_max_num_items". In addition, the first
 * and third variables are rounded down to a multiple of the given data alignment.
 *
 * data_alignment:
 *		If set to '0', uses the default padding, <DEFAULT_MEMORY_ALIGNMENT>.
 *		If set to '1', disables padding.
 *		Otherwise, it must be a positive value expressed in number of
 *		items (not in bytes).
 *
 * max_dimension:
 *		If greater than or equal to the resulting data alignment (i.e.,
 *		either <data_alignment> or <DEFAULT_MEMORY_ALIGNMENT>), uses the
 *		given value as an additional upper limit for matrix dimensions.
 *		That is, the result will be the minimum between <max_dimension>,
 *		and the value calculated from the data alignment and the maximum
 *		number of items.
 *		On "matrix_max_pitch", the result is subsequently rounded down
 *		to a multiple of the data alignment.
 *		It is ignored if set to a non-negative value less than
 *		<data_alignment>.
 *
 * max_nitems:
 *		If set to a positive value, forces the maximum number of items
 *		for a data matrix. Please note that this value HAS PRECEDENCE
 *		over the resulting maximum dimensions. However, it is IGNORED
 *		if set to a value less than the resulting data alignment.
 *		Finally, it must be expressed in number of items, not in bytes.
 *
 * WARNING:
 *	This function must be called *BEFORE* loading any input matrix. Otherwise,
 *	no padding will be set.
 *
 * Returns EXIT_SUCCESS, or EXIT_FAILURE on negative input values.
 */
int set_matrix_limits( index_t data_alignment, index_t max_dimension, size_t max_nitems )
{

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "set_matrix_limits( data_alignment=%" PRI_IDX ", max_dimension=%" PRI_IDX
				", max_nitems=%zu).\n", data_alignment, max_dimension, max_nitems );
	#endif

	if ( (data_alignment < 0) + (max_dimension < 0) ) {
		int const errnum = EINVAL;
		if (data_alignment < 0)
			print_errnum( error_shown_by_all, errnum, "set_matrix_limits(data_alignment=%" PRI_IDX ")", data_alignment );
		if (max_dimension < 0)
			print_errnum( error_shown_by_all, errnum, "set_matrix_limits(max_dimension=%" PRI_IDX ")", max_dimension);
		return EXIT_FAILURE;
	}

	// ------------------

	index_t const l_memory_alignment = ( data_alignment ? data_alignment : DEFAULT_MEMORY_ALIGNMENT );

	index_t const l_max_dim = ( (max_dimension >= l_memory_alignment) ? max_dimension : IDX_MAX );

	// ------------------

	// Maximum number of items.
	size_t const default_max_nitems = SIZE_MAX / ( 2 * sizeof(real) );
	size_t l_max_num_items = ( (max_nitems >= (size_t)l_memory_alignment) ? (MIN(max_nitems,default_max_nitems)) : default_max_nitems );
	l_max_num_items -= ( l_max_num_items % l_memory_alignment );	// Previous multiple of <memory_alignment>.

	// Maximum padded dimension (typically, number of columns).
	index_t l_max_pitch = MIN( (l_max_num_items / 2), (size_t) l_max_dim );
	l_max_pitch -= (l_max_pitch % l_memory_alignment);		// Previous multiple of <memory_alignment>.

	// The other dimension (typically, number of rows).
	index_t const l_max_alignment = MAX( l_memory_alignment, 2 );	// Padding, or 2 columns
	index_t const l_max_non_padded_dim = MIN( (l_max_num_items/l_max_alignment), (size_t) l_max_dim );

	// ------------------

	// Sets output values.
	memory_alignment = l_memory_alignment;
	matrix_max_num_items = l_max_num_items;
	matrix_max_pitch = l_max_pitch;
	matrix_max_non_padded_dim = l_max_non_padded_dim;

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "set_matrix_limits( data_alignment=%" PRI_IDX ", max_dimension=%" PRI_IDX "):\n"
				"\tResulting values: matrix_max_num_items=%zu, matrix_max_pitch=%" PRI_IDX ", matrix_max_non_padded_dim=%"
				PRI_IDX ".\n\n", data_alignment, max_dimension, matrix_max_num_items, matrix_max_pitch,
				matrix_max_non_padded_dim );
	#endif

	return EXIT_SUCCESS;

} // set_matrix_limits

////////////////////////////////////////////////

/*
 * Sets the maximum dimensions and number of items, for the DEFAULT
 * data alignment (<DEFAULT_MEMORY_ALIGNMENT>).
 *
 * The resulting values are stored in the global variables "matrix_max_pitch",
 * "matrix_max_non_padded_dim" and "matrix_max_num_items". In addition, the first
 * and third variables are rounded down to a multiple of the given data alignment.
 *
 * Returns EXIT_SUCCESS.
 */
int set_default_matrix_limits( void )
{

	set_matrix_limits( 0, 0, 0 );

	return EXIT_SUCCESS;

} // set_default_matrix_limits

////////////////////////////////////////////////

/*
 * Computes the padded dimension of "dim".
 *
 * Returns <pitch>, such that:
 *	dim <= pitch
 *	pitch is a multiple of <memory_alignment>
 *
 * WARNING:
 *	Global variable "memory_alignment" must have been properly initialized.
 */
index_t get_padding( index_t dim )
{

	index_t padded_dim = dim;

	// If "dim" is NOT a multiple of <memory_alignment>, computes the next multiple.
	index_t const dim_mod_ma = ( dim % memory_alignment );
	if ( dim_mod_ma )
		padded_dim += (memory_alignment - dim_mod_ma);

	return padded_dim;

} // get_padding

////////////////////////////////////////////////

/*
 * Checks all arguments.
 *
 * Sets 'help' to 'true' if help message was requested ('-h' or '-H' options).
 *
 * Error messages will be shown by process 0 only.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int check_arguments( int argc, char const *restrict *restrict argv, bool *restrict help, struct input_arguments *restrict arguments )
{

	// Limits on matrix dimensions
	index_t const max_pitch = matrix_max_pitch;

	// Checks for invalid parameters
	if ( ! ( (argc > 0) * (uintptr_t) argv * (uintptr_t) help * (uintptr_t) arguments ) ) {
		int const errnum = EFAULT;
		if ( argc <= 0 ) print_errnum( error_shown_by_all, errnum, "check_arguments( argc=%i )", argc );
		if ( ! argv )	print_errnum( error_shown_by_all, errnum, "check_arguments( argv )" );
		if ( ! help )	print_errnum( error_shown_by_all, errnum, "check_arguments( help )" );
		if ( ! arguments ) print_errnum( error_shown_by_all, errnum, "check_arguments( arguments )" );
		return EXIT_FAILURE;
	}

	// ---------------------------

	// Default values
	char const *l_filename = NULL;
	bool l_numeric_hdrs = false;	// Has numeric columns headers.
	bool l_numeric_lbls = false;	// Has numeric row labels.

	index_t l_is_bin = 0;		// Input file is binary.
					// == 0 for ASCII-text format.
					// == 1 for non-native binary format (i.e., double-precision data, and "unsigned int" for dimensions).
					// > 1 for native binary format (i.e., the compiled types for matrix data and dimensions).

	index_t l_save_bin = 0;		// Saves output matrices to binary files.
					// == 0 for ASCII-text format.
					// == 1 for non-native format (i.e., double-precision data, and "unsigned int" for dimensions).
					// > 1 for native format (i.e., the compiled types for matrix data and dimensions).

	index_t l_k = DEFAULT_K;				// Factorization rank.
	index_t l_kp = get_padding( DEFAULT_K );		// (Initial) padded factorization rank.
	index_t l_nIters = DEFAULT_NITERS;			// Maximum number of iterations per run.
	index_t l_niter_test_conv = DEFAULT_NITER_CONV;		// Number of iterations before testing convergence.
	index_t l_stop_threshold = DEFAULT_STOP_THRESHOLD;	// Stopping criterion.
	index_t l_gpu_device = DEFAULT_GPU_DEVICE;		// Device ID (NMF_[m]GPU only).

	index_t l_idx_other_args = 0;	// Index in argv[] with additional executable-specific arguments.

	// ---------------------------

	// Resets getopt(3) variables.
	optind = 0;
	opterr = 0;	// Disables error messages.

	int opt = 0;	// Selected option

	/* Reads option arguments:
	 *
	 *	-b native
	 *	-c
	 *	-e native
	 *	-h
	 *	-i nIters
	 *	-j niter_test_conv
	 *	-k kStart
	 *	-r
	 *	-t stop_threshold
	 *	-y locale (TODO)
	 *	-z gpu_device
	 */

	// NOTE: First colon (':') indicates to return ':' instead of '?' in case of a missing option argument.
	while ( (opt = getopt( argc, (char *const *) argv, ":B:b:CcE:e:HhI:i:J:j:K:k:RrT:t:Z:z:" ) ) != -1 ) {

		switch( opt ) {

			// Input file is binary.
			case 'B':
			case 'b': {
				errno = 0;
				char *endptr = NULL;
				intmax_t const val = strtoimax( optarg, &endptr, 10 );
				if ( *endptr + errno + (val < INTMAX_C(0)) ) {
					print_errnum( error_shown_by_all, errno, "Error: Invalid binary mode for input file: '%s'", optarg );
					append_printed_error( error_shown_by_all, "It must be a non-negative integer value.\n");
					return EXIT_FAILURE;
				}
				l_is_bin = (index_t) ( val ? 2 : 1 );
			} break;


			// Input file has numeric column headers
			case 'C':
			case 'c': {
				l_numeric_hdrs = true;
			} break;


			// Saves as a binary file.
			case 'E':
			case 'e': {
				errno = 0;
				char *endptr = NULL;
				intmax_t const val = strtoimax( optarg, &endptr, 10 );
				if ( *endptr + errno + (val < INTMAX_C(0)) ) {
					print_errnum( error_shown_by_all, errno, "Error: Invalid binary mode for output file(s): '%s'", optarg );
					append_printed_error( error_shown_by_all, "It must be a non-negative integer value.\n" );
					return EXIT_FAILURE;
				}
				l_save_bin = (index_t) ( val ? 2 : 1 );
			} break;


			// Prints a help message.
			case 'H':
			case 'h': {
				*help = true;		// Help is printed on return of this function.
				return EXIT_SUCCESS;
			} // break;	// Unreachable statement


			// nIters
			case 'I':
			case 'i': {
				errno = 0;
				char *endptr = NULL;
				intmax_t const val = strtoimax( optarg, &endptr, 10 );
				if ( *endptr + errno + (val <= INTMAX_C(0)) + (val > (intmax_t) IDX_MAX) ) {
					print_errnum( error_shown_by_all, errno, "Error: Invalid number of iterations: '%s'", optarg);
					append_printed_error( error_shown_by_all, "It must be a positive integer value less than or equal to %"
								PRI_IDX ".\n", IDX_MAX );
					return EXIT_FAILURE;
				}
				l_nIters = (index_t) val;
			} break; // nIters


			// niter_test_conv
			case 'J':
			case 'j': {
				errno = 0;
				char *endptr = NULL;
				intmax_t const val = strtoimax( optarg, &endptr, 10 );
				if ( *endptr + errno + (val <= INTMAX_C(0)) + (val > (intmax_t) IDX_MAX) ) {
					print_errnum( error_shown_by_all, errno, "Error: Invalid number of iterations for convergence test: '%s'",
							optarg );
					append_printed_error( error_shown_by_all, "It must be a positive integer value less than or equal to %"
								PRI_IDX ".\n", IDX_MAX );
					return EXIT_FAILURE;
				}
				l_niter_test_conv = (index_t) val;
			} break; // niter_test_conv


			// Factorization rank
			case 'K':
			case 'k': {
				errno = 0;
				char *endptr = NULL;
				intmax_t const val = strtoimax( optarg, &endptr, 10 );
				if ( *endptr + errno + (val < INTMAX_C(2)) + (val > (intmax_t) max_pitch) ) {
					print_errnum( error_shown_by_all, errno, "Error: invalid factorization rank: '%s'", optarg );
					append_printed_error( error_shown_by_all, "It must be an integer value in the range [2 .. %"
								PRI_IDX "].\n", max_pitch );
					return EXIT_FAILURE;
				}
				l_k = (index_t) val;
				l_kp = get_padding( val );
			} break; // k


			// Input file has numeric row labels
			case 'R':
			case 'r': {
				l_numeric_lbls = true;
			} break;


			// stop_threshold
			case 'T':
			case 't': {
				errno = 0;
				char *endptr = NULL;
				intmax_t const val = strtoimax( optarg, &endptr, 10 );
				if ( *endptr + errno + (val <= INTMAX_C(0)) + (val > (intmax_t) IDX_MAX) ) {
					print_errnum( error_shown_by_all, errno, "Error: invalid stopping threshold '%s'", optarg );
					append_printed_error( error_shown_by_all, "It must be a positive integer value less than or equal to %"
								PRI_IDX ".\n", IDX_MAX );
					return EXIT_FAILURE;
				}
				l_stop_threshold = (index_t) val;
			} break; // stop_threshold


			// Device ID
			case 'Z':
			case 'z': {
				errno = 0;
				char *endptr = NULL;
				intmax_t const val = strtoimax( optarg, &endptr, 10 );
				if ( *endptr + errno + (val < INTMAX_C(0)) + (val > (intmax_t) IDX_MAX) ) {
					print_errnum( error_shown_by_all, errno, "Error: invalid basis device ID number '%s'", optarg );
					append_printed_error( error_shown_by_all, "It must be a non-negative integer value less than or equal to %"
							PRI_IDX ".\n", IDX_MAX );
					return EXIT_FAILURE;
				}
				l_gpu_device = (index_t) val;
			}
			break;


			// Missing argument
			case ':': {
				print_error( error_shown_by_all, "Error: option -%c requires an argument.\nSee help (option '-h').\n",
						optopt );
				return EXIT_FAILURE;
			} // break;	// Unreachable statement


			// Invalid option. It is just ignored.
			case '?':
			default :
			break;

		} // switch( opt )

	} // while there are options to read.

	// -----------------------------

	// Checks non-option argument(s)

	// Filename
	if ( optind >= argc ) {
		print_error( error_shown_by_all, "Error: No filename. Not enough arguments.\nSee help (option '-h').\n" );
		return EXIT_FAILURE;
	}
	l_filename = argv[optind];
	optind++;

	// -----------------------------

	// Additional executable-specific arguments.
	l_idx_other_args = optind;

	// -----------------------------

	// Resets getopt(3) variables.
	optarg = NULL;
	optind = opterr = optopt = 0;

	// --------------------

	// Sets output values.

	struct input_arguments l_arguments;

	l_arguments.filename = l_filename;
	l_arguments.numeric_hdrs = l_numeric_hdrs;
	l_arguments.numeric_lbls = l_numeric_lbls;

	l_arguments.is_bin = l_is_bin;
	l_arguments.save_bin = l_save_bin;

	l_arguments.k = l_k;
	l_arguments.kp = l_kp;
	l_arguments.nIters = l_nIters;
	l_arguments.niter_test_conv = l_niter_test_conv;
	l_arguments.stop_threshold = l_stop_threshold;

	l_arguments.gpu_device = l_gpu_device;

	l_arguments.idx_other_args = l_idx_other_args;

	*arguments = l_arguments;

	return EXIT_SUCCESS;

} // check_arguments

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/* Helper function for next_power_2() and prev_power_2().
 *
 * Computes the next one-less-than-a-power-of-two value.
 *
 * Returns:
 *	The lowest y >= x, such that <y+1> is a power of 2.
 *	0, if x == 0
 */
static size_t next_prev_power_2( size_t x )
{

	for ( size_t i = 0, b = 1 ; i <= sizeof(size_t) ; i++, b <<= 1 )
		x |= ( x >> b );

	return x;

} // next_prev_power_2

// ---------------------------------------------

/*
 * Computes the lowest power-of-two >= x.
 *
 * WARNING:
 *	x <= floor(SIZE_MAX / 2)
 *
 * Returns:
 *	<x>, if it is already a power of two, or x == 0
 *	The next power of two, if "x" is not a power of two.
 */
size_t next_power_2( size_t x )
{

	if ( x & (x-1) ) {	// If it is not already a power of two.

		x = next_prev_power_2( x );	// Next one-less-than-a-power-of-two value.
		x++;
	}

	return x;

} // next_power_2

////////////////////////////////////////////////

/*
 * Computes the highest power-of-two <= x.
 *
 * Returns:
 *	<x>, if it is already a power of two, or x == 0
 *	The previous power of two, if "x" is not a power of two.
 */
size_t prev_power_2( size_t x )
{

	if ( x & (x-1) ) {	// If it is not already a power of two.

		x = next_prev_power_2( x );	// Next one-less-than-a-power-of-two value.
		x = ((x >> 1) + 1);		// or, x -= (x >> 1);
	}

	return x;

} // prev_power_2

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Gets the difference between classification and last_classification vectors
 */
size_t get_difference( index_t const *restrict classification, index_t const *restrict last_classification, index_t m )
{
	size_t diff = 0;

	for ( index_t i = 0 ; i < (m-1); i++ ) {
		for ( index_t j = (i+1) ; j < m ; j++ ) {
			bool conn = ( classification[j] == classification[i] );
			bool conn_last = ( last_classification[j] == last_classification[i] );
			diff += ( conn != conn_last );
		}
	}

	return diff;

} // get_difference

////////////////////////////////////////////////

/*
 * Retrieves a "random" value that can be used as seed.
 *
 * If NMFGPU_FIXED_INIT is non-zero, returns <FIXED_SEED>.
 */
index_t get_seed( void )
{

	index_t seed = FIXED_SEED;

	#if ! NMFGPU_FIXED_INIT

		// Reads the seed from /dev/urandom

		FILE *restrict file = fopen("/dev/urandom", "r");

		if ( ( ! file ) || ( ! fread( &seed, sizeof(index_t), 1, file ) ) ) {

			/* If for whatever reason (e.g., non-Linux system, file
			 * not found, etc.) it failed to read the file, sets
			 * the seed from the clock.
			 */

			if ( file )
				fclose(file);

			errno = 0;

			struct timeval tv;
			gettimeofday(&tv, NULL);
			time_t usec = tv.tv_usec;
			seed = (index_t) usec;

		}

	#endif /* NMFGPU_FIXED_INIT */

	// ----------------------------

	#if NMFGPU_VERBOSE_2
		print_message( verb_shown_by_all, "\nReturned seed: %" PRI_IDX "\n", seed );
	#endif

	return seed;

} // get_seed

////////////////////////////////////////////////
////////////////////////////////////////////////
