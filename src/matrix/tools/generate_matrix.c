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
 * generate_matrix.c
 *	Program to generate a matrix with random values and save it to a file.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Additional information:
 *		NMFGPU_VERBOSE: Shows some messages concerning the progress of the program, as well as some configuration parameters.
 *		NMFGPU_VERBOSE_2: Shows the parameters in some routine calls.
 *
 *	CPU timing:
 *		NMFGPU_PROFILING_GLOBAL: Computes total elapsed time.
 *
 *	Debug / Testing:
 *		NMFGPU_FIXED_INIT: Uses "random" values generated from a fixed seed (defined in common.h).
 *		NMFGPU_DEBUG: Shows the result of each matrix operation and data transfer.
 *
 **********************************************************
 **********************************************************
 **********************************************************
 *
 * NOTE: In order to improve performance:
 *
 *	+ The number of columns is rounded up to a multiple of <memory_alignment>.
 *	  The padded dimension is referred as "pitch".
 *
 *	  This leads to the following limits:
 *		- Maximum number of columns (padded or not): matrix_max_pitch.
 *		- Maximum number of rows: matrix_max_non_padded_dim.
 *		- Maximum number of items: matrix_max_num_items.
 *
 *	  All four GLOBAL variables must be initialized with the
 *	  set_matrix_limits() function.
 *
 **********************************************************
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
 * WARNING:
 *	+ This code requires support for ISO-C99 standard. It can be enabled with 'gcc -std=c99'.
 *
 **********************************************************/

// Required by <stdint.h>
#ifndef __STDC_CONSTANT_MACROS
	#define __STDC_CONSTANT_MACROS (1)
#endif

#include "matrix/matrix_io.h"
#include "matrix/matrix_io_routines.h"
#include "common.h"
#include "real_type.h"
#include "index_type.h"

#if NMFGPU_PROFILING_GLOBAL
	#include <sys/time.h>
#endif
#include <inttypes.h>	/* strtoimax, INTMAX_C, uintptr_t, [u]intmax_t */
#include <math.h>	/* isless, isgreater, isfinite */
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>


////////////////////////////////////////////////
////////////////////////////////////////////////

/* Constants */

/* Default values for some parameters. */

#ifndef DEFAULT_MAXRAND
	#define DEFAULT_MAXRAND ( REAL_C( 1.0 ) )
#endif

#ifndef DEFAULT_MINRAND
	#define DEFAULT_MINRAND ( R_MIN )
#endif

// ---------------------------------------------
// ---------------------------------------------

/* Global variables. */

// Information and/or error messages shown by all processes.
#if NMFGPU_DEBUG
	static bool const dbg_shown_by_all = false;	// Information or error messages on debug.
#endif
static bool const shown_by_all = false;			// Information messages in non-debug mode.
static bool const sys_error_shown_by_all = false;	// System error messages.
static bool const error_shown_by_all = false;		// Error messages on invalid arguments or I/O data.

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Prints all arguments.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int print_generate_matrix_help( char const *restrict const execname )
{

	// Checks for NULL parameters
	if ( ! execname ) {
		print_errnum( error_shown_by_all, EFAULT, "print_generate_matrix_help( execname )" );
		return EXIT_FAILURE;
	}

	int status = EXIT_SUCCESS;

	// ---------------------------

	status = print_message( shown_by_all, "Tool to generate (in CPU) a matrix with random values\n\n"
				"Usage:\n\t%s <filename> <rows> <columns> [ -e <native_format> ] [ <max_value> ]\n"
				"\t%s -h\n\n", execname, execname );

	if ( help_matrix() != EXIT_SUCCESS )
		status = EXIT_FAILURE;

	if ( append_printed_message( shown_by_all, "\nNote: Some of the previous options are read for compatibility reasons, "
				"but they are ignored by the program.\n\n"
				"<rows> <columns>\n\tOutput matrix dimensions (both mandatory if 'help' is not requested).\n"
				"\tNote that <rows> x <columns> must be less than, or equal to, %" PRI_IDX ".\n\n"
				"<max_value>\n\tRange for random values: [0..<max_value>]. "
				"Valid values are in range (%g .. %g). The default is %g\n\n"
				"-h,-H\tPrints this help message.\n\n", IDX_MAX, DEFAULT_MINRAND, R_MAX, DEFAULT_MAXRAND ) != EXIT_SUCCESS )
		status = EXIT_FAILURE;

	return status;

} // print_generate_matrix_help

////////////////////////////////////////////////

/*
 * Checks additional arguments
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int check_additional_arguments( int argc, char const *restrict const *restrict argv, index_t idx_other_args,
					index_t *restrict nrows, index_t *restrict ncols, real *restrict max_value )
{

	// Limits on matrix dimensions
	size_t const max_num_items = matrix_max_num_items;
	index_t const max_pitch = matrix_max_pitch;
	index_t const max_non_padded_dim = matrix_max_non_padded_dim;


	// Checks for NULL parameters
	if ( ! ( ( argc > 0 ) * (uintptr_t) argv * (uintptr_t) nrows * (uintptr_t) ncols * (uintptr_t) max_value ) ) {
		int const errnum = EFAULT;
		if ( argc <= 1 ) print_errnum( error_shown_by_all, errnum, "ncheck_additional_arguments( argc=%i )", argc );
		if ( ! argv )	print_errnum( error_shown_by_all, errnum, "check_additional_arguments( argv )" );
		if ( ! nrows )	print_errnum( error_shown_by_all, errnum, "check_additional_arguments( nrows )" );
		if ( ! ncols )	print_errnum( error_shown_by_all, errnum, "check_additional_arguments( ncols )" );
		if ( ! max_value ) print_errnum( error_shown_by_all, errnum, "check_additional_arguments( max_value )" );
		return EXIT_FAILURE;
	}

	// ---------------------------

	// Default values
	index_t l_nrows = 0, l_ncols = 0;	// Matrix dimensions
	real l_max_value = DEFAULT_MAXRAND;	// Maximum value

	// ---------------------------

	// Fails if there are not more arguments
	if ( idx_other_args >= (index_t) (argc-1) ) {
		print_error( error_shown_by_all, "Error: No matrix dimensions. Not enough arguments.\nSee help (option '-h').\n" );
		return EXIT_FAILURE;
	}

	// ---------------------------

	// Number of rows
	{
		errno = 0;
		char *endptr = NULL;
		intmax_t const val = strtoimax( argv[idx_other_args], &endptr, 10 );
		if ( (*endptr != '\0') + errno + (val <= INTMAX_C(0)) + (val > (intmax_t) max_non_padded_dim) ) {
			print_errnum( error_shown_by_all, errno, "Error. Invalid number of rows: '%s'", argv[idx_other_args]);
			append_printed_error( error_shown_by_all, "It must be a positive integer less than or equal to %" PRI_IDX ".\n",
						max_non_padded_dim );
			return EXIT_FAILURE;
		}
		l_nrows = (index_t) val;
	}
	idx_other_args++;

	// Number of columns
	{
		errno = 0;
		char *endptr = NULL;
		intmax_t const val = strtoimax( argv[idx_other_args], &endptr, 10 );
		if ( (*endptr != '\0') + errno + (val <= INTMAX_C(0)) + (val > (intmax_t) max_pitch) ) {
			print_errnum( error_shown_by_all, errno, "Error. Invalid number of columns: '%s'", argv[idx_other_args]);
			append_printed_error( error_shown_by_all, "It must be a positive integer less than or equal to %" PRI_IDX ".\n",
						max_pitch );
			return EXIT_FAILURE;
		}
		l_ncols = (index_t) val;
	}
	idx_other_args++;


	// Checks number of items.
	{
		uintmax_t const nitems = (uintmax_t) l_nrows * (uintmax_t) l_ncols;
		if ( nitems > (uintmax_t) max_num_items ) {
			print_error( error_shown_by_all, "Error: output matrix will be too large. The number of items must be "
					"less than, or equal to, %" PRI_IDX ".\n", max_num_items );
			return EXIT_FAILURE;
		}
	}

	// ---------------------------

	// Maximum random value (optional)
	if ( idx_other_args < (index_t) argc ) {
		errno = 0;
		char *endptr = NULL;
		l_max_value = STRTOREAL( argv[idx_other_args], &endptr );
		if ( (*endptr != '\0') + errno + (! isfinite(l_max_value)) ) {
			print_errnum( error_shown_by_all, errno, "Error: invalid maximum value: '%s'", argv[idx_other_args] );
			return EXIT_FAILURE;
		}
		if ( islessequal( l_max_value, DEFAULT_MINRAND ) + isgreaterequal( l_max_value, R_MAX ) ) {
			print_errnum( error_shown_by_all, errno, "Error: invalid maximum value: %g", l_max_value );
			append_printed_error( error_shown_by_all, "Valid values are in range (%g .. %g).\n", DEFAULT_MINRAND, R_MAX );
			return EXIT_FAILURE;
		}
		// idx_other_args++; // Not necessary
	}

	// ---------------------------

	*nrows = l_nrows;
	*ncols = l_ncols;
	*max_value = l_max_value;

	return EXIT_SUCCESS;

} // check_additional_arguments

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

	// Default limits for matrix dimensions.
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

	// Checks all arguments
	if ( check_arguments( argc, argv, &help, &arguments ) != EXIT_SUCCESS )
		return EXIT_FAILURE;

	// If help was requested, just prints a help message and returns.
	if ( help )
		return print_generate_matrix_help( argv[0] );

	char const *restrict const filename = arguments.filename;	// Output filename
	index_t const save_bin = arguments.save_bin;			// Output file is binary (native or non-native format).
	index_t const idx_other_args = arguments.idx_other_args;	// Index in argv[] with additional arguments.

	// ----------------------------------------

	// Additional arguments.

	index_t nrows = 0, ncols = 0;
	real max_value = DEFAULT_MAXRAND;

	if ( check_additional_arguments( argc, argv, idx_other_args, &nrows, &ncols, &max_value ) != EXIT_SUCCESS )
		return EXIT_FAILURE;

	index_t const pitch = get_padding( ncols );

	// ----------------------------------------

	// Generates the output matrix
	size_t const nitems = (size_t) nrows * (size_t) ncols;
	size_t const nitems_padding = (size_t) nrows * (size_t) pitch;

	print_message( shown_by_all, "Generating a %" PRI_IDX "-by-%" PRI_IDX " data matrix (%zu"
			" items), with random values in range [0 .. %g]...\n", nrows, ncols, nitems, max_value );

	// Warns if it is a single-row/column matrix.
	if ( (nrows == 1) + (ncols == 1) )
		append_printed_error( shown_by_all, "\nNote, however, that (row or column) vectors are not a valid input for the NMF algorithm.\n" );

	// ----------------------------------------

	#if NMFGPU_PROFILING_GLOBAL
		// Elapsed time
		gettimeofday( &t_tv, NULL );
	#endif

	// ----------------------------------------

	real *restrict const matrix = (real *restrict) malloc( nitems_padding * sizeof(real) );
	if ( ! matrix ) {
		print_errnum( sys_error_shown_by_all, errno, "malloc(matrix, items=%zu, %zu with padding)", nitems, nitems_padding );
		return EXIT_FAILURE;
	}


	// Seed for the random generator.
	index_t seed = get_seed();

	srandom( (unsigned int) seed );	// Initializes the random generator.

	real const denominator = (real) RAND_MAX;

	real *pmatrix = matrix;
	for ( index_t i=0 ; i < nrows ; i++, pmatrix += pitch ) {
		real *pmatrix_r = pmatrix;
		for ( index_t j=0 ; j < ncols ; j++, pmatrix_r++ ) {
			real const numerator = (real) random();
			real const val0_1    = numerator / denominator;	// Value in range [0..1]
			real const val	     = val0_1 * max_value;	// Value in range [0..<max_value>]
			*pmatrix_r = val;
		}
	}

	// ----------------------------------------

	// Saves the output matrix.

	bool const transpose = false;
	bool const verbose = true;
	struct matrix_tags_t const mt = new_empty_matrix_tags();

	status = matrix_save( filename, save_bin, matrix, nrows, ncols, pitch, transpose, &mt, verbose );

	free( matrix );

	if ( status != EXIT_SUCCESS )
		return EXIT_FAILURE;

	// ----------------------------------------

	#if NMFGPU_PROFILING_GLOBAL
		// Elapsed time
		{
			//struct timespec ftv, etv;

			//if ( clock_gettime( CLOCK_REALTIME, &ftv) ) {
				//print_errnum( error_shown_by_all, errno, "clock_gettime(ftv)" );
				//return EXIT_FAILURE;
			//}

			//my_timersub( stv, ftv, etv );
			//float const total_time = etv.tv_sec + ( etv.tv_nsec * 1e-09f );

			struct timeval t_ftv, t_etv;
			gettimeofday( &t_ftv, NULL );
			timersub( &t_ftv, &t_tv, &t_etv );	// etv = ftv - tv
			float const total_time = t_etv.tv_sec + ( t_etv.tv_usec * 1e-06f );

			print_message( shown_by_all, "Done in %g seconds.\n", total_time );
		}
	#endif

	// ----------------------------------------

	return EXIT_SUCCESS;

} // main

////////////////////////////////////////////////
////////////////////////////////////////////////
