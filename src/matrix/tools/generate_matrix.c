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
 *	Timing:
 *		NMFGPU_PROFILING_GLOBAL: Compute total elapsed time.
 *
 *	Debug / Testing:
 *		NMFGPU_FIXED_INIT: Uses "random" values generated from a fixed seed (defined in common.h).
 *		NMFGPU_DEBUG: Shows the result of each matrix operation and data transfer.
 *
 * WARNING:
 *	- Requires support for ISO-C99 standard. It can be enabled with 'gcc -std=c99'.
 *
 **********************************************************/

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <inttypes.h>	/* strtoumax */
#include <math.h>	/* isless, isgreater, isfinite */
#if NMFGPU_PROFILING_GLOBAL
	#include <sys/time.h>
#endif

#include "index_type.h"
#include "real_type.h"
#include "matrix/matrix_io_routines.h"
#include "matrix/matrix_io.h"
#include "common.h"

// ---------------------------------------------
// ---------------------------------------------

/* Constants */

/* Default values for some parameters. */

#ifndef DEFAULT_MAXRAND
	#define DEFAULT_MAXRAND ( REAL_C( 1.0 ) )
#endif


//////////////////////////////////////////////////
//////////////////////////////////////////////////

/*
 * Prints all arguments to the specified file.
 */
static void print_generate_matrix_help( char const *restrict execname, FILE *restrict file )
{

	// Checks for NULL parameters
	if ( ! ( (size_t) execname * (size_t) file ) ) {
		fflush( stdout );
		errno = EFAULT;
		if ( ! execname ) perror("\nprint_generate_matrix_help( execname )");
		if ( ! file )	  perror("\nprint_generate_matrix_help( file )");
		return;
	}

	// ---------------------------

	fprintf( file,	"\nTool to generate (in CPU) a matrix with random values\n\n"
			"Usage:\n\t%s <filename> <rows> <columns> [ -e <native_format> ] [ <max_value> ]\n"
			"\t%s -h\n\n", execname, execname );

	help_matrix( file );

	fprintf( file,	"Note: Some of the previous options are read for compatibility reasons, but they are ignored by the program.\n\n" );

	fprintf( file,	"<rows> <columns>\n\tOutput matrix dimensions (both mandatory if 'help' is not requested).\n"
			"\tNote that <rows> x <columns> must be less than, or equal to, %" PRI_IDX ".\n\n", IDX_MAX);

	fprintf( file,	"<max_value>\n\tRange for random values: [0..<max_value>]. "
			"Valid values are between %g and %g. The default is %g\n\n", R_MIN, R_MAX, DEFAULT_MAXRAND);

	fprintf( file, "-h,-H\tPrints this help message.\n\n" );

} // print_generate_matrix_help

//////////////////////////////////////////////////

/*
 * Checks additional arguments
 * If verbose_error is 'true', shows error messages.
 *
 * Sets "help" to 'true' if not enough arguments.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int check_additional_arguments( int argc, char const *restrict const *restrict argv, bool verbose_error, index_t idx_other_args,
					index_t *restrict nrows, index_t *restrict ncols, real *restrict max_value, bool *restrict help )
{
	// Checks for NULL parameters
	if ( ! ( ( argc > 0 ) * (size_t) argv * (size_t) nrows * (size_t) ncols * (size_t) max_value ) ) {
		fflush( stdout );
		errno = EFAULT;
		if ( ! argv )	perror("\ncheck_additional_arguments( argv )");
		if ( ! nrows )	perror("\ncheck_additional_arguments( nrows )");
		if ( ! ncols )	perror("\ncheck_additional_arguments( ncols )");
		if ( ! max_value ) perror("\ncheck_additional_arguments( max_value )");
		return EXIT_FAILURE;
	}

	// ---------------------------

	// Default values
	index_t l_nrows = 0, l_ncols = 0;	// Matrix dimensions
	real l_max_value = DEFAULT_MAXRAND;	// Maximum value
	*help = false;

	// ---------------------------

	// Fails if no more arguments
	if ( idx_other_args >= (argc-1) ) {
		fflush(stdout);
		if ( verbose_error )
			fprintf( stderr, "\nError: No matrix dimensions. Not enough arguments.\n" );
		*help = true;		// Help is printed on return of this function.
		return EXIT_FAILURE;
	}

	// ---------------------------

	// Number of rows
	{
		errno = 0;
		char *endptr = NULL;
		uintmax_t val = strtoumax( argv[idx_other_args], &endptr, 10 );
		if ( (*endptr != '\0') + errno + (! val) + (val > IDX_MAX) ) {
			fflush(stdout);
			fprintf( stderr, "\nError. Invalid number of rows: '%s'.\n", argv[idx_other_args] );
			if ( val > IDX_MAX )
				fprintf( stderr, "It must be less than or equal to %" PRI_IDX ".\n", IDX_MAX );
			return EXIT_FAILURE;
		}
		l_nrows = (index_t) val;
	}
	idx_other_args++;

	// Number of columns
	{
		errno = 0;
		char *endptr = NULL;
		uintmax_t val = strtoumax( argv[idx_other_args], &endptr, 10 );
		if ( (*endptr != '\0') + errno + (! val) + (val > IDX_MAX) ) {
			fflush(stdout);
			fprintf( stderr, "\nError. Invalid number of columns: '%s'.\n", argv[idx_other_args] );
			if ( val > IDX_MAX )
				fprintf( stderr, "It must be less than or equal to %" PRI_IDX ".\n", IDX_MAX );
			return EXIT_FAILURE;
		}
		l_ncols = (index_t) val;
	}
	idx_other_args++;


	// Checks number of items.
	{
		uintmax_t const nitems = ((uintmax_t) l_nrows * (uintmax_t) l_ncols);
		if ( nitems > (uintmax_t) IDX_MAX ) {
			fflush(stdout);
			fprintf( stderr, "\nError: output matrix too large. The number of items must be less than, or equal to, %"
					PRI_IDX ".\n", IDX_MAX );
			return EXIT_FAILURE;
		}
	}

	// ---------------------------

	// Maximum random value (optional)
	if ( idx_other_args < argc ) {
		errno = 0;
		char *endptr = NULL;
		l_max_value = STRTOREAL( argv[idx_other_args], &endptr );
		if ( (errno + (*endptr != '\0')) || (! isfinite(l_max_value)) ||
			isless( l_max_value, R_MIN ) || isgreater( l_max_value, R_MAX ) ) {
			fflush(stdout);
			fprintf(stderr,"\nError: invalid maximum value: '%s'\nValid values are between %g and %g.\n",
				argv[idx_other_args], R_MIN, R_MAX );
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

//////////////////////////////////////////////////
//////////////////////////////////////////////////

int main( int argc, char const *restrict *restrict argv )
{

	#if NMFGPU_PROFILING_GLOBAL
		// Elapsed time
		struct timeval t_tv;
	#endif

	// ----------------------------------------

	/* Reads all parameters and performs error-checking. */

	bool help = false;			// Help message requested

	struct input_arguments arguments;	// Input arguments

	// Checks all arguments (shows error messages).
	if ( check_arguments( argc, argv, true, &help, &arguments ) == EXIT_FAILURE ) {
		if ( help ) {
			fprintf(stderr, "\n==========\n");
			print_generate_matrix_help( argv[0], stderr );
		}
		fflush(NULL);
		return EXIT_FAILURE;
	}

	// If help was requested, just prints a help message and returns.
	if ( help ) {
		print_generate_matrix_help( argv[0], stdout );
		fflush(NULL);
		return EXIT_SUCCESS;
	}

	char const *restrict const filename = arguments.filename;	// Output filename
	index_t const save_bin = arguments.save_bin;			// Output file is binary (native or non-native format).
	index_t const idx_other_args = arguments.idx_other_args;	// Index in argv[] with additional arguments.

	// ----------------------------------------

	// Additional arguments.

	index_t nrows = 0, ncols = 0;
	real max_value = DEFAULT_MAXRAND;

	if ( check_additional_arguments( argc, argv, true, idx_other_args, &nrows, &ncols, &max_value, &help ) == EXIT_FAILURE ) {
		if ( help ) {
			fprintf(stderr, "\n==========\n");
			print_generate_matrix_help( argv[0], stderr );
		}
		fflush(NULL);
		return EXIT_FAILURE;
	}

	// ----------------------------------------

	#if NMFGPU_DEBUG || NMFGPU_DEBUG_READ_MATRIX || NMFGPU_DEBUG_READ_MATRIX2 \
		|| NMFGPU_DEBUG_READ_FILE || NMFGPU_DEBUG_READ_FILE2
		// Removes the buffer associated to 'stdout' in order to prevent losing messages if the program crashes.
		setbuf( stdout, NULL );
	#endif

	// ----------------------------------------

	// Generates the output matrix
	index_t nitems = nrows * ncols;

	printf( "\nGenerating a %" PRI_IDX "-by-%" PRI_IDX " data matrix (%" PRI_IDX " items, with random values in range [0..%g])...\n",
		nrows, ncols, nitems, max_value );

	// ----------------------------------------

	#if NMFGPU_PROFILING_GLOBAL
		// Elapsed time
		gettimeofday( &t_tv, NULL );
	#endif

	// ----------------------------------------

	real *restrict matrix = malloc( nitems * sizeof(real) );
	if ( ! matrix ) {
		int const err = errno; fflush(stdout); errno = err;
		perror("\nmalloc(matrix)");
		fflush(NULL);
		return EXIT_FAILURE;
	}

	// Seed for the random generator.
	index_t seed = get_seed();

	srandom( (unsigned int) seed );	// Initializes the random generator.

	for ( index_t i=0 ; i < nitems ; i++ ) {
		real val = ( ((real) random() ) / ((real) RAND_MAX) ) * max_value;	// Value in range [0..<max_value>]
		matrix[ i ] = val;
	}

	// ----------------------------------------

	// Saves the output matrix.

	struct matrix_labels ml = NEW_MATRIX_LABELS( NULL, NULL, NULL, NULL, NULL );

	int status = matrix_save( filename, save_bin, matrix, nrows, ncols, false, &ml, ncols );
	if ( status == EXIT_FAILURE ) {
		free( matrix );
		fflush(NULL);
		return EXIT_FAILURE;
	}

	// ----------------------------------------

	#if NMFGPU_PROFILING_GLOBAL
		// Elapsed time
		{
			struct timeval t_ftv, t_etv;
			gettimeofday( &t_ftv, NULL );
			timersub( &t_ftv, &t_tv, &t_etv );	// etv = ftv - tv
			double const total_time = t_etv.tv_sec + ( t_etv.tv_usec * 1e-06 );
			printf( "\nDone in %g seconds.\n", total_time );
		}
	#endif

	// ----------------------------------------

	free( matrix );

	fflush(NULL);

	return EXIT_SUCCESS;

} // main

//////////////////////////////////////////////////
