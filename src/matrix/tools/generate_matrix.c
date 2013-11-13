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
 * WARNING:
 *	- Requires support for ISO-C99 standard. It can be enabled with 'gcc -std=c99'.
 *
 **********************************************************/

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <getopt.h>
#include <inttypes.h>	/* strtoumax */
#include <math.h>	/* isless */

#include "index_type.h"
#include "real_type.h"
#include "matrix_io/matrix_io_routines.h"
#include "matrix_io/matrix_io.h"

// -----------------------------------------------

/* Structure for arguments */

struct input_arguments {
	char const *restrict filename;	// Output filename.
	index_t nrows;			// Number of rows.
	index_t ncols;			// Number of columns.
	bool save_bin;			// Binary output file.
	bool save_non_native;		// Non-"native" binary format (i.e., double-precision data).
	real max_value;			// Maximum value.
};

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
			"Usage:\n\t%s <filename> <rows> <columns> [ -m <max_value> ] [ -bs ]\n\t%s [ -h | --help ]\n\n",
			execname, execname );

	fprintf( file,	"<filename>\tOutput filename (mandatory if 'help' is not requested).\n\n");
	fprintf( file,	"<rows> <columns>\tOutput matrix dimensions (both mandatory if 'help' is not requested).\n"
			"Note that <rows> x <columns> must be less than, or equal to, %" PRI_IDX ".\n\n", IDX_MAX);

	fprintf( file,	"-B,-b\tBinary output file (ASCII-text file by default).\n\n" );
	fprintf( file,	"-M,-m <max_value>\n\tSets the range of random values to [0..<max_value>]. The default is 1.0. "
			"Valid values are between %g and %g.\n\n", R_MIN, R_MAX);
	fprintf( file,	"-S,-s\t(Non-\"native\") Binary format (i.e., double-precision data and unsigned int's). It implies the '-B' option.\n"
			"\tThe default is to use the machine's \"native\" format (i.e., with the compiled types for matrix data "
			"and dimensions).\n\n");

	fprintf( file,	"-h,-H,--help,--HELP\tPrints this help message.\n\n" );

} // print_generate_matrix_help

//////////////////////////////////////////////////

/*
 * Checks all arguments.
 *
 * Sets 'help' to 'true' if help message was requested ('-h', '-H', '--help' and '--HELP' options).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int check_generate_matrix_arguments( int argc, char *restrict const *restrict argv, bool *restrict help,
					struct input_arguments *restrict arguments )
{

	// Checks for NULL parameters
	if ( ! ( (size_t) argv * (size_t) arguments * (size_t) help ) ) {
		fflush( stdout );
		errno = EFAULT;
		if ( ! argv )	perror("\ncheck_generate_matrix_arguments( argv )");
		if ( ! help )	perror("\ncheck_generate_matrix_arguments( help )");
		if ( ! arguments ) perror("\ncheck_generate_matrix_arguments( arguments )");
		return EXIT_FAILURE;
	}

	// ---------------------------

	// Default values
	char const *l_filename = NULL;		// Output filename
	index_t l_nrows = 0;			// Number of rows.
	index_t l_ncols = 0;			// Number of columns.
	bool l_save_bin = false;		// Binary output file.
	bool l_save_non_native = false;		// Non-"native" binary format (i.e., double-precision data).
	real l_max_value = REAL_C( 1.0 );	// Maximum random value.

	int opt = 0;	// Selected option
	opterr = 0;	// Disables error messages.

	// Long options: Help
	struct option const longopt[3] = { { "help", no_argument, NULL, 'h' }, { "HELP", no_argument, NULL, 'H' }, { NULL, 0, NULL, 0 } };

	/*
	 * Reads option arguments:
	 *
	 *	-b ("native"-binary input file)
	 *	-m <max_value> (maximum random value)
	 *	-s (Non-"native"-binary format)
	 */

	// NOTE: First colon (':') indicates to return ':' instead of '?' in case of a missing option argument.
	while ( (opt = getopt_long( argc, (char *const *restrict) argv, ":BbHhM:m:Ss", longopt, NULL) ) != -1 ) {

		switch( opt ) {

			// Binary file format.
			case 'B':
			case 'b':
				l_save_bin = true;
			break;


			// Prints a help message.
			case 'H':
			case 'h':
				*help = true;		// Help is printed on return of this function.
				return EXIT_SUCCESS;
			// break;	// Unreachable statement


			// Maximum random value
			case 'M':
			case 'm':
				errno = 0;
				char *endptr = NULL;
				l_max_value = STRTOREAL( optarg, &endptr );
				if ( (errno + (*endptr != '\0')) || (! isfinite(l_max_value)) ||
					isless( l_max_value, R_MIN ) || isgreater( l_max_value, R_MAX ) ) {
					fflush(stdout);
					fprintf(stderr,"\nError: invalid maximum value: '%s'\nValid values are between %g and %g.\n",
						optarg, R_MIN, R_MAX );
					return EXIT_FAILURE;
				}
			break;


			// Non-"native" binary file.
			case 'S':
			case 's':
				l_save_non_native = true;
				l_save_bin = true;		// Implies "binary format".
			break;


			// Missing argument
			case ':':
				fflush(stdout);
				fprintf( stderr, "\nError: option -%c requires an argument.\n", optopt );
				*help = true;		// Help is printed on return of this function.
				return EXIT_FAILURE;
			// break;	// Unreachable statement


			// Invalid option
			case '?':
				fflush(stdout);
				if ( optopt ) {
					if ( isprint( optopt ) )
						fprintf( stderr, "\nError: invalid option: '-%c'.\n", optopt );
					else
						fprintf( stderr, "\nError: invalid option character: '\\x%x'.\n", optopt );
				}
				else
					fprintf( stderr, "\nError: invalid option: '%s'.\n", argv[optind-1] );
				*help = true;		// Help is printed on return of this function.
				return EXIT_FAILURE;
			// break;	// Unreachable statement

		} // switch( opt )

	} // while there are options to read.

	// -----------------------------

	// Checks non-option argument(s)

	if ( optind >= (argc-3) ) {
		fflush(stdout);
		fprintf( stderr, "\nError: Not enough arguments.\n" );
		*help = true;		// Help is printed on return of this function.
		return EXIT_FAILURE;
	}

	// Output filename
	l_filename = argv[optind];
	optind++;

	// Number of rows
	{
		errno = 0;
		char *endptr = NULL;
		uintmax_t val = strtoumax( argv[optind], &endptr, 10 );
		if ( (*endptr != '\0') + errno + (! val) + (val > IDX_MAX) ) {
			fflush(stdout);
			fprintf( stderr, "\nError. Invalid number of rows: '%s'.\n", optarg );
			if ( val > IDX_MAX )
				fprintf( stderr, "It must be less than or equal to %" PRI_IDX ".\n", IDX_MAX );
			return EXIT_FAILURE;
		}
		l_nrows = (index_t) val;
	}
	optind++;

	// Number of columns
	{
		errno = 0;
		char *endptr = NULL;
		uintmax_t val = strtoumax( argv[optind], &endptr, 10 );
		if ( (*endptr != '\0') + errno + (! val) + (val > IDX_MAX) ) {
			fflush(stdout);
			fprintf( stderr, "\nError. Invalid number of columns: '%s'.\n", optarg );
			if ( val > IDX_MAX )
				fprintf( stderr, "It must be less than or equal to %" PRI_IDX ".\n", IDX_MAX );
			return EXIT_FAILURE;
		}
		l_ncols = (index_t) val;
	}
	// optind++;	// Not necessary

	// -----------------------------

	// Resets extern variables.
	optarg = NULL;
	optind = opterr = optopt = 0;

	// --------------------

	// Sets output values.

	struct input_arguments l_arguments;

	l_arguments.filename = l_filename;
	l_arguments.nrows = l_nrows;
	l_arguments.ncols = l_ncols;

	l_arguments.save_bin = l_save_bin;
	l_arguments.save_non_native = l_save_non_native;
	l_arguments.max_value = l_max_value;

	*arguments = l_arguments;

	return EXIT_SUCCESS;

} // check_generate_matrix_arguments

//////////////////////////////////////////////////

/*
 * Initializes the random generator.
 */
static void init_random()
{

	// Sets the seed.
	unsigned int seed = 0;

	// Reads the seed from the special file /dev/urandom
	FILE *restrict file = fopen("/dev/urandom", "r");

	if ( ! file ) || ( ! fread( &seed, sizeof(unsigned int), 1, file ) ) {

		if ( file )
			fclose(file);

		// Failed to read: Sets the seed from the clock.

		struct timeval tv;
		gettimeofday(&tv, NULL);
		time_t usec = tv.tv_usec;
		seed = (unsigned int) usec;

	}

	// Initializes the random generator.
	srandom( seed );

} // init_random

//////////////////////////////////////////////////
//////////////////////////////////////////////////

int main( int argc, char *restrict const *restrict argv )
{

	/* Reads all parameters and performs error-checking. */

	bool help = false;	// Help message requested

	struct input_arguments arguments;

	// Checks all arguments (shows error messages).
	if ( check_generate_matrix_arguments( argc, argv, &help, &arguments ) == EXIT_FAILURE ) {
		if ( help ) {
			fprintf(stderr, "\n==========\n");
			print_generate_matrix_help( *argv, stderr );
		}
		fflush(NULL);
		return EXIT_FAILURE;
	}

	// If help was requested, just prints a help message and returns.
	if ( help ) {
		print_generate_matrix_help( *argv, stdout );
		fflush(NULL);
		return EXIT_SUCCESS;
	}

	char const *restrict const filename = arguments.filename;	// Output filename
	index_t const nrows = arguments.nrows;				// Number of rows.
	index_t const ncols = arguments.ncols;				// Number of columns.
	bool const save_bin = arguments.save_bin;			// Binary output file.
	bool const save_non_native = arguments.save_non_native;		// Non-"native" binary format (i.e., double-precision data).
	real const max_value = arguments.max_value;			// Maximum value.

	// ----------------------------------------

	// Checks number of items.
	{
		uintmax_t nitems = nrows * ncols;
		if ( nitems > IDX_MAX ) {
			fflush(stdout);
			fprintf( stderr, "\nError: output matrix too large. The number of items must be less than, or equal to, %"
					PRI_IDX ".\n", IDX_MAX );
			return EXIT_FAILURE;
		}
	}

	// -----------------------------------

	// Generates the output matrix
	index_t nitems = nrows * ncols;

	printf( "\nGenerating a %" PRI_IDX "-by-%" PRI_IDX " data matrix (%" PRI_IDX " items, with random values in range [0..%g])...\n",
		nrows, ncols, nitems, max_value );

	real *restrict matrix = malloc( nitems * sizeof(real) );
	if ( ! matrix ) {
		int const err = errno; fflush(stdout); errno = err;
		perror("\nmalloc(matrix)");
		fflush(NULL);
		return EXIT_FAILURE;
	}

	init_random();	// Initializes the random seed.

	for ( index_t i=0 ; i < nitems ; i++ ) {
		real val = ( ((real) random() ) / ((real) RAND_MAX) ) * max_value;	// Value in range [0..<max_value>]
		matrix[ i ] = val;
	}

	// -----------------------------------

	// Saves the output matrix.

	struct matrix_labels ml = NEW_MATRIX_LABELS( NULL, NULL, NULL, NULL, NULL );
	int status = EXIT_SUCCESS;

	printf("Saving output file in ");

	if ( save_bin ) {	// Binary format

		if ( save_non_native ) {
			printf("(non-\"native\") binary format (i.e., with double-precision data, and unsigned int's as matrix dimensions)...\n");
			status = matrix_save_binary( filename, matrix, nrows, ncols, false, &ml, ncols );
		} else {
			printf("\"native\" binary format (i.e., with the compiled types for matrix data and dimensions)...\n");
			status = matrix_save_binary_native( filename, matrix, nrows, ncols, &ml );
		}

	// ASCII text
	} else {
		printf("ASCII-text format...\n");
		status = matrix_save_ascii( filename, matrix, nrows, ncols, false, false, &ml, ncols );
	}

	// -----------------------------------

	if ( status == EXIT_SUCCESS )
		printf("Done.\n");

	free( matrix );

	fflush(NULL);

	return status;

} // main
