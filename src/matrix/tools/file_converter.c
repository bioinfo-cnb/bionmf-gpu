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
 * file_converter.c
 *	Program to perform binary-text file conversions.
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

#include "index_type.h"
#include "real_type.h"
#include "matrix_io/matrix_io_routines.h"
#include "matrix_io/matrix_io.h"

// -----------------------------------------------

/* Structure for arguments */

struct input_arguments {
	char const *restrict filename;	// Input filename.
	bool is_bin;			// Input file is (non-"native") binary.
	bool numeric_hdrs;		// Has numeric columns headers.
	bool numeric_lbls;		// Has numeric row labels.
	bool save_native;		// Save matrix in a "native"-binary file.
};

//////////////////////////////////////////////////

/*
 * Prints all arguments to the specified file.
 */
static void print_file_converter_help( char const *restrict const execname, FILE *restrict file )
{

	// Checks for NULL parameters
	if ( ! ( (size_t) execname * (size_t) file ) ) {
		fflush( stdout );
		errno = EFAULT;
		if ( ! execname ) perror("\nprint_file_converter_help( execname )");
		if ( ! file )	perror("\nprint_file_converter_help( file )");
		return;
	}

	// ---------------------------

	fprintf( file, "\n\t<< ASCII-Binary file converter >>\n\n"
		"Usage:\n\t%s <filename> [ -bcrs ]\n\t%s [ -h | --help ]\n\n", execname, execname );

	fprintf( file, "<filename>\tInput data matrix (mandatory if 'help' is not requested).\n\n");

	fprintf( file, "-B,-b\tInput file is in (non-\"native\") binary format (i.e., double-precision data and unsigned int's).\n"
			"\tOtherwise, (the default) input file is an ASCII-text file.\n\n" );
	fprintf( file, "-C,-c\tInput text file has numeric column headers (disabled by default).\n\n");
	fprintf( file, "-R,-r\tInput text file has numeric row labels (disabled by default).\n\n");
	fprintf( file, "-S,-s\tSaves input matrix as \"native\" binary format (i.e., with the compiled types for matrix data\n"
			"\tand dimensions; disabled by default).\n\n");

	fprintf( file, "-h,-H,--help,--HELP\tPrints this help message.\n\n" );

} // print_file_converter_help

//////////////////////////////////////////////////

/*
 * Checks all arguments.
 *
 * Sets 'help' to 'true' if help message was requested ('-h', '-H', '--help' and '--HELP' options).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int check_file_converter_arguments( int argc, char *restrict const *restrict argv, bool *restrict help,
					struct input_arguments *restrict arguments )
{

	// Checks for NULL parameters
	if ( ! ( (size_t) argv * (size_t) arguments * (size_t) help ) ) {
		fflush( stdout );
		errno = EFAULT;
		if ( ! argv )	perror("\ncheck_file_converter_arguments( argv )");
		if ( ! help )	perror("\ncheck_file_converter_arguments( help )");
		if ( ! arguments ) perror("\ncheck_file_converter_arguments( arguments )");
		return EXIT_FAILURE;
	}

	// ---------------------------

	// Default values
	char const *l_filename = NULL;	// Input filename
	bool l_is_bin = false;		// File is (non-"native") binary.
	bool l_numeric_hdrs = false;	// Has numeric columns headers.
	bool l_numeric_lbls = false;	// Has numeric row labels.
	bool l_save_native = false;	// Save matrix in a "native"-binary file.

	int opt = 0;	// Selected option
	opterr = 0;	// Disables error messages.

	// Long options: Help
	struct option const longopt[3] = { { "help", no_argument, NULL, 'h' }, { "HELP", no_argument, NULL, 'H' }, { NULL, 0, NULL, 0 } };

	/*
	 * Reads option arguments:
	 *
	 *	-b (non-"native"-binary input file
	 *	-c (input file has numeric column headers)
	 *	-h | -H | --help | --HELP
	 *	-r (input file has numeric row labels)
	 *	-s (save matrix in a "native"-binary file)
	 */

	// NOTE: First colon (':') indicates to return ':' instead of '?' in case of a missing option argument.
	while ( (opt = getopt_long( argc, (char *const *restrict) argv, ":BbCcHhRrSs", longopt, NULL) ) != -1 ) {

		switch( opt ) {

			// Input file is (non-"native") binary.
			case 'B':
			case 'b':
				l_is_bin = true;
			break;


			// Input file has numeric column headers
			case 'C':
			case 'c':
				l_numeric_hdrs = true;
			break;


			// Prints a help message.
			case 'H':
			case 'h':
				*help = true;		// Help is printed on return of this function.
				return EXIT_SUCCESS;
			// break;	// Unreachable statement


			// Input file has numeric row labels
			case 'R':
			case 'r':
				l_numeric_lbls = true;
			break;


			// Save input matrix in a "native"-binary file.
			case 'S':
			case 's':
				l_save_native = true;
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
				} else
					fprintf( stderr, "\nError: invalid option: '%s'.\n", argv[optind-1] );
				*help = true;		// Help is printed on return of this function.
				return EXIT_FAILURE;
			// break;	// Unreachable statement

		} // switch( opt )

	} // while there are options to read.

	// -----------------------------

	// Checks non-option argument(s)

	// Input filename
	if ( optind >= argc ) {
		fflush(stdout);
		fprintf( stderr, "\nError: No input filename. Not enough arguments.\n" );
		*help = true;		// Help is printed on return of this function.
		return EXIT_FAILURE;
	}
	l_filename = argv[optind];
	// optind++;	// Not necessary.

	// -----------------------------

	// Resets extern variables.
	optarg = NULL;
	optind = opterr = optopt = 0;

	// --------------------

	// Sets output values.

	struct input_arguments l_arguments;

	l_arguments.filename = l_filename;

	l_arguments.is_bin = l_is_bin;
	l_arguments.numeric_hdrs = l_numeric_hdrs;
	l_arguments.numeric_lbls = l_numeric_lbls;
	l_arguments.save_native = l_save_native;

	*arguments = l_arguments;

	return EXIT_SUCCESS;

} // check_file_converter_arguments

//////////////////////////////////////////////////
//////////////////////////////////////////////////

int main( int argc, char *restrict const *restrict argv )
{

	/* Reads all parameters and performs error-checking. */

	bool help = false;	// Help message requested

	struct input_arguments arguments;

	// Checks all arguments (shows error messages).
	if ( check_file_converter_arguments( argc, argv, &help, &arguments ) == EXIT_FAILURE ) {
		if ( help ) {
			fprintf(stderr, "\n==========\n");
			print_file_converter_help( *argv, stderr );
		}
		fflush(NULL);
		return EXIT_FAILURE;
	}

	// If help was requested, just prints a help message and returns.
	if ( help ) {
		print_file_converter_help( *argv, stdout );
		fflush(NULL);
		return EXIT_SUCCESS;
	}

	char const *restrict const filename = arguments.filename;	// Input data filename.
	bool const is_bin = arguments.is_bin;				// File is (non-"native") binary.
	bool const numeric_hdrs = arguments.numeric_hdrs;		// Has numeric columns headers.
	bool const numeric_lbls = arguments.numeric_lbls;		// Has numeric row labels.
	bool const save_native = arguments.save_native;			// Save input matrix in a "native"-binary file

	// ----------------------------------------

	real *restrict matrix = NULL;
	index_t nrows = 0, ncols = 0;
	struct matrix_labels ml = NEW_MATRIX_LABELS( NULL, NULL, NULL, NULL, NULL );
	int status = EXIT_SUCCESS;

	// --------------------------------------------------------------------- //

	// Loads the file.

	printf("\nLoading input file...\n");

	if ( is_bin ) {	// Input file is (non-native) binary.

		printf("\tFile selected as (non-\"native\") binary (i.e., double-precision data and unsigned int's). Loading...\n");

		status = matrix_load_binary_verb( filename, &matrix, &nrows, &ncols, &ml );

	} else { // Input file is ASCII-text.

		printf("\nFile selected as ASCII-text. Loading...\n"
			"\t\tData matrix selected as having numeric column headers: %s.\n"
			"\t\tData matrix selected as having numeric row labels: %s.\n",
			( numeric_hdrs ? "Yes" : "No" ), ( numeric_lbls ? "Yes" : "No" ) );

		status = matrix_load_ascii_verb( filename, numeric_hdrs, numeric_lbls, &matrix, &nrows, &ncols, &ml );

	} // If file is binary or text.

	if ( status == EXIT_FAILURE ) {
		fprintf( stderr, "Error reading '%s'\n", filename );
		fflush(NULL);
		return EXIT_FAILURE;
	}

	// --------------------------------------------------------------------- //


	// Saves the file with the new format.

	char *restrict const filename_str = malloc( (strlen(filename) + 16)*sizeof(char) );
	if ( ! filename_str ) {
		int const err = errno; fflush(stdout); errno = err;
 		perror("\nmalloc(filename_str)");
		fprintf(stderr,"Error in '%s'\n",argv[0]);
		matrix_clean( matrix, ml );
		fflush(NULL);
		return EXIT_FAILURE;
	}

	printf("Saving output file in ");

	// Saves output as "native" binary.
	if ( save_native ) {
		printf("\"native\" binary format (with the compiled types for matrix data and dimensions)...\n");
		if ( sprintf(filename_str, "%s.native.dat", filename) <= 0 ) {
			int const err = errno; fflush(stdout); errno = err;
			fprintf( stderr, "\nsprintf( filename='%s' ): ", filename );
			if ( err ) fprintf( stderr, "%s\n", strerror(err) );
			fprintf(stderr,"Error setting output filename.\n");
			status = EXIT_FAILURE;
		} else {
			printf("Output file: %s\n",filename_str);
			status = matrix_save_binary_native( filename_str, matrix, nrows, ncols, &ml );
		}
	}

	// Input file is binary. Saves as text.
	else if ( is_bin ) {
		printf("ASCII text...\n");
		if ( sprintf(filename_str, "%s.txt", filename) <= 0 ) {
			int const err = errno; fflush(stdout); errno = err;
			fprintf( stderr, "\nsprintf( filename='%s' ): ", filename );
			if ( err ) fprintf( stderr, "%s\n", strerror(err) );
			fprintf(stderr,"Error setting output filename.\n");
			status = EXIT_FAILURE;
		} else {
			printf("Output file: %s\n",filename_str);
			status = matrix_save_ascii( filename_str, matrix, nrows, ncols, false, false, &ml, ncols );
		}
	}

	// Input file is text. Saves as (non-"native") binary.
	else {
		printf("(non-\"native\") binary format (i.e., double-precision data and unsigned int's)...\n");
		if ( sprintf(filename_str, "%s.dat", filename) <= 0 ) {
			int const err = errno; fflush(stdout); errno = err;
			fprintf( stderr, "\nsprintf( filename='%s' ): ", filename );
			if ( err ) fprintf( stderr, "%s\n", strerror(err) );
			fprintf(stderr,"Error setting output filename.\n");
			status = EXIT_FAILURE;
		} else {
			printf("Output file: %s\n",filename_str);
			status = matrix_save_binary( filename_str, matrix, nrows, ncols, false, &ml, ncols );
		}
	}

	if ( status == EXIT_SUCCESS )
		printf("Done.\n\n");
	else
		fprintf( stderr, "Error in %s\n",argv[0] );

	free(filename_str);

	matrix_clean( matrix, ml );

	fflush(NULL);

	return status;

} // main
