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
#if NMFGPU_PROFILING_GLOBAL
	#include <sys/time.h>
#endif

#include "index_type.h"
#include "real_type.h"
#include "matrix/matrix_io_routines.h"
#include "matrix/matrix_io.h"
#include "common.h"

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
		if ( ! file )	  perror("\nprint_file_converter_help( file )");
		return;
	}

	// ---------------------------

	fprintf( file, "\n\t<< ASCII-Binary file converter >>\n\n"
		"Usage:\n\t%s <filename> [ -b <native_format> ] [ -cr ] [ -e <native_format> ]\n\t%s -h\n\n", execname, execname );

	help_matrix( file );

	fprintf( file, "-h,-H\tPrints this help message.\n\n" );

} // print_file_converter_help

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
			print_file_converter_help( argv[0], stderr );
		}
		fflush(NULL);
		return EXIT_FAILURE;
	}

	// If help was requested, just prints a help message and returns.
	if ( help ) {
		print_file_converter_help( argv[0], stdout );
		fflush(NULL);
		return EXIT_SUCCESS;
	}

	char const *restrict const filename = arguments.filename;	// Input data filename.
	bool const numeric_hdrs = arguments.numeric_hdrs;		// Has numeric columns headers.
	bool const numeric_lbls = arguments.numeric_lbls;		// Has numeric row labels.
	index_t const is_bin = arguments.is_bin;			// Input  file is binary (native or non-native format).
	index_t const save_bin = arguments.save_bin;			// Output file is binary (native or non-native format).

	// ----------------------------------------

	real *restrict matrix = NULL;
	index_t nrows = 0, ncols = 0;
	struct matrix_labels ml = NEW_MATRIX_LABELS( NULL, NULL, NULL, NULL, NULL );
	int status = EXIT_SUCCESS;

	#if NMFGPU_DEBUG || NMFGPU_DEBUG_READ_MATRIX || NMFGPU_DEBUG_READ_MATRIX2 \
		|| NMFGPU_DEBUG_READ_FILE || NMFGPU_DEBUG_READ_FILE2
		// Removes the buffer associated to 'stdout' in order to prevent losing messages if the program crashes.
		setbuf( stdout, NULL );
	#endif

	// --------------------------------------------------------------------- //

	#if NMFGPU_PROFILING_GLOBAL
		// Elapsed time
		gettimeofday( &t_tv, NULL );
	#endif

	// --------------------------------------------------------------------- //

	// Loads the file.

	status = matrix_load( filename, numeric_hdrs, numeric_lbls, is_bin, &matrix, &nrows, &ncols, &ml );
	if ( status == EXIT_FAILURE ) {
		fprintf( stderr, "Error reading '%s'\n", filename );
		fflush(NULL);
		return EXIT_FAILURE;
	}

	// --------------------------------------------------------------------- //

	#if NMFGPU_DEBUG
		// Shows matrix content
		status = matrix_show( matrix, nrows, ncols, ncols, false, &ml );
		if ( status == EXIT_FAILURE ) {
			fprintf( stderr, "Error showing '%s'\n", filename );
			matrix_clean( matrix, ml );
			fflush(NULL);
			return EXIT_FAILURE;
		}
	#endif

	// --------------------------------------------------------------------- //


	// Output filename.

	char *restrict const filename_str = malloc( (strlen(filename) + 16)*sizeof(char) );
	if ( ! filename_str ) {
		int const err = errno; fflush(stdout); errno = err;
 		perror("\nmalloc(filename_str)");
		fprintf(stderr,"Error in '%s'\n",argv[0]);
		matrix_clean( matrix, ml );
		fflush(NULL);
		return EXIT_FAILURE;
	}

	// Saves output as "native" binary.
	if ( save_bin > 1 ) {
		if ( sprintf(filename_str, "%s.native.dat", filename) <= 0 )
			status = EXIT_FAILURE;
	}

	// Saves output as (non-"native") binary.
	else if ( save_bin ) {
		if ( sprintf(filename_str, "%s.dat", filename) <= 0 )
			status = EXIT_FAILURE;
	}

	// Saves output as ASCII text.
	else if ( sprintf(filename_str, "%s.txt", filename) <= 0 )
		status = EXIT_FAILURE;


	if ( status == EXIT_FAILURE )  {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nsprintf( filename='%s' ): ", filename );
		if ( err ) fprintf( stderr, "%s\n", strerror(err) );
		fprintf(stderr,"Error setting output filename.\n");
		free( filename_str ); matrix_clean( matrix, ml );
		fflush(NULL);
		return EXIT_FAILURE;
	}

	printf("\nOutput file: %s\n",filename_str);

	// --------------------------------------------------------------------- //

	// Saves the file.

	status = matrix_save( filename_str, save_bin, matrix, nrows, ncols, false, &ml, ncols, true ); // be verbose
	if ( status == EXIT_FAILURE ) {
		free( filename_str ); matrix_clean( matrix, ml );
		fflush(NULL);
		return EXIT_FAILURE;
	}

	// --------------------------------------------------------------------- //

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

	// --------------------------------------------------------------------- //

	free(filename_str);

	matrix_clean( matrix, ml );

	fflush(NULL);

	return status;

} // main

///////////////////////////////////////////////////////
