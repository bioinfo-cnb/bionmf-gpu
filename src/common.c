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
 * common.c
 *	Some generic definitions, constants, macros and functions used by bioNMF-mGPU.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Debug / Testing:
 *		NMFGPU_FIXED_INIT: Uses "random" values generated from a fixed seed (defined in common.h).
 *
 **********************************************************
 **********************************************************
 **********************************************************
 *
 * Data matrices:
 * 	V (N rows, M columns): Input matrix,
 * 	K: Factorization Rank,
 * 	W (N,K): Output matrix,
 * 	H (K,M): Output matrix,
 * so that V ~ W*H
 *
 * NOTE: In order to improve performance:
 *	- Matrix H is stored in memory as COLUMN-major (i.e., it is transposed).
 *
 *	- All matrices include useless data for padding. Padded dimensions
 *	  are denoted with the 'p' character, e.g., 'Mp' (i.e.,, M + padding)
 *	  or 'Kp' (factorization_rank + padding).
 *
 *	- Padded dimensions are a multiple of memory_alignment
 *	  (a global variable which currently is equal to warpSize or warpSize/2).
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
 * WARNING:
 *	- Requires support for ISO-C99 standard. It can be enabled with 'gcc -std=c99'.
 *
 *********************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <inttypes.h>	/* strtoumax */
#include <unistd.h>	/* getopt */
#include <ctype.h>	/* isprint */
#if ! NMFGPU_FIXED_INIT
	#include <sys/time.h>	/* gettimeofday */
#endif

#include "common.h"

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////

/*
 * Prints to the specified file all arguments regarding
 * the input matrix (e.g., matrix dimensions and format).
 */
void help_matrix( FILE *restrict file )
{

	// Checks for NULL parameters
	if ( ! file ) {
		fflush( stdout );
		errno = EFAULT;
		perror("\nhelp_matrix( file=NULL )");
		return;
	}

	// ---------------------------

	fprintf( file, "<filename>\n\tInput data matrix (mandatory if 'help' is not requested).\n\n");

	fprintf( file, "-B,-b <native>\n\tBinary input file in \"native\" ('-b 1') or \"non-native\" format ('-b 0').\n"
			"\tIn NON-native format, the file is read assuming it was written using double-precision data, and unsigned "
			"integers for matrix\n\tdimensions, regardless how the program was compiled.\n"
			"\tOtherwise, in \"native\" format, the file is read using the data types specified at compilation "
			"(e.g., floats and signed int).\n\tPlease note that native mode skips most error-checking and information messages.\n"
			"\tThe default (if '-b' is not specified) is to read input data from an ASCII-text file.\n\n" );

	fprintf( file, "-C,-c\tInput text file has numeric column headers (disabled by default, ignored for binary files).\n\n");

	fprintf( file, "-R,-r\tInput text file has numeric row labels (disabled by default, ignored for binary files).\n\n");

	fprintf( file, "-E,-e <native>\n\tWrite output files as \"native\" ('-e 1') or \"non-native\" binary format ('-e 0').\n"
			"\tIn NON-native format, the file is written using double-precision data, and unsigned integers for matrix dimensions, "
			"regardless\n\thow the program was compiled.\n"
			"\tOtherwise, in \"native\" or raw format, the file is written using the data types specified at compilation "
			"(e.g., floats and signed int).\n\tPlease note that native mode skips error checking, information messages, and "
			"data transformation (e.g., matrix transposing).\n"
			"\tThe default (if '-e' is not specified) is to write output data to an ASCII-text file.\n\n" );

} // help_matrix

///////////////////////////////////////////////////////

/*
 * Prints to the specified file all arguments regarding
 * the NMF algorithm (e.g., factorization rank, number of iterations, etc).
 */
void help_nmf( FILE *restrict file )
{

	if ( ! file ) {
		fflush( stdout );
		errno = EFAULT;
		perror("\nhelp_nmf( file=NULL )");
		return;
	}

	// ---------------------------

	fprintf( file, "-K,-k <factorization_rank>\n\tFactorization Rank (default: K=%" PRI_IDX ").\n\n", DEFAULT_K );

	fprintf( file, "-I,-i <nIters>\n\tMaximum number of iterations (%" PRI_IDX " by default).\n\n", DEFAULT_NITERS );

	fprintf( file, "-J,-j <niter_test_conv>\n\tPerform a convergence test each <niter_test_conv> iterations (default: %" PRI_IDX ").\n"
			"\tIf this value is greater than <nIters> (see '-i' option), no test is performed\n\n", DEFAULT_NITER_CONV );

	fprintf( file, "-T,-t <stop_threshold>\n\tStopping threshold (default: %" PRI_IDX ").\n"
			"\tWhen matrix H has not changed on the last <stop_threshold> times that the convergence test\n\thas been performed, "
			"it is considered that the algorithm has converged to a solution and stops it.\n\n", DEFAULT_STOP_THRESHOLD );

} // help_nmf

///////////////////////////////////////////////////////

/*
 * Prints to the specified file all arguments regarding
 * bioNMF-mGPU main program (single or multi-GPU version).
 */
void print_nmf_gpu_help( char const *restrict const execname, FILE *restrict file )
{

	// Checks for NULL parameters
	if ( ! ( (size_t) execname * (size_t) file ) ) {
		fflush( stdout );
		errno = EFAULT;
		if ( ! execname ) perror("\nprint_nmf_gpu_help( execname )");
		if ( ! file )	perror("\nprint_nmf_gpu_help( file )");
		return;
	}

	// ---------------------------

	fprintf( file, "\n\t<< bioNMF-mGPU: Non-negative Matrix Factorization on GPU for Biology >>\n\n"
			"Usage:\n\t%s <filename> [ -b native_format ] [ -cr ] [ -k <factorization_rank> ] [ -i <nIters> ] "
			"[ -j <niter_test_conv> ] [ -t <stop_threshold> ] [ -e native_format ] [ -z device_ID ]\n\t%s -h\n\n",
			execname, execname );

	fprintf( file, "---------------\n\nData matrix options:\n\n" );
	help_matrix( file );

	fprintf( file, "---------------\n\nNMF options:\n\n" );
	help_nmf( file );

	fprintf( file, "---------------\n\nOther options:\n\n" );

	fprintf( file, "-Z,-z <GPU device>\n\tGPU device ID to attach on (default: %i).\n\t"
			"On the multi-GPU version, devices will be selected from this value.\n\n", DEFAULT_GPU_DEVICE );

	fprintf( file, "-h,-H\tPrints this help message.\n\n" );

} // print_nmf_gpu_help

///////////////////////////////////////////////////////

/*
 * Checks all arguments.
 *
 * If verbose_error is 'true', shows error messages.
 *
 * Sets 'help' to 'true' if help message was requested ('-h' or '-H' options).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int check_arguments( int argc, char const *restrict *restrict argv, bool verbose_error, bool *restrict help,
			struct input_arguments *restrict arguments )
{

	// Checks for invalid parameters
	if ( ! ( (argc > 0) * (size_t) argv * (size_t) arguments * (size_t) help ) ) {
		fflush( stdout );
		errno = EFAULT;
		if ( ! argv )	perror("\ncheck_arguments( argv )");
		if ( ! help )	perror("\ncheck_arguments( help )");
		if ( ! arguments ) perror("\ncheck_arguments( arguments )");
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
				uintmax_t val = strtoumax( optarg, &endptr, 10 );
				if ( (*endptr != '\0') + errno ) {
					fflush(stdout);
					if ( verbose_error )
						fprintf( stderr, "\nError. Invalid binary mode for input file: '%s'. It must be a "
								"non-negative integer value.\n", optarg );
					return EXIT_FAILURE;
				}
				l_is_bin = (index_t) ( val ? 2 : 1 );
			} break;


			// Input file has numeric column headers
			case 'C':
			case 'c':
				l_numeric_hdrs = true;
			break;


			// Saves as a binary file.
			case 'E':
			case 'e': {
				errno = 0;
				char *endptr = NULL;
				uintmax_t val = strtoumax( optarg, &endptr, 10 );
				if ( (*endptr != '\0') + errno ) {
					fflush(stdout);
					if ( verbose_error )
						fprintf( stderr, "\nError. Invalid binary mode for output file(s): '%s'. It must be a "
								"non-negative integer value.\n", optarg );
					return EXIT_FAILURE;
				}
				l_save_bin = (index_t) ( val ? 2 : 1 );
			} break;


			// Prints a help message.
			case 'H':
			case 'h':
				*help = true;		// Help is printed on return of this function.
				return EXIT_SUCCESS;
			// break;	// Unreachable statement


			// nIters
			case 'I':
			case 'i': {
				errno = 0;
				char *endptr = NULL;
				uintmax_t val = strtoumax( optarg, &endptr, 10 );
				if ( (*endptr != '\0') + errno + (! val) + (val > IDX_MAX) ) {
					fflush(stdout);
					if ( verbose_error )
						fprintf( stderr, "\nError. Invalid number of iterations: '%s'. "
								"It must be a positive integer value less than or equal to %" PRI_IDX "\n",
								optarg, IDX_MAX );
					return EXIT_FAILURE;
				}
				l_nIters = (index_t) val;
			} break; // nIters


			// niter_test_conv
			case 'J':
			case 'j': {
				errno = 0;
				char *endptr = NULL;
				uintmax_t val = strtoumax( optarg, &endptr, 10 );
				if ( (*endptr != '\0') + errno + (! val) + (val > IDX_MAX) ) {
					fflush(stdout);
					if ( verbose_error )
						fprintf( stderr, "\nError. Invalid number of iterations for convergence test: '%s'. "
								"It must be a positive integer value less than or equal to %" PRI_IDX "\n",
								optarg, IDX_MAX );
				}
				l_niter_test_conv = (index_t) val;
			} break; // niter_test_conv


			// kStart
			case 'K':
			case 'k': {
				errno = 0;
				char *endptr = NULL;
				uintmax_t val = strtoumax( optarg, &endptr, 10 );
				if ( (*endptr != '\0') + errno + (val < 2) + (val > IDX_MAX) ) {
					fflush(stdout);
					if ( verbose_error )
						fprintf( stderr, "\nError: invalid factorization rank: '%s'. "
								"It must be an integer value between 2 and %" PRI_IDX "\n", optarg, IDX_MAX );
					return EXIT_FAILURE;
				}
				l_k = (index_t) val;
			} break; // k


			// Input file has numeric row labels
			case 'R':
			case 'r':
				l_numeric_lbls = true;
			break;


			// stop_threshold
			case 'T':
			case 't': {
				errno = 0;
				char *endptr = NULL;
				uintmax_t val = strtoumax( optarg, &endptr, 10 );
				if ( (*endptr != '\0') + errno + (! val) + (val > IDX_MAX) ) {
					fflush(stdout);
					if ( verbose_error )
						fprintf( stderr, "\nError: invalid stopping threshold '%s'. "
								"It must be a positive integer value less than or equal to %" PRI_IDX "\n",
								optarg, IDX_MAX );
					return EXIT_FAILURE;
				}
				l_stop_threshold = (index_t) val;
			} break; // stop_threshold


			// Device ID
			case 'Z':
			case 'z': {
				errno = 0;
				char *endptr = NULL;
				uintmax_t val = strtoumax( optarg, &endptr, 10 );
				if ( (*endptr != '\0') + errno + (val > INT_MAX) ) {
					fflush(stdout);
					if ( verbose_error )
						fprintf( stderr, "\nError: invalid basis device ID number '%s'. "
								"It must be a non-negative integer value less than or equal to %" PRI_IDX "\n",
								optarg, INT_MAX );
					return EXIT_FAILURE;
				}
				l_gpu_device = (index_t) val;
			}
			break;


			// Missing argument
			case ':':
				fflush(stdout);
				if ( verbose_error )
					fprintf( stderr, "\nError: option -%c requires an argument.\n", optopt );
				return EXIT_FAILURE;
			// break;	// Unreachable statement


			// Invalid option
			case '?':
				fflush(stdout);
				if ( verbose_error ) {
					if ( optopt ) {
						if ( isprint( optopt ) )
							fprintf( stderr, "\nError: invalid option: '-%c'.\n", optopt );
						else
							fprintf( stderr, "\nError: invalid option character: '\\x%x'.\n", optopt );
					}
					else
						fprintf( stderr, "\nError: invalid option: '%s'.\n", argv[optind-1] );
				}
				return EXIT_FAILURE;
			// break;	// Unreachable statement

		} // switch( opt )

	} // while there are options to read.

	// -----------------------------

	// Checks non-option argument(s)

	// Filename
	if ( optind >= argc ) {
		fflush(stdout);
		if ( verbose_error )
			fprintf( stderr, "\nError: No filename. Not enough arguments.\n" );
		*help = true;		// Help is printed on return of this function.
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
	l_arguments.nIters = l_nIters;
	l_arguments.niter_test_conv = l_niter_test_conv;
	l_arguments.stop_threshold = l_stop_threshold;

	l_arguments.gpu_device = l_gpu_device;

	l_arguments.idx_other_args = l_idx_other_args;

	*arguments = l_arguments;

	return EXIT_SUCCESS;

} // check_arguments

///////////////////////////////////////////////////////

/*
 * Gets the difference between classification and last_classification vectors
 */
index_t get_difference( index_t const *restrict classification, index_t const *restrict last_classification, index_t m )
{
	index_t diff = 0;

	for ( index_t i = 0 ; i < (m-1); i++ ) {
		for ( index_t j = (i+1) ; j < m ; j++ ) {
			index_t conn = ( classification[j] == classification[i] );
			index_t conn_last = ( last_classification[j] == last_classification[i] );
			diff += ( conn != conn_last );
		}
	}

	return diff;

} // get_difference

///////////////////////////////////////////////////////

/*
 * Retrieves a "random" value that can be used as seed.
 *
 * If NMFGPU_FIXED_INIT is non-zero, returns FIXED_SEED.
 */
index_t get_seed( void )
{

	index_t seed = FIXED_SEED;

	#if ! NMFGPU_FIXED_INIT

		// Reads the seed from the special file /dev/urandom
		FILE *restrict file = fopen("/dev/urandom", "r");

		if ( ( ! file ) || ( ! fread( &seed, sizeof(index_t), 1, file ) ) ) {

			if ( file )
				fclose(file);

			// Failed to read: Sets the seed from the clock.

			struct timeval tv;
			gettimeofday(&tv, NULL);
			time_t usec = tv.tv_usec;
			seed = (index_t) usec;

		}

	#endif /* NMFGPU_FIXED_INIT */

	return seed;

} // get_seed

///////////////////////////////////////////////////////
