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
 * matrix_io.c
 *	I/O methods for working with (labeled) matrices.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Debug / Testing:
 *		NMFGPU_DEBUG_READ_MATRIX: Shows information about the matrix being read (e.g., dimensions, labels, etc).
 *		NMFGPU_DEBUG_READ_MATRIX2: Shows detailed information of every datum read.
 *		NMFGPU_TESTING: When set, methods matrix_show() and matrix_int_show() show *ALL* data in matrix (not just a portion).
 *				It also uses the default decimal precision of "%g" format in printf(3) calls.
 *
 * WARNING:
 *	- Requires support for ISO-C99 standard. It can be enabled with 'gcc -std=c99'.
 *
 **********************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#ifndef __STDC_FORMAT_MACROS
	#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>	/* PRIuMAX, uintmax_t */
#include <unistd.h>	/* unlink */
#include <math.h>	/* isfinite */
#include <ctype.h>	/* isblank, isprint */

#include "matrix/matrix_io.h"

//////////////////////////////////////////////////

// Precision used in printf() functions.
#ifndef DECIMAL_PRECISION
	#if NMFGPU_TESTING
		#define DECIMAL_PRECISION ""
	#else
		#define DECIMAL_PRECISION ".8"
	#endif
#endif

//////////////////////////////////////////////////
//////////////////////////////////////////////////

/*
 * Helper function used by matrix_load_ascii() and matrix_load_ascii_verb().
 *
 * Returns 'true' if the given number is valid (i.e., is NOT 'NaN' or '+/- infinite').
 */
static inline bool is_valid( real value )
{

	// Includes FP_SUBNORMAL numbers as valid values.

	return (bool)isfinite( value );

} // is_valid

// -----------------------------------------------

/*
 * Helper function used by matrix_load_ascii() and matrix_load_ascii_verb().
 *
 * Returns 'true' if 'str' represents one or more numbers, or an empty string.
 * Returns 'false' otherwise.
 */
static bool isnumbers( char const *restrict str )
{

	if ( ! str )
		return false;

	char const *p = str;
	char const *s;
	real value = 0;
	bool isvalid;	// = is_valid( value )
	int c;		// = *p (i.e., first non-numeric character)

	errno = 0;
	do {
		s = p;
		value = STRTOREAL( s, (char **) &p ) ;	// Removes the 'const' qualifier to avoid a "pedantic" warning.
		errno = 0;
		isvalid = is_valid(value);
		c = (int) *p;
	} while ( (s != p) * isvalid * isblank(c) );	// (s != p) && isvalid && isblank(c)

	// Returns 'true' if 'value' is valid and there were no invalid characters (i.e., *p == '\0')
	return ( isvalid * (! c) );	// isvalid && (c == '\0')
	// Note: add "&& (p > str)" to return 'false' on empty strings.

} // isnumbers

// -----------------------------------------------

/*
 * Loads a matrix from an ASCII file.
 *
 * Detects automatically if matrix has name, column headers and/or row labels, as well as data delimiter (space or tab characters).
 *
 * Both matrix dimensions must be >= 2.
 *
 * In addition:
 *	- Outputs information messages.
 *	- Performs error checking.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_load_ascii_verb( char const *restrict filename, bool numeric_hdrs, bool numeric_lbls, real *restrict *restrict const matrix,
			index_t *restrict nrows, index_t *restrict ncols, struct matrix_labels *restrict ml )
{

	// Local values and pointers to output parameters.
	index_t numcols = 0, numrows = 0;
	bool hasname = false, hasheaders = false, haslabels = false;

	// Name, headers and labels
	char *name = NULL; char const *headers = NULL; char *labels = NULL;
	char *p_labels = NULL;	// Pointer to 'labels'
	size_t max_len_labels = 0, len_labels = 0;

	// Array of pointers to headers and labels
	char **pheaders = NULL; char **plabels = NULL;	// Array of pointers to 'headers' and 'labels'
	char **p_plabels = NULL;			// Pointer to 'plabels[]'

	real *restrict data_matrix = NULL;
	real *pmatrix = NULL;

	index_t max_numrows = 512;		// Initial size of plabels and data_matrix

	index_t nlines = 0;			// Number of lines processed.

	int delimiter = (int) '\t';		// Delimiter for tokens (TAB by default).

	// Data in line 1 (first non-blank line): Name and/or columns headers.
	char *restrict dataL1 = NULL;
	char **restrict pdataL1 = NULL;
	size_t len_dataL1 = 0;
	size_t ntokensL1 = 0;		// Number of tokens in line 1.


	// Data in line 2 (first line to have numeric data)
	char *dataL2 = NULL;
	char **pdataL2 = NULL;
	size_t len_dataL2 = 0;
	size_t ntokensL2 = 0;		// Number of tokens in line 2.
	char **p_pdataL2 = NULL;	// Pointer to pdataL2


	/////////////////////////////////


	// Checks for NULL parameters
	if ( ! ( (size_t) matrix * (size_t) nrows * (size_t) ncols * (size_t) ml ) ) {
		fflush( stdout );
		errno = EFAULT;
		if ( ! matrix )	perror("\nmatrix_load_ascii_verb( matrix )");
		if ( ! nrows )	perror("\nmatrix_load_ascii_verb( nrows )");
		if ( ! ncols )	perror("\nmatrix_load_ascii_verb( ncols )");
		if ( ! ml )	perror("\nmatrix_load_ascii_verb( ml )");
		return EXIT_FAILURE;
	}


	// Starts Reading ...

	FILE *restrict const file = fopen( filename, "r" );
	if( ! file ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nfopen '%s' :\n\t%s.\n", filename, strerror(errno) );
		fprintf( stderr, "Error in matrix_load_ascii_verb().\n" );
		return EXIT_FAILURE;
	}


	// Reads line 1
	len_dataL1 = read_line( file, &dataL1 );
	if ( ! dataL1 ) {
		if ( len_dataL1 )
			fprintf(stderr,"Error in matrix_load_ascii_verb().\n");
		else
			fprintf(stderr, "\nError reading input file: file is empty?\n\n");
		fclose(file);
		return EXIT_FAILURE;
	}

	// Divides into tokens by replacing all <delimiter> characters by '\0'.
	ntokensL1 = tokenize( dataL1, &pdataL1, delimiter );
	if ( ! ntokensL1 ) {
		fprintf(stderr,"Error in matrix_load_ascii_verb().\n");
		free(dataL1);
		fclose(file);
		return EXIT_FAILURE;
	}


	/////////////////////////////////


	#if NMFGPU_DEBUG_READ_MATRIX
		printf("ntokensL1=%zu\n",ntokensL1);
		fflush(stdout);
	#endif


	// Checks for overflow.
	if ( (ntokensL1-1) > (IDX_MAX/2) ) {	// -1 because the file may have a "name" field.
		printf( "\t\tNumber of items read on line 1: %zu.", ntokensL1);
		fflush(stdout);
		fprintf(stderr,"\n\nSorry, but your matrix exceeds the limits used for matrix dimensions.\nData matrices are limited to:\n"
				"\t* %" PRI_IDX " rows.\n\t*%" PRI_IDX " columns.\n\t*%" PRI_IDX " items.\n",
				IDX_MAX/2, IDX_MAX/2, IDX_MAX);
		fprintf(stderr, "\nPlease check also for any invalid line terminator. Only LF ('\\n') and CR+LF ('\\r\\n') are accepted.\n");
		free(pdataL1); free(dataL1);
		fclose(file);
		return EXIT_FAILURE;
	} // if overflows.


	/////////////////////////////////


	// Detects if file might have name and/or headers

	// Checks if all tokens, from the second one, are numeric.

	bool has_name_headers = false;	// File might have name and/or headers
	{
		index_t nt = (index_t) ntokensL1;

		index_t i = 1;
		while ( (i < nt) && isnumbers(pdataL1[i]) )
			i++;

		// File might have name and/or headers if:
		has_name_headers = (	(i < nt) +				// Not all tokens, from the second one, are numeric, <OR>
					numeric_hdrs +				// input matrix has numeric column headers, <OR>
					((nt == 1) && !isnumbers(dataL1) ) );	// It has only one (non-numeric) token.
	}

	#if NMFGPU_DEBUG_READ_MATRIX
		printf("has_name_headers=%i\n",has_name_headers);
		fflush(stdout);
	#endif

	if ( has_name_headers ) {

		// File may have name and/or headers

		// Reads the second line.
		len_dataL2 = read_line( file, &dataL2 );
		if ( ! dataL2 ) {
			fflush(stdout);
			if ( len_dataL2 )
				fprintf(stderr,"Error in matrix_load_ascii_verb().\n");
			else {
				fprintf(stderr,"\nError reading input file:\nPremature end of file detected.\n"
						"Is your data matrix stored in a single text line?\n");
				fprintf(stderr, "\nPlease check also for any invalid line terminator. "
						"Only LF ('\\n') and CR+LF ('\\r\\n') are accepted.\n");
			}
			free(pdataL1); free(dataL1);
			fclose(file);
			return EXIT_FAILURE;
		}

		// 'Tokenizes' it
		ntokensL2 = tokenize( dataL2, &pdataL2, delimiter );
		if ( ! ntokensL2 ) {
			fflush(stdout);
			fprintf(stderr,"Error in matrix_load_ascii_verb().\n");
			free(pdataL1); free(dataL1); free(dataL2);
			fclose(file);
			return EXIT_FAILURE;
		}

		// If only one token in line 2.
		if ( ntokensL2 == 1 ) {

			#if NMFGPU_DEBUG_READ_MATRIX
				printf("ntokensL2=1\n");
				fflush(stdout);
			#endif

			/* If line 1 has also only one token, "retokenizes" both L1 and L2 using a space character (' ') as delimiter. */
			if ( ntokensL1 == 1 ) {

				// Sets the new delimiter.
				delimiter = (int) ' ';

				#if NMFGPU_DEBUG_READ_MATRIX
					printf("Retokenizing dataL1...\n");
					fflush(stdout);
				#endif
				free(pdataL1);
				pdataL1 = NULL;
				ntokensL1 = tokenize( dataL1, &pdataL1, delimiter );
				if ( ! ntokensL1 ) {
					fflush(stdout);
					fprintf(stderr,"Error in matrix_load_ascii_verb().\n");
					free(dataL1); free(dataL2); free(pdataL2);
					fclose(file);
					return EXIT_FAILURE;
				}

				#if NMFGPU_DEBUG_READ_MATRIX
					printf("ntokensL1=%zu\n",ntokensL1);
					fflush(stdout);
				#endif

				// Checks for overflow.
				if ( (ntokensL1-1) > (IDX_MAX/2) ) {	// '-1' because the file may have a "name" field.
					printf( "\t\tNumber of items read on line 1: %zu.", ntokensL1);
					fflush(stdout);
					fprintf(stderr, "\n\nSorry, but your matrix exceeds the limits used for matrix dimensions.\n"
							"Data matrices are limited to:\n\t* %" PRI_IDX " rows.\n\t*%" PRI_IDX
							" columns.\n\t*%" PRI_IDX " items.\n", IDX_MAX/2, IDX_MAX/2, IDX_MAX);
					fprintf(stderr, "\nPlease check also for any invalid line terminator. "
							"Only LF ('\\n') and CR+LF ('\\r\\n') are accepted.\n");
					free(pdataL2); free(dataL2); free(pdataL1); free(dataL1);
					fclose(file);
					return EXIT_FAILURE;

				} // if overflows.

				#if NMFGPU_DEBUG_READ_MATRIX
					printf("Retokenizing dataL2...\n");
					fflush(stdout);
				#endif
				free(pdataL2);
				pdataL2 = NULL;
				ntokensL2 = tokenize( dataL2, &pdataL2, delimiter );
				if ( ! ntokensL2 ) {
					fflush(stdout);
					fprintf(stderr,"Error in matrix_load_ascii_verb().\n");
					free(dataL2); free(pdataL1); free(dataL1);
					fclose(file);
					return EXIT_FAILURE;
				}
			}

		} // If retokenize dataL2.

		#if NMFGPU_DEBUG_READ_MATRIX
			printf("ntokensL2=%zu\n",ntokensL2);
			fflush(stdout);
		#endif

		// Number of lines processed.
		nlines = 2;


	} else {  // No name and no headers.

		// Token2 is matrix[0][0] or matrix[0][1]
		hasname = false;
		hasheaders = false;

		// If line 1 has only one token, "retokenizes" it using the space character (' ') as delimiter.
		if ( ntokensL1 == 1 ) {

			// Sets the new delimiter.
			delimiter = (int) ' ';

			#if NMFGPU_DEBUG_READ_MATRIX
				printf("Retokenizing dataL1...\n");
				fflush(stdout);
			#endif
			free(pdataL1);
			pdataL1 = NULL;
			ntokensL1 = tokenize( dataL1, &pdataL1, delimiter );
			if ( ! ntokensL1 ) {
				fflush(stdout);
				fprintf(stderr,"Error in matrix_load_ascii_verb().\n");
				free(dataL1);
				fclose(file);
				return EXIT_FAILURE;
			}

			#if NMFGPU_DEBUG_READ_MATRIX
				printf("ntokensL1=%zu\n",ntokensL1);
				fflush(stdout);
			#endif

		} // if retokenizes L1.

		// Sets L1 as L2 (first non-tag row).
		dataL2 = dataL1;
		len_dataL2 = len_dataL1;
		pdataL2 = pdataL1;
		ntokensL2 = ntokensL1;

		// Number of lines processed.
		nlines = 1;

	} // File might have name and/or headers


	/////////////////////////////////


	// Detects if file might have row labels and sets ntokensL2 (or ntokensL2-1) as the number of columns.
	if ( (! isnumbers(dataL2)) + numeric_lbls ) {	// First token in L2 is not numeric, <OR> Input matrix has numeric row labels.

		// File contains row labels.
		printf("\t\tRow labels detected.\n");
		haslabels = true;

		// numcols = ntokensL2 - 1

		// Checks for invalid number of columns.
		if ( ( ntokensL2-1 <= 1 ) + ( ntokensL2-1 > (IDX_MAX/2) ) ) {
			printf("\t\tNumber of data columns detected (excluding row labels): %zu.\n",ntokensL2-1);
			fflush(stdout);
			if ( ntokensL2-1 <= 1 )
				fprintf( stderr, "\nError reading input file:\nInvalid file format or the number of columns is less than 2.\n"
						"Please remember that columns must be separated by TAB characters (or by single space "
						"characters under certain conditions).\nFinally, please check for any invalid decimal "
						"symbol (e.g., ',' instead of '.').\n\n");
			else
				fprintf( stderr, "\n\nSorry, but your matrix exceeds the limits used for matrix dimensions.\nData matrices "
						"are limited to:\n\t* %" PRI_IDX " rows\n\t*%" PRI_IDX " columns.\n\t* %" PRI_IDX
						" items.\n", IDX_MAX/2, IDX_MAX/2, IDX_MAX );
			free(pdataL1); free(dataL1);
			if ( dataL2 != dataL1 ) { free(pdataL2); free(dataL2); }
			fclose(file);
			return EXIT_FAILURE;
		}

		numcols = (index_t) (ntokensL2 - 1);

		// Uses dataL2 to store row labels (numeric data will be copied to 'data_matrix' later).
		max_len_labels = len_dataL2 + 1;	// Initial size of labels[].
		labels = dataL2;
		len_labels = strlen(dataL2) + 1;	// Length of first label.

		// Uses pdataL2 as plabels. Resizes pdataL2 or max_numrows if necessary.

		if ( (index_t) ntokensL2 < max_numrows ) {
			#if NMFGPU_DEBUG_READ_MATRIX
				printf("Resizing plabels from %zu to %" PRI_IDX "...\n",ntokensL2,max_numrows);
				fflush(stdout);
			#endif
			char **const tmp = (char **) realloc( pdataL2, max_numrows * sizeof(char *) );
			if ( ! tmp ) {
				int const err = errno; fflush(stdout); errno = err;
				perror("\nrealloc( pdataL2 )");
				fprintf(stderr,"Error in matrix_load_ascii_verb().\n");
				free(pdataL1); free(dataL1);
				if ( dataL2 != dataL1 ) { free(pdataL2); free(dataL2); }
				fclose(file);
				return EXIT_FAILURE;
			}
			pdataL2 = tmp;
			if ( dataL1 == dataL2 )
				pdataL1 = tmp;

		} else {

			#if NMFGPU_DEBUG_READ_MATRIX
				printf("Setting max_numrows from %" PRI_IDX " to %zu...\n",max_numrows,ntokensL2);
				fflush(stdout);
			#endif

			max_numrows = (index_t) ntokensL2;
		}

		plabels = pdataL2;
		p_pdataL2 = pdataL2 + 1;	// Numeric data will be copied from the second token.

		// Pointers to place for future second label
		p_plabels = p_pdataL2;	// plabels + 1
		p_labels = labels + len_labels;	// It should be equal to: *p_pdataL2

		#if NMFGPU_DEBUG_READ_MATRIX
			printf("p_labels=%p, (*p_pdataL2)=%p (they should be equal)\n",p_labels,*p_pdataL2);
			fflush(stdout);
		#endif

		printf("\t\tNumber of data columns detected (excluding row labels): %" PRI_IDX ".\n",numcols);

	} else {

		// No labels. numcols = ntokensL2
		haslabels = false;
		p_pdataL2 = pdataL2;

		// If data matrix seems to be numeric only.
		if ( ! has_name_headers )
			printf("\t\tNumeric data only.\n");

		// Checks for invalid number of columns.
		if ( ( ntokensL2 <= 1 ) + ( ntokensL2 > (IDX_MAX/2) ) ) {
			printf("\t\tNumber of data columns detected: %zu.\n",ntokensL2);
			fflush(stdout);
			if ( ntokensL2 <= 1 )
				fprintf( stderr, "\nError reading input file:\nInvalid file format or the number of columns is less than 2.\n"
						"Please remember that columns must be separated by TAB characters (or by single space "
						"characters under certain conditions).\nFinally, please check for any invalid decimal "
						"symbol (e.g., ',' instead of '.').\n\n");
			else
				fprintf( stderr, "\n\nSorry, but your matrix exceeds the limits used for matrix dimensions.\nData matrices "
						"are limited to:\n\t* %" PRI_IDX " rows\n\t*%" PRI_IDX " columns.\n\t* %" PRI_IDX
						" items.\n", IDX_MAX/2, IDX_MAX/2, IDX_MAX );
			free(pdataL1); free(dataL1);
			if ( dataL2 != dataL1 ) { free(pdataL2); free(dataL2); }
			fclose(file);
			return EXIT_FAILURE;
		}

		numcols = (index_t) ntokensL2;

		printf("\t\tNumber of data columns detected: %" PRI_IDX ".\n",numcols);

	} // If file contains row labels

	/////////////////////////////////

	// Compares length of L1 and numcols to definitely know if there are name/headers or not.

	// File might have name and/or headers
	if ( has_name_headers ) {

		// dataL1 != dataL2

		if ( (index_t)(ntokensL1 - 1) == numcols ) {	// Name and headers

			printf("\t\tName (i.e., description string) detected.\n");
			hasname = true;

			// Copies the first token.
			name = (char *restrict) malloc( (strlen(dataL1) + 1) * sizeof(char) );
			if ( ! name ) {
				int const err = errno; fflush(stdout); errno = err;
				perror("\nmalloc( dataL1 )");
				fprintf(stderr,"Error in matrix_load_ascii_verb().\n");
				free(pdataL2); free(dataL2); free(pdataL1); free(dataL1);
				fclose(file);
				return EXIT_FAILURE;
			}
			strcpy( name, dataL1 );

			#if NMFGPU_DEBUG_READ_MATRIX
				printf("Name (len=%zu):'%s'\n",strlen(name),name);
				fflush(stdout);
			#endif

			// Headers
			printf("\t\tColumn headers detected.\n");
			hasheaders = true;

			/* Moves the first header to the beginning of dataL1 in order to
			 * have <dataL1> (i.e., the address returned by malloc(3)) as the address
			 * of the first header, so that it is possible to free(3) "headers".
			 */
			char *p = pdataL1[1];
			headers = memmove( dataL1, p, (strlen(p) + 1) * sizeof(char) ); // Overwrites first token L1.
			if ( ! headers ) {
				int const err = errno; fflush(stdout); errno = err;
				perror("\nmemmove( dataL1 )");
				fprintf(stderr,"Error in matrix_load_ascii_verb().\n");
				free(name);
				free(pdataL2); free(dataL2); free(pdataL1); free(dataL1);
				fclose(file);
				return EXIT_FAILURE;
			}
			pdataL1[1] = dataL1;	// pdataL1[0] == pdataL1[1] == dataL1

			pheaders = (char **restrict) malloc( numcols * sizeof(char *) );
			if ( ! pheaders ) {
				int const err = errno; fflush(stdout); errno = err;
				perror("\nmalloc( pheaders )");
				fprintf(stderr,"Error in matrix_load_ascii_verb().\n");
				free(name);
				free(pdataL2); free(dataL2); free(pdataL1); free(dataL1);
				fclose(file);
				return EXIT_FAILURE;
			}

			// Copies pdataL1 to pheaders.
			memcpy( pheaders, &pdataL1[1], numcols * sizeof(char *) );	// pheaders[i] = pdataL1[i+1]
			free(pdataL1);
			pdataL1 = pheaders;

		} else if ( (index_t) ntokensL1 == numcols ) {	// No name. Headers only.

			printf("\t\tColumn headers detected.\n");
			hasheaders = true;
			headers = dataL1;
			pheaders = pdataL1;

		} else if ( ntokensL1 == 1 ) {	// No headers, name only

			printf("\t\tName (i.e., description string) detected.\n");
			hasname = true;
			name = dataL1;

		} else {	// Error.
			fflush(stdout);
			fprintf(stderr, "\nError reading input file:\nLength of lines 1 (%zu) and 2 (%" PRI_IDX ") mismatch.\n"
					"Invalid file format or data is not separated.\nFinally, please check for any invalid "
					"decimal symbol.\n\n", ntokensL1, numcols );
			free(dataL2); free(pdataL2); free(pdataL1); free(dataL1);
			fclose(file);
			return EXIT_FAILURE;
		} // If there are name and/or headers

	} // If there can be name or headers

	/////////////////////////////////

	// Sets (the rest of) L2 as the first row of data matrix.

	data_matrix = (real *restrict) malloc( max_numrows * numcols * sizeof(real) );
	if ( ! data_matrix ) {
		int const err = errno; fflush(stdout); errno = err;
		perror("\nmalloc( data_matrix ): ");
		fprintf(stderr,"Error in matrix_load_ascii_verb().\n");
		if ( hasname * hasheaders ) free(name);
		free(pdataL1); free(dataL1);
		if ( dataL2 != dataL1 ) { free(pdataL2); free(dataL2); }
		fclose(file);
		return EXIT_FAILURE;
	}

	#if NMFGPU_DEBUG_READ_MATRIX
		printf("Setting first data line (file line %" PRI_IDX ")...\n",nlines);
		fflush(stdout);
	#endif


	pmatrix = data_matrix;
	index_t i;
	for ( i=0; i<numcols; i++,pmatrix++,p_pdataL2++ ) {

		// Token (data) to read.
		char *const val_str = *p_pdataL2;
		char *endptr = NULL;

		#if NMFGPU_DEBUG_READ_MATRIX2
			printf("\tstr='%s'",val_str);
			fflush(stdout);
		#endif

		// Transforms char* to real.
		real const value = STRTOREAL( val_str, &endptr ) ;

		#if NMFGPU_DEBUG_READ_MATRIX2
		{
			printf("val=%g,endptr=",value);
			char const c = *endptr;
			if ( isprint(c) + isblank(c) )
				printf("'%c'\n",c);
			else
				printf("'\\x%X'\n",c);
			fflush(stdout);
		}
		#endif

		// No numeric characters <OR> NaN, inf, underflow/overflow, etc
		if ( !is_valid(value) + (*endptr) ) {	// (*endptr != '\0') || (! is_valid(value))
			fflush(stdout);
			fprintf( stderr, "\nError reading line %" PRI_IDX ", column %" PRI_IDX ". Invalid numeric format: '%s'.\n"
					"Please, check also for invalid decimal symbol (if any).\n\n", nlines, (i + hasheaders + 1), val_str );
			free( data_matrix );
			if ( hasname * hasheaders ) free(name);
			free(pdataL1); free(dataL1);
			if ( dataL2 != dataL1 ) { free(pdataL2); free(dataL2); }
			fclose(file);
			return EXIT_FAILURE;
		}

		// Stores new value.
		*pmatrix = value;

	} // for

	// One matrix row read.
	numrows = 1;

	/////////////////////////////////

	if ( name == dataL1 )	// hasname && (! hasheaders)
		free(pdataL1);

	/////////////////////////////////

	// Reads the rest of matrix data (and possibly labels)

	// Reads the next character.
	int c = fgetc(file);
	if ( c != EOF ) {
		#if NMFGPU_DEBUG_READ_MATRIX
			printf("Line %" PRI_IDX "+1: char=",nlines);
			if ( c == (int) '\n' ) printf("'\\n'.\n");
			else if ( c == (int) '\r' ) printf("'\\r'.\n");
			else if ( isprint(c) + isblank(c) ) printf("'%c'.\n",c);
			else printf("'\\x%X'.\n",c);
			fflush(stdout);
		#endif
		ungetc( c, file);
	}


	// Fails on (premature) End-Of-File (EOF) or error.
	if ( (c == (int) '\r') + (c == (int) '\n') + feof(file) + ferror(file) ) {
		fflush(stdout);
		if ( ferror(file) )
			fprintf(stderr,"\nInternal error in fgetc().\nError in matrix_load_ascii_verb().\n");
		else if ( ( c == (int) '\r') + (c == (int) '\n') )	// EOL
			fprintf(stderr,"\nError reading input file: unexpected end of line.\n"
				"The number of rows is less than 2 or invalid file format.\n\n" );
		else	// EOF
			fprintf(stderr,"\nError reading input file: unexpected end of file.\n"
				"The number of rows is less than 2 or invalid file format.\n\n" );
		struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
		matrix_clean( data_matrix, l_ml );
		return EXIT_FAILURE;
	} // if ! EOF

	// Formats strings to be used with fscanf().
	char format_delimiter[8];		// Format string for delimiter/missing data.
	sprintf( format_delimiter, "%%1[%c\n\r]", delimiter );
	char format_data[16];			// Format string for numeric data: data + delimiter
	sprintf( format_data, "%%" SCNgREAL "%s", format_delimiter );

	#if NMFGPU_DEBUG_READ_MATRIX
		printf("Format delimiter: '%s' (len=%zu)\nFormat data: '%s' (len=%zu)\n",format_delimiter,
			strlen(format_delimiter),format_data,strlen(format_data));
		fflush(stdout);
	#endif

	index_t nitems = numcols;		// Current number of data elements.


	do {

		nlines++;	// A new line will be read.

		#if NMFGPU_DEBUG_READ_MATRIX2
			printf("Reading line %" PRI_IDX "...\n",nlines);
			fflush(stdout);
		#endif

		/////////////////////////////////////////

		// Checks for overflow.
		if ( ((uintmax_t)nitems + (uintmax_t)numcols) > IDX_MAX ) {
			printf("\t\tNumber of matrix rows currently read: %" PRI_IDX ".\n"
				"\t\tNumber of matrix entries currently read: %" PRI_IDX ".\n", numrows, nitems );
			fflush(stdout);
			fprintf(stderr, "\n\nSorry, but your matrix exceeds the limits used for matrix dimensions.\n"
				"Data matrices are limited to:\n\t* %" PRI_IDX " rows.\n\t* %" PRI_IDX
				" columns.\n\t* %" PRI_IDX " items.\n", IDX_MAX/2,IDX_MAX/2,IDX_MAX);
			struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
			matrix_clean( data_matrix, l_ml );
			fclose(file);
			return EXIT_FAILURE;
		} // if overflows.

		/////////////////////////////////////////

		if ( haslabels ) {

			// Reads a token.
			char *restrict data = NULL;
			int last_char = (int) '\0';
			size_t const len_data = read_token( file, delimiter, &data, &last_char );
			if ( ! data ) {
				if ( len_data ) {
					fprintf(stderr,"Error in matrix_load_ascii_verb().\n");
					struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
					matrix_clean( data_matrix, l_ml );
					fclose(file);
					return EXIT_FAILURE;
				}
				// Else, EOF...
				break;
			}
			#if NMFGPU_DEBUG_READ_MATRIX
				printf("Line %" PRI_IDX ": Label(len=%zu): '%s'. Last_char=",nlines,len_data,data);
				if ( last_char == (int) '\n' ) printf("'\\n'.\n");
				else if ( last_char == (int) '\r' ) printf("'\\r'.\n");
				else if ( isprint(last_char) + isblank(last_char) ) printf("'%c'.\n",last_char);
				else printf("'\\x%X'.\n",last_char);
				fflush(stdout);
			#endif
			if ( last_char != delimiter ) { // Blank line.
				free(data);

				// Checks for more blank lines.
				c = last_char;
				index_t const nlines0 = nlines;
				while ( ( c == (int) '\r' ) + ( c == (int) '\n' ) ) {
					if ( c == (int) '\r' )
						fgetc(file);	// Skips, also, the LF character.
					c = fgetc(file);
					nlines++;
				}

				// Just one or more blank lines at the end of the file.
				if ( feof(file) )
					break;

				// Fails on error or if there are more non-blank lines to read.
				fflush(stdout);
				if ( ferror(file) )
					fprintf(stderr,"\nInternal error in fgetc().\nError in matrix_load_ascii_verb().\n");
				else if ( nlines0 < nlines )
					fprintf(stderr,"\nError reading input file: No matrix data between lines %" PRI_IDX " and %"
							PRI_IDX ".\nInvalid file format.\n\n", nlines0, nlines);
				else
					fprintf(stderr,"\nError reading input file: No matrix data in line %" PRI_IDX
							"\nInvalid file format.\n\n", nlines0 );
				struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
				matrix_clean( data_matrix, l_ml );
				fclose(file);
				return EXIT_FAILURE;
			}

			// Before setting the new label, checks if there is enough memory available.
			size_t const len = len_labels + len_data + 1;
			if ( len > max_len_labels ) {	// Allocates more memory
				do {
					max_len_labels *= 2;
				} while ( len >= max_len_labels );

				#if NMFGPU_DEBUG_READ_MATRIX
					printf("Line %" PRI_IDX ": Expanding size of labels to %zu chars.\n",nlines,max_len_labels);
					fflush(stdout);
				#endif

				char *const tmp = (char *) realloc( labels, max_len_labels * sizeof(char) );
				if ( ! tmp ) {
					int const err = errno; fflush(stdout); errno = err;
					perror("\nrealloc( labels )");
					fprintf(stderr,"Error in matrix_load_ascii_verb().\n");
					free(data);
					struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
					matrix_clean( data_matrix, l_ml );
					fclose(file);
					return EXIT_FAILURE;
				}
				labels = tmp;
				p_labels = tmp + len_labels; // Pointer to place for new label.

				#if NMFGPU_DEBUG_READ_MATRIX
					printf("Line %" PRI_IDX ": Retokenizing labels (numrows=%" PRI_IDX ")...\n",
						nlines,numrows);
					fflush(stdout);
				#endif

				// Resets 'plabels'.
				retok( (char const *)labels, (char const **)plabels, numrows );
				p_plabels = plabels + numrows;

			} // If allocate more memory for labels

			// Before setting plabels[], checks if there is enough memory available.
			if ( numrows == max_numrows ) {

				// NOTE: 'max_numrows' will be incremented when allocating more memory for data_matrix[]

				#if NMFGPU_DEBUG_READ_MATRIX
					printf("Line %" PRI_IDX ": Expanding size of plabels to %" PRI_IDX " tokens.\n",
						nlines,2*max_numrows);
					fflush(stdout);
				#endif

				// plabels
				char **const tmp = (char **) realloc( plabels, (2 * max_numrows) * sizeof(char *) );
				if ( ! tmp ) {
					int const err = errno; fflush(stdout); errno = err;
					perror("\nrealloc( plabels )");
					fprintf(stderr,"Error in matrix_load_ascii_verb().\n");
					free(data);
					struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
					matrix_clean( data_matrix, l_ml );
					fclose(file);
					return EXIT_FAILURE;
				}
				plabels = tmp;
				p_plabels = tmp + numrows;

			} // If allocate more memory for plabels

			// Sets vector of pointers to labels
			strcpy(p_labels, data);	// Copies from 'data' the new token.
			*p_plabels = p_labels;	// Sets plabels[numrows].

			p_labels += (len_data + 1);	// pointer to place for next label
			p_plabels++;
			len_labels += (len_data + 1);

			free(data);

		}  // if has row labels

		// Before reading data_matrix checks if there is enough memory available.
		if ( numrows == max_numrows ) {

			// Allocates more memory

			max_numrows *= 2;

			#if NMFGPU_DEBUG_READ_MATRIX
				printf("Line %" PRI_IDX ": Expanding size of data_matrix to %" PRI_IDX " rows.\n",
					nlines, max_numrows);
				fflush(stdout);
			#endif

			// data_matrix
			real *const matrix_tmp = (real *) realloc( data_matrix, max_numrows * numcols * sizeof(real) );
			if ( ! matrix_tmp )  {
				int const err = errno; fflush(stdout); errno = err;
				perror("\nrealloc( data_matrix )");
				fprintf(stderr,"Error in matrix_load_ascii_verb().\n");
				struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
				matrix_clean( data_matrix, l_ml );
				fclose(file);
				return EXIT_FAILURE;
			}
			data_matrix = matrix_tmp;
			pmatrix = matrix_tmp + (numcols * numrows);

		} // If allocate more memory for data_matrix

		#if NMFGPU_DEBUG_READ_MATRIX
			printf("Line %" PRI_IDX ": Reading data_matrix...\n",nlines);
			fflush(stdout);
		#endif

		// Reads the (rest of the) line as data_matrix[numrows][...]
		for ( i=0; i<(numcols-1); i++,pmatrix++ ) {

			real value = REAL_C( 0.0 );

			// If it reads a delimiter, then it is a missing data ('value' is set to 0).
			c = 0;
			int conv = fscanf( file, format_delimiter, &c );
			int max_conv = 1;	// Maximum items to be converted from string.

			#if NMFGPU_DEBUG_READ_MATRIX2
				printf("\tconv1=%i",conv);
			#endif

			if ( ! conv ) { // No missing data: reads current value (data + new_delimiter).
				c = 0;
				conv = fscanf( file, format_data, &value, &c );
				max_conv = 2;

				#if NMFGPU_DEBUG_READ_MATRIX2
					printf(",conv2=%i,c2=",conv);	// conv: number of tokens read.
					if (! c) printf("(empty)");
					else if ( isprint(c) + isblank(c) )	// c: character set as delimiter
						printf("'%c'",c);
					else
						printf("'\\x%X'",c);
				#endif
			}
			#if NMFGPU_DEBUG_READ_MATRIX2
				printf(",value=%g ",value);
				fflush(stdout);
			#endif

			// Fails if premature EOF, premature EOL, or an invalid number (NaN, inf,...).
			if ( (conv < max_conv) + (c != delimiter) + (! is_valid(value)) ) {
				int const err = errno; fflush(stdout); errno = err;
				if ( errno + ferror(file) ) {	// Error.
					if ( errno ) { perror("\nfscanf(value)"); }
					fprintf( stderr, "Error in matrix_load_ascii_verb() reading line %" PRI_IDX ", matrix column %"
							PRI_IDX "\n", nlines, i);
				}
				else if ( (c == (int) '\n') + (c == (int) '\r') )	// Premature EOL.
					fprintf(stderr, "\nError reading input file:\nPremature end-of-line detected in line %" PRI_IDX
							" (%" PRI_IDX " columns found, %" PRI_IDX " expected).\n"
							"Invalid file format.\n\n", nlines,i+haslabels+1,numcols+haslabels);
				else if ( (conv == EOF) + feof(file) ) // Premature EOF.
					fprintf(stderr, "\nError reading input file:\nPremature end-of-file detected in line %" PRI_IDX
							" (%" PRI_IDX " columns found, %" PRI_IDX " expected).\n"
							"Invalid file format.\n\n", nlines,i+haslabels,numcols+haslabels);
				else if ( ! is_valid(value) )	// Not a number
					fprintf(stderr, "\nError reading input file:\nLine %" PRI_IDX ", column %" PRI_IDX
							": Invalid numeric data: %g\n\n", nlines,i+haslabels+1,value);
				else { // Illegal character.
					fprintf(stderr,"\nError reading input file:\nLine %" PRI_IDX ", column %" PRI_IDX
							": Invalid character", nlines,i+haslabels+1);
					c = fgetc( file );	// Reads illegal character
					if ( c != EOF ) {
						if ( isprint(c) ) fprintf(stderr, ": '%c' ('\\x%X')",c,c);
						else fprintf(stderr, ": '\\x%X'",c);
					}
					fprintf(stderr,".\n\n");
				}
				struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
				matrix_clean( data_matrix, l_ml );
				fclose(file);
				return EXIT_FAILURE;
			}

			// Stores new value.
			*pmatrix = value;

		} // for 0<= i < numcols-1

		// Last column: i == numcols - 1
		{

			real value = 0;

			// If it reads an EOL (CR or LF), then it is a missing data (value is set to 0).
			c = (int) '\n';
			int conv = fscanf( file, format_delimiter, &c );

			#if NMFGPU_DEBUG_READ_MATRIX2
				printf(" ... conv1=%i",conv);
				if ( c != (int) '\n' ) {
					printf(",c1=");
					if ( c == (int) '\r' ) printf("'\\r'");
					else if ( isprint(c) + isblank(c) ) printf("'%c'",c);
					else printf("'\\x%X'",c);
				}
				fflush(stdout);
			#endif

			if ( (c == delimiter) + ferror(file) + feof(file) ) {
				int const err = errno; fflush(stdout); errno = err;
				if ( c == delimiter )	// More than numcols columns.
					fprintf(stderr, "\nError reading input file:\nLine %" PRI_IDX ": Invalid file format.\n"
							"There are more than %" PRI_IDX " data columns.\n\n", nlines, numcols);
				else if ( (conv == EOF) + feof(file) ) // Premature EOF.
					fprintf(stderr, "\nError reading input file:\nPremature end-of-file detected in line %" PRI_IDX
							" (%" PRI_IDX " columns found, %" PRI_IDX " expected).\n"
							"Invalid file format.\n\n", nlines,i+haslabels,numcols+haslabels);
				else { // Error.
					if ( errno ) { perror("\nfscanf()"); }
					fprintf( stderr, "Error in matrix_load_ascii_verb() reading line %" PRI_IDX ", matrix column %"
							PRI_IDX "\n", nlines, i);
				}
				struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
				matrix_clean( data_matrix, l_ml );
				fclose(file);
				return EXIT_FAILURE;
			}

			// No missing data: reads last value (plus ['\r' or '\n']).
			if ( ! conv ) {
				c = 0;
				conv = fscanf( file, format_data, &value, &c );

				#if NMFGPU_DEBUG_READ_MATRIX2
					printf(",conv2=%i,c2=",conv);
					if ( c == (int) '\n' ) printf("'\\n'");
					else if ( c == (int) '\r' ) printf("'\\r'");
					else if ( isprint(c) + isblank(c) ) printf("'%c'",c);
					else if ( !c ) printf("(empty)");
					else printf("'\\x%X'",c);
					printf(",value=%g",value);
					fflush(stdout);
				#endif

				// Fails on invalid format or error (conv=EOF), more columns, invalid characters, or invalid value.
				if ( (conv <= 0) + (c == delimiter) + (!(c + feof(file))) + (! is_valid(value)) ) {
					int const err = errno; fflush(stdout); errno = err;
					if ( errno + ferror(file) ) {	// Error.
						if ( errno ) { perror("\nfscanf()"); }
						fprintf( stderr, "Error in matrix_load_ascii_verb() reading line %" PRI_IDX
								", matrix column %" PRI_IDX ".\n\n", nlines, i);
					}
					if ( c == delimiter )	// More than numcols columns.
						fprintf(stderr, "\nError reading input file:\nLine %" PRI_IDX ": There are more than %"
								PRI_IDX " data columns or line finishes with a '%c'.\n\n", nlines,
								numcols, c);
					else if ( ! is_valid(value) )	// Not a number
						fprintf(stderr, "\nError reading input file:\nLine %" PRI_IDX ", column %" PRI_IDX
								": Invalid numeric data: %g\n\n", nlines,i+haslabels+1,value);
					else {	// Illegal character.
						fprintf(stderr, "\nError reading input file:\nLine %" PRI_IDX ", column %" PRI_IDX
								": Invalid character", nlines,i+haslabels+1);
						c = fgetc( file );	// Reads illegal character
						if ( c != EOF ) {
							if ( isprint(c) ) fprintf(stderr, ": '%c' ('\\x%X')",c,c);
							else fprintf(stderr, ": '\\x%X'",c);
						}
						fprintf(stderr,".\n\n");
					}
					struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
					matrix_clean( data_matrix, l_ml );
					fclose(file);
					return EXIT_FAILURE;
				}

			} // If last data was read.
			#if NMFGPU_DEBUG_READ_MATRIX2
			// Missing data.
			else { printf(",value=(emtpy)"); fflush(stdout); }
			#endif

			// If the last character read was a CR ('\r'), reads the corresponding LF character
			if ( c == (int) '\r' ) {

				c = fgetc(file);

				#if NMFGPU_DEBUG_READ_MATRIX2
					if ( c == (int) '\n' ) printf(",c3='\\n'");
					else if ( c == (int) '\r' ) printf(",c3='\\r'");
					else if ( isprint(c) + isblank(c) ) printf(",c3='%c'",c);
					else printf(",c3='\\x%X'",c);
					fflush(stdout);
				#endif

				// Fails if it is not a LF character (assumes this is not an old OS).
				if ( ferror(file) + ( c != (int) '\n' ) ) {
					fflush(stdout);
					if ( c != (int) '\n' ) {
						fprintf(stderr,"\nLine %" PRI_IDX ": Unexpected character after a CR ('\\r'): ",nlines);
						if ( isprint(c) + isblank(c) ) fprintf(stderr,"'%c'.\n\n",c);
						else fprintf(stderr,"'\\x%X'.\n\n",c);
					}
					else
						fprintf(stderr,"\nLine %" PRI_IDX ": Internal error in fgetc(\\n).\n"
							"Error in matrix_load_ascii_verb().\n", nlines);
					struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
					matrix_clean( data_matrix, l_ml );
					fclose(file);
					return EXIT_FAILURE;
				}

			} // if last character read was a CR ('\r').

			#if NMFGPU_DEBUG_READ_MATRIX2
				printf("\n"); fflush(stdout);
			#endif

			// Stores the new value.
			*pmatrix = value;
			pmatrix++;

		} // last column.

		numrows++;
		nitems += numcols;

		// Checks for EOF by reading one more character.
		bool error = false;
		c = fgetc(file);
		if ( c != EOF ) {
			#if NMFGPU_DEBUG_READ_MATRIX
				printf("Line %" PRI_IDX "+1: char=",nlines);
				if ( c == (int) '\n' ) printf("'\\n'.\n");
				else if ( c == (int) '\r' ) printf("'\\r'.\n");
				else if ( isprint(c) + isblank(c) ) printf("'%c'.\n",c);
				else printf("'\\x%X'.\n",c);
				fflush(stdout);
			#endif

			// Checks for blank lines.
			if ( ( c == (int) '\r' ) + ( c == (int) '\n' ) ) {	// ( c == (int) '\r' ) || ( c == (int) '\n' )
				index_t nlines0 = nlines + 1;
				do {
					if ( c == (int) '\r' )
						fgetc(file);	// Skips the corresponding LF
					c = fgetc(file);
					nlines++;
				} while ( ( c == (int) '\r' ) + ( c == (int) '\n' ) );

				if ( ferror(file) ) {
					fflush(stdout);
					fprintf(stderr,"\nInternal error in fgetc().\nError in matrix_load_ascii_verb().\n");
					error = 1;
				}

				// No more lines, stops.
				if ( ( feof(file) ) + ( c == EOF ) )
					break;
				else {	// There are more lines. Invalid format
					fflush(stdout);
					if ( nlines0 < nlines )
						fprintf(stderr,"\nError reading input file: No matrix data between lines %" PRI_IDX
								" and %" PRI_IDX ".\nInvalid file format.\n\n", nlines0, nlines);
					else
						fprintf(stderr,"\nError reading input file: No matrix data in line %" PRI_IDX "\n"
							"Invalid file format.\n\n", nlines0 );
					error = true;
				}
			}

			// There are more lines. Restores the last character read.
			ungetc( c, file);

		} // if ( c != EOF )
		else if ( ferror(file) ) {
			fflush(stdout);
			fprintf(stderr,"\nInternal error in fgetc().\nError in matrix_load_ascii_verb().\n");
			error = true;
		}

		if ( error ) {
			struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
			matrix_clean( data_matrix, l_ml );
			fclose(file);
			return EXIT_FAILURE;
		}

	} while ( c != EOF );

	fclose(file);

	printf("\t\tLoaded a %" PRI_IDX " x %" PRI_IDX " data matrix (%" PRI_IDX " items).\n",numrows,numcols,nitems);

	#if NMFGPU_DEBUG_READ_MATRIX
		fflush(stdout);
	#endif

	// Adjusts allocated memory for labels and data matrix.
	if ( haslabels ) {
		#if NMFGPU_DEBUG_READ_MATRIX
			printf("Resizing labels from %zu to %zu, and plabels from %" PRI_IDX " to %" PRI_IDX "\n",
			       max_len_labels, len_labels,  max_numrows, numrows);
			fflush(stdout);
		#endif
		labels = (char *) realloc( labels, len_labels * sizeof(char) );
		plabels = (char **) realloc( plabels, numrows * sizeof(char *) );
	}
	data_matrix = (real *) realloc( data_matrix, nitems * sizeof(real) );

	#if NMFGPU_DEBUG_READ_MATRIX
		printf("Load matrix finished!\n");
		fflush(stdout);
	#endif

	// Sets output parameters.
	*matrix = data_matrix;
	*nrows = numrows;
	*ncols = numcols;
	*ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );

	return EXIT_SUCCESS;

} // matrix_load_ascii_verb

//////////////////////////////////////////////////

/*
 * Loads a matrix from a (non-"native") binary file (i.e., double-precision data and unsigned int's).
 *
 * Detects automatically if matrix has name, column headers and/or row labels,
 * as well as the used tag delimiter (space or tab character). Skips if 'ml' is NULL.
 * Outputs information messages.
 * Performs error checking.
 *
 * Both matrix dimensions must be >= 2.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_load_binary_verb( char const *restrict filename, real *restrict *restrict matrix, index_t *restrict nrows, index_t *restrict ncols,
				struct matrix_labels *restrict ml )
{

	// Local values and pointers to output parameters.
	index_t numcols = 0, numrows = 0;

	// Name, headers and labels
	char *name = NULL; char *headers = NULL; char *labels = NULL;

	char **pheaders = NULL, **plabels = NULL;	// Array of pointers to headers and labels

	real *restrict data_matrix = NULL;

	/////////////////////////////////

	// Checks for NULL parameters
	if ( ! ( (size_t) matrix * (size_t) nrows * (size_t) ncols * (size_t) ml ) ) {
		fflush( stdout );
		errno = EFAULT;
		if ( ! matrix )	perror("\nmatrix_load_binary_verb( matrix )");
		if ( ! nrows )	perror("\nmatrix_load_binary_verb( nrows )");
		if ( ! ncols )	perror("\nmatrix_load_binary_verb( ncols )");
		if ( ! ml )	perror("\nmatrix_load_binary_verb( ml )");
		return EXIT_FAILURE;
	}

	FILE *restrict const file = fopen( filename, "rb" );
	if( ! file ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nfopen '%s' :\n\t%s.\n", filename, strerror(errno) );
		fprintf( stderr, "Error in matrix_load_binary_verb().\n" );
		return EXIT_FAILURE;
	}

	// Starts reading...

	// Matrix dimensions.
	unsigned int dim[2];
	{
		size_t const nread = fread( dim, sizeof(int), 2, file );
		if ( nread != 2 ) {
			fflush(stdout);
			if ( feof(file) )
				fprintf(stderr, "\nError reading input file:\nPremature end-of-file detected. Invalid file format.\n\n");
			else // error
				fprintf(stderr, "\nInternal error in fread(dim).\nError in matrix_load_binary_verb().\n");
			fclose(file);
			return EXIT_FAILURE;
		}

		uintmax_t const nitems = ((uintmax_t) dim[0]) * ((uintmax_t) dim[1]);
		printf( "\t\tSize of input matrix to read: %u x %u (%" PRIuMAX " items)\n", dim[0], dim[1], nitems );

		// Checks for invalid values.
		if ( ( dim[0] < 2 ) + ( dim[1] < 2 ) + ( nitems > IDX_MAX ) ) {
			fflush(stdout);
			if ( nitems > IDX_MAX )
				fprintf(stderr, "\n\nSorry, but your matrix exceeds the limits used for matrix dimensions.\n"
						"Data matrices are limited to:\n\t* %" PRI_IDX " rows.\n\t* %" PRI_IDX " columns.\n\t* %"
						PRI_IDX " items.\n", IDX_MAX/2, IDX_MAX/2, IDX_MAX);
			else
				fprintf(stderr, "\nError reading input file:\nBoth matrix dimensions must be greater than 1.\n\n");
			fclose(file);
			return EXIT_FAILURE;
		} // if overflows
	}

	// Changes values to index_t
	numrows = (index_t) dim[0];
	numcols = (index_t) dim[1];

	/////////////////////////////////

	// Reads data matrix
	data_matrix = (real *restrict) malloc( numrows * numcols * sizeof(real) );
	if ( ! data_matrix ) {
		int const err = errno; fflush(stdout); errno = err;
		perror( "\nmalloc( data_matrix )" );
		fprintf(stderr,"Error in matrix_load_binary_verb().\n");
		fclose(file);
		return EXIT_FAILURE;
	}


	// In addition, checks for NaN values.

	#if NMFGPU_DEBUG_READ_MATRIX2
		printf("\n");
		fflush(stdout);
	#endif

	real *pmatrix = data_matrix;
	for ( index_t i = 0 ; i < numrows ; i++ ) {

		for ( index_t j = 0 ; j < numcols ; j++, pmatrix++ ) {

			// Reads current data value.
			double value = 0;
			size_t const nread = fread( &value, sizeof(double), 1, file );	// Reads one double-precision value.
			real const num = (real) value;

			#if NMFGPU_DEBUG_READ_MATRIX2
				printf("%g ",value);
				fflush(stdout);
			#endif

			// Checks data.
			if ( ! ( nread * is_valid(num) ) ) {
				fflush(stdout);
				if ( feof(file) )
					fprintf(stderr,"\nError reading input file:\nPremature end-of-file detected.\n"
							"Invalid file format.\n\n");
				else { // error
					fprintf( stderr, "\nError reading input file:\nError reading row %" PRI_IDX ", column %" PRI_IDX,
						i, j );
					if ( (! nread) + ferror(file) )	// Error in file
						fprintf( stderr, ".\nInvalid file format.\n\n" );
					else	// Invalid numeric format
						fprintf( stderr, ": '%g'.\nInvalid numeric format.\n\n", value );
				}
				free( data_matrix );
				fclose(file);
				return EXIT_FAILURE;
			}

			// Stores the new value.
			*pmatrix = num;

		} // for j.

		#if NMFGPU_DEBUG_READ_MATRIX2
			printf("\n");
			fflush(stdout);
		#endif

	} // for i

	// Sets output parameters.
	*matrix = data_matrix;
	*nrows = numrows;
	*ncols = numcols;

	/////////////////////////////////

	// Reads labels, headers and name (as plain text) if they exists.

	char *restrict data = NULL;
	size_t len_data = 0;
	bool haslabels = false;

	// Checks for row labels
	len_data = read_line( file, &data );
	if ( ! data ) {
		fclose(file);
		#if NMFGPU_DEBUG_READ_MATRIX
			printf("No Labels (null).\n"); fflush(stdout);
		#endif
		if ( len_data ) {
			fprintf(stderr,"Error in matrix_load_binary_verb().\n\n");
			fflush(stdout);
			free( data_matrix );
			return EXIT_FAILURE;
		}
		*ml = NEW_MATRIX_LABELS( NULL, NULL, NULL, NULL, NULL );
		return EXIT_SUCCESS;
	}

	if ( len_data >= 3 ) {	// minimum length for two labels.

		char **restrict pdata = NULL;

		// Divides into tokens by replacing all tabs characters by '\0'.
		size_t ntokens = tokenize( data, &pdata, (int) '\t' );

		// If there is only one token, "retokenizes" using space as delimiter.
		if ( ntokens == 1 ) {
			#if NMFGPU_DEBUG_READ_MATRIX
				printf("ntokens Labels=1\n"); fflush(stdout);
			#endif
			free(pdata);
			pdata = NULL;
			ntokens = tokenize( data, &pdata, (int) ' ' );
		} // if ntokens == 1

		#if NMFGPU_DEBUG_READ_MATRIX
			printf("ntokens Labels=%zu\n",ntokens);
			fflush(stdout);
		#endif

		if ( ntokens != (size_t) numrows ) {
			fflush(stdout);
			if ( ntokens ) {
				printf("\t\tRow labels detected.\n");
				fprintf( stderr, "\nError reading input file. Invalid format:\nNumber of row "
						"labels (%zu) and number of matrix rows (%" PRI_IDX ") mismatch.\n", ntokens, numrows );
				if ( ntokens > (size_t) numrows )
					fprintf( stderr, "Please remember to set a '\\n' (new-line) character "
							"between column labels, row labels and the description string.\n\n");
				free(pdata);
			}
			else	// Error.
				fprintf(stderr,"Error in matrix_load_binary_verb().\n\n");
			free(data);
			free( data_matrix );
			fclose(file);
			return EXIT_FAILURE;
		}

		// File contains row labels.
		haslabels = true;
		printf("\t\tRow labels detected.\n");

		labels = data;
		plabels = pdata;

	} else { // No labels.
		#if NMFGPU_DEBUG_READ_MATRIX
			printf("'No Labels' (len=%zu).\n",len_data); fflush(stdout);
		#endif
		free(data);
	}
	data = NULL;


	// Checks for columns headers
	len_data = read_line( file, &data );
	if ( ! data ) {
		fclose(file);
		#if NMFGPU_DEBUG_READ_MATRIX
			printf("No Headers (null).\n"); fflush(stdout);
		#endif
		if ( len_data ) {
			fprintf(stderr,"Error in matrix_load_binary_verb().\n");
			if ( haslabels ) { free(labels); free(plabels); }
			free( data_matrix );
			return EXIT_FAILURE;
		}
		*ml = NEW_MATRIX_LABELS( NULL, NULL, NULL, labels, plabels );
		return EXIT_SUCCESS;
	}

	if ( len_data >= 3 ) { // minimum length for two headers.

		char **restrict pdata = NULL;

		// Divides into tokens by replacing all tabs characters by '\0'.
		size_t ntokens = tokenize( data, &pdata, (int) '\t' );

		// If there is only one token, retokenizes using space as delimiter
		if ( ntokens == 1 ) {
			#if NMFGPU_DEBUG_READ_MATRIX
				printf("ntokens Headers=1\n"); fflush(stdout);
			#endif
			free(pdata);
			pdata = NULL;
			ntokens = tokenize( data, &pdata, (int) ' ' );
		} // if ntokens == 1

		#if NMFGPU_DEBUG_READ_MATRIX
			printf("ntokens Headers=%zu\n",ntokens);
			fflush(stdout);
		#endif

		if ( ntokens != (size_t) numcols ) {
			fflush(stdout);
			if ( ntokens ) {
				printf("\t\tColumn headers detected.\n");
				fflush(stdout);
				fprintf( stderr,"\nError reading input file. Invalid format:\nNumber of column "
						"labels (%zu) and number of matrix columns (%" PRI_IDX ") mismatch.\n", ntokens,
						numcols );
				if ( ntokens > (size_t) numcols )
					fprintf( stderr, "Please remember to set a '\\n' (new-line) character "
							"between headers labels, row labels and the description string.\n\n");
				free(pdata);
			}
			else
				fprintf(stderr,"Error in matrix_load_binary_verb().\n");
			free(data);
			free( data_matrix );
			if ( haslabels ) { free(labels); free(plabels); }
			fclose(file);
			return EXIT_FAILURE;
		}

		printf("\t\tColumn headers detected.\n");

		headers = data;
		pheaders = pdata;

	} else {	// No headers
		#if NMFGPU_DEBUG_READ_MATRIX
			printf("'No Headers' (len=%zu).\n",len_data); fflush(stdout);
		#endif
		free(data);
	}
	data = NULL;


	// Checks for name.
	len_data = read_token( file, (int) '\t', &data, NULL );

	#if NMFGPU_DEBUG_READ_MATRIX
		printf("Name (len=%zu):\n",len_data);
		fflush(stdout);
	#endif

	fclose(file);

	if ( ! data ) {
		struct matrix_labels l_ml = NEW_MATRIX_LABELS( NULL, headers, pheaders, labels, plabels );
		if ( len_data ) {
			fflush(stdout);
			fprintf(stderr,"Error in matrix_load_binary_verb().\n");
			matrix_clean( data_matrix, l_ml );
			return EXIT_FAILURE;
		}
		*ml = l_ml;
		return EXIT_SUCCESS;
	}

	#if NMFGPU_DEBUG_READ_MATRIX
		printf("\t'%s'.\n",data);
		fflush(stdout);
	#endif

	printf("\t\tName (i.e., description string) detected.\n");
	name = data;

	// Sets output parameters.
	*ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );

	return EXIT_SUCCESS;

} // matrix_load_binary_verb

//////////////////////////////////////////////////
//////////////////////////////////////////////////

/*
 * Loads a matrix from an ASCII file.
 *
 * Skips name, headers and labels if 'ml' is set to NULL.
 *
 * If (*matrix) is not NULL, do not allocates memory but uses the supplied one.
 * WARNING: In such case, THERE MUST BE ENOUGH MEMORY ALREADY ALLOCATED.
 *
 * If input file does not have any tag, accepts both tab and space characters as delimiters.
 *
 * If 'transpose' is 'true', transposes matrix in memory as follows:
 * - Matrix dimensions in memory: <ncols> rows, <nrows> columns.
 * - Matrix dimensions in file: <nrows> rows, <ncols> columns.
 * - Reads <ncols> column headers (set as ml->headers) and <nrows> row labels (set as ml->labels).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_load_ascii( char const *restrict filename, index_t nrows, index_t ncols, bool hasname, bool hasheaders, bool haslabels,
			bool transpose, real *restrict *restrict matrix, struct matrix_labels *restrict ml )
{

	if ( ! matrix ) {
		fflush(stdout);
		errno = EFAULT;
		perror("\nmatrix_load_ascii( matrix )");
		return EXIT_FAILURE;
	}

	{
		uintmax_t const nitems = ((uintmax_t) nrows) * ((uintmax_t) ncols);

		if ( (nrows <= 0) + (ncols <= 0) + (nitems > IDX_MAX) ) {
			fflush(stdout);
			errno = EINVAL;
			fprintf( stderr, "\nmatrix_load_ascii( rows=%" PRI_IDX ", columns=%" PRI_IDX " ): %s\n",
				nrows, ncols, strerror(errno));
			if ( nitems > IDX_MAX )
				fprintf(stderr, "Matrix dimensions are too large.\nData matrices are limited to:\n"
						"\t* %" PRI_IDX " rows.\n\t* %" PRI_IDX " columns.\n\t* %" PRI_IDX " items.\n",
						IDX_MAX, IDX_MAX, IDX_MAX);
			return EXIT_FAILURE;

		} // if overflows
	}

	// Starts Reading ...
	FILE *restrict const file = fopen( filename, "r" );
	if( ! file ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nfopen '%s' :\n\t%s.\n", filename, strerror(errno) );
		fprintf(stderr,"Error in matrix_load_ascii().\n");
		return EXIT_FAILURE;
	}

	char *restrict name = NULL;
	char *restrict headers = NULL;
	char **restrict pheaders = NULL;
	char *restrict labels = NULL;
	char **restrict plabels = NULL;

	// Skips name, headers and labels if ml is NULL.
	bool const skip = ! ml;	// ( ml == NULL )

	// Delimiter character (TAB by default).
	int delimiter = (int) '\t';

	// Line number to be read.
	index_t nlines = 1;

	// ----------------------------


	// Name or headers.

	if ( hasname + hasheaders ) {

		char *restrict data = NULL;
		size_t len_data = 0;

		// Reads line 1.
		len_data = read_line( file, &data );
		if ( ! data ) {
			if ( len_data )
				fprintf(stderr,"Error in matrix_load_ascii().\n");
			else {
				fflush(stdout);
				fprintf(stderr,"\nError reading input file: file is empty.\n");
			}
			fclose(file);
			return EXIT_FAILURE;
		}

		if ( skip )
			free(data);

		else if ( hasheaders ) {
			// "Tokenizes" line 1.
			char **restrict pdata = NULL;
			size_t ntokens = tokenize( data, &pdata, delimiter );
			if ( ntokens == 1 ) {	// "Retokenizes" using spaces as delimiter.
				#if NMFGPU_DEBUG_READ_MATRIX
					printf("ntokens Line1=1\n");
					fflush(stdout);
				#endif
				delimiter = (int) ' ';
				free(pdata); pdata = NULL;
				ntokens = tokenize( data, &pdata, delimiter );
			}
			#if NMFGPU_DEBUG_READ_MATRIX
				printf("ntokens Line1=%zu\n",ntokens);
				fflush(stdout);
			#endif
			if ( ntokens != (size_t) (ncols + hasname) ) {	// headers or headers+name
				if ( ntokens ) {
					fflush(stdout);
					fprintf(stderr,"\nError reading input file: invalid file format:\nNumber of column labels (%zu) "
							"and number of matrix columns (%" PRI_IDX ") mismatch.\n",
							ntokens-hasname,ncols);
					free(pdata);
				}
				else
					fprintf(stderr,"Error in matrix_load_ascii().\n");
				free(data);
				fclose(file);
				return EXIT_FAILURE;
			}

			if ( hasname ) {	// Name and headers

				// The first token is the 'name' field, and the rest are column headers.

				// Name: copies the first token.
				name = (char *restrict) malloc( (strlen(data) + 1) * sizeof(char) );
				if ( ! name ) {
					int const err = errno; fflush(stdout); errno = err;
					perror("\nmalloc( name )");
					fprintf(stderr,"Error in matrix_load_ascii().\n");
					free(data); free(pdata);
					fclose(file);
					return EXIT_FAILURE;
				}
				strcpy( name, data );

				/* Headers: Sets remaining tokens, starting from the second one, as column headers.
				 *
				 *	Instead of allocating memory and copying 'data' to 'headers' (starting from the second token), it just
				 *	moves that second token to the "beginning" of 'data' (i.e., to the address returned by the read_line()
				 *	call above) and overwrites the first token (already copied to 'name'). Remaining tokens are kept
				 *	untouched, and the previous place of the second token is left as "garbage".
				 *
				 *	This way data == headers == pheaders[0], and it is possible to call free(headers).
				 */
				char **p_pdata = pdata + 1;	// Second token.
				char *p = *p_pdata;
				headers = memmove( data, p, (strlen(p) + 1) * sizeof(char) );	// Overwrites the first token.
				if ( ! headers )  {
					int const err = errno; fflush(stdout); errno = err;
					perror("\nmemmove( headers )");
					fprintf(stderr,"Error in matrix_load_ascii().\n");
					free(name);
					free(pdata); free(data);
					fclose(file);
					return EXIT_FAILURE;
				}

				// pheaders: Copies pdata[i+1] to pheaders[i]
				pheaders = (char **restrict) malloc( ncols * sizeof(char *) );
				if ( ! pheaders ) {
					int const err = errno; fflush(stdout); errno = err;
					perror("\nmalloc( pheaders )");
					fprintf(stderr,"Error in matrix_load_ascii().\n");
					free(name);
					free(pdata); free(data);
					fclose(file);
					return EXIT_FAILURE;
				}
				memcpy( pheaders, p_pdata, ncols * sizeof(char *) );	// pheaders[i] = pdata[i+1]
				free(pdata);

			} else { // No name, headers only.
				headers = data;
				pheaders = pdata;
			}

		} else // No headers, Name only.
			name = data;

		// Now, reading line 2...
		nlines = 2;

	} // if has name or headers


	// ----------------------------


	// Labels

	size_t max_len_labels = 0;
	size_t len_labels = 0;
	char *p_labels = NULL;
	char **p_plabels = NULL;

	if ( haslabels * ( ! skip ) ) {
		max_len_labels = 64 * nrows;				// Initial size for <nrows> labels of 64 characters each.
		labels = (char *restrict) malloc( max_len_labels * sizeof(char) );	// Memory size will be adjusted later.
		if ( ! labels ) {
			int const err = errno; fflush(stdout); errno = err;
			perror("\nmalloc( labels )");
			fprintf(stderr,"Error in matrix_load_ascii().\n");
			if ( headers ) { free(pheaders); free(headers); }
			if ( name ) free(name);
			fclose(file);
			return EXIT_FAILURE;
		}
		p_labels = labels;

		plabels = (char **restrict) malloc( nrows * sizeof(char *) );
		if ( ! plabels ) {
			int const err = errno; fflush(stdout); errno = err;
			perror("\nmalloc( plabels )");
			fprintf(stderr,"Error in matrix_load_ascii().\n");
			free(labels);
			if ( headers ) { free(pheaders); free(headers); }
			if ( name ) free(name);
			fclose(file);
			return EXIT_FAILURE;
		}
		p_plabels = plabels;
	} // if has labels.

	// ----------------------------

	// Data matrix.

	real *restrict l_matrix = (*matrix);
	if ( ! l_matrix ) {	// Allocates memory
		l_matrix = (real *restrict) malloc( nrows * ncols * sizeof(real) );
		if ( ! l_matrix ) {
			int const err = errno; fflush(stdout); errno = err;
			perror("\nmalloc: ");
			fprintf(stderr,"Error in matrix_load_ascii().\n");
			struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
			clean_matrix_labels(l_ml);
			fclose(file);
			return EXIT_FAILURE;
		}
	}

	// Formats strings to be used with fscanf().
	char format_delimiter[16];				// Format string for delimiter/missing data.
	sprintf( format_delimiter, "%%1[\t \n\r%c]", EOF );	// Sets both, tabs and spaces, as valid delimiters.
	char format_data[16];					// Format string for numeric data: data + delimiter
	sprintf( format_data, "%%" SCNgREAL "%s", format_delimiter );

	#if NMFGPU_DEBUG_READ_MATRIX
		printf("Format delimiter: '%s' (len=%zu)\nFormat data: '%s' (len=%zu)\n",format_delimiter,
			strlen(format_delimiter),format_data,strlen(format_data));
		fflush(stdout);
	#endif


	// ----------------------------


	// Reading file...

	// Steps for outer and inner loops.
	index_t incr_outer_loop = ncols;	// Step for outer loop
	index_t incr_inner_loop = 1;		// Step for inner loop.
	if ( transpose ) {
		incr_outer_loop = 1;
		incr_inner_loop = nrows;
	}

	real *pmatrix = l_matrix;	// &matrix[row][0] (or &matrix[0][row] if transpose)
	for ( index_t r = 0 ; r < nrows ; r++, nlines++, pmatrix += incr_outer_loop ) {

		if ( haslabels ) {
			// Reads a token using the selected delimiter.
			char *restrict data = NULL;
			int last_char = 0;	// Last char read.
			size_t const len_data = read_token( file, delimiter, &data, &last_char );
			if ( ! data ) {
				if ( len_data )
					fprintf(stderr,"Error in matrix_load_ascii().\n");
				else { // EOF
					fflush(stdout);
					fprintf(stderr, "\nError reading data label in row %" PRI_IDX ": premature end of file.\n",
						nlines );
				}
				struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
				clean_matrix_labels(l_ml);
				if (! (*matrix)) free(l_matrix);
				fclose(file);
				return EXIT_FAILURE;
			}
			if ( last_char != delimiter ) { // Blank line.
				free(data);

				// Checks for more blank lines.
				index_t const nlines0 = nlines;
				while ( ( last_char == (int) '\r' ) + ( last_char == (int) '\n' ) ) {
					if ( last_char == (int) '\r' )
						fgetc(file);	// Skips the LF character.
					last_char = fgetc(file);
					nlines++;
				}

				// Just blank lines at the end of the file.
				if ( feof(file) )
					break;

				// Fails on error or if there are more non-blank lines to read.
				fflush(stdout);
				if ( ferror(file) )
					fprintf(stderr,"\nInternal error in fgetc().\nError in matrix_load_ascii().\n");
				else if ( nlines0 < nlines )
					fprintf(stderr,"\nError reading input file: No matrix data between lines %" PRI_IDX " and %"
							PRI_IDX ".\nInvalid file format.\n\n", nlines0, nlines);
				else
					fprintf(stderr,"\nError reading input file: No data for row %" PRI_IDX
							".\nInvalid file format.\n", nlines0 );
				struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
				clean_matrix_labels(l_ml);
				if (! (*matrix)) free(l_matrix);
				fclose(file);
				return EXIT_FAILURE;
			}

			if ( !skip ) {
				// Before setting the label, checks if there is enough memory.
				size_t len = len_labels + len_data + 1;
				if ( len > max_len_labels ) { // Allocates more memory
					do {
						max_len_labels *= 2;
					} while ( len >= max_len_labels );
					char *const tmp = (char *) realloc( labels, max_len_labels * sizeof(char) );
					if ( ! tmp ) {
						int const err = errno; fflush(stdout); errno = err;
						perror("\nrealloc( labels )");
						fprintf( stderr, "Error in matrix_load_ascii().\n" );
						struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
						clean_matrix_labels(l_ml);
						if (! (*matrix)) free( (void *) l_matrix );
						fclose(file);
						return EXIT_FAILURE;
					}
					labels = tmp;
					p_labels = tmp + len_labels; // Pointer to place for new label.

					// Resets 'plabels'.
					retok( (char const *)labels, (char const **)plabels, r );
					p_plabels = plabels + r;	// Pointer to place for new label.

				} // If allocate more memory for labels

				// Sets vector of pointers to labels
				strcpy(p_labels, data);		// Copies the token.
				*p_plabels = p_labels;		// Sets p_plabels[r].

				// pointer to place for next label
				p_labels += (len_data + 1);
				p_plabels++;

				len_labels = len;	// len == (len_labels + len_data + 1)

			} // If set the label.

			free(data);

		} // if haslabels

		#if NMFGPU_DEBUG_READ_MATRIX
			printf("Reading row %" PRI_IDX " (nlines=%" PRI_IDX ")...\n",r,nlines);
			fflush(stdout);
		#endif

		real *pmatrix_r = pmatrix;	// &matrix[row][col] (or &matrix[col][row] if transpose)
		for ( index_t col = 0 ; col < (ncols-1) ; col++, pmatrix_r += incr_inner_loop ) {

			real value = REAL_C( 0.0 );

			// If it reads a delimiter, then it is a missing data ('value' is set to 0).
			int c = 0;
			int conv = fscanf( file, format_delimiter, &c );
			int max_conv = 1;	// Maximum items to be converted from string.

			#if NMFGPU_DEBUG_READ_MATRIX2
				printf("\tconv1=%i",conv);
				fflush(stdout);
			#endif

			if ( ! conv ) { // No missing data: reads current value (data + new_delimiter).
				c = 0;
				conv = fscanf( file, format_data, &value, &c );
				max_conv = 2;
				#if NMFGPU_DEBUG_READ_MATRIX2
					printf(",conv2=%i,c=",conv);
					if (!c) printf("(emtpy)");
					else if ( isprint(c) + isblank(c) ) printf("'%c'",c);
					else printf("'\\x%X'",c);
				#endif
			}
			#if NMFGPU_DEBUG_READ_MATRIX2
				printf(",value=%g ",value);
				fflush(stdout);
			#endif

			// Fails if premature EOF, premature EOL, or not a number (NaN, inf,...).
			if ( (conv < max_conv) + ! (isblank(c) * is_valid(value)) ) {	// conv < max_conv || ! isblank(c) || ! is_valid(value)
				int const err = errno; fflush(stdout); errno = err;
				if ( errno + ferror(file) ) {	// Error.
					if ( errno ) { perror("\nfscanf()"); }
					fprintf( stderr, "Error in matrix_load_ascii() reading row %" PRI_IDX " (line %" PRI_IDX
							"), column %" PRI_IDX "\n\n", r, nlines, col);
				}
				else if ( (c == (int) '\n') + (c == (int) '\r') )	// Premature EOL.
					fprintf(stderr, "\nError reading input file:\nPremature end of line detected in line %" PRI_IDX
							" (%" PRI_IDX " columns found, %" PRI_IDX " expected).\n"
							"Invalid file format.\n\n", nlines, col+haslabels+1, ncols+haslabels);
				else if ( (conv == EOF) + feof(file) ) // Premature EOF.
					fprintf( stderr, "\nError reading input file:\nPremature end of file detected in line %" PRI_IDX
							" (%" PRI_IDX " columns found, %" PRI_IDX " expected).\n"
							"Invalid file format.\n\n", nlines, col+haslabels, ncols+haslabels);
				else if ( ! is_valid(value) )	// Not a number
					fprintf(stderr, "\nError reading input file:\nLine %" PRI_IDX ", column %" PRI_IDX
							": Invalid numeric data: %g\n\n", nlines, col+haslabels+1, value);
				else { // Illegal character.
					fprintf(stderr,"\nError reading input file:\nLine %" PRI_IDX ", column %" PRI_IDX
							": Invalid character", nlines, col+haslabels+1);
					c = fgetc( file );	// Reads illegal character
					if ( c != EOF ) {
						if ( isprint(c) ) fprintf(stderr, ": '%c' ('\\x%X')",c,c);
						else fprintf(stderr, ": '\\x%X'",c);
					}
					fprintf(stderr,".\n\n");
				}
				struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
				clean_matrix_labels(l_ml);
				if (! (*matrix)) free( (void *) l_matrix );
				fclose(file);
				return EXIT_FAILURE;
			}

			*pmatrix_r = value;	// matrix[row][col] (or matrix[col][row] if transpose)

		} // for (0 <= col < ncols-1)

		// last file column: col == ncols-1
		{
			real value = REAL_C( 0.0 );

			// If it reads a '\n', then it is a missing data (value is set to 0).
			int c = (int) '\n';
			int conv = fscanf( file, format_delimiter, &c );

			#if NMFGPU_DEBUG_READ_MATRIX2
				printf(" ... conv1=%i",conv);
				if ( c != (int) '\n' ) {
					printf(",c1=");
					if ( c == (int) '\r' ) printf("'\\r'");
					else if ( isprint(c) + isblank(c) ) printf("'%c'",c);
					else printf("'\\x%X'",c);
				}
				fflush(stdout);
			#endif

			if ( isblank(c) + ferror(file) + feof(file) ) {
				int const err = errno; fflush(stdout); errno = err;
				if ( isblank(c) )	// More than ncols columns.
					fprintf(stderr, "\nError reading input file:\nLine %" PRI_IDX ": Invalid file format.\n"
							"There are more than %" PRI_IDX " data columns.\n\n",nlines,ncols);
				else if ( (conv == EOF) + feof(file) ) // Premature EOF.
					fprintf(stderr, "\nError reading input file:\nPremature end-of-file detected in line %" PRI_IDX
							" (%" PRI_IDX " columns found, %" PRI_IDX " expected).\n"
							"Invalid file format.\n\n", nlines, ncols-1+haslabels, ncols+haslabels);
				else { // Error.
					if ( errno ) { perror("\nfscanf()"); }
					fprintf( stderr, "Error in matrix_load_ascii() reading row %" PRI_IDX ", matrix column %"
							PRI_IDX "\n", r, ncols-1);
				}
				struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
				clean_matrix_labels(l_ml);
				if (! (*matrix)) free( (void *) l_matrix );
				fclose(file);
				return EXIT_FAILURE;
			}

			// No missing data: reads last value (plus ['\r' or '\n']).
			if ( ! conv ) {
				c = 0;
				conv = fscanf( file, format_data, &value, &c );

				#if NMFGPU_DEBUG_READ_MATRIX2
					printf(",conv2=%i,c=",conv);
					if ( c == (int) '\n' ) printf("'\\n'");
					else if ( c == (int) '\r' ) printf("'\\r'");
					else if ( isprint(c) + isblank(c) ) printf("'%c'",c);
					else if (!c) printf("(empty)");
					else printf("'\\x%X'",c);
					printf(",value=%g",value);
					fflush(stdout);
				#endif

				// Fails on invalid format or error (conv=EOF), more columns, invalid characters, or invalid value.
				if ( (conv <= 0) + (c == delimiter) + (! (c + feof(file))) + (! is_valid(value)) ) {
					int const err = errno; fflush(stdout); errno = err;
					if ( errno + ferror(file) ) {	// Error.
						if ( errno ) { perror("\nfscanf()"); }
						fprintf(stderr,"Error in matrix_load_ascii() reading row %" PRI_IDX ", matrix column %"
								PRI_IDX "\n", r, ncols-1);
					}
					else if ( isblank(c) )	// More than ncols columns.
						fprintf( stderr, "\nError reading input file:\nLine %" PRI_IDX ": There are more than %"
								PRI_IDX " data columns or line finishes with a '%c'.\n\n",
								nlines, ncols, c );
					else if ( ! is_valid(value) )	// Not a number
						fprintf(stderr, "\nError reading input file:\nLine %" PRI_IDX ", column %" PRI_IDX
								": Invalid numeric data: %g\n\n", nlines, ncols-1+haslabels+1, value);
					else { // Illegal character.
						fprintf(stderr, "\nError reading input file:\nLine %" PRI_IDX ", column %" PRI_IDX
								": Invalid character", nlines, ncols-1+haslabels+1);
						c = fgetc( file );	// Reads illegal character
						if ( c != EOF ) {
							if ( isprint(c) ) fprintf(stderr, ": '%c' ('\\x%X')",c,c);
							else fprintf(stderr, ": '\\x%X'",c);
						}
						fprintf(stderr,".\n\n");
					}
					struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
					clean_matrix_labels(l_ml);
					if (! (*matrix)) free( (void *) l_matrix );
					fclose(file);
					return EXIT_FAILURE;
				}

			} // If last data was read.
			#if NMFGPU_DEBUG_READ_MATRIX2
			// Missing data
			else { printf(",value=(empty)"); fflush(stdout); }
			#endif

			// If the last character read was a CR ('\r'), reads the corresponding LF ('\n')
			if ( c == (int) '\r' ) {

				c = fgetc(file);

				#if NMFGPU_DEBUG_READ_MATRIX2
					if ( c == (int) '\n' ) printf(",c3='\\n'");
					else if ( c == (int) '\r' ) printf(",c3='\\r'");
					else if ( isprint(c) + isblank(c) ) printf(",c3='%c'",c);
					else printf(",c3='\\x%X'",c);
					fflush(stdout);
				#endif

				// Fails if it is not a LF character (assumes this is not an old OS).
				if ( ( ferror(file) ) + ( c != (int) '\n' ) ) {
					fflush(stdout);
					if ( c != (int) '\n' ) {
						fprintf(stderr,"\nLine %" PRI_IDX ": Unexpected character after a CR ('\\r'): ",nlines);
						if ( isprint(c) + isblank(c) ) fprintf(stderr,"'%c'.\n\n",c);
						else fprintf(stderr,"'\\x%X'.\n\n",c);
					} else
					fprintf(stderr,"\nInternal error in fgetc(\\n).\nError in matrix_load_ascii().\n");
					struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
					clean_matrix_labels(l_ml);
					if (! (*matrix)) free( (void *) l_matrix );
					fclose(file);
					return EXIT_FAILURE;
				}

			} // if last character read was a CR ('\r').

 			#if NMFGPU_DEBUG_READ_MATRIX || NMFGPU_DEBUG_READ_MATRIX2
				printf("\n"); fflush(stdout);
 			#endif

			// Stores new values.
			*pmatrix_r = value;	// matrix[row][ncols-1] (or matrix[ncols-1][row] if transpose).
			pmatrix_r += incr_inner_loop;

		} // last column.

	} // for ( 0 <= r < nrows )

	// Resizes labels.
	if ( haslabels * (! skip) ) {
		#if NMFGPU_DEBUG_READ_MATRIX
			printf("Resizing labels from %zu to %zu\n",max_len_labels,len_labels);
			fflush(stdout);
		#endif
		labels = (char *) realloc( labels, len_labels * sizeof(char) );
	}

	fclose(file);

	// Sets output values.
	*matrix = l_matrix;
	if ( ! skip )
		*ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );

	return EXIT_SUCCESS;

} // matrix_load_ascii

//////////////////////////////////////////////////

/*
 * Loads a matrix from a (non-"native") binary file (i.e., double-precision data and unsigned int's).
 *
 * Detects automatically if matrix has name, column headers and/or row labels,
 * as well as the used tag delimiter (space or tab character). Skips if 'ml' is NULL.
 *
 * If (*matrix) is non-NULL, do not allocates memory but uses the supplied one.
 * WARNING: In such case, THERE MUST BE ENOUGH MEMORY ALREADY ALLOCATED.
 *
 * If 'transpose' is 'true', transposes matrix in memory as follows:
 * - Matrix dimensions in memory: <ncols> rows, <nrows> columns.
 * - Matrix dimensions in file: <nrows> rows, <ncols> columns.
 * - Reads <ncols> column headers (set as ml->headers) and <nrows> row labels (set as ml->labels).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_load_binary( char const *restrict filename, index_t numrows, index_t numcols, bool transpose, real *restrict *restrict matrix,
			struct matrix_labels *restrict ml )
{

	if ( ! matrix ) {
		fflush(stdout);
		errno = EFAULT;
		perror("\nmatrix_load_binary( matrix )");
		return EXIT_FAILURE;
	}

	{
		uintmax_t const nitems = ((uintmax_t) numrows) * ((uintmax_t) numcols);

		if ( (numrows <= 0) + (numcols <= 0) + (nitems > IDX_MAX) ) {
			fflush(stdout);
			errno = EINVAL;
			fprintf( stderr, "\nmatrix_load_binary( rows=%" PRI_IDX ", columns=%" PRI_IDX " ): %s\n",
				numrows, numcols, strerror(errno));
			if ( nitems > IDX_MAX )
				fprintf(stderr, "Matrix dimensions are too large.\nData matrices are limited to:\n"
						"\t* %" PRI_IDX " rows.\n\t* %" PRI_IDX " columns.\n\t* %" PRI_IDX " items.\n",
						IDX_MAX, IDX_MAX, IDX_MAX);
			return EXIT_FAILURE;
		}
	}
	// Starts reading...

	FILE *restrict const file = fopen( filename, "rb" );
	if( ! file ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nfopen '%s' :\n\t%s.\n", filename, strerror(errno) );
		fprintf( stderr, "Error in matrix_load_binary().\n" );
		return EXIT_FAILURE;
	}


	// Matrix dimensions.
	{
		unsigned int dim[2];
		size_t const nread = fread( dim, sizeof(int), 2, file );
		if ( nread != 2 ) {
			fflush(stdout);
			if ( feof(file) )
				fprintf(stderr, "\nError reading input file:\nPremature end-of-file detected. Invalid file format.\n\n");
			else // error
				fprintf(stderr, "\nInternal error in fread(dim).\nError in matrix_load_binary().\n");
			fclose(file);
			return EXIT_FAILURE;
		}

		// Checks dimensions.
		if ( ((unsigned int) numrows != dim[0]) + ((unsigned int) numcols != dim[1]) ) {
			fflush(stdout);
			fprintf(stderr, "\nError reading input file:\nInvalid matrix dimensions: %u x %u (expected: %"
					PRI_IDX " x %" PRI_IDX ")\n", dim[0], dim[1], numrows, numcols);
			fclose(file);
			return EXIT_FAILURE;
		}
	}

	// ------------------------------------------


	// Data matrix

	real *restrict data_matrix = (*matrix);
	if ( ! data_matrix ) {	// Allocates memory
		data_matrix = (real *restrict) malloc( numrows * numcols * sizeof(real) );
		if ( ! data_matrix ) {
			int const err = errno; fflush(stdout); errno = err;
			perror("\nmalloc: ");
			fprintf(stderr,"Error in matrix_load_binary().\n");
			fclose(file);
			return EXIT_FAILURE;
		}
	}


	// ----------------------------------


	// Reading file....


	#if NMFGPU_DEBUG_READ_MATRIX2
		printf("\n");
		fflush(stdout);
	#endif


	// Steps for outer and inner loops.
	index_t incr_outer_loop = numcols;	// Step for outer loop
	index_t incr_inner_loop = 1;	// Step for inner loop.
	if ( transpose ) {
		incr_outer_loop = 1;
		incr_inner_loop = numrows;
	}

	real *pmatrix = data_matrix;	// &matrix[i][0] (or &matrix[0][i] if transpose)
	for ( index_t i = 0 ; i < numrows ; i++, pmatrix += incr_outer_loop ) {

		real *pmatrix_r = pmatrix; // &matrix[i][j] (or &matrix[j][i] if transpose)
		for ( index_t j = 0 ; j < numcols ; j++, pmatrix_r += incr_inner_loop ) {

			// Reads current data value.
			double value = 0;
			size_t const nread = fread( &value, sizeof(double), 1, file ); // Reads one double-precision data.
			real const num = (real) value;

			#if NMFGPU_DEBUG_READ_MATRIX2
				printf("%g ",value); fflush(stdout);
			#endif

			// Checks data.
			if ( ! ( nread * is_valid(num) ) ) {
				fflush(stdout);
				if ( feof(file) )
					fprintf(stderr,"\nError reading input file:\nPremature end-of-file detected.\nInvalid file format.\n\n");
				else { // error
					index_t const r = ( transpose ? j : i );
					index_t const c = ( transpose ? i : j );
					fprintf(stderr, "\nError reading input file:\nError reading row %" PRI_IDX ", column %" PRI_IDX, r, c);
					if ( (! nread) + ferror(file) )	// Error in file
						fprintf( stderr, ".\nInvalid file format.\n\n" );
					else	// Invalid numeric format
						fprintf( stderr, ": '%g'.\nInvalid numeric format.\n\n", value );
				}
				if (! (*matrix)) free( (void *) data_matrix );
				fclose(file);
				return EXIT_FAILURE;
			}

			// Stores new value.
			*pmatrix_r = num;	// matrix[i][j] (or matrix[j][i] if transpose)

		} // for (0 <= j < numcols)

		#if NMFGPU_DEBUG_READ_MATRIX2
			printf("\n");
			fflush(stdout);
		#endif

	} // for (0 <= i < numrows)

	*matrix = data_matrix;


	// ------------------------------


	// Reads labels, headers and name (as plain text) if they exists.


	// Skips matrix labels.
	if ( ! ml ) {
		fclose(file);
		return EXIT_SUCCESS;
	}

	char *restrict name = NULL;
	char *restrict headers = NULL;
	char **restrict pheaders = NULL;
	char *restrict labels = NULL;
	char **restrict plabels = NULL;


	// Checks for row labels
	size_t len = read_line( file, &labels );
	if ( ! labels ) {
		fclose(file);
		if ( len ) {
			fprintf(stderr,"Error in matrix_load_binary().\n\n");
			if (! (*matrix)) free( (void *) data_matrix );
			return EXIT_FAILURE;
		}
		*ml = NEW_MATRIX_LABELS( NULL, NULL, NULL, NULL, NULL );
		return EXIT_SUCCESS;
	}
	if ( len >= 3 ) { // minimum length for two labels.

		// Divides into tokens by replacing all tabs characters by '\0'.
		size_t ntokens = tokenize( labels, &plabels, (int) '\t' );

		// If there is only one token, "retokenizes" using space as delimiter.
		if ( ntokens == 1 ) {
			#if NMFGPU_DEBUG_READ_MATRIX
				printf("\tntokens Labels=1\n"); fflush(stdout);
			#endif
			free(plabels);
			plabels = NULL;
			ntokens = tokenize( labels, &plabels, (int) ' ' );
		} // if ntokens == 1

		#if NMFGPU_DEBUG_READ_MATRIX
			printf("\tntokens Labels=%zu\n",ntokens);
			fflush(stdout);
		#endif

		if ( ntokens != (size_t) numrows ) {
			fflush(stdout);
			if ( ntokens ) {
				fprintf( stderr, "\nError reading input file. Invalid format:\nNumber of row labels (%zu) and number of "
					"matrix rows (%" PRI_IDX ") mismatch.\n", ntokens, numrows );
				if ( ntokens > (size_t) numrows )
					fprintf( stderr, "Please remember to set a '\\n' (new-line) character between column labels, row "
						"labels and the description string.\n\n");
				free(plabels);
			}
			else	// Error.
				fprintf(stderr,"Error in matrix_load_binary().\n\n");
			free(labels);
			if (! (*matrix)) free( (void *) data_matrix );
			fclose(file);
			return EXIT_FAILURE;
		}

	} else {	// No labels.
		#if NMFGPU_DEBUG_READ_MATRIX
			printf("\t'No Labels' (len=%zu)\n",len);
			fflush(stdout);
		#endif
		free(labels);
		labels = NULL;
	}


	// Checks for columns headers
	len = read_line( file, &headers );
	if ( ! headers ) {
		fclose(file);
		if ( len ) {
			fprintf(stderr,"Error in matrix_load_binary().\n");
			if ( labels ) { free(labels); free(plabels); }
			if (! (*matrix)) free( (void *) data_matrix );
			return EXIT_FAILURE;
		}
		*ml = NEW_MATRIX_LABELS( NULL, NULL, NULL, labels, plabels );
		return EXIT_SUCCESS;
	}
	if ( len >= 3 ) { // minimum length for two headers.

		// Divides into tokens by replacing all tabs characters by '\0'.
		size_t ntokens = tokenize( headers, &pheaders, (int) '\t' );

		// If there is only one token, retokenizes using space as delimiter
		if ( ntokens == 1 ) {
			#if NMFGPU_DEBUG_READ_MATRIX
				printf("\tntokens Headers=1\n"); fflush(stdout);
			#endif
			free(pheaders);
			pheaders = NULL;
			ntokens = tokenize( headers, &pheaders, (int) ' ' );
		} // if ntokens == 1

		#if NMFGPU_DEBUG_READ_MATRIX
			printf("\tntokens Headers=%zu\n",ntokens);
		#endif

		if ( ntokens != (size_t) numcols ) {
			fflush(stdout);
			if ( ntokens ) {
				fflush(stdout);
				fprintf( stderr,"\nError reading input file. Invalid format:\nNumber of column labels (%zu) and number of "
						"matrix columns (%" PRI_IDX ") mismatch.\n", ntokens, numcols );
				if ( ntokens > (size_t) numcols )
					fprintf( stderr, "Please remember to set a '\\n' (new-line) character between headers labels, "
							"row labels and the description string.\n\n");
				free(pheaders);
			}
			else
				fprintf(stderr,"Error in matrix_load_binary().\n");
			free(headers);
			if (! (*matrix)) free( (void *) data_matrix );
			if ( labels ) { free(labels); free(plabels); }
			fclose(file);
			return EXIT_FAILURE;
		}

	} else { // No headers
		#if NMFGPU_DEBUG_READ_MATRIX
			printf("\t'No Headers' (len=%zu)\n",len);
			fflush(stdout);
		#endif
		free(headers);
		headers = NULL;
	}


	// Checks for name.
	len = read_token( file, (int) '\t', &name, NULL );

	#if NMFGPU_DEBUG_READ_MATRIX
		printf("\tName (len=%zu):\n",len);
	#endif

	fclose(file);

	if ( (! name) * len ) {
		fprintf(stderr,"Error in matrix_load_binary().\n");
		struct matrix_labels l_ml = NEW_MATRIX_LABELS( NULL, headers, pheaders, labels, plabels );
		clean_matrix_labels(l_ml);
		if (! (*matrix)) free( (void *) data_matrix );
		return EXIT_FAILURE;
	}

	#if NMFGPU_DEBUG_READ_MATRIX
		printf("\t\t'%s'\n",name);
	#endif

	// Sets output matrix labels.
	*ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );

	return EXIT_SUCCESS;

} // matrix_load_binary

//////////////////////////////////////////////////

/*
 * Loads a matrix from a "native" binary file (i.e., with the compiled types for matrix data and dimensions).
 * Detects automatically if matrix has name, column headers and/or row labels, unless 'ml' is set to NULL.
 *
 * If 'matrix' is NULL, skips data matrix (just reads matrix dimensions).
 * Else, if in addition (*matrix != NULL), do not allocates memory for the data matrix, but uses the supplied one.
 *
 * Reads <length> items, starting from the <offset>-th element, if these values are positive (they are
 * ignored otherwise). Skips data matrix if (offset + length) >= matrix dimensions.
 *
 * WARNING:
 *	- For internal use only.
 *	- If *matrix is non-NULL, IT MUST HAVE ENOUGH MEMORY ALREADY ALLOCATED.
 *	- NO ERROR-CHECKING IS PERFORMED (e.g., overflow, invalid values....).
 *
 * Returns EXIT_SUCCESS (or EXIT_FAILURE if could not open filename).
 */
int matrix_load_binary_native( char const *restrict filename, index_t offset, index_t length, real *restrict *restrict matrix,
				index_t *restrict nrows, index_t *restrict ncols, struct matrix_labels *restrict ml )
{

	// Checks for NULL parameters
	if ( ! ( (size_t) nrows * (size_t) ncols ) ) {
		fflush( stdout );
		errno = EFAULT;
		if ( ! nrows ) perror("\nmatrix_load_binary_native( nrows )");
		if ( ! ncols ) perror("\nmatrix_load_binary_native( ncols )");
		return EXIT_FAILURE;
	}


	// Starts Reading ...
	FILE *restrict const file = fopen( filename, "rb" );
	if( ! file ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nfopen '%s' :\n\t%s.\n", filename, strerror(errno) );
		fprintf(stderr,"Error in matrix_load_binary_native().\n");
		return EXIT_FAILURE;
	}


	#if NMFGPU_DEBUG_READ_MATRIX
		printf("Reading file: '%s'\n",filename);
		fflush(stdout);
	#endif

	// Matrix size
	index_t dim[2];
	size_t nread = fread( dim, sizeof(index_t), 2, file );
	if ( nread != 2 ) {
		fflush(stdout);
		if ( feof(file) )
			fprintf(stderr,"\nError reading matrix dimensions:\nPremature end-of-file detected.\n" );
		else // error
			fprintf(stderr,"\nInternal error in function fread (%zu items read, 2 expected).\n", nread);
		fprintf(stderr,"Error in matrix_load_binary_native().\n");
		fclose(file);
		return EXIT_FAILURE;
	}

	index_t nitems;
	{
		uintmax_t nitems_overflow = (uintmax_t) dim[0] * (uintmax_t) dim[1];

		if ( (dim[0] <= 0) || (dim[1] <= 0) || (nitems_overflow > (uintmax_t) IDX_MAX) ) {
			fflush(stdout);
			errno = EINVAL;
			fprintf( stderr, "\nmatrix_load_binary_native( rows=%" PRI_IDX ", columns=%" PRI_IDX " ): %s\n",
				dim[0], dim[1], strerror(errno));
			if ( nitems_overflow > (uintmax_t) IDX_MAX )
				fprintf(stderr, "Matrix dimensions are too large.\nData matrices are limited to:\n"
						"\t* %" PRI_IDX " rows.\n\t* %" PRI_IDX " columns.\n\t* %" PRI_IDX " items.\n",
						IDX_MAX, IDX_MAX, IDX_MAX);
			return EXIT_FAILURE;
		}

		nitems = (index_t) nitems_overflow;
	}

	#if NMFGPU_DEBUG_READ_MATRIX
		printf("\tMatrix Dimensions: %" PRI_IDX "x%" PRI_IDX " (%" PRI_IDX " items)\n", dim[0], dim[1], nitems );
		fflush(stdout);
	#endif

	/////////////////////////////////

	// Reads data matrix

	real *restrict data_matrix = NULL;

	// If data matrix will be read.
	bool const read_matrix = ( ((size_t) matrix) * ((offset+length) <= nitems) );

	bool allocated_memory = false;	// If memory was allocated for data matrix.

	if ( read_matrix ) {

		// If length is not specified, reads all the matrix.
		index_t const size = ( length ? length : nitems ) ;

		data_matrix = (real *restrict) (*matrix);
		if ( ! data_matrix ) {

			#if NMFGPU_DEBUG_READ_MATRIX
				printf("\tAllocating memory for %" PRI_IDX " data items (length=%" PRI_IDX ")...\n", size, length );
				fflush(stdout);
			#endif

			data_matrix = (real *restrict) malloc( size * sizeof(real) );
			if ( ! data_matrix ) {
				int const err = errno; fflush(stdout); errno = err;
				perror("\nmalloc: ");
				fprintf(stderr,"Error in matrix_load_binary_native().\n");
				fclose(file);
				return EXIT_FAILURE;
			}

			allocated_memory = true;

			*matrix = data_matrix;	// Sets the new address.

		}

		if ( offset && fseek( file, offset * sizeof(real), SEEK_CUR ) ) {
			int const err = errno; fflush(stdout); errno = err;
			fprintf( stderr, "\nfseek(%" PRI_IDX "): %s\n", offset, strerror(errno) );
			fprintf(stderr,"Error in matrix_load_binary_native().\n");
			if (allocated_memory) free( (void *) data_matrix );
			fclose(file);
			return EXIT_FAILURE;
		}

		#if NMFGPU_DEBUG_READ_MATRIX
			printf("\tReading %" PRI_IDX " items (length=%" PRI_IDX ") from data matrix (starting at offset %" PRI_IDX ")... ",
				size, length, offset );
			fflush(stdout);
		#endif

		nread = fread( data_matrix, sizeof(real), size, file );
		if ( nread != (size_t) size ) {
			fflush(stdout);
			fprintf(stderr,"\nError reading file: %zu items read, %" PRI_IDX " expected.\n"
					"Error in matrix_load_binary_native().\n", nread, size);
			fclose(file);
			if (allocated_memory) free( (void *) data_matrix );
			return EXIT_FAILURE;
		}

	} else { // Skips all data matrix

		#if NMFGPU_DEBUG_READ_MATRIX
			printf("\tSkipping data matrix... " );
			fflush(stdout);
		#endif

		if ( fseek( file, nitems * sizeof(real), SEEK_CUR ) ) {
			int const err = errno; fflush(stdout); errno = err;
			perror("\nfseek(data_matrix)");
			fprintf(stderr,"Error in matrix_load_binary_native().\n");
			fclose(file);
			return EXIT_FAILURE;
		}

	} // If read data matrix

	#if NMFGPU_DEBUG_READ_MATRIX
		printf("done.\n" );
		fflush(stdout);
	#endif

	*nrows = dim[0];
	*ncols = dim[1];

	/////////////////////////////////


	// Reads headers, labels and name (as plain text) if they exists.

	// Skips them if ml is set to NULL
	if ( ! ml ) {
		fclose(file);
		return EXIT_SUCCESS;
	}

	index_t const numrows = dim[0];
	index_t const numcols = dim[1];

	char *restrict name = NULL;
	char *restrict headers = NULL;
	char **restrict pheaders = NULL;
	char *restrict labels = NULL;
	char **restrict plabels = NULL;


	// Checks for row labels
	size_t len = read_line( file, &labels );
	if ( ! labels ) {
		#if NMFGPU_DEBUG_READ_MATRIX2
			printf("\tNo Labels (null)\n"); fflush(stdout);
		#endif
		fclose(file);
		if ( len ) { // Error
			fprintf(stderr, "Error reading row labels.\nError in matrix_load_binary_native().\n");
			if (allocated_memory) free( (void *) data_matrix );
			return EXIT_FAILURE;
		}
		*ml = NEW_MATRIX_LABELS( NULL, NULL, NULL, NULL, NULL );
		return EXIT_SUCCESS;
	}
	if ( len >= 3 ) { // minimum length for two labels.
		// Divides into tokens by replacing all tabs characters by '\0'.
		size_t ntokens = tokenize( labels, &plabels, (int) '\t' );

		// If there is only one token, "retokenizes" using space as delimiter.
		if ( ntokens == 1 ) {
			#if NMFGPU_DEBUG_READ_MATRIX
				printf("\tntokens Labels=1\n"); fflush(stdout);
			#endif
			free(plabels);
			plabels = NULL;
			ntokens = tokenize( labels, &plabels, (int) ' ' );
		} // if ntokens == 1

		#if NMFGPU_DEBUG_READ_MATRIX2
			printf("\tntokens Labels=%zu\n",ntokens);
			fflush(stdout);
		#endif

		if ( ntokens != (size_t) numrows ) {
			fflush(stdout);
			if ( ntokens ) {
				fprintf( stderr, "\nError reading input file. Invalid format:\nNumber of row labels (%zu) and number of "
					"matrix rows (%" PRI_IDX ") mismatch.\n", ntokens, numrows );
				if ( ntokens > (size_t) numrows )
					fprintf( stderr, "Please remember to set a '\\n' (new-line) character between column labels, row "
						"labels and the description string.\n\n");
				free(plabels);
			}
			else	// Error.
				fprintf(stderr,"Error in matrix_load_binary_native().\n\n");
			free(labels);
			if (! (*matrix)) free( (void *) data_matrix );
			fclose(file);
			return EXIT_FAILURE;
		}

	} else {	// No labels.
		#if NMFGPU_DEBUG_READ_MATRIX2
			printf("\t'No Labels' (len=%zu)\n",len); fflush(stdout);
		#endif
		free(labels);
		labels = NULL;
	}


	// Checks for columns headers
	len = read_line( file, &headers );
	if ( ! headers ) {
		#if NMFGPU_DEBUG_READ_MATRIX2
			printf("\tNo headers (null)\n"); fflush(stdout);
		#endif
		fclose(file);
		if ( len ) {
			fprintf(stderr, "Error reading column headers.\nError in matrix_load_binary_native().\n");
			if ( labels ) { free(plabels); free(labels); }
			if (allocated_memory) free( (void *) data_matrix );
			return EXIT_FAILURE;
		}
		*ml = NEW_MATRIX_LABELS( NULL, NULL, NULL, labels, plabels );
		return EXIT_SUCCESS;
	}
	if ( len >= 3 ) { // minimum length for two headers.
		// Divides into tokens by replacing all tabs characters by '\0'.
		size_t ntokens = tokenize( headers, &pheaders, (int) '\t' );

		// If there is only one token, retokenizes using space as delimiter
		if ( ntokens == 1 ) {
			#if NMFGPU_DEBUG_READ_MATRIX
				printf("\tntokens Headers=1\n"); fflush(stdout);
			#endif
			free(pheaders);
			pheaders = NULL;
			ntokens = tokenize( headers, &pheaders, (int) ' ' );
		} // if ntokens == 1

		#if NMFGPU_DEBUG_READ_MATRIX
			printf("\tntokens Headers=%zu\n",ntokens);
		#endif

		if ( ntokens != (size_t) numcols ) {
			fflush(stdout);
			if ( ntokens ) {
				fflush(stdout);
				fprintf( stderr,"\nError reading input file. Invalid format:\nNumber of column labels (%zu) and number of "
						"matrix columns (%" PRI_IDX ") mismatch.\n", ntokens, numcols );
				if ( ntokens > (size_t) numcols )
					fprintf( stderr, "Please remember to set a '\\n' (new-line) character between headers labels, "
							"row labels and the description string.\n\n");
				free(pheaders);
			}
			else
				fprintf(stderr,"Error in matrix_load_binary_native().\n");
			free(headers);
			if (! (*matrix)) free( (void *) data_matrix );
			if ( labels ) { free(labels); free(plabels); }
			fclose(file);
			return EXIT_FAILURE;
		}

	} else {	// No headers
		#if NMFGPU_DEBUG_READ_MATRIX2
			printf("\t'No headers' (len=%zu)\n",len);
		#endif
		free(headers);
		headers = NULL;
	}


	// Checks for name.
	len = read_token( file, (int) '\t', &name, NULL );

	#if NMFGPU_DEBUG_READ_MATRIX2
		printf("\tName (len=%zu):\n",len);
	#endif

	fclose(file);

	if ( ( ! name ) * len ) {
		fprintf(stderr, "Error reading description string.\nError in matrix_load_binary_native().\n");
		struct matrix_labels l_ml = NEW_MATRIX_LABELS( NULL, headers, pheaders, labels, plabels );
		clean_matrix_labels(l_ml);
		if (allocated_memory) free( (void *) data_matrix );
		return EXIT_FAILURE;
	} // if has name

	#if NMFGPU_DEBUG_READ_MATRIX2
		printf("\t\t'%s'\n",name);
	#endif

	*ml = (struct matrix_labels) NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );

	return EXIT_SUCCESS;

} // matrix_load_binary_native

//////////////////////////////////////////////////

/*
 * Reads input matrix according to the selected file format.
 *
 * is_bin: Reads output matrix from a binary file.
 *		== 0: Disabled. Reads the file as ASCII text.
 *		== 1: Uses "non-native" format (i.e., double-precision data, and "unsigned int" for dimensions).
 *		 > 1: Uses "native" or raw format (i.e., the compiled types for matrix data and dimensions).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_load( char const *restrict filename, bool numeric_hdrs, bool numeric_lbls, index_t is_bin, real *restrict *restrict matrix,
		index_t *restrict nrows, index_t *restrict ncols, struct matrix_labels *restrict ml )
{

	// Checks for NULL parameters
	if ( ! ( (size_t) filename * (size_t) matrix * (size_t) nrows * (size_t) ncols * (size_t) ml ) ) {
		fflush( stdout );
		errno = EFAULT;
		if ( ! filename ) perror("\nmatrix_load( filename )");
		if ( ! matrix )	perror("\nmatrix_load( matrix )");
		if ( ! nrows )	perror("\nmatrix_load( nrows )");
		if ( ! ncols )	perror("\nmatrix_load( ncols )");
		if ( ! ml )	perror("\nmatrix_load( ml )");
		return EXIT_FAILURE;
	}

	if ( is_bin < 0 ) {
		fflush(stdout);
		errno = EINVAL;
		fprintf( stderr, "\nmatrix_load( is_bin=%" PRI_IDX " ): %s\n", is_bin, strerror(errno));
		return EXIT_FAILURE;
	}

	int status = EXIT_SUCCESS;

	// -------------------------------

	// Loads the file.

	printf("\nLoading input file...\n");

	if ( is_bin > 1 ) { // Input file is "native" binary.

		printf("\tFile selected as \"native\" binary (i.e., the file is read using the data types specified at compilation).\n"
			"\tNo error-checking is performed.\n\tLoading...\n");

		status = matrix_load_binary_native( filename, 0, 0, matrix, nrows, ncols, ml );
	}

	// Input file is "non-native" binary.
	else if ( is_bin ) {

		printf("\tFile selected as (non-\"native\") binary (i.e., double-precision data and unsigned integers). Loading...\n");

		status = matrix_load_binary_verb( filename, matrix, nrows, ncols, ml );
	}

	// Input file is ASCII-text.
	else {

		printf("\tFile selected as ASCII text. Loading...\n"
			"\t\tData matrix selected as having numeric column headers: %s.\n"
			"\t\tData matrix selected as having numeric row labels: %s.\n",
			( numeric_hdrs ? "Yes" : "No" ), ( numeric_lbls ? "Yes" : "No" ) );

		status = matrix_load_ascii_verb( filename, numeric_hdrs, numeric_lbls, matrix, nrows, ncols, ml );

	} // If file is (native) binary or text.

	// -------------------------------

	return status;

} // matrix_load

//////////////////////////////////////////////////
//////////////////////////////////////////////////

/*
 * Saves a matrix to an ASCII-text file.
 * Skips name, headers and labels if 'ml' is set to NULL.
 *
 * If 'transpose' is 'true', transposes matrix in file as follows:
 * - Matrix dimensions in memory: <ncols> rows, <nrows> columns.
 * - Matrix dimensions in file: <nrows> rows, <ncols> columns.
 * - Writes <ncols> ml->headers (as column headers) and <nrows> ml->labels (as row labels).
 *
 * ncols <= padding, unless matrix transposing is set (in that case, nrows <= padding).
 *
 * Set 'append' to 'true' to append data to the file.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_save_ascii( char const *restrict filename, real const *restrict matrix, index_t nrows, index_t ncols, bool transpose, bool append,
			struct matrix_labels const *restrict ml, index_t padding )
{

	if ( ! matrix ) {
		fflush(stdout);
		errno = EFAULT;
		perror("\nmatrix_save_ascii( matrix )");
		return EXIT_FAILURE;
	}

	{
		uintmax_t const nitems = ((uintmax_t) nrows) * ((uintmax_t) ncols);

		if ( (nrows <= 0) + (ncols <= 0) + (padding <= 0) + (nitems > IDX_MAX) ) {
			fflush(stdout);
			errno = EINVAL;
			fprintf( stderr, "\nmatrix_save_ascii( rows=%" PRI_IDX ", columns=%" PRI_IDX ", padding=%" PRI_IDX " ): %s\n",
				nrows, ncols, padding, strerror(errno));
			if ( nitems > IDX_MAX )
				fprintf( stderr, "Matrix size (%" PRIuMAX ") exceeds the limits used for matrix dimensions (%" PRI_IDX ").\n",
					nitems, IDX_MAX);
			return EXIT_FAILURE;
		}
	}

	if ( transpose ) {
		if ( nrows > padding ) {
			fflush(stdout);
			errno = EINVAL;
			fprintf( stderr, "\nmatrix_save_ascii( rows=%" PRI_IDX " [number of columns since matrix transposing "
					"is selected], padding=%" PRI_IDX " ): %s\n", nrows, padding, strerror(errno));
			return EXIT_FAILURE;
		}
	} else if ( ncols > padding ) {
			fflush(stdout);
			errno = EINVAL;
			fprintf( stderr, "\nmatrix_save_ascii( columns=%" PRI_IDX ", padding=%" PRI_IDX " ): %s\n",
					nrows, padding, strerror(errno));
			return EXIT_FAILURE;
	}

	// ------------------------

	// File mode: Creates a new text file, <OR>, appends to an existing one.
	char const mode = ( append ? 'a' : 'w' );

	FILE *restrict const file = fopen( filename, &mode );
	if( ! file ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nfopen '%s' :\n\t%s.\n", filename, strerror(errno) );
		fprintf(stderr,"Error in matrix_save_ascii().\n");
		return EXIT_FAILURE;
	}

	char const *restrict name = NULL;
	char const *restrict headers = NULL;
	char const *const *restrict pheaders = NULL;
	char const *restrict labels = NULL;
	char const *const *restrict plabels = NULL;

	if ( ml ) {
		struct matrix_labels lml = *ml;
		name = lml.name;
		headers = lml.headers.tokens;
		pheaders = lml.headers.ptokens;
		labels = lml.labels.tokens;
		plabels = lml.labels.ptokens;
	}


	// Name
	if ( name ) {
		if ( fprintf(file,"%s",name) < 0 ) {
			fflush(stdout);
			fprintf(stderr,"\nInternal error in fprintf(name).\nError in matrix_save_ascii().\n");
			fclose(file);
			return EXIT_FAILURE;
		}
		if ( ( ! headers ) && (fprintf(file,"\n") <= 0) ) {
			fflush(stdout);
			fprintf(stderr, "\nInternal error in fprintf(name\\n).\nError in matrix_save_ascii().\n");
			fclose(file);
			return EXIT_FAILURE;
		}
	}


	// Column headers
	if ( headers ) {

		index_t i = 0;

		// Starts with a delimiter or not.
		if ( ! name ) {
			if ( fprintf(file,"%s",headers) < 0 ) {
				fflush(stdout);
				fprintf(stderr, "\nInternal error in fprintf(pheaders[0]).\nError in matrix_save_ascii().\n");
				fclose(file);
				return EXIT_FAILURE;
			}
			i = 1;
		}

		for ( ; i < ncols ; i++)
			if ( fprintf(file,"\t%s",pheaders[i]) <= 0 ) {
				fflush(stdout);
				fprintf(stderr, "\nInternal error in fprintf(pheaders[%" PRI_IDX "]).\n Error in matrix_save_ascii().\n", i);
				fclose(file);
				return EXIT_FAILURE;
			}

		if ( fprintf(file,"\n") <= 0 ) {
			fflush(stdout);
			fprintf(stderr, "\nInternal error in fprintf(headers\\n).\nError in matrix_save_ascii().\n");
			fclose(file);
			return EXIT_FAILURE;
		}

	} // if headers


	// ----------------------------


	// Writing file...

	// Steps for outer and inner loops.
	index_t incr_outer_loop = padding;	// Step for outer loop
	index_t incr_inner_loop = 1;		// Step for inner loop.
	if ( transpose ) {
		incr_outer_loop = 1;
		incr_inner_loop = padding;
	}

	real const *pmatrix = matrix;	// &matrix[i][0] (or &matrix[0][i] if transpose)
	for ( index_t i = 0 ; i < nrows ; i++, pmatrix += incr_outer_loop ) {

		index_t j = 0;
		real const *pmatrix_r = pmatrix; // &matrix[i][j] (or &matrix[j][i] if transpose)

		if ( labels ) {	// Writes label.

			if ( fprintf(file,"%s",plabels[i]) < 0 ) {
				fflush(stdout);
				fprintf(stderr, "\nInternal error in fprintf(plabels[%" PRI_IDX "]).\nError in matrix_save_ascii().\n", i);
				fclose(file);
				return EXIT_FAILURE;
			}

		} else { // No labels

			// First value.
			if ( fprintf( file, "%" DECIMAL_PRECISION "g", *pmatrix_r ) <= 0 ) {
				fflush(stdout);
				if ( transpose )
					fprintf(stderr, "\nInternal error in fprintf(matrix[0][%" PRI_IDX "]).\nError in matrix_save_ascii().\n",
						i);
				else
					fprintf(stderr, "\nInternal error in fprintf(matrix[%" PRI_IDX "][0]).\nError in matrix_save_ascii().\n",
						i);
				fclose(file);
				return EXIT_FAILURE;
			}
			pmatrix_r += incr_inner_loop;	// &matrix[i][1] (or &matrix[1][i] if transpose)
			j = 1;

		} // If has labels

		for ( ; j < ncols ; j++, pmatrix_r += incr_inner_loop )
			if ( fprintf(file, "\t%" DECIMAL_PRECISION "g", *pmatrix_r) <= 0 ) {
				fflush(stdout);
				if ( transpose )
					fprintf(stderr,"\nInternal error in fprintf(matrix[%" PRI_IDX "][%" PRI_IDX
							"]).\nError in matrix_save_ascii().\n", j, i);
				else
					fprintf(stderr,"\nInternal error in fprintf(matrix[%" PRI_IDX "][%" PRI_IDX
							"]).\nError in matrix_save_ascii().\n", i, j);
				fclose(file);
				return EXIT_FAILURE;
			}

		if ( fprintf(file,"\n") <= 0 ) {
			fflush(stdout);
			if ( transpose )
				fprintf(stderr,"\nInternal error in fprintf(matrix[][%" PRI_IDX "]\\n).\nError in matrix_save_ascii().\n", i);
			else
				fprintf(stderr,"\nInternal error in fprintf(matrix[%" PRI_IDX "][]\\n).\nError in matrix_save_ascii().\n", i);
			fclose(file);
			return EXIT_FAILURE;
		}

	} // for ( 0 <= i < nrows )

	if ( fclose(file) ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nfclose '%s' :\n\t%s.\n", filename, strerror(errno) );
		fprintf(stderr,"Error in matrix_save_ascii().\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // matrix_save_ascii

//////////////////////////////////////////////////

/*
 * Saves <nmatrices> <nrows>-by-<ncols> matrices to a single ASCII-text file.
 * Reads input matrices from "native" binary files (i.e., with the compiled types for matrix data and dimensions).
 * Uses the supplied labels (unless 'ml' is NULL).
 * nmatrices > 1
 *
 * WARNING: 'filename_tmp' MUST HAVE ENOUGH MEMORY AVAILABLE TO STORE ANY OF THE INPUT FILENAMES.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_save_combined_ascii( char const *restrict filename, char const *restrict input_pattern, char const *restrict output_pattern,
				index_t nmatrices, index_t nrows, index_t ncols, struct matrix_labels const *restrict ml,
				char *restrict filename_tmp )
{

	// Checks for NULL parameters
	if ( ! ( (size_t) filename * (size_t) input_pattern * (size_t) output_pattern * (size_t) ml * (size_t) filename_tmp ) ) {
		fflush(stdout);
		fprintf( stderr, "\nmatrix_save_combined_ascii():\n" );
		errno = EFAULT;
		if ( ! filename )	perror("\tfilename");
		if ( ! input_pattern )	perror("\tinput_pattern");
		if ( ! output_pattern )	perror("\toutput_pattern");
		if ( ! filename_tmp )	perror("\tfilename_tmp");
		return EXIT_FAILURE;
	}

	{
		uintmax_t const nitems = ((uintmax_t) nrows) * ((uintmax_t) ncols);
		if ( (nrows <= 0) + (ncols <= 0) + (nitems > IDX_MAX) ) {
			fflush(stdout);
			errno = EINVAL;
			fprintf( stderr, "\nmatrix_save_combined_ascii( rows=%" PRI_IDX ", columns=%" PRI_IDX " ): %s\n",
				nrows, ncols, strerror(errno));
			if ( nitems > IDX_MAX )
				fprintf( stderr, "Matrix size (%" PRIuMAX ") exceeds the limits used for matrix dimensions (%" PRI_IDX ").\n",
						nitems, IDX_MAX);
			return EXIT_FAILURE;
		}
	}

	if ( nmatrices <= 1 ) {
		fflush(stdout);
		errno = EINVAL;
		fprintf( stderr, "\nmatrix_save_combined_ascii( nmatrices=%" PRI_IDX " ): %s\n", nmatrices, strerror(errno));
		return EXIT_FAILURE;
	}


	// --------------------------

	// List of input files
	FILE *restrict *restrict const input_files = (FILE *restrict *restrict) malloc( nmatrices * sizeof(FILE *) );
	if ( ! input_files ) {
		int const err = errno; fflush(stdout); errno = err;
		perror("\nmalloc(input_files[])");
		fprintf(stderr,"Error in matrix_save_combined_ascii().\n");
		return EXIT_FAILURE;
	}

	// Opens all input files.
	for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) {
		if ( sprintf( filename_tmp, input_pattern, filename, ncols, mt ) <= 0 ) {
			int const err = errno; fflush(stdout); errno = err;
			fprintf( stderr, "\nsprintf( base_filename='%s', ncols=%" PRI_IDX ", mt=%" PRI_IDX " ): ",
				filename, ncols, mt );
			if ( err ) fprintf( stderr, "%s\n", strerror(err) );
			fprintf(stderr,"Error setting input filename in matrix_save_combined_ascii()\n");
			for( index_t i = 0 ; i < mt ; i++ ) fclose(input_files[i]);
			free((void *) input_files);
			return EXIT_FAILURE;
		}
		input_files[mt] = (FILE *restrict) fopen( filename_tmp, "rb" );	// Opens for reading in binary mode.
		if( ! input_files[mt] ) {
			int const err = errno; fflush(stdout); errno = err;
			fprintf( stderr, "\nfopen '%s' :\n\t%s.\n", filename_tmp, strerror(errno) );
			fprintf(stderr,"Error in matrix_save_combined_ascii().\n");
			for ( index_t i = 0 ; i < mt ; i++ ) fclose(input_files[i]);
			free((void *) input_files);
			return EXIT_FAILURE;
		}
		// Checks matrix dimensions.
		index_t dim[2] = { 0, 0 };
		size_t const nread = fread( dim, sizeof(index_t), 2, input_files[mt] );
		if ( (nread != 2) + ferror(input_files[mt]) + feof(input_files[mt]) + (dim[0] != nrows) + (dim[1] != ncols) ) {
			fflush(stdout);
			fprintf( stderr, "\nError reading input file %" PRI_IDX "/ %" PRI_IDX " ('%s'):", mt, nmatrices, filename_tmp );
			if ( ( nread != 2 ) + ferror(input_files[mt]) + feof(input_files[mt]) )
				fprintf( stderr, "%zu items read, 2 expected.\n", nread );
			else
				fprintf( stderr, "Invalid input matrix dimensions: %" PRI_IDX " x %" PRI_IDX " (expected: %" PRI_IDX " x %"
						PRI_IDX ").\n", dim[0], dim[1], nrows, ncols );
			fprintf( stderr, "Error in matrix_save_combined_ascii().\n" );
			for ( index_t i = 0 ; i < mt ; i++ ) fclose(input_files[i]);
			free((void *) input_files);
			return EXIT_FAILURE;
		}
	} // For all input files.

	// Opens the output file.
	if ( sprintf( filename_tmp, output_pattern, filename, ncols ) <= 0 ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf(stderr, "\nsprintf( base_filename='%s', ncols=%" PRI_IDX " ): ", filename, ncols );
		if ( err ) fprintf( stderr, "%s\n", strerror(err) );
		fprintf(stderr,"Error setting output filename in matrix_save_combined_ascii()\n");
		for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
		free((void *) input_files);
		return EXIT_FAILURE;
	}
	FILE *restrict const out_file = fopen( filename_tmp, "w" );
	if( ! out_file ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nfopen '%s' :\n\t%s.\n", filename_tmp, strerror(errno) );
		fprintf(stderr,"Error in matrix_save_combined_ascii().\n");
		for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
		free((void *) input_files);
		return EXIT_FAILURE;
	}

	// ----------------------------

	// Writes all column headers.

	char const *restrict name = NULL;
	char const *restrict headers = NULL;
	char const *const *restrict pheaders = NULL;
	char const *restrict labels = NULL;
	char const *const *restrict plabels = NULL;

	if ( ml ) {
		struct matrix_labels lml = *ml;
		name = lml.name;
		headers = lml.headers.tokens;
		pheaders = lml.headers.ptokens;
		labels = lml.labels.tokens;
		plabels = lml.labels.ptokens;
	}

	// ----------------------------

	// Name
	if ( name ) {
		if ( fprintf(out_file,"%s",name) < 0 ) {
			fflush(stdout);
			fprintf(stderr,"\nInternal error in fprintf(name).\nError in matrix_save_combined_ascii().\n");
			for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
			fclose(out_file); free((void *) input_files);
			return EXIT_FAILURE;
		}
		if ( (! headers) && (fprintf(out_file,"\n") <= 0) ) {
			fflush(stdout);
			fprintf(stderr,"\nInternal error in fprintf(name\\n).\nError in matrix_save_combined_ascii().\n");
			for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
			fclose(out_file); free((void *) input_files);
			return EXIT_FAILURE;
		}
	}

	// ----------------------------

	// Column headers
	if ( headers ) {

		index_t i = 0;

		// First column header: starts or not with a delimiter.
		if ( ! name ) {
			if ( fprintf(out_file,"%s",headers) < 0 ) {
				fflush(stdout);
				fprintf(stderr,"\nInternal error in fprintf(pheaders[0]).\nError in matrix_save_combined_ascii().\n");
				for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
				fclose(out_file); free((void *) input_files);
				return EXIT_FAILURE;
			}
			i = 1;
		}

		// Rest of headers for mt == 0 and all headers for mt > 0.
		for( index_t mt = 0 ; mt < nmatrices ; mt++ ) {
			for ( ; i < ncols ; i++ )
				if ( fprintf(out_file,"\t%s",pheaders[i]) <= 0 ) {
					fflush(stdout);
					fprintf(stderr, "\nInternal error in fprintf(pheaders_%" PRI_IDX "[%" PRI_IDX "]).\n"
							"Error in matrix_save_combined_ascii().\n", mt, i);
					for ( index_t j = 0 ; j < nmatrices ; j++ ) fclose(input_files[j]);
					fclose(out_file); free((void *) input_files);
					return EXIT_FAILURE;
				}
			i = 0;	// For mt > 0, starts at header 0.
		}

		if ( fprintf(out_file,"\n") <= 0 ) {
			fflush(stdout);
			fprintf(stderr, "\nInternal error in fprintf(headers\\n).\nError in matrix_save_combined_ascii().\n");
			for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
			fclose(out_file); free((void *) input_files);
			return EXIT_FAILURE;
		}

	} // if headers

	// ----------------------------

	// Allocates memory for one row of data.
	real *restrict data = (real *restrict) malloc( ncols * sizeof(real) );
	if ( ! data ) {
		int const err = errno; fflush(stdout); errno = err;
		perror("\nmalloc(data)");
		fprintf(stderr,"Error in matrix_save_combined_ascii().\n");
		for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
		fclose(out_file); free((void *) input_files);
		return EXIT_FAILURE;
	}

	// ----------------------------

	// Row labels
	if ( labels ) {

		for ( index_t i = 0 ; i < nrows ; i++ ) {	// for each row.

			// Writes the row label
			if ( fprintf(out_file,"%s",plabels[i]) < 0 ) {
				int const err = errno; fflush(stdout); errno = err;
				fprintf(stderr,"\nInternal error in fprintf(plabels[%" PRI_IDX
						"]).\nError in matrix_save_combined_ascii().\n", i);
				for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
				fclose(out_file); free((void *) input_files); free(data);
				return EXIT_FAILURE;
			}

			// Writes row i from each matrix.
			for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) {

				// Reads the entire row from input file <mt>.
				size_t const nread = fread( data, sizeof(real), ncols, input_files[mt] );
				if ( nread != (size_t) ncols ) {
					int const err = errno; fflush(stdout); errno = err;
					fprintf( stderr, "\nError reading input file %" PRI_IDX " (row %" PRI_IDX "): %zu items read, %"
							PRI_IDX " expected.\nError in matrix_save_combined_ascii().\n", mt, i, nread, ncols);
					for ( index_t j = 0 ; j < nmatrices ; j++ ) fclose(input_files[j]);
					fclose(out_file); free((void *) input_files); free(data);
					return EXIT_FAILURE;
				}

				// Writes that row.
				for ( index_t j = 0 ; j < ncols ; j++ )
					if ( fprintf(out_file,"\t%" DECIMAL_PRECISION "g",data[j]) <= 0 ) {
						int const err = errno; fflush(stdout); errno = err;
						fprintf( stderr, "\nInternal error in fprintf(input_matrix_%" PRI_IDX "[%" PRI_IDX "][%"
								PRI_IDX "]).\nError in matrix_save_combined_ascii().\n", mt, i, j);
						for ( index_t p = 0 ; p < nmatrices ; p++ ) fclose(input_files[p]);
						fclose(out_file); free((void *) input_files); free(data);
						return EXIT_FAILURE;
					}

			} // for each input file.

			if ( fprintf(out_file,"\n") <= 0 ) {
				int const err = errno; fflush(stdout); errno = err;
				fprintf(stderr, "\nInternal error in fprintf(input_matrix[%" PRI_IDX
						"][]\\n).\nError in matrix_save_combined_ascii().\n", i);
				for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
				fclose(out_file); free((void *) input_files); free(data);
				return EXIT_FAILURE;
			}

		} // for each row.

	} else { // No labels

		for ( index_t i = 0 ; i < nrows ; i++ ) { // for each row.

			// Writes row i from the first matrix (mt == 0)

			// Reads the entire row from input file 0.
			size_t const nread = fread( data, sizeof(real), ncols, input_files[0] );
			if ( nread != (size_t) ncols ) {
				fflush(stdout);
				fprintf(stderr,"\nError reading input file 0 (row %" PRI_IDX "): %zu items read, %" PRI_IDX
						" expected.\nError in matrix_save_combined_ascii().\n", i, nread, ncols);
				for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
				fclose(out_file); free((void *) input_files); free(data);
				return EXIT_FAILURE;
			}

			// Writes the first data in that row (without delimiter).
			if ( fprintf(out_file,"%" DECIMAL_PRECISION "g",*data) <= 0 ) {
				fflush(stdout);
				fprintf(stderr, "\nInternal error in fprintf(input_matrix_0[%" PRI_IDX
						"][0]).\nError in matrix_save_combined_ascii().\n", i);
				for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
				fclose(out_file); free((void *) input_files); free(data);
				return EXIT_FAILURE;
			}

			// Writes the rest of that row.
			for ( index_t j = 1 ; j < ncols ; j++ )
				if ( fprintf(out_file,"\t%" DECIMAL_PRECISION "g",data[j]) <= 0 ) {
					fflush(stdout);
					fprintf(stderr,"\nInternal error in fprintf(input_matrix_0[%" PRI_IDX "][%" PRI_IDX
							"]).\nError in matrix_save_combined_ascii().\n", i, j);
					for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
					fclose(out_file); free((void *) input_files); free(data);
					return EXIT_FAILURE;
				}


			// ---------------


			// Writes row i from the rest of matrices (mt > 0).

			for( index_t mt = 1 ; mt < nmatrices ; mt++ ) {

				// Reads the entire row from input file <mt>.
				size_t const nread = fread( data, sizeof(real), ncols, input_files[mt] );
				if ( nread != (size_t) ncols ) {
					fflush(stdout);
					fprintf(stderr,"\nError reading input file %" PRI_IDX " (row %" PRI_IDX "): %zu items read, %"
							PRI_IDX " expected.\nError in matrix_save_combined_ascii().\n", mt, i, nread, ncols);
					for ( index_t j = 0 ; j < nmatrices ; j++ ) fclose(input_files[j]);
					fclose(out_file); free((void *) input_files); free(data);
					return EXIT_FAILURE;
				}

				// Writes that row.
				for ( index_t j = 0 ; j < ncols ; j++ )
					if ( fprintf(out_file,"\t%" DECIMAL_PRECISION "g",data[j]) <= 0 ) {
 					fflush(stdout);
						fprintf(stderr,"\nInternal error in fprintf(input_matrix_%" PRI_IDX "[%" PRI_IDX "][%"
								PRI_IDX "]).\nError in matrix_save_combined_ascii().\n", mt, i, j);
						for ( index_t p = 0 ; p < nmatrices ; p++ ) fclose(input_files[p]);
						fclose(out_file); free((void *) input_files); free(data);
						return EXIT_FAILURE;
					}

			} // for each input file.

			if ( fprintf(out_file,"\n") <= 0 ) {
				fflush(stdout);
				fprintf(stderr, "\nInternal error in fprintf(input_matrix[%" PRI_IDX
						"][]\\n).\nError in matrix_save_combined_ascii().\n", i);
				for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
				fclose(out_file); free((void *) input_files); free(data);
				return EXIT_FAILURE;
			}

		} // for each row.

	} // if labels

	// --------------------------

	// Closes all files and cleans up.

	free(data);

	for ( index_t mt = 0 ; mt < nmatrices ; mt++ )
		fclose( input_files[mt] );
	free((void *) input_files);

	if ( fclose(out_file) ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nfclose '%s' :\n\t%s.\n", filename_tmp, strerror(errno) );
		fprintf(stderr,"Error in matrix_save_combined_ascii().\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // matrix_save_combined_ascii

//////////////////////////////////////////////////

/*
 * Saves a matrix to a (non-"native") binary file (i.e., double-precision data and unsigned int's).
 * Skips name, headers and labels if 'ml' is set to NULL.
 *
 * If 'transpose' is 'true', transposes matrix in file as follows:
 * - Matrix dimensions in memory: <ncols> rows, <nrows> columns.
 * - Matrix dimensions in file: <nrows> rows, <ncols> columns.
 * - Writes <ncols> ml->headers (as column headers) and <nrows> ml->labels (as row labels).
 *
 * ncols <= padding, unless matrix transposing is set (in that case, nrows <= padding).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_save_binary( char const *restrict filename, real const *restrict matrix, index_t nrows, index_t ncols, bool transpose,
			struct matrix_labels const *restrict ml, index_t padding )
{

	if ( ! matrix ) {
		fflush(stdout);
		errno = EFAULT;
		perror("\nmatrix_save_binary( matrix )");
		return EXIT_FAILURE;
	}

	{
		uintmax_t const nitems = ((uintmax_t) nrows) * ((uintmax_t) ncols);

		if ( (nrows <= 0) + (ncols <= 0) + (padding <= 0) + (nitems > IDX_MAX) ) {
			fflush(stdout);
			errno = EINVAL;
			fprintf( stderr, "\nmatrix_save_binary( rows=%" PRI_IDX ", columns=%" PRI_IDX ", padding=%" PRI_IDX " ): %s\n",
				nrows, ncols, padding, strerror(errno));
			if ( nitems > IDX_MAX )
				fprintf( stderr, "Matrix size (%" PRIuMAX ") exceeds the limits used for matrix dimensions (%" PRI_IDX ").\n",
					nitems, IDX_MAX);
			return EXIT_FAILURE;
		}
	}

	if ( transpose ) {
		if ( nrows > padding ) {
			fflush(stdout);
			errno = EINVAL;
			fprintf( stderr, "\nmatrix_save_binary( rows=%" PRI_IDX " [number of columns since matrix transposing "
					"is selected], padding=%" PRI_IDX " ): %s\n", nrows, padding, strerror(errno));
			return EXIT_FAILURE;
		}
	} else if ( ncols > padding ) {
		fflush(stdout);
		errno = EINVAL;
		fprintf( stderr, "\nmatrix_save_binary( columns=%" PRI_IDX ", padding=%" PRI_IDX " ): %s\n", nrows, padding, strerror(errno));
		return EXIT_FAILURE;
	}


	FILE *restrict const file = fopen( filename, "wb" );
	if ( ! file ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nfopen '%s' :\n\t%s.\n", filename, strerror(errno) );
		fprintf(stderr,"Error in matrix_save_binary().\n");
		return EXIT_FAILURE;
	}

	// ----------------------------------

	// Dimensions.
	{
		unsigned int const dims[2] = { (unsigned int) nrows, (unsigned int) ncols };
		size_t const nwritten = fwrite( dims, sizeof(int), 2, file );
		if ( nwritten != 2 ) {
			fflush(stdout);
			fprintf(stderr,"\nInternal error in fwrite while writing matrix dimensions.\nError in matrix_save_binary().\n");
			fclose(file);
			return EXIT_FAILURE;
		}
	}

	// ----------------------------------

	// Data matrix

	// Steps for outer and inner loops.
	index_t incr_outer_loop = padding;	// Step for outer loop
	index_t incr_inner_loop = 1;		// Step for inner loop.
	if ( transpose ) {
		incr_outer_loop = 1;
		incr_inner_loop = padding;
	}

	for ( index_t r=0, i=0 ; r < nrows ; r++, i += incr_outer_loop )
		for ( index_t c=0, idx=i ; c < ncols ; c++, idx += incr_inner_loop ) {

			// Writes one double-precision data
			double const value = (double) matrix[ idx ];

			size_t const nwritten = fwrite( &value, sizeof(double), 1, file );
			if ( ! nwritten ) {
				fflush(stdout);
				fprintf(stderr,"\nInternal error writing item %g at row %" PRI_IDX " (of %" PRI_IDX "), and column %"
						PRI_IDX "(of %" PRI_IDX ").\nError in matrix_save_binary().\n", value, r, nrows, c, ncols);
				fclose(file);
				return EXIT_FAILURE;
			}

		}

	// ----------------------------------

	// Returns if there is no labels.

	if ( ! ml ) {
		if ( fclose(file) ) {
			int const err = errno; fflush(stdout); errno = err;
			fprintf( stderr, "\nfclose '%s' :\n\t%s.\n", filename, strerror(errno) );
			fprintf(stderr,"Error in matrix_save_binary().\n");
			return EXIT_FAILURE;
		}
		return EXIT_SUCCESS;
	}

	struct matrix_labels lml = *ml;

	char const *restrict name = lml.name;
	char const *restrict headers = lml.headers.tokens;
	char const *const *restrict pheaders = lml.headers.ptokens;
	char const *restrict labels = lml.labels.tokens;
	char const *const *restrict plabels = lml.labels.ptokens;


	// Returns if there is no labels.
	if ( ! ( (size_t) labels + (size_t) headers + (size_t) name) ) {	// (labels == NULL) && (headers == NULL) && (name == NULL)
		if ( fclose(file) ) {
			int const err = errno; fflush(stdout); errno = err;
			fprintf( stderr, "\nfclose '%s' :\n\t%s.\n", filename, strerror(errno) );
			fprintf(stderr,"Error in matrix_save_binary().\n");
			return EXIT_FAILURE;
		}
		return EXIT_SUCCESS;
	}


	// ----------------------------------


	// Row Labels
	if ( labels ) {
		if ( fprintf(file,"%s",labels) < 0 ) {	// Saves the first label
			fflush(stdout);
			fprintf(stderr,"\nInternal error in fprintf(plabels[0]).\nError in matrix_save_binary().\n");
			fclose(file);
			return EXIT_FAILURE;
		}
		for ( index_t i=1 ; i < nrows ; i++ )
			if ( fprintf(file,"\t%s",plabels[i]) <= 0 ) {
				fflush(stdout);
				fprintf( stderr,"\nInternal error in fprintf(plabels[%" PRI_IDX "]).\nError in matrix_save_binary().\n", i);
				fclose(file);
				return EXIT_FAILURE;
			}
	} // if labels

	if ( fprintf(file,"\n")  <= 0 ) {
		fflush(stdout);
		fprintf(stderr,"\nInternal error in fprintf(plabels[]\\n).\nError in matrix_save_binary().\n");
		fclose(file);
		return EXIT_FAILURE;
	}


	// Column headers
	if ( headers ) {
		if ( fprintf(file,"%s",headers) < 0 ) {	// Saves the first header
			fflush(stdout);
			fprintf(stderr,"\nInternal error in fprintf(pheaders[0]).\nError in matrix_save_binary().\n");
			fclose(file);
			return EXIT_FAILURE;
		}
		for ( index_t j=1 ; j < ncols ; j++ )
			if ( fprintf(file,"\t%s",pheaders[j])  <= 0 ) {
				fflush(stdout);
				fprintf(stderr,"\nInternal error in fprintf(pheaders[%" PRI_IDX "]).\nError in matrix_save_binary().\n",j);
				fclose(file);
				return EXIT_FAILURE;
			}
	} // if headers

	if ( fprintf(file,"\n")  <= 0 ) {
		fflush(stdout);
		fprintf(stderr,"\nInternal error in fprintf(pheaders[]\\n).\nError in matrix_save_binary().\n");
		fclose(file);
		return EXIT_FAILURE;
	}


	// Name
	if ( name && (fprintf(file,"%s",name) < 0) ) {
		fflush(stdout);
		fprintf(stderr,"\nInternal error in fprintf(name).\nError in matrix_save_binary().\n");
		fclose(file);
		return EXIT_FAILURE;
	}


	if ( fclose(file) ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nfclose '%s' :\n\t%s.\n", filename, strerror(errno) );
		fprintf(stderr,"Error in matrix_save_binary().\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // matrix_save_binary

//////////////////////////////////////////////////
//////////////////////////////////////////////////

/*
 * Saves a matrix to a "native" binary file (i.e., with the compiled types for matrix data and dimensions).
 * Skips name, headers and labels if 'ml' is set to NULL.
 *
 * WARNING: For internal use only. No error checking is performed.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_save_binary_native( char const *restrict filename, real const *restrict matrix, index_t nrows, index_t ncols,
				struct matrix_labels const *restrict ml )
{

	if ( ! matrix ) {
		fflush(stdout);
		errno = EFAULT;
		perror("\nmatrix_save_binary_native( matrix )");
		return EXIT_FAILURE;
	}

	if ( (nrows <= 0) + (ncols <= 0) ) {
		fflush(stdout);
		errno = EINVAL;
		fprintf( stderr, "\nmatrix_save_binary_native( rows=%" PRI_IDX ", columns=%" PRI_IDX " ): %s\n",
			nrows, ncols, strerror(errno));
		return EXIT_FAILURE;
	}


	FILE *restrict const file = fopen( filename, "wb" );
	if( ! file ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nfopen '%s' :\n\t%s.\n", filename, strerror(errno) );
		fprintf(stderr,"Error in matrix_save_binary_native().\n");
		return EXIT_FAILURE;
	}


	// ----------------------------------


	// Matrix dimensions.
	{
		const index_t dim[2] = { nrows, ncols };
		size_t const nwritten = fwrite( dim, sizeof(index_t), 2, file );
		if ( nwritten != 2 ) {
			fflush(stdout);
			fprintf(stderr,"\nInternal error in fwrite writing matrix dimensions.\nError in matrix_save_binary_native().\n");
			fclose(file);
			return EXIT_FAILURE;
		}
	}

	// Data matrix
	size_t const nitems = nrows * ncols;
	size_t const nwritten = fwrite( matrix, sizeof(real), nitems, file );
	if ( nwritten != nitems ) {
		fflush(stdout);
		fprintf(stderr,"\nInternal error in function fwrite: %zu items read, %zu expected\n"
				"Error in matrix_save_binary_native().\n",nwritten,nitems);
		fclose(file);
		return EXIT_FAILURE;
	}

	// ----------------------------------


	// Returns if there are no matrix labels.
	if ( ! ml ) {
		if ( fclose(file) ) {
			int const err = errno; fflush(stdout); errno = err;
			fprintf( stderr, "\nfclose '%s' :\n\t%s.\n", filename, strerror(errno) );
			fprintf(stderr,"Error in matrix_save_binary_native().\n");
			return EXIT_FAILURE;
		}
		return EXIT_SUCCESS;
	}

	struct matrix_labels lml = *ml;

	char const *restrict name = lml.name;
	char const *restrict headers = lml.headers.tokens;
	char const *const *restrict pheaders = lml.headers.ptokens;
	char const *restrict labels = lml.labels.tokens;
	char const *const *restrict plabels = lml.labels.ptokens;

	if ( ! ( (size_t) labels + (size_t) headers + (size_t) name) ) {	// (labels == NULL) && (headers == NULL) && (name == NULL)
		if ( fclose(file) ) {
			int const err = errno; fflush(stdout); errno = err;
			fprintf( stderr, "\nfclose '%s' :\n\t%s.\n", filename, strerror(errno) );
			fprintf(stderr,"Error in matrix_save_binary_native().\n");
			return EXIT_FAILURE;
		}
		return EXIT_SUCCESS;
	}

	// ----------------------------------

	// Row Labels
	if ( labels ) {
		if ( fprintf(file,"%s",labels) < 0 ) {	// Saves the first label
			fflush(stdout);
			fprintf(stderr,"\nInternal error in fprintf(plabels[0]).\nError in matrix_save_binary_native().\n");
			fclose(file);
			return EXIT_FAILURE;
		}
		for (index_t i=1 ; i<nrows ; i++)
			if ( fprintf(file,"\t%s",plabels[i]) <= 0 ) {
				fflush(stdout);
				fprintf( stderr,"\nInternal error in fprintf(plabels[%" PRI_IDX
						"]).\nError in matrix_save_binary_native().\n",i);
				fclose(file);
				return EXIT_FAILURE;
			}
	} // if labels

	if ( fprintf(file,"\n")  <= 0 ) {
		fflush(stdout);
		fprintf(stderr,"\nInternal error in fprintf(plabels[]\\n).\nError in matrix_save_binary_native().\n");
		fclose(file);
		return EXIT_FAILURE;
	}

fflush(NULL); printf("MARK2\n"); fflush(NULL);


	// Column headers
	if ( headers ) {
		if ( fprintf(file,"%s",headers) < 0 ) {	// Saves the first header
			fflush(stdout);
			fprintf(stderr,"\nInternal error in fprintf(pheaders[0]).\nError in matrix_save_binary_native().\n");
			fclose(file);
			return EXIT_FAILURE;
		}
		for (index_t j=1; j<ncols; j++)
			if ( fprintf(file,"\t%s",pheaders[j])  <= 0 ) {
				fflush(stdout);
				fprintf(stderr,"\nInternal error in fprintf(pheaders[%" PRI_IDX
						"]).\nError in matrix_save_binary_native().\n",j);
				fclose(file);
				return EXIT_FAILURE;
			}
	} // if headers

	if ( fprintf(file,"\n")  <= 0 ) {
		fflush(stdout);
		fprintf(stderr,"\nInternal error in fprintf(pheaders[]\\n).\nError in matrix_save_binary_native().\n");
		fclose(file);
		return EXIT_FAILURE;
	}


	// Name
	if ( name && fprintf(file,"%s",name)  < 0 ) {
		fflush(stdout);
		fprintf(stderr,"\nInternal error in fprintf(name).\nError in matrix_save_binary_native().\n");
		fclose(file);
		return EXIT_FAILURE;
	}


	if ( fclose(file) ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nfclose '%s' :\n\t%s.\n", filename, strerror(errno) );
		fprintf(stderr,"Error in matrix_save_binary_native().\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // matrix_save_binary_native

//////////////////////////////////////////////////

/*
 * Writes matrix to a file according to the selected file format
 * Skips name, headers and labels if 'ml' is set to NULL.
 *
 * save_bin: Saves output matrix to a binary file.
 *		== 0: Disabled. Saves file as ASCII text.
 *		== 1: Uses "non-native" format (i.e., double-precision data, and "unsigned int" for dimensions).
 *		 > 1: Uses "native" or raw format (i.e., the compiled types for matrix data and dimensions).
 *
 * If 'transpose' is 'true', and save_bin <= 1, transposes matrix in file as follows:
 * - Matrix dimensions in memory: <ncols> rows, <nrows> columns.
 * - Matrix dimensions in file: <nrows> rows, <ncols> columns.
 * - Writes <ncols> ml->headers (as column headers) and <nrows> ml->labels (as row labels).
 *
 * ncols <= padding, unless matrix transposing is set (in that case, nrows <= padding).
 *
 * WARNING:
 *	"Native" mode (i.e., save_bin > 1) skips ALL data transformation (matrix transposing, padding, etc).
 *	All related arguments are ignored. The file is saved in raw format.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_save( char const *restrict filename, index_t save_bin, real *restrict matrix, index_t nrows, index_t ncols, bool transpose,
		struct matrix_labels const *restrict ml, index_t padding )
{

	// Checks for NULL parameters
	if ( ! ( (size_t) filename * (size_t) matrix) ) {
		fflush(stdout);
		errno = EFAULT;
		if ( ! filename ) perror("\nmatrix_save( filename )");
		if ( ! matrix )   perror("\nmatrix_save( matrix )");
		return EXIT_FAILURE;
	}

	if ( (nrows <= 0) + (ncols <= 0) + (save_bin < 0) ) {
		fflush(stdout);
		errno = EINVAL;
		fprintf( stderr, "\nmatrix_save( rows=%" PRI_IDX ", columns=%" PRI_IDX ", save_bin=%" PRI_IDX " ): %s\n",
			nrows, ncols, save_bin, strerror(errno));
		return EXIT_FAILURE;
	}

	int status = EXIT_SUCCESS;

	// -------------------------------

	printf("\nSaving output file as ");

	// Saves output as "native" binary.
	if ( save_bin > 1 ) {
		printf("\"native\" binary (i.e., raw format)...\n");
		if ( padding + transpose )
			printf("\tSkipping all transformation options (matrix transposing, padding, etc.)...\n");
		status = matrix_save_binary_native( filename, matrix, nrows, ncols, ml );
	}

	// Saves output as (non-"native") binary.
	else if ( save_bin ) {
		printf("(non-\"native\") binary (i.e., double-precision data and unsigned integers)...\n");
		status = matrix_save_binary( filename, matrix, nrows, ncols, transpose, ml, padding );
	}

	// Saves output as ASCII text.
	else {
		printf("ASCII text...\n");
		status = matrix_save_ascii( filename, matrix, nrows, ncols, transpose, false, ml, padding );
	}

	// ----------------------------------------

	return status;

} // matrix_save

//////////////////////////////////////////////////
//////////////////////////////////////////////////

/*
 * Transposes matrix using a temporary "native" binary file (i.e., with the compiled types for matrix data and dimensions).
 *
 * <base_filename> is used only if no temporary file from the system can be employed.
 *
 * WARNING:
 *	No error checking is performed.
 *	"matrix" is always changed (even on error).
 */
int matrix_transpose_file( real *restrict matrix, index_t *restrict nrows, index_t *restrict ncols, char const *restrict const base_filename )
{

	// Checks for NULL parameters
	if ( ! ( (size_t) matrix * (size_t) nrows * (size_t) ncols ) ) {
		fflush( stdout );
		fprintf( stderr, "\nmatrix_transpose_file():\n" );
		errno = EFAULT;
		if ( ! matrix ) perror("\tmatrix");
		if ( ! nrows  ) perror("\tnrows" );
		if ( ! ncols  ) perror("\tncols" );
		return EXIT_FAILURE;
	}

	if ( (*nrows <= 0) + (*ncols <= 0) ) {
		fflush(stdout);
		errno = EINVAL;
		fprintf( stderr, "\nmatrix_transpose_file( rows=%" PRI_IDX ", columns=%" PRI_IDX " ): %s\n",
			*nrows, *ncols, strerror(errno));
		return EXIT_FAILURE;
	}

	// ---------------------------

	// Writes matrix by columns to a temporary file.
	bool custom_file = false;		// Uses a custom file instead of one generated by the system.
	char *restrict filename_tmp = NULL;	// Custom filename

	FILE *restrict file = tmpfile();
	if ( ! file ) {

		// Uses a custom file.
		custom_file = true;

		if ( ! base_filename ) {
			fflush( stdout );
			errno = EFAULT;
			perror( "\nmatrix_transpose_file():\n\tbase_filename" );
			return EXIT_FAILURE;
		}

		size_t const len_filename = strlen(base_filename);
		filename_tmp = (char *restrict) malloc( (len_filename + 8) * sizeof(char) );
		if ( ! filename_tmp ) {
			int const err = errno; fflush(stdout); errno = err;
			perror("\nmalloc( filename_tmp )");
			fprintf(stderr,"Error in matrix_transpose_file().\n");
			return EXIT_FAILURE;
		}
		sprintf(filename_tmp, "%s_t.dat", base_filename);

		file = fopen( filename_tmp, "w+b" );	// Open for reading and writing.
		if ( ! file ) {
			int const err = errno; fflush(stdout); errno = err;
			fprintf( stderr, "\nfopen '%s' :\n\t%s.\n", filename_tmp, strerror(errno) );
			fprintf(stderr,"Error in matrix_transpose_file().\n");
			free(filename_tmp);
			return EXIT_FAILURE;
		}

	}

	// ---------------------------

	index_t const numrows = *nrows;
	index_t const numcols = *ncols;

	for ( index_t j=0 ; j<numcols ; j++ ) {
		for ( index_t i=0, idx=j ; i<numrows ; i++, idx+=numcols )
			if ( ! fwrite( &matrix[ idx ], sizeof(real), 1, file ) ) {
				fflush(stdout);
				fprintf(stderr, "\nInternal error in fwrite(temporary_file[ %" PRI_IDX " / %" PRI_IDX "][ %" PRI_IDX " / %"
						PRI_IDX "]).\nError in matrix_transpose_file().\n", j, numcols, i, numrows);
				fclose(file);
				if ( custom_file ) { unlink(filename_tmp); free(filename_tmp); }
				return EXIT_FAILURE;
			}
	} // for j.

	// Now, reads the file.
	rewind(file);

	size_t const nitems = numrows * numcols;
	size_t const nread = fread( matrix, sizeof(real), nitems, file );
	if ( nread != nitems ) {
		fflush(stdout);
		if ( feof(file) )
			fprintf(stderr,"\nfread(): premature end-of-file in temporary file.\nError in matrix_transpose_file().\n");
		else // error
			fprintf(stderr,"\nInternal error in fread( temporary_file ).\nError in matrix_transpose_file().\n");
		fclose(file);
		if ( custom_file ) { unlink(filename_tmp); free(filename_tmp); }
		return EXIT_FAILURE;
	}

	fclose(file);
	if ( custom_file ) { unlink(filename_tmp); free(filename_tmp); }

	*nrows = numcols;
	*ncols = numrows;

	return EXIT_SUCCESS;

} // matrix_transpose_file

//////////////////////////////////////////////////
//////////////////////////////////////////////////

/*
 * Cleans name, headers, labels and matrix.
 *
 * WARNING: This method uses the regular free(3) function.
 */
void matrix_clean( real *restrict matrix, struct matrix_labels ml )
{

	clean_matrix_labels( ml );

	if ( matrix )
		free( (void *) matrix );

} // matrix_clean

//////////////////////////////////////////////////

/*
 * Shows matrix's content (data, name, headers and/or labels).
 *
 * If 'transpose' is 'true', transposes matrix as follows:
 * - Matrix dimensions in memory: <numcols> rows, <numrows> columns.
 * - Matrix dimensions on screen: <numrows> rows, <numcols> columns.
 * - Shows <numcols> ml->headers (as column headers) and <numrows> ml->labels (as row labels).
 *
 * numcols <= padding, unless matrix transposing is set (in that case, numrows <= padding).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int matrix_show( real const *restrict matrix, index_t numrows, index_t numcols, index_t padding, bool transpose,
		struct matrix_labels const *restrict ml )
{

	if ( ! matrix ) {
		fflush(stdout);
		errno = EFAULT;
		perror("\nmatrix_show( matrix )");
		return EXIT_FAILURE;
	}

	{
		uintmax_t const nitems = ((uintmax_t) numrows) * ((uintmax_t) numcols);

		if ( (numrows <= 0) + (numcols <= 0) + (padding <= 0) + (nitems > IDX_MAX) ) {
			fflush(stdout);
			errno = EINVAL;
			fprintf( stderr, "\nmatrix_show( rows=%" PRI_IDX ", columns=%" PRI_IDX ", padding=%" PRI_IDX " ): %s\n",
				numrows, numcols, padding, strerror(errno));
			if ( nitems > IDX_MAX )
				fprintf( stderr, "Matrix size (%" PRIuMAX ") exceeds the limits used for matrix dimensions (%" PRI_IDX ").\n",
					nitems, IDX_MAX);
			return EXIT_FAILURE;
		}
	}

	if ( transpose ) {
		if ( numrows > padding ) {
			fflush(stdout);
			errno = EINVAL;
			fprintf( stderr, "\nmatrix_show( rows=%" PRI_IDX " [number of columns since matrix transposing is selected], "
					"padding=%" PRI_IDX " ): %s\n", numrows, padding, strerror(errno));
			return EXIT_FAILURE;
		}
	} else if ( numcols > padding ) {
			fflush(stdout);
			errno = EINVAL;
			fprintf( stderr, "\nmatrix_show( columns=%" PRI_IDX ", padding=%" PRI_IDX " ): %s\n",
				numrows, padding, strerror(errno));
			return EXIT_FAILURE;
	}

	index_t nrows, ncols;

	#if NMFGPU_TESTING
		ncols = numcols;
		nrows = numrows;
	#else
		if ( numrows == 1 ) {	// 'matrix' is a vector.
			ncols = MIN( numcols, 225 ) ;
			nrows = 1;
		} else {
			ncols = MIN( numcols, 15 ) ;
			nrows = MIN( numrows, 9 ) ;
		}
	#endif


	// --------------------------------


	char const *restrict name = NULL;
	char const *restrict headers = NULL;
	char const *const *restrict pheaders = NULL;
	char const *restrict labels = NULL;
	char const *const *restrict plabels = NULL;

	if ( ml ) {
		struct matrix_labels lml = *ml;
		name = lml.name;
		headers = lml.headers.tokens;
		pheaders = lml.headers.ptokens;
		labels = lml.labels.tokens;
		plabels = lml.labels.ptokens;
	}


	// Name
	if ( name )
		printf("Name: '%s'\n\n", name);


	// Column headers
	if ( headers ) {
		printf( "Headers (%" PRI_IDX "):\n", numcols );
		for( index_t i=0 ; i<ncols ; i++ )
			printf("'%s' ", pheaders[i] );
		if ( ncols < numcols )
			printf(" ... '%s'", pheaders[numcols-1]);
		printf("\n\n");
	}


	// ----------------------------

	// Show matrix (with row labels if exist)

	if ( ((size_t) ml) + transpose ) {	// (ml != NULL) || (transpose == true)
		printf("Data matrix: %" PRI_IDX " rows x %" PRI_IDX " columns.\n", numrows, numcols );
		fflush( stdout );
	}

	// Steps for outer and inner loops.
	index_t incr_outer_loop = padding;	// Step for outer loop
	index_t incr_inner_loop = 1;		// Step for inner loop.
	if ( transpose ) {
		incr_outer_loop = 1;
		incr_inner_loop = padding;
	}

	real const *pmatrix = matrix;	// &matrix[i][0] (or &matrix[0][i] if transpose)
	for ( index_t i=0 ; i<nrows ; i++,pmatrix+=incr_outer_loop ) {

		index_t j = 0;
		real const *pmatrix_r = pmatrix; // &matrix[i][j] (or &matrix[j][i] if transpose)

		printf( "Line %" PRI_IDX ": ", i );

		if ( labels )
			printf( "%s", plabels[i] );

		else { // No labels. Writes the first value.

			printf( "%g", *pmatrix_r );
			pmatrix_r += incr_inner_loop;	// &matrix[i][1] (or &matrix[1][i] if transpose)
			j = 1;

		} // If has labels

		// Rest of values.
		for ( ; j<ncols ; j++,pmatrix_r+=incr_inner_loop )
			printf( " %g", *pmatrix_r );

		if ( ncols < numcols )
			printf( " ... %g", pmatrix[ (numcols - 1) * incr_inner_loop ] ); // NOTE: we use 'pmatrix', not 'pmatrix_r'

		printf( "\n" );

	} // for ( 0 <= i < nrows )

	// Last row.
	if ( nrows < numrows ) {

		index_t i = numrows - 1;
		pmatrix = matrix + i * incr_outer_loop;

		index_t j = 0;
		real const *pmatrix_r = pmatrix; // &matrix[i][j] (or &matrix[j][i] if transpose)

		printf( "...\nLine %" PRI_IDX ": ", i );

		if ( labels )
			printf( "%s", plabels[i] );

		else { // No labels. Writes the first value.

			printf( "%g", *pmatrix_r );
			pmatrix_r += incr_inner_loop;	// &matrix[i][1] (or &matrix[1][i] if transpose)
			j = 1;

		} // If has labels

		// Rest of values.
		for ( ; j<ncols ; j++,pmatrix_r+=incr_inner_loop )
			printf( " %g", *pmatrix_r );

		if ( ncols < numcols )
			printf( " ... %g", pmatrix[ (numcols - 1) * incr_inner_loop ] ); // NOTE: we use 'pmatrix', not 'pmatrix_r'

		printf( "\n" );

	} // if ( nrows < numrows )

	printf( "\n" );

	fflush( stdout );

	return EXIT_SUCCESS;

} // matrix_show

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Loads an integer matrix from an ASCII file.
 * Skips name, headers and labels if 'ml' is set to NULL.
 *
 * WARNING: 'matrix' must be a non-NULL pointer with ENOUGH MEMORY ALREADY ALLOCATED.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_int_load_ascii( char const *restrict filename, index_t nrows, index_t ncols, bool hasname, bool hasheaders, bool haslabels,
			index_t *restrict matrix, struct matrix_labels *restrict ml )
{

	if ( ! matrix ) {
		fflush(stdout);
		errno = EFAULT;
		perror("\nmatrix_int_load_ascii( matrix )");
		return EXIT_FAILURE;
	}

	{
		uintmax_t const nitems = ((uintmax_t) nrows) * ((uintmax_t) ncols);

		if ( (nrows <= 0) + (ncols <= 0) + (nitems > IDX_MAX) ) {
			fflush(stdout);
			errno = EINVAL;
			fprintf( stderr, "\nmatrix_int_load_ascii( rows=%" PRI_IDX ", columns=%" PRI_IDX " ): %s\n",
				nrows, ncols, strerror(errno));
			if ( nitems > IDX_MAX )
				fprintf( stderr, "Matrix size (%" PRIuMAX ") exceeds the limits used for matrix dimensions (%" PRI_IDX ").\n",
					nitems, IDX_MAX);
			return EXIT_FAILURE;
		}
	}


	// Starts Reading ...
	FILE *restrict const file = fopen( filename, "r" );
	if( ! file ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nfopen '%s' :\n\t%s.\n", filename, strerror(errno) );
		fprintf(stderr,"Error in matrix_int_load_ascii().\n");
		return EXIT_FAILURE;
	}

	char *restrict name = NULL;
	char *restrict headers = NULL;
	char **restrict pheaders = NULL;
	char *restrict labels = NULL;
	char **restrict plabels = NULL;

	// Skips name, headers and labels if ml is NULL.
	bool const skip = ! ml;	// ( ml == NULL )


	// ----------------------------


	// Name or headers.

	if ( hasname + hasheaders ) {

		char *restrict data = NULL;
		size_t len_data = 0;

		// Reads line 1.
		len_data = read_line( file, &data );
		if ( ! data ) {
			if ( ! len_data )
				fprintf(stderr,"\nError reading input file: file is empty.\n");
			fprintf(stderr,"Error in matrix_int_load_ascii().\n");
			fclose(file);
			return EXIT_FAILURE;
		}

		if ( skip )
			free(data);

		else if ( hasheaders ) {
			// "Tokenizes" line 1.
			char **restrict pdata = NULL;
			if ( ! tokenize( data, &pdata, (int) '\t' ) ) {
				fprintf(stderr,"Error tokenizing line 1.\nError in matrix_int_load_ascii().\n");
				free(data);
				fclose(file);
				return EXIT_FAILURE;
			}

			if ( hasname ) { // Name and headers

				// Splits line 1 into name and headers.

				// Name: copies the first token.
				name = (char *restrict) malloc( (strlen(data) + 1) * sizeof(char) );
				if ( ! name ) {
					int const err = errno; fflush(stdout); errno = err;
					perror("\nmalloc( name )");
					fprintf(stderr,"Error in matrix_int_load_ascii().\n");
					free(data); free(pdata);
					fclose(file);
					return EXIT_FAILURE;
				}
				strcpy( name, data );

				/* Headers: Sets remaining tokens, starting from the second one, as column headers.
				 *
				 *	Instead of allocating memory and copying 'data' to 'headers' (starting from the second token), it just
				 *	moves that second token to the "beginning" of 'data' (i.e., to the address returned by the read_line()
				 *	call above) and overwrites the first token (already copied to 'name'). Remaining tokens are kept
				 *	untouched, and the previous place of the second token is left as "garbage".
				 *
				 *	This way data == headers == pheaders[0], and it is possible to call free(headers).
				 */
				char **p_pdata = pdata + 1;	// Second token.
				char *p = *p_pdata;
				headers = memmove( data, p, (strlen(p) + 1) * sizeof(char) ); // Overwrites first token.
				if ( ! headers )  {
					int const err = errno; fflush(stdout); errno = err;
					perror("\nmemmove( headers )");
					fprintf(stderr,"Error in matrix_int_load_ascii().\n");
					free(name);
					free(pdata); free(data);
					fclose(file);
					return EXIT_FAILURE;
				}

				// pheaders: Copies pdata[i+1] to pheaders[i]
				pheaders = (char **restrict) malloc( ncols * sizeof(char *) );
				if ( ! pheaders ) {
					int const err = errno; fflush(stdout); errno = err;
					perror("\nmalloc( pheaders )");
					fprintf(stderr,"Error in matrix_int_load_ascii().\n");
					free(name);
					free(pdata); free(data);
					fclose(file);
					return EXIT_FAILURE;
				}
				memcpy( pheaders, p_pdata, ncols * sizeof(char *) ); // pheaders[i] = pdata[i+1]
				free(pdata);

			} else { // No name. Headers only.
				headers = data;
				pheaders = pdata;
			}

		} else // No headers. Name only.
			name = data;

	} // if has name or headers


	// ----------------------------


	// Labels

	size_t max_len_labels = 0;
	size_t len_labels = 0;
	char *p_labels = NULL;
	char **p_plabels = NULL;

	if ( haslabels * ( ! skip ) ) {
		max_len_labels = 64 * nrows;				// Initial size for <nrows> labels of 64 characters each.
		labels = (char *restrict) malloc( max_len_labels * sizeof(char) );	// Memory size will be adjusted later.
		if ( ! labels ) {
			int const err = errno; fflush(stdout); errno = err;
			perror("\nmalloc( labels )");
			fprintf(stderr,"Error in matrix_int_load_ascii().\n");
			if ( headers ) { free(pheaders); free(headers); }
			if ( name ) free(name);
			fclose(file);
			return EXIT_FAILURE;
		}
		p_labels = labels;

		plabels = (char **restrict) malloc( nrows * sizeof(char *) );
		if ( ! plabels ) {
			int const err = errno; fflush(stdout); errno = err;
			perror("\nmalloc( plabels )");
			fprintf(stderr,"Error in matrix_int_load_ascii().\n");
			free(labels);
			if ( headers ) { free(pheaders); free(headers); }
			if ( name ) free(name);
			fclose(file);
			return EXIT_FAILURE;
		}
		p_plabels = plabels;
	} // if has labels.


	// ----------------------------


	// Reading file...

	for ( index_t r=0, idx=0 ; r<nrows ; r++ ) {

		if ( haslabels ) {
			// Reads a token.
			char *restrict data = NULL;
			int last_char = 0;	// Last char read.
			size_t const len_data = read_token( file, (int) '\t', &data, &last_char );
			if ( ! data ) {
				if ( len_data )
					fprintf(stderr,"Error in matrix_int_load_ascii().\n");
				else { // EOF
					fflush(stdout);
					fprintf(stderr, "\nError reading label for matrix row %" PRI_IDX ": premature end of file.\n", r );
				}
				struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
				clean_matrix_labels(l_ml);
				fclose(file);
				return EXIT_FAILURE;
			}
			if ( !skip ) {
				// Before setting the label, checks if there is enough memory.
				size_t len = len_labels + len_data + 1;
				if ( len > max_len_labels ) { // Allocates more memory
					do {
						max_len_labels *= 2;
					} while ( len >= max_len_labels );
					char *const tmp = (char *) realloc( labels, max_len_labels * sizeof(char) );
					if ( ! tmp ) {
						int const err = errno; fflush(stdout); errno = err;
						perror("\nrealloc( labels )");
						fprintf( stderr, "Error in matrix_int_load_ascii().\n" );
						struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
						clean_matrix_labels(l_ml);
						fclose(file);
						return EXIT_FAILURE;
					}
					labels = tmp;
					p_labels = tmp + len_labels; // Pointer to place for new label.

					// Resets 'plabels'.
					retok( (char const *)labels, (char const **)plabels, r );
					p_plabels = plabels + r;	// Pointer to place for new label.

				} // If allocate more memory for labels

				// Sets vector of pointers to labels
				strcpy(p_labels, data);		// Copies the token.
				*p_plabels = p_labels;		// Sets p_plabels[r].

				// pointer to place for next label
				p_labels += (len_data + 1);
				p_plabels++;

				len_labels = len;	// len == (len_labels + len_data + 1)

			} // If set the label.

			free(data);

		} // if haslabels

		#if NMFGPU_DEBUG_READ_MATRIX
			printf("Reading row %" PRI_IDX "...\n",r);
			fflush(stdout);
		#endif

		for ( index_t col=0 ; col<ncols ; col++, idx++ ) {

			int const conv = fscanf( file, "%" PRI_IDX " ", &matrix[ idx ] );
			if ( conv < 1 ) {
				int const err = errno; fflush(stdout); errno = err;
				if ( ferror(file) ) {	// Error.
					perror("\nfscanf()");
					fprintf( stderr, "Error in matrix_int_load_ascii() reading row %" PRI_IDX ", column %" PRI_IDX "\n",
						r, col);
				}
				else if ( (conv == EOF) + feof(file) ) // Premature EOF.
					fprintf( stderr, "\nError reading input file:\nPremature end of file detected in row %" PRI_IDX ".\n"
						"Invalid file format.\n\n", r );
				else	// Not a number
					fprintf( stderr, "\nError reading input file:\nRow %" PRI_IDX ", column %" PRI_IDX
							": Invalid numeric format.\nThe file contains non-numeric data.\n\n",r,col);
				struct matrix_labels l_ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );
				clean_matrix_labels(l_ml);
				fclose(file);
				return EXIT_FAILURE;
			}

		} // for (0 <= col < ncols-1)

	} // for ( 0 <= r < nrows )

	// Resizes labels.
	if ( haslabels * (! skip) ) {
		#if NMFGPU_DEBUG_READ_MATRIX
			printf("Resizing labels from %zu to %zu\n",max_len_labels,len_labels);
			fflush(stdout);
		#endif
		labels = (char *) realloc( labels, len_labels * sizeof(char) );
	}

	fclose(file);

	// Sets output values.
	if ( ! skip )
		*ml = NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );

	return EXIT_SUCCESS;

} // matrix_int_load_ascii

//////////////////////////////////////////////////

/*
 * Loads an integer matrix from a binary file.
 * Skips name, headers and labels if 'ml' is set to NULL.
 *
 * WARNING:
 *	For internal use only. NO ERROR CHECKING IS PERFORMED.
 *	'matrix' must be a non-NULL pointer with ENOUGH MEMORY ALREADY ALLOCATED.
 *
 * Returns EXIT_SUCCESS (or EXIT_FAILURE if could not open filename).
 */
int matrix_int_load_binary( char const *restrict filename, index_t *restrict matrix, index_t nrows, index_t ncols,
				struct matrix_labels *restrict ml )
{

	if ( ! matrix ) {
		fflush(stdout);
		errno = EFAULT;
		perror("\nmatrix_int_load_binary( matrix )");
		return EXIT_FAILURE;
	}

	{
		uintmax_t const nitems = ((uintmax_t) nrows) * ((uintmax_t) ncols);

		if ( (nrows <= 0) + (ncols <= 0) + (nitems > IDX_MAX) ) {
			fflush(stdout);
			errno = EINVAL;
			fprintf( stderr, "\nmatrix_int_load_binary( rows=%" PRI_IDX ", columns=%" PRI_IDX " ): %s\n",
				nrows, ncols, strerror(errno));
			if ( nitems > IDX_MAX )
				fprintf( stderr, "Matrix size (%" PRIuMAX ") exceeds the limits used for matrix dimensions (%" PRI_IDX ").\n",
					nitems, IDX_MAX);
			return EXIT_FAILURE;
		}
	}

	// Starts Reading ...
	FILE *restrict const file = fopen( filename, "rb" );
	if( ! file ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nfopen '%s' :\n\t%s.\n", filename, strerror(errno) );
		fprintf(stderr,"Error in matrix_int_load_binary().\n");
		return EXIT_FAILURE;
	}

	// Reads data matrix
	size_t const nitems = nrows * ncols;
	size_t const nread = fread( matrix, sizeof(index_t), nitems, file );
	if ( nread != nitems ) {
		fflush(stdout);
		fprintf(stderr, "\nError reading input file: %zu items read, %zu expected.\nError in matrix_int_load_binary().\n",
			nread, nitems);
		fclose(file);
		return EXIT_FAILURE;
	}

	/////////////////////////////////


	// Reads headers, labels and name (as plain text) if they exists.

	// Skips them if ml is set to NULL
	if ( ! ml ) {
		fclose(file);
		return EXIT_SUCCESS;
	}

	char *restrict name = NULL;
	char *restrict headers = NULL;
	char **restrict pheaders = NULL;
	char *restrict labels = NULL;
	char **restrict plabels = NULL;


	// Checks for row labels
	size_t len = read_line( file, &labels );
	if ( ! labels ) {
		fclose(file);
		if ( len ) { // Error
			fprintf(stderr, "Error reading row labels.\nError in matrix_int_load_binary().\n");
			return EXIT_FAILURE;
		}
		*ml = NEW_MATRIX_LABELS( NULL, NULL, NULL, NULL, NULL );
		return EXIT_SUCCESS;
	}
	if ( len ) {
		// Divides into tokens by replacing all tabs characters by '\0'.
		if ( ! tokenize( labels, &plabels, (int) '\t' ) ) {
			fprintf(stderr, "Error reading row labels.\nError in matrix_int_load_binary().\n");
			fclose(file);
			free(labels);
			return EXIT_FAILURE;
		}
	}
	else {	// No labels.
		free(labels);
		labels = NULL;
	}


	// Checks for columns headers
	len = read_line( file, &headers );
	if ( ! headers ) {
		fclose(file);
		if ( len ) {
			fprintf(stderr,
				"Error reading column headers.\nError in matrix_int_load_binary().\n");
			if ( labels ) { free(plabels); free(labels); }
			return EXIT_FAILURE;
		}
		*ml = NEW_MATRIX_LABELS( NULL, NULL, NULL, labels, plabels );
		return EXIT_SUCCESS;
	}
	if ( len ) {
		// Divides into tokens by replacing all tabs characters by '\0'.
		if ( ! tokenize( headers, &pheaders, (int) '\t' ) ) {
			fprintf(stderr, "Error reading column headers.\nError in matrix_int_load_binary().\n");
			fclose(file);
			free(headers);
			if ( labels ) { free(plabels); free(labels); }
			return EXIT_FAILURE;
		}
	}
	else{	// No headers
		free(headers);
		headers = NULL;
	}


	// Checks for name.
	len = read_token( file, (int) '\t', &name, NULL );

	fclose(file);

	if ( (! name) * len ) {
		fprintf(stderr, "Error reading description string.\nError in matrix_int_load_binary().\n");
		struct matrix_labels l_ml = NEW_MATRIX_LABELS( NULL, headers, pheaders, labels, plabels );
		clean_matrix_labels(l_ml);
		return EXIT_FAILURE;
	} // if has name

	*ml = (struct matrix_labels) NEW_MATRIX_LABELS( name, headers, pheaders, labels, plabels );

	return EXIT_SUCCESS;

} // matrix_int_load_binary

//////////////////////////////////////////////////
//////////////////////////////////////////////////

/*
 * Saves an integer matrix to an ASCII-text file.
 * Skips name, headers and labels if 'ml' is set to NULL.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_int_save_ascii( char const *restrict filename, index_t const *restrict matrix, index_t nrows, index_t ncols,
				struct matrix_labels const *restrict ml )
{

	if ( ! matrix ) {
		fflush(stdout);
		errno = EFAULT;
		perror("\nmatrix_int_save_ascii( matrix )");
		return EXIT_FAILURE;
	}

	{
		uintmax_t const nitems = ((uintmax_t) nrows) * ((uintmax_t) ncols);

		if ( (nrows <= 0) + (ncols <= 0) + (nitems > IDX_MAX) ) {
			fflush(stdout);
			errno = EINVAL;
			fprintf( stderr, "\nmatrix_int_save_ascii( rows=%" PRI_IDX ", columns=%" PRI_IDX " ): %s\n",
				nrows, ncols, strerror(errno));
			if ( nitems > IDX_MAX )
				fprintf( stderr, "Matrix size (%" PRIuMAX ") exceeds the limits used for matrix dimensions (%" PRI_IDX ").\n",
					nitems, IDX_MAX);
			return EXIT_FAILURE;
		}
	}


	FILE *restrict const file = fopen( filename, "w" );
	if( ! file ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nfopen '%s' :\n\t%s.\n", filename, strerror(errno) );
		fprintf(stderr,"Error in matrix_int_save_ascii().\n");
		return EXIT_FAILURE;
	}


	char const *restrict name = NULL;
	char const *restrict headers = NULL;
	char const *const *restrict pheaders = NULL;
	char const *restrict labels = NULL;
	char const *const *restrict plabels = NULL;

	if ( ml ) {
		struct matrix_labels lml = *ml;
		name = lml.name;
		headers = lml.headers.tokens;
		pheaders = lml.headers.ptokens;
		labels = lml.labels.tokens;
		plabels = lml.labels.ptokens;
	}


	// Name
	if ( name ) {
		if ( fprintf(file,"%s",name) < 0 ) {
			fflush(stdout);
			fprintf(stderr,
				"\nInternal error in fprintf(name).\nError in matrix_int_save_ascii().\n");
			fclose(file);
			return EXIT_FAILURE;
		}
		if ( headers == NULL && fprintf(file,"\n") <= 0 ) {
			fflush(stdout);
			fprintf(stderr,
				"\nInternal error in fprintf(name\\n).\nError in matrix_int_save_ascii().\n");
			fclose(file);
			return EXIT_FAILURE;
		}
	}


	// Column headers
	if ( headers ) {

		index_t i = 0;

		// Starts with a delimiter or not.
		if ( ! name ) {
			if ( fprintf(file,"%s",headers) < 0 ) {
				fflush(stdout);
				fprintf(stderr, "\nInternal error in fprintf(pheaders[0]).\nError in matrix_int_save_ascii().\n");
				fclose(file);
				return EXIT_FAILURE;
			}
			i = 1;
		}

		for (; i<ncols; i++)
			if ( fprintf(file,"\t%s",pheaders[i]) <= 0 ) {
				fflush(stdout);
				fprintf(stderr, "\nInternal error in fprintf(pheaders[%" PRI_IDX "]).\nError in matrix_int_save_ascii().\n", i);
				fclose(file);
				return EXIT_FAILURE;
			}

		if ( fprintf(file,"\n") <= 0 ) {
			fflush(stdout);
			fprintf(stderr,"\nInternal error in fprintf(headers\\n).\nError in matrix_int_save_ascii().\n");
			fclose(file);
			return EXIT_FAILURE;
		}

	} // if headers


	// Row labels
	if ( labels ) {
		for ( index_t i=0, idx=0 ; i<nrows ; i++ ) {
			if ( fprintf(file,"%s",plabels[i]) < 0 ) {
				fflush(stdout);
				fprintf(stderr, "\nInternal error in fprintf(plabels[%" PRI_IDX "]).\nError in matrix_int_save_ascii().\n",i);
				fclose(file);
				return EXIT_FAILURE;
			}
			for (index_t j=0 ; j<ncols ; j++, idx++)
				if ( fprintf(file,"\t%" PRI_IDX "",matrix[idx]) <= 0 ) {
					fflush(stdout);
					fprintf(stderr,"\nInternal error in fprintf(matrix[%" PRI_IDX "][%" PRI_IDX
							"]).\nError in matrix_int_save_ascii().\n",i,j);
					fclose(file);
					return EXIT_FAILURE;
				}
			if ( fprintf(file,"\n") <= 0 ) {
				fflush(stdout);
				fprintf(stderr,"\nInternal error in fprintf(matrix[%" PRI_IDX "][]\\n).\nError in matrix_int_save_ascii().\n",i);
				fclose(file);
				return EXIT_FAILURE;
			}
		} // for
	} else { // No labels
		for ( index_t i=0, idx=0 ; i<nrows ; i++ ) {
			if ( fprintf(file,"%" PRI_IDX "",matrix[idx]) <= 0 ) {
				fflush(stdout);
				fprintf(stderr, "\nInternal error in fprintf(matrix[%" PRI_IDX "][0]).\nError in matrix_int_save_ascii().\n",i);
				fclose(file);
				return EXIT_FAILURE;
			}
			idx++;
			for ( index_t j=1 ; j<ncols ; j++, idx++)
				if ( fprintf(file,"\t%" PRI_IDX "",matrix[idx]) <= 0 ) {
					fflush(stdout);
					fprintf(stderr,"\nInternal error in fprintf(matrix[%" PRI_IDX "][%" PRI_IDX
							"]).\nError in matrix_int_save_ascii().\n",i,j);
					fclose(file);
					return EXIT_FAILURE;
				}
			if ( fprintf(file,"\n") <= 0 ) {
				fflush(stdout);
				fprintf(stderr,"\nInternal error in fprintf(matrix[%" PRI_IDX "][]\\n).\nError in matrix_int_save_ascii().\n",i);
				fclose(file);
				return EXIT_FAILURE;
			}
		} // for
	} // if labels

	if ( fclose(file) ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nfclose '%s' :\n\t%s.\n", filename, strerror(errno) );
		fprintf(stderr,"Error in matrix_int_save_ascii().\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // matrix_int_save_ascii

//////////////////////////////////////////////////

/*
 * Saves an integer matrix to a binary file.
 * Skips name, headers and labels if 'ml' is set to NULL.
 *
 * WARNING: For internal use only. No error checking is performed.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_int_save_binary( char const *restrict filename, index_t const *restrict matrix, index_t nrows, index_t ncols,
				struct matrix_labels const *restrict ml )
{

	if ( ! matrix ) {
		fflush(stdout);
		errno = EFAULT;
		perror("\nmatrix_int_save_binary( matrix )");
		return EXIT_FAILURE;
	}

	{
		uintmax_t const nitems = ((uintmax_t) nrows) * ((uintmax_t) ncols);

		if ( (nrows <= 0) + (ncols <= 0) + (nitems > IDX_MAX) ) {
			fflush(stdout);
			errno = EINVAL;
			fprintf( stderr, "\nmatrix_int_save_binary( rows=%" PRI_IDX ", columns=%" PRI_IDX " ): %s\n",
				nrows, ncols, strerror(errno));
			if ( nitems > IDX_MAX )
				fprintf( stderr, "Matrix size (%" PRIuMAX ") exceeds the limits used for matrix dimensions (%" PRI_IDX ").\n",
					nitems, IDX_MAX);
			return EXIT_FAILURE;
		}
	}

	FILE *restrict const file = fopen( filename, "wb" );
	if( ! file ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nfopen '%s' :\n\t%s.\n", filename, strerror(errno) );
		fprintf(stderr,"Error in matrix_int_save_binary().\n");
		return EXIT_FAILURE;
	}

	// Data matrix
	size_t const nitems = nrows * ncols;
	size_t const nwritten = fwrite( matrix, sizeof(index_t), nitems, file );
	if ( nwritten != nitems ) {
		fflush(stdout);
		fprintf(stderr,"\nInternal error in fwrite(). %zu items written, %zu expected.\nError in matrix_int_save_binary().\n",
			nwritten, nitems );
		fclose(file);
		return EXIT_FAILURE;
	}

	// ----------------------------------


	// Returns if there is no labels.
	if ( ! ml ) {
		if ( fclose(file) ) {
			int const err = errno; fflush(stdout); errno = err;
			fprintf( stderr, "\nfclose '%s' :\n\t%s.\n", filename, strerror(errno) );
			fprintf(stderr,"Error in matrix_int_save_binary().\n");
			return EXIT_FAILURE;
		}
		return EXIT_SUCCESS;
	}

	struct matrix_labels lml = *ml;

	char const *restrict name = lml.name;
	char const *restrict headers = lml.headers.tokens;
	char const *const *restrict pheaders = lml.headers.ptokens;
	char const *restrict labels = lml.labels.tokens;
	char const *const *restrict plabels = lml.labels.ptokens;


	// Returns if there is no labels.
	if ( ! ( (size_t) labels + (size_t) headers + (size_t) name) ) {	// (labels == NULL) && (headers == NULL) && (name == NULL)
		if ( fclose(file) ) {
			int const err = errno; fflush(stdout); errno = err;
			fprintf( stderr, "\nfclose '%s' :\n\t%s.\n", filename, strerror(errno) );
			fprintf(stderr,"Error in matrix_int_save_binary().\n");
			return EXIT_FAILURE;
		}
		return EXIT_SUCCESS;
	}


	// ----------------------------------


	// Row Labels
	if ( labels ) {
		if ( fprintf(file,"%s",labels) < 0 ) {	// Saves the first label
			fflush(stdout);
			fprintf(stderr,"\nInternal error in fprintf(plabels[0]).\nError in matrix_int_save_binary().\n");
			fclose(file);
			return EXIT_FAILURE;
		}
		for (index_t i=1; i<nrows; i++)
			if ( fprintf(file,"\t%s",plabels[i]) <= 0 ) {
				fflush(stdout);
				fprintf( stderr,"\nInternal error in fprintf(plabels[%" PRI_IDX "]).\nError in matrix_int_save_binary().\n",i);
				fclose(file);
				return EXIT_FAILURE;
			}
	} // if labels

	if ( fprintf(file,"\n")  <= 0 ) {
		fflush(stdout);
		fprintf(stderr,"\nInternal error in fprintf(plabels[]\\n).\nError in matrix_int_save_binary().\n");
		fclose(file);
		return EXIT_FAILURE;
	}


	// Column headers
	if ( headers ) {
		if ( fprintf(file,"%s",headers) < 0 ) {	// Saves the first header
			fflush(stdout);
			fprintf(stderr,"\nInternal error in fprintf(pheaders[0]).\nError in matrix_int_save_binary().\n");
			fclose(file);
			return EXIT_FAILURE;
		}
		for (index_t i=1; i<ncols; i++)
			if ( fprintf(file,"\t%s",pheaders[i])  <= 0 ) {
				fflush(stdout);
				fprintf(stderr,"\nInternal error in fprintf(pheaders[%" PRI_IDX "]).\nError in matrix_int_save_binary().\n",i);
				fclose(file);
				return EXIT_FAILURE;
			}
	} // if headers

	if ( fprintf(file,"\n")  <= 0 ) {
		fflush(stdout);
		fprintf(stderr,"\nInternal error in fprintf(pheaders[]\\n).\nError in matrix_int_save_binary().\n");
		fclose(file);
		return EXIT_FAILURE;
	}


	// Name
	if ( name && fprintf(file,"%s",name)  < 0 ) {
		fflush(stdout);
		fprintf(stderr,"\nInternal error in fprintf(name).\nError in matrix_int_save_binary().\n");
		fclose(file);
		return EXIT_FAILURE;
	}


	if ( fclose(file) ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf( stderr, "\nfclose '%s' :\n\t%s.\n", filename, strerror(errno) );
		fprintf(stderr,"Error in matrix_int_save_binary().\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // matrix_int_save_binary

//////////////////////////////////////////////////

/*
 * Shows matrix's content (data, name, headers and/or labels).
 *
 * If 'transpose' is 'true', transposes matrix as follows:
 * - Matrix dimensions in memory: <numcols> rows, <numrows> columns.
 * - Matrix dimensions on screen: <numrows> rows, <numcols> columns.
 * - Shows <numcols> ml->headers (as column headers) and <numrows> ml->labels (as row labels).
 *
 * numcols <= padding, unless matrix transposing is set (in that case, numrows <= padding).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int matrix_int_show( index_t const *restrict matrix, index_t numrows, index_t numcols, index_t padding, bool transpose,
			struct matrix_labels const *restrict ml )
{

	if ( ! matrix ) {
		fflush(stdout);
		errno = EFAULT;
		perror("\nmatrix_int_show( matrix )");
		return EXIT_FAILURE;
	}

	{
		uintmax_t const nitems = ((uintmax_t) numrows) * ((uintmax_t) numcols);

		if ( (numrows <= 0) + (numcols <= 0) + (padding <= 0) + (nitems > IDX_MAX) ) {
			fflush(stdout);
			errno = EINVAL;
			fprintf( stderr, "\nmatrix_int_show( rows=%" PRI_IDX ", columns=%" PRI_IDX ", padding=%" PRI_IDX " ): %s\n",
				numrows, numcols, padding, strerror(errno));
			if ( nitems > IDX_MAX )
				fprintf( stderr, "Matrix size (%" PRIuMAX ") exceeds the limits used for matrix dimensions (%" PRI_IDX ").\n",
					nitems, IDX_MAX);
			return EXIT_FAILURE;
		}
	}

	if ( transpose ) {
		if ( numrows > padding ) {
			fflush(stdout);
			errno = EINVAL;
			fprintf( stderr, "\nmatrix_int_show( rows=%" PRI_IDX " [number of columns since matrix transposing is selected], "
					"padding=%" PRI_IDX " ): %s\n", numrows, padding, strerror(errno));
			return EXIT_FAILURE;
		}
	} else if ( numcols > padding ) {
			fflush(stdout);
			errno = EINVAL;
			fprintf( stderr, "\nmatrix_int_show( columns=%" PRI_IDX ", padding=%" PRI_IDX " ): %s\n",
				numrows, padding, strerror(errno));
			return EXIT_FAILURE;
	}

	index_t nrows, ncols;

	#if NMFGPU_TESTING
		ncols = numcols;
		nrows = numrows;
	#else
		if ( numrows == 1 ) {	// 'matrix' is a vector.
			ncols = MIN( numcols, 225 ) ;
			nrows = 1;
		} else {
			ncols = MIN( numcols, 15 ) ;
			nrows = MIN( numrows, 9 ) ;
		}
	#endif


	// --------------------------------


	char const *restrict name = NULL;
	char const *restrict headers = NULL;
	char const *const *restrict pheaders = NULL;
	char const *restrict labels = NULL;
	char const *const *restrict plabels = NULL;

	if ( ml ) {
		struct matrix_labels lml = *ml;
		name = lml.name;
		headers = lml.headers.tokens;
		pheaders = lml.headers.ptokens;
		labels = lml.labels.tokens;
		plabels = lml.labels.ptokens;
	}


	// Name
	if ( name )
		printf("Name: '%s'\n\n", name);


	// Column headers
	if ( headers ) {
		printf( "Headers (%" PRI_IDX "):\n", numcols );
		for( index_t i=0; i<ncols; i++ )
			printf("'%s' ", pheaders[i] );
		if ( ncols < numcols )
			printf(" ... '%s'", pheaders[numcols-1]);
		printf("\n\n");
	}


	// ----------------------------

	// Show matrix (with row labels if exist)

	if ( ((size_t) ml) + transpose ) {	// (ml != NULL) || (transpose == true)
		printf("Data matrix: %" PRI_IDX " rows x %" PRI_IDX " columns.\n", numrows, numcols );
		fflush( stdout );
	}

	// Steps for outer and inner loops.
	index_t incr_outer_loop = padding;	// Step for outer loop
	index_t incr_inner_loop = 1;		// Step for inner loop.
	if ( transpose ) {
		incr_outer_loop = 1;
		incr_inner_loop = padding;
	}

	index_t const *pmatrix = matrix;	// &matrix[i][0] (or &matrix[0][i] if transpose)
	for ( index_t i=0 ; i<nrows ; i++,pmatrix+=incr_outer_loop ) {

		index_t j = 0;
		index_t const *pmatrix_r = pmatrix; // &matrix[i][j] (or &matrix[j][i] if transpose)

		printf( "Line %" PRI_IDX ": ", i );

		if ( labels )
			printf( "%s", plabels[i] );

		else { // No labels. Writes the first value.

			printf( "%" PRI_IDX, *pmatrix_r );
			pmatrix_r += incr_inner_loop;	// &matrix[i][1] (or &matrix[1][i] if transpose)
			j = 1;

		} // If has labels

		// Rest of values.
		for ( ; j<ncols ; j++,pmatrix_r+=incr_inner_loop )
			printf( " %" PRI_IDX, *pmatrix_r );

		if ( ncols < numcols )
			printf( " ... %" PRI_IDX, pmatrix[ (numcols - 1) * incr_inner_loop ] ); // NOTE: we use 'pmatrix', not 'pmatrix_r'

		printf( "\n" );

	} // for ( 0 <= i < nrows )

	// Last row.
	if ( nrows < numrows ) {

		index_t i = numrows - 1;
		pmatrix = matrix + i * incr_outer_loop;

		index_t j = 0;
		index_t const *pmatrix_r = pmatrix; // &matrix[i][j] (or &matrix[j][i] if transpose)

		printf( "...\nLine %" PRI_IDX ": ", i );

		if ( labels )
			printf( "%s", plabels[i] );

		else { // No labels. Writes the first value.

			printf( "%" PRI_IDX, *pmatrix_r );
			pmatrix_r += incr_inner_loop;	// &matrix[i][1] (or &matrix[1][i] if transpose)
			j = 1;

		} // If has labels

		// Rest of values.
		for ( ; j<ncols ; j++,pmatrix_r+=incr_inner_loop )
			printf( " %" PRI_IDX, *pmatrix_r );

		if ( ncols < numcols )
			printf( " ... %" PRI_IDX, pmatrix[ (numcols - 1) * incr_inner_loop ] ); // NOTE: we use 'pmatrix', not 'pmatrix_r'

		printf( "\n" );

	} // if ( nrows < numrows )

	printf( "\n" );

	fflush( stdout );

	return EXIT_SUCCESS;

} // matrix_int_show

//////////////////////////////////////////////////
