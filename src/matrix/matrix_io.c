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
 * matrix_io.c
 *	I/O methods for working with tagged matrices.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Debug / Testing:
 *		NMFGPU_DEBUG_READ_MATRIX: Shows information about the matrix being read (e.g., dimensions, labels, etc).
 *		NMFGPU_DEBUG_READ_MATRIX2: Shows detailed information of every datum read.
 *		NMFGPU_TESTING: When set, methods matrix_show() and matrix_int_show() show *ALL* data in matrix (not just a portion).
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

// Required by <inttypes.h>
#ifndef __STDC_FORMAT_MACROS
	#define __STDC_FORMAT_MACROS (1)
#endif
#ifndef __STDC_CONSTANT_MACROS
	#define __STDC_CONSTANT_MACROS (1)
#endif
#include "matrix/matrix_io.h"
#include "common.h"

#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>	/* unlink */
#include <ctype.h>	/* isprint */
#include <inttypes.h>	/* PRIxxx, xxx_C, uintxxx_t */
#include <math.h>	/* isless */


////////////////////////////////////////////////
////////////////////////////////////////////////

/* Constants */

// Signature for binary files.
#undef BIN_FILE_SIGNATURE
#define BIN_FILE_SIGNATURE ( 0xB1023F )

// ---------------------------------------------
// ---------------------------------------------

/* Data types */

// Data type used on "generic" I/O functions.
typedef union generic_data_t {
	real r;
	index_t i;
} data_t;

// ---------------------------------------------
// ---------------------------------------------

/* Global variables */

// Simple variable and macro to test Endiannes.
int const endiannes_test = 1;
#undef IS_BIG_ENDIAN
#define IS_BIG_ENDIAN() ( ! *((unsigned char const *restrict) &endiannes_test) )


////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Returns 'true' if the given number is valid (i.e., is NOT 'NaN' or '+/- infinite').
 */
static bool is_valid( real value )
{

	int const value_type = fpclassify( value );

	return (bool) ( (value_type == FP_ZERO) + (value_type == FP_NORMAL) );

} // is_valid

// ---------------------------------------------

/*
 * Reads a single line from an ASCII-text file.
 * It is stored in pmatrix[ 0...((ncols-1)*incr_ncols) ].
 *
 * If "real_data" is 'true', data are read as 'real' type. Otherwise, they are read as index_t values.
 *
 * If "haslabels" is 'true', a label string is read at the beginning of the line. In addition, if "skip_labels" is 'false',
 * it is appended to *labels, and "labels_length" is updated. If more memory was allocated, "max_labels_length" is also updated.
 * On the other hand, if "haslabels" is 'false' or "skip_labels" is 'true', "labels" is not referenced.
 *
 * Consecutive newline characters and/or an end of file are valid only at the end of the line, in a non-empty file.
 * Otherwise, they are reported as "invalid file format".
 *
 * Finally, sets psum_row, sum_cols, and min_value, if they are non-NULL.
 *
 * NOTE: Although some arguments are redundant (e.g., delimiter, delimiter_scn_format, data_scn_format), it is much
 * faster to pass them as parameters, than using snprintf(3) on every call to generate the same formats strings for scanf(3).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int matrix_read_line( FILE *restrict file, index_t current_line, index_t ncols, index_t incr_ncols, bool haslabels, bool skip_labels,
				int delimiter, char const *restrict delimiter_scn_format, char const *restrict data_scn_format, bool real_data,
				void *restrict pmatrix, void *restrict psum_row, void *restrict sum_cols, void *restrict min_value,
				char *restrict *restrict labels, size_t *restrict labels_length, size_t *restrict max_labels_length )
{

	// Datatype-dependent parameters.

	// Formats strings to be used with fscanf().

	// Format string for delimiter/missing data.
	size_t const delim_scnfmt_size = 2;
	char delimiter_scn_format[ delim_scnfmt_size ];
	sprintf( delimiter_scn_format, "%c", delimiter );

	// Format string for numeric data: data + delimiter
	size_t const data_scnfmt_size = 8;
	char data_scn_format[ data_scnfmt_size ];
	sprintf( data_scn_format, "%%" SCNgREAL "%c", delimiter );

	#if NMFGPU_DEBUG_READ_MATRIX
		print_message( false, "Format delimiter: '%s' (len=%zu)\nFormat data: '%s' (len=%zu)\n\n", delimiter_scn_format,
				strlen(delimiter_scn_format), data_scn_format, strlen(data_scn_format) );
	#endif



	// Size of data.
	size_t const data_size = ( real_data ? sizeof(real) : sizeof(index_t) );





	if ( haslabels ) {

		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( false, "\tReading label (skip=%i)...\n", skip_labels );
		#endif

		// Reads a token.
		char *restrict data = NULL;
		int last_char = (int) '\0';
		size_t const len_data = read_token( file, delimiter, &data, &last_char );
		if ( ! data ) {
			if ( len_data )
				print_error( true, "Error in matrix_read_line()\n" );
			else
				print_error( false, "\nError reading input file:\nPremature end-of-file detected at line %" PRI_IDX
						". Invalid file format.\n", current_line );
			return EXIT_FAILURE;
		}

		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( false, "\t\tLabel(len=%zu): '%s'. Last_char=", len_data, data );
			if ( last_char == (int) '\n' ) print_message( false, "'\\n'.\n");
			else if ( last_char == (int) '\r' ) print_message( false, "'\\r'.\n");
			else if ( isprint(last_char) + isblank(last_char) ) print_message( false, "'%c'.\n", last_char );
			else if ( last_char == EOF ) print_message( false, "EOF.\n", last_char );
			else print_message( false, "'\\x%X'.\n", last_char );
		#endif

		if ( last_char != delimiter ) {	// Blank line.
			print_error( false, "\nError reading input file: No matrix data at line %" PRI_IDX ". Invalid file format.\n",
					current_line );
			free(data);
			return EXIT_FAILURE;
		}

		// ---------------------------

		if ( ! skip_labels ) {

			char *labels_str = *labels;			// String where labels are stored.
			size_t len_labels = *labels_length;		// Current number of characters stored in labels_str.
			size_t max_len_labels = *max_labels_length;	// Current size of labels_str, in number of characters, not in bytes.

			// Before setting the new label, checks if there is enough memory available.
			size_t const len = len_labels + len_data + 1;

			if ( len > max_len_labels ) {	// Allocates more memory

				// Sets to the next power of 2.
				max_len_labels = prev_power_2( len ) * 2;

				#if NMFGPU_DEBUG_READ_MATRIX
					print_message( false, "\t\tExpanding size of labels to %zu chars.\n", max_len_labels );
				#endif

				char *restrict const tmp = (char *restrict) realloc( labels_str, max_len_labels * sizeof(char) );
				if ( ! tmp ) {
					print_errnum( true, errno, "Error in matrix_read_line(): realloc( labels, max_labels_length=%zu ) ",
							max_len_labels );
					free(data);
					return EXIT_FAILURE;
				}
				labels_str = tmp;
				*labels = tmp;
				*max_labels_length = max_len_labels;

			} // If allocate more memory for labels

			// Sets the new label
			strcpy( &labels_str[ len_labels ], data );	// Copies from 'data' the new token.
			len_labels = len;				// len == (len_labels + len_data + 1)
			*labels_length = len;

		} // If set the label

		free( data );

	} // if haslabels

	// ----------------------------

	// Reads the data matrix

	#if NMFGPU_DEBUG_READ_MATRIX
		print_message( false, "\tReading data matrix (ncols=%" PRI_IDX ", incr_ncols=%" PRI_IDX ")...\n", ncols, incr_ncols );
	#endif

	void *pmatrix_r = pmatrix;

	// First ncols-1 columns
	for ( index_t col = 0 ; col < (ncols-1) ; col++, pmatrix_r += (incr_ncols * data_size) ) {

		char const fmt
		char c[2] = { 0, 0 };			// Delimiter to be read.
		int conv = 0;				// Number of matched items.
		data_t value;				// Value to be read.
		memset( &value, 0, sizeof(value) );


		// Tries to read a value and a delimiter.
		errno = 0;
		conv = fscanf( file, data_scn_format, &value, c );

		#if NMFGPU_DEBUG_READ_MATRIX2
		{
			int const err = errno;
			bool const shown_by_all = false;
			print_message( shown_by_all, "\tconv=%i,c=", conv );	// conv: number of tokens read.
			if ( ! c[0] ) print_message( shown_by_all, "(empty)" );
			else if ( (int) c[0] == delimiter ) print_message( shown_by_all, "delim ('%c')", c[0] );
			else print_message( shown_by_all, (isgraph(c[0]) ? "%c" : "'\\0x%X'"), c[0] );
			if ( real_data )
				print_message( shown_by_all, ",value=%g", value.r );
			else
				print_message( shown_by_all, ",value=%" PRI_IDX, value.i );
			errno = err;
		}
		#endif

		/* If no conversion was performed, it might be a missing data,
		 * so tries to just read the delimiter.
		 */
		if ( ! conv ) {
			char const fmt[2] = { delimiter, '\0' };
			c[1] = c[0] = 0
			errno = 0;
			conv = fscanf( file, fmt, c );

			#if NMFGPU_DEBUG_READ_MATRIX2
			{
				int const err = errno;
				bool const shown_by_all = false;
				print_message( shown_by_all, " (missing data) ; conv2=%i,c2=", conv );
				if ( ! c[0] ) print_message( shown_by_all, "(empty)" );
				else if ( (int) c[0] == delimiter ) print_message( shown_by_all, "delim ('%c')", c[0] );
				else print_message( shown_by_all, (isgraph(c) ? "%c" : "'\\0x%X'"), c[0] );
				errno = err;
			}
			#endif
		}

		// Fails on error or premature EOF/EOL.
		if ( (int) c[0] != delimiter ) {
			if ( ferror(file) )	// Error.
				print_errnum( true, errno, "Error in matrix_read_line() reading line %" PRI_IDX
						", column %" PRI_IDX ": fscanf()", current_line, col + haslabels + 1 );
			else if ( feof(file) )
				print_error( false, "\nError reading input file:\nPremature end of file detected in line %" PRI_IDX
						" (%" PRI_IDX " columns found, %" PRI_IDX " expected).\nInvalid file format.\n",
						current_line, col + haslabels, ncols + haslabels );
			else {	// Invalid data or premature EOL.
				bool const shown_by_all = false;
				int const chr = ( c[0] ? c[0] : fgetc( file ) );		// (First) Illegal character.
				if ( (chr == (int) '\n') + (chr == (int) '\r') )	// Premature EOL.
					print_error( shown_by_all, "\nError reading input file:\nPremature end of line detected at line %"
							PRI_IDX " (%" PRI_IDX " columns found, %" PRI_IDX " expected).\nInvalid file format.\n",
							current_line, col + haslabels, ncols + haslabels );
				else {
					print_error( shown_by_all, "\nError reading input file at line %" PRI_IDX ", column %" PRI_IDX
								". Unexpected character: ", current_line, col + haslabels + 1 );
					print_error( shown_by_all, (isgraph(chr) ? "'%c'\n" : "'\\0x%X'\n"), chr );
				}
			}
			return EXIT_FAILURE;
		}

		// Sets the new value.
		memcpy( pmatrix_r, &value, data_size );

	} // for (0 <= col < ncols-1)

	// last file column: col = ncols-1
	{
		index_t const col = ncols - 1;

		char c[3] = { 0, 0, 0 };		// Newline to be read ("\r\n" or "\n"), followed by '\0'.
		int conv = 0;				// Number of matched items.
		data_t value;				// Value to be read.
		memset( &value, 0, sizeof(value) );

		// Tries to read a value and the newline.
		errno = 0;
		conv = fscanf( file, data_eol_scn_fmt, &value, c );

		#if NMFGPU_DEBUG_READ_MATRIX2
		{
			bool const shown_by_all = false;
			int const err = errno;
			print_message( shown_by_all, "\tconv(last)=%i,c(last)=", conv );	// conv: number of tokens read.
			if ( c[0] ) {
				for ( size_t i = 0 ; i < 2 ; i++ )
					switch( c[i] ) {
						case '\0': break;
						case '\r': { print_message( shown_by_all, "\\r" ); } break;
						case '\n': { print_message( shown_by_all, "\\n" ); } break;
						case '\t': { print_message( shown_by_all, "\\t" ); } break;
						case  ' ': { print_message( shown_by_all, "' '" ); } break;
						default  : { print_message( shown_by_all, (isgraph(c[0]) ? "'%c'" : "'\\0x%X'"), c[i] ); } break;
					}
			} else print_message( shown_by_all, "(empty)" );
			if ( real_data )
				print_message( shown_by_all, ",value(last)=%g", value.r );
			else
				print_message( shown_by_all, ",value(last)=%" PRI_IDX, value.i );
			errno = err;
		}
		#endif

		/* If no conversion was performed, it might be a missing data,
		 * so tries to just read the newline.
		 */
		if ( ! conv ) {
			c[2] = c[1] = c[0] = 0;
			errno = 0;
			conv = fscanf( file, "%2[\r\n]", c );

			#if NMFGPU_DEBUG_READ_MATRIX2
			{
				int const err = errno;
				bool const shown_by_all = false;
				print_message( shown_by_all, " (missing data) ; conv2(last)=%i,c2(last)=", conv );
				if ( c[0] ) {
					for ( size_t i = 0 ; i < 2 ; i++ )
						switch( c[i] ) {
							case '\0': break;
							case '\r': { print_message( shown_by_all, "\\r" ); } break;
							case '\n': { print_message( shown_by_all, "\\n" ); } break;
						}
				} else print_message( shown_by_all, "(empty)" );
				errno = err;
			}
			#endif
		}

		// Fails on error, or if neither EOL nor EOF was read.
		if ( ferror(file) + (! (c[0] + feof(file))) + ((c[0] == '\r') * (c[1] != '\n')) ) {
			if ( ferror(file) )	// Error.
				print_errnum( true, errno, "Error in matrix_read_line() at line %" PRI_IDX
						", column %" PRI_IDX ": fscanf()", current_line, col + haslabels + 1 );

			else if ( c[0] == '\r' ) {  // (c[0] == '\r' && c[1] != '\r')
				print_error( shown_by_all, "\nError reading input file at line %" PRI_IDX ", column %" PRI_IDX
						". Unexpected end-of-line: '\\r", current_line, col + haslabels + 1 );
				print_error( shown_by_all, ( isgraph(chr) ? ("%c'\n") : ("\\0x%X'\n") ), chr );

			} else { // Invalid datum: (!c[0] && !EOF)
				bool const shown_by_all = false;
				int const chr = fgetc( file );		// (First) Illegal character.
				if ( chr == delimiter )			// There are more than <ncols> columns.
					print_error( shown_by_all, "\nError reading input file at line %" PRI_IDX ": Invalid file format.\n"
							"There are more than %" PRI_IDX " columns.\n", current_line, ncols + haslabels );
				else {
					print_error( shown_by_all, "\nError reading input file at line %" PRI_IDX ", column %" PRI_IDX
							". Unexpected character: ", current_line, col + haslabels + 1 );
					print_error( shown_by_all, ( isgraph(chr) ? ("'%c'\n") : ("'\\0x%X'\n") ), chr );
				}
			}
			return EXIT_FAILURE;
		}

		// Returns any character that belongs to the next line.
		if ( ((c[0] == '\n') * c[1]) && (ungetc(c[1], file) == EOF) ) {
			print_errnum( true, errno, "Error in matrix_read_line() at line %" PRI_IDX
					", column %" PRI_IDX ": ungetc()", current_line, col + haslabels + 1 );
			return EXIT_FAILURE;
		}

		// Finally, sets the new value.
		memcpy( pmatrix_r, &value, data_size );

	} // last column.

	// -----------------------------


		// If the last character read was a CR ('\r'), reads the corresponding LF ('\n') character.
		if ( c[0] == '\r' ) {

			int const chr = fgetc(file);

			#if NMFGPU_DEBUG_READ_MATRIX2
				print_message( false, "c_final=" );
				if ( ! chr ) print_message( false, "(empty)\n" );
				else if ( chr == (int) '\n' ) print_message( false, "'\\n'\n" );
				else if ( chr == (int) '\r' ) print_message( false, "'\\r'\n" );
				else if ( chr == EOF ) print_message( false, "EOF\n", chr);
				else print_message( false, ( (isprint(chr) + isblank(chr)) ? "'%c'\n" : "'\\x%X'\n"), chr );
			#endif

			// Fails on error, or if it is not an LF or an EOF character.
			if ( ferror(file) + ( (chr != (int)'\n') * (chr != EOF) ) ) {
				if ( ferror(file) )
					print_error( true, "\nError in matrix_read_line() reading line %" PRI_IDX ": fgetc('\\n').\n",
							current_line );
				else {
					print_error( false, "\nError reading input file in line %" PRI_IDX ", column %" PRI_IDX
							": unexpected character after a CR ('\\r'): ", current_line, col + haslabels + 1 );
					print_error( false, ( (isprint(chr) + isblank(chr)) ?
							("'%c'.\nInvalid file format.\n") : ("'\\x%X'.\nInvalid file format.\n") ), chr );
				}
				return EXIT_FAILURE;
			}

		} // if last character read was a CR ('\r').

	// -----------------------------


	void *psum_cols = sum_cols;

			// It also checks for invalid format on 'real'-type data.
			bool const invalid_real_value = real_data * ( ! is_valid( value.r ) );

					else if ( invalid_real_value )
						print_error( false, ": %g\n", value.r );


			// Sets some of the output values, if they are non-NULL.
			if ( real_data ) {

				real *const psr = (real *) psum_row;
				if ( psr )
					*psr += value.r;

				real *const psc = (real *) psum_cols;
				if ( psc )
					*psc += value.r;

			} else {

				index_t *const psr = (index_t *) psum_row;
				if ( psr )
					*psr += value.i;

				index_t *const psc = (index_t *) psum_cols;
				if ( psc )
					*psc += value.i;
			}


		// ... and updates the minimum, if non-NULL.
		if ( min_value ) {
			if ( real_data ) {
				real *const pmv = (real *) min_value;
				*pmv = MIN( *pmv, value.r );	// value.r has been checked already.
			} else {
				index_t *const pmv = (index_t *) min_value;
				*pmv = MIN( *pmv, value.i );
			}
		}

		if ( psum_cols )
			psum_cols += (incr_ncols * data_size);





	return EXIT_SUCCESS;

} // matrix_read_line

// ---------------------------------------------

/*
 * Loads a matrix from an ASCII-text file.
 *
 * If "real_data" is 'true', data are read as 'real' type. Otherwise, they are read as index_t values.
 *
 * Skips name, headers and labels if 'mt' is NULL.
 *
 * If (*matrix)  is non-NULL, do not allocates memory but uses the supplied one.
 * WARNING: In such case, THERE MUST BE ENOUGH MEMORY ALREADY ALLOCATED.
 *
 * Detects the symbol used as delimiter (tab or white space).
 *
 * If "transpose" is 'true', transposes matrix in memory as follows:
 *	- Reads from file <nrows> rows, and <ncols> columns.
 *	- Writes to memory: <ncols> rows, and <nrows> columns (padded to <pitch>).
 *	- Reads <ncols> column headers (set as mt->headers) and <nrows> row labels (set as mt->labels).
 *
 * WARNING:
 *	- If "transpose" is 'true', nrows must be <= pitch. Else, ncols must be <= pitch
 *	- NO error checking is performed to detect negative data or empty rows/columns.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int matrix_read_ascii( char const *restrict filename, index_t nrows, index_t ncols, index_t pitch, bool hasname, bool hasheaders,
			bool haslabels, bool transpose, bool real_data, void *restrict *restrict matrix, struct matrix_tags_t *restrict mt )
{

	// Size of data
	size_t const data_size = ( real_data ? sizeof(real) : sizeof(index_t) );

	// Limits on matrix dimensions
	size_t const max_num_items = matrix_max_num_items;
	index_t const max_pitch = matrix_max_pitch;
	index_t const max_non_padded_dim = matrix_max_non_padded_dim;

	// Name and headers
	char *restrict name = NULL;
	struct tag_t headers = new_empty_tag();

	// Labels
	struct tag_t labels = new_empty_tag();
	size_t max_len_labels = 0;		// Allocated memory for labels.tokens
	size_t len_labels = 0;			// Memory currently used in labels.tokens.

	// Skips name, headers and labels if mt is NULL.
	bool const skip = ! mt;	// ( mt == NULL )

	// Delimiter character (TAB by default).
	int delimiter = (int) '\t';

	// Line number to be read.
	index_t nlines = 1;

	////////////////////////////////

	// Checks for NULL pointers
	if ( ! ( (uintptr_t) filename * (uintptr_t) matrix ) ) {
		int const errnum = EFAULT;
		bool const shown_by_all = false;
		if ( ! filename ) print_errnum( shown_by_all, errnum, "\nmatrix_read_ascii( filename )" );
		if ( ! matrix ) print_errnum( shown_by_all, errnum, "\nmatrix_read_ascii( matrix )" );
		return EXIT_FAILURE;
	}

	// Checks matrix dimensions
	{
		index_t const dim_major = ( transpose ? ncols : nrows );
		uintmax_t const nitems = (uintmax_t) dim_major * (uintmax_t) pitch;

		if ( (nrows <= 0) + (ncols <= 0) + (pitch <= 0) ) {
			print_error( false, "\nError in matrix_read_ascii( rows=%" PRI_IDX ", columns=%" PRI_IDX ", pitch=%" PRI_IDX
					"transpose=%i ): Invalid matrix dimensions.\n", nrows, ncols, pitch, transpose );
			return EXIT_FAILURE;
		}

		if ( (pitch > max_pitch) + (dim_major > max_non_padded_dim) + (nitems > (uintmax_t) max_num_items) ) {
			print_error( false, "\nSorry, but your matrix exceeds the limits for matrix dimensions.\n"
					"On this system and with the given input arguments, data matrices are limited to:\n"
					"\t* %" PRI_IDX " rows.\n\t*%" PRI_IDX " columns.\n\t*%zu items.\n",
					max_non_padded_dim, max_pitch, max_num_items );
			return EXIT_FAILURE;
		}

		if ( transpose ) {
			if ( nrows > pitch ) {
				print_error( false, "\nError in matrix_read_ascii( rows=%" PRI_IDX ", pitch=%" PRI_IDX
						", transpose: yes ): Invalid values.\n", nrows, pitch );
				return EXIT_FAILURE;
			}
		} else if ( ncols > pitch ) {
			print_error( false, "\nError in matrix_read_ascii( columns=%" PRI_IDX ", pitch=%" PRI_IDX " ): Invalid values.\n",
					ncols, pitch );
				return EXIT_FAILURE;
		}
	}

	// -----------------------------

	// Starts Reading ...

	FILE *restrict const file = fopen( filename, "r" );
	if ( ! file ) {
		print_errnum( true, errno, "Error in matrix_read_ascii(): fopen( %s )", filename );
		return EXIT_FAILURE;
	}

	// ----------------------------

	// Name and/or headers.

	if ( hasname + hasheaders ) {

		size_t ntokens = 0;

		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( false, "\t\tReading name/headers...\n" );
		#endif

		int const status = read_tag( file, delimiter, &headers, NULL, &ntokens );

		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( false, "\t\t\tstatus=%i, ntokens=%zu\n", status, ntokens );
		#endif

		if ( status ) {	// EOF, error, or invalid format
			if ( status == 1 )	// EOF
				print_error( false, "\nError reading input file: file is empty?\n" );
			else if ( status == 2 )	// Internal error.
				print_error( true, "Error in matrix_read_ascii( line 1 ).\n" );
			else	// 3: Maximum line length
				print_error( false, "Is your data matrix stored in a single text line?\nPlease check also for "
						"any invalid line terminator. Only LF ('\\n') and CR+LF ('\\r\\n') are accepted.\n" );
			fclose(file);
			return EXIT_FAILURE;
		}

		if ( skip ) {
			clean_tag( headers );
			headers = new_empty_tag();
		} else {

			// Checks for proper number of tokens.

			// If only one token, and there should be more, "retokenizes" using space as delimiter.
			if ( (ntokens == 1) * hasheaders * (hasname + ncols - 1) ) {	// (hasname || ncols > 1)

				// Sets the new delimiter.
				delimiter = (int) ' ';

				#if NMFGPU_DEBUG_READ_MATRIX
					print_message( false, "\t\t\t\"Retokenizing\" data on line 1 (hasname=%i, hasheaders=%i, ncols=%"
							PRI_IDX ")...\n", hasname, hasheaders, ncols );
				#endif

				// Removes previous array of pointers to tokens.
				char *restrict data = (char *restrict) headers.tokens;
				free( (void *) headers.ptokens );
				ntokens = 0;

				headers = tokenize( data, delimiter, &ntokens );

				#if NMFGPU_DEBUG_READ_MATRIX
					print_message( false, "\t\t\tLine 1 (headers): %s, num_tokensL1=%zu\n",
						       ( headers.tokens ? "non-NULL" : "NULL" ), ntokens );
				#endif

				if ( ! headers.tokens ) {
					print_error( true, "Error in matrix_read_ascii(): re-tokenize line 1 (hasname=%i, hasheaders=%i, "
							"ncols=%" PRI_IDX ").\n", hasname, hasheaders, ncols );
					free( (void *) data );
					fclose(file);
					return EXIT_FAILURE;
				}

			} // If only one token

			if ( ntokens != (size_t) ((hasheaders * ncols) + hasname) ) {
				print_error( false, "\nError reading input file: Invalid number of items in line 1: %zu read, %"
						PRI_IDX " expected.\nInvalid file format.\n", ntokens, (hasheaders * ncols) + hasname );
				clean_tag(headers);
				fclose(file);
				return EXIT_FAILURE;
			}

			// ------------------------

			if ( hasname * hasheaders ) {	// Name and headers

				/* The first token is the 'name' field, and the rest are column headers.
				 * size_of(headers.ptokens): ncols + 1
				 */

				char *restrict data = (char *restrict) headers.tokens;
				char **restrict pdata = (char **restrict) headers.ptokens;

				// Name: copies the first token.
				name = (char *restrict) malloc( (strlen(data) + 1) * sizeof(char) );
				if ( ! name ) {
					print_errnum( true, errno, "Error in matrix_read_ascii(): malloc( name, size=%zu )",
							strlen(data) + 1 );
					clean_tag(headers);
					fclose(file);
					return EXIT_FAILURE;
				}
				strcpy( name, data );

				#if NMFGPU_DEBUG_READ_MATRIX
					print_message( false, "\t\tName (len=%zu):'%s'\n", strlen(name), name );
				#endif

				/* Headers: Sets remaining tokens as column headers.
				 *
				 * Moves the second token to the beginning of "data", overwriting the first token (which was already
				 * copied into "name"). Remaining tokens are kept untouched, and the previous place of the second
				 * token is left as "garbage".
				 *
				 * Therefore, data == pdata[0], and it is possible to call free(3) on them.
				 */
				if ( ! memmove( data, pdata[1], (strlen(pdata[1]) + 1) * sizeof(char) ) ) {
					print_errnum( true, errno, "Error in matrix_read_ascii(): memmove( headers, size=%zu )",
							strlen(pdata[1]) + 1 );
					free((void *)name); clean_tag(headers);
					fclose(file);
					return EXIT_FAILURE;
				}

				// Adjusts pdata[] to <ncols> tokens
				pdata[ 0 ] = data;
				for ( index_t i = 1 ; i < ncols ; i++ )
					pdata[ i ] = pdata[ i+1 ];

				pdata = (char **restrict) realloc( pdata, ncols * sizeof(char *) );
				if ( ! pdata ) {
					print_errnum( true, errno, "Error in matrix_read_ascii(): realloc( pheaders, ncols=%" PRI_IDX
							" )", ncols );
					free((void *)name); clean_tag(headers);
					fclose(file);
					return EXIT_FAILURE;
				}

				headers = new_tag( data, pdata );

			} else if ( hasname ) { // No headers, name only.

				name = (char *restrict) headers.tokens;	// The only token.

				if ( headers.ptokens )
					free( (void *)headers.ptokens );

				headers = new_empty_tag();
			}
			// Else, headers only

		} // if skip labels.

		// Now, reading line 2...
		nlines = 2;

	} // if has name and/or headers


	// ----------------------------


	// Labels

	if ( haslabels * ( ! skip ) ) {

		max_len_labels = 64 * prev_power_2( nrows );	// Initial size. It will be adjusted later.

		char *restrict const labels_str = (char *restrict) malloc( max_len_labels * sizeof(char) );
		if ( ! labels_str ) {
			print_errnum( true, errno, "Error in matrix_read_ascii(): malloc( labels_str, size=%zu )", max_len_labels );
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );	// labels == NULL
			clean_matrix_tags(l_mt);
			fclose(file);
			return EXIT_FAILURE;
		}
		labels.tokens = (char const *restrict) labels_str;

		char **restrict const plabels = (char **restrict) malloc( (size_t) nrows * sizeof(char *) );
		if ( ! plabels ) {
			print_errnum( true, errno, "Error in matrix_read_ascii(): malloc( plabels, nrows=%" PRI_IDX " )", nrows );
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
			clean_matrix_tags(l_mt);
			fclose(file);
			return EXIT_FAILURE;
		}
		labels.ptokens = (char const *const *restrict) plabels;

	} // if has labels.


	// ----------------------------


	// Data matrix.

	void *restrict l_matrix = (*matrix);
	if ( ! l_matrix ) {	// Allocates memory
		index_t const dim_major = ( transpose ? ncols : nrows );
		l_matrix = (void *restrict) malloc( (size_t) dim_major * (size_t) pitch * data_size );
		if ( ! l_matrix ) {
			print_errnum( true, errno, "Error in matrix_read_ascii(): malloc( l_matrix, nrows=%" PRI_IDX ", ncols=%"
					PRI_IDX ", pitch= %" PRI_IDX ", transpose: %i, real_data: %i, data_size=%zu )",
					nrows, ncols, pitch, transpose, real_data, data_size );
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
			clean_matrix_tags(l_mt);
			fclose(file);
			return EXIT_FAILURE;
		}
	}


	// ----------------------------


	// Format strings to be used with fscanf().

	// Format string for delimiter/missing data.
	size_t const delim_scnfmt_size = 8;
	char delimiter_scn_format[ delim_scnfmt_size ];
	sprintf( delimiter_scn_format, "%%1[%c\n\r]", delimiter );

	// Format string for numeric data: data + delimiter
	size_t const data_scnfmt_size = 2 * delim_scnfmt_size;
	char data_scn_format[ data_scnfmt_size ];
	sprintf( data_scn_format, ( real_data ? "%%" SCNgREAL "%s" : "%%" SCN_IDX "%s" ), delimiter_scn_format );

	#if NMFGPU_DEBUG_READ_MATRIX
		print_message( false, "Format delimiter: '%s' (len=%zu)\nFormat data: '%s' (len=%zu)\n\n", delimiter_scn_format,
				strlen(delimiter_scn_format), data_scn_format, strlen(data_scn_format) );
	#endif


	// ----------------------------


	// Reading file...

	// Step sizes for outer and inner loops.
	index_t incr_outer_loop = pitch;	// Step size for outer loop
	index_t incr_inner_loop = 1;		// Step size for inner loop.
	if ( transpose ) {
		incr_outer_loop = 1;
		incr_inner_loop = pitch;
	}

	void *pmatrix = l_matrix;	// &matrix[row][0] (or &matrix[0][row] if transpose)

	for ( index_t r = 0 ; r < nrows ; r++, nlines++, pmatrix += (incr_outer_loop * data_size) ) {

		// WARNING: Does NOT check negative values or empty rows/columns.
		void *const psum_row = NULL;
		void *const psum_cols = NULL;
		void *const pmin_value = NULL;

		// Reads the row label (if any) and a full matrix row (in file).
		int const status = matrix_read_line( file, nlines, ncols, incr_inner_loop, haslabels, skip, delimiter, delimiter_scn_format,
							data_scn_format, real_data, pmatrix, psum_row, psum_cols, pmin_value,
							(char *restrict *restrict) &(labels.tokens), &len_labels, &max_len_labels );

		if ( status != EXIT_SUCCESS ) {
			print_error( false, "\nError reading input file.\n" );
			if (! (*matrix)) free( (void *) l_matrix );
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
			clean_matrix_tags(l_mt);
			fclose(file);
			return EXIT_FAILURE;
		}

	} // for ( 0 <= r < nrows )

	// -----------------------------

	// Resizes labels.tokens
	if ( haslabels * (! skip) ) {

		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( false, "Resizing labels from %zu to %zu\n", max_len_labels, len_labels );
		#endif

		char *restrict data = (char *restrict) realloc( (void *) labels.tokens, len_labels * sizeof(char) );
		if ( ! data ) {
			print_errnum( true, errno, "Error in matrix_read_ascii(): realloc( labels, len=%zu )", len_labels );
			if (! (*matrix)) free( (void *) l_matrix );
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
			clean_matrix_tags(l_mt);
			fclose(file);
			return EXIT_FAILURE;
		}
	}

	fclose(file);

	// Sets output values.
	*matrix = l_matrix;
	if ( ! skip )
		*mt = new_matrix_tags( name, headers, labels );

	return EXIT_SUCCESS;

} // matrix_read_ascii

////////////////////////////////////////////////

/*
 * Returns 'true' if "str" represents one or more numbers, or an empty string.
 */
static bool isnumber( char const *restrict str )
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

} // isnumber

// ---------------------------------------------

/*
 * Loads a real-type matrix from an ASCII-text file.
 *
 * Detects automatically if matrix has name, column headers and/or row labels,
 * as well as the employed delimiter symbol (space or tab character).
 *
 * In addition, outputs information messages, and performs error checking.
 *
 * The number of columns is rounded up to a multiple of <memory_alignment>.
 *
 * WARNING:
 *	- Both matrix dimensions must be >= 2.
 *	- All rows and columns must have at least one positive value (i.e., greater than 0).
 *	- Negative values are NOT allowed.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_load_ascii_verb( char const *restrict filename, bool numeric_hdrs, bool numeric_lbls, real *restrict *restrict matrix,
				index_t *restrict nrows, index_t *restrict ncols, index_t *restrict pitch, struct matrix_tags_t *restrict mt )
{

	// Local values and pointers to output parameters.
	index_t numcols = INDEX_C(0), numrows = INDEX_C(0), l_pitch = INDEX_C(0);
	bool hasheaders = false, haslabels = false;

	// Limits on matrix dimensions
	size_t const max_num_items = matrix_max_num_items;
	index_t const max_pitch = matrix_max_pitch;
	index_t const max_non_padded_dim = matrix_max_non_padded_dim;

	// Name and headers
	char *restrict name = NULL;
	struct tag_t headers = new_empty_tag();

	// Labels
	struct tag_t labels = new_empty_tag();
	size_t max_len_labels = 0;			// Allocated memory for labels.tokens
	size_t len_labels = 0;				// Memory currently used in labels.tokens.

	// Data matrix
	real *restrict data_matrix = NULL;
	index_t max_numrows = MIN( INDEX_C(512), max_non_padded_dim );	// Initial size for data_matrix
	index_t nlines = 0;				// Number of lines processed.

	int delimiter = (int) '\t';			// Delimiter for tokens (TAB by default).

	int status = 0;

	// Sums of columns: used to check that all columns have at least one positive value.
	real *restrict sum_cols = NULL;			// Size: numcols

	// Data in line 1: Name and/or columns headers.
	struct tag_t tokensL1 = new_empty_tag();
	size_t len_tokensL1 = 0;			// Length of line 1.
	size_t num_tokensL1 = 0;			// Number of tokens in line 1.

	// Data in line 2 (first line to have numeric data)
	struct tag_t tokensL2 = new_empty_tag();
	size_t len_tokensL2 = 0;			// Length of line 2.
	size_t num_tokensL2 = 0;			// Number of tokens in line 2.

	/////////////////////////////////

	// Checks for NULL parameters
	if ( ! ( (uintptr_t) filename * (uintptr_t) matrix * (uintptr_t) nrows * (uintptr_t) ncols * (uintptr_t) pitch * (uintptr_t) mt ) ) {
		int const errnum = EFAULT;
		bool const shown_by_all = false;
		if ( ! filename ) print_errnum( shown_by_all, errnum, "\nmatrix_load_ascii_verb( filename )" );
		if ( ! matrix )	print_errnum( shown_by_all, errnum, "\nmatrix_load_ascii_verb( matrix )" );
		if ( ! nrows )	print_errnum( shown_by_all, errnum, "\nmatrix_load_ascii_verb( nrows )" );
		if ( ! ncols )	print_errnum( shown_by_all, errnum, "\nmatrix_load_ascii_verb( ncols )" );
		if ( ! pitch )	print_errnum( shown_by_all, errnum, "\nmatrix_load_ascii_verb( pitch )" );
		if ( ! mt )	print_errnum( shown_by_all, errnum, "\nmatrix_load_ascii_verb( mt )" );
		return EXIT_FAILURE;
	}

	/////////////////////////////////

	// Starts Reading ...

	FILE *restrict const file = fopen( filename, "r" );
	if ( ! file ) {
		print_errnum( true, errno, "Error in matrix_load_ascii_verb(): fopen( %s )", filename );
		return EXIT_FAILURE;
	}


	/////////////////////////////////


	// Reads line 1

	status = read_tag( file, delimiter, &tokensL1, &len_tokensL1, &num_tokensL1 );
	if ( status ) {	// EOF, error, or invalid format
		if ( status == 1 )	// EOF
			print_error( false, "\nError reading input file: file is empty?\n" );
		else if ( status == 2 )	// Internal error.
			print_error( true, "Error in matrix_load_ascii_verb( line 1 ).\n" );
		else	// 3: Maximum line length
			print_error( false, "Is your data matrix stored in a single text line?\nPlease check also for "
					"any invalid line terminator. Only LF ('\\n') and CR+LF ('\\r\\n') are accepted.\n" );
		fclose(file);
		return EXIT_FAILURE;
	}


	/////////////////////////////////


	#if NMFGPU_DEBUG_READ_MATRIX
		print_message( false, "Line1: len=%zu, ntokens=%zu\n", len_tokensL1, num_tokensL1 );
	#endif


	// Checks for overflow (compares with <max_pitch> to make sure that padding will not overflow).
	if ( (num_tokensL1-1) > (size_t) max_pitch ) {	// - 1, because the file may have a "name" field.
		print_message( false, "\n\t\tNumber of items read on line 1: %zu.\n", num_tokensL1 );
		print_error( false, "\nSorry, but your matrix exceeds the limits for matrix dimensions.\n"
				"On this system and with the given input arguments, data matrices are limited to:\n"
				"\t* %" PRI_IDX " rows.\n\t*%" PRI_IDX " columns.\n\t*%zu items.\n\nPlease check also for any "
				"invalid line terminator. Only LF ('\\n') and CR+LF ('\\r\\n') are accepted.\n\n",
				max_non_padded_dim, max_pitch, max_num_items );
		clean_tag(tokensL1);
		fclose(file);
		return EXIT_FAILURE;
	} // if overflows.


	/////////////////////////////////


	// Detects if file might have name and/or headers

	// Checks if all tokens, from the second one, are numeric.

	bool has_name_headers = false;	// File might have name and/or headers
	{

		char const *const *pdataL1 = tokensL1.ptokens;

		index_t nt = (index_t) num_tokensL1;

		index_t i = 1;
		while ( (i < nt) && isnumber(pdataL1[i]) )
			i++;

		// File might have name and/or headers if:
		has_name_headers = (	(i < nt) +				   // Not all tokens, from the second one, are numeric, <OR>
					numeric_hdrs +				   // input matrix has numeric column headers, <OR>
					( (nt == 1) && (! isnumber(pdataL1[0])) ) ); // It has only one (non-numeric) token.
	}

	#if NMFGPU_DEBUG_READ_MATRIX
		print_message( false, "has_name_headers=%i\n", has_name_headers );
	#endif

	if ( has_name_headers ) {

		// File may have name and/or headers

		// Reads line 2.
		status = read_tag( file, delimiter, &tokensL2, &len_tokensL2, &num_tokensL2 );

		if ( status ) {	// EOF, error, or invalid format
			if ( status == 2 )	// Internal error.
				print_error( true, "Error in matrix_load_ascii_verb( line 2 ).\n" );
			else {	// EOF or maximum line length
				if ( status == 1 )	// EOF
					print_error( false, "\nError reading input file: premature end of file.\n" );
				print_error( false, "Is your data matrix stored in a single text line?\nPlease check also for "
						"any invalid line terminator; only LF ('\\n') and CR+LF ('\\r\\n') are accepted.\n" );
			}
			clean_tag(tokensL1);
			fclose(file);
			return EXIT_FAILURE;
		}

		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( false, "num_tokensL2=%zu\n", num_tokensL2 );
		#endif

		// If both L1 and L2 have only one token, "retokenizes" them using a space character (' ') as delimiter.
		if ( (num_tokensL2 * num_tokensL1) == 1 ) {

			// Sets the new delimiter.
			delimiter = (int) ' ';

			char *restrict data = NULL;

			#if NMFGPU_DEBUG_READ_MATRIX
				print_message( false, "Retokenizing dataL1...\n" );
			#endif

			// Removes previous array of pointers to tokens.
			data = (char *restrict) tokensL1.tokens;
			free( (void *) tokensL1.ptokens );
			num_tokensL1 = 0;

			tokensL1 = tokenize( data, delimiter, &num_tokensL1 );

			#if NMFGPU_DEBUG_READ_MATRIX
				print_message( false, "tokensL1: %s, num_tokensL1=%zu\n", ( tokensL1.tokens ? "non-NULL" : "NULL" ),
						num_tokensL1 );
			#endif

			if ( ! tokensL1.tokens ) {
				print_error( true, "Error in matrix_load_ascii_verb(): re-tokenize line 1.\n" );
				clean_tag(tokensL2);
				free( (void *) data );
				fclose(file);
				return EXIT_FAILURE;
			}

			// Checks for overflow.
			if ( (num_tokensL1-1) > (size_t) max_pitch ) {	// '-1' because the file may have a "name" field.
				print_message( false, "\n\t\tNumber of items read on line 1: %zu.\n", num_tokensL1 );
				print_error( false, "\nSorry, but your matrix exceeds the limits used for matrix dimensions.\n"
						"On this system and with the given input arguments, data matrices are limited to:\n\t* %"
						PRI_IDX " rows.\n\t*%" PRI_IDX " columns.\n\t*%zu items.\n\nPlease check also for any invalid "
						"line terminator. Only LF ('\\n') and CR+LF ('\\r\\n') are accepted.\n\n",
						max_non_padded_dim, max_pitch, max_num_items );
				clean_tag(tokensL2);
				clean_tag(tokensL1);
				fclose(file);
				return EXIT_FAILURE;
			}

			// ----------------------

			#if NMFGPU_DEBUG_READ_MATRIX
				print_message( false, "Retokenizing dataL2...\n" );
			#endif

			// Removes previous array of pointers to tokens.
			data = (char *restrict) tokensL2.tokens;
			free( (void *) tokensL2.ptokens );
			num_tokensL2 = 0;

			tokensL2 = tokenize( data, delimiter, &num_tokensL2 );

			#if NMFGPU_DEBUG_READ_MATRIX
				print_message( false, "tokensL2: %s, num_tokensL2=%zu\n", ( tokensL2.tokens ? "non-NULL" : "NULL" ),
						num_tokensL2 );
			#endif

			if ( ! tokensL2.tokens ) {
				print_error( true, "Error in matrix_load_ascii_verb(): re-tokenize line 2.\n" );
				free( (void *) data );
				clean_tag(tokensL1);
				fclose(file);
				return EXIT_FAILURE;
			}

		} // If "retokenize" both L1 and L2.

		// Number of lines processed.
		nlines = 2;

	} else {  // No name and no headers.

		// tokensL2 is matrix[0][0] or matrix[0][1]
		hasheaders = false;

		// If line 1 has only one token, "retokenizes" it using the space character (' ') as delimiter.
		if ( num_tokensL1 == 1 ) {

			#if NMFGPU_DEBUG_READ_MATRIX
				print_message( false, "Retokenizing dataL1...\n" );
			#endif

			// New delimiter.
			delimiter = (int) ' ';

			// Removes previous array of pointers to tokens.
			char *restrict data = (char *restrict) tokensL1.tokens;
			free( (void *) tokensL1.ptokens );
			num_tokensL1 = 0;

			tokensL1 = tokenize( data, delimiter, &num_tokensL1 );

			#if NMFGPU_DEBUG_READ_MATRIX
				print_message( false, "tokensL1: %s, num_tokensL1=%zu\n", ( tokensL1.tokens ? "non-NULL" : "NULL" ),
						num_tokensL1 );
			#endif

			if ( ! tokensL1.tokens ) {
				print_error( true, "Error in matrix_load_ascii_verb( re-tokenize line 1 ).\n" );
				clean_tag(tokensL2);
				free( (void *) data );
				fclose(file);
				return EXIT_FAILURE;
			}

		} // if retokenizes L1.

		// Sets L1 as L2 (first row with numeric data).
		tokensL2 = tokensL1;
		len_tokensL2 = len_tokensL1;
		num_tokensL2 = num_tokensL1;

		tokensL1 = new_empty_tag();
		len_tokensL1 = num_tokensL1 = 0;

		// Number of lines processed.
		nlines = 1;

	} // File might have name and/or headers


	/////////////////////////////////


	// Detects if file may have row labels and sets num_tokensL2 (or num_tokensL2-1) as the number of columns.

	if ( (! isnumber(tokensL2.tokens)) + numeric_lbls ) {	// First token in L2 is not numeric, or input matrix has numeric row labels.

		// File contains row labels.
		numcols = (index_t) (num_tokensL2 - 1);
		haslabels = true;
		print_message( false, "\t\tRow labels detected.\n\t\tNumber of data columns detected (excluding row labels): %"
				PRI_IDX ".\n", numcols );

	} else {
		// No labels: numcols = num_tokensL2

		// If data matrix seems to be numeric only.
		if ( ! has_name_headers )
			print_message( false, "\t\tNumeric-only data.\n" );

		numcols = (index_t) num_tokensL2;

		print_message( false, "\t\tNumber of data columns detected: %" PRI_IDX ".\n", numcols );

	} // If file contains row labels

	// Checks for invalid number of columns.
	if ( ( numcols <= 1 ) + ( (num_tokensL2 - haslabels) > (size_t) max_pitch ) ) {
		if ( numcols <= 1 )
			print_error( false, "\nError reading input file:\nInvalid file format or the number of columns is less than 2.\n"
					"Please remember that columns must be separated by TAB characters (or by single space "
					"characters under certain conditions).\nFinally, please check for any invalid decimal "
					"symbol (e.g., ',' instead of '.').\n\n" );
		else
			print_error( false, "\n\nSorry, but your matrix exceeds the limits used for matrix dimensions.\n"
					"On this system and with the given input arguments, data matrices are limited to:\n\t* %" PRI_IDX
					" rows\n\t*%" PRI_IDX " columns.\n\t* %zu items.\n\n", max_non_padded_dim, max_pitch, max_num_items );
		clean_tag(tokensL2);
		clean_tag(tokensL1);
		fclose(file);
		return EXIT_FAILURE;
	}

	// Padded dimension
	l_pitch = get_padding( numcols );

	/////////////////////////////////


	// Compares length of L1 and numcols to definitely know if there are name/headers or not.

	// File might have name and/or headers
	if ( has_name_headers ) {

		// tokensL1 != tokensL2

		if ( (num_tokensL1 - 1) == (size_t) numcols ) {	// Name and headers

			print_message( false, "\t\tName (i.e., description string) detected.\n" );

			// Splits tokensL1 as name and headers.

			char *restrict data = (char *restrict) tokensL1.tokens;
			char **restrict pdata = (char **restrict) tokensL1.ptokens;

			// Copies the first token.
			name = (char *restrict) malloc( (strlen(data) + 1) * sizeof(char) );
			if ( ! name ) {
				print_errnum( true, errno, "Error in matrix_load_ascii_verb(): malloc( name, size=%zu )", strlen(data) + 1 );
				clean_tag(tokensL2); clean_tag(tokensL1);
				fclose(file);
				return EXIT_FAILURE;
			}
			strcpy( name, data );

			#if NMFGPU_DEBUG_READ_MATRIX
				print_message( false, "\t\tName (len=%zu):'%s'\n", strlen(name), name );
			#endif

			// Headers
			print_message( false, "\t\tColumn headers detected.\n" );
			hasheaders = true;

			/* Headers: Sets remaining tokens as column headers.
			 *
			 * Moves the second token to the beginning of "tokensL1.tokens", overwriting the first token (which was already
			 * copied into 'name'). Remaining tokens are kept untouched, and the previous place of the second token is left
			 * as "garbage".
			 *
			 * Therefore, headers.tokens == headers.ptokens[0], and it is possible to call clean_tag(headers).
			 */
			if ( ! memmove( data, pdata[1], (strlen(pdata[1]) + 1) * sizeof(char) ) ) {
				print_errnum( true, errno, "Error in matrix_load_ascii_verb(): memmove( headers, size=%zu )",
						strlen(pdata[1]) + 1 );
				struct matrix_tags_t l_mt = new_matrix_tags( name, tokensL2, tokensL1 );
				clean_matrix_tags(l_mt);
				fclose(file);
				return EXIT_FAILURE;
			}

			// Adjusts pdata[] to <numcols> tokens
			pdata[ 0 ] = data;
			for ( index_t j = 0 ; j < numcols ; j++ )
				pdata[ j ] = pdata[ j+1 ];

			pdata = (char **restrict) realloc( pdata, numcols * sizeof(char *) );
			if ( ! pdata ) {
				print_errnum( true, errno, "Error in matrix_load_ascii_verb(): realloc( pheaders, numcols=%" PRI_IDX " )",
						numcols );
				struct matrix_tags_t l_mt = new_matrix_tags( name, tokensL2, tokensL1 );
				clean_matrix_tags(l_mt);
				fclose(file);
				return EXIT_FAILURE;
			}

			headers = new_tag( data, pdata );

		} else if ( num_tokensL1 == (size_t) numcols ) {	// No name, headers only.

			print_message( false, "\t\tColumn headers detected.\n" );
			hasheaders = true;
			headers = tokensL1;

		} else if ( num_tokensL1 == 1 ) {	// No headers, name only

			print_message( false, "\t\tName (i.e., description string) detected.\n" );
			name = (char *restrict) tokensL1.tokens;
			if ( tokensL1.ptokens )
				free( (void *)tokensL1.ptokens );

		} else {	// Error.

			print_error( false, "\nError reading input file: length of lines 1 (%zu) and 2 (%" PRI_IDX ") mismatch.\n"
					"Invalid file format or data are not separated.\nFinally, please check for any invalid "
					"decimal symbol.\n", num_tokensL1, numcols );
			clean_tag(tokensL2);
			clean_tag(tokensL1);
			fclose(file);
			return EXIT_FAILURE;

		} // If there are name and/or headers

		tokensL1 = new_empty_tag();

	} // If there can be name or headers

	// From here: tokensL1 == NULL


	/////////////////////////////////


	// Sets (the rest of) L2 as the first row of data matrix.


	data_matrix = (real *restrict) malloc( (size_t) max_numrows * (size_t) l_pitch * sizeof(real) );
	if ( ! data_matrix ) {
		print_errnum( true, errno, "Error in matrix_load_ascii_verb(): malloc( data_matrix, max_numrows=%" PRI_IDX ", numcols=%"
				PRI_IDX ", l_pitch=%" PRI_IDX " )", max_numrows, numcols, l_pitch );
		struct matrix_tags_t l_mt = new_matrix_tags( name, headers, tokensL2 );
		clean_matrix_tags(l_mt);
		fclose(file);
		return EXIT_FAILURE;
	}

	// Sums of columns: used to check that all columns have at least one positive value.
	sum_cols = (real *restrict) malloc( (size_t) l_pitch * sizeof(real) );
	if ( ! sum_cols ) {
		print_errnum( true, errno, "Error in matrix_load_ascii_verb(): malloc( sum_cols, l_pitch=%" PRI_IDX " )", l_pitch );
		struct matrix_tags_t l_mt = new_matrix_tags( name, headers, tokensL2 );
		matrix_clean(data_matrix, l_mt);
		fclose(file);
		return EXIT_FAILURE;
	}

	#if NMFGPU_DEBUG_READ_MATRIX
		print_message( false, "Setting first data line (file line %" PRI_IDX ")...\n", nlines );
	#endif

	{
		real sum_row = REAL_C( 0.0 );
		real min_value = R_MAX;

		for ( index_t j = 0 ; j < numcols; j++ ) {

			// Token (data) to read.
			char const *const val_str = tokensL2.ptokens[ j + haslabels ];
			char *endptr = NULL;

			#if NMFGPU_DEBUG_READ_MATRIX2
				print_message( false, "\tstr='%s'", val_str );
			#endif

			// Transforms char* to real.
			errno = 0;
			real const value = STRTOREAL( val_str, &endptr );

			#if NMFGPU_DEBUG_READ_MATRIX2
			{
				int const err = errno;
				print_message( false, "val=%g,endptr=", value );
				char const c = *endptr;
				print_message( false, ( (isprint(c) + isblank(c)) ? ("'%c'\n") : ("'\\x%X'\n") ), c );
				errno = err;
			}
			#endif

			// No numeric characters <OR> NaN, inf, underflow/overflow, etc
			if ( errno + (! is_valid(value)) + (*endptr) ) {	// (errno != 0) || (*endptr != '\0') || (! is_valid(value))
				print_errnum( false, errno, "\nError reading line %" PRI_IDX ", column %" PRI_IDX
						". Invalid numeric value: '%s'", nlines, (j + haslabels + 1), val_str );
				if ( ! errno )
					print_error( false, "Please, check also for invalid decimal symbols (if any).\n" );
				free(sum_cols);
				struct matrix_tags_t l_mt = new_matrix_tags( name, headers, tokensL2 );
				matrix_clean(data_matrix, l_mt);
				fclose(file);
				return EXIT_FAILURE;
			}

			// Stores the new value.
			data_matrix[ j ] = value;
			sum_cols[ j ] = value;
			sum_row += value;
			min_value = MIN( min_value, value );	// value has been checked already.

		} // for 0..numcols

		// Fails on "empty" row or there is a negative value.
		if ( (sum_row < R_MIN) + (min_value < REAL_C(0.0)) ) {
			print_error( false, "\nError in input file at line %" PRI_IDX ": ", nlines );
			if ( min_value < REAL_C( 0.0 ) )
				print_error( false, "negative value(s) detected (e.g., %g).\n", min_value );
			else
				print_error( false, "\"empty\" row detected.\nAll rows and columns must "
						"have at least one value greater than or equal to %" PRI_IDX "\n", R_MIN );
			free(sum_cols);
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, tokensL2 );
			matrix_clean(data_matrix, l_mt);
			fclose(file);
			return EXIT_FAILURE;
		}
	}

	numrows = 1;				  // One matrix row was read.
	size_t nitems = numcols;		  // Current number of data elements: numrows * numcols
	size_t nitems_p = l_pitch;		  // Similar to "nitems", but using "l_pitch" instead of "numcols"
	real *pmatrix = &data_matrix[ nitems_p ]; // Pointer to next row (numrows * l_pitch)


	/////////////////////////////////


	/* Reuses tokensL2 to store row labels (first row label is already there).
	 * Remaining memory (whose data were copied to data_matrix already) will be overwritten with future row labels.
	 */
	if ( haslabels ) {
		labels = tokensL2;
		max_len_labels = len_tokensL2 + 1;		// Allocated memory.
		len_labels = strlen(tokensL2.tokens) + 1;	// Memory currently used (first label only).
	}

	// From here: tokensL2 == NULL.
	tokensL2 = new_empty_tag();


	/////////////////////////////////


	do {

		nlines++;	// A new line is going to be read.

		#if NMFGPU_DEBUG_READ_MATRIX2
			print_message( false, "\n==============\nReading line %" PRI_IDX "...\n", nlines );
		#endif

		/////////////////////////////////////////

		// Checks for overflow.
		if ( ( ((uintmax_t) nitems_p + (uintmax_t) l_pitch) > (uintmax_t) max_num_items ) +
			( (uintmax_t) numrows >= (uintmax_t) max_non_padded_dim ) ) {
			print_message( false, "\t\tNumber of matrix rows currently read: %" PRI_IDX ".\n"
					"\t\tNumber of matrix entries currently read: %zu.\n", numrows, nitems );
			print_error( false, "\nSorry, but your matrix exceeds the limits used for matrix dimensions.\n"
				"On this system and with the given input arguments, data matrices are limited to:\n\t* %" PRI_IDX
				" rows.\n\t* %" PRI_IDX " columns.\n\t* %zu items.\n\n", max_non_padded_dim, max_pitch, max_num_items );
			free(sum_cols);
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
			matrix_clean( data_matrix, l_mt );
			fclose(file);
			return EXIT_FAILURE;
		}

		/////////////////////////////////////////

		/* Allocates more memory for data_matrix[] and sum_rows[], if necessary.
		 * NOTE: labels.ptokens[] will be adjusted after this loop.
		 */

		if ( numrows >= max_numrows ) {

			max_numrows *= 2;

			#if NMFGPU_DEBUG_READ_MATRIX
				print_message( false, "\tExpanding memory for a total of %" PRI_IDX " rows: ", max_numrows );
			#endif

			// data_matrix
			real *restrict const tmp = (real *restrict) realloc( data_matrix, (size_t) max_numrows * (size_t) l_pitch * sizeof(real) );
			if ( ! tmp )  {
				print_errnum( true, errno, "Error in matrix_load_ascii_verb(): realloc( data_matrix, max_numrows=%"
						PRI_IDX", numcols=%" PRI_IDX ", l_pitch=%" PRI_IDX " )", max_numrows, numcols, l_pitch );
				free(sum_cols);
				struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
				matrix_clean( data_matrix, l_mt );
				fclose(file);
				return EXIT_FAILURE;
			}
			data_matrix = tmp;
			pmatrix = &tmp[ nitems_p ];

		} // If allocate more memory for data_matrix

		// Reads the row label (if any) and a full matrix row.
		index_t const incr_numcols = 1;
		bool const skip_labels = false;
		real sum_row = REAL_C( 0.0 );
		real min_value = R_MAX;
		bool const real_data = true;
		status = matrix_read_line( file, nlines, numcols, incr_numcols, haslabels, skip_labels, delimiter, delimiter_scn_format,
					data_scn_format, real_data, pmatrix, &sum_row, sum_cols, &min_value,
					(char *restrict *restrict) &(labels.tokens), &len_labels, &max_len_labels );

		if ( (status != EXIT_SUCCESS) + (sum_row < R_MIN) + (min_value < REAL_C(0.0)) ) {
			if ( status != EXIT_SUCCESS )
				print_error( false, "\nError reading input file.\n" );
			else {
				print_error( false, "\nError reading input file at line: %" PRI_IDX ": ", nlines );
				if ( min_value < REAL_C(0.0) )
					print_error( false, "negative value(s) detected (e.g., %g).\n", min_value );
				else	// sum_row < R_MIN
					print_error( false, "\"empty\" row detected.\nAll rows and columns must "
							"have at least one value greater than or equal to %" PRI_IDX "\n", R_MIN );
			}
			free(sum_cols);
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
			matrix_clean( data_matrix, l_mt );
			fclose(file);
			return EXIT_FAILURE;
		}

		numrows++;
		nitems += (size_t) numcols;	// == numrows * numcols
		nitems_p += (size_t) l_pitch;	// == numrows * l_pitch
		pmatrix += l_pitch;

		// -----------------------------

		// Checks for blank lines or EOF.
		{
			uintmax_t l = UINTMAX_C( 0 );	// Number of blank lines detected, plus one.
			char chr[3] = { 0, 0, 0 };	// Newline to be read ("\r\n" or "\n"), followed by '\0'.
			int conv = 0;			// Number of matched items.

			do {
				errno = 0;
				conv = fscanf( file, "%2[\r\n]", chr );

				#if NMFGPU_DEBUG_READ_MATRIX2
				{
					bool const shown_by_all = false;
					int const err = errno;
					print_message( shown_by_all, "\tconv(next line)=%i,chr=", conv );	// conv: number of tokens read.
					if ( chr[0] ) {
						for ( size_t i = 0 ; i < 2 ; i++ )
							switch( chr[i] ) {
								case '\0': break;
								case '\r': { print_message( shown_by_all, "\\r" ); } break;
								case '\n': { print_message( shown_by_all, "\\n" ); } break;
								case '\t': { print_message( shown_by_all, "\\t" ); } break;
								case  ' ': { print_message( shown_by_all, "' '" ); } break;
								default  : {
									print_message(shown_by_all,(isgraph(c[0]) ? "'%c'" : "'\\0x%X'"),chr[i]);
								} break;
							}
					} else
						print_message( shown_by_all, "(empty)" );
					errno = err;
				}
				#endif

				l++;

			} while ( conv == 2 );


			switch( conv ) {

				case EOF: {
					if ( ferror(file) ) {
						print_errnum( true, errno, "Error in matrix_read_line() at line %" PRIuMAX
								": fscanf( blank_lines=%" PRIuMAX " )", current_line + l, l );
						return EXIT_FAILURE;
					}
				} break;

				case 1: {
					if
				}

			}






			int c[] = 0;
			while ( ( c == (int) '\r' ) + ( c == (int) '\n' ) ) {
				if ( c == (int) '\r' )
					fgetc(file);	// Skips the corresponding LF
				c = fgetc(file);
				l++;
			}
		}




		// Checks for EOF by reading one more character.
		c = fgetc(file);

		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( false, "Line %" PRI_IDX "+1 (checks for EOF): char=", nlines );
			if ( c == (int) '\n' ) print_message( false, "'\\n'.");
			else if ( c == (int) '\r' ) print_message( false, "'\\r'.");
			else if ( isprint(c) + isblank(c) ) print_message( false, "'%c'.\n", c);
			else if ( c == EOF ) print_message( false, "EOF.\n" );
			else print_message( false, "'\\x%X'.\n", c);
		#endif

		// Checks for blank lines.
		uintmax_t l = UINTMAX_C( 0 );
		while ( ( c == (int) '\r' ) + ( c == (int) '\n' ) ) {
			if ( c == (int) '\r' )
				fgetc(file);	// Skips the corresponding LF
			c = fgetc(file);
			l++;
		}

		if ( c != EOF ) { // There are more data lines.

			if ( l ) { // There were also blank lines. Invalid format

				#if NMFGPU_DEBUG_READ_MATRIX
					print_message( false, "\tThere were %" PRIuMAX " blank lines: last char: ", l );
					print_message( false, ( isgraph(c) ? ("'%c'.\n") : ("'\\x%X'.\n") ), c );
				#endif

				print_error( false, "\nError reading input file: No matrix data in line %" PRI_IDX "\n"
						"Invalid file format.\n\n", nlines + 1 );
				free(sum_cols);
				struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
				matrix_clean( data_matrix, l_mt );
				fclose(file);
				return EXIT_FAILURE;

			} else {	// No blank lines. Restores the last character read.
				c = ungetc( c, file );
				#if NMFGPU_DEBUG_READ_MATRIX
					if ( c == EOF ) print_error( true, "\tError: EOF from ungetc().\n" );
				#endif
			}
		}

		if ( ferror(file) ) {
			print_error( true, "Error in matrix_load_ascii_verb(): fgetc()/ungetc().\n");
			free(sum_cols);
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
			matrix_clean( data_matrix, l_mt );
			fclose(file);
			return EXIT_FAILURE;
		}

	} while ( c != EOF );

	fclose(file);

	print_message( false, "\t\tLoaded a %" PRI_IDX " x %" PRI_IDX " data matrix (%zu items).\n", numrows, numcols, nitems );

	// Fails on "empty" columns
	// NOTE: There are faster alternatives, but we want to tell the user which column is "empty".
	for ( index_t i = 0 ; i < numcols ; i++ )
		if ( sum_cols[ i ] < R_MIN ) {
			print_error( false, "\nError in input file: column %" PRI_IDX " is \"empty\".\nAll rows and columns must "
					"have at least one value greater than or equal to %" PRI_IDX "\n", i + hasheaders + 1, R_MIN );
			free(sum_cols);
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
			matrix_clean( data_matrix, l_mt );
			fclose(file);
			return EXIT_FAILURE;
		}
	free(sum_cols);

	// Adjusts allocated memory for labels and data matrix.
	if ( haslabels ) {
		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( false, "\t\tResizing labels from %zu to %zu, and plabels from %" PRI_IDX " to %" PRI_IDX "\n",
					max_len_labels, len_labels, max_numrows, numrows );
		#endif

		errno = 0;
		char const *restrict const data = (char const *restrict) realloc( (void *) labels.tokens, len_labels * sizeof(char) );
		if ( ! data ) {
			print_errnum( true, errno, "Error in matrix_load_ascii_verb(): realloc( labels, len_labels=%zu )", len_labels );
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
			matrix_clean( data_matrix, l_mt );
			fclose(file);
			return EXIT_FAILURE;
		}
		labels.tokens = data;

		errno = 0;
		char **restrict const pdata = (char **restrict) realloc( (void *) labels.ptokens, (size_t) numrows * sizeof(char *) );
		if ( ! pdata ) {
			print_errnum( true, errno, "Error in matrix_load_ascii_verb(): realloc( plabels, numrows=%" PRI_IDX " )", numrows );
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
			matrix_clean( data_matrix, l_mt );
			fclose(file);
			return EXIT_FAILURE;
		}
		labels.ptokens = (char const *const *restrict) pdata;

		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( false, "\t\"Retokenizing\" labels...\n" );
		#endif

		// Resets labels.ptokens[].
		retok( labels, numrows );
	}

	// Adjusts memory used by data_matrix
	{
		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( false, "Resizing matrix from %" PRI_IDX "rows to %" PRI_IDX "...\n", max_numrows, numrows );
		#endif
		real *restrict const tmp = (real *restrict) realloc( data_matrix, nitems_p * sizeof(real) );
		if ( ! tmp ) {
			print_errnum( true, errno, "Error in matrix_load_ascii_verb(): realloc( data_matrix, numrows=%" PRI_IDX " )", numrows );
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
			matrix_clean( data_matrix, l_mt );
			fclose(file);
			return EXIT_FAILURE;
		}
		data_matrix = tmp;
	}

	#if NMFGPU_DEBUG_READ_MATRIX
		print_message( false, "\tLoad matrix finished!\n");
	#endif

	// Sets output parameters.
	*matrix = data_matrix;
	*nrows = numrows;
	*ncols = numcols;
	*pitch = l_pitch;
	*mt = new_matrix_tags( name, headers, labels );

	return EXIT_SUCCESS;

} // matrix_load_ascii_verb

////////////////////////////////////////////////

/*
 * Loads a real-type matrix from an ASCII file.
 *
 * Skips name, headers and labels if 'mt' is NULL.
 *
 * If (*matrix) is non-NULL, do not allocates memory but uses the supplied one.
 * WARNING: In such case, THERE MUST BE ENOUGH MEMORY ALREADY ALLOCATED.
 *
 * If input file does not have labels, accepts both tab and space characters as delimiters.
 *
 * If 'transpose' is 'true', transposes matrix in memory as follows:
 * - Matrix dimensions in memory: <ncols> rows, <nrows> columns, padded to <pitch>.
 * - Matrix dimensions in file: <nrows> rows, <ncols> columns.
 * - Reads <ncols> column headers (set as mt->headers) and <nrows> row labels (set as mt->labels).
 *
 * WARNING:
 *	- If "transpose" is 'true', nrows must be <= pitch. Else, ncols must be <= pitch
 *	- NO error checking is performed to detect negative data or empty rows/columns.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_load_ascii( char const *restrict filename, index_t nrows, index_t ncols, index_t pitch, bool hasname, bool hasheaders, bool haslabels,
			bool transpose, real *restrict *restrict matrix, struct matrix_tags_t *restrict mt )
{

	bool const real_data = true;

	return matrix_read_ascii( filename, nrows, ncols, pitch, hasname, hasheaders, haslabels, transpose, real_data,
				(void *restrict *restrict) matrix, mt );

} // matrix_load_ascii

////////////////////////////////////////////////

/*
 * Loads a index_t-type matrix from an ASCII file.
 *
 * Skips name, headers and labels if 'mt' is NULL.
 *
 * If (*matrix) is non-NULL, do not allocates memory but uses the supplied one.
 * WARNING: In such case, THERE MUST BE ENOUGH MEMORY ALREADY ALLOCATED.
 *
 * If input file does not have labels, accepts both tab and space characters as delimiters.
 *
 * If 'transpose' is 'true', transposes matrix in memory as follows:
 * - Matrix dimensions in memory: <ncols> rows, <nrows> columns, padded to <pitch>.
 * - Matrix dimensions in file: <nrows> rows, <ncols> columns.
 * - Reads <ncols> column headers (set as mt->headers) and <nrows> row labels (set as mt->labels).
 *
 * WARNING:
 *	- If "transpose" is 'true', nrows must be <= pitch. Else, ncols must be <= pitch
 *	- NO error checking is performed to detect negative data or empty rows/columns.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_int_load_ascii( char const *restrict filename, index_t nrows, index_t ncols, index_t pitch, bool hasname, bool hasheaders,
			bool haslabels, bool transpose, index_t *restrict *restrict matrix, struct matrix_tags_t *restrict mt )
{

	bool const real_data = false;

	return matrix_read_ascii( filename, nrows, ncols, pitch, hasname, hasheaders, haslabels, transpose, real_data,
				(void *restrict *restrict) matrix, mt );

} // matrix_int_load_ascii

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Swaps all bytes to the inverse order.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE on invalid arguments.
 */
static void reverse_bytes( void const *restrict in, size_t size, void *restrict out )
{

	unsigned char const *const p_in = (unsigned char const *restrict) in;
	unsigned char *const p_out = (unsigned char *restrict) out;

	for ( size_t i = 0, j = size-1 ; i < size ; i++, j-- )
		p_out[ j ] = p_in[ i ];

	return EXIT_SUCCESS;

} // reverse_bytes

// ---------------------------------------------

/*
 * Reads the signature from a "formated" binary file: a 32-bits unsigned integer
 * in little-endian format.
 *
 * Returns EXIT_SUCCESS, or EXIT_FAILURE if the signature is invalid.
 */
static int read_signature( FILE *restrict file )
{

	uint32_t const valid_signature = BIN_FILE_SIGNATURE;
	uint32_t const file_signature = UINT32_C( 0 );

	// -----------------------------

	if ( ! fread( &file_signature, sizeof(uint32_t), 1, file ) ) {
		if ( feof(file) )
			print_error( false, "\nError reading input file:\nPremature end-of-file detected.\n"
					"Invalid file format.\n\n" );
		else if ( ferror(file) )
			print_error( true, "Error in read_signature(): fread().\n" );
		return EXIT_FAILURE;
	}

	// If this system is big-endian, reverses the byte order.
	if ( IS_BIG_ENDIAN() ) {
		uint32_t value = UINT32_C( 0 );
		reverse_bytes( &file_signature, sizeof(uint32_t), &value );
		file_signature = value;
	}

	// -----------------------------

	if ( file_signature != valid_signature ) {
		print_error( false, "\nError reading input file:\nInvalid signature: %" PRIX32 "\nInvalid file format.\n\n", file_signature );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // read_signature

// ---------------------------------------------

/*
 * Loads a matrix from a "formated" binary file: double-precision data in
 * little-endian format).
 *
 * If (*matrix) is non-NULL, do not allocates memory, but uses the supplied one.
 * In that case, THERE MUST BE ENOUGH MEMORY ALREADY ALLOCATED (padding included).
 *
 * If "transpose" is 'true', transposes matrix in memory as follows:
 *	- Matrix dimensions in memory: <*ncols> rows, <*nrows> columns, padded to <*pitch>.
 *	- Matrix dimensions in file: <*nrows> rows, <*ncols> columns.
 *
 * If "check_errors" is 'true', makes sure that:
 *	- All rows and columns must have at least one positive value (i.e., greater than 0).
 *	- There are no negative values.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int matrix_read_binary( FILE *restrict file, bool transpose, bool check_errors, index_t nrows, index_t ncols, index_t pitch,
				real *restrict *restrict matrix )
{

	// Allocates memory, if necessary.

	real *restrict data_matrix = *matrix;

	if ( ! data_matrix ) {
		index_t const dim_major = ( transpose ? numcols : numrows );
		data_matrix = (real *restrict) malloc( (size_t) dim_major * (size_t) pitch * sizeof(real) );
		if ( ! data_matrix ) {
			print_errnum( true, errno, "Error in matrix_read_binary(): malloc( data_matrix, numrows=%" PRI_IDX ", numcols=%"
					PRI_IDX ", pitch=%" PRI_IDX ", transpose=%i )", numrows, numcols, pitch, transpose );
			return EXIT_FAILURE;
		}
	}

	// -----------------------------

	// Starts reading...

	#if NMFGPU_DEBUG_READ_MATRIX2
		print_message( false, "\tReading data (transpose=%i)...\n\n", transpose );
	#endif

	// Step sizes for outer and inner loops.
	index_t incr_outer_loop = pitch;	// Step size for outer loop
	index_t incr_inner_loop = INDEX_C(1);	// Step size for inner loop.
	if ( transpose ) {
		incr_outer_loop = INDEX_C(1);
		incr_inner_loop = pitch;
	}

	// Sums of inner loop: used to check that all columns/rows have at least one positive value.
	real *restrict sum_inner_loop = NULL;
	if ( check_errors ) {
		sum_inner_loop = (real *restrict) calloc( numcols, sizeof(real) ); // NOTE: this is calloc(3), not malloc(3).
		if ( ! sum_inner_loop ) {
			print_errnum( true, errno, "Error in matrix_read_binary(): calloc( sum_inner_loop, numcols=%" PRI_IDX " )", numcols );
			if ( ! (*matrix) )
				free( (void *) data_matrix );
			return EXIT_FAILURE;
		}
	}

	real *pmatrix = data_matrix;
	for ( index_t i=0 ; i < numrows ; i++, pmatrix += incr_outer_loop ) {

		real sum_outer_loop = REAL_C( 0.0 );
		real min_value = R_MAX;

		real *pmatrix_r = pmatrix;
		for ( index_t j=0 ; j < numcols ; j++, pmatrix_r += incr_inner_loop ) {

			// Reads current data value.
			double value = 0.0;
			size_t const nread = fread( &value, sizeof(double), 1, file );	// Reads one double-precision value.

			// If this system is big-endian, reverses the byte order.
			if ( IS_BIG_ENDIAN() ) {
				double be_value = 0.0;
				reverse_bytes( &value, sizeof(double), &be_value );
				value = be_value;
			}
			real const num = (real) value;

			#if NMFGPU_DEBUG_READ_MATRIX2
				print_message( false, "%g ", value );
			#endif

			// Checks data.
			if ( ! ( nread * is_valid(num) ) ) {
				index_t r, c;
				if ( transpose ) { r = j; c = i; }
				else { r = i; c = j; }
				if ( feof(file) )
					print_error( false, "\nError reading input file:\nPremature end-of-file detected.\n"
							"Invalid file format.\n\n");
				else if ( ferror(file) )
					print_error( true, "Error in matrix_load_binary_verb(): fread( row %" PRI_IDX ", column %" PRI_IDX
							", transposed dimensions: %s )\n", r, c, (transpose ? "yes" : "no") );
				else	// ! is_valid(num)
					print_error( false, "\nError reading input file (row %" PRI_IDX ", column %" PRI_IDX ", transposed "
							"dimensions: %s): '%g'.\nInvalid numeric or file format.\n\n", r, c,
							(transpose ? "yes" : "no") );
				if ( check_errors ) free( sum_inner_loop );
				if ( ! (*matrix) )  free( (void *) data_matrix );
				return EXIT_FAILURE;
			}

			// Stores the new value.
			*pmatrix_r = num;

			if ( check_errors ) {
				sum_inner_loop[ j ] += num;
				sum_outer_loop += num;
				min_value = MIN( min_value, num );	// num has been checked already.
			}

		} // for j.

		#if NMFGPU_DEBUG_READ_MATRIX2
			print_message( false, "\n" );
		#endif

		// Fails on "empty" row/column, or negative value(s).
		if ( check_errors * ((sum_outer_loop < R_MIN) + (min_value < REAL_C(0.0))) ) {
			print_error( false, "\nError reading input file at row %" PRI_IDX " (transposed dimensions: %s): ", i,
					(transpose ? "yes" : "no") );
			if ( min_value < REAL_C(0.0) )
				print_error( false, "negative value(s) detected (e.g., %g).\n", min_value );
			else	// sum_outer_loop < R_MIN
				print_error( false, "\"empty\" row detected.\nAll rows and columns must "
						"have at least one value greater than or equal to %" PRI_IDX "\n", R_MIN );
			free( sum_inner_loop );
			if ( ! (*matrix) )  free( (void *) data_matrix );
			return EXIT_FAILURE;
		}

	} // for i

	#if NMFGPU_DEBUG_READ_MATRIX2
		print_message( false, "\n" );
	#endif

	// Fails on "empty" columns/rows
	if ( check_errors ) {
		// NOTE: There are faster alternatives, but we want to tell the user which row/column is "emtpy".
		for ( index_t j=0 ; j < numcols ; j++ )
			if ( sum_inner_loop[ j ] < R_MIN ) {
				print_error( false, "\nError in input file: column %" PRI_IDX " (transposed dimensions: %s) is \"empty\".\n"
						"All rows and columns must have at least one value greater than or equal to %" PRI_IDX "\n",
						j, (transpose ? "yes" : "no"), R_MIN );
				free( sum_inner_loop );
				if ( ! (*matrix) )  free( (void *) data_matrix );
				return EXIT_FAILURE;
			}
		free( sum_inner_loop );
	}

	// Sets output parameters.
	*matrix = data_matrix;

	return EXIT_SUCCESS;

} // matrix_read_binary

// ---------------------------------------------

/*
 * Reads labels, headers and name (as plain text) if they exists.
 * It also detects automatically the used delimiter symbol (space or tab character).
 *
 * verbose: If 'true', shows messages concerning the label type found.
 *
 * NOTE: This function is intended to be used when reading tag elements from a BINARY file.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int matrix_read_tags( FILE *restrict file, index_t numrows, index_t numcols, bool verbose, struct matrix_tags_t *restrict mt )
{

	// Name, headers and labels
	char *restrict name = NULL;
	struct tag_t headers = new_empty_tag();
	struct tag_t labels = new_empty_tag();

	size_t ntokens = 0;
	size_t len = 0;

	int status = 0;

	int const delimiter = (int) '\t';

	// -----------------------------

	// Checks for row labels

	#if NMFGPU_DEBUG_READ_MATRIX
		print_message( false, "\t\tReading row labels...\n" );
	#endif

	status = read_tag( file, delimiter, &labels, &len, &ntokens );

	if ( status ) {	// EOF, error, or invalid format
		if ( status >= 2 ) {	// Error or invalid format.
			if ( status == 2 )	// Internal error.
				print_error( true, "Error in matrix_read_tags( row labels ).\n" );
			else	// 3: Invalid format.
				print_error( false, "Please remember to set a '\\n' (new-line) character "
						"between column headers, row labels and the description string.\n\n" );
			return EXIT_FAILURE;
		}
		// Else, EOF
		return EXIT_SUCCESS;
	}
	// Else, success: one or more row labels were read.

	if ( len ) {	// Non-empty line: (len > 0) && (ntokens >= 1)

		if ( verbose )
			print_message( false, "\t\tRow labels detected.\n" );

		// If there is only one token, and there should be more, "retokenizes" using space as delimiter.
		if ( ( ntokens == 1 ) * ( numrows > 1 ) ) {

			#if NMFGPU_DEBUG_READ_MATRIX
				print_message( false, "\t\t\t\"Retokenizes\" using space...\n" );
			#endif

			// Removes previous array of pointers to tokens.
			char *restrict data = (char *restrict) labels.tokens;
			free( (void *) labels.ptokens );
			ntokens = 0;

			labels = tokenize( data, (int) ' ', &ntokens );

			#if NMFGPU_DEBUG_READ_MATRIX
				print_message( false, "\t\t\tResulting number of tokens (with space): %zu\n", ntokens );
			#endif

			if ( ! labels.tokens ) {
				print_error( true, "Error in matrix_read_tags().\n" );
				free( (void *) data );
				return EXIT_FAILURE;
			}

		} // If it must "retokenize" the string

		if ( ntokens != (size_t) numrows ) {
			print_error( false, "\nError reading input file: %zu row labels found, %" PRI_IDX " expected.\n", ntokens, numrows );
			clean_tag( labels );
			return EXIT_FAILURE;
		}

	} else {	// Just a newline was read: (len == 0) && (ntokens == 1)
		clean_tag( labels );
		labels = new_empty_tag();
	}

	// --------------------------

	// Checks for column headers

	#if NMFGPU_DEBUG_READ_MATRIX
		print_message( false, "\t\tReading column headers...\n" );
	#endif

	ntokens = 0;
	len = 0;

	status = read_tag( file, delimiter, &headers, &len, &ntokens );

	if ( status ) {	// EOF, error, or invalid format
		if ( status >= 2 ) {	// Error or invalid format.
			if ( status == 2 )	// Internal error.
				print_error( true, "Error in matrix_read_tags( column headers ).\n" );
			else	// 3: Invalid format.
				print_error( false, "Please remember to set a '\\n' (new-line) character "
						"between column headers, row labels and the description string.\n\n" );
			return EXIT_FAILURE;
		}
		// Else, EOF
		*mt = new_matrix_tags( NULL, new_empty_tag(), labels );
		return EXIT_SUCCESS;
	}

	// Else, success: one or more row labels were read.

	if ( len ) {	// Non-empty line: (len > 0) && (ntokens >= 1)

		if ( verbose )
			print_message( false, "\t\tColumn headers detected.\n" );

		// If there is only one token, and there should be more, "retokenizes" using space as delimiter.
		if ( ( ntokens == 1 ) * ( numcols > 1 ) ) {

			#if NMFGPU_DEBUG_READ_MATRIX
				print_message( false, "\t\t\t\"Retokenizes\" using space...\n" );
			#endif

			// Removes previous array of pointers to tokens.
			char *restrict data = (char *restrict) headers.tokens;
			free( (void *) headers.ptokens );
			ntokens = 0;

			headers = tokenize( data, (int) ' ', &ntokens );

			#if NMFGPU_DEBUG_READ_MATRIX
				print_message( false, "\t\t\tResulting number of tokens (with space): %zu\n", ntokens );
			#endif

			if ( ! headers.tokens ) {
				print_error( true, "Error in matrix_read_tags().\n" );
				free( (void *) data );
				clean_tag( labels );
				return EXIT_FAILURE;
			}

		} // If it must "retokenize" the string

		if ( ntokens != (size_t) numcols ) {
			print_error( false, "\nError reading input file: %zu row labels found, %" PRI_IDX " expected.\n", ntokens, numcols );
			clean_tag( headers );
			clean_tag( labels );
			return EXIT_FAILURE;
		}

	} else {	// Just a newline was read: (len == 0) && (ntokens == 1)
		clean_tag( headers );
		headers = new_empty_tag();
	}

	// --------------------------

	// Checks for name.

	#if NMFGPU_DEBUG_READ_MATRIX
		print_message( false, "\t\tReading name...\n" );
	#endif

	len = read_line( file, &name );

	#if NMFGPU_DEBUG_READ_MATRIX
		print_message( false, "\t\t\tName (len=%zu): '%s'\n", len, name );
	#endif

	if ( len ) {	// Success, error, or invalid format.
		if ( ! name ) {	// Error or invalid format.
			if ( len == 1 )	// Internal error.
				print_error( true, "Error in matrix_read_tags( name ).\n" );
			struct matrix_tags_t l_mt = new_matrix_tags( NULL, headers, labels );
			clean_matrix_tags( l_mt );
			return EXIT_FAILURE;
		}
		// Else, success
		if ( verbose )
			print_message( false, "\t\tName (i.e., description string) detected.\n" );
	}
	// Else, nothing read: just a newline (name != NULL), or EOF (name == NULL).

	*mt = new_matrix_tags( name, headers, labels );

	return EXIT_SUCCESS;

} // matrix_read_tags

////////////////////////////////////////////////

/*
 * Loads a real-type matrix from a "formated" binary file: double-precision data,
 * and 32-bits unsigned integers for matrix dimensions and the file signature,
 * all of them in little-endian format.
 *
 * Detects automatically if matrix has name, column headers and/or row labels,
 * as well as the employed delimiter symbol (space or tab character).
 *
 * In addition, outputs information messages, and performs error checking.
 *
 * The number of columns is rounded up to a multiple of <memory_alignment>.
 *
 * WARNING:
 *	- Both matrix dimensions must be >= 2.
 *	- All rows and columns must have at least one positive value (i.e., greater than 0).
 *	- Negative values are NOT allowed.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_load_binary_verb( char const *restrict filename, real *restrict *restrict matrix, index_t *restrict nrows, index_t *restrict ncols,
				index_t *restrict pitch, struct matrix_tags_t *restrict mt )
{

	index_t numcols = INDEX_C(0), numrows = INDEX_C(0), l_pitch = INDEX_C(0);

	int status = EXIT_SUCCESS;

	// Limits on matrix dimensions
	size_t const max_num_items = matrix_max_num_items;
	index_t const max_pitch = matrix_max_pitch;
	index_t const max_non_padded_dim = matrix_max_non_padded_dim;

	/////////////////////////////////

	// Checks for NULL parameters
	if ( ! ( (uintptr_t) filename * (uintptr_t) matrix * (uintptr_t) nrows * (uintptr_t) ncols * (uintptr_t) pitch * (uintptr_t) mt ) ) {
		bool const shown_by_all = false;
		int const errnum = EFAULT;
		if ( ! filename ) print_errnum( shown_by_all, errnum, "\nmatrix_load_binary_verb( filename )" );
		if ( ! matrix )	print_errnum( shown_by_all, errnum, "\nmatrix_load_binary_verb( matrix )" );
		if ( ! nrows )	print_errnum( shown_by_all, errnum, "\nmatrix_load_binary_verb( nrows )" );
		if ( ! ncols )	print_errnum( shown_by_all, errnum, "\nmatrix_load_binary_verb( ncols )" );
		if ( ! pitch )	print_errnum( shown_by_all, errnum, "\nmatrix_load_binary_verb( pitch )" );
		if ( ! mt )	print_errnum( shown_by_all, errnum, "\nmatrix_load_binary_verb( mt )" );
		return EXIT_FAILURE;
	}

	// ------------------------------------

	FILE *restrict const file = fopen( filename, "rb" );
	if ( ! file ) {
		print_errnum( true, errno, "Error in matrix_load_binary_verb(): fopen( %s )", filename );
		return EXIT_FAILURE;
	}

	// Checks file signature
	if ( read_signature( file ) != EXIT_SUCCESS ) {
		fclose(file);
		return EXIT_FAILURE;
	}

	// ------------------------------------

	// Reads matrix dimensions.
	{
		uint32_t dim[2] = { UINT32_C(0), UINT32_C(0) };

		size_t const nread = fread( dim, sizeof(uint32_t), 2, file );
		if ( nread != 2 ) {
			if ( feof(file) )
				print_error( false, "\nError reading input file:\nPremature end-of-file detected. Invalid file format.\n\n");
			else // error
				print_error( true, "Error in matrix_load_binary_verb(): fread( dim[2] ).\n" );
			fclose(file);
			return EXIT_FAILURE;
		}

		// Changes to big-endian, if necessary.
		if ( IS_BIG_ENDIAN() ) {
			uint32_t value = UINT32_C( 0 );
			reverse_bytes( &dim[0], sizeof(uint32_t), &value );
			dim[0] = value;
			value = UINT32_C( 0 );
			reverse_bytes( &dim[1], sizeof(uint32_t), &value );
			dim[1] = value;
		}

		uint_fast64_t const nitems = ((uint_fast64_t) dim[0]) * ((uint_fast64_t) dim[1]);

		print_message( false, "\t\tSize of input matrix to read: %" PRIu32 " x %" PRIu32 " (%" PRIuFAST64 " items)\n",
				dim[0], dim[1], nitems );

		if ( (dim[0] < 2) + (dim[1] < 2) + ((size_t) dim[0] > (size_t) max_non_padded_dim) +
			((size_t) dim[1] > (size_t) max_pitch) + (nitems > (uint_fast64_t) max_num_items) ) {
			if ( (dim[0] < 2) + (dim[1] < 2) )
				print_error( false, "\nError reading input file: both matrix dimensions must be greater than 1.\n\n" );
			else
				print_error( false, "\n\nSorry, but your matrix exceeds the limits used for matrix dimensions.\n"
						"On this system and with the given input arguments, data matrices are limited to:\n\t* %"
						PRI_IDX " rows.\n\t* %" PRI_IDX " columns.\n\t* %zu items.\n", max_non_padded_dim,
						max_pitch, max_num_items );
			fclose(file);
			return EXIT_FAILURE;
		}

		// Changes values to index_t
		numrows = (index_t) dim[0];
		numcols = (index_t) dim[1];
		l_pitch = get_padding( numcols );

	} // Reads matrix dimensions.

	// ------------------------------------

	// Reads matrix data

	bool const transpose = false;
	bool const check_errors = true;

	status = matrix_read_binary( file, transpose, check_errors, numrows, numcols, l_pitch, matrix );

	if ( status != EXIT_SUCCESS ) {
		fclose(file);
		return EXIT_FAILURE;
	}

	//-------------------------------------

	// Reads labels, headers and name (as plain text) if they exists.

	status = matrix_read_tags( file, numrows, numcols, true, mt );	// (verbose mode)

	fclose(file);

	if ( status != EXIT_SUCCESS ) {
		free( (void *) *matrix );
		*matrix = NULL;
		return EXIT_FAILURE;
	}

	*nrows = numrows;
	*ncols = numcols;
	*pitch = l_pitch;

	return EXIT_SUCCESS;

} // matrix_load_binary_verb

////////////////////////////////////////////////

/*
 * Loads a real-type matrix from a "formated" binary file: double-precision data,
 * and 32-bits unsigned integers for matrix dimensions and the file signature,
 * all of them in little-endian format.
 *
 * Detects automatically if matrix has name, column headers and/or row labels, as well as the used
 * delimiter symbol (space or tab character). Skips all of them if 'mt' is NULL.
 *
 * If (*matrix) is non-NULL, do not allocates memory but uses the supplied one.
 * In such case, THERE MUST BE ENOUGH MEMORY ALREADY ALLOCATED (padding included).
 *
 * If 'transpose' is 'true', transposes matrix in memory as follows:
 * - Matrix dimensions in memory: <numcols> rows, <numrows> columns, padded to <pitch>.
 * - Matrix dimensions in file: <numrows> rows, <numcols> columns.
 * - Reads <numcols> column headers (set as mt->headers) and <numrows> row labels (set as mt->labels).
 *
 * Fails if dimensions stored in file mismatch with numrows and numcols.
 *
 * WARNING:
 *	- If "transpose" is 'true', numrows must be <= pitch. Else, numcols must be <= pitch
 *	- NO error-checking is performed to detect negative data or empty rows/columns.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_load_binary( char const *restrict filename, index_t numrows, index_t numcols, index_t pitch, bool transpose,
			real *restrict *restrict matrix, struct matrix_tags_t *restrict mt )
{

	int status = EXIT_SUCCESS;

	// Limits on matrix dimensions
	size_t const max_num_items = matrix_max_num_items;
	index_t const max_pitch = matrix_max_pitch;
	index_t const max_non_padded_dim = matrix_max_non_padded_dim;

	//////////////////////////////

	// Checks for NULL parameters
	if ( ! ( (uintptr_t) filename * (uintptr_t) matrix ) ) {
		bool const shown_by_all = false;
		int const errnum = EFAULT;
		if ( ! filename ) print_errnum( shown_by_all, errnum, "\nmatrix_load_binary( filename )" );
		if ( ! matrix )	print_errnum( shown_by_all, errnum, "\nmatrix_load_binary( matrix )" );
		return EXIT_FAILURE;
	}

	// Checks provided matrix dimensions.
	{
		index_t const dim_major = ( transpose ? numcols : numrows );
		uintmax_t const nitems = ((uintmax_t) dim_major) * ((uintmax_t) pitch);

		if ( (numrows <= 0) + (numcols <= 0) + (pitch <= 0) + (pitch > max_pitch) + (dim_major > max_non_padded_dim) +
			(nitems > (uintmax_t) max_num_items) ) {
			if ( (numrows <= 0) + (numcols <= 0) + (pitch <= 0) )
				print_error( false, "\nError in matrix_load_binary( rows=%" PRI_IDX ", columns=%" PRI_IDX ", pitch=%"
						PRI_IDX "): Invalid matrix dimensions.\n", numrows, numcols, pitch );
			else
				print_error( false, "\n\nSorry, but your matrix exceeds the limits used for matrix dimensions.\n"
						"On this system and with the given input arguments, data matrices are limited to:\n\t* %"
						PRI_IDX " rows.\n\t* %" PRI_IDX " columns.\n\t* %zu items.\n", max_non_padded_dim,
						max_pitch, max_num_items );
			return EXIT_FAILURE;
		}
		if ( transpose ) {
			if ( numrows > pitch ) {
				print_error( false, "\nError in matrix_load_binary( rows=%" PRI_IDX ", pitch=%" PRI_IDX
						", transpose: yes ): Invalid values.\n", numrows, pitch );
				return EXIT_FAILURE;
			}
		} else if ( numcols > pitch ) {
			print_error( false, "\nError in matrix_load_binary( columns=%" PRI_IDX ", pitch=%" PRI_IDX " ): Invalid values.\n",
					numcols, pitch );
				return EXIT_FAILURE;
		}
	}

	// ------------------------------------

	#if NMFGPU_DEBUG_READ_MATRIX
		print_message( false, "\nReading '%s'...\n", filename );
	#endif

	FILE *restrict const file = fopen( filename, "rb" );
	if ( ! file ) {
		print_errnum( true, errno, "Error in matrix_load_binary(): fopen( %s )", filename );
		return EXIT_FAILURE;
	}

	// Checks file signature
	if ( read_signature( file ) != EXIT_SUCCESS ) {
		fclose(file);
		return EXIT_FAILURE;
	}

	// ------------------------------------

	// Reads matrix dimensions.
	{
		uint32_t dim[2] = { UINT32_C(0), UINT32_C(0) };

		size_t const nread = fread( dim, sizeof(uint32_t), 2, file );
		if ( nread != 2 ) {
			if ( feof(file) )
				print_error( false, "\nError reading input file:\nPremature end-of-file detected. Invalid file format.\n\n");
			else // error
				print_error( true, "\nError in matrix_load_binary(): fread( dim[2] ).\n" );
			fclose(file);
			return EXIT_FAILURE;
		}

		// Changes to big-endian, if necessary.
		if ( IS_BIG_ENDIAN() ) {
			uint32_t value = UINT32_C( 0 );
			reverse_bytes( &dim[0], sizeof(uint32_t), &value );
			dim[0] = value;
			value = UINT32_C( 0 );
			reverse_bytes( &dim[1], sizeof(uint32_t), &value );
			dim[1] = value;
		}

		#if NMFGPU_DEBUG_READ_MATRIX
		{
			uint_fast64_t nitems = (uint_fast64_t) dim[0] * (uint_fast64_t) dim[1];
			print_message( false, "\tMatrix Dimensions: %" PRIu32 "x%" PRIu32 " (%" PRIuFAST64 " items)\n",
					dim[0], dim[1], nitems );
		}
		#endif

		// Checks file dimensions.
		if ( (numrows != (index_t) dim[0]) + (numcols != (index_t) dim[1]) ) {
		print_error( false, "\nError in matrix_load_binary( transpose=%i ): matrix dimensions mismatch: %" PRIu32 " x %" PRIu32
				" read, %" PRI_IDX " x %" PRI_IDX " expected.\n", transpose, dim[0], dim[1], numrows, numcols );
			fclose(file);
			return EXIT_FAILURE;
		}
	} // Reads matrix dimensions.

	// ------------------------------------

	// Reads matrix data

	bool const check_errors = false;

	status = matrix_read_binary( file, transpose, check_errors, numrows, numcols, matrix );
	if ( status != EXIT_SUCCESS ) {
		print_error( true, "Error in matrix_load_binary()\n" );
		fclose(file);
		return EXIT_FAILURE;
	}

	//-------------------------------------

	// Reads labels, headers and name (as plain text) if they exists.

	if ( mt ) {
		status = matrix_read_tags( file, numrows, numcols, false, mt );	// (non-verbose mode)
		if ( status != EXIT_SUCCESS ) {
			print_error( true, "Error in matrix_load_binary()\n" );
			if ( ! (*matrix) ) {
				free( (void *) *matrix );
				*matrix = NULL;
			}
		}
	}

	fclose(file);

	return status;

} // matrix_load_binary

////////////////////////////////////////////////

/*
 * Loads a matrix from a "native" binary file (i.e., with the native endiannes,
 * and the compiled types for matrix data and dimensions; no file signature).
 *
 * Reads <rows_to_read> full rows, starting at <starting_row>.
 *
 * Detects automatically if matrix has name, column headers and/or row labels,
 * unless 'mt' is NULL.
 *
 * If "matrix" is NULL, skips data matrix; just reads matrix dimensions (and
 * labels, if mt is non-NULL). Else, and if in addition, *matrix is non-NULL,
 * do not allocates memory for the data matrix, but uses the supplied one.
 *
 * If <starting_row + rows_to_read> is greater than the number of rows stored in
 * the file, skips matrix data. In contrast, if "rows_to_read" is zero, reads all
 * rows starting at <starting_row>.
 *
 * In verbose mode, shows information messages, such as matrix dimensions and
 * labels found.
 *
 * If *file_nrows, *file_ncols and *pitch are zero, they are used to return the
 * matrix dimensions stored in the file, and the pitch used in memory (if any
 * matrix row is read), respectively. In contrast, if the two former are non-zero,
 * they are used to check the values stored in the file, failing if they differ.
 *
 * The number of columns is always rounded up to a multiple of <memory_alignment>.
 *
 * WARNING:
 *	- For internal-use only (i.e., for temporary files).
 *	- If *matrix is non-NULL, IT MUST HAVE ENOUGH MEMORY ALREADY ALLOCATED,
 *	  padding included.
 *	- NO ERROR-CHECKING IS PERFORMED (e.g., overflow, invalid values,
 *	  negative data, etc).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_load_binary_native( char const *restrict filename, index_t starting_row, index_t rows_to_read, bool verbose, size_t data_size,
				void *restrict *restrict matrix, index_t *restrict file_nrows, index_t *restrict file_ncols,
				index_t *restrict pitch, struct matrix_tags_t *restrict mt )
{

	index_t numcols = INDEX_C(0), numrows = INDEX_C(0), l_pitch = INDEX_C(0);

	void *restrict data_matrix = NULL;

	bool read_matrix = false;	// If data matrix was read.

	bool memory_allocated = false;	// If memory was allocated for data matrix.

	/////////////////////////////////

	// Checks for NULL parameters
	if ( ! ( (uintptr_t) filename * (uintptr_t) file_nrows * (uintptr_t) file_ncols * (uintptr_t) pitch ) ) {
		bool const shown_by_all = false;
		int const errnum = EFAULT;
		if ( ! filename ) print_errnum( shown_by_all, errnum, "\nmatrix_load_binary_native( filename )" );
		if ( ! file_nrows ) print_errnum( shown_by_all, errnum, "\nmatrix_load_binary_native( file_nrows )" );
		if ( ! file_ncols ) print_errnum( shown_by_all, errnum, "\nmatrix_load_binary_native( file_ncols )" );
		if ( ! pitch )	print_errnum( shown_by_all, errnum, "\nmatrix_load_binary_native( pitch )" );
		return EXIT_FAILURE;
	}

	// Checks for invalid parameters
	if ( (! data_size) + (starting_row < 0) + (rows_to_read < 0) + (*file_nrows < 0) + (*file_ncols < 0) ) {
		bool const shown_by_all = false;
		int const errnum = EINVAL;
		if ( ! data_size ) print_errnum( shown_by_all, errnum, "\nmatrix_load_binary_native( data_size=%zu )", data_size );
		if ( starting_row < 0 ) print_errnum(shown_by_all,errnum,"\nmatrix_load_binary_native( starting_row=%" PRI_IDX " )",starting_row);
		if ( rows_to_read < 0 ) print_errnum(shown_by_all,errnum,"\nmatrix_load_binary_native( rows_to_read=%" PRI_IDX " )",rows_to_read);
		if ( *file_nrows < 0 ) print_errnum(shown_by_all,errnum,"\nmatrix_load_binary_native( file_nrows=%" PRI_IDX " )",*file_nrows );
		if ( *file_ncols < 0 ) print_errnum(shown_by_all,errnum,"\nmatrix_load_binary_native( file_ncols=%" PRI_IDX " )",*file_ncols );
		return EXIT_FAILURE;
	}

	// ------------------------------------

	#if NMFGPU_DEBUG_READ_MATRIX
		print_message( false, "\nReading '%s'...\n", filename );
	#endif

	FILE *restrict const file = fopen( filename, "rb" );
	if ( ! file ) {
		print_errnum( true, errno, "Error in matrix_load_binary_native(): fopen( %s )", filename );
		return EXIT_FAILURE;
	}

	// ------------------------------------

	// Reads matrix dimensions.
	{
		index_t dim[2] = { INDEX_C(0), INDEX_C(0) };

		size_t const nread = fread( dim, sizeof(index_t), 2, file );
		if ( nread != 2 ) {
			if ( feof(file) )
				print_error( false, "\nError reading input file:\nPremature end-of-file detected. Invalid file format.\n\n");
			else // error
				print_error( true, "\nError in matrix_load_binary_native(): fread( dim[2] ).\n" );
			fclose(file);
			return EXIT_FAILURE;
		}

		uintmax_t const nitems = ((uintmax_t) dim[0]) * ((uintmax_t) dim[1]);

		if ( verbose )
			print_message( false, "\t\tMatrix dimensions: %" PRI_IDX " x %" PRI_IDX " (%" PRIuMAX " items)\n",
					dim[0], dim[1], nitems );

		if ((dim[0] <= 0) + (dim[1] <= 0) + (dim[0] > max_non_padded_dim) + (dim[1] > max_pitch) + (nitems > (uintmax_t)max_num_items)) {
			if ( (dim[0] < 2) + (dim[1] < 2) )
				print_error( false, "\nError reading input file: both matrix dimensions must be greater than zero.\n\n" );
			else
				print_error( false, "\n\nSorry, but your matrix exceeds the limits used for matrix dimensions.\n"
						"On this system and with the given input arguments, data matrices are limited to:\n\t* %"
						PRI_IDX " rows.\n\t* %" PRI_IDX " columns.\n\t* %zu items.\n", max_non_padded_dim,
						max_pitch, max_num_items );
			fclose(file);
			return EXIT_FAILURE;
		}

		// ------------------------------------

		// Checks matrix dimensions, if provided.

		if ( (*file_nrows + *file_ncols) * ((*file_nrows != (index_t) dim[0]) + (*file_ncols != (index_t) dim[1])) ) {
			print_error( false, "\nError in matrix_load_binary( transpose=%i ): matrix dimensions mismatch: %" PRIu32 " x %" PRIu32
				" read, %" PRI_IDX " x %" PRI_IDX " expected.\n", transpose, dim[0], dim[1], *file_nrows, *file_ncols );
			fclose(file);
			return EXIT_FAILURE;
		}

		numrows = dim[0];
		numcols = dim[1];
		l_pitch = get_padding( numcols );

	} // Reads matrix dimensions

	// ------------------------------------

	// Reads matrix data

	// If data matrix will be read.
	read_matrix = ( ((uintptr_t) matrix) * ((starting_row + rows_to_read) <= numrows) );

	if ( read_matrix ) {

		/* If the number of rows to read was not specified, reads all
		 * the matrix, starting at <starting_row>.
		 */
		if ( ! rows_to_read )
			rows_to_read = numrows - starting_row;

		data_matrix = (real *restrict) (*matrix);

		// Allocates memory, if necessary.
		if ( ! data_matrix ) {

			#if NMFGPU_DEBUG_READ_MATRIX
				size_t const items_to_read = (size_t) rows_to_read * numcols;
				print_message( false, "\tAllocating memory for %zu items: %" PRI_IDX " rows, %" PRI_IDX " columns (%"
						PRI_IDX " with padding)...\n", items_to_read, rows_to_read, numcols, l_pitch );
			#endif

			size_t const items_to_read_padding = (size_t) rows_to_read * l_pitch;

			data_matrix = malloc( items_to_read_padding * data_size );
			if ( ! data_matrix ) {
				print_errnum( true, errno, "Error in matrix_load_binary_native(): malloc( data_matrix, rows=%" PRI_IDX
						", numcols=%" PRI_IDX ", padding=%" PRI_IDX " )", rows_to_read, numcols, l_pitch );
				fclose(file);
				return EXIT_FAILURE;
			}

			memory_allocated = true;
		}

		// ------------------------------------

		// "Jumps" to row <starting_row>.
		{
			#if NMFGPU_DEBUG_READ_MATRIX
				print_message( false, "\tSkipping %" PRI_IDX " rows...\n", starting_row );
			#endif

			size_t const offset = starting_row * numcols;	// Offset in file, not in memory.

			if ( offset && fseek( file, offset * data_size, SEEK_CUR ) ) {
				print_errnum( true, errno, "Error in matrix_load_binary_native(): fseek( starting_row=%" PRI_IDX ", numcols=%"
						PRI_IDX " )", starting_row, numcols );
				if ( memory_allocated )
					free( (void *) data_matrix );
				fclose(file);
				return EXIT_FAILURE;
			}
		}

		// ------------------------------------

		// Reads the data matrix.

		if ( verbose )
			print_message( false, "\tReading an %" PRI_IDX "-by-%" PRI_IDX " data matrix, starting at row %" PRI_IDX "...\n",
					rows_to_read, numcols, starting_row );

		void *pmatrix = data_matrix;
		for ( index_t i=0 ; i < rows_to_read ; i++, pmatrix += (l_pitch * data_size) ) {

			size_t const nread = fread( pmatrix, data_size, numcols, file );

			if ( nread != (size_t) numcols ) {
				if ( ferror(file) )
					print_errnum( true, errno, "Error in matrix_load_binary_native(): fread( row %" PRI_IDX "/%" PRI_IDX
							", %" PRI_IDX " columns)", i, rows_to_read, numcols );
				else	// EOF
					print_error( false, "\nError reading input file:\nPremature end-of-file detected. "
							"Invalid file format.\n\n");
				if ( memory_allocated )
					free( (void *) data_matrix );
				fclose(file);
				return EXIT_FAILURE;
			}

		}

	} else { // Skips all data matrix

		size_t const nitems = (size_t) numrows * numcols;

		if ( verbose )
			print_message( false, "\tSkipping data matrix...\n" );

		if ( fseek( file, nitems * data_size, SEEK_CUR ) ) {
			print_errnum( true, errno, "Error in matrix_load_binary_native(): fseek( matrix_size: %zu items )", nitems );
			fclose(file);
			return EXIT_FAILURE;
		}

	} // If read data matrix

	/////////////////////////////////


	// Reads headers, labels and name (as plain text) if they exists.

	if ( mt && ( matrix_read_tags( file, numrows, numcols, verbose, mt ) != EXIT_SUCCESS ) ) {
		print_error( true, "Error in matrix_load_binary_native()\n" );
		if ( read_matrix * memory_allocated )
			free( (void *) data_matrix );
		fclose(file);
		return EXIT_FAILURE;
	}

	fclose(file);

	// ------------------------------------

	// Sets output values.

	*file_nrows = numrows;
	*file_ncols = numcols;
	*pitch = l_pitch;
	if ( read_matrix * memory_allocated )
		*matrix = data_matrix;

	return EXIT_SUCCESS;

} // matrix_load_binary_native

////////////////////////////////////////////////

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
		index_t *restrict nrows, index_t *restrict ncols, index_t *restrict pitch, struct matrix_tags_t *restrict mt )
{

	// Checks for NULL parameters
	if ( ! ( (uintptr_t) filename * (uintptr_t) matrix * (uintptr_t) nrows * (uintptr_t) ncols * (uintptr_t) pitch * (uintptr_t) mt ) ) {
		bool const shown_by_all = false;
		int const errnum = EFAULT;
		if ( ! filename ) print_errnum( shown_by_all, errnum, "\nmatrix_load( filename )" );
		if ( ! matrix )	print_errnum( shown_by_all, errnum, "\nmatrix_load( matrix )" );
		if ( ! nrows )	print_errnum( shown_by_all, errnum, "\nmatrix_load( nrows )" );
		if ( ! ncols )	print_errnum( shown_by_all, errnum, "\nmatrix_load( ncols )" );
		if ( ! pitch )	print_errnum( shown_by_all, errnum, "\nmatrix_load( pitch )" );
		if ( ! mt )	print_errnum( shown_by_all, errnum, "\nmatrix_load( mt )" );
		return EXIT_FAILURE;
	}

	if ( is_bin < 0 ) {
		print_errnum( false, EINVAL, "\nError in matrix_load( is_bin=%" PRI_IDX " )", is_bin );
		return EXIT_FAILURE;
	}

	// Initializes matrix dimensions
	*pitch = *ncols = *nrows = 0;

	int status = EXIT_SUCCESS;

	// -------------------------------

	// Loads the file.

	print_message( false, "\nLoading input file...\n" );

	if ( is_bin > 1 ) { // Input file is "native" binary.

		print_message( false, "\tFile selected as \"native\" binary (i.e., the file is read using the data types specified "
					"at compilation).\n\tNo error-checking is performed.\n\tLoading...\n" );

		index_t const starting_row = 0;
		index_t const rows_to_read = 0;
		bool const verbose = true;
		size_t const data_size = sizeof(real);

		status = matrix_load_binary_native( filename, starting_row, rows_to_read, verbose, data_size, (void *restrict *restrict) matrix,
							nrows, ncols, pitch, mt );
	}

	// Input file is "non-native" binary.
	else if ( is_bin ) {

		print_message( false, "\tFile selected as (non-\"native\") binary (i.e., double-precision data and unsigned integers).\n"
				"\tLoading...\n" );

		status = matrix_load_binary_verb( filename, matrix, nrows, ncols, pitch, mt );
	}

	// Input file is ASCII-text.
	else {

		print_message( false, "\tFile selected as ASCII text. Loading...\n"
			"\t\tData matrix selected as having numeric column headers: %s.\n"
			"\t\tData matrix selected as having numeric row labels: %s.\n",
			( numeric_hdrs ? "Yes" : "No" ), ( numeric_lbls ? "Yes" : "No" ) );

		status = matrix_load_ascii_verb( filename, numeric_hdrs, numeric_lbls, matrix, nrows, ncols, pitch, mt );

	} // If file is (native) binary or text.

	// -------------------------------

	return status;

} // matrix_load

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Saves a matrix to an ASCII-text file.
 * Skips name, headers and labels if 'mt' is NULL.
 *
 * If 'transpose' is 'true', transposes matrix in file as follows:
 * - Matrix dimensions in memory: <ncols> rows, <nrows> columns (padded to <pitch>)
 * - Matrix dimensions in file: <nrows> rows, <ncols> columns.
 * - Writes <ncols> mt->headers (as column headers) and <nrows> mt->labels (as row labels).
 *
 * ncols <= pitch, unless matrix transposing is set (in that case, nrows <= pitch).
 *
 * Set 'append' to 'true' to append data to the file.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int matrix_write_ascii( char const *restrict filename, bool real_data, void const *restrict matrix, index_t nrows, index_t ncols,
				index_t padding, bool transpose, bool append, struct matrix_tags_t const *restrict mt )
{

	int status = EXIT_SUCCESS;
	int const delimiter = (int) '\t';

	char const *restrict name = NULL;
	struct tag_t labels = new_empty_tag();
	struct tag_t headers = new_empty_tag();

	bool hasheaders = false;
	bool haslabels = false;

	////////////////////////////////

	// Checks for NULL parameters
	if ( ! ( (uintptr_t) filename * (uintptr_t) matrix ) ) {
		int const errnum = EFAULT;
		bool const shown_by_all = false;
		if ( ! filename ) print_errnum( shown_by_all, errnum, "\nmatrix_write_ascii( filename )" );
		if ( ! matrix )	print_errnum( shown_by_all, errnum, "\nmatrix_write_ascii( matrix )" );
		return EXIT_FAILURE;
	}

	// Checks matrix dimensions.
	{
		uintmax_t const nitems = ((uintmax_t) nrows) * ((uintmax_t) ncols);

		if ( (nrows <= 0) + (ncols <= 0) + (nitems > (uintmax_t) max_num_items) ) {
			print_error( false, "\nError in matrix_write_ascii( rows=%" PRI_IDX ", columns=%" PRI_IDX ", padding=%" PRI_IDX
						" transpose=%i ): Invalid matrix dimensions.\n", nrows, ncols, padding, transpose );
			if ( (nrows > 0) * (ncols > 0) ) // nitems > max_num_items
				print_error( false, "Matrix size (%" PRIuMAX " items) exceeds the limits used for matrix dimensions (%zu "
							"items).\n", nitems, max_num_items );
			return EXIT_FAILURE;
		}

		if ( transpose ) {
			if ( nrows > padding ) {
				print_error( false, "\nError in matrix_write_ascii( rows=%" PRI_IDX " [number of columns since matrix "
						"transposing is selected], padding=%" PRI_IDX " ): Invalid values.\n", nrows, padding );
				return EXIT_FAILURE;
			}
		} else if ( ncols > padding ) {
			print_error( false, "\nError in matrix_write_ascii( columns=%" PRI_IDX ", padding=%" PRI_IDX " ): Invalid values.\n",
					nrows, padding );
				return EXIT_FAILURE;
		}
	} // Checks matrix dimensions

	// -----------------------------

	// File mode: Creates a new text file, <OR>, appends data to an existing one.
	char const mode = ( append ? 'a' : 'w' );

	FILE *restrict const file = fopen( filename, &mode );
	if ( ! file ) {
		print_errnum( true, errno, "Error in matrix_write_ascii(): fopen( %s, mode='%c' )", filename, mode );
		return EXIT_FAILURE;
	}

	// -----------------------------

	// Writes tag elements.

	if ( mt ) {
		name = mt->name;
		headers = mt->headers;
		labels = mt->labels;

		hasheaders = (bool) headers.tokens;
		haslabels = (bool) labels.tokens;
	}

	// Name
	if ( name ) {
		struct tag_t const tag_name = new_tag( (char *restrict)name, (char **restrict)&name );	// Fakes a struct tag_t

		status = write_tag( file, tag_name, "name", 1, delimiter, false, ! hasheaders ); // No prefix; suffix only if no headers.
		if ( status != EXIT_SUCCESS ) {
			print_error( true, "Error in matrix_write_ascii()\n" );
			fclose(file);
			return EXIT_FAILURE;
		}
	}

	// Column headers
	if ( hasheaders ) {
		status = write_tag( file, headers, "headers", ncols, delimiter, name, true ); // Prefix only if has name; suffix always.
		if ( status != EXIT_SUCCESS ) {
			print_error( true, "Error in matrix_write_ascii()\n" );
			fclose(file);
			return EXIT_FAILURE;
		}
	}

	// ----------------------------

	// Writing data...

	// Step sizes for outer and inner loops.
	index_t incr_outer_loop = padding;	// Step size for outer loop
	index_t incr_inner_loop = 1;		// Step size for inner loop.
	if ( transpose ) {
		incr_outer_loop = 1;
		incr_inner_loop = padding;
	}
	errno = 0;

	size_t const data_size = ( real_data ? sizeof(real) : sizeof(index_t) );

	void const *pmatrix = matrix;	// &matrix[i][0] (or &matrix[0][i], if transpose)
	for ( index_t i = 0 ; i < nrows ; i++, pmatrix += (incr_outer_loop * data_size) ) {

		// Writes label.
		if ( haslabels ) {
			if ( fprintf( file, "%s", labels.ptokens[i] ) < 0 ) {	// < 0: *ptokens[i] might be NULL.
				print_errnum( true, errno, "Error in matrix_write_ascii( %s ): fprintf(plabels[%" PRI_IDX "])",
						( real_data ? "real" : "index_t" ), i );
				fclose(file);
				return EXIT_FAILURE;
			}
		} else {
			data_t val;
			int conv;

			memcpy( &val, pmatrix, data_size );

			if ( real_data )
				conv = fprintf( file, "%g", val.r );
			else
				conv = fprintf( file, "%" PRI_IDX, val.i );

			if ( conv <= 0 ) {
				print_errnum( true, errno, "Error in matrix_write_ascii( %s ): fprintf( %s %" PRI_IDX " of %" PRI_IDX
						", item 0 of %" PRI_IDX " )", ( real_data ? "real" : "index_t" ),
						( transpose ? "column" : "row" ), i, nrows, ncols );
				fclose(file);
				return EXIT_FAILURE;
			}
		} // if haslabels

		void const *p = pmatrix + ( (! haslabels) * incr_inner_loop * data_size );
		for ( index_t j = (! haslabels) ; j < ncols ; j++, p += (incr_inner_loop * data_size) ) {

			data_t val;
			int conv;

			memcpy( &val, p, data_size );

			if ( real_data )
				conv = fprintf( file, "%c%g", delimiter, val.r );
			else
				conv = fprintf( file, "%c%" PRI_IDX, delimiter, val.i );

			if ( conv <= 0 ) {
				print_errnum( true, errno, "Error in matrix_write_ascii( %s ): fprintf( %s %" PRI_IDX " of %" PRI_IDX ", item %"
						PRI_IDX " of %" PRI_IDX " )", ( real_data ? "real" : "index_t" ), ( transpose ? "column" : "row" ),
						i, nrows, j, ncols );
				fclose(file);
				return EXIT_FAILURE;
			}
		}

		if ( fprintf( file, "\n" ) <= 0 ) {
			print_errnum( true, errno, "Error in matrix_write_ascii( %s ): fprintf('\\n' at %s %" PRI_IDX " of %" PRI_IDX ")",
					( real_data ? "real" : "index_t" ), ( transpose ? "column" : "row" ), i, nrows );
			fclose(file);
			return EXIT_FAILURE;
		}

	} // for ( 0 <= i < nrows )

	if ( fclose(file) ) {
		print_errnum( true, errno, "Error in matrix_write_ascii(): fclose()" );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // matrix_write_ascii

// ---------------------------------------------

/*
 * Saves a real-type matrix to an ASCII-text file.
 * Skips name, headers and labels if 'mt' is NULL.
 *
 * If 'transpose' is 'true', transposes matrix in file as follows:
 * - Matrix dimensions in memory: <ncols> rows, <nrows> columns.
 * - Matrix dimensions in file: <nrows> rows, <ncols> columns.
 * - Writes <ncols> mt->headers (as column headers) and <nrows> mt->labels (as row labels).
 *
 * ncols <= padding, unless matrix transposing is set (in that case, nrows <= padding).
 *
 * Set 'append' to 'true' to append data to the file.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_save_ascii( char const *restrict filename, real const *restrict matrix, index_t nrows, index_t ncols, bool transpose, bool append,
			struct matrix_tags_t const *restrict mt, index_t padding )
{

	bool const real_data = true;

	return matrix_write_ascii( filename, real_data, (void const *restrict) matrix, nrows, ncols, transpose, append, mt, padding );

} // matrix_save_ascii

////////////////////////////////////////////////

/*
 * Saves an index_t-type matrix to an ASCII-text file.
 * Skips name, headers and labels if 'mt' is NULL.
 *
 * If 'transpose' is 'true', transposes matrix in file as follows:
 * - Matrix dimensions in memory: <ncols> rows, <nrows> columns.
 * - Matrix dimensions in file: <nrows> rows, <ncols> columns.
 * - Writes <ncols> mt->headers (as column headers) and <nrows> mt->labels (as row labels).
 *
 * ncols <= padding, unless matrix transposing is set (in that case, nrows <= padding).
 *
 * Set 'append' to 'true' to append data to the file.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_int_save_ascii( char const *restrict filename, index_t const *restrict matrix, index_t nrows, index_t ncols, bool transpose,
				bool append, struct matrix_tags_t const *restrict mt, index_t padding )
{

	bool const real_data = false;

	return matrix_write_ascii( filename, real_data, (void const *restrict) matrix, nrows, ncols, transpose, append, mt, padding );

} // matrix_int_save_ascii

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Writes to a file the given real-type values in a single line, separated by the given delimiter.
 *
 * prefix: If 'true', also writes a delimiter character before the first value.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int matrix_write_line( FILE *restrict file, real const *restrict pmatrix, index_t nitems, int delimiter, bool prefix )
{

	errno = 0;

	// Starts with no prefix
	if ( (! prefix) && (fprintf( file, "%g", *pmatrix ) <= 0) ) {
		print_errnum( true, errno, "Error in matrix_write_line(): fprintf( item 0 of %" PRI_IDX " )", nitems );
		return EXIT_FAILURE;
	}

	for ( index_t j = (! prefix) ; j < nitems ; j++ ) {
		int const conv = fprintf( file, "%c%g", delimiter, pmatrix[ j ] );
		if ( conv <= 0 ) {
			print_errnum( true, errno, "Error in matrix_write_line(): fprintf( item %" PRI_IDX " of %" PRI_IDX " )", j, nitems );
			return EXIT_FAILURE;
		}
	}

	return EXIT_SUCCESS;

} // matrix_write_line

// ---------------------------------------------

/*
 * Saves <nmatrices> <nrows>-by-<ncols> real-type matrices to a single ASCII-text file.
 * Reads input matrices from "native"-binary files (i.e., with the compiled types for matrix data and dimensions).
 * Uses the supplied labels (unless 'mt' is NULL).
 * nmatrices > 1
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_save_combined_ascii( char const *restrict filename, char const *restrict input_pattern, char const *restrict output_pattern,
				index_t nmatrices, index_t nrows, index_t ncols, struct matrix_tags_t const *restrict mt )
{

	// Temporary storage for filenames.
	char *restrict filename_tmp = NULL;
	size_t str_len = 0;	// strlen( filename_tmp )

	int status = EXIT_SUCCESS;
	int const delimiter = (int) '\t';

	char const *restrict name = NULL;
	struct tag_t labels = new_empty_tag();
	struct tag_t headers = new_empty_tag();

	bool hasheaders = false;
	bool haslabels = false;

	////////////////////////////////

	// Checks for NULL parameters
	if ( ! ( (uintptr_t) filename * (uintptr_t) input_pattern * (uintptr_t) output_pattern ) ) {
		int const errnum = EFAULT;
		bool const shown_by_all = false;
		if ( ! filename )	print_errnum( shown_by_all, errnum, "\nmatrix_save_combined_ascii( filename )" );
		if ( ! input_pattern )	print_errnum( shown_by_all, errnum, "\nmatrix_save_combined_ascii( input_pattern )" );
		if ( ! output_pattern )	print_errnum( shown_by_all, errnum, "\nmatrix_save_combined_ascii( output_pattern )" );
		return EXIT_FAILURE;
	}

	// Checks other parameters
	{
		uintmax_t const nitems = ((uintmax_t) nmatrices) * ((uintmax_t) nrows) * ((uintmax_t) ncols);

		if ( (nmatrices <= 1) + (nrows <= 0) + (ncols <= 0) + (nitems > (uintmax_t) max_num_items) ) {
			print_error( false, "\nError in matrix_save_combined_ascii( nmatrices=%" PRI_IDX ", rows=%" PRI_IDX
					", columns=%" PRI_IDX " ): Invalid parameters.\n", nmatrices, nrows, ncols );
			if ( (nrows > 0) * (ncols > 0) ) { // (nitems > max_num_items) || (nmatrices == 1)
				if ( nmatrices == 1 )
					print_error( false, "This method is intended for at least two matrices.\n" );
				else
					print_error( false, "The size of the resulting matrix ( %" PRIuMAX " items ) exceeds the limits "
							"used for matrix dimensions (%zu items).\n", nitems, max_num_items );
			}
			return EXIT_FAILURE;
		}
	}

	// --------------------------

	// List of input files

	FILE *restrict *restrict const input_files = (FILE *restrict *restrict) malloc( nmatrices * sizeof(FILE *) );
	if ( ! input_files ) {
		print_errnum( true, errno, "Error in matrix_save_combined_ascii(): malloc( input_files[], size=%" PRI_IDX " )", nmatrices );
		return EXIT_FAILURE;
	}

	// --------------------------

	// Temporary storage for filenames.
	{
		// Computes (very roughly) the required amount of memory

		size_t const len_ip = strlen( input_pattern );
		size_t const len_op = strlen( output_pattern );

		str_len = strlen( filename ) + MAX( len_ip, len_op );

		index_t val = MAX( ncols, nmatrices );
		index_t num_digits = 1;
		while ( val >= 10 ) {
			val /= 10;
			num_digits++;
		}
		str_len += (2 * num_digits);

		str_len++;	// To store the null character.

		// Allocates that memory
		filename_tmp = (char *restrict) malloc( str_len * sizeof(char) );
		if ( ! filename_tmp ) {
			print_errnum( true, errno, "Error in matrix_save_combined_ascii(): malloc( filename_tmp, size=%zu )", str_len );
			free((void *)input_files);
			return EXIT_FAILURE;
		}

	} // Temporary storage for filenames.

	// --------------------------

	// Opens all input files.

	for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) {

		errno = 0;
		status = snprintf( filename_tmp, str_len, input_pattern, filename, ncols, mt );
		if ( (status <= 0) + ((size_t)status >= str_len) ) {
			print_errnum( (status <= 0), errno, "Error matrix_save_combined_ascii(): snprintf( base_filename='%s', ncols=%"
					PRI_IDX ", input file mt=%" PRI_IDX " of nmatrices=%" PRI_IDX ", length=%zu )", filename, ncols,
					mt, nmatrices, str_len );
			if ( status > 0 )	// (status >= str_len)
				print_error( false, "The resulting string was truncated; %i bytes are required at least.\n", status + 1 );
			for ( index_t i = 0 ; i < mt ; i++ ) fclose(input_files[i]);
			free((void *)filename_tmp); free((void *)input_files);
			return EXIT_FAILURE;
		}

		input_files[mt] = (FILE *restrict) fopen( filename_tmp, "rb" );	// Opens for reading in binary mode.
		if( ! input_files[mt] ) {
			print_errnum( true, errno, "Error in matrix_save_combined_ascii(): fopen( input file '%s' )", filename_tmp );
			for ( index_t i = 0 ; i < mt ; i++ ) fclose(input_files[i]);
			free((void *)filename_tmp); free((void *)input_files);
			return EXIT_FAILURE;
		}

		// Checks matrix dimensions.
		index_t dim[2] = { 0, 0 };

		size_t const nread = fread( dim, sizeof(index_t), 2, input_files[mt] );
		if ( (nread != 2) + (dim[0] != nrows) + (dim[1] != ncols) ) {
			if ( ferror( input_files[mt] ) )
				print_errnum( true, errno, "Error in matrix_save_combined_ascii(): fread( dim[2], ncols=%" PRI_IDX
						", mt=%" PRI_IDX ", nmatrices=%" PRI_IDX " )" );
			else if ( feof( input_files[mt] ) )
				print_error( false, "\nError in matrix_save_combined_ascii() reading dimensions in file %" PRI_IDX " of "
						PRI_IDX ": Premature end of file detected.\nInvalid file format.\n", mt, nmatrices );
			else	// (dim[0] != nrows) || (dim[1] != ncols)
				print_error( false, "\nError in matrix_save_combined_ascii() reading dimensions in file %" PRI_IDX " of "
						PRI_IDX ": Invalid input matrix dimensions.\n%" PRI_IDX " x %" PRI_IDX " read, %" PRI_IDX
						" x %" PRI_IDX "expected.\n", mt, nmatrices, dim[0], dim[1], nrows, ncols );
			for ( index_t i = 0 ; i < mt ; i++ ) fclose(input_files[i]);
			fclose(input_files[mt]);
			free((void *)filename_tmp); free((void *)input_files);
			return EXIT_FAILURE;
		}
	} // For all input files.

	// --------------------------

	// Opens the output file.

	status = snprintf( filename_tmp, str_len, output_pattern, filename, ncols );
	if ( (status <= 0) + ((size_t)status >= str_len) ) {
		print_errnum( (status <= 0), errno, "Error matrix_save_combined_ascii(): snprintf( output file, ncols=%" PRI_IDX
				", length=%zu )", filename, ncols, str_len );
		if ( status > 0 )	// (status >= str_len)
			print_error( false, "The resulting string was truncated; %i bytes are required at least.\n", status + 1 );
		for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
		free((void *)filename_tmp); free((void *)input_files);
		return EXIT_FAILURE;
	}

	FILE *restrict const out_file = fopen( filename_tmp, "w" );
	if( ! out_file ) {
		print_errnum( true, errno, "Error in matrix_save_combined_ascii(): fopen( output file '%s' )", filename_tmp );
		for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
		free((void *)filename_tmp); free((void *)input_files);
		return EXIT_FAILURE;
	}
	free((void *)filename_tmp);

	// ----------------------------

	// Writes all matrix tag elements.

	if ( mt ) {
		name = mt->name;
		headers = mt->headers;
		labels = mt->labels;

		hasheaders = (bool) headers.tokens;
		haslabels = (bool) headers.tokens;
	}

	// Name
	if ( name ) {
		struct tag_t const tag_name = new_tag( (char *restrict)name, (char **restrict)&name );	// Fakes a struct tag_t

		status = write_tag( out_file, tag_name, "name", 1, delimiter, false, ! hasheaders ); // No prefix; suffix only if no headers.
		if ( status != EXIT_SUCCESS ) {
			print_error( true, "Error in matrix_save_combined_ascii().\n" );
			fclose(out_file);
			for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
			free((void *)input_files);
			return EXIT_FAILURE;
		}
	}

	// Column headers
	if ( hasheaders ) {

		// First header: prefix only if has name; no suffix.
		status = write_tag( out_file, headers, "headers", ncols, delimiter, name, false );
		if ( status != EXIT_SUCCESS ) {
			print_error( true, "Error in matrix_save_combined_ascii() writing column headers (0 of %" PRI_IDX " matrices)\n",
					nmatrices );
			fclose(out_file);
			for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
			free((void *)input_files);
			return EXIT_FAILURE;
		}

		// Rest of headers: All prefixed; no suffix
		for ( index_t mt = 1 ; mt < nmatrices ; mt++ ) {
			status = write_tag( out_file, headers, "headers", ncols, delimiter, true, false );
			if ( status != EXIT_SUCCESS ) {
				print_error( true, "Error in matrix_save_combined_ascii() writing column headers (%" PRI_IDX " of %"
						PRI_IDX " matrices)\n", mt, nmatrices );
				fclose(out_file);
				for ( index_t i = 0 ; i < nmatrices ; i++ ) fclose(input_files[i]);
				free((void *)input_files);
				return EXIT_FAILURE;
			}
		}

		errno = 0;
		if ( fprintf( out_file, "\n" ) <= 0 ) {
			print_errnum( true, errno, "Error in matrix_save_combined_ascii(): fprintf('\\n') after headers" );
			fclose(out_file);
			for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
			free((void *)input_files);
			return EXIT_FAILURE;
		}

	} // if hasheaders

	// ----------------------------

	// Allocates memory for a single data row.
	real *restrict const data = (real *restrict) malloc( ncols * sizeof(real) );
	if ( ! data ) {
		print_errnum( true, errno, "Error in matrix_save_combined_ascii(): malloc(data, size=ncols=%" PRI_IDX ")", ncols );
		fclose(out_file);
		for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
		free((void *)input_files);
		return EXIT_FAILURE;
	}

	// ----------------------------

	// Processes the rest of data.

	for ( index_t i = 0 ; i < nrows ; i++ ) {

		// Writes label.
		errno = 0;
		if ( haslabels && (fprintf(out_file,"%s",labels.ptokens[i]) < 0) ) {	// < 0: *ptokens[i] might be NULL
			print_errnum( true, errno, "Error in matrix_save_combined_ascii(): fprintf(plabels[%" PRI_IDX "])", i );
			free(data); fclose(out_file);
			for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
			free((void *)input_files);
			return EXIT_FAILURE;
		}

		// Writes row i from input file 0.
		{
			// Reads the entire row <i> from input file 0.
			size_t const nread = fread( data, sizeof(real), ncols, input_files[ 0 ] );
			if ( nread != (size_t) ncols ) {
				if ( ferror( input_files[ 0 ] ) )
					print_errnum( true, errno, "Error in matrix_save_combined_ascii(): fread( row=%" PRI_IDX
							", file 0 of %" PRI_IDX ", ncols=%" PRI_IDX " )", i, nmatrices, ncols );
				else
					print_error( false, "Error in matrix_save_combined_ascii() reading row %" PRI_IDX " from file 0 of %"
							PRI_IDX " (ncols=%" PRI_IDX ").\nPremature end of file.\n", i, nmatrices, ncols );
				free(data); fclose(out_file);
				for ( index_t j = 0 ; j < nmatrices ; j++ ) fclose(input_files[j]);
				free((void *)input_files);
				return EXIT_FAILURE;
			}

			// Writes that row: prefixes only if there are labels.
			status = matrix_write_line( out_file, data, ncols, delimiter, haslabels );
			if ( status != EXIT_SUCCESS ) {
				print_error( true, "Error in matrix_save_combined_ascii() writing row %" PRI_IDX " from file 0 of %"
						PRI_IDX " (ncols=%" PRI_IDX ").\n", i, nmatrices, ncols );
				free(data); fclose(out_file);
				for ( index_t j = 0 ; j < nmatrices ; j++ ) fclose(input_files[j]);
				free((void *)input_files);
				return EXIT_FAILURE;
			}
		} // input file 0

		// Writes row i from the rest of input files
		for ( index_t mt = 1 ; mt < nmatrices ; mt++ ) {

			// Reads the entire row <i> from input file <mt>.
			size_t const nread = fread( data, sizeof(real), ncols, input_files[ mt ] );
			if ( nread != (size_t) ncols ) {
				if ( ferror( input_files[ mt ] ) )
					print_errnum( true, errno, "Error in matrix_save_combined_ascii(): fread( row=%" PRI_IDX ", file %"
							PRI_IDX " of %" PRI_IDX ", ncols=%" PRI_IDX " )", i, mt, nmatrices, ncols );
				else
					print_error( false, "Error in matrix_save_combined_ascii() reading row %" PRI_IDX " from file %"
							PRI_IDX " of %" PRI_IDX " (ncols=%" PRI_IDX ").\nPremature end of file.\n", i, mt,
							nmatrices, ncols );
				free(data); fclose(out_file);
				for ( index_t j = 0 ; j < nmatrices ; j++ ) fclose(input_files[j]);
				free((void *)input_files);
				return EXIT_FAILURE;
			}

			// Writes that row: prefixes always.
			status = matrix_write_line( out_file, data, ncols, delimiter, true );
			if ( status != EXIT_SUCCESS ) {
				print_error( true, "Error in matrix_save_combined_ascii() writing row %" PRI_IDX " from file %"
						PRI_IDX " of %" PRI_IDX " (ncols=%" PRI_IDX ").\n", i, mt, nmatrices, ncols );
				free(data); fclose(out_file);
				for ( index_t j = 0 ; j < nmatrices ; j++ ) fclose(input_files[j]);
				free((void *)input_files);
				return EXIT_FAILURE;
			}

		} // for each input file.

		errno = 0;
		if ( fprintf( out_file, "\n" ) <= 0 ) {
			print_errnum( true, errno, "Error in matrix_save_combined_ascii(): fprintf('\\n') after row %" PRI_IDX, i );
			free(data); fclose(out_file);
			for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
			free((void *)input_files);
			return EXIT_FAILURE;
		}

	} // for each row.

	// --------------------------

	// Closes all files and cleans up.

	free(data);

	for ( index_t mt = 0 ; mt < nmatrices ; mt++ )
		fclose( input_files[mt] );
	free( (void *) input_files );

	errno = 0;
	if ( fclose(out_file) ) {
		print_errnum( true, errno, "Error in matrix_save_combined_ascii(): fclose( output file )" );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // matrix_save_combined_ascii

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Writes labels, headers and name (as plain text).
 *
 * NOTE: This function is intended to be used when writing tag elements to a BINARY file.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int matrix_write_tags( FILE *restrict file, index_t num_headers, index_t num_labels, struct matrix_tags_t mt, int delimiter )
{

	char const *restrict name = mt.name;
	struct tag_t headers = mt.headers;
	struct tag_t labels = mt.labels;

	// -----------------------------

	// Row labels
	{
		bool const suffix = ( (uintptr_t) headers.tokens + (uintptr_t)name );
		int const status = write_tag( file, labels, "labels", num_labels, delimiter, false, suffix ); // No prefix.
		if ( status != EXIT_SUCCESS ) {
			print_error( true, "Error in matrix_write_tags()\n" );
			return EXIT_FAILURE;
		}
	}

	// -----------------------------

	// Column headers
	{
		bool const suffix = name;	// name != NULL
		int const status = write_tag( file, headers, "headers", num_headers, delimiter, false, suffix ); // No prefix
		if ( status != EXIT_SUCCESS ) {
			print_error( true, "Error in matrix_write_tags()\n" );
			return EXIT_FAILURE;
		}
	}

	// -----------------------------

	// Name
	{
		struct tag_t const tag_name = new_tag( (char *restrict)name, (char **restrict)&name );	// Fakes a struct tag_t
		bool const suffix = name;	// name != NULL

		int const status = write_tag( file, tag_name, "name", 1, delimiter, false, suffix ); // No prefix
		if ( status != EXIT_SUCCESS ) {
			print_error( true, "Error in matrix_write_tags()\n" );
			return EXIT_FAILURE;
		}
	}

	return EXIT_SUCCESS;

} // matrix_write_tags

// ---------------------------------------------

/*
 * Saves a matrix to a (non-"native") binary file (i.e., double-precision data and unsigned integers).
 * Skips name, headers and labels if 'mt' is NULL.
 *
 * If 'transpose' is 'true', transposes matrix in file as follows:
 * - Matrix dimensions in memory: <ncols> rows, <nrows> columns.
 * - Matrix dimensions in file: <nrows> rows, <ncols> columns.
 * - Writes <ncols> mt->headers (as column headers) and <nrows> mt->labels (as row labels).
 *
 * ncols <= padding, unless matrix transposing is set (in that case, nrows <= padding).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_save_binary( char const *restrict filename, real const *restrict matrix, index_t nrows, index_t ncols, bool transpose,
			struct matrix_tags_t const *restrict mt, index_t padding )
{

	int const delimiter = (int) '\t';

	////////////////////////////////

	// Checks for NULL parameters
	if ( ! ( (uintptr_t) filename * (uintptr_t) matrix ) ) {
		int const errnum = EFAULT;
		bool const shown_by_all = false;
		if ( ! filename ) print_errnum( shown_by_all, errnum, "\nmatrix_save_binary( filename )" );
		if ( ! matrix )	print_errnum( shown_by_all, errnum, "\nmatrix_save_binary( matrix )" );
		return EXIT_FAILURE;
	}

	// Checks matrix dimensions.
	{
		uintmax_t const nitems = ((uintmax_t) nrows) * ((uintmax_t) ncols);

		if ( (nrows <= 0) + (ncols <= 0) + (nitems > (uintmax_t) max_num_items) ) {
			print_error( false, "\nError in matrix_save_binary( rows=%" PRI_IDX ", columns=%" PRI_IDX ", padding=%" PRI_IDX
						" transpose=%i ): Invalid matrix dimensions.\n", nrows, ncols, padding, transpose );
			if ( (nrows > 0) * (ncols > 0) ) // nitems > max_num_items
				print_error( false, "Matrix size (%" PRIuMAX " items) exceeds the limits used for matrix dimensions (%zu "
							"items).\n", nitems, max_num_items );
			return EXIT_FAILURE;
		}

		if ( transpose ) {
			if ( nrows > padding ) {
				print_error( false, "\nError in matrix_save_binary( rows=%" PRI_IDX " [number of columns since matrix "
						"transposing is selected], padding=%" PRI_IDX " ): Invalid values.\n", nrows, padding );
				return EXIT_FAILURE;
			}
		} else if ( ncols > padding ) {
			print_error( false, "\nError in matrix_save_binary( columns=%" PRI_IDX ", padding=%" PRI_IDX " ): Invalid values.\n",
					nrows, padding );
				return EXIT_FAILURE;
		}
	} // Checks matrix dimensions

	// -----------------------------

	FILE *restrict const file = fopen( filename, "wb" );
	if ( ! file ) {
		print_errnum( true, errno, "Error in matrix_save_binary(): fopen( %s )", filename );
		return EXIT_FAILURE;
	}

	// ----------------------------------

	// Writes matrix dimensions.
	{
		unsigned int const dims[2] = { (unsigned int) nrows, (unsigned int) ncols };
		size_t const nwritten = fwrite( dims, sizeof(unsigned int), 2, file );
		if ( nwritten != 2 ) {
			print_error( true, "\nError in matrix_save_binary(): fwrite( dim, size=2 ).\n" );
			fclose(file);
			return EXIT_FAILURE;
		}
	}

	// ----------------------------------

	// Writes data matrix

	// Steps for outer and inner loops.
	index_t incr_outer_loop = padding;	// Step for outer loop
	index_t incr_inner_loop = 1;		// Step for inner loop.
	if ( transpose ) {
		incr_outer_loop = 1;
		incr_inner_loop = padding;
	}

	real const *pmatrix = matrix;
	for ( index_t r=0 ; r < nrows ; r++, pmatrix += incr_outer_loop ) {

		real const *pmatrix_r = pmatrix;

		for ( index_t c=0 ; c < ncols ; c++, pmatrix_r += incr_inner_loop ) {

			// Writes one double-precision data
			double const value = (double) *pmatrix_r;

			size_t const nwritten = fwrite( &value, sizeof(double), 1, file );
			if ( ! nwritten ) {
				print_errnum( true, errno, "Error in matrix_save_binary(): fwrite( row %" PRI_IDX ", column %" PRI_IDX
							", transposed dimensions: %s )", r, c, (transpose ? "yes" : "no") );
				fclose(file);
				return EXIT_FAILURE;
			}

		} // for c

	} // for r

	// ----------------------------------

	// Writes matrix labels, if any

	if ( mt && (matrix_write_tags( file, ncols, nrows, *mt, delimiter ) != EXIT_SUCCESS) ) {
		fclose(file);
		return EXIT_FAILURE;
	}

	if ( fclose(file) ) {
		print_errnum( true, errno, "Error in matrix_save_binary(): fclose( %s )", filename );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // matrix_save_binary

////////////////////////////////////////////////

/*
 * Saves a matrix to a "native" binary file (i.e., with the compiled types for matrix data and dimensions).
 * Skips name, headers and labels if 'mt' is NULL.
 *
 * WARNING:
 *	- For internal use only (i.e., for temporary files).
 *	- NO ERROR-CHECKING IS PERFORMED (e.g., overflow, invalid values...).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_save_binary_native( char const *restrict filename, void const *restrict matrix, index_t nrows, index_t ncols, size_t data_size,
				struct matrix_tags_t const *restrict mt )
{

	int const delimiter = (int) '\t';

	////////////////////////////////

	// Checks for NULL parameters
	if ( ! ( (uintptr_t) filename * (uintptr_t) matrix * data_size ) ) {
		int const errnum = EFAULT;
		bool const shown_by_all = false;
		if ( ! filename ) print_errnum( shown_by_all, errnum, "\nmatrix_save_binary_native( filename )" );
		if ( ! matrix )	print_errnum( shown_by_all, errnum, "\nmatrix_save_binary_native( matrix )" );
		if ( ! data_size ) print_errnum( shown_by_all, EINVAL, "\nmatrix_save_binary_native( data_size )" );
		return EXIT_FAILURE;
	}

	// Checks matrix dimensions.
	{
		uintmax_t const nitems = ((uintmax_t) nrows) * ((uintmax_t) ncols);

		if ( (nrows <= 0) + (ncols <= 0) + (nitems > (uintmax_t) max_num_items) ) {
			print_error( false, "\nError in matrix_save_binary_native( rows=%" PRI_IDX ", columns=%" PRI_IDX
					" ): Invalid matrix dimensions.\n", nrows, ncols );
			if ( (nrows > 0) * (ncols > 0) ) // nitems > max_num_items
				print_error( false, "Matrix size (%" PRIuMAX " items) exceeds the limits used for matrix dimensions (%zu "
							"items).\n", nitems, max_num_items );
			return EXIT_FAILURE;
		}
	}

	size_t const nitems = nrows * ncols;

	// -----------------------------

	FILE *restrict const file = fopen( filename, "wb" );
	if ( ! file ) {
		print_errnum( true, errno, "Error in matrix_save_binary_native(): fopen( %s )", filename );
		return EXIT_FAILURE;
	}

	// ----------------------------------

	// Writes matrix dimensions.
	{
		index_t const dims[2] = { nrows, ncols };
		size_t const nwritten = fwrite( dims, sizeof(index_t), 2, file );
		if ( nwritten != 2 ) {
			print_error( true, "\nError in matrix_save_binary_native(): fwrite( dim, size=2 ).\n" );
			fclose(file);
			return EXIT_FAILURE;
		}
	}

	// ----------------------------------

	// Writes data matrix

	size_t const nwritten = fwrite( matrix, data_size, nitems, file );
	if ( nwritten != (size_t) nitems ) {
		print_errnum( true, errno, "Error in matrix_save_binary_native(): fwrite( %" PRI_IDX "x%" PRI_IDX " = %zu )",
				nrows, ncols, nitems );
		fclose(file);
		return EXIT_FAILURE;
	}

	// ----------------------------------

	// Writes matrix labels, if any

	if ( mt && (matrix_write_tags( file, ncols, nrows, *mt, delimiter ) != EXIT_SUCCESS) ) {
		fclose(file);
		return EXIT_FAILURE;
	}

	if ( fclose(file) ) {
		print_errnum( true, errno, "Error in matrix_save_binary_native(): fclose( %s )", filename );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // matrix_save_binary_native

////////////////////////////////////////////////

/*
 * Writes matrix to a file according to the selected file format.
 * Skips name, headers and labels if 'mt' is NULL.
 *
 * save_bin: Saves output matrix to a binary file.
 *		== 0: Disabled. Saves the file as ASCII text.
 *		== 1: Uses "non-native" format (i.e., double-precision data, and "unsigned int" for dimensions).
 *		 > 1: Uses "native" or raw format (i.e., the compiled types for matrix data and dimensions).
 *
 * If 'transpose' is 'true' and save_bin <= 1, transposes matrix in file as follows:
 * - Matrix dimensions in memory: <ncols> rows, <nrows> columns.
 * - Matrix dimensions in file: <nrows> rows, <ncols> columns.
 * - Writes <ncols> mt->headers (as column headers) and <nrows> mt->labels (as row labels).
 *
 * ncols <= padding, unless matrix transposing is set (in that case, nrows <= padding).
 *
 * If verbose is 'true', shows some information messages (e.g., file format).
 *
 * WARNING:
 *	"Native" mode (i.e., save_bin > 1) skips ALL data transformation (matrix transposing, padding, etc).
 *	All related arguments are ignored, and the file is saved in raw format.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_save( char const *restrict filename, index_t save_bin, real const *restrict matrix, index_t nrows, index_t ncols, bool transpose,
		struct matrix_tags_t const *restrict mt, index_t padding, bool verbose )
{

	// Checks for NULL parameters
	if ( ! ( (uintptr_t) filename * (uintptr_t) matrix ) ) {
		int const errnum = EFAULT;
		bool const shown_by_all = false;
		if ( ! filename ) print_errnum( shown_by_all, errnum, "\nmatrix_save( filename )" );
		if ( ! matrix )	print_errnum( shown_by_all, errnum, "\nmatrix_save( matrix )" );
		return EXIT_FAILURE;
	}

	if ( (nrows <= 0) + (ncols <= 0) + (save_bin < 0) ) {
		if ( (nrows <= 0) + (ncols <= 0) )
			print_error( false, "\nError in matrix_save( rows=%" PRI_IDX ", columns=%" PRI_IDX " ): Invalid matrix dimensions.\n",
					nrows, ncols );
		if ( save_bin < 0 )
			print_errnum( false, EINVAL, "\nError in matrix_save( save_bin=%" PRI_IDX " )", save_bin );
		return EXIT_FAILURE;
	}

	int status = EXIT_SUCCESS;

	// -------------------------------

	if ( verbose )
		print_message( false, "\nSaving output file...\n" );

	// Saves output as "native" binary.
	if ( save_bin > 1 ) {
		if ( verbose ) {
			print_message( false, "\tFile selected as \"native\" binary (i.e., the file is written using the data types specified "
					"at compilation).\n\tNo error-checking is performed.\n" );
			if ( transpose )
				print_message( false, "\tSkipping all transformation options (matrix transposing, padding, etc.)...\n");
		}
		status = matrix_save_binary_native( filename, matrix, nrows, ncols, sizeof(real), mt );
	}

	// Saves output as (non-"native") binary.
	else if ( save_bin ) {
		if ( verbose )
			print_message(false,"\tFile selected as (non-\"native\") binary (i.e., double-precision data and unsigned integers).\n");

		status = matrix_save_binary( filename, matrix, nrows, ncols, transpose, mt, padding );
	}

	// Saves output as ASCII text.
	else {
		if ( verbose )
			print_message( false, "\tFile selected as ASCII text.\n" );
		status = matrix_save_ascii( filename, matrix, nrows, ncols, transpose, false, mt, padding );
	}

	// ----------------------------------------

	return status;

} // matrix_save

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Prints matrix's content (data, name, headers and/or labels).
 * Skips name, headers and labels if 'mt' is NULL.
 *
 * If 'transpose' is 'true', transposes matrix as follows:
 * - Matrix dimensions in memory: <numcols> rows, <numrows> columns.
 * - Matrix dimensions on screen: <numrows> rows, <numcols> columns.
 * - Shows <numcols> mt->headers (as column headers) and <numrows> mt->labels (as row labels).
 *
 * numcols <= padding, unless matrix transposing is set (in that case, numrows <= padding).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int matrix_print( bool real_data, void const *restrict matrix, index_t numrows, index_t numcols, index_t padding, bool transpose,
			bool shown_by_all, struct matrix_tags_t const *restrict mt )
{

	int status = EXIT_SUCCESS;

	char const *restrict name = NULL;
	struct tag_t labels = new_empty_tag();
	struct tag_t headers = new_empty_tag();

	bool hasheaders = false;
	bool haslabels = false;

	// Portion of "matrix" to be shown.
	index_t nrows = numrows;
	index_t ncols = numcols;

	// Maximum length of each row label to show.
	int const label_precision = 5;

	////////////////////////////////

	// Checks for NULL parameters
	if ( ! matrix ) {
		print_errnum( false, EFAULT, "\nprint_show( matrix )" );
		return EXIT_FAILURE;
	}

	// Checks matrix dimensions.
	{
		uintmax_t const nitems = ((uintmax_t) nrows) * ((uintmax_t) ncols);

		if ( (nrows <= 0) + (ncols <= 0) + (nitems > (uintmax_t) max_num_items) ) {
			print_error( false, "\nError in print_show( rows=%" PRI_IDX ", columns=%" PRI_IDX ", padding=%" PRI_IDX
						" transpose=%i ): Invalid matrix dimensions.\n", nrows, ncols, padding, transpose );
			if ( (nrows > 0) * (ncols > 0) ) // nitems > max_num_items
				print_error( false, "Matrix size (%" PRIuMAX " items) exceeds the limits used for matrix dimensions (%zu "
							"items).\n", nitems, max_num_items );
			return EXIT_FAILURE;
		}

		if ( transpose ) {
			if ( nrows > padding ) {
				print_error( false, "\nError in print_show( rows=%" PRI_IDX " [number of columns since matrix "
						"transposing is selected], padding=%" PRI_IDX " ): Invalid values.\n", nrows, padding );
				return EXIT_FAILURE;
			}
		} else if ( ncols > padding ) {
			print_error( false, "\nError in print_show( columns=%" PRI_IDX ", padding=%" PRI_IDX " ): Invalid values.\n",
					nrows, padding );
				return EXIT_FAILURE;
		}
	} // Checks matrix dimensions

	// --------------------------------

	// Checks if there are tag elements to show...

	if ( mt ) {
		name = mt->name;
		headers = mt->headers;
		labels = mt->labels;

		hasheaders = (bool) headers.tokens;
		haslabels = (bool) labels.tokens;
	}

	// -----------------------------

	// Matrix dimensions to show.

	#if ! NMFGPU_TESTING
		if ( nrows == 1 )	// 'matrix' is a vector.
			ncols = MIN( ncols, 225 ) ;
		else {
			ncols = MIN( ncols, 15 ) ;
			nrows = MIN( nrows, 9 ) ;
		}

		// Decrements ncols by 1 if there are row labels.
		ncols -= haslabels;
	#endif

	// -----------------------------

	// Writes all tags.

	// Name
	if ( name ) {
		struct tag_t const tag_name = new_tag( (char *restrict)name, (char **restrict)&name );	// Fakes a struct tag_t

		if ( show_tag( tag_name, "Name", 1, 1, true, shown_by_all ) != EXIT_SUCCESS )
			return EXIT_FAILURE;
	}

	// Column headers
	if ( hasheaders && ( show_tag( headers, "Headers", numcols, ncols, true, shown_by_all ) != EXIT_SUCCESS ) )
		return EXIT_FAILURE;

	// ----------------------------

	// Prints matrix, with row labels if exist

	if ( ((uintptr_t) name + hasheaders + haslabels + transpose) &&
		(print_message( shown_by_all, "Data matrix: %" PRI_IDX " rows x %" PRI_IDX " columns.\n", numrows, numcols ) != EXIT_SUCCESS) )
		return EXIT_FAILURE;


	// Warns about possibly truncated row labels.
	if ( haslabels ) {
		status = print_message( shown_by_all, "Please note that row labels will show up to the first %i characters.\n\n",
					label_precision );
		if ( status != EXIT_SUCCESS )
			return EXIT_FAILURE;
	}


	// Step sizes for outer and inner loops.
	index_t incr_outer_loop = padding;	// Step size for outer loop
	index_t incr_inner_loop = 1;		// Step size for inner loop.
	if ( transpose ) {
		incr_outer_loop = 1;
		incr_inner_loop = padding;
	}
	errno = 0;

	size_t const data_size = ( real_data ? sizeof(real) : sizeof(index_t) );

	void const *pmatrix = matrix;	// &matrix[i][0] (or &matrix[0][i], if transpose)
	for ( index_t i = 0 ; i < nrows ; i++, pmatrix += (incr_outer_loop * data_size) ) {

		if ( print_message( shown_by_all, "Line %" PRI_IDX ":", i ) != EXIT_SUCCESS )
			return EXIT_FAILURE;

		// Prints a (truncated) row label.
		if ( haslabels && (print_message( shown_by_all, " %.*s:", label_precision, labels.ptokens[i] ) != EXIT_SUCCESS) )
			return EXIT_FAILURE;

		void const *p = pmatrix;
		for ( index_t j = 0 ; j < ncols ; j++, p += (incr_inner_loop * data_size) ) {

			data_t val;
			memcpy( &val, p, data_size );

			if ( real_data )
				status = print_message( shown_by_all, " %g", val.r );
			else
				status = print_message( shown_by_all, " %" PRI_IDX, val.i );

			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;

		} // for

		// Last column.
		if ( ncols < numcols ) {

			data_t val;
			p = pmatrix + ( (numcols-1) * incr_inner_loop * data_size );

			memcpy( &val, p, data_size );

			if ( real_data )
				status = print_message( shown_by_all, " ... %g", val.r );
			else
				status = print_message( shown_by_all, " ... %" PRI_IDX, val.i );

			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		}

		if ( print_message( shown_by_all, "\n" ) != EXIT_SUCCESS )
			return EXIT_FAILURE;

	} // for i=[0..nrows)

	// Last row.
	if ( nrows < numrows ) {

		index_t const i = numrows - 1;
		pmatrix = matrix + (i * incr_outer_loop * data_size);

		if ( print_message( shown_by_all, "...\nLine %" PRI_IDX ":", i ) != EXIT_SUCCESS )
			return EXIT_FAILURE;

		// Prints a (truncated) row label.
		if ( haslabels && (print_message( shown_by_all, " %.*s:", label_precision, labels.ptokens[i] ) != EXIT_SUCCESS) )
			return EXIT_FAILURE;

		void const *p = pmatrix;
		for ( index_t j = 0 ; j < ncols ; j++, p += (incr_inner_loop * data_size) ) {

			data_t val;
			memcpy( &val, p, data_size );

			if ( real_data )
				status = print_message( shown_by_all, " %g", val.r );
			else
				status = print_message( shown_by_all, " %" PRI_IDX, val.i );

			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;

		} // for

		// Last column.
		if ( ncols < numcols ) {

			data_t val;
			p = pmatrix + ( (numcols-1) * incr_inner_loop * data_size );

			memcpy( &val, p, data_size );

			if ( real_data )
				status = print_message( shown_by_all, " ... %g", val.r );
			else
				status = print_message( shown_by_all, " ... %" PRI_IDX, val.i );

			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		}

		if ( print_message( shown_by_all, "\n" ) != EXIT_SUCCESS )
			return EXIT_FAILURE;

	} // if ( nrows < numrows )

	return print_message( shown_by_all, "\n" );

} // matrix_print

// ---------------------------------------------

/*
 * Shows matrix's content (data, name, headers and/or labels).
 * Skips name, headers and labels if 'mt' is NULL.
 *
 * If 'transpose' is 'true', transposes matrix as follows:
 * - Matrix dimensions in memory: <numcols> rows, <numrows> columns.
 * - Matrix dimensions on screen: <numrows> rows, <numcols> columns.
 * - Shows <numcols> mt->headers (as column headers) and <numrows> mt->labels (as row labels).
 *
 * numcols <= padding, unless matrix transposing is set (in that case, numrows <= padding).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int matrix_show( real const *restrict matrix, index_t numrows, index_t numcols, index_t padding, bool transpose, bool shown_by_all,
		struct matrix_tags_t const *restrict mt )
{

	bool const real_data = true;

	return matrix_print( real_data, matrix, numrows, numcols, padding, transpose, shown_by_all, mt );

} // matrix_show

////////////////////////////////////////////////

/*
 * Shows matrix's content (data, name, headers and/or labels).
 * Skips name, headers and labels if 'mt' is NULL.
 *
 * If 'transpose' is 'true', transposes matrix as follows:
 * - Matrix dimensions in memory: <numcols> rows, <numrows> columns.
 * - Matrix dimensions on screen: <numrows> rows, <numcols> columns.
 * - Shows <numcols> mt->headers (as column headers) and <numrows> mt->labels (as row labels).
 *
 * numcols <= padding, unless matrix transposing is set (in that case, numrows <= padding).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int matrix_int_show( index_t const *restrict matrix, index_t numrows, index_t numcols, index_t padding, bool transpose, bool shown_by_all,
			struct matrix_tags_t const *restrict mt )
{

	bool const real_data = false;

	return matrix_print( real_data, matrix, numrows, numcols, padding, transpose, shown_by_all, mt );

} // matrix_int_show

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Transposes a matrix using a temporary file.
 *
 * <base_filename> is used only if no temporary file from the system can be employed. It is never referenced otherwise.
 *
 * WARNING:
 *	- Pointer "matrix" is ALWAYS CHANGED, even on error.
 *	- NO ERROR-CHECKING IS PERFORMED (e.g., overflow, invalid values...).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int matrix_transpose_file( void *restrict matrix, index_t *restrict nrows, index_t *restrict ncols, size_t data_size,
				char const *restrict base_filename )
{

	index_t numcols = 0, numrows = 0;

	bool custom_file = false;		// Uses a custom file rather than one generated by the system.
	char *restrict filename_tmp = NULL;	// Custom filename (used only if no temporary file from the system can be employed).

	////////////////////////////////

	// Checks for NULL parameters
	if ( ! ( (uintptr_t) matrix * (uintptr_t) nrows * (uintptr_t) ncols * (uintptr_t) data_size ) ) {
		bool const shown_by_all = false;
		int const errnum = EFAULT;
		if ( ! matrix )	print_errnum( shown_by_all, errnum, "\nmatrix_transpose_file( matrix )" );
		if ( ! nrows )	print_errnum( shown_by_all, errnum, "\nmatrix_transpose_file( nrows )" );
		if ( ! ncols )	print_errnum( shown_by_all, errnum, "\nmatrix_transpose_file( ncols )" );
		if ( ! data_size ) print_errnum( shown_by_all, EINVAL, "\nmatrix_transpose_file( data_size )" );
		return EXIT_FAILURE;
	}

	numrows = *nrows;
	numcols = *ncols;

	if ( (numrows <= 0) + (numcols <= 0) ) {
		print_error( false, "\nError in matrix_save_binary_native( rows=%" PRI_IDX ", columns=%" PRI_IDX
				" ): Invalid matrix dimensions.\n", numrows, numcols );
		return EXIT_FAILURE;
	}

	// ---------------------------

	// Opens a temporary file

	FILE *restrict file = tmpfile();

	if ( ! file ) {

		// Uses a custom file.
		custom_file = true;

		if ( ! base_filename ) {
			print_errnum( false, EFAULT, "\nmatrix_transpose_file( base_filename )" );
			return EXIT_FAILURE;
		}

		size_t const len = strlen(base_filename) + 8;
		filename_tmp = (char *restrict) malloc( len * sizeof(char) );
		if ( ! filename_tmp ) {
			print_errnum( true, errno, "Error in matrix_transpose_file(): malloc( filename_tmp, length=%zu )", len );
			return EXIT_FAILURE;
		}
		errno = 0;
		int const conv = snprintf( filename_tmp, len, "%s_t.dat", base_filename );
		if ( (conv <= 0) + ((size_t) conv >= len) ) {
			print_errnum( (conv <= 0), errno, "Error in matrix_transpose_file(): snprintf( filename_tmp, length=%zu )", len );
			if ( conv > 0 )	// conv >= len
				print_error( false, "The resulting string was truncated; %i bytes are required at least.\n", conv + 1 );
			free(filename_tmp);
			return EXIT_FAILURE;
		}

		file = fopen( filename_tmp, "w+b" );	// Open for reading and writing.
		if ( ! file ) {
			print_errnum( true, errno, "Error in matrix_transpose_file(): fopen( %s )", filename_tmp );
			free(filename_tmp);
			return EXIT_FAILURE;
		}

	} // If requires a custom file.

	// ---------------------------

	// Writes matrix by columns

	void const *pmatrix = matrix;
	for ( index_t j = 0 ; j < numcols ; j++, pmatrix += data_size ) {

		void const *pmatrix_r = pmatrix;

		for ( index_t i=0 ; i<numrows ; i++, pmatrix_r += (numcols * data_size) )

			if ( fwrite( pmatrix_r, data_size, 1, file ) != 1 ) {
				print_errnum( true, errno, "Error in matrix_transpose_file(): fwrite( row %" PRI_IDX ", column %"
						PRI_IDX " )", i, j );
				fclose(file);
				if ( custom_file ) { unlink(filename_tmp); free(filename_tmp); }
				return EXIT_FAILURE;
			}
	} // for j

	// ---------------------------

	// Now, reads the file.

	rewind(file);

	size_t const nitems = numrows * numcols;
	size_t const nread = fread( matrix, data_size, nitems, file );
	if ( nread != nitems ) {
		if ( ferror(file) )
			print_errnum( true, errno, "Error in matrix_transpose_file(): fread( %zu items )", nitems );
		else	// EOF
			print_error( true, "Error in matrix_transpose_file(): fread( %zu items ): Premature end-of-file detected "
					"(%zu items read).\n", nitems, nread );
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

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Cleans name, headers, labels and matrix.
 */
void matrix_clean( void *restrict matrix, struct matrix_tags_t mt )
{

	clean_matrix_tags( mt );

	if ( matrix )
		free( (void *) matrix );

} // matrix_clean

////////////////////////////////////////////////
////////////////////////////////////////////////
