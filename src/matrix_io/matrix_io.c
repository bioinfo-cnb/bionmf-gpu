/************************************************************************
 *
 * NMF-mGPU - Non-negative Matrix Factorization on multi-GPU systems.
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
 **********************************************************/

#include "matrix_io/matrix_io.h"
#include "matrix_io/matrix_io_routines.h"
#include "common.h"
#include "index_type.h"
#include "real_type.h"

#include <math.h>	/* isless */
#include <ctype.h>	/* isprint */
#include <unistd.h>	/* unlink */
#include <stdio.h>
#include <inttypes.h>	/* PRIxxx, xxx_C, uintxxx_t */
#include <string.h>
#include <errno.h>
#include <stdbool.h>
#include <stdlib.h>

////////////////////////////////////////////////
////////////////////////////////////////////////

/* Constants */

/* Signatures for binary files.
 *
 * WARNING: Length must NOT exceed 32 bits (i.e., 8 hex characters).
 */
#undef BIN_FILE_SIGNATURE_REAL
#define BIN_FILE_SIGNATURE_REAL ( 0xEDB10EA1 )

#undef BIN_FILE_SIGNATURE_INDEX
#define BIN_FILE_SIGNATURE_INDEX ( 0xEDB101DE )

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

/* "Private" global variables */

// Simple variable and macro to test Endiannes.
static int const endiannes_test = 1;
#undef IS_BIG_ENDIAN
#define IS_BIG_ENDIAN() ( ! *((unsigned char const *restrict) &endiannes_test) )


// Information and/or error messages shown by all processes.
#if NMFGPU_DEBUG_READ_MATRIX || NMFGPU_DEBUG_READ_MATRIX2 || NMFGPU_DEBUG_WRITE_MATRIX
	static bool const dbg_shown_by_all = false;	// Information or error messages on debug.
#endif
static bool const shown_by_all = false;			// Information messages in non-debug mode.
static bool const sys_error_shown_by_all = true;	// System error messages.
static bool const error_shown_by_all = false;		// Error messages on invalid arguments or I/O data.

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Checks matrix dimensions.
 *
 * In verbose mode, shows matrix dimensions.
 *
 * ncols <= pitch.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_check_dimensions( char const *restrict const function_name, index_t nrows, index_t ncols, index_t pitch, bool transpose,
				bool verbose )
{

	// Limits on matrix dimensions
	size_t const max_num_items = matrix_max_num_items;
	index_t const max_pitch = matrix_max_pitch;
	index_t const max_non_padded_dim = matrix_max_non_padded_dim;

	int status = EXIT_SUCCESS;

	// -----------------------------

	/* In "normal" mode:	  nrows <= max_non_padded_dim
	 * In "transposing" mode: nrows <= max_non_padded_dim  &&  ncols <= max_non_padded_dim
	 */
	index_t const max_dim = ( transpose ? (MAX(ncols, nrows)) : nrows );

	index_t const min_cols = MIN( memory_alignment, ncols );

	uintmax_t const nitems = (uintmax_t) nrows * (uintmax_t) pitch;

	#if ! (NMFGPU_DEBUG_READ_MATRIX || NMFGPU_DEBUG_READ_MATRIX2)
		if ( verbose )
	#endif
			print_message( shown_by_all, "\t\tMatrix dimensions: %" PRI_IDX " x %" PRI_IDX " (%" PRIuMAX " items)\n",
					nrows, ncols, nitems );

	if ( (nrows <= 0) + (ncols <= 0) + (pitch < min_cols) + (pitch > max_pitch) + (max_dim > max_non_padded_dim) +
		(nitems > (uintmax_t) max_num_items) ) {
		print_error( error_shown_by_all, "Error in %s( rows=%" PRI_IDX ", columns=%" PRI_IDX ", pitch=%" PRI_IDX ", transpose=%i ): ",
				function_name, nrows, ncols, pitch, transpose );
		if ( (nrows <= 0) + (ncols <= 0) )
			append_printed_error( error_shown_by_all, "Invalid matrix dimensions.\n" );
		else if ( pitch < min_cols )
			append_printed_error( error_shown_by_all, "Invalid padded dimension (pitch). It must NOT be less than the number of "
					"columns and the data alignment (currently set to % " PRI_IDX ")\n", memory_alignment );
		else {	// ( (pitch > max_pitch) + (max_dim > max_non_padded_dim) + (nitems > (uintmax_t) max_num_items) )
			if ( verbose )
				append_printed_error( error_shown_by_all, "Sorry, but your matrix exceeds the limits used for matrix dimensions.\n"
						"On this system, and with the given input arguments, data matrices are limited to:\n" );
			else
				append_printed_error( error_shown_by_all, "Data matrices are limited to:\n" );
			append_printed_error( error_shown_by_all, "\t* %" PRI_IDX " rows.\n\t*%" PRI_IDX " columns.\n\t*%zu items.\n",
					max_non_padded_dim, max_pitch, max_num_items );
		}
		status = EXIT_FAILURE;
	}

	// -----------------------------

	return status;

} // matrix_check_dimensions

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
 * Data is read according to the format defined in "data_scn_fmt[]" and "data_size".
 *
 * If "haslabels" is 'true', a label string is read at the beginning of the line. In addition, if "skip_labels" is 'false',
 * it is appended to *labels, and "labels_length" is updated. If more memory was allocated, "max_labels_length" is also updated.
 * On the other hand, if "haslabels" is 'false' or "skip_labels" is 'true', "labels" is not referenced.
 *
 * WARNING: NO ERROR CHECKING IS PERFORMED for negative data, empty rows/columns, etc.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int matrix_read_line( FILE *restrict file, index_t current_line, char const *restrict data_scn_fmt, size_t data_size, int delimiter,
				index_t ncols, index_t incr_ncols, void *restrict pmatrix, bool haslabels, bool skip_labels,
				char *restrict *restrict labels, size_t *restrict labels_length, size_t *restrict max_labels_length )
{

	#if NMFGPU_DEBUG_READ_MATRIX2
		bool const real_data = strstr( data_scn_fmt, SCNgREAL );
	#endif

	// -----------------------------

	// Reads the row label (if any).

	if ( haslabels ) {

		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( dbg_shown_by_all, "\tReading label (skip=%i)...\n", skip_labels );
		#endif

		// Reads a token.
		char *restrict data = NULL;
		int last_char = (int) '\0';
		size_t const len_data = read_token( file, delimiter, &data, &last_char );
		if ( ! data ) {
			if ( len_data )
				print_error( error_shown_by_all, "Error in matrix_read_line()\n" );
			else
				print_error( error_shown_by_all, "Error reading input file:\nPremature end-of-file detected at line %"
						PRI_IDX ". Invalid file format.\n", current_line );
			return EXIT_FAILURE;
		}

		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( dbg_shown_by_all, "\t\tLabel(len=%zu): '%s'. Last_char=", len_data, data );
			switch( last_char ) {
				case EOF : { append_printed_message( dbg_shown_by_all, "EOF.\n" ); } break;
				case '\r': { append_printed_message( dbg_shown_by_all, "\\r.\n" ); } break;
				case '\n': { append_printed_message( dbg_shown_by_all, "\\n.\n" ); } break;
				case '\t': { append_printed_message( dbg_shown_by_all, "\\t.\n" ); } break;
				case  ' ': { append_printed_message( dbg_shown_by_all, "' '.\n" ); } break;
				default  : {
					append_printed_message( dbg_shown_by_all, (isgraph(last_char) ? "'%c'.\n" : "'\\0x%X'.\n"), last_char);
				}
				break;
			}
		#endif

		if ( last_char != delimiter ) {	// Blank line.
			print_error( error_shown_by_all, "Error reading input file: No matrix data at line %" PRI_IDX
					". Invalid file format.\n", current_line );
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
					print_message( dbg_shown_by_all, "\t\tExpanding size of labels to %zu chars.\n", max_len_labels );
				#endif

				char *restrict const tmp = (char *restrict) realloc( labels_str, max_len_labels * sizeof(char) );
				if ( ! tmp ) {
					print_errnum( sys_error_shown_by_all, errno, "Error in matrix_read_line(): realloc( labels, "
							"max_labels_length=%zu )", max_len_labels );
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
		print_message( dbg_shown_by_all, "\tReading data matrix (ncols=%" PRI_IDX ", incr_ncols=%" PRI_IDX ", delimiter='",
				ncols, incr_ncols );
		switch( delimiter ) {
			case '\t': { append_printed_message( dbg_shown_by_all, "\\t" ); } break;
			case  ' ': { append_printed_message( dbg_shown_by_all, " " ); } break;
			default  : { append_printed_message( dbg_shown_by_all, (isgraph(delimiter) ? "%c" : "\\0x%X"), delimiter); }
				break;
		}
		append_printed_message( dbg_shown_by_all, "', data_scn_fmt[%zu]='%s')...\n", strlen(data_scn_fmt), data_scn_fmt );
	#endif

	void *pmatrix_r = pmatrix;

	// First ncols-1 columns
	for ( index_t col = 0 ; col < (ncols-1) ; col++, pmatrix_r += (incr_ncols * data_size) ) {

		char c[2] = { 0, 0 };			// Delimiter to read.
		int conv = 0;				// Number of matched items.
		data_t value;				// Value to read.
		value.r = REAL_C( 0.0 );		// Initialized as real-type, since sizeof(real) >= sizeof(index_t)

		/* Reads a value.
		 * If no conversion is performed, it may be a missing data or
		 * an invalid file.
		 */
		errno = 0;
		conv = fscanf( file, data_scn_fmt, &value );

		#if NMFGPU_DEBUG_READ_MATRIX2
		{
			int const err = errno;
			append_printed_message( dbg_shown_by_all, "\tconv=%i,value=", conv );
			if ( real_data )
				append_printed_message( dbg_shown_by_all, "%g(real)", value.r );
			else
				append_printed_message( dbg_shown_by_all, "%" PRI_IDX "(index_t)", value.i );
			errno = err;
		}
		#endif

		// Reads the delimiter.
		if ( conv != EOF ) {
			// Format string for fscanf(3). NOTE: automatic skip of whitespace characters is disabled.
			char const delim_scn_fmt[8] = { '%', '[', delimiter, ']', 0, 0, 0, 0 };
			errno = 0;
			conv = fscanf( file, delim_scn_fmt, c );

			#if NMFGPU_DEBUG_READ_MATRIX2
			{
				int const err = errno;
				append_printed_message( dbg_shown_by_all, ",conv2=%i,c=", conv );
				if ( ! c[0] ) append_printed_message( dbg_shown_by_all, "(empty)" );
				else if ( (int) c[0] == delimiter ) append_printed_message( dbg_shown_by_all, "'%c' (delim)", c[0] );
				else append_printed_message( dbg_shown_by_all, (isgraph(c) ? "%c" : "'\\0x%X'"), c[0] );
				errno = err;
			}
			#endif
		}

		// Fails on error or premature EOF/EOL.
		if ( (int) c[0] != delimiter ) {
			if ( ferror(file) )	// Error.
				print_errnum( sys_error_shown_by_all, errno, "Error in matrix_read_line() reading line %" PRI_IDX
						", column %" PRI_IDX ": fscanf()", current_line, col + haslabels + 1 );
			else if ( feof(file) )
				print_error( error_shown_by_all, "Error reading input file:\nPremature end of file detected in line %"
						PRI_IDX " (%" PRI_IDX " columns found, %" PRI_IDX " expected).\nInvalid file format.\n",
						current_line, col + haslabels, ncols + haslabels );
			else {	// Invalid data or premature EOL.
				int const chr = ( c[0] ? c[0] : fgetc( file ) );	// (First) Illegal character.
				if ( (chr == (int) '\n') + (chr == (int) '\r') )	// Premature EOL.
					print_error( error_shown_by_all, "Error reading input file:\nPremature end of line detected at line %"
							PRI_IDX " (%" PRI_IDX " columns found, %" PRI_IDX " expected).\nInvalid file format.\n",
							current_line, col + haslabels, ncols + haslabels );
				else {
					print_error( error_shown_by_all, "Error reading input file at line %" PRI_IDX ", column %" PRI_IDX
							". Unexpected character: ", current_line, col + haslabels + 1 );
					append_printed_error( error_shown_by_all, (isgraph(chr) ? "'%c'\n" : "'\\0x%X'\n"), chr );
					if ( chr == (int) ',' )
						append_printed_error( error_shown_by_all, "Please remember that dot ('.') is the only "
									"accepted decimal symbol, and there is no thousand separator.\n" );
				}
			}
			return EXIT_FAILURE;
		}

		// Sets the new value.
		if ( ! memcpy( pmatrix_r, &value, data_size ) ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_read_line() at line %" PRI_IDX ", column %" PRI_IDX
					": memcpy(matrix <-- file)", current_line, col + haslabels + 1 );
			return EXIT_FAILURE;
		}

	} // for (0 <= col < ncols-1)

	// last file column: col = ncols-1
	{
		index_t const col = ncols - 1;

		char c[3] = { 0, 0, 0 };		// Newline to read ("\r\n" or "\n"), followed by '\0'.
		int conv = 0;				// Number of matched items.
		data_t value;				// Value to read.
		value.r = REAL_C( 0.0 );		// Initialized as real-type, since sizeof(real) >= sizeof(index_t)

		/* Reads a value.
		 * If no conversion is performed, it may be a missing data or
		 * an invalid file.
		 */
		errno = 0;
		conv = fscanf( file, data_scn_fmt, &value );

		#if NMFGPU_DEBUG_READ_MATRIX2
		{
			int const err = errno;
			append_printed_message( dbg_shown_by_all, "\tconv(eol)=%i,value(eol)=", conv );
			if ( real_data )
				append_printed_message( dbg_shown_by_all, "%g", value.r );
			else
				append_printed_message( dbg_shown_by_all, "%" PRI_IDX, value.i );
			errno = err;
		}
		#endif

		// Reads the newline.
		if ( conv != EOF ) {
			errno = 0;
			conv = fscanf( file, "%2[\r\n]", c );

			#if NMFGPU_DEBUG_READ_MATRIX2
			{
				int const err = errno;
				append_printed_message( dbg_shown_by_all, ",conv2(last)=%i,c(last)=", conv );
				if ( c[0] ) {
					for ( size_t i = 0 ; i < 2 ; i++ )
						switch( c[i] ) {
							case '\0': break;
							case '\r': { append_printed_message( dbg_shown_by_all, "\\r" ); } break;
							case '\n': { append_printed_message( dbg_shown_by_all, "\\n" ); } break;
						}
				} else append_printed_message( dbg_shown_by_all, "(empty)" );
				errno = err;
			}
			#endif
		}

		// Fails on error, or if neither EOL nor EOF was read.
		if ( ferror(file) + (! (c[0] + feof(file))) + ((c[0] == '\r') * (c[1] != '\n')) + (c[1] == '\r') ) {
			if ( ferror(file) )	// Error.
				print_errnum( sys_error_shown_by_all, errno, "Error in matrix_read_line() at line %" PRI_IDX
						", column %" PRI_IDX ": fscanf()", current_line, col + haslabels + 1 );

			else if ((c[0] == '\r') + (c[1] == '\r')) {  // (c[0] == '\r' && c[1] != '\n') || (c[1] == '\r')
				print_error( error_shown_by_all, "Error in input file at line %" PRI_IDX ", column %" PRI_IDX
						". Invalid character sequence for end-of-line:", current_line, col + haslabels + 1 );
				for ( size_t i = 0 ; i < 2 ; i++ )
					switch( c[i] ) {
						case '\0': { append_printed_error( error_shown_by_all, " '\\0'" ); } break;
						case '\r': { append_printed_error( error_shown_by_all, " '\\r'" ); } break;
						case '\n': { append_printed_error( error_shown_by_all, " '\\n'" ); } break;
						default: {
							append_printed_error( error_shown_by_all, ( isgraph(c[i]) ? (" '%c'") : (" '\\0x%X'") ),
										c[i] );
						} break;
					}
				append_printed_error( error_shown_by_all, "\nOnly UNIX (LF, '\\n') and MS-DOS/Windows (CR+LF, '\\r\\n'),"
						" are accepted.\n" );

			} else { // Invalid datum: (!c[0] && !EOF)
				int const chr = fgetc( file );		// (First) Illegal character.
				if ( chr == delimiter )			// There are more than <ncols> columns.
					print_error( error_shown_by_all, "Error in input file at line %" PRI_IDX ": Invalid file format.\n"
							"There are more than %" PRI_IDX " columns.\n", current_line, ncols + haslabels );
				else {
					print_error( error_shown_by_all, "Error in input file at line %" PRI_IDX ", column %" PRI_IDX
							". Unexpected character: ", current_line, col + haslabels + 1 );
					append_printed_error( error_shown_by_all, ( isgraph(chr) ? ("'%c'\n") : ("'\\0x%X'\n") ), chr );
					if ( chr == (int) ',' )
						append_printed_error( error_shown_by_all, "Please remember that dot ('.') is the only "
								"accepted decimal symbol, and there is no thousand separator.\n" );
				}
			}
			return EXIT_FAILURE;
		}

		// Returns any character that belongs to the next line.
		if ( ((c[0] == '\n') * c[1]) && (ungetc(c[1], file) == EOF) ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_read_line() at line %" PRI_IDX
					", column %" PRI_IDX ": ungetc()", current_line, col + haslabels + 1 );
			return EXIT_FAILURE;
		}

		// Finally, sets the new value.
		if ( ! memcpy( pmatrix_r, &value, data_size ) ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_read_line() at line %" PRI_IDX ", column %" PRI_IDX
					": memcpy(matrix <-- file)", current_line, col + haslabels + 1 );
			return EXIT_FAILURE;
		}

	} // last column.

	// -----------------------------

	#if NMFGPU_DEBUG_READ_MATRIX2
	{
		int const err = errno;
		append_printed_message( dbg_shown_by_all, "\n" );
		errno = err;
	}
	#endif

	return EXIT_SUCCESS;

} // matrix_read_line

////////////////////////////////////////////////

/*
 * Returns 'true' if "str" represents one or more numbers, or an empty string.
 */
static bool is_number( char const *restrict str )
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

} // is_number

// ---------------------------------------------

/* Checks for blank lines or EOF.
 *
 * Fails if, after one or more blank lines, there is more matrix data, or if
 * the input files contains more than a fixed number of blank lines.
 *
 * Return EXIT_SUCCESS or EXIT_FAILURE
 */
static int check_blank_lines( FILE *restrict file, index_t current_line )
{

	/* NOTE:
	 * Input file is considered as INVALID if it has <MAX_NUM_BLANK_LINES>
	 * or more CONSECUTIVE blank lines.
	 */
	#define MAX_NUM_BLANK_LINES (16)
	size_t num_bl = 0;
	bool bad_eol = false;

	char last_char = 0;

	do {
		char chr[3] = { 0, 0, 0 };	// Newline to read ("\r\n" or "\n"), followed by '\0'.
		errno = 0;

		#if NMFGPU_DEBUG_READ_MATRIX2
			int const conv =
		#endif
		fscanf( file, "%2[\r\n]", chr );

		#if NMFGPU_DEBUG_READ_MATRIX2
		{
			int const err = errno;
			print_message( dbg_shown_by_all, "conv(line " PRI_IDX " + %zu)=%i,chr=", current_line, num_bl, conv );
			if ( chr[0] ) {
				for ( size_t i = 0 ; i < 2 ; i++ )
					switch( chr[i] ) {
						case '\0': break;
						case '\r': { append_printed_message( dbg_shown_by_all, "\\r" ); } break;
						case '\n': { append_printed_message( dbg_shown_by_all, "\\n" ); } break;
					}
				append_printed_message( dbg_shown_by_all, "\n" );
			} else
				append_printed_message( dbg_shown_by_all, "(empty)\n" );
			errno = err;
		}
		#endif

		/* Counts blank lines as follows:
		 *	chr[0..1]: '\n' '\n'				: +2 lines
		 *	chr[0..1]: '\r' '\n' || '\n' '\0'		: +1 line
		 *	chr[0..1]: '\0'					: +0 lines (conv == 0)
		 *	chr[0..1]: '\n' '\r' || '\r' '\0' || '\r' '\r'  : Error
		 */
		bad_eol = ( ((chr[0] == '\r') * (chr[1] != '\n')) + (chr[1] == '\r') ); // Bad EOL
		num_bl += (chr[0] == '\n') + (chr[1] == '\n') - bad_eol; // -bad_eol: To correct num_bl.

		last_char = chr[1];

	} while ( (last_char == (int) '\n') * (num_bl < MAX_NUM_BLANK_LINES) );	// && (! bad_eol)

	// Error or invalid file format.
	if ( ferror(file) + bad_eol + (num_bl * (! feof(file))) ) {
		if ( ferror(file) )
			print_errnum( sys_error_shown_by_all, errno, "Error in check_blank_lines() at line %" PRI_IDX
					": fscanf( blank_lines=%zu )", current_line, num_bl );
		else if ( bad_eol )
			print_error( error_shown_by_all, "Error in input file at line %" PRI_IDX ": invalid end-of-line.\n"
					"Only UNIX (LF, '\\n') and MS-DOS/Windows (CR+LF, '\\r\\n') styles are accepted.\n",
					current_line + num_bl );
		else if ( num_bl >= MAX_NUM_BLANK_LINES )
			print_error( error_shown_by_all, "Warning: There are more than %d blank lines starting at file line %" PRI_IDX
					". It seems NOT to be a valid file. Aborting...\n", MAX_NUM_BLANK_LINES, current_line+1);
		else
			print_error( error_shown_by_all, "Error in input file: No matrix data at line %" PRI_IDX "\n"
					"Invalid file format.\n\n", current_line + 1 );
		return EXIT_FAILURE;
	}

	#undef MAX_NUM_BLANK_LINES

	return EXIT_SUCCESS;

} // check_blank_lines

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
		if ( ! filename ) print_errnum( error_shown_by_all, errnum, "matrix_load_ascii_verb( filename )" );
		if ( ! matrix )	print_errnum( error_shown_by_all, errnum, "matrix_load_ascii_verb( matrix )" );
		if ( ! nrows )	print_errnum( error_shown_by_all, errnum, "matrix_load_ascii_verb( nrows )" );
		if ( ! ncols )	print_errnum( error_shown_by_all, errnum, "matrix_load_ascii_verb( ncols )" );
		if ( ! pitch )	print_errnum( error_shown_by_all, errnum, "matrix_load_ascii_verb( pitch )" );
		if ( ! mt )	print_errnum( error_shown_by_all, errnum, "matrix_load_ascii_verb( mt )" );
		return EXIT_FAILURE;
	}

	/////////////////////////////////

	// Starts Reading ...

	FILE *restrict const file = fopen( filename, "r" );
	if ( ! file ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_ascii_verb(): fopen( %s )", filename );
		return EXIT_FAILURE;
	}


	/////////////////////////////////


	// Reads line 1

	status = read_tag( file, delimiter, &tokensL1, &len_tokensL1, &num_tokensL1 );
	if ( status ) {	// EOF, error, or invalid format
		if ( status == 1 )	// EOF
			print_error( error_shown_by_all, "Error reading input file: file is empty?\n" );
		else if ( status == 2 )	// Internal error.
			print_error( error_shown_by_all, "Error in matrix_load_ascii_verb( line 1 ).\n" );
		else	// 3: Maximum line length
			print_error( error_shown_by_all, "Is your data matrix stored in a single text line?\nPlease check also for "
					"any invalid line terminator. Only LF ('\\n') and CR+LF ('\\r\\n') are accepted.\n" );
		fclose(file);
		return EXIT_FAILURE;
	}


	/////////////////////////////////


	#if NMFGPU_DEBUG_READ_MATRIX
		print_message( dbg_shown_by_all, "Line1: len=%zu, ntokens=%zu\n", len_tokensL1, num_tokensL1 );
	#endif


	// Checks for overflow (compares with <max_pitch> to make sure that padding will not overflow).
	if ( (num_tokensL1-1) > (size_t) max_pitch ) {	// - 1, because the file may have a "name" field.
		print_message( error_shown_by_all, "\t\tNumber of items read on line 1: %zu.\n", num_tokensL1 );
		append_printed_error( error_shown_by_all, "\nSorry, but your matrix exceeds the limits for matrix dimensions.\n"
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
		while ( (i < nt) && is_number(pdataL1[i]) )
			i++;

		// File might have name and/or headers if:
		has_name_headers = (	(i < nt) +				   // Not all tokens, from the second one, are numeric, <OR>
					numeric_hdrs +				   // input matrix has numeric column headers, <OR>
					( (nt == 1) && (! is_number(pdataL1[0])) ) ); // It has only one (non-numeric) token.
	}

	#if NMFGPU_DEBUG_READ_MATRIX
		print_message( dbg_shown_by_all, "has_name_headers=%i\n", has_name_headers );
	#endif

	if ( has_name_headers ) {

		// File may have name and/or headers

		// Reads line 2.
		status = read_tag( file, delimiter, &tokensL2, &len_tokensL2, &num_tokensL2 );

		if ( status ) {	// EOF, error, or invalid format
			if ( status == 2 )	// Internal error.
				print_error( error_shown_by_all, "Error in matrix_load_ascii_verb( line 2 ).\n" );
			else {	// EOF or maximum line length
				if ( status == 1 )	// EOF
					print_error( error_shown_by_all, "Error reading input file: premature end of file.\n" );
				append_printed_error( error_shown_by_all, "Is your data matrix stored in a single text line?\n"
						"Please check also for any invalid line terminator; only LF ('\\n') and CR+LF ('\\r\\n') "
						"are accepted.\n" );
			}
			clean_tag(tokensL1);
			fclose(file);
			return EXIT_FAILURE;
		}

		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( dbg_shown_by_all, "num_tokensL2=%zu\n", num_tokensL2 );
		#endif

		// If both L1 and L2 have only one token, "retokenizes" them using a space character (' ') as delimiter.
		if ( (num_tokensL2 * num_tokensL1) == 1 ) {

			// Sets the new delimiter.
			delimiter = (int) ' ';

			char *restrict data = NULL;

			#if NMFGPU_DEBUG_READ_MATRIX
				print_message( dbg_shown_by_all, "Retokenizing dataL1...\n" );
			#endif

			// Removes previous array of pointers to tokens.
			data = (char *restrict) tokensL1.tokens;
			free( (void *) tokensL1.ptokens );
			num_tokensL1 = 0;

			tokensL1 = tokenize( data, delimiter, &num_tokensL1 );

			#if NMFGPU_DEBUG_READ_MATRIX
				print_message( dbg_shown_by_all, "tokensL1: %s, num_tokensL1=%zu\n", ( tokensL1.tokens ? "non-NULL" : "NULL" ),
						num_tokensL1 );
			#endif

			if ( ! tokensL1.tokens ) {
				print_error( error_shown_by_all, "Error in matrix_load_ascii_verb(): re-tokenize line 1.\n" );
				clean_tag(tokensL2);
				free( (void *) data );
				fclose(file);
				return EXIT_FAILURE;
			}

			// Checks for overflow.
			if ( (num_tokensL1-1) > (size_t) max_pitch ) {	// '-1' because the file may have a "name" field.
				print_message( error_shown_by_all, "\t\tNumber of items read on line 1: %zu.\n", num_tokensL1 );
				append_printed_error( error_shown_by_all, "\nSorry, but your matrix exceeds the limits used for matrix "
							"dimensions.\nOn this system and with the given input arguments, data matrices "
							"are limited to:\n\t* %" PRI_IDX " rows.\n\t*%" PRI_IDX " columns.\n\t*%zu items.\n\n"
							"Please check also for any invalid line terminator. Only LF ('\\n') and CR+LF "
							"('\\r\\n') are accepted.\n\n", max_non_padded_dim, max_pitch, max_num_items );
				clean_tag(tokensL2);
				clean_tag(tokensL1);
				fclose(file);
				return EXIT_FAILURE;
			}

			// ----------------------

			#if NMFGPU_DEBUG_READ_MATRIX
				print_message( dbg_shown_by_all, "Retokenizing dataL2...\n" );
			#endif

			// Removes previous array of pointers to tokens.
			data = (char *restrict) tokensL2.tokens;
			free( (void *) tokensL2.ptokens );
			num_tokensL2 = 0;

			tokensL2 = tokenize( data, delimiter, &num_tokensL2 );

			#if NMFGPU_DEBUG_READ_MATRIX
				print_message( dbg_shown_by_all, "tokensL2: %s, num_tokensL2=%zu\n", ( tokensL2.tokens ? "non-NULL" : "NULL" ),
						num_tokensL2 );
			#endif

			if ( ! tokensL2.tokens ) {
				print_error( error_shown_by_all, "Error in matrix_load_ascii_verb(): re-tokenize line 2.\n" );
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
				print_message( dbg_shown_by_all, "Retokenizing dataL1...\n" );
			#endif

			// New delimiter.
			delimiter = (int) ' ';

			// Removes previous array of pointers to tokens.
			char *restrict data = (char *restrict) tokensL1.tokens;
			free( (void *) tokensL1.ptokens );
			num_tokensL1 = 0;

			tokensL1 = tokenize( data, delimiter, &num_tokensL1 );

			#if NMFGPU_DEBUG_READ_MATRIX
				print_message( dbg_shown_by_all, "tokensL1: %s, num_tokensL1=%zu\n", ( tokensL1.tokens ? "non-NULL" : "NULL" ),
						num_tokensL1 );
			#endif

			if ( ! tokensL1.tokens ) {
				print_error( error_shown_by_all, "Error in matrix_load_ascii_verb( re-tokenize line 1 ).\n" );
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

	if ( (! is_number(tokensL2.tokens)) + numeric_lbls ) {	// First token in L2 is not numeric, or input matrix has numeric row labels.

		// File contains row labels.
		numcols = (index_t) (num_tokensL2 - 1);
		haslabels = true;
		print_message( shown_by_all, "\t\tRow labels detected.\n\t\tNumber of data columns detected (excluding row labels): %"
				PRI_IDX ".\n", numcols );

	} else {
		// No labels: numcols = num_tokensL2

		// If data matrix seems to be numeric only.
		if ( ! has_name_headers )
			print_message( shown_by_all, "\t\tNumeric-only data.\n" );

		numcols = (index_t) num_tokensL2;

		print_message( shown_by_all, "\t\tNumber of data columns detected: %" PRI_IDX ".\n", numcols );

	} // If file contains row labels

	// Checks for invalid number of columns.
	if ( ( numcols <= 1 ) + ( (num_tokensL2 - haslabels) > (size_t) max_pitch ) ) {
		if ( numcols <= 1 )
			print_error( error_shown_by_all, "Error reading input file:\nInvalid file format or the number of "
					"columns is less than 2.\n"
					"Please remember that columns must be separated by TAB characters (or by single space "
					"characters under certain conditions).\nFinally, please check for any invalid decimal "
					"symbol (e.g., ',' instead of '.').\n\n" );
		else
			print_error( error_shown_by_all, "Sorry, but your matrix exceeds the limits used for matrix dimensions.\n"
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

			print_message( shown_by_all, "\t\tName (i.e., description string) detected.\n" );

			// Splits tokensL1 as name and headers.

			char *restrict data = (char *restrict) tokensL1.tokens;
			char **restrict pdata = (char **restrict) tokensL1.ptokens;

			// Copies the first token.
			name = (char *restrict) malloc( (strlen(data) + 1) * sizeof(char) );
			if ( ! name ) {
				print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_ascii_verb(): malloc( name, size=%zu )",
						strlen(data) + 1 );
				clean_tag(tokensL2); clean_tag(tokensL1);
				fclose(file);
				return EXIT_FAILURE;
			}
			strcpy( name, data );

			#if NMFGPU_DEBUG_READ_MATRIX
				print_message( dbg_shown_by_all, "\t\tName (len=%zu):'%s'\n", strlen(name), name );
			#endif

			// Headers
			print_message( shown_by_all, "\t\tColumn headers detected.\n" );
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
				print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_ascii_verb(): memmove( headers, size=%zu )",
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
				print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_ascii_verb(): realloc( pheaders, numcols=%"
						PRI_IDX " )", numcols );
				struct matrix_tags_t l_mt = new_matrix_tags( name, tokensL2, tokensL1 );
				clean_matrix_tags(l_mt);
				fclose(file);
				return EXIT_FAILURE;
			}

			headers = new_tag( data, pdata );

		} else if ( num_tokensL1 == (size_t) numcols ) {	// No name, headers only.

			print_message( shown_by_all, "\t\tColumn headers detected.\n" );
			hasheaders = true;
			headers = tokensL1;

		} else if ( num_tokensL1 == 1 ) {	// No headers, name only

			print_message( shown_by_all, "\t\tName (i.e., description string) detected.\n" );
			name = (char *restrict) tokensL1.tokens;
			if ( tokensL1.ptokens )
				free( (void *)tokensL1.ptokens );

		} else {	// Error.

			print_error( error_shown_by_all, "Error reading input file: length of lines 1 (%zu) and 2 (%" PRI_IDX ") mismatch.\n"
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
		print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_ascii_verb(): malloc( data_matrix, max_numrows=%"
				PRI_IDX ", numcols=%" PRI_IDX ", l_pitch=%" PRI_IDX " )", max_numrows, numcols, l_pitch );
		struct matrix_tags_t l_mt = new_matrix_tags( name, headers, tokensL2 );
		clean_matrix_tags(l_mt);
		fclose(file);
		return EXIT_FAILURE;
	}

	// Sums of columns: used to check that all columns have at least one positive value.
	sum_cols = (real *restrict) malloc( l_pitch * sizeof(real) );
	if ( ! sum_cols ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_ascii_verb(): malloc( sum_cols, l_pitch=%"
				PRI_IDX " )", l_pitch );
		struct matrix_tags_t l_mt = new_matrix_tags( name, headers, tokensL2 );
		matrix_clean(data_matrix, l_mt);
		fclose(file);
		return EXIT_FAILURE;
	}

	#if NMFGPU_DEBUG_READ_MATRIX
		print_message( dbg_shown_by_all, "Setting first data line (file line %" PRI_IDX ")...\n", nlines );
	#endif

	{
		real sum_row = REAL_C( 0.0 );
		real min_value = R_MAX;

		for ( index_t j = 0 ; j < numcols; j++ ) {

			// Token (data) to read.
			char const *const val_str = tokensL2.ptokens[ j + haslabels ];
			char *endptr = NULL;

			#if NMFGPU_DEBUG_READ_MATRIX2
				append_printed_message( dbg_shown_by_all, "\tstr='%s'", val_str );
			#endif

			// Transforms char* to real.
			errno = 0;
			real const value = STRTOREAL( val_str, &endptr );

			#if NMFGPU_DEBUG_READ_MATRIX2
			{
				int const err = errno;
				append_printed_message( dbg_shown_by_all, "val=%g,endptr=", value );
				char const c = *endptr;
				append_printed_message( dbg_shown_by_all, ( (isprint(c) + isblank(c)) ? ("'%c'\n") : ("'\\x%X'\n") ), c );
				errno = err;
			}
			#endif

			// No numeric characters <OR> NaN, inf, underflow/overflow, etc
			if ( errno + (! is_valid(value)) + (*endptr) ) {	// (errno != 0) || (*endptr != '\0') || (! is_valid(value))
				print_errnum( error_shown_by_all, errno, "Error reading line %" PRI_IDX ", column %" PRI_IDX
						". Invalid numeric value: '%s'", nlines, (j + haslabels + 1), val_str );
				if ( ! errno )
					append_printed_error( error_shown_by_all, "Please, check also for invalid decimal symbols (if any).\n" );
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
			print_error( error_shown_by_all, "Error in input file at line %" PRI_IDX ": ", nlines );
			if ( min_value < REAL_C( 0.0 ) )
				append_printed_error( error_shown_by_all, "negative value(s) detected (e.g., %g).\n", min_value );
			else
				append_printed_error( error_shown_by_all, "\"empty\" row detected.\nAll rows and columns must "
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

	// Checks for blank lines or EOF.
	status = check_blank_lines( file, nlines );
	if ( status != EXIT_SUCCESS ) {
		free(sum_cols);
		struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
		matrix_clean( data_matrix, l_mt );
		fclose(file);
		return EXIT_FAILURE;
	}

	/////////////////////////////////

	// Format string for fscanf(3)
	char data_scn_fmt[8];
	snprintf( data_scn_fmt, 8, "%%" SCNgREAL );

	/////////////////////////////////


	while ( ! feof(file) ) {

		nlines++;	// A new line is going to be read.

		#if NMFGPU_DEBUG_READ_MATRIX2
			print_message( dbg_shown_by_all, "==============\nReading line %" PRI_IDX "...\n", nlines );
		#endif

		/////////////////////////////////////////

		// Checks for overflow.
		if ( ( ((uintmax_t) nitems_p + (uintmax_t) l_pitch) > (uintmax_t) max_num_items ) +
			( (uintmax_t) numrows >= (uintmax_t) max_non_padded_dim ) ) {
			print_message( error_shown_by_all, "\t\tNumber of matrix rows currently read: %" PRI_IDX ".\n"
					"\t\tNumber of matrix entries currently read: %zu.\n", numrows, nitems );
			append_printed_error( error_shown_by_all, "\nSorry, but your matrix exceeds the limits used for matrix dimensions.\n"
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
				print_message( dbg_shown_by_all, "\tExpanding memory for a total of %" PRI_IDX " rows...\n", max_numrows );
			#endif

			// data_matrix
			real *restrict const tmp = (real *restrict) realloc( data_matrix, (size_t) max_numrows * (size_t) l_pitch * sizeof(real) );
			if ( ! tmp )  {
				print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_ascii_verb(): realloc( data_matrix, "
						"max_numrows=%" PRI_IDX", numcols=%" PRI_IDX ", l_pitch=%" PRI_IDX " )", max_numrows, numcols,
						l_pitch );
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
		{
			size_t const data_size = sizeof(real);
			index_t const incr_numcols = 1;
			bool const skip_labels = false;

			status = matrix_read_line( file, nlines, data_scn_fmt, data_size, delimiter, numcols, incr_numcols, pmatrix, haslabels,
							skip_labels, (char *restrict *restrict) &(labels.tokens), &len_labels, &max_len_labels );

			if ( status != EXIT_SUCCESS ) {
				print_error( error_shown_by_all, "Error reading input file.\n" );
				free(sum_cols);
				struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
				matrix_clean( data_matrix, l_mt );
				fclose(file);
				return EXIT_FAILURE;
			}
		}

		// Checks all data read, computes the minimum value, and updates sum_cols[] and sum_row[]
		{
			real sum_row = REAL_C( 0.0 );
			real min_value = R_MAX;

			for ( index_t j = 0 ; j < numcols; j++ ) {

				real const value = pmatrix[j];

				// Checks for NaN, inf, underflow/overflow, etc.
				if ( ! is_valid( value ) ) {
					print_error( error_shown_by_all, "Error at line %" PRI_IDX ", column %" PRI_IDX
							". Invalid numeric value: '%s'\n", nlines, (j + haslabels + 1) );
					free(sum_cols);
					struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
					matrix_clean( data_matrix, l_mt );
					fclose(file);
					return EXIT_FAILURE;
				}

				if ( value < min_value )
					min_value = value;

				sum_row += value;

				sum_cols[ j ] += value;

			} // for

			if ( (sum_row < R_MIN) + (min_value < REAL_C(0.0)) ) {
				print_error( error_shown_by_all, "Error in input file at line: %" PRI_IDX ": ", nlines );
				if ( min_value < REAL_C(0.0) )
					append_printed_error( error_shown_by_all, "negative value(s) detected (e.g., %g).\n", min_value );
				else	// sum_row < R_MIN
					append_printed_error( error_shown_by_all, "\"empty\" row detected.\nAll rows and columns must "
							"have at least one value greater than or equal to %" PRI_IDX "\n", R_MIN );
				free(sum_cols);
				struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
				matrix_clean( data_matrix, l_mt );
				fclose(file);
				return EXIT_FAILURE;
			}

		} // Checks data

		numrows++;
		nitems += (size_t) numcols;	// == numrows * numcols
		nitems_p += (size_t) l_pitch;	// == numrows * l_pitch
		pmatrix += l_pitch;

		// -----------------------------

		// Checks for blank lines or EOF.

		status = check_blank_lines( file, nlines );
		if ( status != EXIT_SUCCESS ) {
			free(sum_cols);
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
			matrix_clean( data_matrix, l_mt );
			fclose(file);
			return EXIT_FAILURE;
		}


	} // while ( ! EOF )

	fclose(file);

	print_message( shown_by_all, "\t\tLoaded a %" PRI_IDX " x %" PRI_IDX " data matrix (%zu items).\n", numrows, numcols, nitems );

	// Fails on "empty" columns
	// NOTE: There are faster alternatives, but we want to tell the user which column is "empty".
	for ( index_t i = 0 ; i < numcols ; i++ )
		if ( sum_cols[ i ] < R_MIN ) {
			print_error( error_shown_by_all, "Error in input file: column %" PRI_IDX " is \"empty\".\nAll rows and columns must "
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
			print_message( dbg_shown_by_all, "\t\tResizing labels from %zu to %zu, and plabels from %" PRI_IDX " to %" PRI_IDX "\n",
					max_len_labels, len_labels, max_numrows, numrows );
		#endif

		errno = 0;
		char const *restrict const data = (char const *restrict) realloc( (void *) labels.tokens, len_labels * sizeof(char) );
		if ( ! data ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_ascii_verb(): realloc( labels, len_labels=%zu )",
					len_labels );
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
			matrix_clean( data_matrix, l_mt );
			fclose(file);
			return EXIT_FAILURE;
		}
		labels.tokens = data;

		errno = 0;
		char **restrict const pdata = (char **restrict) realloc( (void *) labels.ptokens, numrows * sizeof(char *) );
		if ( ! pdata ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_ascii_verb(): realloc( plabels, numrows=%"
					PRI_IDX " )", numrows );
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
			matrix_clean( data_matrix, l_mt );
			fclose(file);
			return EXIT_FAILURE;
		}
		labels.ptokens = (char const *const *restrict) pdata;

		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( dbg_shown_by_all, "\t\"Retokenizing\" labels...\n" );
		#endif

		// Resets labels.ptokens[].
		retok( labels, numrows );
	}

	// Adjusts memory used by data_matrix
	{
		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( dbg_shown_by_all, "Resizing matrix from %" PRI_IDX "rows to %" PRI_IDX "...\n", max_numrows, numrows );
		#endif
		real *restrict const tmp = (real *restrict) realloc( data_matrix, nitems_p * sizeof(real) );
		if ( ! tmp ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_ascii_verb(): realloc( data_matrix, numrows=%"
					PRI_IDX " )", numrows );
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
			matrix_clean( data_matrix, l_mt );
			fclose(file);
			return EXIT_FAILURE;
		}
		data_matrix = tmp;
	}

	#if NMFGPU_DEBUG_READ_MATRIX
		print_message( dbg_shown_by_all, "\tLoad matrix finished!\n");
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
 * Loads a matrix from an ASCII file.
 *
 * Skips name, headers and labels if "mt" is NULL.
 *
 * If (*matrix) is non-NULL, do not allocates memory but uses the supplied one.
 * WARNING: In such case, THERE MUST BE ENOUGH MEMORY ALREADY ALLOCATED.
 *
 * If input file does not have labels, accepts both tab and space characters as delimiters.
 *
 * If "transpose" is 'true':
 *	- Reads from file:
 *		<ncols> rows and <nrows> columns.
 *		<ncols> row labels (set as mt->headers), and <nrows> column headers (set as mt->labels).
 *	- Writes to "*matrix": <nrows> rows and <ncols> columns (padded to <pitch>).
 *
 * ncols <= pitch
 *
 * WARNING:
 *	- NO ERROR CHECKING IS PERFORMED for negative data, empty rows/columns, etc.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_load_ascii( char const *restrict filename, index_t nrows, index_t ncols, index_t pitch, bool real_data, bool hasname,
			bool hasheaders, bool haslabels, bool transpose, void *restrict *restrict matrix, struct matrix_tags_t *restrict mt )
{

	// Format string for fscanf(3).
	char data_scn_fmt[8];
	snprintf( data_scn_fmt, 8, "%%%s", ( real_data ? SCNgREAL : SCN_IDX ) );
	size_t const data_size = ( real_data ? sizeof(real) : sizeof(index_t) );

	// Name, headers and labels
	char *restrict name = NULL;
	struct tag_t headers = new_empty_tag();
	struct tag_t labels = new_empty_tag();

	size_t max_len_labels = 0;		// Allocated memory for labels.tokens
	size_t len_labels = 0;			// Memory currently used in labels.tokens.

	bool memory_allocated = false;		// If memory for data matrix was allocated.

	index_t const major_dim = ( transpose ? ncols : nrows );
	index_t const minor_dim = ( transpose ? nrows : ncols );

	// Skips name, headers and labels if mt is NULL.
	bool const skip = ( ! mt );

	// Delimiter character (TAB by default).
	int delimiter = (int) '\t';

	// Line number to read.
	index_t nlines = 1;

	////////////////////////////////

	// Checks for NULL pointers
	if ( ! ( (uintptr_t) filename * (uintptr_t) matrix ) ) {
		int const errnum = EFAULT;
		if ( ! filename ) print_errnum( error_shown_by_all, errnum, "matrix_load_ascii( filename )" );
		if ( ! matrix ) print_errnum( error_shown_by_all, errnum, "matrix_load_ascii( matrix )" );
		return EXIT_FAILURE;
	}

	// Checks matrix dimensions (quiet mode)
	if ( matrix_check_dimensions( "matrix_load_ascii", nrows, ncols, pitch, transpose, false ) != EXIT_SUCCESS )
		return EXIT_FAILURE;

	// -----------------------------

	// Starts Reading ...

	FILE *restrict const file = fopen( filename, "r" );
	if ( ! file ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_ascii(): fopen( %s )", filename );
		return EXIT_FAILURE;
	}

	// ----------------------------

	// Name and/or headers (set later as mt->labels if in matrix-transposing mode).

	if ( hasname + hasheaders ) {

		struct tag_t tags = new_empty_tag();
		size_t ntokens = 0;

		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( dbg_shown_by_all, "\t\tReading name/headers...\n" );
		#endif

		int const status = read_tag( file, delimiter, &tags, NULL, &ntokens );

		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( dbg_shown_by_all, "\t\t\tstatus=%i, ntokens=%zu\n", status, ntokens );
		#endif

		if ( status ) {	// EOF, error, or invalid format
			if ( status == 1 )	// EOF
				print_error( error_shown_by_all, "Error reading input file: file is empty?\n" );
			else if ( status == 2 )	// Internal error.
				print_error( error_shown_by_all, "Error in matrix_load_ascii( line 1 ).\n" );
			else	// 3: Maximum line length
				print_error( error_shown_by_all, "Is your data matrix stored in a single text line?\nPlease check also for "
						"any invalid line terminator. Only LF ('\\n') and CR+LF ('\\r\\n') are accepted.\n" );
			fclose(file);
			return EXIT_FAILURE;
		}

		if ( skip ) {
			clean_tag( tags );
			tags = new_empty_tag();
		} else {

			// Checks for proper number of tokens.

			// If only one token, and there should be more, "retokenizes" using space as delimiter.
			if ( (ntokens == 1) * hasheaders * (hasname + minor_dim - 1) ) {	// (hasname || minor_dim > 1)

				// Sets the new delimiter.
				delimiter = (int) ' ';

				#if NMFGPU_DEBUG_READ_MATRIX
					print_message( dbg_shown_by_all, "\t\t\t\"Retokenizing\" data on line 1 (hasname=%i, hasheaders=%i,"
							" minor_dim=%" PRI_IDX ")...\n", hasname, hasheaders, minor_dim );
				#endif

				// Removes previous array of pointers to tokens.
				char *restrict data = (char *restrict) tags.tokens;
				free( (void *) tags.ptokens );
				ntokens = 0;

				tags = tokenize( data, delimiter, &ntokens );

				#if NMFGPU_DEBUG_READ_MATRIX
					print_message( dbg_shown_by_all, "\t\t\tLine 1 (tags): %s, num_tokensL1=%zu\n",
						       ( tags.tokens ? "non-NULL" : "NULL" ), ntokens );
				#endif

				if ( ! tags.tokens ) {
					print_error( error_shown_by_all, "Error in matrix_load_ascii(): re-tokenize line 1 (hasname=%i,"
							" hasheaders=%i, minor_dim=%" PRI_IDX ").\n", hasname, hasheaders, minor_dim );
					free( (void *) data );
					fclose(file);
					return EXIT_FAILURE;
				}

			} // If only one token

			if ( ntokens != (size_t) ((hasheaders * minor_dim) + hasname) ) {
				print_error( error_shown_by_all, "Error reading input file: Invalid number of items in line 1: %zu read, %"
						PRI_IDX " expected.\nInvalid file format.\n", ntokens, (hasheaders * minor_dim) + hasname );
				clean_tag(tags);
				fclose(file);
				return EXIT_FAILURE;
			}

			// ------------------------

			if ( hasname * hasheaders ) {	// Name and headers

				/* The first token is the 'name' field, and the rest are headers.
				 * size_of(tags.ptokens): minor_dim + 1
				 */

				char *restrict data = (char *restrict) tags.tokens;
				char **restrict pdata = (char **restrict) tags.ptokens;

				// Name: copies the first token.
				name = (char *restrict) malloc( (strlen(data) + 1) * sizeof(char) );
				if ( ! name ) {
					print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_ascii(): malloc( name, size=%zu )",
							strlen(data) + 1 );
					clean_tag(tags);
					fclose(file);
					return EXIT_FAILURE;
				}
				strcpy( name, data );

				#if NMFGPU_DEBUG_READ_MATRIX
					print_message( dbg_shown_by_all, "\t\tName (len=%zu):'%s'\n", strlen(name), name );
				#endif

				/* Sets remaining tokens as headers.
				 *
				 * Moves the second token to the beginning of "data", overwriting the first token (which was already
				 * copied into "name"). Remaining tokens are kept untouched, and the previous place of the second
				 * token is left as "garbage".
				 *
				 * This way, data == pdata[0], and it is possible to call free(3) on them.
				 */
				if ( ! memmove( data, pdata[1], (strlen(pdata[1]) + 1) * sizeof(char) ) ) {
					print_errnum(sys_error_shown_by_all, errno, "Error in matrix_load_ascii(): memmove( tags, size=%zu )",
							strlen( pdata[1] ) + 1 );
					free((void *)name); clean_tag(tags);
					fclose(file);
					return EXIT_FAILURE;
				}

				// Adjusts pdata[] to <minor_dim> tokens
				pdata[ 0 ] = data;
				for ( index_t i = 1 ; i < minor_dim ; i++ )
					pdata[ i ] = pdata[ i+1 ];

				pdata = (char **restrict) realloc( pdata, minor_dim * sizeof(char *) );
				if ( ! pdata ) {
					print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_ascii(): realloc( pheaders, "
							"minor_dim=%" PRI_IDX " )", minor_dim );
					free((void *)name); clean_tag(tags);
					fclose(file);
					return EXIT_FAILURE;
				}

				tags = new_tag( data, pdata );

			} else if ( hasname ) { // No headers, name only.

				name = (char *restrict) tags.tokens;	// The only token.

				if ( tags.ptokens )
					free( (void *)tags.ptokens );

				tags = new_empty_tag();
			}
			// Else, headers only

		} // if skip tags.

		headers = tags;

		// Now, reading line 2...
		nlines = 2;

	} // if has name and/or headers

	// ----------------------------

	// Labels (set later as mt->headers if in matrix-transposing mode)

	if ( haslabels * ( ! skip ) ) {

		max_len_labels = 64 * prev_power_2( major_dim );	// Initial size. It will be adjusted later.

		char *restrict const labels_str = (char *restrict) malloc( max_len_labels * sizeof(char) );
		if ( ! labels_str ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_ascii(): malloc( labels_str, size=%zu )",
					max_len_labels );
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );	// labels == NULL
			clean_matrix_tags(l_mt);
			fclose(file);
			return EXIT_FAILURE;
		}
		labels.tokens = (char const *restrict) labels_str;

		char **restrict const plabels = (char **restrict) malloc( (size_t) major_dim * sizeof(char *) );
		if ( ! plabels ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_ascii(): malloc( plabels, major_dim=%"
					PRI_IDX " )", major_dim );
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );	// labels.tokens != NULL
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
		memory_allocated = true;
		size_t const nitems = (size_t) nrows * (size_t) pitch;
		l_matrix = (void *restrict) malloc( nitems * data_size );
		if ( ! l_matrix ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_ascii(): malloc( l_matrix, nrows=%" PRI_IDX
					", ncols=%" PRI_IDX ", pitch= %" PRI_IDX ", transpose: %i, data_size=%zu )", nrows, ncols, pitch,
					transpose, data_size );
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
			clean_matrix_tags(l_mt);
			fclose(file);
			return EXIT_FAILURE;
		}
	}

	// ----------------------------

	// Reading file...

	// Step sizes for outer and inner loops.
	index_t incr_outer_loop = ( transpose ? 1 : pitch );
	index_t incr_inner_loop = ( transpose ? pitch : 1 );

	void *pmatrix = l_matrix;
	for ( index_t r = 0 ; r < major_dim ; r++, nlines++, pmatrix += (incr_outer_loop * data_size) ) {

		// Reads a full file line, including the row label, if any.
		int const status = matrix_read_line( file, nlines, data_scn_fmt, data_size, delimiter, minor_dim, incr_inner_loop, pmatrix,
							haslabels, skip, (char *restrict *restrict) &(labels.tokens), &len_labels,
							&max_len_labels);

		if ( status != EXIT_SUCCESS ) {
			if ( memory_allocated ) free( (void *) l_matrix );
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
			clean_matrix_tags(l_mt);
			fclose(file);
			return EXIT_FAILURE;
		}

	} // for ( 0 <= r < major_dim )

	fclose(file);

	// -----------------------------

	// Resizes labels.tokens (note: len_labels <= max_len_labels)
	if ( haslabels * (! skip) ) {

		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( dbg_shown_by_all, "Resizing labels from %zu to %zu\n", max_len_labels, len_labels );
		#endif

		char *restrict const data = (char *restrict) realloc( (void *) labels.tokens, len_labels * sizeof(char) );
		if ( ! data ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_ascii(): realloc( labels, len=%zu )", len_labels );
			if ( memory_allocated ) free( (void *) l_matrix );
			struct matrix_tags_t l_mt = new_matrix_tags( name, headers, labels );
			clean_matrix_tags(l_mt);
			return EXIT_FAILURE;
		}

		// Adjusts the array of pointers to labels
		retok( labels, major_dim );
	}

	// -----------------------------

	// Sets output values.

	if ( memory_allocated )
		*matrix = l_matrix;

	if ( ! skip ) {
		// Swaps row headers and column labels on matrix-transposing mode
		struct tag_t h_tags = ( transpose ? labels  : headers );
		struct tag_t l_tags = ( transpose ? headers : labels  );

		*mt = new_matrix_tags( name, h_tags, l_tags );
	}

	return EXIT_SUCCESS;

} // matrix_load_ascii

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

	for ( size_t i = 0, j = (size-1) ; i < size ; i++, j-- )
		p_out[ j ] = p_in[ i ];

} // reverse_bytes

// ---------------------------------------------

/*
 * Reads the signature from a "formatted" binary file: a 32-bits unsigned integer
 * in little-endian format.
 *
 * Returns EXIT_SUCCESS, or EXIT_FAILURE if the signature is invalid.
 */
static int read_signature( FILE *restrict file, bool real_data )
{

	uint32_t const valid_signature = ( real_data ? BIN_FILE_SIGNATURE_REAL : BIN_FILE_SIGNATURE_INDEX );
	uint32_t file_signature = UINT32_C( 0 );

	// -----------------------------

	errno = 0;
	if ( ! fread( &file_signature, sizeof(uint32_t), 1, file ) ) {
		if ( feof(file) )
			print_error( error_shown_by_all, "Error reading input file:\nPremature end-of-file detected.\n"
					"Invalid file format.\n\n" );
		else if ( ferror(file) )
			print_errnum( sys_error_shown_by_all, errno, "Error in read_signature(): fread()" );
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
		print_error( error_shown_by_all, "Error reading input file:\nInvalid signature: %" PRIX32 "\nInvalid file format.\n\n",
				file_signature );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // read_signature

// ---------------------------------------------

/*
 * Loads a matrix from a "formatted" binary file: double-precision data in
 * little-endian format.
 *
 * If (*matrix) is non-NULL, do not allocates memory, but uses the supplied one.
 * In that case, THERE MUST BE ENOUGH MEMORY ALREADY ALLOCATED (padding included).
 *
 * If "transpose" is 'true':
 *	- Reads from file:
 *		<ncols> rows and <nrows> columns.
 *		<ncols> row labels (set as mt->headers), and <nrows> column headers (set as mt->labels).
 *	- Writes to "*matrix": <nrows> rows and <ncols> columns (padded to <pitch>).
 *
 * ncols <= pitch
 *
 * If "check_errors" is 'true', makes sure that:
 *	- All rows and columns must have at least one positive value (i.e., greater than 0).
 *	- There are no negative values.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int matrix_read_binary( FILE *restrict file, bool check_errors, index_t nrows, index_t ncols, index_t pitch, bool transpose,
				real *restrict *restrict matrix )
{

	bool memory_allocated = false;	// If memory for data matrix was allocated.

	index_t const major_dim = ( transpose ? ncols : nrows );
	index_t const minor_dim = ( transpose ? nrows : ncols );

	// -----------------------------

	// Allocates memory, if necessary.

	real *restrict data_matrix = *matrix;
	if ( ! data_matrix ) {
		memory_allocated = true;
		size_t const nitems = (size_t) nrows * (size_t) pitch;
		data_matrix = (real *restrict) malloc( nitems * sizeof(real) );
		if ( ! data_matrix ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_read_binary(): malloc( data_matrix, nrows=%" PRI_IDX
					", ncols=%" PRI_IDX ", pitch=%" PRI_IDX ", transpose=%i )", nrows, ncols, pitch, transpose );
			return EXIT_FAILURE;
		}
	}

	// -----------------------------

	// Starts reading...

	#if NMFGPU_DEBUG_READ_MATRIX2
		print_message( dbg_shown_by_all, "\tReading data (transpose=%i)...\n\n", transpose );
	#endif

	// Step sizes for outer and inner loops.
	index_t incr_outer_loop = ( transpose ? 1 : pitch );
	index_t incr_inner_loop = ( transpose ? pitch : 1 );

	// Sums of inner loop: used to check that all items in within the minor dimension, have at least one positive value.
	real *restrict sum_inner_loop = NULL;
	if ( check_errors ) {
		sum_inner_loop = (real *restrict) calloc( minor_dim, sizeof(real) ); // NOTE: this is calloc(3), not malloc(3).
		if ( ! sum_inner_loop ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_read_binary(): calloc( sum_inner_loop, minor_dim=%"
					PRI_IDX " )", minor_dim );
			if ( memory_allocated )
				free( (void *) data_matrix );
			return EXIT_FAILURE;
		}
	}

	real *pmatrix = data_matrix;
	for ( index_t i=0 ; i < major_dim ; i++, pmatrix += incr_outer_loop ) {

		real sum_outer_loop = REAL_C( 0.0 );
		real min_value = R_MAX;

		real *pmatrix_r = pmatrix;
		for ( index_t j=0 ; j < minor_dim ; j++, pmatrix_r += incr_inner_loop ) {

			// Reads current data value.
			errno = 0;
			double value = 0.0;
			size_t const nread = fread( &value, sizeof(double), 1, file );	// Reads one double-precision value.

			// If this system is big-endian, reverses the byte order.
			if ( IS_BIG_ENDIAN() ) {
				double be_value = 0.0;	// Big-Endian value
				reverse_bytes( &value, sizeof(double), &be_value );
				value = be_value;
			}
			real const num = (real) value;

			#if NMFGPU_DEBUG_READ_MATRIX2
				append_printed_message( dbg_shown_by_all, "%g ", value );
			#endif

			// Checks data.
			if ( ! ( nread * is_valid(num) ) ) {
				if ( feof(file) )
					print_error( error_shown_by_all, "Error reading input file:\nPremature end-of-file detected.\n"
							"Invalid file format.\n\n");
				else if ( ferror(file) )
					print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_binary_verb( transpose=%i ): "
							"fread( major dim %" PRI_IDX "/%" PRI_IDX ", minor_dim %" PRI_IDX "/%" PRI_IDX ")",
							transpose, i, major_dim, j, minor_dim );
				else	// ! is_valid(num)
					print_error( error_shown_by_all, "Error reading input file (transpose=%i, major dim %" PRI_IDX "/%"
							PRI_IDX ", minor_dim %" PRI_IDX "/%" PRI_IDX "): '%g'.\n"
							"Invalid numeric or file format.\n\n", transpose, i, major_dim, j, minor_dim );
				if ( check_errors ) free( sum_inner_loop );
				if ( memory_allocated )  free( (void *) data_matrix );
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
			append_printed_message( dbg_shown_by_all, "\n" );
		#endif

		// Fails on "empty" major dimension, or negative value(s).
		if ( check_errors * ((sum_outer_loop < R_MIN) + (min_value < REAL_C(0.0))) ) {
			print_error( error_shown_by_all, "Error in input file at major dim %" PRI_IDX "/% " PRI_IDX " (transpose=%i): ",
					i, major_dim, transpose );
			if ( min_value < REAL_C(0.0) )
				print_error( error_shown_by_all, "negative value(s) detected (e.g., %g).\n", min_value );
			else	// sum_outer_loop < R_MIN
				print_error( error_shown_by_all, "\"empty\" row/column detected.\nAll rows and columns must "
						"have at least one value greater than or equal to %" PRI_IDX "\n", R_MIN );
			free( sum_inner_loop );
			if ( memory_allocated )  free( (void *) data_matrix );
			return EXIT_FAILURE;
		}

	} // for i

	#if NMFGPU_DEBUG_READ_MATRIX2
		append_printed_message( dbg_shown_by_all, "\n" );
	#endif

	// Fails on "empty" columns/rows
	if ( check_errors ) {
		// NOTE: There are faster alternatives, but we want to tell the user which item from the minor dimension is "emtpy".
		for ( index_t j=0 ; j < minor_dim ; j++ )
			if ( sum_inner_loop[ j ] < R_MIN ) {
				print_error( error_shown_by_all, "Error in input file at minor dim %" PRI_IDX "/%" PRI_IDX
						" (transpose=%i): the item is \"empty\".\nAll rows and columns must have "
						"at least one value greater than or equal to %" PRI_IDX "\n", j, minor_dim,
						transpose, R_MIN );
				free( sum_inner_loop );
				if ( memory_allocated )  free( (void *) data_matrix );
				return EXIT_FAILURE;
			}
		free( sum_inner_loop );
	}

	// Sets output parameters.
	if ( memory_allocated )
		*matrix = data_matrix;

	return EXIT_SUCCESS;

} // matrix_read_binary

// ---------------------------------------------

/*
 * Loads a matrix from a "native" binary file (i.e., with the native endiannes,
 * and the compiled types for matrix data).
 *
 * If (*matrix) is non-NULL, do not allocates memory, but uses the supplied one.
 * In that case, THERE MUST BE ENOUGH MEMORY ALREADY ALLOCATED (padding included).
 *
 * WARNING:
 *	- For internal-use only (i.e., for temporary files).
 *	- NO ERROR-CHECKING IS PERFORMED (e.g., overflow, invalid values,
 *	  negative data, etc).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
static int matrix_read_binary_native( FILE *restrict file, index_t nrows, index_t ncols, index_t pitch, size_t data_size,
					void *restrict *restrict matrix )
{

	bool memory_allocated = false;	// If memory for data matrix was allocated.

	// -----------------------------

	// Allocates memory, if necessary.

	void *restrict data_matrix = *matrix;
	if ( ! data_matrix ) {
		memory_allocated = true;
		size_t const nitems = (size_t) nrows * (size_t) pitch;
		data_matrix = (void *restrict) malloc( nitems * data_size );
		if ( ! data_matrix ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_read_binary_native(): malloc( data_matrix, nrows=%"
					PRI_IDX ", ncols=%" PRI_IDX ", pitch=%" PRI_IDX ", data_size=%zu )", nrows, ncols, pitch, data_size );
			return EXIT_FAILURE;
		}
	}

	// -----------------------------

	// Starts reading...

	void *pmatrix = data_matrix;
	for ( index_t i=0 ; i < nrows ; i++, pmatrix += (pitch * data_size) ) {

		errno = 0;
		size_t const nread = fread( pmatrix, data_size, ncols, file );

		if ( nread != (size_t) ncols ) {
			if ( ferror(file) )
				print_errnum( sys_error_shown_by_all, errno, "Error in matrix_read_binary_native(): fread( row %" PRI_IDX
						"/%" PRI_IDX ", %" PRI_IDX " columns)", i, nrows, ncols );
			else	// EOF
				print_error( error_shown_by_all, "Error reading input file:\nPremature end-of-file detected. "
						"Invalid file format.\n\n");
			if ( memory_allocated )
				free( (void *) data_matrix );
			return EXIT_FAILURE;
		}

	} // for i

	// -----------------------------

	// Sets output parameters.
	if ( memory_allocated )
		*matrix = data_matrix;

	return EXIT_SUCCESS;

} // matrix_read_binary_native

// ---------------------------------------------

/*
 * Reads labels, headers and name (as plain text) if they exists.
 * It also detects automatically the used delimiter symbol (space or tab character).
 *
 * verbose: If 'true', shows messages concerning the label type found.
 *
 * If "transpose" is 'true', reads from file:
 *	<ncols> row labels, set as mt->headers.
 *	<nrows> column headers, set as mt->labels.
 *
 * NOTE: This function is intended for reading tag elements from BINARY files.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int matrix_read_tags( FILE *restrict file, index_t nrows, index_t ncols, bool transpose, bool verbose, struct matrix_tags_t *restrict mt)
{

	index_t const major_dim = ( transpose ? ncols : nrows );
	index_t const minor_dim = ( transpose ? nrows : ncols );

	int const delimiter = (int) '\t';

	struct matrix_tags_t l_mt = new_empty_matrix_tags();

	// -----------------------------

	// Checks for row labels
	{

		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( dbg_shown_by_all, "\t\tReading row labels...\n" );
		#endif

		struct tag_t tags = new_empty_tag();

		size_t ntokens = 0;
		size_t len = 0;

		int status = read_tag( file, delimiter, &tags, &len, &ntokens );

		if ( status ) {	// EOF, error, or invalid format
			if ( status >= 2 ) {	// Error or invalid format.
				if ( status == 2 )	// Internal error.
					print_error( error_shown_by_all, "Error in matrix_read_tags( row labels ).\n" );
				else	// 3: Invalid format.
					print_error( error_shown_by_all, "Please remember to set a newline ('\\n') character "
							"between column headers, row labels and the description string.\n\n" );
				return EXIT_FAILURE;
			}
			// Else, EOF
			*mt = l_mt;
			return EXIT_SUCCESS;
		}
		// Else, success: one or more row labels were read.

		if ( len ) {	// Non-empty line: (len > 0) && (ntokens >= 1)

			if ( verbose )
				print_message( shown_by_all, "\t\tRow labels detected.\n" );

			// If there is only one token, and there should be more, "retokenizes" using space as delimiter.
			if ( ( ntokens == 1 ) * ( major_dim > 1 ) ) {

				#if NMFGPU_DEBUG_READ_MATRIX
					print_message( dbg_shown_by_all, "\t\t\t\"Retokenizes\" using space...\n" );
				#endif

				// Removes previous array of pointers to tokens.
				char *restrict data = (char *restrict) tags.tokens;
				free( (void *) tags.ptokens );
				ntokens = 0;

				tags = tokenize( data, (int) ' ', &ntokens );

				#if NMFGPU_DEBUG_READ_MATRIX
					print_message( dbg_shown_by_all, "\t\t\tResulting number of tokens (with space): %zu\n", ntokens );
				#endif

				if ( ! tags.tokens ) {
					print_error( error_shown_by_all, "Error in matrix_read_tags().\n" );
					free( (void *) data );
					return EXIT_FAILURE;
				}

			} // If it must "retokenize" the string

			if ( ntokens != (size_t) major_dim ) {
				print_error( error_shown_by_all, "Error reading input file: %zu row labels found, %" PRI_IDX " expected.\n",
						ntokens, major_dim );
				clean_tag( tags );
				return EXIT_FAILURE;
			}

		} else {	// Just a newline was read: (len == 0) && (ntokens == 1)
			clean_tag( tags );
			tags = new_empty_tag();
		}

		// Sets tags as labels or headers according to transpose.
		if ( transpose )
			l_mt.headers = tags;
		else
			l_mt.labels = tags;

	} // row labels

	// --------------------------

	// Checks for column headers
	{
		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( dbg_shown_by_all, "\t\tReading column headers...\n" );
		#endif

		struct tag_t tags = new_empty_tag();

		size_t ntokens = 0;
		size_t len = 0;

		int status = read_tag( file, delimiter, &tags, &len, &ntokens );

		if ( status ) {	// EOF, error, or invalid format
			if ( status >= 2 ) {	// Error or invalid format.
				if ( status == 2 )	// Internal error.
					print_error( error_shown_by_all, "Error in matrix_read_tags( column headers ).\n" );
				else	// 3: Invalid format.
					print_error( error_shown_by_all, "Please remember to set a newline ('\\n') character "
							"between column headers, row labels and the description string.\n\n" );
				clean_matrix_tags( l_mt );
				return EXIT_FAILURE;
			}
			// Else, EOF
			*mt = l_mt;
			return EXIT_SUCCESS;
		}

		// Else, success: one or more row labels were read.

		if ( len ) {	// Non-empty line: (len > 0) && (ntokens >= 1)

			if ( verbose )
				print_message( shown_by_all, "\t\tColumn headers detected.\n" );

			// If there is only one token, and there should be more, "retokenizes" using space as delimiter.
			if ( ( ntokens == 1 ) * ( minor_dim > 1 ) ) {

				#if NMFGPU_DEBUG_READ_MATRIX
					print_message( dbg_shown_by_all, "\t\t\t\"Retokenizes\" using space...\n" );
				#endif

				// Removes previous array of pointers to tokens.
				char *restrict data = (char *restrict) tags.tokens;
				free( (void *) tags.ptokens );
				ntokens = 0;

				tags = tokenize( data, (int) ' ', &ntokens );

				#if NMFGPU_DEBUG_READ_MATRIX
					print_message( dbg_shown_by_all, "\t\t\tResulting number of tokens (with space): %zu\n", ntokens );
				#endif

				if ( ! tags.tokens ) {
					print_error( error_shown_by_all, "Error in matrix_read_tags().\n" );
					free( (void *) data );
					clean_matrix_tags( l_mt );
					return EXIT_FAILURE;
				}

			} // If it must "retokenize" the string

			if ( ntokens != (size_t) minor_dim ) {
				print_error( error_shown_by_all, "Error reading input file: %zu row labels found, %" PRI_IDX " expected.\n",
						ntokens, minor_dim );
				clean_tag( tags );
				clean_matrix_tags( l_mt );
				return EXIT_FAILURE;
			}

		} else {	// Just a newline was read: (len == 0) && (ntokens == 1)
			clean_tag( tags );
			tags = new_empty_tag();
		}

		// Sets tags as labels or headers according to transpose.
		if ( transpose )
			l_mt.labels = tags;
		else
			l_mt.headers = tags;

	} // column headers

	// --------------------------

	// Checks for name.
	{
		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( dbg_shown_by_all, "\t\tReading name...\n" );
		#endif

		char *restrict name = NULL;

		size_t len = read_line( file, &name );

		#if NMFGPU_DEBUG_READ_MATRIX
			print_message( dbg_shown_by_all, "\t\t\tName (len=%zu): '%s'\n", len, name );
		#endif

		if ( len ) {	// Success, error, or invalid format.
			if ( ! name ) {	// Error or invalid format.
				if ( len == 1 )	// Internal error.
					print_error( error_shown_by_all, "Error in matrix_read_tags( name ).\n" );
				clean_matrix_tags( l_mt );
				return EXIT_FAILURE;
			}
			// Else, success
			if ( verbose )
				print_message( shown_by_all, "\t\tName (i.e., description string) detected.\n" );
		}
		// Else, nothing read: just a newline (name != NULL), or EOF (name == NULL).

		l_mt.name = (char const *restrict) name;
	}

	// --------------------------

	*mt = l_mt;

	return EXIT_SUCCESS;

} // matrix_read_tags

////////////////////////////////////////////////

/*
 * Loads a real-type matrix from a "formatted" binary file: double-precision data,
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

	bool const check_errors = true;
	bool const transpose = false;
	bool const verbose = true;

	int status = EXIT_SUCCESS;

	/////////////////////////////////

	// Checks for NULL parameters
	if ( ! ( (uintptr_t) filename * (uintptr_t) matrix * (uintptr_t) nrows * (uintptr_t) ncols * (uintptr_t) pitch * (uintptr_t) mt ) ) {
		int const errnum = EFAULT;
		if ( ! filename ) print_errnum( error_shown_by_all, errnum, "matrix_load_binary_verb( filename )" );
		if ( ! matrix )	print_errnum( error_shown_by_all, errnum, "matrix_load_binary_verb( matrix )" );
		if ( ! nrows )	print_errnum( error_shown_by_all, errnum, "matrix_load_binary_verb( nrows )" );
		if ( ! ncols )	print_errnum( error_shown_by_all, errnum, "matrix_load_binary_verb( ncols )" );
		if ( ! pitch )	print_errnum( error_shown_by_all, errnum, "matrix_load_binary_verb( pitch )" );
		if ( ! mt )	print_errnum( error_shown_by_all, errnum, "matrix_load_binary_verb( mt )" );
		return EXIT_FAILURE;
	}

	// ------------------------------------

	FILE *restrict const file = fopen( filename, "rb" );
	if ( ! file ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_binary_verb(): fopen( %s )", filename );
		return EXIT_FAILURE;
	}

	// Checks file signature (real type)
	if ( read_signature( file, true ) != EXIT_SUCCESS ) {
		fclose(file);
		return EXIT_FAILURE;
	}

	// ------------------------------------

	// Reads matrix dimensions.
	{
		uint32_t dim[2] = { UINT32_C(0), UINT32_C(0) };

		errno = 0;
		size_t const nread = fread( dim, sizeof(uint32_t), 2, file );
		if ( nread != 2 ) {
			if ( feof(file) )
				print_error( error_shown_by_all, "Error reading input file:\nPremature end-of-file detected. "
						"Invalid file format.\n\n");
			else // error
				print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_binary_verb(): fread( dim[2] )" );
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

		// Changes values to index_t
		numrows = (index_t) dim[0];
		numcols = (index_t) dim[1];
		l_pitch = get_padding( numcols );

		// Check matrix dimensions (verbose mode)
		status = matrix_check_dimensions( "input file ", numrows, numcols, l_pitch, transpose, true );

		if ( (status != EXIT_SUCCESS) + (numrows < 2) + (numcols < 2) ) {
			if ( status == EXIT_SUCCESS )	// (numrows < 2) + (numcols < 2)
				print_error( error_shown_by_all,
						"\nError reading input file: both matrix dimensions must be greater than 1.\n\n" );
			fclose(file);
			return EXIT_FAILURE;
		}

	} // Reads matrix dimensions.

	// ------------------------------------

	// Reads matrix data

	status = matrix_read_binary( file, check_errors, numrows, numcols, l_pitch, transpose, matrix );

	if ( status != EXIT_SUCCESS ) {
		fclose(file);
		return EXIT_FAILURE;
	}

	//-------------------------------------

	// Reads labels, headers and name (as plain text) if they exists.

	status = matrix_read_tags( file, numrows, numcols, transpose, verbose, mt );

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
 * Loads a real-type matrix from a "formatted" binary file: double-precision data,
 * and 32-bits unsigned integers for matrix dimensions and the file signature,
 * all of them in little-endian format.
 *
 * Detects automatically if matrix has name, column headers and/or row labels,
 * as well as the employed delimiter symbol (space or tab character). Skips all
 * of them if "mt" is NULL.
 *
 * If (*matrix) is non-NULL, do not allocates memory but uses the supplied one.
 * In such case, THERE MUST BE ENOUGH MEMORY ALREADY ALLOCATED (padding included).
 *
 * If "transpose" is 'true':
 *	- Reads from file:
 *		<ncols> rows and <nrows> columns.
 *		<ncols> row labels (set as mt->headers), and <nrows> column headers (set as mt->labels).
 *	- Writes to "*matrix": <nrows> rows and <ncols> columns (padded to <pitch>).
 *
 * ncols <= pitch
 *
 * Fails if dimensions stored in file mismatch with "nrows" and "ncols".
 *
 * WARNING:
 *	- NO ERROR CHECKING IS PERFORMED for negative data, empty rows/columns, etc.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_load_binary( char const *restrict filename, index_t nrows, index_t ncols, index_t pitch, bool transpose,
			real *restrict *restrict matrix, struct matrix_tags_t *restrict mt )
{

	index_t const major_dim = ( transpose ? ncols : nrows );
	index_t const minor_dim = ( transpose ? nrows : ncols );

	bool const verbose = false;	// (quiet mode)

	int status = EXIT_SUCCESS;

	//////////////////////////////

	// Checks for NULL parameters
	if ( ! ( (uintptr_t) filename * (uintptr_t) matrix ) ) {
		int const errnum = EFAULT;
		if ( ! filename ) print_errnum( error_shown_by_all, errnum, "matrix_load_binary( filename )" );
		if ( ! matrix )	print_errnum( error_shown_by_all, errnum, "matrix_load_binary( matrix )" );
		return EXIT_FAILURE;
	}

	// Checks provided matrix dimensions.
	status = matrix_check_dimensions( "matrix_load_binary", nrows, ncols, pitch, transpose, verbose );
	if ( status != EXIT_SUCCESS )
		return EXIT_FAILURE;

	// ------------------------------------

	#if NMFGPU_DEBUG_READ_MATRIX
		print_message( dbg_shown_by_all, "Reading '%s'...\n", filename );
	#endif

	FILE *restrict const file = fopen( filename, "rb" );
	if ( ! file ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_binary(): fopen( %s )", filename );
		return EXIT_FAILURE;
	}

	// Checks file signature (real type)
	if ( read_signature( file, true ) != EXIT_SUCCESS ) {
		fclose(file);
		return EXIT_FAILURE;
	}

	// ------------------------------------

	// Reads matrix dimensions.
	{
		uint32_t dim[2] = { UINT32_C(0), UINT32_C(0) };

		errno = 0;
		size_t const nread = fread( dim, sizeof(uint32_t), 2, file );
		if ( nread != 2 ) {
			if ( feof(file) )
				print_error( error_shown_by_all, "Error reading input file:\nPremature end-of-file detected. "
						"Invalid file format.\n\n");
			else // error
				print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_binary(): fread( dim[2] )" );
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
			uint_fast64_t const nitems = (uint_fast64_t) dim[0] * (uint_fast64_t) dim[1];
			print_message( dbg_shown_by_all, "\tMatrix Dimensions (file): %" PRIu32 "x%" PRIu32 " (%" PRIuFAST64 " items)\n",
					dim[0], dim[1], nitems );
		}
		#endif

		// Checks file dimensions.
		if ( (major_dim != (index_t) dim[0]) + (minor_dim != (index_t) dim[1]) ) {
			print_error( error_shown_by_all, "Error in matrix_load_binary( transpose=%i ): matrix dimensions mismatch: %"
					PRIu32 " x %" PRIu32 " read, %" PRI_IDX " x %" PRI_IDX " expected.\n",
					transpose, dim[0], dim[1], major_dim, minor_dim );
			fclose(file);
			return EXIT_FAILURE;
		}
	} // Reads matrix dimensions.

	// ------------------------------------

	// Reads matrix data

	bool const check_errors = false;
	bool const memory_allocated = ( ! (*matrix) );	// Memory for the data matrix will be allocated.

	status = matrix_read_binary( file, check_errors, nrows, ncols, pitch, transpose, matrix );

	if ( status != EXIT_SUCCESS ) {
		print_error( error_shown_by_all, "Error in matrix_load_binary()\n" );
		fclose(file);
		return EXIT_FAILURE;
	}

	//-------------------------------------

	// Reads labels, headers and name (as plain text) if they exists.

	if ( mt ) {
		status = matrix_read_tags( file, nrows, ncols, transpose, verbose, mt );
		if ( status != EXIT_SUCCESS ) {
			print_error( error_shown_by_all, "Error in matrix_load_binary()\n" );
			if ( memory_allocated ) {
				free( (void *) *matrix );
				*matrix = NULL;
			}
			fclose(file);
			return EXIT_FAILURE;
		}
	}

	fclose(file);

	return EXIT_SUCCESS;

} // matrix_load_binary

////////////////////////////////////////////////

/*
 * Loads a matrix from a "native" binary file (i.e., with the native endianness,
 * the compiled types for matrix data and dimensions, and no file signature).
 *
 * Detects automatically if matrix has name, column headers and/or row labels,
 * as well as the employed delimiter symbol (space or tab character). Skips all
 * of them if "mt" is NULL.
 *
 * If (*matrix) is non-NULL, do not allocates memory but uses the supplied one.
 * In such case, THERE MUST BE ENOUGH MEMORY ALREADY ALLOCATED (padding included).
 *
 * In verbose mode, shows information messages, such as matrix dimensions and
 * labels found.
 *
 * If *nrows or *ncols are non-zero, they are compared with the matrix dimensions
 * stored in the file, failing if they differ.
 *
 * If *pitch is zero, the number of columns is rounded up to a multiple of
 * <memory_alignment>. Otherwise, the given pitch is used as padded dimension,
 * and must NOT be less than the number of columns.
 *
 * WARNING:
 *	- For internal use only (i.e., for temporary files).
 *	- If *matrix is non-NULL, IT MUST HAVE ENOUGH MEMORY ALREADY ALLOCATED,
 *	  padding included.
 *	- NO ERROR CHECKING IS PERFORMED (e.g., overflow, invalid values,
 *	  negative data, etc).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int matrix_load_binary_native( char const *restrict filename, void *restrict *restrict matrix, index_t *restrict nrows, index_t *restrict ncols,
				index_t *restrict pitch, size_t data_size, bool verbose, struct matrix_tags_t *restrict mt )
{

	index_t numcols = INDEX_C(0), numrows = INDEX_C(0), l_pitch = INDEX_C(0);

	bool const transpose = false;	// No matrix transposing

	int status = EXIT_SUCCESS;

	/////////////////////////////////


	// Checks for NULL parameters.
	if ( ! ( (uintptr_t) filename * (uintptr_t) matrix * (uintptr_t) nrows * (uintptr_t) ncols * (uintptr_t) pitch ) ) {
		int const errnum = EFAULT;
		if ( ! filename ) print_errnum( error_shown_by_all, errnum, "matrix_load_binary_native( filename )" );
		if ( ! matrix ) print_errnum( error_shown_by_all, errnum, "matrix_load_binary_native( matrix )" );
		if ( ! nrows ) print_errnum( error_shown_by_all, errnum, "matrix_load_binary_native( nrows )" );
		if ( ! ncols ) print_errnum( error_shown_by_all, errnum, "matrix_load_binary_native( ncols )" );
		if ( ! pitch )	print_errnum( error_shown_by_all, errnum, "matrix_load_binary_native( pitch )" );
		return EXIT_FAILURE;
	}

	// Checks for invalid parameters
	if ( (! data_size) + (*nrows < 0) + (*ncols < 0) + (*pitch < 0) ) {
		int const errnum = EINVAL;
		if ( ! data_size ) print_errnum( error_shown_by_all, errnum, "matrix_load_binary_native( data_size=%zu )", data_size );
		if ( *nrows < 0 ) print_errnum( error_shown_by_all, errnum, "matrix_load_binary_native( nrows=%" PRI_IDX " )", *nrows );
		if ( *ncols < 0 ) print_errnum( error_shown_by_all, errnum, "matrix_load_binary_native( ncols=%" PRI_IDX " )", *ncols );
		if ( *pitch < 0 ) print_errnum( error_shown_by_all, errnum, "matrix_load_binary_native( pitch=%" PRI_IDX " )", *pitch );
		return EXIT_FAILURE;
	}

	// ------------------------------------

	#if NMFGPU_DEBUG_READ_MATRIX
		print_message( dbg_shown_by_all, "Reading '%s'...\n", filename );
	#endif

	FILE *restrict const file = fopen( filename, "rb" );
	if ( ! file ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_binary_native(): fopen( %s )", filename );
		return EXIT_FAILURE;
	}

	// ------------------------------------

	// Reads matrix dimensions.
	{
		index_t dim[2] = { INDEX_C(0), INDEX_C(0) };

		errno = 0;
		size_t const nread = fread( dim, sizeof(index_t), 2, file );
		if ( nread != 2 ) {
			if ( feof(file) )
				print_error( error_shown_by_all, "Error reading input file:\nPremature end-of-file detected. "
						"Invalid file format.\n\n");
			else // error
				print_errnum( sys_error_shown_by_all, errno, "Error in matrix_load_binary_native(): fread( dim[2] )" );
			fclose(file);
			return EXIT_FAILURE;
		}

		// Changes values to index_t
		numrows = dim[0];
		numcols = dim[1];
		l_pitch = ( *pitch ? *pitch : get_padding( numcols ) );

		status = matrix_check_dimensions( "matrix_load_binary_native", numrows, numcols, l_pitch, transpose, verbose );

		if ( status != EXIT_SUCCESS ) {
			fclose(file);
			return EXIT_FAILURE;
		}

		// ------------------------------------

		// Checks matrix dimensions, if provided.

		if ( (*nrows * (*nrows != numrows)) + (*ncols * (*ncols != numcols)) ) {
			print_error( error_shown_by_all, "Error in matrix_load_binary_native(): matrix dimensions mismatch:\n" );
			if ( *nrows * (*nrows != numrows) )
				append_printed_error ( error_shown_by_all, "\t*nrows(argument)=%" PRI_IDX ", nrows(file)=%" PRI_IDX "\n",
						*nrows, numrows );
			if ( *ncols * (*ncols != numcols) )
				append_printed_error ( error_shown_by_all, "\t*ncols(argument)=%" PRI_IDX ", ncols(file)=%" PRI_IDX "\n",
						*ncols, numcols );
			fclose(file);
			return EXIT_FAILURE;
		}

	} // Reads matrix dimensions

	// ------------------------------------

	// Reads matrix data

	bool const memory_allocated = ( ! *matrix );	// If memory for data matrix will be allocated.
	status = matrix_read_binary_native( file, numrows, numcols, l_pitch, data_size, matrix );

	if ( status != EXIT_SUCCESS ) {
		print_error( error_shown_by_all, "Error in matrix_load_binary_native()\n" );
		fclose(file);
		return EXIT_FAILURE;
	}

	//-------------------------------------

	// Reads labels, headers and name (as plain text) if requested.

	if ( mt ) {
		status = matrix_read_tags( file, numrows, numcols, transpose, verbose, mt );
		if ( status != EXIT_SUCCESS ) {
			print_error( error_shown_by_all, "Error in matrix_load_binary_native()\n" );
			if ( memory_allocated ) {
				free( (void *) *matrix );
				*matrix = NULL;
			}
			fclose( file );
			return EXIT_FAILURE;
		}
	}

	fclose( file );

	//-------------------------------------

	// Sets output values

	if ( ! *nrows ) *nrows = numrows;
	if ( ! *ncols ) *ncols = numcols;
	if ( ! *pitch ) *pitch = l_pitch;

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
		int const errnum = EFAULT;
		if ( ! filename ) print_errnum( error_shown_by_all, errnum, "matrix_load( filename )" );
		if ( ! matrix )	print_errnum( error_shown_by_all, errnum, "matrix_load( matrix )" );
		if ( ! nrows )	print_errnum( error_shown_by_all, errnum, "matrix_load( nrows )" );
		if ( ! ncols )	print_errnum( error_shown_by_all, errnum, "matrix_load( ncols )" );
		if ( ! pitch )	print_errnum( error_shown_by_all, errnum, "matrix_load( pitch )" );
		if ( ! mt )	print_errnum( error_shown_by_all, errnum, "matrix_load( mt )" );
		return EXIT_FAILURE;
	}

	if ( is_bin < 0 ) {
		print_errnum( error_shown_by_all, EINVAL, "Error in matrix_load( is_bin=%" PRI_IDX " )", is_bin );
		return EXIT_FAILURE;
	}

	// Initializes matrix dimensions
	*pitch = *ncols = *nrows = 0;

	int status = EXIT_SUCCESS;

	// -------------------------------

	// Loads the file.

	print_message( shown_by_all, "Loading input file...\n" );

	if ( is_bin > 1 ) { // Input file is "native" binary.

		append_printed_message( shown_by_all, "\tFile selected as \"native\" binary (i.e., the file is read using the data "
					"types specified at compilation).\n\tNo error-checking is performed.\n\tLoading...\n" );

		bool const verbose = true;
		size_t const data_size = sizeof(real);

		status = matrix_load_binary_native( filename, (void *restrict *restrict) matrix, nrows, ncols, pitch, data_size, verbose, mt );
	}

	// Input file is "non-native" binary.
	else if ( is_bin ) {

		append_printed_message( shown_by_all, "\tFile selected as (non-\"native\") binary (i.e., double-precision data and "
					"unsigned integers).\n\tLoading...\n" );
		status = matrix_load_binary_verb( filename, matrix, nrows, ncols, pitch, mt );
	}

	// Input file is ASCII-text.
	else {

		append_printed_message( shown_by_all, "\tFile selected as ASCII text. Loading...\n"
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
 * Saves data matrix to an ASCII-text file.
 * Skips name, headers and labels if "mt" is NULL.
 *
 * If "transpose" is 'true':
 * - Reads from "matrix": <nrows> rows and <ncols> columns (padded to <pitch>).
 * - Writes to file:
 *	<ncols> rows and <nrows> columns.
 *	<ncols> row labels from mt->headers, and <nrows> column headers from mt->labels.
 *
 * ncols <= pitch.
 *
 * Set "append" to 'true' to append data to the file.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_save_ascii( char const *restrict filename, void const *restrict matrix, index_t nrows, index_t ncols, index_t pitch, bool real_data,
			bool transpose, bool append, struct matrix_tags_t const *restrict mt )
{

	int status = EXIT_SUCCESS;

	int const delimiter = (int) '\t';

	index_t const major_dim = ( transpose ? ncols : nrows );
	index_t const minor_dim = ( transpose ? nrows : ncols );

	char const *restrict name = NULL;
	struct tag_t labels = new_empty_tag();
	struct tag_t headers = new_empty_tag();

	bool hasname = false;
	bool hasheaders = false;
	bool haslabels = false;

	////////////////////////////////

	// Checks for NULL parameters
	if ( ! ( (uintptr_t) filename * (uintptr_t) matrix ) ) {
		int const errnum = EFAULT;
		if ( ! filename ) print_errnum( error_shown_by_all, errnum, "matrix_save_ascii( filename )" );
		if ( ! matrix )	print_errnum( error_shown_by_all, errnum, "matrix_save_ascii( matrix )" );
		return EXIT_FAILURE;
	}

	// Checks matrix dimensions (quiet mode)
	status = matrix_check_dimensions( "matrix_save_ascii", nrows, ncols, pitch, transpose, false );
	if ( status != EXIT_SUCCESS )
		return EXIT_FAILURE;

	// -----------------------------

	// File mode: Creates a new text file, <OR>, appends data to an existing one.
	char const mode = ( append ? 'a' : 'w' );

	FILE *restrict const file = fopen( filename, &mode );
	if ( ! file ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_ascii(): fopen( %s, mode='%c' )", filename, mode );
		return EXIT_FAILURE;
	}

	// -----------------------------

	// Writes tag elements.

	if ( mt ) {

		name = mt->name;
		headers = ( transpose ? mt->labels  : mt->headers );
		labels  = ( transpose ? mt->headers : mt->labels  );

		hasname = name;
		hasheaders = headers.tokens;
		haslabels  = labels.tokens;

		// Name
		if ( hasname ) {
			struct tag_t const tag_name = new_tag( (char *restrict)name, (char **restrict)&name );	// Fakes a tag_t struct
			bool const prefix = false;
			bool const suffix = ( ! hasheaders );

			status = write_tag( file, tag_name, "name", 1, delimiter, prefix, suffix );
			if ( status != EXIT_SUCCESS ) {
				print_error( error_shown_by_all, "Error in matrix_save_ascii( transpose=%i )\n", transpose );
				fclose(file);
				return EXIT_FAILURE;
			}
		}

		// Column headers
		if ( hasheaders ) {
			bool const prefix = hasname;
			bool const suffix = true;

			status = write_tag( file, headers, "headers", minor_dim, delimiter, prefix, suffix );
			if ( status != EXIT_SUCCESS ) {
				print_error( error_shown_by_all, "Error in matrix_save_ascii( transpose=%i )\n", transpose );
				fclose(file);
				return EXIT_FAILURE;
			}
		}

	} // mt != NULL


	// ----------------------------

	// Writing data...

	// Step sizes for outer and inner loops.
	index_t incr_outer_loop = ( transpose ? 1 : pitch );
	index_t incr_inner_loop = ( transpose ? pitch : 1 );
	errno = 0;

	size_t const data_size = ( real_data ? sizeof(real) : sizeof(index_t) );

	void const *pmatrix = matrix;
	for ( index_t i = 0 ; i < major_dim ; i++, pmatrix += (incr_outer_loop * data_size) ) {

		// Writes label.
		if ( haslabels ) {
			if ( fprintf( file, "%s", labels.ptokens[i] ) < 0 ) {	// < 0: *ptokens[i] might be NULL.
				print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_ascii( transpose=%i ): "
						"fprintf(plabels, item %" PRI_IDX "/%" PRI_IDX ")", transpose, i, major_dim );
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
				print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_ascii( transpose=%i ): fprintf( major "
						"dim %" PRI_IDX "/%" PRI_IDX ", minor dim 0/%" PRI_IDX " )", transpose, i, major_dim, minor_dim );
				fclose(file);
				return EXIT_FAILURE;
			}
		} // if haslabels

		void const *p = pmatrix + ( (! haslabels) * incr_inner_loop * data_size );
		for ( index_t j = (! haslabels) ; j < minor_dim ; j++, p += (incr_inner_loop * data_size) ) {

			data_t val;
			int conv;

			memcpy( &val, p, data_size );

			if ( real_data )
				conv = fprintf( file, "%c%g", delimiter, val.r );
			else
				conv = fprintf( file, "%c%" PRI_IDX, delimiter, val.i );

			if ( conv <= 0 ) {
				print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_ascii( transpose=%i ): fprintf( "
						"major dim %" PRI_IDX "/%" PRI_IDX ", minor dim %" PRI_IDX "/%" PRI_IDX " )", transpose, i,
						major_dim, j, minor_dim );
				fclose(file);
				return EXIT_FAILURE;
			}
		}

		if ( fprintf( file, "\n" ) <= 0 ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_ascii( transpose=%i ): "
					"fprintf('\\n', major dim %" PRI_IDX "/%" PRI_IDX ")", transpose, i, major_dim );
			fclose(file);
			return EXIT_FAILURE;
		}

	} // for ( 0 <= i < nrows )

	if ( fclose(file) ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_ascii(): fclose( %s, opened in mode: '%c' )",
				filename, mode );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // matrix_save_ascii

////////////////////////////////////////////////
////////////////////////////////////////////////

/* Helper function for matrix_save_combined_ascii()
 *
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
	if ( (! prefix) && (fprintf( file, "%g", pmatrix[0] ) <= 0) ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in matrix_write_line(): fprintf( item 0 of %" PRI_IDX " )", nitems );
		return EXIT_FAILURE;
	}

	for ( index_t j = (! prefix) ; j < nitems ; j++ ) {
		int const conv = fprintf( file, "%c%g", delimiter, pmatrix[ j ] );
		if ( conv < 2 ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_write_line(): fprintf( item %" PRI_IDX " of %"
					PRI_IDX " )", j, nitems );
			return EXIT_FAILURE;
		}
	}

	return EXIT_SUCCESS;

} // matrix_write_line

// ---------------------------------------------

/*
 * Saves <nmatrices> <nrows>-by-<ncols> real-type matrices to a single ASCII-text file.
 *
 * Reads input matrices from "native"-binary files (i.e., with the compiled types for matrix data and dimensions).
 *
 * Uses the supplied labels (unless "mt" is NULL).
 *
 * WARNING:
 *	- nmatrices > 1
 *	- The resulting matrix must not overflow the limits for matrix dimensions.
 *	  That is,
 *		(nmatrices * ncols) <= matrix_max_pitch
 *		(nmatrices * (nrows * ncols)) <= matrix_max_num_items
 *		nrows <= matrix_max_non_padded_dim
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_save_combined_ascii( char const *restrict filename, char const *restrict input_pattern, char const *restrict output_pattern,
				index_t nmatrices, index_t nrows, index_t ncols, struct matrix_tags_t const *restrict mt )
{

	// Dimension limits for the resulting matrix.
	size_t const max_num_items = matrix_max_num_items;
	index_t const max_pitch = matrix_max_pitch;
	index_t const max_non_padded_dim = matrix_max_non_padded_dim;

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
		if ( ! filename )	print_errnum( error_shown_by_all, errnum, "matrix_save_combined_ascii( filename )" );
		if ( ! input_pattern )	print_errnum( error_shown_by_all, errnum, "matrix_save_combined_ascii( input_pattern )" );
		if ( ! output_pattern )	print_errnum( error_shown_by_all, errnum, "matrix_save_combined_ascii( output_pattern )" );
		return EXIT_FAILURE;
	}

	// Checks other parameters
	{
		uintmax_t const width = ((uintmax_t) nmatrices) * ((uintmax_t) ncols);
		uintmax_t const nitems = width * ((uintmax_t) nrows);

		if ( (nmatrices <= 1) + (nrows <= 0) + (ncols <= 0) + (width > (uintmax_t) max_pitch) + (nrows > max_non_padded_dim) +
			(nitems > (uintmax_t) max_num_items) ) {
			if ( (nmatrices <= 1) + (nrows <= 0) + (ncols <= 0) )
				print_error( error_shown_by_all, "Error in matrix_save_combined_ascii( nmatrices=%" PRI_IDX ", rows=%" PRI_IDX
						", columns=%" PRI_IDX " ): Invalid parameters.\n", nmatrices, nrows, ncols );
			else
				print_error( error_shown_by_all, "Sorry, but the output matrix exceeds the limits for matrix dimensions.\n"
						"On this system and with the given input arguments, all data matrices are limited to:\n"
						"\t* %" PRI_IDX " rows.\n\t*%" PRI_IDX " columns.\n\t*%zu items.\n",
						max_non_padded_dim, max_pitch, max_num_items );
			return EXIT_FAILURE;
		}
	}

	// --------------------------

	// List of input files

	FILE *restrict *restrict const input_files = (FILE *restrict *restrict) malloc( nmatrices * sizeof(FILE *) );
	if ( ! input_files ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_combined_ascii(): malloc( input_files[], size=%"
				PRI_IDX " )", nmatrices );
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
			print_errnum( sys_error_shown_by_all, errno,
					"Error in matrix_save_combined_ascii(): malloc( filename_tmp, size=%zu )", str_len );
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
				append_printed_error( error_shown_by_all, "The resulting string was truncated; %i bytes are required at least.\n",
						status + 1 );
			for ( index_t i = 0 ; i < mt ; i++ ) fclose(input_files[i]);
			free((void *)filename_tmp); free((void *)input_files);
			return EXIT_FAILURE;
		}

		input_files[mt] = (FILE *restrict) fopen( filename_tmp, "rb" );	// Opens for reading in binary mode.
		if( ! input_files[mt] ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_combined_ascii(): fopen( input file '%s' )",
					filename_tmp );
			for ( index_t i = 0 ; i < mt ; i++ ) fclose(input_files[i]);
			free((void *)filename_tmp); free((void *)input_files);
			return EXIT_FAILURE;
		}

		// NOTE: There is no signature on "native"-binary files.

		// Checks matrix dimensions.
		index_t dim[2] = { 0, 0 };

		errno = 0;
		size_t const nread = fread( dim, sizeof(index_t), 2, input_files[mt] );
		if ( (nread != 2) + (dim[0] != nrows) + (dim[1] != ncols) ) {
			if ( ferror( input_files[mt] ) )
				print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_combined_ascii(): fread( dim[2], ncols=%"
						PRI_IDX ", mt=%" PRI_IDX ", nmatrices=%" PRI_IDX " )" );
			else if ( feof( input_files[mt] ) )
				print_error( error_shown_by_all, "Error in matrix_save_combined_ascii() reading dimensions in file %" PRI_IDX
						" of " PRI_IDX ": Premature end of file detected.\nInvalid file format.\n", mt, nmatrices );
			else	// (dim[0] != nrows) || (dim[1] != ncols)
				print_error( error_shown_by_all, "Error in matrix_save_combined_ascii() reading dimensions in file %" PRI_IDX
						" of " PRI_IDX ": Invalid input matrix dimensions.\n%" PRI_IDX " x %" PRI_IDX " read, %" PRI_IDX
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
			append_printed_error( error_shown_by_all, "The resulting string was truncated; %i bytes are required at least.\n",
					status + 1 );
		for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
		free((void *)filename_tmp); free((void *)input_files);
		return EXIT_FAILURE;
	}

	FILE *restrict const out_file = fopen( filename_tmp, "w" );
	if( ! out_file ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_combined_ascii(): fopen( output file '%s' )", filename_tmp );
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

		hasheaders = headers.tokens;
		haslabels = headers.tokens;

		// Name
		if ( name ) {
			struct tag_t const tag_name = new_tag( (char *restrict)name, (char **restrict)&name );	// Fakes a tag_t struct

			// No prefix; suffix only if no headers.
			status = write_tag( out_file, tag_name, "name", 1, delimiter, false, (! hasheaders) );
			if ( status != EXIT_SUCCESS ) {
				print_error( error_shown_by_all, "Error in matrix_save_combined_ascii().\n" );
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
				print_error( error_shown_by_all, "Error in matrix_save_combined_ascii() writing column headers (0 of %"
						PRI_IDX " matrices)\n", nmatrices );
				fclose(out_file);
				for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
				free((void *)input_files);
				return EXIT_FAILURE;
			}

			// Rest of headers: All prefixed; no suffix
			for ( index_t mt = 1 ; mt < nmatrices ; mt++ ) {
				status = write_tag( out_file, headers, "headers", ncols, delimiter, true, false );
				if ( status != EXIT_SUCCESS ) {
					print_error( error_shown_by_all, "Error in matrix_save_combined_ascii() writing column headers (%"
							PRI_IDX " of %" PRI_IDX " matrices)\n", mt, nmatrices );
					fclose(out_file);
					for ( index_t i = 0 ; i < nmatrices ; i++ ) fclose(input_files[i]);
					free((void *)input_files);
					return EXIT_FAILURE;
				}
			}

			errno = 0;
			if ( fprintf( out_file, "\n" ) <= 0 ) {
				print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_combined_ascii(): fprintf('\\n') "
						"after headers" );
				fclose(out_file);
				for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
				free((void *)input_files);
				return EXIT_FAILURE;
			}

		} // if hasheaders

	} // if mt != NULL

	// ----------------------------

	// Allocates memory for a single data row.
	real *restrict const data = (real *restrict) malloc( ncols * sizeof(real) );
	if ( ! data ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_combined_ascii(): malloc(data, size=ncols=%"
				PRI_IDX ")", ncols );
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
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_combined_ascii(): fprintf(plabels[%"
					PRI_IDX "])", i );
			free(data); fclose(out_file);
			for ( index_t mt = 0 ; mt < nmatrices ; mt++ ) fclose(input_files[mt]);
			free((void *)input_files);
			return EXIT_FAILURE;
		}

		// Writes row i from input file 0.
		{
			// Reads the entire row <i> from input file 0.
			errno = 0;
			size_t const nread = fread( data, sizeof(real), ncols, input_files[ 0 ] );
			if ( nread != (size_t) ncols ) {
				if ( ferror( input_files[ 0 ] ) )
					print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_combined_ascii(): fread( row=%"
							PRI_IDX ", file 0 of %" PRI_IDX ", ncols=%" PRI_IDX " )", i, nmatrices, ncols );
				else
					print_error( error_shown_by_all, "Error in matrix_save_combined_ascii() reading row %" PRI_IDX
							" from file 0 of %" PRI_IDX " (ncols=%" PRI_IDX ").\nPremature end of file.\n", i,
							nmatrices, ncols );
				free(data); fclose(out_file);
				for ( index_t j = 0 ; j < nmatrices ; j++ ) fclose(input_files[j]);
				free((void *)input_files);
				return EXIT_FAILURE;
			}

			// Writes that row: prefixes only if there are labels.
			status = matrix_write_line( out_file, data, ncols, delimiter, haslabels );
			if ( status != EXIT_SUCCESS ) {
				print_error( error_shown_by_all, "Error in matrix_save_combined_ascii() writing row %" PRI_IDX
						" from file 0 of %" PRI_IDX " (ncols=%" PRI_IDX ").\n", i, nmatrices, ncols );
				free(data); fclose(out_file);
				for ( index_t j = 0 ; j < nmatrices ; j++ ) fclose(input_files[j]);
				free((void *)input_files);
				return EXIT_FAILURE;
			}
		} // input file 0

		// Writes row i from the rest of input files
		for ( index_t mt = 1 ; mt < nmatrices ; mt++ ) {

			// Reads the entire row <i> from input file <mt>.
			errno = 0;
			size_t const nread = fread( data, sizeof(real), ncols, input_files[ mt ] );
			if ( nread != (size_t) ncols ) {
				if ( ferror( input_files[ mt ] ) )
					print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_combined_ascii(): fread( row=%"
							PRI_IDX ", file %" PRI_IDX " of %" PRI_IDX ", ncols=%" PRI_IDX " )",
							i, mt, nmatrices, ncols );
				else
					print_error( error_shown_by_all, "Error in matrix_save_combined_ascii() reading row %" PRI_IDX
							" from file %" PRI_IDX " of %" PRI_IDX " (ncols=%" PRI_IDX
							").\nPremature end of file.\n", i, mt,
							nmatrices, ncols );
				free(data); fclose(out_file);
				for ( index_t j = 0 ; j < nmatrices ; j++ ) fclose(input_files[j]);
				free((void *)input_files);
				return EXIT_FAILURE;
			}

			// Writes that row: prefixes always.
			status = matrix_write_line( out_file, data, ncols, delimiter, true );
			if ( status != EXIT_SUCCESS ) {
				print_error( error_shown_by_all, "Error in matrix_save_combined_ascii() writing row %" PRI_IDX " from file %"
						PRI_IDX " of %" PRI_IDX " (ncols=%" PRI_IDX ").\n", i, mt, nmatrices, ncols );
				free(data); fclose(out_file);
				for ( index_t j = 0 ; j < nmatrices ; j++ ) fclose(input_files[j]);
				free((void *)input_files);
				return EXIT_FAILURE;
			}

		} // for each input file.

		errno = 0;
		if ( fprintf( out_file, "\n" ) <= 0 ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_combined_ascii(): fprintf('\\n') after row %"
					PRI_IDX, i );
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
		print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_combined_ascii(): fclose( output file )" );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // matrix_save_combined_ascii

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Writes the signature to a "formatted" binary file: a 32-bits unsigned integer
 * in little-endian format.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int write_signature( FILE *restrict file, bool real_data )
{

	uint32_t file_signature = ( real_data ? BIN_FILE_SIGNATURE_REAL : BIN_FILE_SIGNATURE_INDEX );

	// -----------------------------

	// If this system is big-endian, reverses the byte order.
	if ( IS_BIG_ENDIAN() ) {
		uint32_t value = UINT32_C( 0 );
		reverse_bytes( &file_signature, sizeof(uint32_t), &value );
		file_signature = value;
	}

	// -----------------------------

	errno = 0;
	if ( ! fwrite( &file_signature, sizeof(uint32_t), 1, file ) ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in write_signature(): fwrite()" );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // write_signature

// ---------------------------------------------

/*
 * Writes labels, headers and name (as plain text).
 *
 * If "transpose" is 'true', writes to file:
 *	<ncols> row labels from mt.headers.
 *	<nrows> column headers from mt.labels
 *
 * NOTE: This function is intended for writing tag elements to BINARY files.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int matrix_write_tags( FILE *restrict file, index_t nrows, index_t ncols, bool transpose, struct matrix_tags_t mt, int delimiter )
{

	index_t const major_dim = ( transpose ? ncols : nrows );
	index_t const minor_dim = ( transpose ? nrows : ncols );

	char const *restrict name = mt.name;
	const struct tag_t headers = ( transpose ? mt.labels  : mt.headers );
	const struct tag_t labels  = ( transpose ? mt.headers : mt.labels  );

	bool const hasheaders = headers.tokens;
	bool const hasname  = name;

	bool const prefix = false;	// No prefix

	// -----------------------------

	// Row labels
	{
		bool const suffix = ( hasheaders + hasname );	// Writes suffix only if there are other tags to write.

		int const status = write_tag( file, labels, "labels", major_dim, delimiter, prefix, suffix );
		if ( status != EXIT_SUCCESS ) {
			print_error( error_shown_by_all, "Error in matrix_write_tags( transpose=%i )\n", transpose );
			return EXIT_FAILURE;
		}
	}

	// -----------------------------

	// Column headers
	{
		bool const suffix = hasname;	// Writes suffix only if there are other tags to write.

		int const status = write_tag( file, headers, "headers", minor_dim, delimiter, prefix, suffix );
		if ( status != EXIT_SUCCESS ) {
			print_error( error_shown_by_all, "Error in matrix_write_tags( transpose=%i )\n", transpose );
			return EXIT_FAILURE;
		}
	}

	// -----------------------------

	// Name
	{
		struct tag_t const tag_name = new_tag( (char *restrict)name, (char **restrict)&name );	// Fakes a tag_t struct

		bool const suffix = hasname;	// Writes suffix only if there is a "name" tag.

		int const status = write_tag( file, tag_name, "name", 1, delimiter, prefix, suffix );
		if ( status != EXIT_SUCCESS ) {
			print_error( error_shown_by_all, "Error in matrix_write_tags( transpose=%i )\n", transpose );
			return EXIT_FAILURE;
		}
	}

	return EXIT_SUCCESS;

} // matrix_write_tags

// ---------------------------------------------

/*
 * Saves a real-type matrix from a "formatted" binary file: double-precision data,
 * and 32-bits unsigned integers for matrix dimensions and the file signature,
 * all of them in little-endian format.
 *
 * Skips name, headers and labels if "mt" is NULL.
 *
 * If "transpose" is 'true':
 * - Reads from "matrix": <nrows> rows and <ncols> columns (padded to <pitch>).
 * - Writes to file:
 *	<ncols> rows and <nrows> columns.
 *	<ncols> row labels from mt->headers, and <nrows> column headers from mt->labels.
 *
 * ncols <= pitch.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_save_binary( char const *restrict filename, real const *restrict matrix, index_t nrows, index_t ncols, index_t pitch, bool transpose,
			struct matrix_tags_t const *restrict mt )
{

	int const delimiter = (int) '\t';

	index_t const major_dim = ( transpose ? ncols : nrows );
	index_t const minor_dim = ( transpose ? nrows : ncols );

	////////////////////////////////

	// Checks for NULL parameters
	if ( ! ( (uintptr_t) filename * (uintptr_t) matrix ) ) {
		int const errnum = EFAULT;
		if ( ! filename ) print_errnum( error_shown_by_all, errnum, "matrix_save_binary( filename )" );
		if ( ! matrix )	print_errnum( error_shown_by_all, errnum, "matrix_save_binary( matrix )" );
		return EXIT_FAILURE;
	}

	// Checks matrix dimensions (quiet mode)
	if ( matrix_check_dimensions( "matrix_save_binary", nrows, ncols, pitch, transpose, false ) != EXIT_SUCCESS )
		return EXIT_FAILURE;

	// -----------------------------

	FILE *restrict const file = fopen( filename, "wb" );
	if ( ! file ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_binary(): fopen( %s )", filename );
		return EXIT_FAILURE;
	}

	// ----------------------------------

	// Writes file signature (real type)
	if ( write_signature( file, true ) != EXIT_SUCCESS ) {
		fclose(file);
		return EXIT_FAILURE;
	}

	// ------------------------------------

	// Writes matrix dimensions.
	{
		uint32_t dim[2] = { major_dim, minor_dim };

		// Changes to little-endian, if necessary.
		if ( IS_BIG_ENDIAN() ) {
			uint32_t value = UINT32_C( 0 );
			reverse_bytes( &dim[0], sizeof(uint32_t), &value );
			dim[0] = value;
			value = UINT32_C( 0 );
			reverse_bytes( &dim[1], sizeof(uint32_t), &value );
			dim[1] = value;
		}

		errno = 0;
		size_t const nwritten = fwrite( dim, sizeof(uint32_t), 2, file );
		if ( nwritten != 2 ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_binary(): fwrite( dim[2] )" );
			fclose(file);
			return EXIT_FAILURE;
		}

	} // Writes matrix dimensions.

	// ----------------------------------

	// Writes all matrix data

	// Step sizes for outer and inner loops.
	index_t incr_outer_loop = ( transpose ? 1 : pitch );
	index_t incr_inner_loop = ( transpose ? pitch : 1 );

	real const *pmatrix = matrix;
	for ( index_t r=0 ; r < major_dim ; r++, pmatrix += incr_outer_loop ) {

		real const *pmatrix_r = pmatrix;

		for ( index_t c=0 ; c < minor_dim ; c++, pmatrix_r += incr_inner_loop ) {

			// Writes one double-precision data
			double value = (double) *pmatrix_r;

			// If this system is big-endian, reverses the byte order.
			if ( IS_BIG_ENDIAN() ) {
				double le_value = 0.0;	// Little-Endian value
				reverse_bytes( &value, sizeof(double), &le_value );
				value = le_value;
			}

			errno = 0;
			size_t const nwritten = fwrite( &value, sizeof(double), 1, file );
			if ( ! nwritten ) {
				print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_binary( transpose=%i ): "
						"fwrite( major dim %" PRI_IDX "/%" PRI_IDX ", minor dim %" PRI_IDX "/%" PRI_IDX " )",
						transpose, r, major_dim, c, minor_dim );
				fclose(file);
				return EXIT_FAILURE;
			}

		} // for c

	} // for r

	// ----------------------------------

	// Writes matrix labels, if any

	if ( mt && (matrix_write_tags( file, nrows, ncols, transpose, *mt, delimiter ) != EXIT_SUCCESS) ) {
		fclose(file);
		return EXIT_FAILURE;
	}

	if ( fclose(file) ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_binary(): fclose( %s )", filename );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // matrix_save_binary

////////////////////////////////////////////////

/*
 * Saves a matrix to a "native" binary file (i.e., with the native endiannes,
 * and the compiled types for matrix data and dimensions; no file signature).
 *
 * Skips name, headers and labels if "mt" is NULL.
 *
 * WARNING:
 *	- For internal use only (i.e., for temporary files).
 *	- NO ERROR-CHECKING IS PERFORMED (e.g., overflow, invalid values, etc).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_save_binary_native( char const *restrict filename, void const *restrict matrix, index_t nrows, index_t ncols, index_t pitch,
				size_t data_size, struct matrix_tags_t const *restrict mt )
{

	int const delimiter = '\t';

	bool const transpose = false;	// No matrix transposing

	////////////////////////////////

	// Checks for NULL parameters
	if ( ! ( (uintptr_t) filename * (uintptr_t) matrix * (uintptr_t) data_size ) ) {
		int const errnum = EFAULT;
		if ( ! filename ) print_errnum( error_shown_by_all, errnum, "matrix_save_binary_native( filename )" );
		if ( ! matrix )	print_errnum( error_shown_by_all, errnum, "matrix_save_binary_native( matrix )" );
		if ( ! data_size ) print_errnum( error_shown_by_all, EINVAL, "matrix_save_binary_native( data_size )" );
		return EXIT_FAILURE;
	}

	// Checks matrix dimensions (quiet mode).
	if ( matrix_check_dimensions( "matrix_save_binary_native", nrows, ncols, pitch, transpose, false ) != EXIT_SUCCESS )
		return EXIT_FAILURE;

	// -----------------------------

	FILE *restrict const file = fopen( filename, "wb" );
	if ( ! file ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_binary_native(): fopen( %s )", filename );
		return EXIT_FAILURE;
	}

	// ----------------------------------

	// Writes matrix dimensions.
	{
		errno = 0;
		index_t const dims[2] = { nrows, ncols };
		size_t const nwritten = fwrite( dims, sizeof(index_t), 2, file );
		if ( nwritten != 2 ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_binary_native(): fwrite( dim, size=2 )" );
			fclose(file);
			return EXIT_FAILURE;
		}
	}

	// ----------------------------------

	#if NMFGPU_DEBUG_WRITE_MATRIX
		print_message( dbg_shown_by_all, "\tWriting data matrix...\n" );
	#endif

	// Writes data row by row in order to skip the padding zone.

	void const *pmatrix = matrix;
	for ( index_t r=0 ; r < nrows ; r++, pmatrix += (pitch * data_size) ) {

		size_t const nwritten = fwrite( pmatrix, data_size, ncols, file );

		if ( nwritten != (size_t) ncols ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_binary_native(): fwrite( row=%" PRI_IDX
					", ncols=%" PRI_IDX ")", r, ncols );
			fclose(file);
			return EXIT_FAILURE;
		}

	}

	// ----------------------------------

	#if NMFGPU_DEBUG_WRITE_MATRIX
		print_message( dbg_shown_by_all, "\tWriting matrix tags...\n" );
	#endif

	// Writes matrix labels, if any

	if ( mt && (matrix_write_tags( file, nrows, ncols, transpose, *mt, delimiter ) != EXIT_SUCCESS) ) {
		fclose(file);
		return EXIT_FAILURE;
	}

	// ----------------------------------

	#if NMFGPU_DEBUG_WRITE_MATRIX
		print_message( dbg_shown_by_all, "\tDone.\n" );
	#endif

	if ( fclose(file) ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in matrix_save_binary_native(): fclose( %s )", filename );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // matrix_save_binary_native

////////////////////////////////////////////////

/*
 * Writes matrix to a file according to the selected file format.
 * Skips name, headers and labels if "mt" is NULL.
 *
 * save_bin: Saves output matrix to a binary file.
 *		== 0: Disabled. Saves the file as ASCII text.
 *		== 1: Uses "non-native" format (i.e., double-precision data, and "unsigned int" for dimensions).
 *		 > 1: Uses "native" or raw format (i.e., the compiled types for matrix data and dimensions).
 *
 * If "transpose" is 'true':
 * - Reads from "matrix": <nrows> rows and <ncols> columns (padded to <pitch>).
 * - Writes to file:
 *	<ncols> rows and <nrows> columns.
 *	<ncols> row labels from mt->headers, and <nrows> column headers from mt->labels.
 *
 * ncols <= pitch.
 *
 * If verbose is 'true', shows some information messages (e.g., file format).
 *
 * WARNING:
 *	"Native" mode (i.e., save_bin > 1) skips ALL data transformation (matrix transposing, padding, etc).
 *	All related arguments are ignored, and the file is saved in raw format.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_save( char const *restrict filename, index_t save_bin, real const *restrict matrix, index_t nrows, index_t ncols, index_t pitch,
		bool transpose, struct matrix_tags_t const *restrict mt, bool verbose )
{

	if ( save_bin < 0 ) {
		print_errnum( error_shown_by_all, EINVAL, "Error in matrix_save( save_bin=%" PRI_IDX " )", save_bin );
		return EXIT_FAILURE;
	}

	int status = EXIT_SUCCESS;

	// -------------------------------

	if ( verbose )
		print_message( shown_by_all, "\nSaving output file...\n" );

	// Saves output as "native" binary.
	if ( save_bin > 1 ) {
		if ( verbose ) {
			append_printed_message( shown_by_all, "\tFile selected as \"native\" binary (i.e., the file is written using the "
						"data types specified at compilation).\n\tNo error-checking is performed.\n" );
			if ( transpose )
				append_printed_message( shown_by_all, "\tSkipping all transformation options (e.g., matrix transposing)...\n" );
		}
		status = matrix_save_binary_native( filename, (void const *restrict) matrix, nrows, ncols, pitch, sizeof(real), mt );
	}

	// Saves output as (non-"native") binary.
	else if ( save_bin ) {
		if ( verbose )
			append_printed_message( shown_by_all, "\tFile selected as (non-\"native\") binary (i.e., double-precision data and "
						"32-bits unsigned integers matrix dimensions).\n");

		status = matrix_save_binary( filename, matrix, nrows, ncols, pitch, transpose, mt );
	}

	// Saves output as ASCII text.
	else {
		if ( verbose )
			append_printed_message( shown_by_all, "\tFile selected as ASCII text.\n" );

		bool const append = false;
		bool const real_data = true;

		status = matrix_save_ascii( filename, matrix, nrows, ncols, pitch, real_data, transpose, append, mt );
	}

	// ----------------------------------------

	return status;

} // matrix_save

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Shows matrix's content (data, name, headers and/or labels).
 * Skips name, headers and labels if "mt" is NULL.
 *
 * If "transpose" is 'true':
 * - Reads from "matrix": <nrows> rows and <ncols> columns (padded to <pitch>).
 * - Shows:
 *	<ncols> rows and <nrows> columns.
 *	<ncols> row labels from mt->headers, and <nrows> column headers from mt->labels.
 *
 * ncols <= pitch.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int matrix_show( void const *restrict matrix, index_t nrows, index_t ncols, index_t pitch, bool real_data, bool transpose, bool all_processes,
		struct matrix_tags_t const *restrict mt )
{

	// Full matrix dimensions
	index_t const full_major_dim = ( transpose ? ncols : nrows );
	index_t const full_minor_dim = ( transpose ? nrows : ncols );

	// Portion of input matrix to print.
	#if NMFGPU_TESTING
		// Testing mode: prints the whole matrix.
		index_t const short_major_dim = full_major_dim;
		index_t const short_minor_dim = full_minor_dim;
	#else
		// Prints just a portion of the input matrix.
		index_t const short_major_dim = MIN( full_major_dim, 9 );
		index_t const short_minor_dim = MIN( full_minor_dim, ( (full_major_dim > 1) ? 15 : 225 ) );
	#endif

	int status = EXIT_SUCCESS;

	char const *restrict name = NULL;
	struct tag_t labels = new_empty_tag();
	struct tag_t headers = new_empty_tag();

	bool hasname = false;
	bool hasheaders = false;
	bool haslabels = false;

	// Maximum length of each row label to show.
	int const label_precision = 5;

	////////////////////////////////

	// Checks for NULL parameters
	if ( ! matrix ) {
		print_errnum( error_shown_by_all, EFAULT, "matrix_show( matrix )" );
		return EXIT_FAILURE;
	}

	// Checks matrix dimensions (quiet mode).
	status = matrix_check_dimensions( "matrix_show", nrows, ncols, pitch, transpose, false );
	if ( status != EXIT_SUCCESS )
		return EXIT_FAILURE;

	// -----------------------------

	// Prints tag elements.

	if ( mt ) {
		name = mt->name;
		headers = ( transpose ? mt->labels  : mt->headers );
		labels  = ( transpose ? mt->headers : mt->labels  );

		hasname = name;
		hasheaders = headers.tokens;
		haslabels  = labels.tokens;

		bool const prefix = true;

		// Name
		if ( hasname ) {
			struct tag_t const tag_name = new_tag( (char *restrict)name, (char **restrict)&name );	// Fakes a tag_t struct

			status = show_tag( tag_name, "Name", 1, 1, prefix, all_processes );
			if ( status != EXIT_SUCCESS ) {
				print_error( error_shown_by_all, "Error in matrix_show( transpose=%i )\n", transpose );
				return EXIT_FAILURE;
			}
		}

		// Column headers
		if ( hasheaders ) {
			status = show_tag( headers, "Headers", full_minor_dim, short_minor_dim, prefix, all_processes );
			if ( status != EXIT_SUCCESS ) {
				print_error( error_shown_by_all, "Error in matrix_show( transpose=%i )\n", transpose );
				return EXIT_FAILURE;
			}
		}

	} // if mt != NULL

	// ----------------------------

	// Prints matrix, with row labels, if exist

	if ( hasname + hasheaders + haslabels + transpose ) {
		status = print_message( all_processes, "Data matrix: %" PRI_IDX " rows x %" PRI_IDX " columns.\n",
					full_major_dim, full_minor_dim );
		if ( status != EXIT_SUCCESS )
			return EXIT_FAILURE;
	}

	// Warns about possibly truncated row labels.
	if ( haslabels ) {
		status = print_message( all_processes, "Please note that only %i characters will be shown for each row label.\n\n",
					label_precision );
		if ( status != EXIT_SUCCESS )
			return EXIT_FAILURE;
	}


	// Step sizes for outer and inner loops.
	index_t incr_outer_loop = ( transpose ? 1 : pitch );
	index_t incr_inner_loop = ( transpose ? pitch : 1 );
	errno = 0;

	size_t const data_size = ( real_data ? sizeof(real) : sizeof(index_t) );

	void const *pmatrix = matrix;
	for ( index_t i = 0 ; i < short_major_dim ; i++, pmatrix += (incr_outer_loop * data_size) ) {

		if ( append_printed_message( all_processes, "Line %" PRI_IDX ":", i ) != EXIT_SUCCESS )
			return EXIT_FAILURE;

		// Prints a (truncated) row label.
		if ( haslabels && (append_printed_message( all_processes, " %.*s.:", label_precision, labels.ptokens[i] ) != EXIT_SUCCESS) )
			return EXIT_FAILURE;

		void const *p = pmatrix;
		for ( index_t j = 0 ; j < short_minor_dim ; j++, p += (incr_inner_loop * data_size) ) {

			data_t val;
			memcpy( &val, p, data_size );

			if ( real_data )
				status = append_printed_message( all_processes, " %g", val.r );
			else
				status = append_printed_message( all_processes, " %" PRI_IDX, val.i );

			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;

		} // for

		// Last column.
		if ( short_minor_dim < full_minor_dim ) {

			data_t val;
			p = pmatrix + ( (size_t) (full_minor_dim-1) * incr_inner_loop * data_size );

			memcpy( &val, p, data_size );

			if ( real_data )
				status = append_printed_message( all_processes, " ... %g", val.r );
			else
				status = append_printed_message( all_processes, " ... %" PRI_IDX, val.i );

			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		}

		if ( append_printed_message( all_processes, "\n" ) != EXIT_SUCCESS )
			return EXIT_FAILURE;

	} // for i=[0..short_major_dim)

	// Last row.
	if ( short_major_dim < full_major_dim ) {

		index_t const i = full_major_dim - 1;
		pmatrix = matrix + ((size_t) i * incr_outer_loop * data_size);

		if ( append_printed_message( all_processes, "...\nLine %" PRI_IDX ":", i ) != EXIT_SUCCESS )
			return EXIT_FAILURE;

		// Prints a (truncated) row label.
		if ( haslabels && (append_printed_message( all_processes, " %.*s:", label_precision, labels.ptokens[i] ) != EXIT_SUCCESS) )
			return EXIT_FAILURE;

		void const *p = pmatrix;
		for ( index_t j = 0 ; j < short_minor_dim ; j++, p += (incr_inner_loop * data_size) ) {

			data_t val;
			memcpy( &val, p, data_size );

			if ( real_data )
				status = append_printed_message( all_processes, " %g", val.r );
			else
				status = append_printed_message( all_processes, " %" PRI_IDX, val.i );

			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;

		} // for

		// Last column.
		if ( short_minor_dim < full_minor_dim ) {

			data_t val;
			p = pmatrix + ( (size_t)(full_minor_dim-1) * incr_inner_loop * data_size );

			memcpy( &val, p, data_size );

			if ( real_data )
				status = append_printed_message( all_processes, " ... %g", val.r );
			else
				status = append_printed_message( all_processes, " ... %" PRI_IDX, val.i );

			if ( status != EXIT_SUCCESS )
				return EXIT_FAILURE;
		}

		if ( append_printed_message( all_processes, "\n" ) != EXIT_SUCCESS )
			return EXIT_FAILURE;

	} // if ( short_major_dim < full_major_dim )

	status = append_printed_message( all_processes, "\n" );

	return status;

} // matrix_show

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Transposes a matrix using a temporary file.
 *
 * If "mt" is non-NULL, swaps row labels and column headers.
 *
 * <base_filename> is used only if no temporary file from the system can be employed.
 * It is never referenced otherwise.
 *
 * ncols <= pitch
 * Then, "pitch" it is updated to <nrows>, rounded up to <memory_alignment>.
 *
 * WARNING:
 *	- Pointer "matrix" is ALWAYS CHANGED, even on error.
 *	- NO ERROR-CHECKING IS PERFORMED (e.g., overflow, invalid values...).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int matrix_transpose_file( void *restrict matrix, index_t *restrict nrows, index_t *restrict ncols, index_t *restrict pitch, size_t data_size,
				struct matrix_tags_t *restrict mt, char const *restrict base_filename )
{

	index_t numcols = 0, numrows = 0, l_pitch = 0;

	bool custom_file = false;		// Uses a custom file, rather than one generated by the system.
	char *restrict filename_tmp = NULL;	// Custom filename (used only if no temporary file from the system can be employed).

	int status = EXIT_SUCCESS;

	////////////////////////////////

	// Checks for NULL parameters
	if ( ! ( (uintptr_t) matrix * (uintptr_t) nrows * (uintptr_t) ncols * (uintptr_t) pitch * (uintptr_t) data_size ) ) {
		int const errnum = EFAULT;
		if ( ! matrix )	print_errnum( error_shown_by_all, errnum, "matrix_transpose_file( matrix )" );
		if ( ! nrows )	print_errnum( error_shown_by_all, errnum, "matrix_transpose_file( nrows )" );
		if ( ! ncols )	print_errnum( error_shown_by_all, errnum, "matrix_transpose_file( ncols )" );
		if ( ! pitch )	print_errnum( error_shown_by_all, errnum, "matrix_transpose_file( pitch )" );
		if ( ! data_size ) print_errnum( error_shown_by_all, EINVAL, "matrix_transpose_file( data_size )" );
		return EXIT_FAILURE;
	}

	numrows = *nrows;
	numcols = *ncols;
	l_pitch = *pitch;

	// Checks matrix dimensions.
	bool const transpose = true;
	bool const verbose = false;

	status = matrix_check_dimensions( "matrix_transpose_file", numrows, numcols, l_pitch, transpose, verbose );

	if ( status != EXIT_SUCCESS )
		return EXIT_FAILURE;


	// ---------------------------

	// Opens a temporary file

	FILE *restrict file = tmpfile();

	if ( ! file ) {

		// Uses a custom file.
		custom_file = true;

		if ( ! base_filename ) {
			print_errnum( error_shown_by_all, EFAULT, "matrix_transpose_file( base_filename )" );
			return EXIT_FAILURE;
		}

		size_t const len = strlen(base_filename) + 8;
		filename_tmp = (char *restrict) malloc( len * sizeof(char) );
		if ( ! filename_tmp ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_transpose_file(): malloc( filename_tmp, length=%zu )",
					len );
			return EXIT_FAILURE;
		}
		errno = 0;
		int const conv = sprintf( filename_tmp, "%s_t.dat", base_filename );
		if ( conv <= 0 ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_transpose_file(): sprintf( filename_tmp, %s_t.dat )",
					base_filename );
			free(filename_tmp);
			return EXIT_FAILURE;
		}

		file = fopen( filename_tmp, "w+b" );	// Open for reading and writing.
		if ( ! file ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in matrix_transpose_file(): fopen( %s )", filename_tmp );
			free(filename_tmp);
			return EXIT_FAILURE;
		}

	} // If requires a custom file.

	// ---------------------------

	// Writes matrix by columns
	void const *pmatrix = (void const *restrict) matrix;
	for ( index_t j = 0 ; j < numcols ; j++, pmatrix += data_size ) {

		void const *pmatrix_r = pmatrix;

		for ( index_t i=0 ; i<numrows ; i++, pmatrix_r += (l_pitch * data_size) )

			if ( fwrite( pmatrix_r, data_size, 1, file ) != 1 ) {
				print_errnum( sys_error_shown_by_all, errno, "Error in matrix_transpose_file(): fwrite( row %"
						PRI_IDX ", column %" PRI_IDX " )", i, j );
				fclose(file);
				if ( custom_file ) { unlink(filename_tmp); free(filename_tmp); }
				return EXIT_FAILURE;
			}
	} // for j

	// ---------------------------

	// Now, reads the file.

	rewind(file);

	// Swaps matrix dimensions.
	{
		index_t const tmp = numrows;
		numrows = numcols;
		numcols = tmp;
	}
	l_pitch = get_padding( numcols );	// New padding

	void *pmatrix_r = matrix;
	for ( index_t i=0 ; i<numrows ; i++, pmatrix_r += (l_pitch * data_size) ) {
		errno = 0;
		size_t const nread = fread( pmatrix_r, data_size, numcols, file );
		if ( nread != (size_t) numcols ) {
			if ( ferror(file) )
				print_errnum( sys_error_shown_by_all, errno, "Error in matrix_transpose_file(): fread( %zu items read, %"
						PRI_IDX " expected )", nread, numcols );
			else	// EOF
				print_error( error_shown_by_all, "Error in matrix_transpose_file(): fread( %" PRI_IDX
						" items ): Premature end-of-file detected (%zu items read).\n", numcols, nread );
			fclose(file);
			if ( custom_file ) { unlink(filename_tmp); free(filename_tmp); }
			return EXIT_FAILURE;
		}
	}
	fclose(file);
	if ( custom_file ) { unlink(filename_tmp); free(filename_tmp); }

	// ---------------------------

	// Swaps matrix tags, if any
	if ( mt ) {
		struct matrix_tags_t l_mt = swap_matrix_tags( *mt );
		*mt = l_mt;
	}

	// ---------------------------

	*nrows = numrows;
	*ncols = numcols;
	*pitch = l_pitch;

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
