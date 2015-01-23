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
 * matrix_io_routines.c
 *	I/O methods for working with tagged matrices.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Debug / Testing:
 *		NMFGPU_DEBUG_READ_FILE2: Shows information about the line/token being read.
 *
 **********************************************************
 **********************************************************
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

#include "matrix_io/matrix_io_routines.h"
#include "common.h"
#include "index_type.h"

#include <math.h>	/* log10f, ceilf */
#include <ctype.h>	/* isblank, isprint */
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>	/* uintptr_t, uintmax_t */
#include <stdbool.h>
#include <stdlib.h>

////////////////////////////////////////////////
////////////////////////////////////////////////

/* "Private" global variables */

// Information and/or error messages shown by all processes.
#if NMFGPU_DEBUG_READ_FILE2
	static bool const dbg_shown_by_all = false;	// Information or error messages on debug.
#endif
static bool const sys_error_shown_by_all = true;	// System error messages.
static bool const error_shown_by_all = false;		// Error messages on invalid arguments or I/O data.

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Reads a single line from an ASCII-text file.
 *
 * Return values:
 *	In the following cases, *str is NULL:
 *		0 on end of file.
 *		1 on internal errors or any invalid parameter.
 *		2 on invalid file format or maximum line length is reached.
 *
 *	In the following cases, *str is NON-NULL, but any end-of-line character is removed:
 *		0 if just a '\n' was read.
 *		Length of line on success.
 */
size_t read_line( FILE *restrict file, char *restrict *restrict str )
{

	if ( ! ( (uintptr_t) file * (uintptr_t) str ) ) {	// (file == NULL) || (str == NULL)
		int const errnum = EFAULT;
		if ( ! file ) print_errnum( error_shown_by_all, errnum, "read_line(file)" );
		if ( str ) { *str = NULL; } else { print_errnum( error_shown_by_all, errnum, "read_line(str)"); }
		return 1;
	}

	// ---------------------

	// Maximum line length, in number of characters.
	size_t const max_size = matrix_max_num_items;

	// Initial buffer size (actually, the half of), in number of characters.
	size_t max_len_data = 256;

	char *restrict data = NULL;
	size_t len_data = 0;
	char c = '\0';

	do {
		// Allocates (more) memory.
		{
			max_len_data *= 2;
			max_len_data = MIN( max_len_data, max_size );
			char *const tmp = (char *) realloc(data, max_len_data * sizeof(char));
			if ( ! tmp ) {
				print_errnum( sys_error_shown_by_all, errno, "Error in read_line(): realloc( data, size=%zu )",
						max_len_data );
				*str = NULL;
				free(data);
				return 1;
			}
			data = tmp;
		}

		char *p = data + len_data;	// Overwrites the '\0' (if exists).

		errno = 0;
		if ( ! fgets(p, max_len_data - len_data, file) ) {

			if ( errno + ferror(file) ) {
				print_errnum( sys_error_shown_by_all, errno, "Error in read_line(): fgets(p, size=%zu)",
						max_len_data - len_data);
				*str = NULL;
				free(data);
				return 1;
			}

			// Empty file
			if ( ! len_data ) {
				*str = NULL;
				free(data);
				return 0;
			}

			// EOF and nothing read.
			break;
		}

		len_data += strlen(p);
		c = data[ len_data - 1 ];	// Before the '\0'

	} while ( (c != '\n') * (! feof(file)) * (max_len_data < max_size) );

	// Maximum line length reached
	if ( (len_data >= (max_size-1)) * (c != '\n') * (! feof(file)) ) {
		print_error( error_shown_by_all, "Maximum line length (%zu characters) reached.\n", max_size );
		*str = NULL;
		free(data);
		return 2;
	}

	// At this point: c == EOL (i.e., '\r\n' or just '\n') || EOF detected

	// Deletes any existing new-line sequence.
	if ( c == '\n' ) {
		len_data--;
		if ( len_data && (data[ len_data - 1 ] == '\r') )
			len_data--;
	}
	data[ len_data ] = '\0';	// Makes sure the string finishes with a '\0'.

	#if NMFGPU_DEBUG_READ_FILE2
	////////////////////////////////
		print_message( dbg_shown_by_all, "\tLen=%zu(%zu):%s.\n\tResizing data from %zu to %zu.\n",
				len_data, strlen(data), data, max_len_data, len_data + 1 );
	////////////////////////////////
	#endif

	// Adjusts memory size.
	*str = (char *) realloc( data, (len_data + 1) * sizeof(char) );	// +1: to keep the '\0'
	if ( ! *str ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in read_line(): realloc( data, size=%zu )", len_data + 1 );
		len_data = 1;
		free(data);
	}

	return len_data;

} // read_line

////////////////////////////////////////////////

/*
 * Reads a single token (i.e., a word) from and ASCII-text file
 *
 * If 'last_char' is non-NULL, sets the last character read.
 *
 * Returns:
 *	In the following cases, *str is NULL:
 *		0 on end of file.
 *		1 on internal errors or any invalid parameter.
 *		2 on invalid file format or maximum line length is reached.
 *
 *	In the following cases, *str is NON-NULL, but any end-of-line character is removed:
 *		0 if just a '\n' or a delimiter was read.
 *		Number of tokens on success.
 */
size_t read_token( FILE *restrict file, int delimiter, char *restrict *restrict str, int *restrict last_char )
{

	if ( ! ((uintptr_t) file * (uintptr_t) str * (delimiter > 0)) ) {	// (file == NULL) || (str == NULL) || (delimiter <= 0)
		int const errnum = EFAULT;
		if ( ! file ) print_errnum( error_shown_by_all, errnum, "read_token(file)" );
		if ( str ) { *str = NULL; } else { print_errnum( error_shown_by_all, errnum, "read_token(str)"); }
		if ( delimiter <= 0 ) print_errnum( error_shown_by_all, EINVAL, "read_token( delimiter=%i )", delimiter );
		if ( last_char ) *last_char = 0;
		return 1;
	}

	// ---------------------

	// Maximum token length, in number of characters.
	size_t const max_size = matrix_max_num_items;

	// Initial buffer size (actually, the half of) in number of characters.
	size_t max_len_data = 32;

	char *restrict data = NULL;
	size_t len_data = 0;
	char c[2] = { 0, 0 };

	// Reads until <delimiter>, EOL or EOF is reached.
	do {
		// Allocates (more) memory
		{
			max_len_data = MIN( max_size, (max_len_data * 2) );

			#if NMFGPU_DEBUG_READ_FILE2
			////////////////////////////////
				print_message( dbg_shown_by_all, "\t(Re)allocating memory for %zu characters (current len_data=%zu)...\n",
					       max_len_data, len_data );
			////////////////////////////////
			#endif

			char *restrict const tmp = (char *restrict) realloc(data, max_len_data * sizeof(char));
			if ( ! tmp ) {
				print_errnum( sys_error_shown_by_all, errno, "Error in read_token(): realloc( data, size=%zu )",
						max_len_data );
				if ( last_char ) *last_char = 0;
				*str = NULL;
				free(data);
				return 1;
			}
			data = tmp;
		}

		// Reads more characters (overwrites '\0').
		char *p = data + len_data;
		*p = '\0';

		/* Reads up to <max_len_data-len_data-1> characters stopping on
		 * a <delimiter> or a '\n', which are also read.
		 *
		 * The format expression consists on two parts:
		 *	1) To match the token, mandatory.
		 *	2) To match a delimiter or a newline character ('\n'), optional.
		 *
		 * Any existing CR ('\r') character is matched on the "token" section.
		 */
		size_t const str_size = 32;
		char str_format[ str_size ];
		sprintf( str_format, "%%%zu[^%c\n]%%1[%c\n]", max_len_data-len_data-1, delimiter, delimiter );

		#if NMFGPU_DEBUG_READ_FILE2
		////////////////////////////////
			print_message( dbg_shown_by_all, "\tFormat data: '%s' (len=%zu)\n", str_format, strlen(str_format) );
		////////////////////////////////
		#endif

		c[1] = c[0] = 0;
		errno = 0;
		int conv = fscanf( file, str_format, p, c );

		#if NMFGPU_DEBUG_READ_FILE2
		////////////////////////////////
		{
			int const err = errno;
			print_message( dbg_shown_by_all, "\tOriginal(len=%zu):'%s',conv=%i,c=", strlen(p), p, conv );
			switch( c[0] ) {
				case 0	 : { append_printed_message( dbg_shown_by_all, "(empty)" ); } break;
				case '\r': { append_printed_message( dbg_shown_by_all, "\\r" ); } break;
				case '\n': { append_printed_message( dbg_shown_by_all, "\\n" ); } break;
				case '\t': { append_printed_message( dbg_shown_by_all, "\\t" ); } break;
				case  ' ': { append_printed_message( dbg_shown_by_all, "' '" ); } break;
				default  : { append_printed_message( dbg_shown_by_all, (isgraph(c[0]) ? "'%c'" : "'\\0x%X'"), c[0] ); } break;
			}
			append_printed_message( dbg_shown_by_all, "\n" );
			errno = err;
		}
		////////////////////////////////
		#endif

		// NOTE: From here: (strlen(p) > 0) || (strlen(p) == c == 0)

		if ( ! conv ) {	// No matching performed

			/* No token was matched, but there still to read a <delimiter> or a LF ('\n') character.
			 * Any existing LF ('\r') character would have been stored in 'p' and would have represented a
			 * successful match (i.e., conv > 0).
			 */
			errno = 0;
			conv = c[0] = fgetc( file );
			c[1] = 0;
		}
		if ( conv == EOF ) {
			if ( ferror(file) ) {
				print_errnum( sys_error_shown_by_all, errno, "Error in read_token(): %s",
						( (conv != c[0]) ? "fscanf(p)" : "fgetc()" ) );
				if ( last_char ) *last_char = 0;
				*str = NULL;
				free(data);
				return 1;
			}

			// Empty file
			if ( ! len_data ) {	// && (c == '\0' || c == EOF)
				if ( last_char ) { *last_char = EOF; }
				*str = NULL;
				free(data);
				return 0;
			}

			// End of file with no conversion: stops reading.
			break;
		}

		len_data += strlen(p);	// >= 0

	} while ( (! c[0]) * (max_len_data < max_size) );	// c == 0: There still more characters to read...

	// Maximum line length reached
	if ( (len_data >= (max_size-1)) * (!c[0]) ) {
		print_error( error_shown_by_all, "Maximum token length (%zu characters) reached.\n", max_size );
		if ( last_char ) *last_char = 0;
		*str = NULL;
		free(data);
		return 2;
	}

	// At this point: (len_data == strlen(data) >= 0) && (c == delimiter || c == '\n' || EOF)

	// Deletes any existing new-line sequence.
	if ( len_data && (data[ (len_data - 1) ] == '\r') )	// && (c == '\n')
		len_data--;
	data[ len_data ] = '\0';	// Makes sure the string finishes with a '\0'.

	#if NMFGPU_DEBUG_READ_FILE2
	////////////////////////////////
		print_message( dbg_shown_by_all, "\tLen=%zu (%zu):'%s'.\n", len_data, strlen(data), data );
	////////////////////////////////
	#endif

	// Returns the last char read if requested.
	if ( last_char ) {
		if ( len_data * (!c[0]) )
			c[0] = data[len_data-1];	// last character.
		*last_char = c[0];
	}

	// Adjusts memory size.
	*str = (char *) realloc( data, (len_data + 1) * sizeof(char) );	// +1 to keep the '\0'
	if ( ! *str ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in read_token(): realloc( data, size=%zu )", len_data + 1 );
		len_data = 1;
		if ( last_char ) { *last_char = 0; }
		*str = NULL;
		free( data );
	}

	return len_data;

} // read_token

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Parses a string into a sequence of tokens.
 *
 * The 'delimiter' argument specifies the character that delimits the tokens in the parsed string.
 *
 * In contrast to strtok(3), contiguous delimiter characters represent EMPTY tokens.
 * Similarly for delimiters at the start or end of the string.
 *
 * Returns:
 *	- A struct tag_t containing str and a new vector of pointers to,
 *	- number of tokens (>=1)
 *	- An empty struct tag_t, and 0 on error.
 */
struct tag_t tokenize( char *restrict str, int delimiter, size_t *restrict ntokens )
{

	if ( ! ( (uintptr_t) str * (uintptr_t) ntokens ) ) {	// (str == NULL) || (ntokens == NULL)
		int const errnum = EFAULT;
		if ( ! str ) print_errnum( error_shown_by_all, errnum, "tokenize( str )");
		if ( ! ntokens ) print_errnum( error_shown_by_all, errnum, "tokenize( ntokens )");
		return new_empty_tag();
	}

	// ---------------------------

	// Maximum number of tokens
	size_t const max_len_pstr = matrix_max_num_items;

	// Initial size.
	size_t len_pstr = MIN( 256, max_len_pstr );

	// Array of pointers to each token.
	char **restrict pstr = (char **restrict) malloc( len_pstr * sizeof(char *) );
	if ( ! pstr ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in tokenize(): malloc( pstr, size=%zu )", len_pstr );
		return new_empty_tag();
	}

	char *ct = str;		// Starting address of current token.
	char *nt = NULL;	// Address of the next token.

	pstr[ 0 ] = ct;		// First token
	size_t num_tokens = 1;

	while ( (uintptr_t)(nt = strchr(ct,delimiter)) * (num_tokens < len_pstr) ) {

		*nt = '\0';	// Replaces delimiter by '\0'
		ct = nt + 1;	// Next token becomes the current one

		// Stores the address of the (new) current token.
		pstr[ num_tokens ] = ct;	// *ct is either a "regular" character, or a '\0'.
		num_tokens++;
	}

	// If there are tokens to process, allocates more memory.
	while ( (uintptr_t)nt * (len_pstr < max_len_pstr) ) {

		len_pstr = MIN( (len_pstr * 2), max_len_pstr );
		char **restrict const tmp = (char **restrict) realloc( pstr, len_pstr * sizeof(char *) );
		if ( ! tmp ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in tokenize(): realloc( pstr, size=%zu )", len_pstr );
			free(pstr);
			return new_empty_tag();
		}
		pstr = tmp;

		// Processes the pending token.
		do {
			*nt = '\0';	// Replaces delimiter by '\0'
			ct = nt + 1;	// Next token becomes the current one

			// Stores the address of the (new) current token
			pstr[ num_tokens ] = ct;	// *ct is either a "regular" character, or a '\0'.
			num_tokens++;

		} while ( (uintptr_t)(nt = strchr(ct,delimiter)) * (num_tokens < len_pstr) );

	} // while nt != NULL and we can allocate more memory.

	// Maximum number of tokens reached.
	if ( (len_pstr >= (max_len_pstr-1)) * (uintptr_t)nt ) {
		print_error( error_shown_by_all, "Maximum number of tokens (%zu) reached.\n", max_len_pstr );
		free(pstr);
		return new_empty_tag();
	}

	// From here: (nt == NULL), and ct points to the last token.

	// Adjusts memory size.
	{
		char **restrict const tmp = (char **restrict) realloc( pstr, num_tokens * sizeof(char *) );
		if ( ! tmp ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in tokenize(): realloc( pstr, size=%" PRI_IDX " )", num_tokens );
			free( pstr );
			return new_empty_tag();
		}
		pstr = tmp;
	}

	// ---------------------------

	*ntokens = num_tokens;
	return new_tag( str, pstr );

} // tokenize

////////////////////////////////////////////////

/*
 * Returns a struct tag_t with the specified arguments.
 *
 * tokens: Parsed string with tokens.
 * ptokens: Array of pointers to each token.
 */
struct tag_t new_tag( char *restrict tokens, char **restrict ptokens )
{

	struct tag_t tag;
	tag.tokens = (char const *restrict) tokens;
	tag.ptokens = (char const *const *restrict) ptokens;

	return tag;

} // new_tag

////////////////////////////////////////////////

/*
 * Returns an empty struct tag_t.
 */
struct tag_t new_empty_tag( void )
{

	struct tag_t const tag = new_tag( NULL, NULL );

	return tag;

} // new_empty_tag

////////////////////////////////////////////////

/*
 * Generates a list of tokens of the form: "<token_prefix><t><token_suffix>", where t0 <= t < (t0 + num_tokens).
 *
 * The list is returned in "tag".
 * (t0 + num_tokens) must not be greater than MAX( matrix_max_pitch, matrix_max_non_padded_dim ).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int generate_tag(char const *restrict token_prefix, char const *restrict token_suffix, index_t t0, index_t num_tokens, struct tag_t *restrict tag)
{

	if ( ( t0 < 0 ) + ( num_tokens <= 0 ) + ( ! tag ) ) {
		int const errnum = EINVAL;
		if ( t0 < 0 ) print_errnum( error_shown_by_all, errnum, "generate_tag( t0=%" PRI_IDX " )", t0 );
		if ( num_tokens <= 0 ) print_errnum( error_shown_by_all, errnum, "generate_tag( num_tokens=%" PRI_IDX " )", num_tokens );
		if ( ! tag ) print_errnum( error_shown_by_all, EFAULT, "generate_tag( tag )" );
		return EXIT_FAILURE;
	}

	index_t const max_num_tokens = MAX( matrix_max_pitch, matrix_max_non_padded_dim );

	if ( num_tokens > max_num_tokens ) {
		print_error( error_shown_by_all, "generate_tag( num_tokens=%" PRI_IDX " ): Excessive number of tokens. Maximum allowed: %"
				PRI_IDX ".\n", num_tokens, max_num_tokens );
		return EXIT_FAILURE;
	}

	if ( (IDX_MAX - num_tokens) < t0 ) {
		print_error( error_shown_by_all, "generate_tag( t0=%" PRI_IDX ", num_tokens=%" PRI_IDX " ): The provided parameters "
				"exceed the limits used for matrix dimensions.\n\t<t0 + num_tokens> must NOT be greater than %"
				PRI_IDX ".\n", t0, num_tokens, IDX_MAX );
		return EXIT_FAILURE;
	}

	// ----------------------------

	char const null_char = '\0';
	char const *tk_p = &null_char;
	char const *tk_s = &null_char;

	if ( token_prefix )
		tk_p = token_prefix;

	if ( token_suffix )
		tk_s = token_suffix;

	// ----------------------------

	/* Roughly estimation of the total length:
	 *
	 * 1) Number of characters required by the last token: (t0 + num_tokens - 1).
	 * 2) Adds the length of both suffix and prefix.
	 * 3) Multiplies the result by <num_tokens>.
	 */
	index_t const last_token = t0 + num_tokens - 1 - 1;	// The additional "-1" might save one character.
	size_t max_len_tokens = (size_t) ceilf( log10f( last_token ) );
	max_len_tokens += (strlen( tk_p ) + strlen( tk_s ) + 1);	// +1 for the delimiter
	max_len_tokens *= num_tokens;
	if ( max_len_tokens > matrix_max_num_items )
		print_error( error_shown_by_all, "generate_tag(): The provided parameters exceed the maximum length for a line (%zu).\n",
				matrix_max_num_items );

	// ----------------------------

	// Tokens.
	char *restrict const tokens = (char *restrict) malloc( max_len_tokens * sizeof(char) ); // Size will be adjusted later.
	if ( ! tokens ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in generate_tag(): malloc( tokens, size=%zu )", max_len_tokens );
		return EXIT_FAILURE;
	}
	size_t len_tokens = 0;	// Current length for tokens.


	// Array of pointers to 'tokens'.
	char const **restrict const ptokens = (char const **restrict) malloc( num_tokens * sizeof(char *) );
	if ( ! ptokens ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in generate_tag(): malloc( ptokens, size=%zu )", num_tokens );
		free( (void *)tokens );
		return EXIT_FAILURE;
	}

	// ----------------------------

	// Generates the tokens.

	for( index_t i = 0, t = t0 ; i < num_tokens ; i++, t++ ) {

		ptokens[ i ] = (char const *) &tokens[ len_tokens ];

		int const len = sprintf( &tokens[ len_tokens ], "%s%" PRI_IDX "%s", tk_p, t, tk_s );

		len_tokens += ((size_t)len + 1);	// (+1 to include the '\0').

	} // for 0 <= i < num_tokens.


	// Adjusts the memory used.
	char *restrict l_tokens = (char *restrict) realloc( tokens, len_tokens * sizeof(char) );
	if ( ! l_tokens ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in generate_tag(): realloc( tokens, size=%zu )", len_tokens );
		free( (void *)ptokens ); free( (void *)tokens );
		return EXIT_FAILURE;
	}

	// ----------------------------

	// Output values
	struct tag_t l_tag = new_tag( l_tokens, (char **restrict) ptokens );

	// If for any reason the address has changed, it is necessary to "retokenize" the string.
	if ( l_tokens != tokens )
		retok( l_tag, num_tokens );

	// Finally, sets the outputs.
	*tag = l_tag;

	return EXIT_SUCCESS;

} // generate_tag

////////////////////////////////////////////////

/*
 * Searches for tokens and resets the array of pointers to.
 *
 * WARNING:
 *	0 < num_tokens <= length( tag.ptokens[] )
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int retok( struct tag_t tag, index_t num_tokens )
{

	char const *restrict str = tag.tokens;
	char const **restrict pstr = (char const **restrict) tag.ptokens;

	if ( (! ((uintptr_t) str * (uintptr_t) pstr)) + (num_tokens <= 0) ) {	// str == NULL || ntokens == NULL || num_tokens <= 0
		int const errnum = EFAULT;
		if ( ! str )  print_errnum( error_shown_by_all, errnum, "retok( tag.tokens )");
		if ( ! pstr ) print_errnum( error_shown_by_all, errnum, "retok( tag.ptokens[] )");
		if ( num_tokens <= 0 ) print_errnum( error_shown_by_all, EINVAL, "retok( num_tokens )" );
		return EXIT_FAILURE;
	}

	// --------------------------------------

	char const *s = str;
	pstr[ 0 ] = str;	// First token

	for ( index_t i = 1 ; i < num_tokens ; i++ ) {
		s += (strlen(s) + 1);	// Jumps to the next token.
		pstr[ i ] = s;
	}

	return EXIT_SUCCESS;

} // retok

////////////////////////////////////////////////

/*
 * Reads a single line which is parsed and separated into a sequence of tokens.
 *
 * The "delimiter" argument specifies the character that delimits the tokens in the parsed string.
 *
 * The resulting sequence of tokens is stored in "tag". If len_tokens is non-NULL, stores the total
 * length of the string. Similarly, if ntokens is non-NULL, stores the number of tokens.
 *
 * If just a newline is read, tag->tokens stores an empty string, which is pointed by tag->ptokens[0].
 * In addition, *len_tokens is set to '0' and *ntokens to '1', if they are non-NULL.
 *
 * Return codes:
 *	0 on success (tag is non-NULL).
 *	1 on end of file (nothing was read).
 *	2 on internal error or invalid parameter(s).
 *	3 on invalid file format, or maximum line length was reached.
 */
int read_tag( FILE *restrict file, int delimiter, struct tag_t *restrict tag, size_t *restrict len_tokens, size_t *restrict ntokens )
{

	if ( ! tag ) {
		print_errnum( error_shown_by_all, EFAULT, "read_tag( tag )" );
		return 2;
	}

	char *restrict data = NULL;
	size_t len_data = 0;
	size_t num_tokens = 0;
	struct tag_t l_tag = new_empty_tag();

	// ---------------------

	// Reads a single line.

	len_data = read_line( file, &data );

	#if NMFGPU_DEBUG_READ_FILE2
		print_message( dbg_shown_by_all, "\t\t\tdata: %s, len_data: %zu\n", ( data ? "non-NULL" : "NULL" ), len_data );
	#endif

	if ( ! data ) {	// EOF, error, or invalid file format.
		if ( len_data == 1 )	// Error
			print_error( error_shown_by_all, "Error in read_tag().\n" );
		return (len_data + 1);
	}

	// ---------------------

	// Divides into tokens using the given delimiter.

	l_tag = tokenize( data, delimiter, &num_tokens );

	#if NMFGPU_DEBUG_READ_FILE2
		print_message( dbg_shown_by_all, "\t\t\ttag: %s, num_tokens: %zu\n", ( l_tag.tokens ? "non-NULL" : "NULL" ), num_tokens );
	#endif

	if ( ! l_tag.tokens ) {
		print_error( error_shown_by_all, "Error in read_tag().\n" );
		free( (void *)data );
		return 2;
	}

	// ---------------------

	// Sets output values

	if ( len_tokens )
		*len_tokens = len_data;

	if ( ntokens )
		*ntokens = num_tokens;

	*tag = l_tag;

	return 0;

} // read_tag

////////////////////////////////////////////////

/*
 * Writes to a file the given tag element in a single line, separated by the given delimiter.
 *
 * prefix: If 'true', also writes a delimiter character before the first token.
 * suffix: If 'true', writes a newline character after the last token.
 *
 * If num_tokens == 0, no field in tag is referenced.
 * If num_tokens == 1, only tag.tokens is referenced.
 * Labels are written only if tag.tokens is non-NULL, regardless of "num_tokens".
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int write_tag( FILE *restrict file, struct tag_t tag, char const *restrict const tokens_name, index_t num_tokens, int delimiter, bool prefix,
		bool suffix )
{

	if ( (! ((uintptr_t) file * (uintptr_t) tokens_name)) + (num_tokens < 0) + (delimiter <= 0) ) {
		int const err_null = EFAULT;
		int const err_inval = EINVAL;
		if ( ! file ) print_errnum( error_shown_by_all, err_null, "write_tag( file )" );
		if ( ! tokens_name )  print_errnum( error_shown_by_all, err_null, "write_tag( tokens_name )" );
		if ( num_tokens < 0 ) print_errnum( error_shown_by_all, err_inval, "write_tag( num_tokens=%" PRI_IDX " )", num_tokens );
		if ( delimiter <= 0 ) print_errnum( error_shown_by_all, err_inval, "write_tag( delimiter=%i )", delimiter );
		return EXIT_FAILURE;
	}

	char const *restrict const data = tag.tokens;
	char const *const *restrict const pdata = tag.ptokens;
	errno = 0;

	// -----------------------------

	// Prefix.
	if ( prefix && (fprintf( file, "%c", delimiter ) <= 0) ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in write_tag(): fprintf( %s, prefix='%c' )", tokens_name, delimiter );
		return EXIT_FAILURE;
	}

	// Tokens
	if ( (uintptr_t)num_tokens * (uintptr_t)data ) {

		if ( fprintf( file, "%s", data ) <= 0 ) {
			print_errnum( sys_error_shown_by_all, errno, "Error in write_tag(): fprintf( %s, token 0/%" PRI_IDX " )",
					tokens_name, num_tokens );
			return EXIT_FAILURE;
		}

		for ( index_t i = 1 ; i < num_tokens ; i++ )
			if ( fprintf( file, "%c%s", delimiter, pdata[ i ] ) <= 0 ) {
				print_errnum( sys_error_shown_by_all, errno, "Error in write_tag(): fprintf( %s, token %" PRI_IDX "/%"
						PRI_IDX " )", tokens_name, i, num_tokens );
				return EXIT_FAILURE;
			}

	} // If there are tokens.

	// Suffix
	if ( suffix && (fprintf( file, "\n" ) <= 0) ) {
		print_errnum( sys_error_shown_by_all, errno, "Error in write_tag(): fprintf( %s, suffix='\\n' )", tokens_name );
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

} // write_tag

////////////////////////////////////////////////

/*
 * Prints the given tag element in a single-line message.
 *
 * num_tokens: Number of tokens in memory (i.e., length(tag.ptokens[])).
 *
 * pnumtokens: Number of tokens to print BEFORE the last one.
 *	That is, it prints tag.ptokens[ 0..(pnumtokens-1) ], followed by
 *	tag.ptokens[ num_tokens-1 ], if pnumtokens < num_tokens.
 *
 * prefix: If 'true', prints <tokens_name> and <num_tokens> before the first token.
 *	Otherwise, tokens_name is not referenced.
 *
 * pnumtokens must NOT be greater than num_tokens
 * If tag.tokens is NULL, nothing is printed and 'EXIT_SUCCESS' is returned.
 * If pnumtokens == 0 or num_tokens == 0, just prints the prefix (if true).
 * If pnumtokens == 1, only tag.tokens is referenced.
 * If pnumtokens > 1, tag.ptokens[] must be non-NULL.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int show_tag( struct tag_t tag, char const *restrict const tokens_name, index_t num_tokens, index_t pnumtokens, bool prefix, bool all_processes )
{

	char const *restrict const data = tag.tokens;
	char const *const *restrict const pdata = tag.ptokens;

	if ( (num_tokens < 0) + (pnumtokens < 0) + (pnumtokens > num_tokens) ) {
		print_errnum( error_shown_by_all, EINVAL, "show_tag( pnumtokens=%" PRI_IDX ", num_tokens=%" PRI_IDX " )",
				pnumtokens, num_tokens );
		return EXIT_FAILURE;
	}

	if ( (prefix * (! tokens_name)) + ((pnumtokens > 1) * (! pdata)) ) {
		int const errnum = EFAULT;
		if ( prefix * (! tokens_name) )     print_errnum( error_shown_by_all, errnum, "show_tag( tokens_name )" );
		if ( (pnumtokens > 1) * (! pdata) ) print_errnum( error_shown_by_all, errnum, "show_tag( pdata )" );
		return EXIT_FAILURE;
	}

	if ( ! data )
		return EXIT_SUCCESS;


	// -----------------------------


	// Prefix.
	if ( prefix && (print_message( all_processes, "%s (%" PRI_IDX "):\n", tokens_name, num_tokens ) != EXIT_SUCCESS) )
		return EXIT_FAILURE;

	// Tokens
	if ( pnumtokens ) {

		// First token
		if ( append_printed_message( all_processes, "'%s'", data ) != EXIT_SUCCESS )
			return EXIT_FAILURE;

		for ( index_t i = 1 ; i < pnumtokens ; i++ )
			if ( append_printed_message( all_processes, " '%s'", pdata[ i ] ) != EXIT_SUCCESS )
				return EXIT_FAILURE;
	}

	// Last token
	if ( (pnumtokens < num_tokens) &&
		(append_printed_message( all_processes, " ... '%s'", pdata[ num_tokens - 1 ] ) != EXIT_SUCCESS) )
		return EXIT_FAILURE;

	return append_printed_message( all_processes, "\n" );

} // show_tag

////////////////////////////////////////////////

/*
 * Cleans the given tag element.
 */
void clean_tag( struct tag_t tag )
{

	if ( tag.ptokens )
		free((void *) tag.ptokens);

	if ( tag.tokens )
		free((void *) tag.tokens);

} // clean_tag

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Returns a new struct matrix_tags_t with the specified arguments.
 *
 * name: A short description string.
 * headers: A struct tag_t with column headers.
 * labels: A struct tag_t with row labels.
 */
struct matrix_tags_t new_matrix_tags( char *restrict name, struct tag_t headers, struct tag_t labels )
{

	struct matrix_tags_t mt;

	mt.name = (char const *restrict) name;
	mt.headers = headers;
	mt.labels = labels;

	return mt;

} // new_matrix_tags

////////////////////////////////////////////////

/*
 * Returns an empty struct matrix_tags_t.
 */
struct matrix_tags_t new_empty_matrix_tags( void )
{

	char *restrict const name = NULL;
	struct tag_t const tags_h = new_empty_tag();
	struct tag_t const tags_l = new_empty_tag();

	struct matrix_tags_t mt = new_matrix_tags( name, tags_h, tags_l );

	return mt;

} // new_empty_matrix_tags

////////////////////////////////////////////////

/*
 * Swaps row labels and column headers.
 */
struct matrix_tags_t swap_matrix_tags( struct matrix_tags_t mt )
{

	struct matrix_tags_t new_mt = new_matrix_tags( (char *restrict) mt.name, mt.labels, mt.headers );

	return new_mt;

} // swap_matrix_tags

////////////////////////////////////////////////

/*
 * Cleans all matrix tag elements (name, column headers and row labels).
 */
void clean_matrix_tags( struct matrix_tags_t mt )
{

	clean_tag( mt.labels );
	clean_tag( mt.headers );

	if ( mt.name )
		free((void *) mt.name);

} // clean_matrix_tags

////////////////////////////////////////////////
////////////////////////////////////////////////
