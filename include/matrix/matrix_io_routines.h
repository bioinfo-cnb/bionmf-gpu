/************************************************************************
 *
 * NMF-mGPU -- Non-negative Matrix Factorization on multi-GPU systems.
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
 * matrix_io_routines.h
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
 ****************
 *
 * WARNING:
 *	+ This code requires support for ISO-C99 standard. It can be enabled with 'gcc -std=c99'.
 *
 **********************************************************/

#if ! NMFGPU_MATRIX_IO_ROUTINES_H
#define NMFGPU_MATRIX_IO_ROUTINES_H (1)

#include "index_type.h"

#include <stdbool.h>
#include <stdio.h>

////////////////////////////////////////////////
////////////////////////////////////////////////

/* Selects the appropriate "restrict" keyword. */

#undef RESTRICT

#if __CUDACC__				/* CUDA source code */
	#define RESTRICT __restrict__
#else					/* C99 source code */
	#define RESTRICT restrict
#endif

/* Always process this header as C code, not C++. */
#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------
// ---------------------------------------------

/* Type definitions for matrix tags */

// Structure for a tag element.
struct tag_t {
	char const *RESTRICT tokens;		// String with tokens.
	char const *const *RESTRICT ptokens;	// Array of pointers to each token.
};

// Structure for tag elements.
struct matrix_tags_t {
	char const *RESTRICT name;	// Description string.
	struct tag_t headers;		// Column headers.
	struct tag_t labels;		// Row labels.
};

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
size_t read_line( FILE *RESTRICT file, char *RESTRICT *RESTRICT str );

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
size_t read_token( FILE *RESTRICT file, int delimiter, char *RESTRICT *RESTRICT str, int *RESTRICT last_char );

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
struct tag_t tokenize( char *RESTRICT str, int delimiter, size_t *RESTRICT ntokens );

////////////////////////////////////////////////

/*
 * Returns a struct tag_t with the specified arguments.
 *
 * tokens: Parsed string with tokens.
 * ptokens: Array of pointers to each token.
 */
struct tag_t new_tag( char *RESTRICT tokens, char **RESTRICT ptokens );

////////////////////////////////////////////////

/*
 * Returns an empty struct tag_t.
 */
struct tag_t new_empty_tag( void );

////////////////////////////////////////////////

/*
 * Generates a list of tokens of the form: "<token_prefix><t><token_suffix>", where t0 <= t < (t0 + num_tokens).
 *
 * The list is returned in "tag".
 * (t0 + num_tokens) must not be greater than MAX( matrix_max_pitch, matrix_max_non_padded_dim ).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int generate_tag(char const *RESTRICT token_prefix, char const *RESTRICT token_suffix, index_t t0, index_t num_tokens, struct tag_t *RESTRICT tag);

////////////////////////////////////////////////

/*
 * Searches for tokens and resets the array of pointers to.
 *
 * WARNING:
 *	0 < num_tokens <= length( tag.ptokens[] )
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int retok( struct tag_t tag, index_t num_tokens );

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
int read_tag( FILE *RESTRICT file, int delimiter, struct tag_t *RESTRICT tag, size_t *RESTRICT len_tokens, size_t *RESTRICT ntokens );

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
int write_tag( FILE *RESTRICT file, struct tag_t tag, char const *RESTRICT const tokens_name, index_t num_tokens, int delimiter, bool prefix,
		bool suffix );

////////////////////////////////////////////////

/*
 * Prints the given tag element in a single-line message.
 *
 * num_tokens: Number of tokens in memory (i.e., length(tag.ptokens[])).
 *
 * pnumtokens: Number of tokens to be printed BEFORE the last one.
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
int show_tag( struct tag_t tag, char const *RESTRICT const tokens_name, index_t num_tokens, index_t pnumtokens, bool prefix, bool all_processes );

////////////////////////////////////////////////

/*
 * Cleans the given tag element.
 */
void clean_tag( struct tag_t tag );

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Returns a new struct matrix_tags_t with the specified arguments.
 *
 * name: A short description string.
 * headers: A struct tag_t with column headers.
 * labels: A struct tag_t with row labels.
 */
struct matrix_tags_t new_matrix_tags( char *RESTRICT name, struct tag_t headers, struct tag_t labels );

////////////////////////////////////////////////

/*
 * Returns an empty struct matrix_tags_t.
 */
struct matrix_tags_t new_empty_matrix_tags( void );

////////////////////////////////////////////////

/*
 * Swaps row labels and column headers.
 */
struct matrix_tags_t swap_matrix_tags( struct matrix_tags_t mt );

////////////////////////////////////////////////

/*
 * Cleans all matrix tag elements (name, column headers and row labels).
 */
void clean_matrix_tags( struct matrix_tags_t mt );

////////////////////////////////////////////////
////////////////////////////////////////////////

#undef RESTRICT	/* To select the appropriate "restrict" keyword. */

/* Always process this header as C code, not C++. */
#ifdef __cplusplus
}
#endif

////////////////////////////////////////////////
////////////////////////////////////////////////

#endif /* NMFGPU_MATRIX_IO_ROUTINES_H */
