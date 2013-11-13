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
 * matrix_io_routines.h
 *	I/O methods for working with (labeled) matrices.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Debug / Testing:
 *		NMFGPU_DEBUG_READ_FILE2: Shows information about the line/token being read.
 *
 * WARNING:
 *	- Requires support for ISO-C99 standard. It can be enabled with 'gcc -std=c99'.
 *
 **********************************************************/

#if ! NMFGPU_MATRIX_IO_ROUTINES_H
#define NMFGPU_MATRIX_IO_ROUTINES_H (1)

#include "index_type.h"

//////////////////////////////////////////////////

/* Selects the appropriate "restrict" keyword. */

#undef RESTRICT

#if defined(__CUDACC__)			/* CUDA source code */
	#define RESTRICT __restrict__
#else					/* C99 source code */
	#define RESTRICT restrict
#endif

//////////////////////////////////////////////////
//////////////////////////////////////////////////

/* Type definition for column and row labels */

// Structure for tags (column/row labels)
struct tags {
	char const *RESTRICT tokens;		// String with tokens.
	char const *const *RESTRICT ptokens;	// Pointer to each token.
};
#ifndef NEW_TAGS
	#define NEW_TAGS(t,p) ((struct tags) { (char const *) (t), (char const *const *) (p) })
#endif

// Structure for matrix labels.
struct matrix_labels {
	char const *RESTRICT name;	// Description string
	struct tags headers;		// Column headers.
	struct tags labels;		// Row labels.
};
#ifndef NEW_MATRIX_LABELS
	#define NEW_MATRIX_LABELS(n,h,ph,l,pl) ((struct matrix_labels) { (char const *) (n), NEW_TAGS( (h), (ph) ), NEW_TAGS( (l), (pl) ) })
#endif

//////////////////////////////////////////////////
//////////////////////////////////////////////////

/* Always process this header as C code, not C++. */
#ifdef __cplusplus
extern "C" {
#endif

//////////////////////////////////////////////////

/*
 * Reads a single line from an ASCII-text file.
 *
 * Returns:
 *	0 on End-Of-File (*str is set to NULL) or if just a single '\n' was read (*str is NOT NULL).
 *	1 on internal errors (*str is set to NULL) or any NULL parameter.
 *	Length of line on success (0 if only a '\n' was read), *str is NOT NULL.
 */
size_t read_line( FILE *RESTRICT file, char *RESTRICT *RESTRICT str );

// ----------------------------------------------

/*
 * Reads a single token (i.e., a word) from and ASCII-text file
 *
 * If 'last_char' is non-NULL, sets the last character read.
 *
 * Returns:
 *	0 on End-Of-File (*str is set to NULL), or if just a '\n' or a delimiter was read.
 *	1 on internal errors (*str is set to NULL) or any NULL parameter.
 *	Length of line on success (0 if only a '\n' or a delimiter was read).
 */
size_t read_token( FILE *RESTRICT file, int delimiter, char *RESTRICT *RESTRICT str, int *RESTRICT last_char );

// ----------------------------------------------

/*
 * Parses a string into a sequence of tokens.
 *
 * The 'delimiter' argument specifies the character that delimits the tokens in the parsed string.
 *
 * In contrast to strtok(3), contiguous delimiter characters represent EMPTY tokens.
 * Similarly for delimiters at the start or end of the string.
 *
 * Returns the number of tokens (>=1) and a vector of pointers to, or 0 and NULL on error.
 */
size_t tokenize( char *RESTRICT str, char **RESTRICT *RESTRICT pstr, int delimiter );

// ----------------------------------------------

/*
 * Searches for tokens and sets pointers to.
 *
 * WARNING: numtokens <= length( pstr )
 */
void retok( char const *RESTRICT str, char const **RESTRICT pstr, index_t numtokens );

// ----------------------------------------------

/*
 * Generates a label string in the form: "<tag_prefix><t><tag_suffix>", where t0 <= t < (t0+num_tags).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int generate_labels( char const *RESTRICT tag_prefix, char const *RESTRICT tag_suffix, index_t t0, index_t num_tags,
			struct tags *RESTRICT labels );

// ----------------------------------------------

/*
 * Cleans labels.
 */
void clean_labels( struct tags labels );

// ----------------------------------------------

/*
 * Cleans all matrix labels (name, column headers and row labels).
 */
void clean_matrix_labels( struct matrix_labels ml );

//////////////////////////////////////////////////

/* Always process this header as C code, not C++. */
#ifdef __cplusplus
}
#endif

//////////////////////////////////////////////////
//////////////////////////////////////////////////

#undef RESTRICT	/* To select the appropriate "restrict" keyword. */

//////////////////////////////////////////////////

#endif /* NMFGPU_MATRIX_IO_ROUTINES_H */
