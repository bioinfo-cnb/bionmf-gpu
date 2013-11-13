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
 * matrix_io_routines.c
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>	/* SIZE_MAX */

#include "matrix/matrix_io_routines.h"

//////////////////////////////////////////////////
//////////////////////////////////////////////////

/*
 * Reads a single line from an ASCII-text file.
 *
 * Returns:
 *	0 on End-Of-File (*str is set to NULL) or if just a single '\n' was read (*str is NOT NULL).
 *	1 on internal errors (*str is set to NULL) or any NULL parameter.
 *	Length of line on success (0 if only a '\n' was read), *str is NOT NULL.
 */
size_t read_line( FILE *restrict file, char *restrict *restrict str )
{

	if ( ! ( (size_t) file * (size_t) str ) ) {	// (file == NULL) || (str == NULL)
		fflush( stdout );
		if ( ! file ) { errno = EBADF; perror("\nread_line(file)"); }
		if ( str ) { *str = NULL; } else { errno = EFAULT; perror("\nread_line(str)"); }
		return 1;
	}

	// ---------------------

	// Maximum line length (in number of items, not in bytes)
	size_t max_size = SIZE_MAX / 4;

	// Initial buffer size (actually, the half of) in items, not it bytes.
	size_t max_len_data = 256;				// <= max_size/2

	char *restrict data = NULL;
	size_t len_data = 0;
	char c = '\0';

	do {
		// Allocates (more) memory.
		{
			max_len_data *= 2;
			char *const tmp = realloc(data, max_len_data * sizeof(char));
			if ( ! tmp ) {
				int const err = errno; fflush(stdout); errno = err;
				perror("\nrealloc(data)");
				fprintf(stderr,"Error in read_line().\n");
				*str = NULL;
				free(data);
				return 1;
			}
			data = tmp;
		}

		char *p = data + len_data;	// Overwrites the '\0' (if exists).

		if ( ! fgets(p, max_len_data - len_data, file) ) {

			if ( errno + ferror(file) ) {
				int const err = errno; fflush(stdout); errno = err;
				if ( errno )
					perror("\nfgets(p)");
				else
					fprintf(stderr,"\nInternal error in fgets(p):\n");
				fprintf(stderr,"Error in read_line().\n");
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

	} while ( (c != '\n') * (! feof(file)) * (max_len_data <= (max_size/2)) );

	// Maximum line length reached
	if ( (len_data == (max_size-1)) * (c != '\n') * (! feof(file)) ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf(stderr,"\nread_line(): Maximum line length (%zu items) reached.\n", max_size );
		*str = NULL;
		free(data);
		return 1;
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
		printf("\tLen=%zu(%zu):%s.\n",len_data,strlen(data),data);
		printf("Resizing data from %zu to %zu\n",max_len_data,len_data+1);
	#endif

	// Adjusts memory size.
	*str = realloc( data, (len_data + 1) * sizeof(char) );	// +1: to keep the '\0'
	if ( ! *str ) {
		int const err = errno; fflush(stdout); errno = err;
		perror("\nrealloc(data)");
		fprintf(stderr,"Error in read_line().\n");
		len_data = 1;
		free( data );
	}

	return len_data;

} // read_line

//////////////////////////////////////////////////

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
size_t read_token( FILE *restrict file, int delimiter, char *restrict *restrict str, int *restrict last_char )
{

	if ( ! ( (size_t) file * (size_t) str ) ) {	// (file == NULL) || (str == NULL)
		fflush( stdout );
		if ( ! file ) { errno = EBADF; perror("\nread_token(file)"); }
		if ( str ) { *str = NULL; } else { errno = EFAULT; perror("\nread_token(str)"); }
		if ( last_char ) *last_char = 0;
		return 1;
	}

	// ---------------------

	// Maximum line length (in number of items, not in bytes)
	size_t max_size = SIZE_MAX / 4;

	// Initial buffer size (actually, the half of) in items, not it bytes.
	size_t max_len_data = 32;				// <= max_size/2

	char *restrict data = NULL;
	size_t len_data = 0;
	int c = 0;	// = (int) '\0';

	// Reads until <delimiter>, EOL or EOF is reached.
	do {
		// Allocates (more) memory
		{
			max_len_data *= 2;
			char *const tmp = realloc(data, max_len_data * sizeof(char));
			if ( ! tmp ) {
				int const err = errno; fflush(stdout); errno = err;
				perror("\nrealloc( data )");
				fprintf(stderr,"Error in read_token().\n");
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

		/* Reads up to max_len_data-len_data-1 characters stopping on <delimiter> or '\n'. The delimiter is also read.
		 * NOTE:
		 *	Any existing CR ('\r') character is matched on the first part of the expression.
		 *	Only a *FULL* match is processed.
		 */
		char str_format[32];
		sprintf( str_format, "%%%zu[^%c\n]%%1[%c\n]", max_len_data-len_data-1, delimiter, delimiter );

		#if NMFGPU_DEBUG_READ_FILE2
			printf("\tFormat data: '%s' (len=%zu)\n",str_format,strlen(str_format));
		#endif

		int conv = fscanf( file, str_format, p, &c );

		#ifdef NMFGPU_DEBUG_READ_FILE2
			printf("\tOriginal(len=%zu):'%s'.\n",strlen(p),p);
		#endif

		// NOTE: From here: (strlen(p) > 0 && c > 0) \\ (strlen(p) == c == 0)
		if ( ! conv ) {	// No matching performed

			/* No token was matched, but there still to be read a <delimiter> or a LF ('\n') character.
			 * Any existing LF ('\r') character would had been stored in 'p' and would had represented
			 * a successful match (i.e., conv > 0).
			 */
			c = fgetc( file );
			conv = c;
		}
		if ( conv == EOF ) {
			if ( ferror(file) ) {
				int const err = errno; fflush(stdout); errno = err;
				perror("\nfscanf(p)");
				fprintf(stderr,"Error in read_token().\n");
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

	} while ( (!c) * (max_len_data <= (max_size/2)) );

	// Maximum line length reached
	if ( (len_data == (max_size-1)) * (!c) ) {
		int const err = errno; fflush(stdout); errno = err;
		fprintf(stderr,"\nread_token(): Maximum token length (%zu) reached.\n", max_size );
		if ( last_char ) *last_char = 0;
		*str = NULL;
		free(data);
		return 1;
	}

	// At this point: (len_data == strlen(data) >= 0) && (c == delimiter || c == '\n' || EOF)

	// Deletes any existing new-line sequence.
	if ( len_data && (data[ (len_data - 1) ] == '\r') )	// && (c == '\n')
		len_data--;
	data[ len_data ] = '\0';	// Makes sure the string finishes with a '\0'.

	#if NMFGPU_DEBUG_READ_FILE2
		printf("\tLen=%zu (%zu):'%s'.\n",len_data,strlen(data),data);
	#endif

	// Returns the last char read if requested.
	if ( last_char ) {
		if ( len_data * (!c) )
			c = (int) data[len_data-1];	// last character.
		*last_char = c;
	}

	// Adjusts memory size.
	*str = realloc( data, (len_data + 1) * sizeof(char) );	// +1 to keep the '\0'
	if ( ! *str ) {
		int const err = errno; fflush(stdout); errno = err;
		perror("\nrealloc(data)");
		fprintf(stderr,"Error in read_token().\n");
		len_data = 1;
		if ( last_char ) { *last_char = 0; }
		*str = NULL;
		free( data );
	}

	return len_data;

} // read_token

//////////////////////////////////////////////////

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
size_t tokenize( char *restrict str, char **restrict *restrict pstr, int delimiter )
{

	if ( ! ( (size_t) str * (size_t) pstr ) ) {	// (str == NULL) || (pstr == NULL)
		fflush( stdout );
		errno = EFAULT;
		if ( ! str ) { perror("\ntokenize(str)"); }
		if ( pstr ) { *pstr = NULL; } else { perror("\ntokenize(pstr)"); }
		return 0;
	}

	// ---------------------------

	size_t max_len_pstr = 128;	// Initial size for 128 tokens

	// Pointer to each token.
	char **restrict l_pstr = malloc( max_len_pstr * sizeof(char *) );
	if ( ! l_pstr ) {
		int const err = errno; fflush(stdout); errno = err;
		perror("\nmalloc(l_pstr)");
		fprintf(stderr,"Error in tokenize().\n");
		*pstr = NULL;
		return 0;
	}

	l_pstr[ 0 ] = str;		// First token
	size_t num_tokens = 1;

	char *ct = str;			// Starting address of current token.
	char *nt = NULL;		// Address of a new token.
	while ( (nt = strchr(ct,delimiter)) ) {

		*nt = '\0';	// Replaces delimiter by '\0'
		nt++;		// New token

		// Sets the vector of pointers

		// First, allocates more memory if necessary.
		if ( num_tokens == max_len_pstr ) {
			max_len_pstr *= 2;
			char **const tmp = realloc( l_pstr, max_len_pstr * sizeof(char *) );
			if ( ! tmp ) {
				int const err = errno; fflush(stdout); errno = err;
				perror("\nrealloc(l_pstr)");
				fprintf(stderr,"Error in tokenize().\n");
				free(l_pstr);
				*pstr = NULL;
				return 0;
			}
			l_pstr = tmp;
		}

		// Stores the address of the new token (nt).
		l_pstr[ num_tokens ] = nt;
		num_tokens++;

		// New token becomes the current one
		ct = nt;

	} // while

	// From here: (nt == NULL), and ct points to the last token.

	// Adjusts memory size.
	*pstr = realloc( l_pstr, num_tokens * sizeof(char *) );
	if ( ! (*pstr) ) {
		int const err = errno; fflush(stdout); errno = err;
		perror("\nrealloc(data)");
		fprintf(stderr,"Error in tokenize().\n");
		free( l_pstr );
		num_tokens = 0;
	}

	return num_tokens;

} // tokenize

//////////////////////////////////////////////////

/*
 * Searches for tokens and sets pointers to.
 *
 * WARNING: numtokens <= length( pstr )
 */
void retok( char const *restrict str, char const **restrict pstr, index_t numtokens )
{

	if ( (size_t) str * (size_t) pstr ) {	// ( str != NULL ) && ( pstr != NULL )

		char const *s = str;

		for ( index_t i = 0 ; i < numtokens ; i++ ) {
			pstr[i] = s;		// Sets pointer to current token.
			s += (strlen(s) + 1);	// Jumps to the next token.
		}

	}

} // retok

//////////////////////////////////////////////////

/*
 * Generates a label string in the form: "<tag_prefix><t><tag_suffix>", where t0 <= t < (t0+num_tags).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int generate_labels( char const *restrict tag_prefix, char const *restrict tag_suffix, index_t t0, index_t num_tags,
			struct tags *restrict labels )
{

	if ( t0 < 0 ) {
		fflush( stdout );
		errno = EINVAL;
		fprintf( stderr, "\ngenerate_labels( t0=%" PRI_IDX " ): %s\n", t0, strerror(errno) );
		return EXIT_FAILURE;
	}

	if ( num_tags <= 0 ) {
		fflush( stdout );
		errno = EINVAL;
		fprintf( stderr, "\ngenerate_labels( num_tags=%" PRI_IDX " ): %s\n", num_tags, strerror(errno) );
		return EXIT_FAILURE;
	}

	if ( ! labels ) {
		fflush( stdout );
		errno = EFAULT;
		perror( "\ngenerate_labels( labels ).\n" );
		return EXIT_FAILURE;
	}


	char const null_char = '\0';
	char const *tg_p = &null_char;
	char const *tg_s = &null_char;

	if ( tag_prefix )
		tg_p = tag_prefix;

	if ( tag_suffix )
		tg_s = tag_suffix;


	// Maximum length.
	size_t max_len_tokens = strlen( tg_p ) + strlen( tg_s ) + 1; // (+1 to include the '\0').

	// Adjusts 'max_len_tokens' according to the number of digits of the ending tag (t0+num_tags-1).
	{
		index_t tmp = t0 + num_tags - 2; // ending_tag - 1: The last '-1' might save one digit.
		index_t num_digits = 1;
		while ( tmp >= 10 ) {
			tmp /= 10;
			num_digits++;
		}
		max_len_tokens = (max_len_tokens + num_digits) * num_tags;
	}

	// Labels.
	char *restrict const tokens = malloc( max_len_tokens * sizeof(char) );	// Size will be adjusted later.
	if ( ! tokens ) {
		int const err = errno; fflush(stdout); errno = err;
		perror("\nmalloc(tokens)");
		fprintf(stderr,"Error in generate_labels().\n");
		return EXIT_FAILURE;
	}
	size_t len_tokens = 0;	// Current length for tokens.
	char *p_tokens = tokens;

	// Array of pointers to 'tokens'.
	char const **restrict const ptokens = malloc( num_tags * sizeof(char *) );
	if ( ! ptokens ) {
		int const err = errno; fflush(stdout); errno = err;
		perror("\nmalloc(ptokens)");
		fprintf(stderr,"Error in generate_labels().\n");
		free(tokens);
		return EXIT_FAILURE;
	}

	// Generates the tokens.
	for( index_t i = 0, t = t0; i < num_tags; i++, t++ ) {

		size_t const len = sprintf( p_tokens, "%s%" PRI_IDX "%s", tg_p, t, tg_s) + 1; // (+1 to include the '\0').

		ptokens[ i ] = p_tokens;

		// Points to place for next label.
		p_tokens += len;
		len_tokens += len;

	} // for 0 <= i < num_tags.

	// Adjusts memory used.
	char const *const l_tokens = realloc( tokens, len_tokens * sizeof(char) );
	if ( ! l_tokens ) {
		int const err = errno; fflush(stdout); errno = err;
		perror("\nrealloc(tokens)");
		fprintf(stderr,"Error in generate_labels().\n");
		free( ptokens ); free( tokens );
		return EXIT_FAILURE;
	}

	// If for any reason the address has changed, it is necessary to "retokenize" the string.
	if ( l_tokens != tokens )
		retok( l_tokens, ptokens, num_tags );

	// Finally, sets the outputs.
	(*labels) = NEW_TAGS( l_tokens, ptokens );

	return EXIT_SUCCESS;

} // generate_labels

//////////////////////////////////////////////////

/*
 * Cleans labels.
 */
void clean_labels( struct tags labels )
{

	if ( labels.ptokens )
		free((void *) labels.ptokens);

	if ( labels.tokens )
		free((void *) labels.tokens);

} // clean_labels

//////////////////////////////////////////////////

/*
 * Cleans all matrix labels (name, column headers and row labels).
 */
void clean_matrix_labels( struct matrix_labels ml )
{

	clean_labels( ml.labels );
	clean_labels( ml.headers );

	if ( ml.name )
		free((void *) ml.name);

} // clean_matrix_labels.

//////////////////////////////////////////////////
