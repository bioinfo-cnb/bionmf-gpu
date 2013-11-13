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
 * matrix_io.h
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

#if ! NMFGPU_MATRIX_IO_H
#define NMFGPU_MATRIX_IO_H (1)

#include "index_type.h"
#include "real_type.h"
#include "matrix/matrix_io_routines.h"

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

/* Always process this header as C code, not C++. */
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

//////////////////////////////////////////////////
//////////////////////////////////////////////////

/*
 * Loads a matrix from an ASCII file.
 *
 * Detects automatically if matrix has name, column headers and/or row labels, as well as data delimiter (space or tab characters).
 * Outputs information messages.
 * Performs error checking.
 *
 * Both matrix dimensions must be >= 2.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_load_ascii_verb( char const *RESTRICT filename, bool numeric_hdrs, bool numeric_lbls, real *RESTRICT *RESTRICT const matrix,
				index_t *RESTRICT nrows, index_t *RESTRICT ncols, struct matrix_labels *RESTRICT ml );

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
int matrix_load_binary_verb( char const *RESTRICT filename, real *RESTRICT *RESTRICT matrix, index_t *RESTRICT nrows, index_t *RESTRICT ncols,
				struct matrix_labels *RESTRICT ml );

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
 * If input file does not have labels, accepts both tab and space characters as delimiters.
 *
 * If 'transpose' is 'true', transposes matrix in memory as follows:
 * - Matrix dimensions in memory: <ncols> rows, <nrows> columns.
 * - Matrix dimensions in file: <nrows> rows, <ncols> columns.
 * - Reads <ncols> column headers (set as ml->headers) and <nrows> row labels (set as ml->labels).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_load_ascii( char const *RESTRICT filename, index_t nrows, index_t ncols, bool hasname, bool hasheaders, bool haslabels,
			bool transpose, real *RESTRICT *RESTRICT matrix, struct matrix_labels *RESTRICT ml );

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
int matrix_load_binary( char const *RESTRICT filename, index_t numrows, index_t numcols, bool transpose, real *RESTRICT *RESTRICT matrix,
			struct matrix_labels *RESTRICT ml );

//////////////////////////////////////////////////

/*
 * Loads a matrix from a "native" binary file (i.e., with the compiled types for matrix data and dimensions).
 * Detects automatically if matrix has name, column headers and/or row labels, unless 'ml' is set to NULL.
 *
 * If 'matrix' is NULL, skips data matrix (just reads matrix dimensions).
 * Else, if in addition, (*matrix != NULL), do not allocates memory for the data matrix, but uses the supplied one.
 *
 * Reads <length> items, starting from the <offset>-th element, if these values are positive (they are
 * ignored otherwise). Skips data matrix if (offset + length) >= matrix_dimensions.
 *
 * WARNING:
 *	- For internal use only.
 *	- If *matrix is non-NULL, IT MUST HAVE ENOUGH MEMORY ALREADY ALLOCATED.
 *	- NO ERROR-CHECKING IS PERFORMED (e.g., overflow, invalid values....).
 *
 * Returns EXIT_SUCCESS (or EXIT_FAILURE if could not open filename).
 */
int matrix_load_binary_native( char const *RESTRICT filename, index_t offset, index_t length, real *RESTRICT *RESTRICT matrix,
				index_t *RESTRICT nrows, index_t *RESTRICT ncols, struct matrix_labels *RESTRICT ml );

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
int matrix_save_ascii( char const *RESTRICT filename, real const *RESTRICT matrix, index_t nrows, index_t ncols, bool transpose, bool append,
			struct matrix_labels const *RESTRICT ml, index_t padding );

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
int matrix_save_combined_ascii( char const *RESTRICT filename, char const *RESTRICT input_pattern, char const *RESTRICT output_pattern,
				index_t nmatrices, index_t nrows, index_t ncols, struct matrix_labels const *RESTRICT ml,
				char *RESTRICT filename_tmp );

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
int matrix_save_binary( char const *RESTRICT filename, real const *RESTRICT matrix, index_t nrows, index_t ncols, bool transpose,
			struct matrix_labels const *RESTRICT ml, index_t padding );

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
int matrix_save_binary_native( char const *RESTRICT filename, real const *RESTRICT matrix, index_t nrows, index_t ncols,
				struct matrix_labels const *RESTRICT ml );

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
int matrix_transpose_file( real *RESTRICT matrix, index_t *RESTRICT nrows, index_t *RESTRICT ncols, char const *RESTRICT const base_filename );

//////////////////////////////////////////////////

/*
 * Cleans name, headers, labels and matrix.
 *
 * WARNING: This method uses the regular free(3) function.
 */
void matrix_clean( real *RESTRICT matrix, struct matrix_labels ml );

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
int matrix_show( real const *RESTRICT matrix, index_t numrows, index_t numcols, index_t padding, bool transpose,
		struct matrix_labels const *RESTRICT ml );

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
int matrix_int_load_ascii( char const *RESTRICT filename, index_t nrows, index_t ncols, bool hasname, bool hasheaders, bool haslabels,
			index_t *RESTRICT matrix, struct matrix_labels *RESTRICT ml );

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
int matrix_int_load_binary( char const *RESTRICT filename, index_t *RESTRICT matrix, index_t nrows, index_t ncols,
				struct matrix_labels *RESTRICT ml );

//////////////////////////////////////////////////
//////////////////////////////////////////////////

/*
 * Saves an integer matrix to an ASCII-text file.
 * Skips name, headers and labels if 'ml' is set to NULL.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_int_save_ascii( char const *RESTRICT filename, index_t const *RESTRICT matrix, index_t nrows, index_t ncols,
				struct matrix_labels const *RESTRICT ml );

//////////////////////////////////////////////////
/*
 * Saves an integer matrix to a binary file.
 * Skips name, headers and labels if 'ml' is set to NULL.
 *
 * WARNING: For internal use only. No error checking is performed.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_int_save_binary( char const *RESTRICT filename, index_t const *RESTRICT matrix, index_t nrows, index_t ncols,
				struct matrix_labels const *RESTRICT ml );

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
int matrix_int_show( index_t const *RESTRICT matrix, index_t numrows, index_t numcols, index_t padding, bool transpose,
			struct matrix_labels const *RESTRICT ml );

//////////////////////////////////////////////////
//////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

//////////////////////////////////////////////////
//////////////////////////////////////////////////

#undef RESTRICT

//////////////////////////////////////////////////

#endif /* NMFGPU_MATRIX_IO_H */
