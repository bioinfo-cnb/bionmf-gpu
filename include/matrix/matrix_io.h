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
 * matrix_io.h
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

#if ! NMFGPU_MATRIX_IO_H
#define NMFGPU_MATRIX_IO_H (1)

#include "matrix/matrix_io_routines.h"
#include "real_type.h"
#include "index_type.h"

#include <stdbool.h>

////////////////////////////////////////////////
////////////////////////////////////////////////

/* Selects the appropriate "restrict" keyword. */

#undef RESTRICT

#if defined(__CUDACC__)			/* CUDA source code */
	#define RESTRICT __restrict__
#else					/* C99 source code */
	#define RESTRICT restrict
#endif

/* Always process this header as C code, not C++. */
#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////
////////////////////////////////////////////////

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
int matrix_load_ascii_verb( char const *RESTRICT filename, bool numeric_hdrs, bool numeric_lbls, real *RESTRICT *RESTRICT const matrix,
				index_t *RESTRICT nrows, index_t *RESTRICT ncols, index_t *RESTRICT pitch, struct matrix_tags_t *RESTRICT mt );

////////////////////////////////////////////////

/*
 * Loads a real-type matrix from an ASCII file.
 *
 * Skips name, headers and labels if 'mt' is NULL.
 *
 * If (*matrix) is non-NULL, do not allocates memory but uses the supplied one.
 * WARNING: In such case, THERE MUST BE ENOUGH MEMORY ALREADY ALLOCATED.
 *
 * If input file does not have any tag, accepts both tab and space characters as delimiters.
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
int matrix_load_ascii( char const *RESTRICT filename, index_t nrows, index_t ncols, index_t pitch, bool hasname, bool hasheaders, bool haslabels,
			bool transpose, real *RESTRICT *RESTRICT matrix, struct matrix_tags_t *RESTRICT mt );

////////////////////////////////////////////////

/*
 * Loads a index_t-type matrix from an ASCII file.
 *
 * Skips name, headers and labels if 'mt' is NULL.
 *
 * If (*matrix) is non-NULL, do not allocates memory but uses the supplied one.
 * WARNING: In such case, THERE MUST BE ENOUGH MEMORY ALREADY ALLOCATED.
 *
 * If input file does not have any tag, accepts both tab and space characters as delimiters.
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
int matrix_int_load_ascii( char const *RESTRICT filename, index_t nrows, index_t ncols, index_t pitch, bool hasname, bool hasheaders,
			bool haslabels, bool transpose, index_t *RESTRICT *RESTRICT matrix, struct matrix_tags_t *RESTRICT mt );

////////////////////////////////////////////////
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
int matrix_load_binary_verb( char const *RESTRICT filename, real *RESTRICT *RESTRICT matrix, index_t *RESTRICT nrows, index_t *RESTRICT ncols,
				index_t *RESTRICT pitch, struct matrix_tags_t *RESTRICT mt );

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
int matrix_load_binary( char const *RESTRICT filename, index_t numrows, index_t numcols, index_t pitch, bool transpose,
			real *RESTRICT *RESTRICT matrix, struct matrix_tags_t *RESTRICT mt );

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
int matrix_load_binary_native( char const *RESTRICT filename, index_t starting_row, index_t rows_to_read, bool verbose, size_t data_size,
				void *RESTRICT *RESTRICT matrix, index_t *RESTRICT file_nrows, index_t *RESTRICT file_ncols,
				index_t *RESTRICT pitch, struct matrix_tags_t *RESTRICT mt );

////////////////////////////////////////////////

/*
 * Reads input matrix according to the selected file format
 *
 * is_bin: Reads output matrix from a binary file.
 *		== 0: Disabled. Reads the file as ASCII text.
 *		== 1: Uses "non-native" format (i.e., double-precision data, and "unsigned int" for dimensions).
 *		 > 1: Uses "native" or raw format (i.e., the compiled types for matrix data and dimensions).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_load( char const *RESTRICT filename, bool numeric_hdrs, bool numeric_lbls, index_t is_bin, real *RESTRICT *RESTRICT matrix,
		index_t *RESTRICT nrows, index_t *RESTRICT ncols, index_t *RESTRICT pitch, struct matrix_tags_t *RESTRICT mt );

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

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
int matrix_save_ascii( char const *RESTRICT filename, real const *RESTRICT matrix, index_t nrows, index_t ncols, bool transpose, bool append,
			struct matrix_tags_t const *RESTRICT mt, index_t padding );

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
int matrix_int_save_ascii( char const *RESTRICT filename, index_t const *RESTRICT matrix, index_t nrows, index_t ncols, bool transpose,
				bool append, struct matrix_tags_t const *RESTRICT mt, index_t padding );

////////////////////////////////////////////////

/*
 * Saves <nmatrices> <nrows>-by-<ncols> real-type matrices to a single ASCII-text file.
 * Reads input matrices from "native"-binary files (i.e., with the compiled types for matrix data and dimensions).
 * Uses the supplied tags (unless 'mt' is NULL).
 * nmatrices > 1
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_save_combined_ascii( char const *RESTRICT filename, char const *RESTRICT input_pattern, char const *RESTRICT output_pattern,
				index_t nmatrices, index_t nrows, index_t ncols, struct matrix_tags_t const *RESTRICT mt );

////////////////////////////////////////////////

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
int matrix_save_binary( char const *RESTRICT filename, real const *RESTRICT matrix, index_t nrows, index_t ncols, bool transpose,
			struct matrix_tags_t const *RESTRICT mt, index_t padding );

////////////////////////////////////////////////

/*
 * Saves a matrix to a "native" binary file (i.e., with the compiled types for matrix data and dimensions).
 * Skips name, headers and labels if 'mt' is NULL.
 *
 * WARNING:
 *	- For internal use only  (i.e., for temporary files).
 *	- NO ERROR-CHECKING IS PERFORMED (e.g., overflow, invalid values...).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_save_binary_native( char const *RESTRICT filename, void const *RESTRICT matrix, index_t nrows, index_t ncols, size_t data_size,
				struct matrix_tags_t const *RESTRICT mt );

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
int matrix_save( char const *RESTRICT filename, index_t save_bin, real const *RESTRICT matrix, index_t nrows, index_t ncols, bool transpose,
		struct matrix_tags_t const *RESTRICT mt, index_t padding, bool verbose );

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

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
int matrix_show( real const *RESTRICT matrix, index_t numrows, index_t numcols, index_t padding, bool transpose, bool shown_by_all,
		struct matrix_tags_t const *RESTRICT mt );

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
int matrix_int_show( index_t const *RESTRICT matrix, index_t numrows, index_t numcols, index_t padding, bool transpose, bool shown_by_all,
			struct matrix_tags_t const *RESTRICT mt );

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
int matrix_transpose_file( void *RESTRICT matrix, index_t *RESTRICT nrows, index_t *RESTRICT ncols, size_t data_size,
				char const *RESTRICT base_filename );

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Cleans name, headers, labels and matrix.
 */
void matrix_clean( void *RESTRICT matrix, struct matrix_tags_t mt );

////////////////////////////////////////////////
////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

#undef RESTRICT

////////////////////////////////////////////////
////////////////////////////////////////////////

#endif /* NMFGPU_MATRIX_IO_H */
