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
 * Checks matrix dimensions.
 *
 * In verbose mode, shows matrix dimensions.
 *
 * ncols <= pitch.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_check_dimensions( char const *RESTRICT const function_name, index_t nrows, index_t ncols, index_t pitch, bool transpose, bool verbose );

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
 * Loads a matrix from an ASCII file.
 *
 * Skips name, headers and labels if "mt" is NULL.
 *
 * If (*matrix) is non-NULL, do not allocates memory but uses the supplied one.
 * WARNING: In such case, THERE MUST BE ENOUGH MEMORY ALREADY ALLOCATED.
 *
 * If input file does not have any tag, accepts both tab and space characters as delimiters.
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
int matrix_load_ascii( char const *RESTRICT filename, index_t nrows, index_t ncols, index_t pitch, bool real_data, bool hasname, bool hasheaders,
			bool haslabels, bool transpose, void *RESTRICT *RESTRICT matrix, struct matrix_tags_t *RESTRICT mt );

////////////////////////////////////////////////
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
int matrix_load_binary_verb( char const *RESTRICT filename, real *RESTRICT *RESTRICT matrix, index_t *RESTRICT nrows, index_t *RESTRICT ncols,
				index_t *RESTRICT pitch, struct matrix_tags_t *RESTRICT mt );

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
int matrix_load_binary( char const *RESTRICT filename, index_t nrows, index_t ncols, index_t pitch, bool transpose,
			real *RESTRICT *RESTRICT matrix, struct matrix_tags_t *RESTRICT mt );

////////////////////////////////////////////////

/*
 * Loads a matrix from a "native" binary file (i.e., with the native endiannes,
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
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_load_binary_native( char const *RESTRICT filename, void *RESTRICT *RESTRICT matrix, index_t *RESTRICT nrows, index_t *RESTRICT ncols,
				index_t *RESTRICT pitch, size_t data_size, bool verbose, struct matrix_tags_t *RESTRICT mt );

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
int matrix_save_ascii( char const *RESTRICT filename, void const *RESTRICT matrix, index_t nrows, index_t ncols, index_t pitch, bool real_data,
			bool transpose, bool append, struct matrix_tags_t const *RESTRICT mt );

////////////////////////////////////////////////

/*
 * Saves <nmatrices> <nrows>-by-<ncols> real-type matrices to a single ASCII-text file.
 *
 * Reads input matrices from "native"-binary files (i.e., with the compiled types for matrix data and dimensions).
 *
 * Uses the supplied tags (unless "mt" is NULL).
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
int matrix_save_combined_ascii( char const *RESTRICT filename, char const *RESTRICT input_pattern, char const *RESTRICT output_pattern,
				index_t nmatrices, index_t nrows, index_t ncols, struct matrix_tags_t const *RESTRICT mt );

////////////////////////////////////////////////

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
int matrix_save_binary( char const *RESTRICT filename, real const *RESTRICT matrix, index_t nrows, index_t ncols, index_t pitch, bool transpose,
			struct matrix_tags_t const *RESTRICT mt );

////////////////////////////////////////////////

/*
 * Saves a matrix to a "native" binary file (i.e., with the native endiannes,
 * and the compiled types for matrix data and dimensions; no file signature).
 *
 * Skips name, headers and labels if "mt" is NULL.
 *
 * WARNING:
 *	- For internal use only  (i.e., for temporary files).
 *	- NO ERROR-CHECKING IS PERFORMED (e.g., overflow, invalid values, etc).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int matrix_save_binary_native( char const *RESTRICT filename, void const *RESTRICT matrix, index_t nrows, index_t ncols, index_t pitch,
				size_t data_size, struct matrix_tags_t const *RESTRICT mt );

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
int matrix_save( char const *RESTRICT filename, index_t save_bin, real const *RESTRICT matrix, index_t nrows, index_t ncols, index_t pitch,
		bool transpose, struct matrix_tags_t const *RESTRICT mt, bool verbose );

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
int matrix_show( void const *RESTRICT matrix, index_t nrows, index_t ncols, index_t pitch, bool real_data, bool transpose, bool all_processes,
		struct matrix_tags_t const *RESTRICT mt );

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
 * Then, "pitch" it is updated to <nrows> rounded up to <memory_alignment>.
 *
 * WARNING:
 *	- Pointer "matrix" is ALWAYS CHANGED, even on error.
 *	- NO ERROR-CHECKING IS PERFORMED (e.g., overflow, invalid values...).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE
 */
int matrix_transpose_file( void *RESTRICT matrix, index_t *RESTRICT nrows, index_t *RESTRICT ncols, index_t *RESTRICT pitch, size_t data_size,
				struct matrix_tags_t *RESTRICT mt, char const *RESTRICT base_filename );

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
