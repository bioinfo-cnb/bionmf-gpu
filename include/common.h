/************************************************************************
 *
 * NMF-mGPU - Non-negative Matrix Factorization on multi-GPU systems.
 *
 * Copyright (C) 2011-2015:
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
 * common.h
 *	Some generic definitions, constants, macros and functions used by NMF-mGPU.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Additional information:
 *		NMFGPU_VERBOSE_2: Shows the parameters in some routine calls.
 *
 *	Debug / Testing:
 *		NMFGPU_FIXED_INIT: Uses "random" values generated from the fixed seed: FIXED_SEED.
 *
 **********************************************************
 **********************************************************
 **********************************************************
 *
 * Data matrices:
 *	V (N rows, M columns): input matrix
 *	W (N,K): output matrix
 *	H (K,M): output matrix,
 * such that: V  ~  W * H.
 *
 * Arguments:
 *	Matrix V (and its dimensions)
 *	K: Factorization Rank
 *
 *
 * NOTE: In order to improve performance:
 *
 *	+ Matrix H is stored in memory as COLUMN-major (i.e., it is transposed).
 *
 *	+ All matrices include useless data for padding. Padded dimensions
 *	  are denoted with the 'p' character. For instance:
 *		Mp, which is equal to <M + padding>
 *		Kp, which is equal to <K + padding>.
 *
 *	  Data alignment is controlled by the global variable: memory_alignment.
 *
 *	  This leads to the following limits:
 *		- Maximum number of columns (padded or not): matrix_max_pitch.
 *		- Maximum number of rows: matrix_max_non_padded_dim.
 *		- Maximum number of items: matrix_max_num_items.
 *
 *	  All four GLOBAL variables must be initialized with the set_matrix_limits()
 *	  function.
 *
 ****************
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
 * Multi-GPU version:
 *
 * When the input matrix V is distributed among multiple devices each host thread processes
 * the following sets of rows and columns:
 *
 *	Vrow[ 1..NpP ][ 1..M ] <-- V[ bN..(bN+NpP) ][ 1..M ]	(i.e., NpP rows, starting from bN)
 *	Vcol[ 1..N ][ 1..MpP ] <-- V[ 1..N ][ bM..(bM+MpP) ]	(i.e., MpP columns, starting from bM)
 *
 * Such sets allow to update the corresponding rows and columns of W and H, respectively.
 *
 * Note that each host thread has a private full copy of matrices W and H, which must be synchronized
 * after being updated.
 *
 ****************
 *
 * Large input matrix (blockwise processing):
 *
 * If the input matrix (or the portion assigned to this device) is too large for the GPU memory,
 * it must be blockwise processed as follow:
 *
 *	d_Vrow[1..BLN][1..Mp] <-- Vrow[ offset..(offset + BLN) ][1..Mp]			(i.e., BLN <= NpP rows)
 *	d_Vcol[1..N][1..BLMp] <-- Vcol[1..N][ offset_Vcol..(offset_Vcol + BLMp) ]	(i.e., BLM <= MpP columns)
 *
 * Note that padded dimensions are denoted with the suffix 'p' (e.g., Mp, BLMp, etc).
 *
 * In any case, matrices W and H are fully loaded into the GPU memory.
 *
 * Information for blockwise processing is stored in two block_t structures (one for each dimension).
 * Such structures ('block_N' and 'block_M') are initialized in the init_block_conf() routine.
 *
 ****************
 *
 * Mapped Memory on integrated GPUs:
 *
 * On integrated systems, such as notebooks, where device memory and host memory are physically the
 * same (but disjoint regions), any data transfer between host and device memory is superfluous.
 * In such case, host memory is mapped into the address space of the device, and all transfer
 * operations are skipped. Memory for temporary buffers (e.g., d_WH or d_Aux) is also allocated
 * on the HOST and then mapped. This saves device memory, which is typically required for graphics/video
 * operations.
 *
 * This feature is disabled if NMFGPU_FORCE_BLOCKS is non-zero.
 *
 *********************************************************/

#if ! NMFGPU_COMMON_H
#define NMFGPU_COMMON_H (1)

#include "real_type.h"
#include "index_type.h"

#include <stdbool.h>

////////////////////////////////////////////////
////////////////////////////////////////////////

/* Data types */

// File formats:
typedef enum file_format {

	// ASCII-text file.
	ASCII_TEXT_FMT,

	/* "Non-native" binary file:
	 *	Little endian.
	 *	Double-precision data
	 *	32-bits unsigned integers for matrix dimensions.
	 *	File signature.
	 */
	NON_NATIVE_BINARY_FMT,

	/* "Native" binary file:
	 *	Native endiannes.
	 *	Compiled types for matrix data
	 *	Compiled signedness for matrix dimensions.
	 *	NO file signature.
	 */
	NATIVE_BINARY_FMT,

} file_fmt_t;


// Structure for arguments
struct input_arguments {

	char const *restrict filename;
	bool numeric_hdrs;		// Input matrix has numeric columns headers (ignored for binary files).
	bool numeric_lbls;		// Input matrix has numeric row labels (ignored for binary files).

	file_fmt_t input_file_fmt;	// Input file format.
	file_fmt_t output_file_fmt;	// Output file format.

	index_t k;			// Starting factorization rank.
	index_t kp;			// Padded factorization rank.
	index_t nIters;			// Maximum number of iterations per run.
	index_t niter_test_conv;	// Number of iterations before testing convergence.
	index_t stop_threshold;		// Stopping criterion.

	index_t gpu_device;		// GPU device ID (NMF_[m]GPU only).

	index_t idx_other_args;		// Index in argv[] with additional executable-specific arguments.
};

// ---------------------------------------------
// ---------------------------------------------

/* Global variables */

extern index_t process_id;		// Current process ID.
extern index_t num_processes;		// (Maximum) Number of processes on the system.
extern index_t num_act_processes;	// Number of "active" (i.e., not-idle) processes (<= num_processes).

// Matrix dimension limits (NOTE: they may be modified if the program is executed in a GPU device).
extern index_t memory_alignment;		// Data alignment on memory.
extern size_t matrix_max_num_items;		// Maximum number of items in a matrix.
extern index_t matrix_max_pitch;		// Maximum multiple of <memory_alignment>.
extern index_t matrix_max_non_padded_dim;	// Maximum non-padded dimension.

// Matrix dimensions:
extern index_t N;	// Number of rows of input matrix V.
extern index_t M;	// Number of columns of input matrix V.
extern index_t K;	// Factorization rank.

// Dimensions for multi-process version:
extern index_t NpP;	// Number of rows of V assigned to this process (NpP <= N).
extern index_t MpP;	// Number of columns of V assigned to this process (MpP <= M).
extern index_t bN;	// Starting row ((bN + NpP) <= N).
extern index_t bM;	// Starting column ((bM + MpPp) <= Mp).

// Padded dimensions:
extern index_t Mp;	// <M> rounded up to the next multiple of <memory_alignment>.
extern index_t Kp;	// <K> rounded up to the next multiple of <memory_alignment>.
extern index_t MpPp;	// <MpP> rounded up to the next multiple of <memory_alignment> (MpPp <= Mp).

// Classification vectors.
extern index_t *restrict classification;
extern index_t *restrict last_classification;

// HOST matrices
extern real *restrict W;
extern real *restrict H;
extern real *restrict Vcol;	// Block of NpP rows from input matrix V.
extern real *Vrow;		// Block of MpP columns from input matrix V.

// File extensions for output files.
extern char const *restrict const file_extension[ 3 ];


////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Prints the given message composed by the format string "fmt" and the subsequent
 * arguments, if any.
 *
 * If "all_processes" is 'true', the message is printed by all existing processes.
 * In addition, if (num_processes > 1), the process ID is also printed.
 *
 * The string is always printed to the standard output stream ('stdout').
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int print_message( bool all_processes, char const *restrict const fmt, ... );

////////////////////////////////////////////////

/*
 * Prints the given message, composed by the format string "fmt" and the subsequent
 * arguments, if any.
 *
 * This method is intended for successive portions of a message that was previously
 * printed, so it will never be prefixed by a newline nor the process ID.
 *
 * If "all_processes" is 'true', the message is printed by all existing processes.
 *
 * The string is always printed to the standard output stream ('stdout').
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int append_printed_message( bool all_processes, char const *restrict const fmt, ... );

////////////////////////////////////////////////

/*
 * Prints the given error message, composed by the format string "fmt" and the
 * subsequent arguments, if any.
 *
 * If "all_processes" is 'true', the message is printed by all existing processes.
 * In addition, if (num_processes > 1), the process ID is also printed.
 *
 * The string is always printed to the standard error stream ('stderr'). The
 * standard output stream ('stdout') is previously flushed for all processes.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int print_error( bool all_processes, char const *restrict const fmt, ... );

////////////////////////////////////////////////

/*
 * Prints the given error message, composed by the format string "fmt" and the
 * subsequent arguments, if any.
 *
 * This method is intended for successive portions of a message that was previously
 * printed, so it will never be prefixed by a newline nor the process ID.
 *
 * If "all_processes" is 'true', the message is printed by all existing processes.
 *
 * The string is always printed to the standard error stream ('stderr'). The
 * standard output stream ('stdout') is previously flushed for all processes.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int append_printed_error( bool all_processes, char const *restrict const fmt, ... );

////////////////////////////////////////////////

/*
 * Prints the given error message, composed by the format string "fmt" and the
 * subsequent arguments, if any.
 *
 * If "all_processes" is 'true', the message is printed by all existing processes.
 * In addition, if (num_processes > 1), the process ID is also printed.
 *
 * The string is always printed to the standard error stream ('stderr'). The
 * standard output stream ('stdout') is previously flushed for all processes.
 *
 * Finally, if errnum is non-zero, this function behaves similar to perror(3).
 * That is, it appends to the message a colon, the string given by strerror(errnum)
 * and a newline character. Otherwise, it just prints a newline.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int print_errnum( bool all_processes, int errnum, char const *restrict const fmt, ... );

////////////////////////////////////////////////

/*
 * Prints the given error message, composed by the format string "fmt" and the
 * subsequent arguments, if any.
 *
 * This method is intended for successive portions of a message that was previously
 * printed, so it will never be prefixed by a newline nor the process ID.
 *
 * If "all_processes" is 'true', the message is printed by all existing processes.
 *
 * The string is always printed to the standard error stream ('stderr'). The
 * standard output stream ('stdout') is previously flushed for all processes.
 *
 * Finally, if errnum is non-zero, this function behaves similar to perror(3).
 * That is, it appends to the message a colon, the string given by strerror(errnum)
 * and a newline character. Otherwise, it just prints a newline.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int append_printed_errnum( bool all_processes, int errnum, char const *restrict const fmt, ... );

////////////////////////////////////////////////

/*
 * Flushes the buffer associated to the standard output stream (stdout).
 *
 * If "permanently" is 'true', the buffer is also disabled.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int flush_output( bool permanently );

////////////////////////////////////////////////

/*
 * Prints all arguments regarding the file formats.
 *
 * This message is printed by process 0 only.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int help_file_formats( void );

////////////////////////////////////////////////

/*
 * Prints all arguments regarding the NMF algorithm (e.g., factorization rank, number of iterations, etc).
 *
 * This message is printed by process 0 only.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int help_nmf( void );

////////////////////////////////////////////////

/*
 * Prints all arguments regarding the main program <execname>.
 *
 * This message is printed by process 0 only.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int print_nmf_gpu_help( char const *restrict const execname );

////////////////////////////////////////////////

/*
 * Sets the maximum dimensions and number of items, for the given data
 * alignment and dimension limit.
 *
 * The resulting values are stored in the global variables "matrix_max_pitch",
 * "matrix_max_non_padded_dim" and "matrix_max_num_items". In addition, the first
 * and third variables are rounded down to a multiple of the given data alignment.
 *
 * data_alignment:
 *		If set to '0', uses the default padding, <DEFAULT_MEMORY_ALIGNMENT>.
 *		If set to '1', disables padding.
 *		Otherwise, it must be a positive value expressed in number of
 *		real-type items (not in bytes).
 *
 * max_dimension:
 *		If greater than or equal to the resulting data alignment (i.e.,
 *		either <data_alignment> or <DEFAULT_MEMORY_ALIGNMENT>), uses the
 *		given value as an additional upper limit for matrix dimensions.
 *		That is, the result will be the minimum between <max_dimension>,
 *		and the value calculated from the data alignment and the maximum
 *		number of items.
 *		On "matrix_max_pitch", the result is subsequently rounded down
 *		to a multiple of the data alignment.
 *		It is ignored if set to a non-negative value less than
 *		<data_alignment>.
 *
 * max_nitems:
 *		If set to a positive value, forces the maximum number of items
 *		for a data matrix. Please note that this value HAS PRECEDENCE
 *		over the resulting maximum dimensions. However, it is IGNORED
 *		if set to a value less than the resulting data alignment.
 *		Finally, it must be expressed in number of items, not in bytes.
 *
 * WARNING:
 *	This function must be called *BEFORE* loading any input matrix. Otherwise,
 *	no padding will be set.
 *
 * Returns EXIT_SUCCESS, or EXIT_FAILURE on negative input values.
 */
int set_matrix_limits( index_t data_alignment, index_t max_dimension, size_t max_nitems );

////////////////////////////////////////////////

/*
 * Sets the maximum dimensions and number of items, for the DEFAULT
 * data alignment (<DEFAULT_MEMORY_ALIGNMENT>).
 *
 * The resulting values are stored in the global variables "matrix_max_pitch",
 * "matrix_max_non_padded_dim" and "matrix_max_num_items". In addition, the first
 * and third variables are rounded down to a multiple of the given data alignment.
 *
 * Returns EXIT_SUCCESS.
 */
int set_default_matrix_limits( void );

////////////////////////////////////////////////

/*
 * Computes the padded dimension of "dim". That is, the next multiple of <memory_alignment>.
 *
 * Computes the padded dimension of "dim".
 *
 * Returns <pitch>, such that:
 *	dim <= pitch
 *	pitch is a multiple of <memory_alignment>
 *
 * WARNING:
 *	Global variable "memory_alignment" must have been properly initialized.
 */
index_t get_padding( index_t dim );

////////////////////////////////////////////////

/*
 * Checks all arguments.
 *
 * Sets 'help' to 'true' if help message was requested ('-h' or '-H' options).
 *
 * Error messages will be shown by process 0 only.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int check_arguments( int argc, char const *restrict *restrict argv, bool *restrict help, struct input_arguments *restrict arguments );

////////////////////////////////////////////////

/*
 * Computes the lowest power-of-two >= x.
 *
 * WARNING:
 *	x <= floor(SIZE_MAX / 2)
 *
 * Returns:
 *	<x>, if it is already a power of two, or x == 0
 *	The next power of two, if "x" is not a power of two.
 */
size_t next_power_2( size_t x );

////////////////////////////////////////////////

/*
 * Computes the highest power-of-two <= x.
 *
 * Returns:
 *	<x>, if it is already a power of two, or x == 0
 *	The previous power of two, if "x" is not a power of two.
 */
size_t prev_power_2( size_t x );

////////////////////////////////////////////////

/*
 * Gets the difference between classification and last_classification vectors
 */
size_t get_difference( index_t const *restrict classification, index_t const *restrict last_classification, index_t m );

////////////////////////////////////////////////

/*
 * Retrieves a "random" value that can be used as seed.
 *
 * If NMFGPU_FIXED_INIT is non-zero, returns <FIXED_SEED>.
 */
index_t get_seed( void );

////////////////////////////////////////////////
////////////////////////////////////////////////

#endif /* NMFGPU_COMMON_H */
