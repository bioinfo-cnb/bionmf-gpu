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
 * index_type.h
 *	Definition of the 'index' data type.
 *
 * NOTE: The following macro constant(s) can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Data type:
 *		NMFGPU_UINDEX: Makes use of UNSIGNED integers for index type.
 *				Otherwise, uses signed integers.
 *
 *		NMFGPU_MPI: Defines some MPI-related constants.
 *
 *********************************************************/

#if ! NMFGPU_INDEX_TYPE_H
#define NMFGPU_INDEX_TYPE_H (1)

#if NMFGPU_MPI
	#include <mpi.h> /* MPI_UNSIGNED, MPI_INT */
#endif

#include <limits.h>	/* [U]INT_MAX */

////////////////////////////////////////////////
////////////////////////////////////////////////

/* Data-type definitions */

/* "Index" data type
 *
 * NOTE: UNSIGNED integers may produce faster code.
 */
#if NMFGPU_UINDEX	/* Unsigned indexes */
	typedef unsigned int index_t;
#else			/* Signed indexes */
	typedef int index_t;
#endif

// Index-type data on MPI
#if NMFGPU_MPI && (! defined(NMFGPU_MPI_INDEX_T))
	#if NMFGPU_UINDEX		/* Unsigned indexes */
		#define NMFGPU_MPI_INDEX_T MPI_UNSIGNED
	#else				/* Signed indexes */
		#define NMFGPU_MPI_INDEX_T MPI_INT
	#endif
#endif

// ---------------------------------------------
// ---------------------------------------------

/* Constants */

// Generic index_t-related constants

// Maximum value.
#ifndef IDX_MAX
	#if NMFGPU_UINDEX	/* Unsigned indexes */
		#define IDX_MAX ( UINT_MAX )
	#else
		#define IDX_MAX (  INT_MAX )
	#endif
#endif

// Minimum value
#ifndef IDX_MIN
	#if NMFGPU_UINDEX
		#define IDX_MIN ( INDEX_C(0) )
	#else
		#define IDX_MIN ( INT_MIN )
	#endif
#endif

// Conversion format to string for the printf(3) family of functions.
#ifndef PRI_IDX
	#if NMFGPU_UINDEX
		#define PRI_IDX "u"
	#else
		#define PRI_IDX "d"
	#endif
#endif

/* Conversion format from string for the scanf(3) family of functions.
 * WARNING: Please, do NOT set to longer than 6 characters.
 */
#ifndef SCN_IDX
	#if NMFGPU_UINDEX
		#define SCN_IDX "u"
	#else
		#define SCN_IDX "d"
	#endif
#endif

// ---------------------------------------------

// gpu_size_t-related constants

// Maximum value
#ifndef GPUSIZE_MAX
	#define GPUSIZE_MAX ( IDX_MAX )
#endif

// ---------------------------------------------
// ---------------------------------------------

/* Macro Functions */

// Converts constant values to the right data type (i.e., appends an 'u' on unsigned indexes).
#ifndef INDEX_C
	#if NMFGPU_UINDEX	/* Unsigned indexes */
		#define INDEX_C(c) (c ## U)
	#else			/* Signed indexes */
		#define INDEX_C
	#endif
#endif

/* Converts constants to the right data type (i.e., appends an 'u' on unsigned indexes).
 * Useful when the argument is not a value but a macro constant that must also be expanded.
 */
#ifndef INDEX_MC
	#define INDEX_MC(m) ( INDEX_C( (m) ) )
#endif


// Minimum and maximum values.
#undef MAX
#define MAX(a,b) ( ( (a) >= (b) ) ? (a) : (b) )

#undef MIN
#define MIN(a,b) ( ( (a) <= (b) ) ? (a) : (b) )

#undef MIN3
#define MIN3(a,b,c) ( MIN( (MIN((a),(b))), (c)) )

////////////////////////////////////////////////
////////////////////////////////////////////////

#endif /* NMFGPU_INDEX_TYPE_H */
