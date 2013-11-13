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
 * WARNING:
 *	- Requires support for ISO-C99 standard. It can be enabled with 'gcc -std=c99'.
 *	- Some (optional) useful features for processing single-precision data are only available when '_GNU_SOURCE' is defined.
 *	  Such features can be enabled with 'gcc -D_GNU_SOURCE'
 *
 *********************************************************/

#if ! NMFGPU_INDEX_TYPE_H
#define NMFGPU_INDEX_TYPE_H (1)

#include <limits.h>	/* [U]INT_MAX */
/* WARNING: Do NOT include <limits.h> if using ICC v10
 * Instead, please use the following definitions:
 *
 * #ifndef INT_MAX
 *	#define INT_MAX (2147483647)
 * #endif
 * #ifndef UINT_MAX
 *	#define UINT_MAX (4294967295u)
 * #endif
 */

///////////////////////////////////////////////////////

/* Data-type definitions */

// 'Index' data type
#if NMFGPU_UINDEX	/* Unsigned indexes */
	typedef unsigned int index_t;
#else
	typedef int index_t;
#endif

// ---------------------------------------------

/* Constants */

// Maximum value
#ifndef IDX_MAX
	#if NMFGPU_UINDEX
		#define IDX_MAX ( UINT_MAX )
	#else
		#define IDX_MAX (  INT_MAX )
	#endif
#endif

// Minimum value
#ifndef IDX_MIN
	#if NMFGPU_UINDEX	/* Unsigned indexes */
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
		#define PRI_IDX "i"
	#endif
#endif

// ---------------------------------------------

/* Macro Functions */

// Converts constant values to the right data type (i.e., appends an 'u' on unsigned indexes).
#ifndef INDEX_C
	#if NMFGPU_UINDEX	/* Unsigned indexes */
		#define INDEX_C(c) (c ## u)
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
#ifndef MIN
	#define MIN(a,b) ( ( (a) <= (b) ) ? (a) : (b) )
#endif
#ifndef MIN3
	#define MIN3(a,b,c) ( ( (a) <= (b) ) ? (MIN((a),(c))) : (MIN((b),(c))) )
#endif
#ifndef MAX
	#define MAX(a,b) ( ( (a) >= (b) ) ? (a) : (b) )
#endif

/*
 * Computes the highest power of 2 <= x.
 * Returns the same value (x) if it is already a power of 2, or is zero.
 */
#ifndef PREV_POWER_2
	#define PREV_POWER_2(x) \
		( ( (x) & ((x)-1) ) ? \
			({ \
				index_t i = (x); \
				for ( index_t n = 0, b = 1 ; n <= (index_t) sizeof(index_t) ; n++, b <<= 1 ) { i |= ( i >> b ); } \
				i -= (i >> 1); \
			}) : \
			(x) \
		)
#endif

// ---------------------------------------------

/* CUDA DEVICE Functions */

#if __CUDA_ARCH__

	/*
	 * On Compute Capability 1.x, a 24-bits integer product is faster than the regular 32-bits operand: '*'.
	 */
	#ifndef IMUL
		#if __CUDA_ARCH__ < 200	/* Compute Capability 1.x */
			#ifdef NMFGPU_UINDEX
				#define IMUL __umul24
			#else
				#define IMUL __mul24
			#endif
		#else			/* Compute Capability >= 2.0 */
			// Just uses the regular operand.
			#define IMUL(x,y) ( (x) * (y) )
		#endif
	#endif

#endif /* __CUDA_ARCH__ */

///////////////////////////////////////////////////////

#endif /* NMFGPU_INDEX_TYPE_H */
