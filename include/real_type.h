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
 * real_type.h
 *	Definition of the 'real' data type.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Data type:
 *		NMFGPU_SINGLE_PREC: Makes use of single-precision data (i.e., 'float').
 *
 * WARNING:
 *	- Requires support for ISO-C99 standard. It can be enabled with 'gcc -std=c99'.
 *
 *********************************************************/

#if ! NMFGPU_REAL_TYPE_H
#define NMFGPU_REAL_TYPE_H (1)

#include <math.h>	/* sqrt(f) */
#include <string.h>	/* strtof, strtod, */
#include <float.h>	/* DBL_EPSILON, FLT_MAX, etc */

////////////////////////////////////////////////
////////////////////////////////////////////////

/* Data-type definitions */

// 'Real' data type.
typedef
#if NMFGPU_SINGLE_PREC	/* Single precision */
	float
#else	/* Double precision */
	double
#endif
		real;

// ---------------------------------------------
// ---------------------------------------------

/* Constants */

// Machine epsilon (a.k.a. unit roundoff or EPS)
#ifndef R_EPSILON
	#if NMFGPU_SINGLE_PREC		/* Single precision */
		#define R_EPSILON ( FLT_EPSILON )
	#else				/* Double precision */
		#define R_EPSILON ( DBL_EPSILON )
	#endif
#endif

// Maximum FINITE value.
#ifndef R_MAX
	#if NMFGPU_SINGLE_PREC		/* Single precision */
		#define R_MAX ( FLT_MAX )
	#else				/* Double precision */
		#define R_MAX ( DBL_MAX )
	#endif
#endif

// Minimum normalized positive value.
#ifndef R_MIN
	#if NMFGPU_SINGLE_PREC		/* Single precision */
		#define R_MIN ( FLT_MIN )
	#else				/* Double precision */
		#define R_MIN ( DBL_MIN )
	#endif
#endif

/* Conversion format from string used in the scanf(3) family of functions.
 * WARNING: Please, do NOT set to longer than 6 characters.
 */
#ifndef SCNgREAL
	#if NMFGPU_SINGLE_PREC
		#define SCNgREAL "g"
	#else
		#define SCNgREAL "lg"
	#endif
#endif

// ---------------------------------------------
// ---------------------------------------------

/* Macro Functions */

// Converts constant values to the right data type (i.e., appends an 'f' on single-precision mode).
#ifndef REAL_C
	#if NMFGPU_SINGLE_PREC	/* Single precision */
		#define REAL_C(c) (c ## f)
	#else			/* Double precision */
		#define REAL_C
	#endif
#endif

/* Converts constants to the right data type (i.e., appends an 'f' on single-precision mode).
 * Useful when the argument is not a value but a macro constant that must also be expanded.
 */
#ifndef REAL_MC
	#define REAL_MC(m) ( REAL_C( m ) )
#endif

// ---------------------------------------------
// ---------------------------------------------

/* C HOST Functions */

// Function to convert from string to real data type.
#ifndef STRTOREAL
	#if NMFGPU_SINGLE_PREC
		#if (_XOPEN_SOURCE >= 600) || defined(_ISOC99_SOURCE) || (_POSIX_C_SOURCE >= 200112L)
			#define STRTOREAL strtof
		#else
			#define STRTOREAL(nptr,endptr) ( (float) strtod((nptr),(endptr)) )
		#endif
	#else
		#define STRTOREAL strtod
	#endif
#endif

// Square root function
#ifndef SQRTR
	#if NMFGPU_SINGLE_PREC
		#if (_XOPEN_SOURCE >= 600) || defined(_ISOC99_SOURCE) || (_POSIX_C_SOURCE >= 200112L) || \
			defined(_BSD_SOURCE) || defined(_SVID_SOURCE)
			#define SQRTR sqrtf
		#else
			#define SQRTR(x) ( (float) sqrt( (double) (x) ) )
		#endif
	#else
		#define SQRTR sqrt
	#endif
#endif

////////////////////////////////////////////////
////////////////////////////////////////////////

#endif /* NMFGPU_REAL_TYPE_H */
