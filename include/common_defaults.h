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
 * common_defaults.h
 *	Some generic definitions, constants, macros and functions used by common.c.
 *
 *********************************************************/

#if ! NMFGPU_COMMON_DEFAULTS_H
#define NMFGPU_COMMON_DEFAULTS_H (1)

#include "real_type.h"
#include "index_type.h"

#include <stdlib.h>

////////////////////////////////////////////////
////////////////////////////////////////////////

/* Constants */

// Fixed seed for the random-values generator.
#ifndef FIXED_SEED
	#define FIXED_SEED ( INDEX_C(3) )
#endif

// ---------------------------------------------

// File extension for ASCII-text files.
#if ! TEXT_FILE_EXT
	#define TEXT_FILE_EXT ".txt"
#endif

// File extension for non-native binary files.
#if ! NON_NATIVE_BIN_FILE_EXT
	#define NON_NATIVE_BIN_FILE_EXT ".dat"
#endif

// File extension for native binary files.
#if ! NATIVE_BIN_FILE_EXT
	#define NATIVE_BIN_FILE_EXT ".native.dat"
#endif

// ---------------------------------------------

// Default values of some input parameters.
#ifndef DEFAULT_K
	#define DEFAULT_K ( INDEX_C(2) )
#endif

#ifndef DEFAULT_NITERS
	#define DEFAULT_NITERS ( INDEX_C(2000) )
#endif

#ifndef DEFAULT_NITER_CONV
	#define DEFAULT_NITER_CONV ( INDEX_C(10) )
#endif

#ifndef DEFAULT_STOP_THRESHOLD
	#define DEFAULT_STOP_THRESHOLD ( INDEX_C(40) )
#endif

#ifndef DEFAULT_GPU_DEVICE
	#define DEFAULT_GPU_DEVICE ( INDEX_C(0) )
#endif

// ---------------------------------------------

// Default alignment for data on memory.
#ifndef DEFAULT_MEMORY_ALIGNMENT		/* 64 bytes, expressed in real-type items. */
	#define DEFAULT_MEMORY_ALIGNMENT ( (index_t) ( 64 / sizeof(real) ) )
#endif

////////////////////////////////////////////////
////////////////////////////////////////////////

#endif /* NMFGPU_COMMON_DEFAULTS_H */
