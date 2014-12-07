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
 * GPU_kernels.cu
 *	Kernel code to be executed on the device (GPU).
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Data type:
 *		NMFGPU_SINGLE_PREC: Makes use of single-precision data (i.e., 'float').
 *
 **********************************************************
 *
 * NOTE: In order to improve performance:
 *
 *	+ All matrices include useless data for padding. It is sometimes denoted as 'pitch'.
 *
 *********************************************************/

#include "GPU_kernels.cuh"

////////////////////////////////////////////////
////////////////////////////////////////////////

/* Data types */

/*
 * Floating-point data on GPU shared memory:
 *
 * On devices of Compute Capability 1.x, accesses to 'double'-type data
 * are compiled into two separated 32-bit requests with a stride of two.
 * This results in a 2-way bank conflict.
 *
 * To avoid this, (64-bits) doubles are stored as two independent
 * (32-bits) integers on different regions of the shared memory.
 * Every load/store operation consists, then, in a "gather"/"scatter"
 * process.
 *
 * In order to hide the resulting different types (and number) of pointer
 * to shared memory, a new data type is defined below. Note that it will
 * be used just for local variables in kernel code. The shared memory
 * will still being a simple array of real- or int-type data.
 */
#if ( __CUDA_ARCH__ < 200 ) && ( ! NMFGPU_SINGLE_PREC )

	/* Double-precision data on Compute Capability 1.x */
	typedef struct rp2shm {
		int *__restrict__ hi;	// Pointer to an 'int' containing the higher 32 bits of a double value.
		int *lo;		// Pointer to an 'int' containing the lower  32 bits of a double value.
	} rp2shm_t;

#else	/* single-precision data <OR> Compute Capability >= 2.0 */

	typedef real * rp2shm_t;

#endif

// ---------------------------------------------
// ---------------------------------------------

/* DEVICE-ONLY GLOBAL Variables */

#if (! __CUDA_ARCH__) || (__CUDA_ARCH__ >= 120)	/* Compute Capability >= 1.2 */

	// HACK: Trick to avoid useless warnings...
	#if __CUDA_ARCH__ >= 120
		static
	#else
		extern
	#endif

		/* Global variable used in kernel_reduce_to_row().
		 *	Please see the threadFenceReduction example in CUDA Samples for details.
		 */
		__device__ unsigned int retirement_counter = 0U;

#endif

// ---------------------------------------------
// ---------------------------------------------

/* HOST-ONLY GLOBAL variables */

extern index_t computeCapability;		// Compute Capability (major).

////////////////////////////////////////////////
////////////////////////////////////////////////

#if __CUDA_ARCH__	/* Helps to prevent 'unused-function' warning messages on cudafe++, when ! __CUDA_ARCH__ */

/*
 * Computes the maximum of two floating-point values.
 *
 * Returns:
 *	Nan, if both arguments are NaN.
 *	The numeric argument, if the other is NaN.
 *	The maximum value, if both arguments are numeric.
 */
static __device__ real device_fmax( real valA, real valB )
{

	real max_val = REAL_C( 0.0 );

	#if __CUDA_ARCH__	/* Reduces work on cudafe(++) when looking for HOST code.*/

		#if NMFGPU_SINGLE_PREC

			/* Single Precision */
			max_val = fmaxf( valA, valB );

		#else	/* Double precision */
			max_val = fmax( valA, valB );

		#endif

	#endif

	return max_val;

} // device_fmax

#endif	/* __CUDA_ARCH__ */

////////////////////////////////////////////////

#if __CUDA_ARCH__	/* Helps to prevent 'unused-function' warning messages on cudafe++, when ! __CUDA_ARCH__ */

/*
 * Divides two floating-point values: valA / valB
 *
 * On single-precision data, a faster (but less-precise) function may be employed,
 * if the compilation flags '-use_fast_math' and '-prec-div', were specified.
 *
 * WARNING:
 *	valB != 0
 *	No error checking is performed.
 */
static __device__ real device_fdiv( real valA, real valB )
{

	real div_val = REAL_C( 0.0 );

	#if __CUDA_ARCH__	/* Reduces work on cudafe(++) when looking for HOST code.*/

		#if NMFGPU_SINGLE_PREC

			/* Single Precision:
			 *	This operation can be replaced by a faster less-precise
			 *	function if the flags '-use_fast_math' and '-prec-div'
			 *	were specified.
			 */
			div_val = fdividef( valA, valB );

		#else	/* Double precision: just use the regular operator. */
			div_val = valA / valB;

		#endif

	#endif

	return div_val;

} // device_fdiv

#endif	/* __CUDA_ARCH__ */

////////////////////////////////////////////////

/*
 * Non-negative integers product:
 *
 * fast_prod:	'True' to perform a (non-negative) integer product in "fast" mode.
 *
 *		On Compute Capability 1.x, it is a 24-bits product, which
 *		IGNORES the higher 8 bits of both (32-bits) input operands, and
 *		returns the lower 32 bits of the 48-bits result.
 *		ATTENTION: Both input operands are processed as UNSIGNED, so do
 *		NOT use this function with negative values.
 *
 *		In contrast, in "extended" mode, the integer product is performed
 *		with the regular operator "*", which on Compute Capability 1.x
 *		is compiled into multiple assembler instructions. However, it
 *		is required when any of the operands is greater than or equal
 *		to 2**24. In this case, the signedness is honored.
 *
 *		On Compute Capability >= 2.0, this flag is IGNORED, since "*"
 *		already performs the fastest operation.
 */
template < bool fast_prod >
static __device__ index_t idxmul( index_t a, index_t b )
{

	index_t p = INDEX_C( 0 );

	#if __CUDA_ARCH__	/* Reduces work on cudafe(++) when looking for HOST code.*/

	// 24-bits Integer product, on Compute Capability 1.x only
	#if __CUDA_ARCH__ < 200
		bool const idxmul24 = fast_prod;
	#else
		bool const idxmul24 = false;
	#endif

	// ------------------------------------------------

	if ( idxmul24 )
		p = __umul24( a, b );	// ATTENTION: Do NOT use this function with negative values.
	else
		p = a * b;

	#endif /* __CUDA_ARCH__ */

	return p;

} // idxmul

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

#if __CUDA_ARCH__	/* Helps to prevent 'unused-function' warning messages on cudafe++, when ! __CUDA_ARCH__ */

/*
 * Returns a new pointer to shared memory, starting at address <pshm>.
 *
 * length: Number of items covered by the new pointer (i.e., the array length).
 *	   Actually, it is required only for double-precision data on Compute
 *	   Capability 1.x, and will be ignored otherwise.
 */
static __device__ rp2shm_t new_rp2shm( void *__restrict__ pshm, index_t const length )
{

	rp2shm_t rp2shm;

	#if ( __CUDA_ARCH__ < 200 ) && ( ! NMFGPU_SINGLE_PREC )

		/* Double-precision data on Compute Capability 1.x */
		rp2shm.hi = (int *) pshm;
		rp2shm.lo = &rp2shm.hi[ length ];

	#else	/* Single-precision data <OR> Compute Capability >= 2.0 */
		rp2shm = (rp2shm_t) pshm;
	#endif

	return rp2shm;

} // new_rp2shm

#endif	/* __CUDA_ARCH__ */

////////////////////////////////////////////////

/*
 * Loads a value from the shared-memory area pointed to by "rp2shm", with the
 * given index. That is, returns the equivalent to '&rp2shm[ index ]'.
 *
 * volatile_mode: If 'true', loads the value from a volatile pointer. That is,
 *		  using a real load instruction, rather than reusing some
 *		  register. If set to 'false', lets the compiler to decide.
 *
 * Returns a real-type value.
 */
template < bool volatile_mode >
static __device__ real load_from_rp2shm( rp2shm_t rp2shm, index_t index )
{
	real val = REAL_C( 0.0 );

	#if __CUDA_ARCH__	/* Reduces work on cudafe(++) when looking for HOST code.*/

		#if ( __CUDA_ARCH__ < 200 ) && ( ! NMFGPU_SINGLE_PREC )

			/* Double-precision data on Compute Capability 1.x */
			int val_hi, val_lo;

			if ( volatile_mode ) {
				int volatile *const vp_hi = rp2shm.hi;
				int volatile *const vp_lo = rp2shm.lo;
				val_hi = vp_hi[ index ];
				val_lo = vp_lo[ index ];

			} else {
				val_hi = rp2shm.hi[ index ];
				val_lo = rp2shm.lo[ index ];
			}

			val = __hiloint2double( val_hi, val_lo );

		#else	/* Single-precision data <OR> Compute Capability >= 2.0 */

			if ( volatile_mode ) {
				real volatile *const vp_val = (real *) rp2shm;
				val = vp_val[ index ];

			} else
				val = rp2shm[ index ];

		#endif

	#endif

	return val;

} // load_from_rp2shm

////////////////////////////////////////////////

/*
 * Stores "value" into the shared-memory area pointed to by "rp2shm", with the
 * given index. That is, the equivalent to 'rp2shm[ index ] = value;'.
 *
 * volatile_mode: If 'true', stores "value" through a volatile pointer. That is,
 *		  using a real store instruction, rather than reusing some
 *		  register. If set to 'false', lets the compiler to decide.
 */
template < bool volatile_mode >
static __device__ void store_into_rp2shm( real value, rp2shm_t rp2shm, index_t index )
{

	#if __CUDA_ARCH__	/* Reduces work on cudafe(++) when looking for HOST code.*/

		#if ( __CUDA_ARCH__ < 200 ) && ( ! NMFGPU_SINGLE_PREC )

			/* Double-precision data on Compute Capability 1.x */

			int const val_hi = __double2hiint( value );
			int const val_lo = __double2loint( value );

			if ( volatile_mode ) {
				int volatile *const vp_hi = rp2shm.hi;
				int volatile *const vp_lo = rp2shm.lo;
				vp_hi[ index ] = val_hi;
				vp_lo[ index ] = val_lo;

			} else {
				rp2shm.hi[ index ] = val_hi;
				rp2shm.lo[ index ] = val_lo;
			}

		#else	/* Single-precision data <OR> Compute Capability >= 2.0 */

			if ( volatile_mode ) {
				real volatile *const vp_val = rp2shm;
				vp_val[ index ] = value;

			} else
				rp2shm[ index ] = value;

		#endif

	#endif

} // store_into_rp2shm

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

#if __CUDA_ARCH__ >= 300	/* Prevents 'unused-function' warning messages */

/* Warp Shuffle XOR (variable exchange between threads within a warp)
 * Calculates a source line ID by performing a bitwise XOR of the
 * caller's lane ID with the given bitmask.
 *
 * The width is fixed to the warp size.
 *
 * Currently, only 4-bytes quantities can be exchanged. For 8-bytes data,
 * two exchanges must be performed.
 *
 * WARNING: For Compute Capability >= 3.0 ONLY.
 */
static __device__ real real_shfl_xor( real variable, index_t laneMask )
{

	real value = REAL_C( 0.0 );

	#if __CUDA_ARCH__ >= 300 /* Compute Capability >= 3.0, only. */

		#if NMFGPU_SINGLE_PREC

			/* Single Precision */
			value = __shfl_xor( variable, laneMask );

		#else	/* Double Precision */

			int val_hi = __double2hiint( variable );
			int val_lo = __double2loint( variable );

			val_hi = __shfl_xor( val_hi, laneMask );
			val_lo = __shfl_xor( val_lo, laneMask );

			value = __hiloint2double( val_hi, val_lo );
		#endif

	#endif

	return value;

} // real_shfl_xor

#endif /* __CUDA_ARCH__ >= 300 */

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Helper method of kernel_reduce_to_row().
 *
 * Reduces a portion of d_A[] to a row. That is, returns sum( d_A[...][tx] ).
 *
 * block size: bs = blockDim.x * blockDim.y
 * matrix_size <= (gridDim.y * gridDim.x) * items_per_thread * bs
 * gridDim.y <= gridDim.x
 *
 * ext_grid:	'True' on an "extended" grid (i.e., gridDim.y > 1).
 *		In this case, it requires "single_Blk" to be 'false'.
 *		Otherwise, it forces blockIdx.y = 0 and gridDim.y = 1.
 *
 * single_Blk:	'True' if there is only a single block (i.e., gridDim.x == gridDim.y == 1,
 *		and  blockIdx.y == blockIdx.x == 0).
 *		In this case, it forces blockIdx.y = blockIdx.x = 0, and gridDim.x = gridDim.y = 1.
 *		In addition, it requires "ext_grid" to be 'false'.
 *
 * items_per_thread: Number of loads from global memory performed by each thread at a time.
 *
 * fast_prod:	'True' to perform integer products in "fast" mode.
 *		On Compute Capability 1.x, it is a 24-bits product, which IGNORES the higher
 *		8 bits of both (32-bits) input operands, and returns the lower 32 bits of
 *		the 48-bits result. This mode requires input operands less than 2**24.
 *		In contrast, in "extended" mode, some integer products are performed with
 *		the regular operator "*", which on Compute Capability 1.x, is compiled into
 *		multiple assembler instructions. This mode must be used when the number of
 *		blocks is greater than or equal to 2**24. In that case, only the related
 *		operations are performed with "*". The rest, are still performed in 24-bits.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		"fast_prod" must be 'false' if (gridDim.x * gridDim.y) >= 2**24
 *
 *	- No code is compiled on the following condition(s), since it would never be executed:
 *
 *		ext_grid == single_Blk == true
 *
 *		On Compute Capability >= 2.0:
 *			fast_prod == false:
 *				All products are performed with "*", which is already the fastest mode.
 *
 *		On Compute Capability >= 3.0:
 *			(sizeof(index_t) == sizeof(int)) && (items_per_thread > 2) && (ext_grid == true):
 *				On such architectures, maxGridSizeX >= INT_MAX. So, if sizeof(index_t) == sizeof(int),
 *				then (IDX_MAX / maxGridSizeX) <= 2. Therefore, if items_per_thread > 2, then
 *				(maxGridSizeX * items_per_thread) > IDX_MAX >= height
 *
 * 		On Compute Capability 1.x:
 *			(fast_prod == ext_grid == false):
 *				maxGridSizeX == maxGridSizeY < 2**16
 *
 * Returns the sum of the corresponding column. That is, sum(d_A[...][tx]).
 */
template < bool ext_grid, bool single_Blk, index_t items_per_thread, bool fast_prod >
static __device__ real reduce_gmem_to_row( real const *__restrict__ d_A, size_t matrix_size, index_t bs, index_t offset )
{

	// Resulting value.
	real sum_val = REAL_C( 0.0 );

	#if __CUDA_ARCH__	/* Reduces work on cudafe(++) when looking for HOST code.*/

	// Skips instantiated code that actually is never executed.
	if ( (! (ext_grid && single_Blk))
		#if __CUDA_ARCH__ >= 200
			&& fast_prod
			#if __CUDA_ARCH__ >= 300
				&& ( (sizeof(index_t) != sizeof(int)) || (items_per_thread <= 2) || (! ext_grid) )
			#endif
		#else
			&& ( fast_prod || ext_grid )
		#endif
	   )
	{

		// Grid "extension"
		bool const grid_extension = ext_grid;

		// Single Block
		bool const single_block = single_Blk;

		// Number of items read from global memory by each thread.
		index_t const num_items = items_per_thread;

		// Integer products in "fast" mode.
		bool const fast_product = fast_prod;

		// ------------------------------------------------

		// Block index
		index_t const bx = ( single_block ? 0 : blockIdx.x );
		index_t const by = ( grid_extension ? blockIdx.y : 0 );

		// Grid dimensions
		index_t const gdx = ( single_block ? 1 : gridDim.x );

		// ------------------------------------------------

		/* Each threads reads <num_items> elements from d_A[]
		 * with a distance equal to the block size ("bs").
		 */

		// Index to elements.
		size_t elemIdx[ num_items ];

		// "Active" block size.
		index_t const act_bs = idxmul<true>( num_items, bs );	// num_items * (blockDim.y * blockDim.x)

		/* Grid layout:
		 *
		 *	By_0:	Bx_0 (width=bdx, height=num_items * bdy)
		 *		Bx_1 (width=bdx, height=num_items * bdy)
		 *		...
		 *		Bx_{GDX-1}
		 *
		 *	By_1:	Bx_0
		 *		Bx_1 (width=bdx, height=num_items * bdy)
		 *	...
		 *
		 * Index of current block:  block_index  = By_i * GDX + Bx_j
		 * Offset to current block: block_offset = block_index  * act_bs
		 * Index to first element:  elemIdx	 = block_offset + offset
		 */

		// First element.
		elemIdx[ 0 ] = idxmul<true>( by, gdx ) + bx;
		elemIdx[ 0 ] = (size_t) idxmul<fast_product>( elemIdx[ 0 ], act_bs ) + offset;

		// Rest of elements.
		#pragma unroll
		for ( index_t i = 1 ; i < num_items ; i++ )
			elemIdx[ i ] = elemIdx[ i-1 ] + bs;

		// ------------------------------------------------

		// Each threads processes (up to) <num_items> elements from global memory.

		if ( (! single_block) && ( elemIdx[ num_items-1 ] < matrix_size ) ) {

			real value[ num_items ];

			#pragma unroll
			for ( index_t i = 0 ; i < num_items ; i++ )
				value[ i ] = d_A[ elemIdx[ i ] ];

			#pragma unroll
			for ( index_t i = 0 ; i < num_items ; i++ )
				sum_val += value[ i ];

		// Single-block mode, or matrix height is not multiple of <num_items>
		} else {

			/* Single-block mode:
			 *	Each threads reads <num_items> elements at a time until the block reaches <matrix_size>.
			 */
			while ( single_block && ( elemIdx[ num_items-1 ] < matrix_size ) ) {

				real value[ num_items ];

				#pragma unroll
				for ( index_t i = 0 ; i < num_items ; i++ )
					value[ i ] = d_A[ elemIdx[ i ] ];

				#pragma unroll
				for ( index_t i = 0 ; i < num_items ; i++ )
					elemIdx[ i ] += act_bs;

				#pragma unroll
				for ( index_t i = 0 ; i < num_items ; i++ )
					sum_val += value[ i ];
			}

			// Remaining elements...
			#pragma unroll
			for ( index_t i = 0 ; i < (num_items-1) ; i++ ) {
				if ( elemIdx[ i ] < matrix_size )
					sum_val += d_A[ elemIdx[ i ] ];
			}

		} // if multi-blocks mode

	} // Skips instantiated code that actually is never executed.

	#endif /* if __CUDA_ARCH__ > 0 */

	return sum_val;

} // reduce_gmem_to_row

// ---------------------------------------------

/*
 * Helper method of kernel_reduce_to_row().
 *
 * Reduces the shared-memory block to a row. That is, returns SUM( shmem[..][tx] )
 *
 * Size of shared-memory block: blockDim.x * blockDim.y
 *
 * block_width: Overrides blockDim.x
 *		Uses the "half-warpSize mode" on Compute Capability 1.x,
 *		if block_width == 16 (i.e warpSize/2).
 *		Disabled/ignored if set to 0.
 *
 * block_height: A power of 2 that overrides blockDim.y.
 *		 Disabled/ignored if set to 0.
 *
 * WARNING:
 *
 *	- blockDim.y > 1
 *
 *	- block_width == 0, implies that blockDim.x > warpSize.
 *
 *	- No code is compiled on the following conditions, since it would never be executed:
 *
 *		block_height == 1
 *
 *		(block_width > 0) && (block_width != warpSize)
 *
 *		On Compute Capability 1.x:
 *			(block_width > 0) && (block_width != 32) && (block_width != 16)
 *
 *			block_width == block_height == 0
 *
 * Returns the sum of the corresponding column. That is, sums[ty...(bdy-1)][tx]
 */
template < index_t block_width, index_t block_height >
static __device__ real reduce_shmem_to_row( index_t bs, index_t offset, real current_sum_val )
{

	// Resulting value.
	real sum_val = current_sum_val;

	#if __CUDA_ARCH__	/* Reduces work on cudafe(++) when looking for HOST code.*/

	// Skips instantiated code that actually is never executed.
	if ( ( block_height != 1 )
		#if __CUDA_ARCH__ >= 200
			&& ( (! block_width) || (block_width == warpSize) )
		#else
			&& ( ((! block_width) && block_height) || (block_width == 32) || (block_width == 16) )
		#endif
	   )
	{

		// ------------------------------------------------

		/* "half-warpSize mode" on Compute Capability 1.x,
		 * if block_width == 16 (i.e., warpSize/2)
		 */
		#if __CUDA_ARCH__ < 200
			bool const half_warpSize = ( block_width == 16 );
		#else
			bool const half_warpSize = false;
		#endif

		// ------------------------------------------------

		// Thread index
		index_t const ty = ( (block_height == 1) ? 0 : threadIdx.y );

		// Block dimensions
		index_t const bdx = ( block_width  ? block_width  : blockDim.x );	// == pitch
		index_t const bdy = ( block_height ? block_height : blockDim.y );	// A power of 2.

		index_t const half_bdy = (bdy >> 1);

		// ------------------------------------------------

		/* All threads store the result in a shared-memory block: sums[bdy][bdx],
		 * which is then reduced to a single row.
		 */
		extern __shared__ real smem[];
		rp2shm_t sums = new_rp2shm( smem, bs );

		if ( half_warpSize && (bdy == 2) ) {

			/* In 'half-warp-size' mode, uses volatile pointer(s). In this case,
			 * no subsequent synchronization will be necessary.
			 */
			store_into_rp2shm<true>( sum_val, sums, offset );	// sums[ty][tx]

		} else if ( ty >= half_bdy ) {
			// Only threads in the bottom half initialize the shared memory.
			store_into_rp2shm<false>( sum_val, sums, offset );	// sums[ty][tx]
		}

		// ------------------------------------------------

		/* Reduces sums[bdy][bdx] to a single row:
		 *
		 * for ( index_t half = bdy/2 ; half > 0 ; half /= 2 ) {	// bdy is a power of 2.
		 *
		 *	__syncthreads();
		 *
		 *	// Threads in the upper half read values from the lower half.
		 *	if ( ty < half )
		 *		sums[ty][tx] += sums[ty+half][tx];
		 * }
		 *
		 * Comparisons are NOT performed using <ty> and <half>. Instead, (offset == ty*bdx + tx)
		 * and (bound == half*bdx) are employed, respectively.
		 */

		/* half = bdy/2 .. 32.
		 * NOTE: It is never executed if maxThreadsPerBlock <= 1024, which is true for Compute
		 * Capability <= 3.5 (perhaps also for newer architectures).
		 */
		#if __CUDA_ARCH__ > 500

			for ( index_t half = half_bdy, bound = idxmul<true>( half, bdx ) ; half > 16 ; half >>= 1, bound >>= 1 ) {

				__syncthreads();

				// if (ty < half)
				if ( offset < bound ) {

					// sums[ty][tx] is already in sum_val
					// // sum_val += load_from_rp2shm<false>( sums, offset );

					// sums[ty+half][tx]
					sum_val += load_from_rp2shm<false>( sums, (offset + bound) );

					// New final value in sums[ty][tx]
					store_into_rp2shm<false>( sum_val, sums, offset );
				}
			}
		#endif

		// ------------------------------------------------

		// half == 16 .. 1
		switch ( bdy ) {

			// half == 16
			case 32: {
				index_t const bound = ( bdx << 4 );	// bdx * 16
				__syncthreads();
				if ( offset < bound ) {
					sum_val += load_from_rp2shm<false>( sums, (offset + bound) );
					store_into_rp2shm<false>( sum_val, sums, offset );
				}
			}

			// half == 8
			case 16: {
				index_t const bound = ( bdx << 3 );	// bdx * 8
				__syncthreads();
				if ( offset < bound ) {
					sum_val += load_from_rp2shm<false>( sums, (offset + bound) );
					store_into_rp2shm<false>( sum_val, sums, offset );
				}
			}

			// half == 4
			case 8: {
				index_t const bound = ( bdx << 2 );	// bdx * 4
				__syncthreads();
				if ( offset < bound ) {
					sum_val += load_from_rp2shm<false>( sums, (offset + bound) );
					store_into_rp2shm<false>( sum_val, sums, offset );
				}
			}

			// half == 2
			case 4: {
				index_t const bound = ( bdx << 1 );	// bdx * 2
				__syncthreads();
				if ( offset < bound ) {
					// Uses volatile pointers only on half_warpSize mode
					sum_val += load_from_rp2shm< half_warpSize >( sums, (offset + bound) );
					store_into_rp2shm< half_warpSize >( sum_val, sums, offset );
				}
			}

			// half == 1
			case 2: {
				index_t const bound = bdx;

				if ( ! half_warpSize )
					__syncthreads();

				if ( offset < bound )	// Uses volatile pointers only on half_warpSize mode
					sum_val += load_from_rp2shm< half_warpSize >( sums, (offset + bound) );	// sums[1][tx]
			}

		} // switch

	} // Skips instantiated code that actually is never executed.

	#endif	/* __CUDA_ARCH__ */

	return sum_val;	// sums[ty][tx]

} // reduce_shmem_to_row

// ---------------------------------------------

/*
 * d_C[i] = sum( d_A[i][...] )
 *
 * Reduces d_A[] to <gridDim.y * gridDim.x> rows that are stored in d_Tmp[].
 * Then, d_Tmp[] is reduced to a single row, which is stored in d_C[].
 *
 * blockDim.x == pitch
 * blockDim.y is a power of 2.
 * matrix_size <= (gridDim.y * gridDim.x) * items_per_thread * (block_height * block_width)
 * Size_of(d_Tmp) >= gridDim.y * gridDim.x * blockDim.x
 * Size_of(d_C) >= blockDim.x
 * gridDim.y <= gridDim.x
 * (gridDim.y * gridDim.x) <= UINT_MAX
 *
 * ext_grid:	'True' on an "extended" grid (i.e., gridDim.y > 1).
 *		In this case, it requires "single_Blk" to be 'false'.
 *		Otherwise, it forces blockIdx.y = 0 and gridDim.y = 1.
 *
 * single_Blk:	'True' if there is only a single block (i.e., gridDim.x == gridDim.y == 1,
 *		and  blockIdx.y == blockIdx.x == 0).
 *		In this case, it forces blockIdx.y = blockIdx.x = 0, and gridDim.x = gridDim.y = 1.
 *		In addition, it requires "ext_grid" to be 'false'.
 *
 * block_width: Overrides blockDim.x.
 *		On Compute Capability 1.x, it is denoted as "'half-warp-size' mode" when set to 16.
 *		In such case, and if block_height == 2, it makes use of volatile pointers to shared memory.
 *		Disabled/ignored if set to 0.
 *
 * block_height: A power of 2 that overrides blockDim.y.
 *		 When set to '1' (denoted as "'single-row' mode"), no shared memory is used, except for a boolean value
 *		 used on the reduction of d_Tmp[] to a row, in multi-block mode. Single-row mode also forces blockIdx.y = 0
 *		 Disabled/ignored if set to 0.
 *
 * items_per_thread: Number of loads from global memory performed by each thread at a time.
 *
 * fast_prod:	'True' to perform integer products in "fast" mode.
 *		On Compute Capability 1.x, it is a 24-bits product, which IGNORES the higher
 *		8 bits of both (32-bits) input operands, and returns the lower 32 bits of
 *		the 48-bits result. This mode requires input operands less than 2**24.
 *		In contrast, in "normal" mode, some integer products are performed with the
 *		regular operator "*", which on Compute Capability 1.x, is compiled into
 *		multiple assembler instructions. This mode must be used when the number of
 *		blocks is greater than or equal to 2**24. In that case, only the related
 *		operations are performed with "*". The rest, are still performed in 24-bits.
 *
 * Required size of shared memory:
 *	block_height != 1: block_width * block_height * sizeof(real) bytes.
 *	Else:
 *		multi-blocks mode:   sizeof(bool) bytes.
 *		Else (i.e., single block): 0 bytes.
 *
 * WARNING:
 *
 *	- block_width == 0, implies that blockDim.x > warpSize.
 *
 *	- On Compute Capability < 1.2, the resulting rows in d_Tmp[] will NOT be reduced.
 *	  Therefore, a second  call must be performed on d_Tmp[], with ext_grid = false,
 *	  single_Blk = true, and fast_prod = true.
 *	  Such call should be something like this:
 *		index_t const new_size = gridDim.y * gridDim.x * blockDim.x;
 *		kernel_reduce_to_row< false, true, ..., true ><<< 1, ... >>>( d_Tmp, new_size, NULL, d_C );
 *
 *	- On Compute Capability 1.x:
 *		"fast_prod" must be 'false' if (gridDim.x * gridDim.y) >= 2**24
 *
 *	- No code is compiled on the following condition(s), since it would never be executed:
 *
 *		ext_grid == single_Blk == true
 *
 * 		(block_width > 0) && (block_width != warpSize)
 *
 *		On Compute Capability >= 2.0:
 *			fast_prod == false:
 *				All products are performed with "*", which is already the fastest mode.
 *
 *		On Compute Capability >= 3.0
 *			(sizeof(index_t) == sizeof(int)) && (items_per_thread > 2) && (ext_grid == true):
 *				On such architectures, maxGridSizeX >= INT_MAX. So, if sizeof(index_t) == sizeof(int),
 *				then (IDX_MAX / maxGridSizeX) <= 2. Therefore, if items_per_thread > 2, then
 *				(maxGridSizeX * items_per_thread) > IDX_MAX >= height
 *
 *		On Compute Capability 1.x:
 *			(block_width > 0) && (block_width != warpSize) && (block_width != 16)
 *
 *			block_width == block_height == 0
 *
 *			(fast_prod == ext_grid == false):
 *				maxGridSizeX == maxGridSizeY < 2**16
 */
template < bool ext_grid, bool single_Blk, index_t block_width, index_t block_height, index_t items_per_thread, bool fast_prod >
static __global__ void kernel_reduce_to_row( real const *__restrict__ d_A, size_t matrix_size, real *__restrict__ d_Tmp,
						real *__restrict__ d_C)
{

	#if __CUDA_ARCH__	/* Reduces work on cudafe(++) when looking for HOST code.*/

	// Skips instantiated code that actually is never executed.
	if ( (! (ext_grid && single_Blk))
		#if __CUDA_ARCH__ >= 200
			&& ( (! block_width) || (block_width >= warpSize) )
			&& fast_prod
			#if __CUDA_ARCH__ >= 300
				&& ( (sizeof(index_t) != sizeof(int)) || (items_per_thread <= 2) || (! ext_grid) )
			#endif
		#else
			&& ( ((! block_width) && block_height) || (block_width == 32) || (block_width == 16) )
			&& ( fast_prod || ext_grid )
		#endif
	   )
	{

		// Grid "extension"
		bool const grid_extension = ext_grid;

		// Single Block
		bool const single_block = single_Blk;

		// Number of loads from global memory performed by each thread at a time.
		index_t const num_items = items_per_thread;

		// Integer products in "fast" mode.
		bool const fast_product = fast_prod;

		// ------------------------------------------------

		// Thread index
		index_t const tx = threadIdx.x;
		index_t const ty = ( (block_height == 1) ? 0 : threadIdx.y );

		// Block dimensions
		index_t const bdx = ( block_width  ? block_width  : blockDim.x );	// == pitch
		index_t const bdy = ( block_height ? block_height : blockDim.y );	// A power of 2.

		// Grid dimensions
		index_t const gdx = ( single_block ? 1 : gridDim.x );

		// Block size: blockDim.y * blockDim.x
		index_t const bs = idxmul<true>( bdy, bdx );

		// Offset from current block: threadIdx.y * blockDim.x + threadIdx.x
		index_t const offset = idxmul<true>( ty, bdx ) + tx;

		// ------------------------------------------------

		// Reduces a portion of d_A[] to a row. That is, sum(d_A[...][tx]).
		real sum_val = reduce_gmem_to_row< grid_extension, single_block, num_items, fast_product >( d_A, matrix_size, bs, offset );

		/* All threads store their results (i.e., "sum_val") in a shared-memory block,
		 * which is then reduced to a row.
		 *
		 * The returned value correspond to: sums[ty...(bdy-1)][tx]
		 */
		if ( bdy > 1 )
			sum_val = reduce_shmem_to_row< block_width, block_height >( bs, offset, sum_val );

		// ------------------------------------------------

		// Multi-blocks mode:
		if ( gdx > 1 ) {

			// Block index
			index_t const bx = blockIdx.x;
			index_t const by = ( grid_extension ? blockIdx.y : 0 );

			// ------------------------------------------------

			/* Threads with ty == 0 store their partial result in d_Tmp[by*gdx+bx][tx],
			 * which is then reduced to a row.
			 */
			{
				// Index to store the resulting value:
				size_t idx = idxmul<true>( by, gdx ) + bx;
				idx = idxmul< fast_product >( idx, bdx ) + tx;

				if ( (bdy == 1) || (offset < bdx) )	// if ( ty == 0 )
					d_Tmp[ idx ] = sum_val;		//	d_Tmp[by*gdx+bx][tx] = shmem[0][tx]
			}

			// ------------------------------------------------

			/* On Compute Capability >= 1.2:
			 *	The last block to finish, reduces the resulting rows in d_Tmp[ gdy*gdx ][ bdx ].
			 *	Please, see the threadfenceReduction example in CUDA SDK for details.
			 *
			 * On Compute Capability < 1.2, a second call must be performed on d_Tmp[], with grid_extension = false,
			 * single_block = true, and fast_product = true, to perform this action.Such call should be something
			 * like this:
			 *	index_t const new_size = gridDim.y * gridDim.x * blockDim.x;
			 *	kernel_reduce_to_row< false, true, ..., true><<< 1, ... >>>( d_Tmp, new_size, NULL, d_C );
			 *
			 * NOTE:
			 *	Actually, devices of Compute Capability 1.1 are able to execute this code, but not at
			 *	a 100% of occupancy, due to the required number of registers.
			 */

			#if (__CUDA_ARCH__ >= 120)

				// Grid dimensions
				index_t const gdy = ( grid_extension ? gridDim.y : 1 );

				// Is this block the last to finish the partial reduction of d_A[] ?
				extern __shared__ real smem[];
				bool *__restrict__ const p_isLastBlock = (bool *) &smem[0];

				// ------------------------------------------------

				// Thread ID is (0,0).
				bool const t0 = ! (tx + ty);

				// Total number of blocks
				index_t const num_blocks = idxmul<true>( gdy, gdx );

				// Size of current partial sum in d_Tmp. It is > bdx
				size_t const d_Tmp_size = idxmul< fast_product >( num_blocks, bdx );

				// ------------------------------------------------

				/* All threads wait until the latest writes to global memory are finished.
				 * NOTE:
				 * On Compute Capability 1.x, this functions is implemented
				 * with about 20 assembler instructions.
				 */
				__threadfence();

				// Thread 0 from each block checks if such block is the last one to finish.

				unsigned int ticket = 0U;	// Block position in line.
				bool l_isLastBlock = false;	// ticket == (num_blocks-1)

				if ( t0 ) {
					*p_isLastBlock = false;	// Initializes the shared memory.
					ticket = atomicInc( &retirement_counter, num_blocks );	// NOTE: num_blocks must be <= UINT_MAX
				}

				// Checks if it is the last block
				if ( ticket == (num_blocks-1) )
					*p_isLastBlock = true;	// Only the "last" thread 0 will store 'true' in shared memory.

				// ------------------------------------------------

				// All threads withing the block wait until "p_isLastBlock" has been written to shared memory.
				__syncthreads();

				l_isLastBlock = *p_isLastBlock;

				// The last block sums the results of all other blocks.
				if ( l_isLastBlock ) {

					// d_Tmp[ gdy*gdx ][ bdx ] is reduced to a row using a single block.

					// Reduces a portion of d_Tmp[] to a row. That is, sum(d_Tmp[...][tx]).
					sum_val = reduce_gmem_to_row< false, true, num_items, true >( d_Tmp, d_Tmp_size, bs, offset );

					/* All threads store their results (i.e., "sum_val") in a shared-memory block,
					 * which is then reduced to a row.
					 *
					 * The returned value correspond to: sums[ty...(bdy-1)][tx]
					 */
					if ( bdy > 1 )
						sum_val = reduce_shmem_to_row< block_width, block_height >( bs, offset, sum_val );

					// The resulting value is stored in d_C[0][tx]
					if ( (bdy == 1) || (offset < bdx) ) {		// if ( ty == 0 )
						d_C[ offset ] = sum_val;		//	d_C[0][tx] = shmem[0][tx]

						// In addition, any of threads reset the counter (do not care which).
						retirement_counter = 0U;
					}

				} // if last_block

			#endif /* __CUDA_ARCH__ >= 120 */

		// "Single-block" mode
		} else {
			// The result is directly stored in d_C[0][tx].
			if ( (bdy == 1) || (offset < bdx) )		// if ( ty == 0 )
				d_C[ offset ] = sum_val;		//	d_C[0][tx] = shmem[0][tx]

		} // if ( gdx > 1 )

	} // Condition to skip instantiated code that actually is never executed.

	#endif /* if __CUDA_ARCH__ > 0 */

} // kernel_reduce_to_row

// ---------------------------------------------

/*
 * d_accum_A[j] = SUM( d_A[...][j] )
 *
 * height <= (dimGrid.y * dimGrid.x) * REDUCE_TO_ROW__ITEMS_PER_THREAD * block_height
 * d_Tmp: Temporary storage. Ignored if grid_length == 1.
 * size_of( d_Tmp ) >= (dimGrid.y * dimGrid.x) * pitch
 * length( d_accum_A ) >= pitch
 *
 * block_height <= (maxThreadsPerBlock / pitch), and must be a power of 2.
 * (gridDim.y * gridDim.x) <= UINT_MAX
 * "pitch" must be a multiple of 'memory_alignment'.
 *
 * WARNING:
 *	height > 1
 */
__host__ void reduce_to_row( real const *__restrict__ d_A, index_t height, index_t pitch, real *__restrict__ d_Tmp, index_t block_height,
				dim3 dimGrid, cudaStream_t stream_AccA, real *__restrict__ d_accum_A )
{

	// Grid dimensions:
	index_t const grid_length = dimGrid.x;
	index_t const grid_extension = dimGrid.y;

	// Matrix size
	size_t const matrix_size = (size_t) height * (size_t) pitch;

	// Number of loads from global memory performed by each thread at a time.
	index_t const num_items = REDUCE_TO_ROW__ITEMS_PER_THREAD;

	/* Integer product mode:
	 *
	 * On Compute Capability 1.x, if the number of blocks is greater than
	 * or equal to 2**24, some of the integer products must be performed
	 * with the regular operator "*", which is slower on this architecture.
	 * NOTE: This can happen on EXTENDED grids, only.
	 * Otherwise (i.e., with less blocks), it uses the "fast" mode.
	 *
	 * On Compute Capability >= 2.0, "*" is already the "fast" mode.
	 */
	bool const fast_product = (computeCapability > 1) + (((size_t) grid_extension * (size_t) grid_length) < (1 << 24));

	// --------------------------------

	/* Selects the kernel to launch according to block configuration.
	 *
	 * NOTE: The following conditions will be checked on kernel compilation:
	 *
	 *	- No grid extension is required on Compute Capability >= 3.0, if
	 *		(sizeof(index_t) == sizeof(int)) && (num_items > 2)
	 *
	 *	- Integer products in NON-extended grids are always performed in "fast" mode.
	 */

	if ( pitch > 32 ) {

		/* NOTE:
		 *	Since "pitch" is not known at compile time, sets the corresponding
		 *	template parameter to 0, so the kernel uses the "blockDim.x"
		 *	built-in variable.
		 */
		index_t const blk_width = 0;

		if ( grid_length > 1 ) {	// multi-blocks grid.

			bool const single_Blk = false;

			switch ( block_height ) {

				/* NOTE:
				 *	For block_height >= 16, the template parameter is set to 0,
				 *	so the kernel uses the "blockDim.y" built-in variable.
				 *
				 *	It is never executed on Compute Capability 1.x, so
				 *	"fast_product" is always 'true'
				 */
				default: {

					dim3 const dimBlock( pitch, block_height );
					size_t const shmem_size = pitch * block_height * sizeof(real);
					index_t const blk_height = 0;

					bool const fast_prod = true;
					if ( grid_extension > 1 )
						kernel_reduce_to_row< true, single_Blk, blk_width, blk_height, num_items, fast_prod >
									<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
					else
						kernel_reduce_to_row< false, single_Blk, blk_width, blk_height, num_items, fast_prod >
									<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
				} break;

				// NOTE: More than 256 threads.
				case 8: {
					index_t const blk_height = 8;
					dim3 const dimBlock( pitch, blk_height );
					size_t const shmem_size = pitch * blk_height * sizeof(real);

					if ( grid_extension > 1 ) {

						bool const ext_grid = true;
						if ( fast_product )
							kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, true >
										<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
											( d_A, matrix_size, d_Tmp, d_accum_A );
						else
							kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, false >
										<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
											( d_A, matrix_size, d_Tmp, d_accum_A );

					// No grid extension
					} else {
						bool const fast_prod = true;
						kernel_reduce_to_row< false, single_Blk, blk_width, blk_height, num_items, fast_prod >
									<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
					}
				} break;

				case 4: {
					index_t const blk_height = 4;
					dim3 const dimBlock( pitch, blk_height );
					size_t const shmem_size = pitch * blk_height * sizeof(real);

					if ( grid_extension > 1 ) {

						bool const ext_grid = true;
						if ( fast_product )
							kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, true >
										<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
											( d_A, matrix_size, d_Tmp, d_accum_A );
						else
							kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, false >
										<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
											( d_A, matrix_size, d_Tmp, d_accum_A );
					// No grid extension
					} else {
						bool const fast_prod = true;
						kernel_reduce_to_row< false, single_Blk, blk_width, blk_height, num_items, fast_prod >
									<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
					}
				} break;

				case 2: {
					index_t const blk_height = 2;
					dim3 const dimBlock( pitch, blk_height );
					size_t const shmem_size = pitch * blk_height * sizeof(real);

					if ( grid_extension > 1 ) {

						bool const ext_grid = true;
						if ( fast_product )
							kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, true >
										<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
											( d_A, matrix_size, d_Tmp, d_accum_A );
						else
							kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, false >
										<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
											( d_A, matrix_size, d_Tmp, d_accum_A );
					// No grid extension
					} else {
						bool const fast_prod = true;
						kernel_reduce_to_row< false, single_Blk, blk_width, blk_height, num_items, fast_prod >
									<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
					}
				} break;

				/* NOTE:
				 *	block_height == 1 is referred as "'single-row' mode".
				 *	Shared memory is allocated for just a single boolean value.
				 */
				case 1: {
					index_t const blk_height = 1;
					dim3 const dimBlock( pitch, blk_height );
					size_t const shmem_size = sizeof(bool);

					if ( grid_extension > 1 ) {

						bool const ext_grid = true;
						if ( fast_product )
							kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, true >
										<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
											( d_A, matrix_size, d_Tmp, d_accum_A );
						else
							kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, false >
										<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
											( d_A, matrix_size, d_Tmp, d_accum_A );
					// No grid extension
					} else {
						bool const fast_prod = true;
						kernel_reduce_to_row< false, single_Blk, blk_width, blk_height, num_items, fast_prod >
									<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
					}
				} break;

			} // switch( block_height )
		}

		// Just a single block
		else {
			dim3 const dimBlock( pitch, block_height );
			dim3 const dimGrid_SB( 1, 1 );

			// No shared memory is required if block_height == 1
			size_t const shmem_size = ( (block_height > 1) ? (pitch * block_height * sizeof(real)) : 0 );

			real *__restrict__ const pTmp = NULL;	// No temporary buffer is required.

			/* NOTE:
			 *	For the block height, sets the template parameter to 0,
			 *	so the kernel uses the "blockDim.y" built-in variable.
			 */
			index_t const blk_height = 0;

			bool const ext_grid = false;
			bool const single_Blk = true;
			bool const fast_prod = true;

			kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, fast_prod >
						<<< dimGrid_SB, dimBlock, shmem_size, stream_AccA >>>
							( d_A, matrix_size, pTmp, d_accum_A );
		}

	} // pitch > 32

	// pitch == 32. Typical warp size for most (or all) architectures.
	else if ( pitch == 32 ) {

		index_t const blk_width = 32;

		if ( grid_length > 1 ) {	// multi-blocks grid.

			bool const single_Blk = false;

			switch ( block_height ) {

				// NOTE: 512 threads.
				case 16: {
					index_t const blk_height = 16;
					dim3 const dimBlock( blk_width, blk_height );
					size_t const shmem_size = blk_width * blk_height * sizeof(real);

					if ( grid_extension > 1 ) {

						bool const ext_grid = true;
						if ( fast_product )
							kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, true >
										<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
											( d_A, matrix_size, d_Tmp, d_accum_A );
						else
							kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, false >
										<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
											( d_A, matrix_size, d_Tmp, d_accum_A );
					// No grid extension
					} else {
						bool const fast_prod = true;
						kernel_reduce_to_row< false, single_Blk, blk_width, blk_height, num_items, fast_prod >
									<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
					}
				} break;

				case 8: {
					index_t const blk_height = 8;
					dim3 const dimBlock( blk_width, blk_height );
					size_t const shmem_size = blk_width * blk_height * sizeof(real);

					if ( grid_extension > 1 ) {

						bool const ext_grid = true;
						if ( fast_product )
							kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, true >
										<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
											( d_A, matrix_size, d_Tmp, d_accum_A );
						else
							kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, false >
										<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
											( d_A, matrix_size, d_Tmp, d_accum_A );
					// No grid extension
					} else {
						bool const fast_prod = true;
						kernel_reduce_to_row< false, single_Blk, blk_width, blk_height, num_items, fast_prod >
									<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
					}
				} break;

				/* NOTE:
				 *	For other values, the template parameter is set to 0,
				 *	so the kernel uses the "blockDim.y" built-in variable.
				 */
				default: {
					dim3 const dimBlock( blk_width, block_height );
					size_t const shmem_size = blk_width * block_height * sizeof(real);
					index_t const blk_height = 0;

					if ( grid_extension > 1 ) {

						bool const ext_grid = true;
						if ( fast_product )
							kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, true >
										<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
											( d_A, matrix_size, d_Tmp, d_accum_A );
						else
							kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, false >
										<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
											( d_A, matrix_size, d_Tmp, d_accum_A );
					// No grid extension
					} else {
						bool const fast_prod = true;
						kernel_reduce_to_row< false, single_Blk, blk_width, blk_height, num_items, fast_prod >
									<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
					}
				} break;

			} // switch( block_height )
		}

		// Just a single block
		else {
			dim3 const dimBlock( blk_width, block_height );
			dim3 const dimGrid_SB( 1, 1 );

			// No shared memory is required if block_height == 1
			size_t const shmem_size = ( (block_height > 1) ? (pitch * block_height * sizeof(real)) : 0 );

			real *__restrict__ const pTmp = NULL;	// No temporary buffer is required.

			/* NOTE:
			 *	For the block height, sets the template parameter to 0,
			 *	so the kernel uses the "blockDim.y" built-in variable.
			 */
			index_t const blk_height = 0;

			bool const ext_grid = false;
			bool const single_Blk = true;
			bool const fast_prod = true;

			kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, fast_prod >
						<<< dimGrid_SB, dimBlock, shmem_size, stream_AccA >>>
							( d_A, matrix_size, pTmp, d_accum_A );
		}

	} // pitch == 32

	/* pitch == 16.
	 * NOTE: Intended for Compute Capability 1.x ONLY, where it is referred as "'half-warp-size' mode".
	 */
	else {

		index_t const blk_width = 16;

		if ( grid_length > 1 ) {	// multi-blocks grid.

			bool const single_Blk = false;

			switch ( block_height ) {

				case 16: {
					index_t const blk_height = 16;
					dim3 const dimBlock( blk_width, blk_height );
					size_t const shmem_size = blk_width * blk_height * sizeof(real);

					if ( grid_extension > 1 ) {

						bool const ext_grid = true;
						if ( fast_product )
							kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, true >
										<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
											( d_A, matrix_size, d_Tmp, d_accum_A );
						else
							kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, false >
										<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
											( d_A, matrix_size, d_Tmp, d_accum_A );
					// No grid extension
					} else {
						bool const fast_prod = true;
						kernel_reduce_to_row< false, single_Blk, blk_width, blk_height, num_items, fast_prod >
									<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
					}
				} break;

				/* NOTE:
				 *	For other values, the template parameter is set to 0,
				 *	so the kernel uses the "blockDim.y" built-in variable.
				 */
				default: {
					dim3 const dimBlock( blk_width, block_height );
					size_t const shmem_size = blk_width * block_height * sizeof(real);
					index_t const blk_height = 0;

					if ( grid_extension > 1 ) {

						bool const ext_grid = true;
						if ( fast_product )
							kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, true >
										<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
											( d_A, matrix_size, d_Tmp, d_accum_A );
						else
							kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, false >
										<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
											( d_A, matrix_size, d_Tmp, d_accum_A );
					// No grid extension
					} else {
						bool const fast_prod = true;
						kernel_reduce_to_row< false, single_Blk, blk_width, blk_height, num_items, fast_prod >
									<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
					}
				} break;

			} // switch( block_height )
		}

		// Just a single block
		else {
			dim3 const dimBlock( blk_width, block_height );
			dim3 const dimGrid_SB( 1, 1 );

			// No shared memory is required if block_height == 1
			size_t const shmem_size = ( (block_height > 1) ? (pitch * block_height * sizeof(real)) : 0 );

			real *__restrict__ const pTmp = NULL;	// No temporary buffer is required.

			/* NOTE:
			 *	For the block height, sets the template parameter to 0,
			 *	so the kernel uses the "blockDim.y" built-in variable.
			 */
			index_t const blk_height = 0;

			bool const ext_grid = false;
			bool const single_Blk = true;
			bool const fast_prod = true;

			kernel_reduce_to_row< ext_grid, single_Blk, blk_width, blk_height, num_items, fast_prod >
						<<< dimGrid_SB, dimBlock, shmem_size, stream_AccA >>>
							( d_A, matrix_size, pTmp, d_accum_A );
		}

	} // pitch == 16

} // reduce_to_row

////////////////////////////////////////////////

/*
 * d_A = d_B <op> d_A
 *
 * <op> is "./" or "-"
 *
 * matrix_size <= (gridDim.y * gridDim.x) * items_per_thread * blockDim.x
 * blockDim.y == 1  &  threadIdx.y == 0
 *
 * gridDim.y <= gridDim.x
 *
 * ext_grid: 'True' on an "extended" grid (i.e., gridDim.y > 1).
 *		   If set to 'false', forces blockIdx.y = 0 and gridDim.y = 1.
 *
 * items_per_thread: Number of loads from global memory performed by each thread at a time.
 *
 * div_operator: 'True' if the operation to perform, is a floating-point division.
 *		 Otherwise, a subtraction is performed.
 *
 * fast_prod: 'True' to perform integer products in "fast" mode.
 *		On Compute Capability 1.x, it is a 24-bits product, which IGNORES the higher
 *		8 bits of both (32-bits) input operands, and returns the lower 32 bits of
 *		the 48-bits result. This mode requires input operands less than 2**24.
 *		In contrast, in "normal" mode, some integer products are performed with the
 *		regular operator "*", which on Compute Capability 1.x, is compiled into
 *		multiple assembler instructions. This mode must be used when the number of
 *		blocks is greater than or equal to 2**24. In that case, only the related
 *		operations are performed with "*". The rest, are still performed in 24-bits.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		"fast_prod" must be 'false' if (gridDim.x * gridDim.y) >= 2**24
 *
 *	- No code is compiled on the following condition(s), since it would never be executed:
 *
 *		On Compute Capability >= 2.0:
 *			fast_prod == false:
 *				All products are performed with "*", which is already the fastest mode.
 *
 *		On Compute Capability >= 3.0
 *			(sizeof(size_t) == sizeof(index_t)) && (ext_grid == true):
 *				On such architectures, maxGridSizeX ~ IDX_MAX. That is,
 *				(IDX_MAX / maxGridSizeX) <= 2.
 *				So, if sizeof(size_t) == sizeof(index_t), then
 *				(matrix_size / maxGridSizeX) <= 2, which is certainly less than
 *				<memory_alignment> (i.e., the minimum block size). Therefore,
 *				(maxGridSizeX * act_bs) > matrix_size for any (active) block size.
 *
 * 		On Compute Capability 1.x:
 *			(fast_prod == ext_grid == false):
 *				maxGridSizeX == maxGridSizeY < 2**16
 */
template < bool ext_grid, index_t items_per_thread, bool div_operator, bool fast_prod >
static __global__ void kernel_div_sub( real *__restrict__ d_A, real const *__restrict__ d_B, size_t matrix_size )
{

	#if __CUDA_ARCH__	/* Reduces work on cudafe(++) when looking for HOST code.*/

	// Skips instantiated code that actually is never executed.
	if (
		#if __CUDA_ARCH__ >= 200
			fast_prod
			#if __CUDA_ARCH__ >= 300
				&& ( (sizeof(size_t) > sizeof(index_t)) || (! ext_grid) )
			#endif
		#else
			( fast_prod || ext_grid )
		#endif
	   )
	{

		// Grid "extension"
		bool const grid_extension = ext_grid;

		// Number of items read from global memory by each thread.
		index_t const num_items = items_per_thread;

		// Integer products in "fast" mode.
		bool const fast_product = fast_prod;

		// ------------------------------------------------

		// Thread index
		index_t const tx = threadIdx.x;

		// Block index
		index_t const bx = blockIdx.x;
		index_t const by = ( grid_extension ? blockIdx.y : 0 );

		// Block dimensions
		index_t const bdx = blockDim.x;

		// Grid dimensions
		index_t const gdx = gridDim.x;

		// ------------------------------------------------

		/* Each threads reads <num_items> elements from d_A[]
		 * and d_B[] with a distance equal to the block size.
		 */

		// Index to elements.
		size_t elemIdx[ num_items ];

		// Block size
		index_t const bs = bdx;

		// Offset from current block: threadIdx.x
		index_t const offset = tx;

		// "Active" block size.
		index_t const act_bs = idxmul<true>( num_items, bs );	// num_items * blockDim.x

		/* Grid layout (1-D thread blocks):
		 *
		 *	By_0:	Bx_0 Bx_1 ... Bx_{GDX-1}
		 *	By_1:	Bx_0 Bx_1 ... Bx_{GDX-1}
		 *	...
		 *
		 * Index of current block:  block_index  = By_i * GDX + Bx_j
		 * Offset to current block: block_offset = block_index * act_bs
		 * Index to first element:  elemIdx	 = block_offset + offset
		 */

		// First element.
		elemIdx[ 0 ] = idxmul<true>( by, gdx ) + bx;
		elemIdx[ 0 ] = (size_t) idxmul<fast_product>( elemIdx[ 0 ], act_bs ) + offset;

		// Rest of elements.
		#pragma unroll
		for ( index_t i = 1 ; i < num_items ; i++ )
			elemIdx[ i ] = elemIdx[ i-1 ] + bs;

		// ------------------------------------------------

		// Each threads processes (up to) <num_items> elements from global memory.

		if ( elemIdx[ (num_items-1) ] < matrix_size ) {

			/* Compute Capability 1.0 - 1.1 */
			#if __CUDA_ARCH__ < 120

				// NOTE: The "if" statement must be OUTSIDE of the loop in order to unroll it.

				if ( div_operator ) {
					#pragma unroll
					for ( index_t i = 0 ; i < num_items ; i++ )
						d_A[ elemIdx[ i ] ] = device_fdiv( d_B[ elemIdx[ i ] ], d_A[ elemIdx[ i ] ] );	// A = B / A
				} else {
					#pragma unroll
					for ( index_t i = 0 ; i < num_items ; i++ )
						d_A[ elemIdx[ i ] ] = ( d_B[ elemIdx[ i ] ] - d_A[ elemIdx[ i ] ] );	// A = B - A
				}


			/* Compute Capability 1.2, and beyond: "faster" code (i.e., exposes a higher degree of
			 * ILP: Instruction-Level Parallelism), but requires more registers. */
			#else
				real value_A[ num_items ], value_B[ num_items ];

				#pragma unroll
				for ( index_t i = 0 ; i < num_items ; i++ )
					value_A[ i ] = d_A[ elemIdx[ i ] ];

				#pragma unroll
				for ( index_t i = 0 ; i < num_items ; i++ )
					value_B[ i ] = d_B[ elemIdx[ i ] ];

				// NOTE: Compute Capability 1.x: The "if" statement must be OUTSIDE of the loop in order to unroll it.
				if ( div_operator ) {
					#pragma unroll
					for ( index_t i = 0 ; i < num_items ; i++ )
						value_A[ i ] = device_fdiv( value_B[ i ], value_A[ i ] );	// A = B / A
				} else {
					#pragma unroll
					for ( index_t i = 0 ; i < num_items ; i++ )
						value_A[ i ] = ( value_B[ i ] - value_A[ i ] );		// A = B - A
				}

				#pragma unroll
				for ( index_t i = 0 ; i < num_items ; i++ )
					d_A[ elemIdx[ i ] ] = value_A[ i ];

			#endif /* Compute Capability */


		// Matrix height is not multiple of <num_items>
		} else {
			// NOTE: Compute Capability 1.x: The "if" statement must be OUTSIDE of the loop in order to unroll it.
			if ( div_operator ) {
				#pragma unroll
				for ( index_t i = 0 ; i < (num_items-1) ; i++ )
					if ( elemIdx[ i ] < matrix_size ) {
						real value_A = d_A[ elemIdx[ i ] ];
						real const value_B = d_B[ elemIdx[ i ] ];
						value_A = device_fdiv( value_B, value_A );	// A = B / A
						d_A[ elemIdx[ i ] ] = value_A;
					}
			} else {
				#pragma unroll
				for ( index_t i = 0 ; i < (num_items-1) ; i++ )
					if ( elemIdx[ i ] < matrix_size ) {
						real value_A = d_A[ elemIdx[ i ] ];
						real const value_B = d_B[ elemIdx[ i ] ];
						value_A = ( value_B - value_A );	// A = B - A
						d_A[ elemIdx[ i ] ] = value_A;
					}
			}
		}

	} // Skips instantiated code that actually is never executed.

	#endif /* if __CUDA_ARCH__ > 0 */

} // kernel_div

// ---------------------------------------------

/*
 * d_A = d_B <op> d_A
 *
 * <op> is "./" or "-"
 *
 * matrix_size <= (dimGrid.y * dimGrid.x * DIV_SUB__ITEMS_PER_THREAD * block_size)
 * block_size <= maxThreadsPerBlock
 * dimGrid.x  <= maxGridSizeX
 * dimGrid.y <= MIN( dimGrid.x, maxGridSizeY )
 *
 * div_operator: 'True' if operation to perform is a floating-point division.
 *		Otherwise, a subtraction is performed.
 */
__host__ void div_sub( real *__restrict__ d_A, real const *__restrict__ d_B, size_t matrix_size, index_t block_size, dim3 dimGrid,
			bool div_operator, cudaStream_t stream_A )
{

	// Grid dimensions:
	index_t const grid_length = dimGrid.x;
	index_t const grid_extension = dimGrid.y;

	// Number of loads from global memory performed by each thread at a time.
	index_t const num_items = DIV_SUB__ITEMS_PER_THREAD;

	/* Integer product mode:
	 *
	 * On Compute Capability 1.x, if the number of blocks is greater than
	 * or equal to 2**24, some of the integer products must be performed
	 * with the regular operator "*", which is slower on this architecture.
	 * NOTE: This can happen on EXTENDED grids, only.
	 * Otherwise (i.e., with less blocks), it uses the "fast" mode.
	 *
	 * On Compute Capability >= 2.0, "*" is already the "fast" mode.
	 */
	bool const fast_product = (computeCapability > 1) + (((size_t) grid_extension * (size_t) grid_length) < (1 << 24));

	dim3 const dimBlock( block_size, 1 );	// 1-D blocks

	size_t const shmem_size = 0;

	// --------------------------------

	/* Selects the kernel to launch according to the selected operator.
	 *
	 * NOTE: The following conditions will be checked on kernel compilation:
	 *
	 *	- No grid extension is required on Compute Capability >= 3.0, if
	 *		sizeof(size_t) == sizeof(int)
	 *
	 *	- Integer products in NON-extended grids are always performed in "fast" mode.
	 */

	// Division
	if ( div_operator ) {

		bool const div_op = true;

		if ( grid_extension > 1 ) {

			bool const extended_grid = true;
			if ( fast_product )
				kernel_div_sub< extended_grid, num_items, div_op, true >
						<<< dimGrid, dimBlock, shmem_size, stream_A >>>( d_A, d_B, matrix_size );
			else
				kernel_div_sub< extended_grid, num_items, div_op, false >
						<<< dimGrid, dimBlock, shmem_size, stream_A >>>( d_A, d_B, matrix_size );
		// No grid extension
		} else {
			bool const fast_prod = true;
				kernel_div_sub< false, num_items, div_op, fast_prod >
						<<< dimGrid, dimBlock, shmem_size, stream_A >>>( d_A, d_B, matrix_size );
		}

	// Subtraction
	} else {
		bool const div_op = false;

		if ( grid_extension > 1 ) {

			bool const extended_grid = true;
			if ( fast_product )
				kernel_div_sub< extended_grid, num_items, div_op, true >
						<<< dimGrid, dimBlock, shmem_size, stream_A >>>( d_A, d_B, matrix_size );
			else
				kernel_div_sub< extended_grid, num_items, div_op, false >
						<<< dimGrid, dimBlock, shmem_size, stream_A >>>( d_A, d_B, matrix_size );
		// No grid extension
		} else {
			bool const fast_prod = true;
				kernel_div_sub< false, num_items, div_op, fast_prod >
						<<< dimGrid, dimBlock, shmem_size, stream_A >>>( d_A, d_B, matrix_size );
		}

	} // if div_operator

} // div_sub

////////////////////////////////////////////////

/*
 * d_A[i][j] = d_A[i][j] .* d_Aux[i][j] ./ d_accum_b[j]
 *
 * blockDim.x == pitch
 * matrix_size <= ((gridDim.y * gridDim.x) * items_per_thread * blockDim.y) * blockDim.x
 * Size_of(d_accum_b) >= blockDim.x
 * gridDim.y <= gridDim.x
 *
 * ext_grid: 'True' on an "extended" grid (i.e., gridDim.y > 1).
 *		   If set to 'false', forces blockIdx.y = 0 and gridDim.y = 1.
 *
 * single_row:	'True' for single-row blocks (i.e., blockDim.y == 1 and threadIdx.y == 0).
 *		It forces threadIdx.y = 0 and blockDim.y = 1.
 *		In addition, no shared memory is used.
 *
 * items_per_thread: Number of loads from global memory performed by each thread at a time.
 *
 * fast_prod: 'True' to perform integer products in "fast" mode.
 *		On Compute Capability 1.x, it is a 24-bits product, which IGNORES the higher
 *		8 bits of both (32-bits) input operands, and returns the lower 32 bits of
 *		the 48-bits result. This mode requires input operands less than 2**24.
 *		In contrast, in "normal" mode, some integer products are performed with the
 *		regular operator "*", which on Compute Capability 1.x, is compiled into
 *		multiple assembler instructions. This mode must be used when the number of
 *		blocks is greater than or equal to 2**24. In that case, only the related
 *		operations are performed with "*". The rest, are still performed in 24-bits.
 *
 * Required size of shared memory:
 *	Multi-rows blocks:	  blockDim.x * sizeof(real) bytes.
 *	Else (i.e., single-row blocks): 0 bytes.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		"fast_prod" must be 'false' if (gridDim.x * gridDim.y) >= 2**24
 *
 *	- No code is compiled on the following condition(s), since it would never be executed:
 *
 *		On Compute Capability >= 2.0:
 *			fast_prod == false:
 *				All products are performed with "*", which is already the fastest mode.
 *
 *		On Compute Capability >= 3.0:
 *			(sizeof(index_t) == sizeof(int)) && (items_per_thread > 2) && (ext_grid == true):
 *				On such architectures, maxGridSizeX >= INT_MAX. So, if sizeof(index_t) == sizeof(int),
 *				then (IDX_MAX / maxGridSizeX) <= 2. Therefore, if items_per_thread > 2, then
 *				(maxGridSizeX * items_per_thread) > IDX_MAX >= height
 *
 * 		On Compute Capability 1.x:
 *			(fast_prod == ext_grid == false):
 *				maxGridSizeX == maxGridSizeY < 2**16
 */
template <bool ext_grid, bool single_row, index_t items_per_thread, bool fast_prod >
static __global__ void kernel_mul_div( real *__restrict__ d_A, real const *__restrict__ d_Aux, real const *__restrict__ d_accum_b,
					size_t matrix_size )
{

	#if __CUDA_ARCH__	/* Reduces work on cudafe(++) when looking for HOST code.*/

	// Skips instantiated code that actually is never executed.
	if (
		#if __CUDA_ARCH__ >= 200
			fast_prod
			#if __CUDA_ARCH__ >= 300
				&& ( (sizeof(index_t) != sizeof(int)) || (items_per_thread <= 2) || (! ext_grid) )
			#endif
		#else
			( fast_prod || ext_grid )
		#endif
	   )
	{

		// Grid "extension"
		bool const grid_extension = ext_grid;

		// Number of items read from global memory by each thread.
		index_t const num_items = items_per_thread;

		// Integer products in "fast" mode.
		bool const fast_product = fast_prod;

		// ------------------------------------------------

		// Thread index
		index_t const tx = threadIdx.x;
		index_t const ty = ( single_row ? 0 : threadIdx.y );

		// Block index
		index_t const bx = blockIdx.x;
		index_t const by = ( grid_extension ? blockIdx.y : 0 );

		// Block dimensions
		index_t const bdx = blockDim.x;				// == pitch
		index_t const bdy = ( single_row ? 1 : blockDim.y );

		// Grid dimensions
		index_t const gdx = gridDim.x;

		// ------------------------------------------------

		/* Multi-rows mode:
		 *	Threads on first row store d_accum_b[ tx ] into shared memory: acc[ tx ]
		 */

		// Shared memory for vector d_accum_b[]. Length: blockDim.x
		extern __shared__ real smem[];
		rp2shm_t acc = new_rp2shm( smem, bdx );

		if ( ! ( single_row || ty ) )
			store_into_rp2shm< false >( d_accum_b[ tx ], acc, tx );	// acc[ tx ] = d_accum_b[ tx ]

		// ------------------------------------------------

		/* Each threads reads <num_items> elements from d_A[]
		 * with a distance equal to the block size ("bs").
		 */

		// Index to elements.
		size_t elemIdx[ num_items ];

		// Block size
		index_t const bs = idxmul<true>( bdy, bdx );

		// Offset from current block: threadIdx.y * blockDim.x + threadIdx.x
		index_t const offset = idxmul<true>( ty, bdx ) + tx;

		// "Active" block size.
		index_t const act_bs = idxmul<true>( num_items, bs );	// num_items * (blockDim.y * blockDim.x)

		/* Grid layout:
		 *
		 *	By_0:	Bx_0 (width=bdx, height=num_items * bdy)
		 *		Bx_1 (width=bdx, height=num_items * bdy)
		 *		...
		 *		Bx_{GDX-1}
		 *
		 *	By_1:	Bx_0
		 *		Bx_1 (width=bdx, height=num_items * bdy)
		 *	...
		 *
		 * Index of current block:  block_index  = By_i * GDX + Bx_j
		 * Offset to current block: block_offset = block_index  * act_bs
		 * Index to first element:  elemIdx	 = block_offset + offset
		 */

		// First element.
		elemIdx[ 0 ] = idxmul<true>( by, gdx ) + bx;
		elemIdx[ 0 ] = (size_t) idxmul<fast_product>( elemIdx[ 0 ], act_bs ) + offset;

		// Rest of elements.
		#pragma unroll
		for ( index_t i = 1 ; i < num_items ; i++ )
			elemIdx[ i ] = elemIdx[ i-1 ] + bs;

		// ------------------------------------------------

		// Loads d_accum_b[ tx ] from global or shared memory.

		real val_accum = REAL_C( 0.0 );

		if ( single_row )
			val_accum = d_accum_b[ tx ];
		else {
			__syncthreads();
			val_accum = load_from_rp2shm<false>( acc, tx );
		}

		// ------------------------------------------------

		// Each threads processes (up to) <num_items> elements from global memory.

		if ( elemIdx[ (num_items-1) ] < matrix_size ) {

			/* Compute Capability 1.0 - 1.1 */
			#if __CUDA_ARCH__ < 120

				#pragma unroll
				for ( index_t i = 0 ; i < num_items ; i++ )
					d_A[ elemIdx[ i ] ] = device_fdiv( ( d_A[ elemIdx[ i ] ] * d_Aux[ elemIdx[ i ] ] ), val_accum );


			/* Compute Capability 1.2, and beyond: "faster" code (i.e., exposes a higher degree of
			 * ILP: Instruction-Level Parallelism), but requires more registers. */
			#else
				real value_A[ num_items ], value_Aux[ num_items ];

				#pragma unroll
				for ( index_t i = 0 ; i < num_items ; i++ )
					value_A[ i ] = d_A[ elemIdx[ i ] ];

				#pragma unroll
				for ( index_t i = 0 ; i < num_items ; i++ )
					value_Aux[ i ] = d_Aux[ elemIdx[ i ] ];

				#pragma unroll
				for ( index_t i = 0 ; i < num_items ; i++ )
					value_A[ i ] *= value_Aux[ i ];

				#pragma unroll
				for ( index_t i = 0 ; i < num_items ; i++ )
					value_A[ i ] = device_fdiv( value_A[ i ], val_accum );

				#pragma unroll
				for ( index_t i = 0 ; i < num_items ; i++ )
					d_A[ elemIdx[ i ] ] = value_A[ i ];

			#endif /* Compute Capability */


		// Matrix height is not multiple of <num_items>
		} else {
			#pragma unroll
			for ( index_t i = 0 ; i < (num_items-1) ; i++ )
				if ( elemIdx[ i ] < matrix_size ) {
					real value_A = d_A[ elemIdx[ i ] ];
					real const value_Aux = d_Aux[ elemIdx[ i ] ];
					value_A = device_fdiv( (value_A * value_Aux), val_accum );
					d_A[ elemIdx[ i ] ] = value_A;
				}
		}

	} // Skips instantiated code that actually is never executed.

	#endif /* if __CUDA_ARCH__ > 0 */

} // kernel_mul_div

// ---------------------------------------------

/*
 * d_A[i][j] = d_A[i][j] .* d_Aux[i][j] ./ d_accum_b[j]
 *
 * height <= (dimGrid.y * dimGrid.x) * MUL_DIV__ITEMS_PER_THREAD * block_height
 * Size_of(d_accum_b) >= pitch
 * block_height <= (maxThreadsPerBlock / pitch)
 * dimGrid.x  <= maxGridSizeX
 * dimGrid.y <= MIN( dimGrid.x, maxGridSizeY )
 * "pitch" must be a multiple of 'memory_alignment'.
 */
__host__ void mul_div( real *__restrict__ d_A, real const *__restrict__ d_Aux, real const *__restrict__ d_accum_b, index_t height, index_t pitch,
			index_t block_height, dim3 dimGrid, cudaStream_t stream_A )
{

	// Grid dimensions:
	index_t const grid_length = dimGrid.x;
	index_t const grid_extension = dimGrid.y;

	// Matrix size
	size_t const matrix_size = (size_t) height * (size_t) pitch;

	// Number of loads from global memory performed by each thread at a time.
	index_t const num_items = MUL_DIV__ITEMS_PER_THREAD;

	/* Integer product mode:
	 *
	 * On Compute Capability 1.x, if the number of blocks is greater than
	 * or equal to 2**24, some of the integer products must be performed
	 * with the regular operator "*", which is slower on this architecture.
	 * NOTE: This can happen on EXTENDED grids, only.
	 * Otherwise (i.e., with less blocks), it uses the "fast" mode.
	 *
	 * On Compute Capability >= 2.0, "*" is already the "fast" mode.
	 */
	bool const fast_product = (computeCapability > 1) + (((size_t) grid_extension * (size_t) grid_length) < (1 << 24));

	// Block dimensions
	dim3 const dimBlock( pitch, block_height );

	// --------------------------------

	/* Selects the kernel to launch according to grid and block configurations.
	 *
	 * NOTE: The following conditions will be checked on kernel compilation:
	 *
	 *	- No grid extension is required on Compute Capability >= 3.0, if
	 *		(sizeof(index_t) == sizeof(int)) && (num_items > 2)
	 *
	 *	- Integer products in NON-extended grids are always performed in "fast" mode.
	 */

	// Multi-row blocks
	if ( block_height > 1 ) {

		bool const single_row = false;

		size_t const shmem_size = pitch * sizeof(real);

		// "Extended" grid.
		if ( grid_extension > 1 ) {

			bool const extended_grid = true;

			if ( fast_product )
				kernel_mul_div< extended_grid, single_row, num_items, true >
						<<< dimGrid, dimBlock, shmem_size, stream_A >>>( d_A, d_Aux, d_accum_b, matrix_size );
			else
				kernel_mul_div< extended_grid, single_row, num_items, false >
						<<< dimGrid, dimBlock, shmem_size, stream_A >>>( d_A, d_Aux, d_accum_b, matrix_size );

		// No "grid extension" required
		} else {

			bool const fast_prod = true;

			kernel_mul_div< false, single_row, num_items, fast_prod >
					<<< dimGrid, dimBlock, shmem_size, stream_A >>>( d_A, d_Aux, d_accum_b, matrix_size );

		} // grid extension


	// Single-row blocks
	} else {

		bool const single_row = true;

		size_t const shmem_size = 0;

		// "Extended" grid.
		if ( grid_extension > 1 ) {

			bool const extended_grid = true;

			if ( fast_product )
				kernel_mul_div< extended_grid, single_row, num_items, true >
						<<< dimGrid, dimBlock, shmem_size, stream_A >>>( d_A, d_Aux, d_accum_b, matrix_size );
			else
				kernel_mul_div< extended_grid, single_row, num_items, false >
						<<< dimGrid, dimBlock, shmem_size, stream_A >>>( d_A, d_Aux, d_accum_b, matrix_size );

		// No "grid extension" required
		} else {

			bool const fast_prod = true;

			kernel_mul_div< false, single_row, num_items, fast_prod >
					<<< dimGrid, dimBlock, shmem_size, stream_A >>>( d_A, d_Aux, d_accum_b, matrix_size );

		} // grid extension

	} // Multi- or single-row blocks

} // mul_div

////////////////////////////////////////////////

/*
 * d_A = MAX( d_A, R_MIN )
 *
 * blockDim.x == pitch
 * matrix_size <= ((gridDim.y * gridDim.x) * items_per_thread * blockDim.y) * blockDim.x
 * Size_of(d_accum_b) >= blockDim.x
 * gridDim.y <= gridDim.x
 *
 * ext_grid: 'True' on an "extended" grid (i.e., gridDim.y > 1).
 *		   If set to 'false', forces blockIdx.y = 0 and gridDim.y = 1.
 *
 * items_per_thread: Number of loads from global memory performed by each thread at a time.
 *
 * fast_prod: 'True' to perform integer products in "fast" mode.
 *		On Compute Capability 1.x, it is a 24-bits product, which IGNORES the higher
 *		8 bits of both (32-bits) input operands, and returns the lower 32 bits of
 *		the 48-bits result. This mode requires input operands less than 2**24.
 *		In contrast, in "normal" mode, some integer products are performed with the
 *		regular operator "*", which on Compute Capability 1.x, is compiled into
 *		multiple assembler instructions. This mode must be used when the number of
 *		blocks is greater than or equal to 2**24. In that case, only the related
 *		operations are performed with "*". The rest, are still performed in 24-bits.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		"fast_prod" must be 'false' if (gridDim.x * gridDim.y) >= 2**24
 *
 *	- No code is compiled on the following condition(s), since it would never be executed:
 *
 *		On Compute Capability >= 2.0:
 *			fast_prod == false:
 *				All products are performed with "*", which is already the fastest mode.
 *
 *		On Compute Capability >= 3.0:
 *			(sizeof(index_t) == sizeof(int)) && (items_per_thread > 2) && (ext_grid == true):
 *				On such architectures, maxGridSizeX >= INT_MAX. So, if sizeof(index_t) == sizeof(int),
 *				then (IDX_MAX / maxGridSizeX) <= 2. Therefore, if items_per_thread > 2, then
 *				(maxGridSizeX * items_per_thread) > IDX_MAX >= height
 *
 *		On Compute Capability 1.x:
 *			(fast_prod == ext_grid == false):
 *				maxGridSizeX == maxGridSizeY < 2**16
 */
template < bool ext_grid, index_t items_per_thread, bool fast_prod >
static __global__ void kernel_adjust( real *__restrict__ d_A, size_t matrix_size )
{

	#if __CUDA_ARCH__	/* Reduces work on cudafe(++) when looking for HOST code.*/

	// Skips instantiated code that actually is never executed.
	if (
		#if __CUDA_ARCH__ >= 200
			fast_prod
			#if __CUDA_ARCH__ >= 300
				&& ( (sizeof(index_t) != sizeof(int)) || (items_per_thread <= 2) || (! ext_grid) )
			#endif
		#else
			( fast_prod || ext_grid )
		#endif
	   )
	{

		// Grid "extension"
		bool const grid_extension = ext_grid;

		// Number of items read from global memory by each thread.
		index_t const num_items = items_per_thread;

		// Integer products in "fast" mode.
		bool const fast_product = fast_prod;

		// ------------------------------------------------

		// Thread index
		index_t const tx = threadIdx.x;
		index_t const ty = threadIdx.y;

		// Block index
		index_t const bx = blockIdx.x;
		index_t const by = ( grid_extension ? blockIdx.y : 0 );

		// Block dimensions
		index_t const bdx = blockDim.x;				// == pitch
		index_t const bdy = blockDim.y;

		// Grid dimensions
		index_t const gdx = gridDim.x;

		// ------------------------------------------------

		/* Each threads reads <num_items> elements from d_A[]
		 * with a distance equal to the block size ("bs").
		 */

		// Index to elements.
		size_t elemIdx[ num_items ];

		// Block size
		index_t const bs = idxmul<true>( bdy, bdx );

		// Offset from current block: threadIdx.y * blockDim.x + threadIdx.x
		index_t const offset = idxmul<true>( ty, bdx ) + tx;

		// "Active" block size.
		index_t const act_bs = idxmul<true>( num_items, bs );	// num_items * (blockDim.y * blockDim.x)

		/* Grid layout:
		 *
		 *	By_0:	Bx_0 (width=bdx, height=num_items * bdy)
		 *		Bx_1 (width=bdx, height=num_items * bdy)
		 *		...
		 *		Bx_{GDX-1}
		 *
		 *	By_1:	Bx_0
		 *		Bx_1 (width=bdx, height=num_items * bdy)
		 *	...
		 *
		 * Index of current block:  block_index  = By_i * GDX + Bx_j
		 * Offset to current block: block_offset = block_index  * act_bs
		 * Index to first element:  elemIdx	 = block_offset + offset
		 */

		// First element.
		elemIdx[ 0 ] = idxmul<true>( by, gdx ) + bx;
		elemIdx[ 0 ] = (size_t) idxmul<fast_product>( elemIdx[ 0 ], act_bs ) + offset;

		// Rest of elements.
		#pragma unroll
		for ( index_t i = 1 ; i < num_items ; i++ )
			elemIdx[ i ] = elemIdx[ i-1 ] + bs;

		// ------------------------------------------------

		// Each threads processes (up to) <num_items> elements from global memory.

		if ( elemIdx[ (num_items-1) ] < matrix_size ) {

			/* Compute Capability 1.0 - 1.1 */
			#if __CUDA_ARCH__ < 120

				#pragma unroll
				for ( index_t i = 0 ; i < num_items ; i++ ) {

					real value = d_A[ elemIdx[ i ] ];
					value = device_fmax( value, R_MIN );	// If d_A[] is 'NaN', returns 'R_MIN'

					// if ( isnan( value ) || (value < R_MIN) )
					if ( value == R_MIN )
						d_A[ elemIdx[ i ] ] = value;
				}


			/* Compute Capability 1.2, and beyond: "faster" code (i.e., exposes a higher degree of
			 * ILP: Instruction-Level Parallelism), but requires more registers. */
			#else
				real value[ num_items ];

				#pragma unroll
				for ( index_t i = 0 ; i < num_items ; i++ )
					value[ i ] = d_A[ elemIdx[ i ] ];

				#pragma unroll
				for ( index_t i = 0 ; i < num_items ; i++ )
					value[ i ] = device_fmax( value[ i ], R_MIN );	// If d_A[] is 'NaN', returns 'R_MIN'

				#pragma unroll
				for ( index_t i = 0 ; i < num_items ; i++ )
					if ( value[ i ] == R_MIN )			// ( isnan( value ) || (value < R_MIN) )
						d_A[ elemIdx[ i ] ] = value[ i ];

			#endif /* Compute Capability */


		// Matrix height is not multiple of <num_items>
		} else {

			#pragma unroll
			for ( index_t i = 0 ; i < (num_items-1) ; i++ ) {

				if ( elemIdx[ i ] < matrix_size ) {

					real value = d_A[ elemIdx[ i ] ];
					value = device_fmax( value, R_MIN );	// If d_A[] is 'NaN', returns 'R_MIN'
					if ( value == R_MIN )			// if ( isnan( value ) || (value < R_MIN) )
						d_A[ elemIdx[ i ] ] = value;
				}
			}
		}

	} // Skips instantiated code that actually is never executed.

	#endif /* if __CUDA_ARCH__ > 0 */

} // kernel_adjust

// ---------------------------------------------

/*
 * d_A = MAX( d_A , R_MIN )
 *
 * Adjusts d_A[ height ][ pitch ] to avoid underflow.
 *
 * height <= (dimGrid.y * dimGrid.x) * ADJUST__ITEMS_PER_THREAD * block_height
 * block_height <= (maxThreadsPerBlock / pitch)
 * dimGrid.x  <= maxGridSizeX
 * dimGrid.y <= MIN( dimGrid.x, maxGridSizeY )
 * "pitch" must be a multiple of 'memory_alignment'.
 */
__host__ void adjust( real *__restrict__ d_A, index_t height, index_t pitch, index_t block_height, dim3 dimGrid, cudaStream_t stream_A )
{

	// Grid dimensions:
	index_t const grid_length = dimGrid.x;
	index_t const grid_extension = dimGrid.y;

	// Matrix size
	size_t const matrix_size = (size_t) height * (size_t) pitch;

	// Number of loads from global memory performed by each thread at a time.
	index_t const num_items = ADJUST__ITEMS_PER_THREAD;

	/* Integer product mode:
	 *
	 * On Compute Capability 1.x, if the number of blocks is greater than
	 * or equal to 2**24, some of the integer products must be performed
	 * with the regular operator "*", which is slower on this architecture.
	 * NOTE: This can happen on EXTENDED grids, only.
	 * Otherwise (i.e., with less blocks), it uses the "fast" mode.
	 *
	 * On Compute Capability >= 2.0, "*" is already the "fast" mode.
	 */
	bool const fast_product = (computeCapability > 1) + (((size_t) grid_extension * (size_t) grid_length) < (1 << 24));

	// Block dimensions
	dim3 const dimBlock( pitch, block_height );

	size_t const shmem_size = 0;

	// --------------------------------

	/* Selects the kernel to launch according to grid configuration.
	 *
	 * NOTE: The following conditions will be checked on kernel compilation:
	 *
	 *	- No grid extension is required on Compute Capability >= 3.0, if
	 *		(sizeof(index_t) == sizeof(int)) && (num_items > 2)
	 *
	 *	- Integer products in NON-extended grids are always performed in "fast" mode.
	 */

	// "Extended" grid.
	if ( grid_extension > 1 ) {

		bool const extended_grid = true;

		if ( fast_product )
			kernel_adjust< extended_grid, num_items, true >
					<<< dimGrid, dimBlock, shmem_size, stream_A >>>( d_A, matrix_size );
		else
			kernel_adjust< extended_grid, num_items, false >
					<<< dimGrid, dimBlock, shmem_size, stream_A >>>( d_A, matrix_size );

	// No "grid extension" required
	} else {

		bool const fast_prod = true;

		kernel_adjust< false, num_items, fast_prod >
				<<< dimGrid, dimBlock, shmem_size, stream_A >>>( d_A, matrix_size );

	} // grid extension

} // adjust

////////////////////////////////////////////////

/*
 * Helper method of kernel_idx_max().
 *
 * Computes the maximum value in a portion of d_A and its column index.
 * That is, returns "max_val_idx", and "max_val" such that:
 *	max_val == d_A[i][max_val_idx] == max( d_A[i][...] ),
 * where
 *	0 <= max_val_idx <= width <= pitch
 *
 * (blockDim.x * items_per_thread) >= width
 * matrix_size <= ((gridDim.y * gridDim.x) * blockDim.y) * pitch
 * gridDim.y <= gridDim.x
 *
 * ext_grid: 'True' on an "extended" grid (i.e., gridDim.y > 1).
 *		   If set to 'false', forces blockIdx.y = 0 and gridDim.y = 1.
 *
 * block_width: Overrides blockDim.x.
 *		Disabled/ignored if set to 0.
 *
 * items_per_thread: Number of loads from global memory performed by each thread at a time.
 *
 * fast_prod: 'True' to perform integer products in "fast" mode.
 *		On Compute Capability 1.x, it is a 24-bits product, which IGNORES the higher
 *		8 bits of both (32-bits) input operands, and returns the lower 32 bits of
 *		the 48-bits result. This mode requires input operands less than 2**24.
 *		In contrast, in "extended" mode, some integer products are performed with
 *		the regular operator "*", which on Compute Capability 1.x, is compiled into
 *		multiple assembler instructions. This mode must be used when the number of
 *		blocks is greater than or equal to 2**24. In that case, only the related
 *		operations are performed with "*". The rest, are still performed in 24-bits.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		"fast_prod" must be 'false' if (gridDim.x * gridDim.y) >= 2**24
 *
 *	- No code is compiled on the following condition(s), since it would never be executed:
 *
 *		(block_width > 0) && (block_width != warpSize)
 *
 *		On Compute Capability >= 2.0:
 *			fast_prod == false:
 *				All products are performed with "*", which is already the fastest mode.
 *
 *		On Compute Capability >= 3.0:
 *			(sizeof(index_t) == sizeof(int)) && (items_per_thread > 2) && (ext_grid == true):
 *				On such architectures, maxGridSizeX >= INT_MAX. So, if sizeof(index_t) == sizeof(int),
 *				then (IDX_MAX / maxGridSizeX) <= 2. Therefore, if items_per_thread > 2, then
 *				(maxGridSizeX * items_per_thread) > IDX_MAX >= height
 *
 *		On Compute Capability 1.x:
 *			(block_width > 0) && (block_width != warpSize) && (block_width != 16)
 *
 *			(fast_prod == ext_grid == false):
 *				maxGridSizeX == maxGridSizeY < 2**16
 */
template < bool ext_grid, index_t block_width, index_t items_per_thread, bool fast_prod >
static __device__ void idx_max_gmem( real const *__restrict__ d_A, index_t width, index_t pitch, size_t matrix_size,
					real *__restrict__ max_val, index_t *__restrict__ max_val_idx )
{

	#if __CUDA_ARCH__	/* Reduces work on cudafe(++) when looking for HOST code.*/

	if (
		#if __CUDA_ARCH__ >= 200
			( (! block_width) || (block_width == warpSize) )
			&& fast_prod
			#if __CUDA_ARCH__ >= 300
				&& ( (sizeof(index_t) != sizeof(int)) || (items_per_thread <= 2) || (! ext_grid) )
			#endif
		#else
			( (! block_width) || (block_width == warpSize) || (block_width == 16) )
			&& ( fast_prod || ext_grid )
		#endif
	   )
	{

		// Grid "extension"
		bool const grid_extension = ext_grid;

		// Number of items read from global memory by each thread.
		index_t const num_items = items_per_thread;

		// Integer products in "fast" mode.
		bool const fast_product = fast_prod;

		// ------------------------------------------------

		// Thread index
		index_t const tx = threadIdx.x;
		index_t const ty = threadIdx.y;

		// Block index
		index_t const bx = blockIdx.x;
		index_t const by = ( grid_extension ? blockIdx.y : 0 );

		// Block dimensions
		index_t const bdx = ( block_width ? block_width : blockDim.x );	// (pitch/items_per_thread) <= bdx <= pitch
		index_t const bdy = blockDim.y;

		// Grid dimensions
		index_t const gdx = gridDim.x;

		// ------------------------------------------------

		/* Each threads processes <num_items> elements from d_A[]
		 * which have a distance equal to the block width ("bdx").
		 *
		 * Magnitudes computed with "pitch" (rather than "bdx"),
		 * are named "active".
		 */

		// "Active" block size: blockDim.y * pitch
		index_t const act_bs = idxmul<true>( bdy, pitch );	// (pitch/items_per_thread) <= bdx <= pitch

		// "Active" offset within the block to current row: threadIdx.y * pitch
		index_t const act_offset_from_block = idxmul<true>( ty, pitch );

		/* Grid layout:
		 *
		 *	By_0:	Bx_0 (width=bdx, active_width=pitch, height=bdy)
		 *		Bx_1 (width=bdx, active_width=pitch, height=bdy)
		 *		...
		 *		Bx_{GDX-1}
		 *
		 *	By_1:	Bx_0
		 *		Bx_1 (width=bdx, active_width=pitch, height=bdy)
		 *	...
		 *
		 * Index of current block:  block_index	 = By_i * GDX + Bx_j
		 * Offset to current block: block_offset = block_index  * act_bs
		 * Offset to current row:   row_offset	 = block_offset + act_offset_from_block
		 * Index of element:	    row_offset + tx
		 */

		size_t row_offset = idxmul<true>( by, gdx ) + bx;
		row_offset = (size_t) idxmul<fast_product>( row_offset, act_bs ) + act_offset_from_block;

		// Column index of elements.
		index_t colIdx[ num_items ];
		colIdx[ 0 ] = tx;

		#pragma unroll
		for ( index_t i = 1 ; i < num_items ; i++ )
			colIdx[ i ] = colIdx[ i-1 ] + bdx;

		// ------------------------------------------------

		// Each threads processes (up to) <num_items> elements from global memory.

		// Maximum value and its column index
		real l_max_val = REAL_C( -1.0 );
		index_t l_max_val_idx = INDEX_C( 0 );

		if ( row_offset < matrix_size ) {

			if ( colIdx[ (num_items-1) ] < width ) {

				// First value
				l_max_val = d_A[ row_offset + colIdx[ 0 ] ];
				l_max_val_idx = colIdx[ 0 ];

				// Rest of values
				real value[ (num_items-1) ];

				#pragma unroll
				for ( index_t i = 0 ; i < (num_items-1) ; i++ )
					value[ i ] = d_A[ row_offset + colIdx[ (i+1) ] ];

				#pragma unroll
				for ( index_t i = 0 ; i < (num_items-1) ; i++ )
					if ( value[ i ] > l_max_val ) {
						l_max_val = value[ i ];
						l_max_val_idx = colIdx[ (i+1) ];
					}

			// Matrix width is not multiple of <num_items>
			} else if ( colIdx[ 0 ] < width ) {

				l_max_val = d_A[ row_offset + colIdx[ 0 ] ];
				l_max_val_idx = colIdx[ 0 ];

				#pragma unroll
				for ( index_t i = 1 ; i < (num_items-1) ; i++ ) {
					real const value = d_A[ row_offset + colIdx[ i ] ];
					if ( colIdx[ i ] < width ) {
						if ( value > l_max_val ) {
							l_max_val = value;
							l_max_val_idx = colIdx[ i ];
						}
					}
				}
			}

		} // if ( row_offset < matrix_size )

		// Returns selected values
		*max_val = l_max_val;
		*max_val_idx = l_max_val_idx;

	} // Skips instantiated code that actually is never executed.

	#endif /* if __CUDA_ARCH__ > 0 */

} // idx_max_gmem

// ---------------------------------------------

/*
 * Helper method of idx_max_shmem().
 *
 * This helper method computes the maximum value in shared memory, within a warp.
 * Therefore, no synchronization is performed. Instead, it uses volatile pointers.
 *
 * block_width: A power of 2 that overrides blockDim.x.
 *		Uses the "Warp-size mode" when 0 < block_width == warpSize
 *		In addition, on Compute Capability 1.x, if block_width == 16 (i.e warpSize/2).
 *		it is denoted as "half-warpsize mode".
 *		Disabled/ignored if set to 0.
 *
 * WARNING:
 *
 *	- The operation is only performed between the lower 32 threads within a warp.
 *
 *	- block_width == 0, implies that blockDim.x > warpSize.
 *
 *	- No code is compiled on the following conditions, since it would never be executed:
 *
 *		(block_width > 0) && (block_width != warpSize)
 *
 *		On Compute Capability 1.x
 *			(block_width > 0) && (block_width != warpSize) && (block_width != 16)
 */
template < index_t block_width >
static __device__ void idx_max_shmem_warp( index_t bs, index_t offset, index_t *__restrict__ maxValIdx, rp2shm_t maxVal,
						real *__restrict__ max_val, index_t *__restrict__ max_val_idx )
{

	#if __CUDA_ARCH__	/* Reduces work on cudafe(++) when looking for HOST code.*/

	// Skips instantiated code that actually is never executed.
	if ( (! block_width) || (block_width == warpSize)
		#if __CUDA_ARCH__ < 200
			|| (block_width == 16)
		#endif
	   )
	{

		// ------------------------------------------------

		/* Uses the "Warp-size mode" if 0 < block_width == warpSize
		 * On Compute Capability 1.x, it may be also 16 == (warpSize/2).
		 */
		bool const warpSize_mode = block_width;


		/* "half-warpSize mode" on Compute Capability 1.x,
		 * if block_width == 16 (i.e., warpSize/2)
		 */
		#if __CUDA_ARCH__ < 200
			bool const half_warpSize = ( block_width == 16 );
		#else
			bool const half_warpSize = false;
		#endif

		// ------------------------------------------------

		// Thread index
		index_t const tx = threadIdx.x;

		// ------------------------------------------------

		// Current Maximum value and its column index.
		real l_max_val = *max_val;
		index_t l_max_val_idx  = *max_val_idx;

		// ------------------------------------------------

		/* In warpSize mode, all threads must initialize the
		 * shared memory:
		 *	maxVal[ty][tx] and maxValIdx[ty][tx]
		 */

		if ( warpSize_mode ) {
			index_t volatile *const v_maxValIdx = maxValIdx;
			v_maxValIdx[ offset ] = l_max_val_idx;
			store_into_rp2shm<true>( l_max_val, maxVal, offset );
		}

		// ------------------------------------------------

		// half = 16
		if ( ! half_warpSize ) {

			index_t const half = 16;

			index_t volatile *const v_maxValIdx = maxValIdx;

			// l_max_val = load_from_rp2shm< true >( maxVal, offset ); // maxVal[ty][tx] is already in l_max_val
			// l_max_val_idx = v_maxValIdx[ offset ];		   // maxValIdx[ty][tx] is already in l_max_val_idx

			if ( tx < half ) {

				real const val = load_from_rp2shm<true>( maxVal, (offset + half) ); // maxVal[ty][tx+half]
				index_t const idx = v_maxValIdx[ (offset + half) ];		    // maxValIdx[ty][tx+half]

				/* Updates the local variables, and also stores the new values in
				 * shared memory, in maxVal[ty][tx] and maxValIdx[ty][tx], respectively.
				 */
				if ( l_max_val < val ) {
					l_max_val = val;
					l_max_val_idx = idx;

					store_into_rp2shm<true>( val, maxVal, offset );
					v_maxValIdx[ offset ] = idx;
				}
			}

		} // half = 16

		// ------------------------------------------------

		// half = 8, 4, 2 (i.e., 3 times)
		#pragma unroll
		for ( index_t i = 0, bw = 16 ; i < 3 ; i++, bw >>= 1 ) {

			index_t const half = ( bw >> 1 );

			index_t volatile *const v_maxValIdx = maxValIdx;

			// l_max_val = load_from_rp2shm< true >( maxVal, offset ); // maxVal[ty][tx] is already in l_max_val
			// l_max_val_idx = v_maxValIdx[ offset ];		   // maxValIdx[ty][tx] is already in l_max_val_idx

			if ( tx < half ) {

				real const val = load_from_rp2shm<true>( maxVal, (offset + half) ); // maxVal[ty][tx+half]
				index_t const idx = v_maxValIdx[ (offset + half) ];		    // maxValIdx[ty][tx+half]

				/* Updates the local variables, and also stores the new values in
				 * shared memory, in maxVal[ty][tx] and maxValIdx[ty][tx], respectively.
				 */
				if ( l_max_val < val ) {
					l_max_val = val;
					l_max_val_idx = idx;

					store_into_rp2shm<true>( val, maxVal, offset );
					v_maxValIdx[ offset ] = idx;
				}
			}

		} // for half = 8 .. 2

		// ------------------------------------------------

		// half == 1
		{

			index_t const half = 1;

			index_t volatile *const v_maxValIdx = maxValIdx;

			// l_max_val = load_from_rp2shm< true >( maxVal, offset ); // maxVal[ty][tx] is already in l_max_val
			// l_max_val_idx = v_maxValIdx[ offset ];		   // maxValIdx[ty][tx] is already in l_max_val_idx

			if ( ! tx ) {	// ( tx < half )

				real const val = load_from_rp2shm<true>( maxVal, (offset + half) ); // maxVal[ty][tx+half]
				index_t const idx = v_maxValIdx[ offset + half ];		    // maxValIdx[ty][tx+half]

				/* Updates the local variables. It is not necessary to
				 * store them in shared memory.
				 */
				if ( l_max_val < val ) {
					l_max_val = val;
					l_max_val_idx = idx;

					// store_into_rp2shm<true>( val, maxVal, offset );	// Not necessary
					// v_maxValIdx[ offset ] = idx;				// Not necessary
				}
			}

		} // half == 1

		// ------------------------------------------------

		// Returns the selected values
		*max_val = l_max_val;
		*max_val_idx = l_max_val_idx;

	} // Skips instantiated code that actually is never executed.

	#endif /* __CUDA_ARCH__ */

} // idx_max_shmem_warp

// ---------------------------------------------

/*
 * Helper method of idx_max_shmem().
 *
 * Computes the maximum value stored in a shared-memory block.
 * It also returns the corresponding column index, which is also stored in the shared-memory block.
 * That is, returns "max_val_idx", and "max_val" such that:
 *	max_val = shmem[i][j] == max( shmem[i][...] )
 *	max_val_idx = shmem[ i + bs ][j],
 * where
 *	0 <= max_val_idx <= width <= pitch
 *
 * Size of shared-memory block: bs * ( sizeof(index_t) + sizeof(real) ),
 * where bs = (blockDim.x * blockDim.y).
 *
 * block_width: A power of 2 that overrides blockDim.x.
 *		Uses the "Warp-size mode" when 0 < block_width == warpSize
 *		In addition, on Compute Capability 1.x, if block_width == 16 (i.e warpSize/2),
 *		it is denoted as "half-warpsize mode".
 *		Disabled/ignored if set to 0.
 *
 * WARNING:
 *
 *	- block_width == 0, implies that blockDim.x > warpSize.
 *
 *	- No code is compiled on the following condition(s), since it would never be executed:
 *
 *		(block_width > 0) && (block_width != warpSize)
 *
 *		On Compute Capability 1.x
 *			(block_width > 0) && (block_width != warpSize) && (block_width != 16)
 */
template < index_t block_width >
static __device__ void idx_max_shmem( real *__restrict__ max_val, index_t *__restrict__ max_val_idx )
{

	#if __CUDA_ARCH__	/* Reduces work on cudafe(++) when looking for HOST code.*/

	// Skips instantiated code that actually is never executed.
	if ( (! block_width) || (block_width == warpSize)
		#if __CUDA_ARCH__ < 200
			|| (block_width == 16)
		#endif
	   )
	{

		// ------------------------------------------------

		/* Uses the "Warp-size mode" if 0 < block_width == warpSize
		 * On Compute Capability 1.x, it may be also 16 == (warpSize/2).
		 */
		bool const warpSize_mode = block_width;

		// ------------------------------------------------

		// Thread index
		index_t const tx = threadIdx.x;
		index_t const ty = threadIdx.y;

		// Block dimensions
		index_t const bdx = ( block_width ? block_width : blockDim.x );
		index_t const bdy = blockDim.y;

		// ------------------------------------------------

		// Current Maximum value and its column index.
		real l_max_val = *max_val;
		index_t l_max_val_idx  = *max_val_idx;

		// Block size: blockDim.y * blockDim.x
		index_t const bs = idxmul<true>( bdy, bdx );

		// Offset from current block: threadIdx.y * blockDim.x + threadIdx.x
		index_t const offset = idxmul<true>( ty, bdx ) + tx;

		// ------------------------------------------------

		/* Shared memory:
		 *	max_values[ bs ]: maximum values.
		 *	idx_max_values[ bs ]: column index of maximum values.
		 *
		 * Below the warp size, data will be loaded/stored through
		 * volatile pointers.
		 */
		extern __shared__ real smem[];
		real *__restrict__ const psmem = smem;
		index_t *const maxValIdx = (index_t *) psmem;
		rp2shm_t maxVal = new_rp2shm( &maxValIdx[ bs ], bs );


		// ------------------------------------------------

		/* The following operations are performed only if not in warpSize mode.
		 * That is, if bdx > warpSize
		 */
		if ( ! warpSize_mode ) {

			/* All threads initialize the shared memory:
			 *	maxVal[ty][tx] and maxValIdx[ty][tx]
			 *
			 * (although it is required only for those whose ty >= bdx/2).
			 */
			maxValIdx[ offset ] = l_max_val_idx;
			store_into_rp2shm<false>( l_max_val, maxVal, offset );

			// ------------------------------------------------

			/* Computes the maximum value.
			 *
			 * for ( index_t half = bdx/2 ; half > 0 ; half /= 2 ) {	// bdx is a power of 2.
			 *
			 *	__syncthreads();
			 *
			 *	// Threads on the left half read values from the right section.
			 *	if ( tx < half ) && ( maxVal[ty][tx] < maxVal[ty][tx+half] ) {
			 *
			 *		maxVal[ty][tx] = maxVal[ty][tx+half];
			 *
			 *		maxValIdx[ty][tx] = maxValIdx[ty][tx+half];
			 *	}
			 * }
			 *
			 * Please note that position [ty][tx] is referenced with: offset == ty*bdx + tx.
			 */

			// half = bdx/2 ... 2*warpSize
			for ( index_t half = (bdx >> 1) ; half > warpSize ; half >>= 1 ) {	// bdx is a power of 2.

				__syncthreads();

				// l_max_val = load_from_rp2shm< false >( maxVal, offset ); // maxVal[ty][tx] is already in l_max_val
				// l_max_val_idx = maxValIdx[ offset ];			    // maxValIdx[ty][tx] is already in l_max_val_idx

				if ( tx < half ) {

					real const val = load_from_rp2shm<false>( maxVal, (offset + half) ); // maxVal[ty][tx+half]
					index_t const idx = maxValIdx[ offset + half ];			     // maxValIdx[ty][tx+half]

					/* Updates the local variables, and also stores the new values in
					* shared memory, in maxVal[ty][tx] and maxValIdx[ty][tx], respectively.
					*/
					if ( l_max_val < val ) {
						l_max_val = val;
						l_max_val_idx = idx;

						store_into_rp2shm<false>( val, maxVal, offset );
						maxValIdx[ offset ] = idx;
					}
				}

			} // half = (bdx/2) ... (2*warpSize)

			// ------------------------------------------------

			// half = warpSize
			{

				// Waits until the last write to shared memory has been completed.
				__syncthreads();


				index_t const half = warpSize;	// <= (bdx/2)

				/* No synchronization is required below the warp size.
				 * Instead, it uses "volatile" pointers to shared memory.
				 */
				index_t volatile *const v_maxValIdx = maxValIdx;

				// l_max_val = load_from_rp2shm< true >( maxVal, offset ); // maxVal[ty][tx] is already in l_max_val
				// l_max_val_idx = v_maxValIdx[ offset ];		   // maxValIdx[ty][tx] is already in l_max_val_idx

				if ( tx < half ) {
					real const val = load_from_rp2shm<true>( maxVal, (offset + half) ); // maxVal[ty][tx+half]
					index_t const idx = v_maxValIdx[ (offset + half) ];		    // maxValIdx[ty][tx+half]

					/* Updates the local variables.
					 *
					 * On Compute Capability < 3.0, the new values are also stored in
					 * shared memory, in maxVal[ty][tx] and maxValIdx[ty][tx], respectively.
					 * This is not required on devices with higher capability, since
					 * warp-shuffle functions will be used.
					 */
					if ( l_max_val < val ) {
						l_max_val = val;
						l_max_val_idx = idx;

						#if __CUDA_ARCH__ < 300
							store_into_rp2shm<true>( val, maxVal, offset );
							v_maxValIdx[ offset ] = idx;
						#endif
					}
				}

			} // half = warpSize

		} // NOT in warpSize mode ( half = bdx/2 ... warpSize )

		// ------------------------------------------------

		// half = warpSize/2 .. 1

		#if __CUDA_ARCH__ < 300

			/* Compute Capabilities 1.x and 2.x */

			/* Uses volatile pointers to shared memory.
			 * NOTE: Assumes warpSize == 32.
			 */
			idx_max_shmem_warp<block_width>( bs, offset, maxValIdx, maxVal, &l_max_val, &l_max_val_idx );


		#else	/* Compute Capability >= 3.0 */

			/* Shared memory is not any more required. Instead, it uses warp shuffle functions.
			 *
			 * NOTE:
			 * Unfortunately, "warpSize" is not recognized as a constant for loop unrolling
			 * (at least on CUDA <= 5.5). Therefore, the required loop is split in two ranges
			 * ([(warpSize/2)..32] and [16..1]) in order to guarantee that the latter will be
			 * unrolled. In contrast, the former will not be even compiled until warpSize > 32.
			 */

			// half = (warpSize / 2) .. 32
			#pragma unroll
			for ( index_t half = (warpSize >> 1) ; half > 16 ; half >>= 1 ) {

				real const val = real_shfl_xor( l_max_val, half );
				index_t const idx = (index_t) __shfl_xor( (int) l_max_val_idx, (int) half );

				if ( l_max_val < val ) {
					l_max_val = val;
					l_max_val_idx = idx;
				}

			} // half = (warpSize / 2) .. 32

			// half = 16 .. 1
			#pragma unroll
			for ( index_t half = 16 ; half > 0 ; half >>= 1 ) {

				real const val = real_shfl_xor( l_max_val, half );
				index_t const idx = (index_t) __shfl_xor( (int) l_max_val_idx, (int) half );

				if ( l_max_val < val ) {
					l_max_val = val;
					l_max_val_idx = idx;
				}

			} // half = 16 .. 1

		#endif	/* half = (warpSize/2) .. 1, for Compute Capability < 3.0 */

		// ------------------------------------------------

		// Returns the selected values
		*max_val = l_max_val;
		*max_val_idx = l_max_val_idx;

	} // Skips instantiated code that actually is never executed.

	#endif /* __CUDA_ARCH__ */

} // idx_max_shmem

// ---------------------------------------------

/*
 * Computes the maximum value of each row in d_A[] and stores its column index in d_Idx[].
 * That is, returns d_Idx[i], such that:
 *	d_A[i][ d_Idx[i] ] == max( d_A[i][...] ).
 *
 * blockDim.x <= width <= pitch
 * matrix_size <= (gridDim.y * gridDim.x * blockDim.y) * pitch
 * Size_of(d_Idx) >= (gridDim.y * gridDim.x * blockDim.y)
 * gridDim.y <= gridDim.x
 *
 * ext_grid: 'True' on an "extended" grid (i.e., gridDim.y > 1).
 *		   If set to 'false', forces blockIdx.y = 0 and gridDim.y = 1.
 *
 * block_width: Overrides blockDim.x.
 *		Uses the "Warp-size mode" when 0 < block_width == warpSize
 *		Nevertheless, on Compute Capability 1.x, block_width may be 16 (i.e warpSize/2).
 *		This will be denoted as "half-warpsize mode".
 *		Disabled/ignored if set to 0.
 *
 * items_per_thread: Number of loads from global memory performed by each thread at a time.
 *
 * fast_prod: 'True' to perform integer products in "fast" mode.
 *		On Compute Capability 1.x, it is a 24-bits product, which IGNORES the higher
 *		8 bits of both (32-bits) input operands, and returns the lower 32 bits of
 *		the 48-bits result. This mode requires input operands less than 2**24.
 *		In contrast, in "extended" mode, some integer products are performed with
 *		the regular operator "*", which on Compute Capability 1.x, is compiled into
 *		multiple assembler instructions. This mode must be used when the number of
 *		blocks is greater than or equal to 2**24. In that case, only the related
 *		operations are performed with "*". The rest, are still performed in 24-bits.
 *
 * Size of shared-memory block: bs * ( sizeof(index_t) + sizeof(real) ),
 * where bs = (blockDim.x * blockDim.y).
 *
 * WARNING:
 *
 *	- block_width == 0, implies that blockDim.x > warpSize.
 *
 *
 *	- On Compute Capability 1.x:
 *		"fast_prod" must be 'false' if (gridDim.x * gridDim.y) >= 2**24
 *
 *
 *	- No code is compiled on the following condition(s), since it would never be executed:
 *
 *		(block_width > 0) && (block_width != warpSize)
 *
 *		On Compute Capability >= 2.0:
 *			fast_prod == false:
 *				All products are performed with "*", which is already the fastest mode.
 *
 *		On Compute Capability >= 3.0:
 *			(sizeof(index_t) == sizeof(int)) && (items_per_thread > 2) && (ext_grid == true):
 *				On such architectures, maxGridSizeX >= INT_MAX. So, if sizeof(index_t) == sizeof(int),
 *				then (IDX_MAX / maxGridSizeX) <= 2. Therefore, if items_per_thread > 2, then
 *				(maxGridSizeX * items_per_thread) > IDX_MAX >= height
 *
 *		On Compute Capability 1.x:
 *			(block_width > 0) && (block_width != warpSize) && (block_width != 16)
 *
 *			(fast_prod == ext_grid == false):
 *				maxGridSizeX == maxGridSizeY < 2**16
 */
template < bool ext_grid, index_t block_width, index_t items_per_thread, bool fast_prod >
static __global__ void kernel_idx_max( real const *__restrict__ d_A, index_t width, index_t pitch, size_t matrix_size,
					index_t *__restrict__ d_Idx )
{

	#if __CUDA_ARCH__	/* Reduces work on cudafe(++) when looking for HOST code.*/

	// Skips instantiated code that actually is never executed.
	if (
		#if __CUDA_ARCH__ >= 200
			( (! block_width) || (block_width == warpSize) )
			&& fast_prod
			#if __CUDA_ARCH__ >= 300
				&& ( (sizeof(index_t) != sizeof(int)) || (items_per_thread <= 2) || (! ext_grid) )
			#endif
		#else
			( (! block_width) || (block_width == 32) || (block_width == 16) )
			&& ( fast_prod || ext_grid )
		#endif
	   )
	{

		// Grid "extension"
		bool const grid_extension = ext_grid;

		// Number of items read from global memory by each thread.
		index_t const num_items = items_per_thread;

		// Integer products in "fast" mode.
		bool const fast_product = fast_prod;

		// ------------------------------------------------

		// Thread index
		index_t const tx = threadIdx.x;
		index_t const ty = threadIdx.y;

		// Block index
		index_t const bx = blockIdx.x;
		index_t const by = ( grid_extension ? blockIdx.y : 0 );

		// Block dimensions
		index_t const bdy = blockDim.y;

		// Grid dimensions
		index_t const gdx = gridDim.x;

		// ------------------------------------------------

		// Maximum value and its column index
		real max_val = REAL_C( -1.0 );
		index_t max_val_idx = INDEX_C( 0 );

		// Each threads computes the maximum value of a portion from d_A[].
		idx_max_gmem< grid_extension, block_width, num_items, fast_product >
				( d_A, width, pitch, matrix_size, &max_val, &max_val_idx );

		// Each thread stores its selected values in shared memory and computes the maximum.
		idx_max_shmem< block_width >( &max_val, &max_val_idx );

		// ------------------------------------------------

		/* Threads with tx == 0 store their selected index d_Idx[ rowIdx ], where
		 * rowIdx = (by * gdx + bx) * bdy + ty
		 */

		/* Grid layout:
		 *
		 *	By_0:	Bx_0 (height=bdy)
		 *		Bx_1 (height=bdy)
		 *		...
		 *		Bx_{GDX-1}
		 *
		 *	By_1:	Bx_0
		 *		Bx_1 (height=bdy)
		 *	...
		 *
		 * Index of current block: block_index = By_i * GDX + Bx_j
		 * Index of current row:	rowIdx = block_index * bdy + ty
		 */
		index_t rowIdx = idxmul<true>( by, gdx ) + bx;
		rowIdx = idxmul<fast_product>( rowIdx, bdy ) + ty;

		if ( ! tx )
			d_Idx[ rowIdx ] = max_val_idx;

	} // Skips instantiated code that actually is never executed.

	#endif /* if __CUDA_ARCH__ > 0 */

} // kernel_idx_max

// ---------------------------------------------

/*
 * Computes the maximum value of each row in d_A[] and stores its column index in d_Idx[].
 * That is, returns d_Idx[i], such that:
 *	d_A[i][ d_Idx[i] ] == max( d_A[i][...] ).
 * where
 *	0 <= max_val_idx <= width <= pitch
 *
 * height <= (dimGrid.y * dimGrid.x) * dimBlock.y <= size_of( d_Idx )
 * dimBlock.x must be a power of 2, and <= maxThreadsPerBlock
 * dimBlock.y <= (maxThreadsPerBlock / pitch).
 * dimGrid.x  <= maxGridSizeX
 * dimGrid.y <= MIN( dimGrid.x, maxGridSizeY )
 */
__host__ void idx_max( real const *__restrict__ d_A, index_t height, index_t width, index_t pitch, dim3 dimBlock, dim3 dimGrid,
			cudaStream_t stream_A, index_t *__restrict__ d_Idx )
{

	// Block dimensions:
	index_t const block_width = dimBlock.x;
	index_t const block_height = dimBlock.y;

	// Grid dimensions:
	index_t const grid_length = dimGrid.x;
	index_t const grid_extension = dimGrid.y;

	// Matrix size
	size_t const matrix_size = (size_t) height * (size_t) pitch;

	// Number of loads from global memory performed by each thread at a time.
	index_t const num_items = IDX_MAX__ITEMS_PER_THREAD;

	/* Integer product mode:
	 *
	 * On Compute Capability 1.x, if the number of blocks is greater than
	 * or equal to 2**24, some of the integer products must be performed
	 * with the regular operator "*", which is slower on this architecture.
	 * NOTE: This can happen on EXTENDED grids, only.
	 * Otherwise (i.e., with less blocks), it uses the "fast" mode.
	 *
	 * On Compute Capability >= 2.0, "*" is already the "fast" mode.
	 */
	bool const fast_product = (computeCapability > 1) + (((size_t) grid_extension * (size_t) grid_length) < (1 << 24));

	// --------------------------------

	/* Selects the kernel to launch according to grid and block configurations.
	 *
	 * NOTE: The following conditions will be checked on kernel compilation:
	 *
	 *	- No grid extension is required on Compute Capability >= 3.0, if
	 *		(sizeof(index_t) == sizeof(int)) && (num_items > 2)
	 *
	 *	- Integer products in NON-extended grids are always performed in "fast" mode.
	 */

	switch ( block_width ) {

		/* NOTE:
		 *	For higher values, the template parameter is set to 0,
		 *	so the kernel uses the "blockDim.x" built-in variable.
		 */
		default: {

			size_t const shmem_size = block_width * block_height * (sizeof(real) + sizeof(index_t));
			dim3 const dimBlock( block_width, block_height );
			index_t const bw = 0;

			if ( grid_extension > 1 ) {

				bool const extended_grid = true;

				if ( fast_product )
					kernel_idx_max< extended_grid, bw, num_items, true >
								<<< dimGrid, dimBlock, shmem_size, stream_A >>>
									( d_A, width, pitch, matrix_size, d_Idx );
				else
					kernel_idx_max< extended_grid, bw, num_items, false >
								<<< dimGrid, dimBlock, shmem_size, stream_A >>>
									( d_A, width, pitch, matrix_size, d_Idx );
			// No grid extension
			} else {
				bool const fast_prod = true;
				kernel_idx_max< false, bw, num_items, fast_prod >
							<<< dimGrid, dimBlock, shmem_size, stream_A >>>
								( d_A, width, pitch, matrix_size, d_Idx );
			}
		} break;

		case 32: {	// Typical warp size for most (or all) architectures.

			index_t const bw = 32;
			size_t const shmem_size = bw * block_height * (sizeof(real) + sizeof(index_t));
			dim3 const dimBlock( bw, block_height );

			if ( grid_extension > 1 ) {

				bool const extended_grid = true;

				if ( fast_product )
					kernel_idx_max< extended_grid, bw, num_items, true >
								<<< dimGrid, dimBlock, shmem_size, stream_A >>>
									( d_A, width, pitch, matrix_size, d_Idx );
				else
					kernel_idx_max< extended_grid, bw, num_items, false >
								<<< dimGrid, dimBlock, shmem_size, stream_A >>>
									( d_A, width, pitch, matrix_size, d_Idx );
			// No grid extension
			} else {
				bool const fast_prod = true;
				kernel_idx_max< false, bw, num_items, fast_prod >
							<<< dimGrid, dimBlock, shmem_size, stream_A >>>
								( d_A, width, pitch, matrix_size, d_Idx );
			}
		} break;

		case 16: { // On Compute Capability 1.x, this is the half of the warp size.

			index_t const bw = 16;
			size_t const shmem_size = bw * block_height * (sizeof(real) + sizeof(index_t));
			dim3 const dimBlock( bw, block_height );

			if ( grid_extension > 1 ) {

				bool const extended_grid = true;

				if ( fast_product )
					kernel_idx_max< extended_grid, bw, num_items, true >
								<<< dimGrid, dimBlock, shmem_size, stream_A >>>
									( d_A, width, pitch, matrix_size, d_Idx );
				else
					kernel_idx_max< extended_grid, bw, num_items, false >
								<<< dimGrid, dimBlock, shmem_size, stream_A >>>
									( d_A, width, pitch, matrix_size, d_Idx );
			// No grid extension
			} else {
				bool const fast_prod = true;
				kernel_idx_max< false, bw, num_items, fast_prod >
							<<< dimGrid, dimBlock, shmem_size, stream_A >>>
								( d_A, width, pitch, matrix_size, d_Idx );
			}
		} break;

	} // switch

} // idx_max

////////////////////////////////////////////////
////////////////////////////////////////////////
