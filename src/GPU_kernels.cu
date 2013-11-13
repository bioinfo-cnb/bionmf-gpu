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
 * GPU_kernels.cu
 *	Kernel code to be executed on the device (GPU).
 *
 * NOTE:
 *	- All matrices include useless data for padding. It is sometimes denoted as 'pitch'.
 *
 *********************************************************/

#include "GPU_kernels.cuh"

// =======================================================================

#if __CUDA_ARCH__ >= 110	/* Compute Capability >= 1.1 */

	/* Global variable used in kernel_reduce_to_row().
	 *	Please see the threadfenceReduction example in CUDA SDK for details.
	 */
	__device__ unsigned int retirement_count = 0;
#endif

// =======================================================================
// =======================================================================

/*
 * Helper method of kernel_reduce_to_row().
 *
 * Reduces a portion of d_A[] to a row. That is, returns sum( d_A[...][tx] ).
 *
 * block size: bs = blockDim.x * blockDim.y
 * matrix_size <= gridDim.y * gridDim.x * items_per_thread * bs
 * gridDim.y <= gridDim.x
 *
 * ext_grid:	'True' on a "extended" grid (i.e., gridDim.y > 1).
 *		If set to 'false', forces blockIdx.y = 0 and gridDim.y = 1.
 *		Otherwise, forces "single_Blk" to 'false'.
 *
 * single_Blk:	'True' if there is only a single block (i.e., gridDim.x == gridDim.y == 1  and  blockIdx.y == blockIdx.x == 0).
 *		It forces blockIdx.y = blockIdx.x = 0, and gridDim.x = gridDim.y = 1.
 *		It also forces "ext_grid" to 'false'.
 *
 * items_per_thread: Number of loads from global memory performed by each thread at a time.
 *
 * If both parameters "ext_grid" and "single_Blk" are set to 'true', they are ignored and overridden to 'false'.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		In a non-"extended" grid (i.e., with gridDim.y == 1):
 *			matrix_size <= (gridDim.x * items_per_thread * bs) must be < 2**24
 *		In any case, (gridDim.y * gridDim.x) must be < 2**24
 *
 *	- No code is compiled on the following condition(s), since it would never be executed:
 *		On Compute Capability 3.0 - 3.5:
 *			(sizeof(index_t) == sizeof(int)) && (ext_grid == true).
 *				Since (maxGridSizeX == INT_MAX), then (IDX_MAX / maxGridSizeX) <= 2
 *				Therefore, (gridDim.x * bs) > IDX_MAX, and there is no need to set gridDim.y > 1.
 *
 * Returns the sum of the corresponding column. That is, sum(d_A[...][tx]).
 */
template < bool ext_grid = false, bool single_Blk = false, index_t items_per_thread = REDUCE_TO_ROW__ITEMS_PER_THREAD >
__device__ static real reduce_gmem_to_row( real const *__restrict__ d_A, index_t matrix_size, index_t bs, index_t offset )
{

	// Resulting value.
	real sum_val = REAL_C( 0.0 );

	#if __CUDA_ARCH__

	// Skips instantiated code that actually is never executed.
	#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ <= 350)
	if ( (sizeof(index_t) != sizeof(int)) || (! ext_grid) )
	#endif
	{

		// Grid "extension"
		bool const grid_extension = ext_grid && ( ! single_Blk );

		// Single Block
		bool const single_block = single_Blk && ( ! ext_grid );

		// Number of items read from global memory by each thread.
		index_t const num_items = items_per_thread;

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
		index_t elemIdx[ num_items ];

		// "Active" block size.
		index_t const act_bs = IMUL( num_items, bs );	// num_items * (blockDim.y * blockDim.x)

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
		elemIdx[ 0 ] = IMUL( by, gdx ) + bx;
		if ( grid_extension )
			elemIdx[ 0 ] *= act_bs;				// Unconditional 32-bit product for all Compute Capabilities.
		else
			elemIdx[ 0 ] = IMUL( elemIdx[ 0 ], act_bs );	// 24-bit product on Compute Capability 1.x; 32-bit operation, otherwise.
		elemIdx[ 0 ] += offset;

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

			// TODO: Perform a parallel reduction
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
			for ( index_t i = 0 ; i < num_items-1 ; i++ ) {
				if ( elemIdx[ i ] < matrix_size )
					sum_val += d_A[ elemIdx[ i ] ];
			}

		} // if multi-blocks mode

	} // Skips instantiated code that actually is never executed.

	#endif /* if __CUDA_ARCH__ > 0 */

	return sum_val;

} // reduce_gmem_to_row

// -----------------------------------------------------------------------

/*
 * Helper method of kernel_reduce_to_row().
 *
 * Reduces the shared-memory block to a row. That is, returns SUM( shmem[..][tx] )
 *
 * Size of shared-memory block: bs = blockDim.x * blockDim.y
 *
 * block_width: Overrides blockDim.x
 *		On Compute Capability 1.x, it is denoted as "'half-warp-size' mode" when set to 16.
 *		In that case, and if block_height == 2, it makes use of volatile pointers to shared memory.
 *		Disabled/ignored if set to 0.
 *
 * block_height: A power of 2 that overrides blockDim.y.
 *		 Disabled/ignored if set to 0.
 *
 * WARNING:
 *	- No code is compiled on the following condition(s), since it would never be executed:
 *
 *		block_height == 1
 *
 *		On Compute Capabilities 1.0 - 3.5:
 *			block_height > 32
 *
 *		On Compute Capabilities 2.0 - 3.5:
 *			block_width  < 32
 *			(block_width * block_height) > 1024
 *
 *		On Compute Capability 1.x:
 *			(block_width * block_height) > 512
 *
 * Returns the sum of the corresponding column. That is, sums[ty...(bdy-1)][tx]
 */
template < index_t block_width = 0, index_t block_height = 0 >
__device__ static real reduce_shmem_to_row( index_t offset, real sum_val )
{

	#if __CUDA_ARCH__

	// Skips instantiated code that actually is never executed.
	if ( ( block_height != 1 )
		#if __CUDA_ARCH__ <= 350
			&& (block_height <= 32)
		#endif
		#if __CUDA_ARCH__ < 200
			&& ((block_width * block_height) <=  512)
		#else
			&& ((! block_width) || (block_width >= 32))
			#if __CUDA_ARCH__ <= 350
				&& ((block_width * block_height) <= 1024)
			#endif
		#endif
	   ) {

		// ------------------------------------------------

		// Uses the "'half-warp-size' mode" on Compute Capability 1.x if block_width == 16 (i.e., warpSize/2)
		#if __CUDA_ARCH__ < 200
			bool const half_warpSize = ( block_width == 16 );
		#else
			bool const half_warpSize = false;
		#endif

		// ------------------------------------------------

		// Thread index
		index_t const tx = threadIdx.x;
		index_t const ty = ( (block_height == 1) ? 0 : threadIdx.y );

		// Block dimensions
		index_t const bdx = ( block_width  ? block_width  : blockDim.x );	// == pitch
		index_t const bdy = ( block_height ? block_height : blockDim.y );	// A power of 2.

		// Block size.
		index_t const bs = IMUL( bdx, bdy );

		// Overrides offset from current block: threadIdx.y * blockDim.x + threadIdx.x
		if ( block_width + (block_height == 1) )
			offset = IMUL( ty, bdx ) + tx;

		// ------------------------------------------------

		// Shared memory for matrix reduction. Used in "multi-rows" mode.
		extern __shared__ real smem[];
		DECLARE_POINTER_SM_VAR( sums, , const, smem, 0, bs );

		/* On Compute Capability 1.x, no synchronization is required below the warp size.
		 * Instead, a "volatile" pointer to shared memory is used.
		 */
		DECLARE_SM_POINTER( vsums, volatile, const, sums, 0 );

		// ------------------------------------------------

		/* All threads store the result in a shared-memory block: sums[bdy][bdx],
		 * which is then reduced to a single row.
		 */

		if ( half_warpSize && ((! block_height) || (block_height == 2)) ) {

			// Uses the volatile pointer (no synchronization is necessary).
			STORE_IN_SM( vsums, offset, sum_val );	// vsums[ty][tx]

		} else {
			STORE_IN_SM(  sums, offset, sum_val );	//  sums[ty][tx]
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

		switch ( bdy ) {

			#if __CUDA_ARCH__ > 350

				// half >= 32. It is never executed on Compute Capability <= 3.5
				default: {
					// for ( half=(bdy/2) ; half > 16 ; half /= 2 )
					for ( index_t height = bdy, bound = (bs >> 1) ; height > 32 ; height >>= 1, bound >>= 1 ) {

						__syncthreads();

						// if (ty < half)
						if ( offset < bound ) {

							// sums[ty][tx] is already in sum_val
							// // sum_val = LOAD_FROM_SM( sums, offset );

							// sums[ty+half][tx]
							sum_val += LOAD_FROM_SM( sums, (offset + bound) );

							// New final value in sums[ty][tx]
							STORE_IN_SM( sums, offset, sum_val );
						}
					}
				}
			#endif

			// half == 16
			case 32: {
				index_t const bound = ( bdx << 4 );	// bdx * 16
				__syncthreads();
				if ( offset < bound ) {
					sum_val += LOAD_FROM_SM( sums, (offset + bound) );
					STORE_IN_SM( sums, offset, sum_val );
				}
			}

			// half == 8
			case 16: {
				index_t const bound = ( bdx << 3 );	// bdx * 8
				__syncthreads();
				if ( offset < bound ) {
					sum_val += LOAD_FROM_SM( sums, (offset + bound) );
					STORE_IN_SM( sums, offset, sum_val );
				}
			}

			// half == 4
			case 8: {
				index_t const bound = ( bdx << 2 );	// bdx * 4
				__syncthreads();
				if ( offset < bound ) {
					sum_val += LOAD_FROM_SM( sums, (offset + bound) );
					STORE_IN_SM( sums, offset, sum_val );
				}
			}

			// half == 2
			case 4: {
				index_t const bound = ( bdx << 1 );	// bdx * 2
				__syncthreads();
				if ( offset < bound ) {
					if ( half_warpSize )
						sum_val += LOAD_FROM_SM( vsums, (offset + bound) );
					else
						sum_val += LOAD_FROM_SM(  sums, (offset + bound) );
				}
			}

			// half == 1
			case 2: {
				index_t const bound = bdx;

				if ( ! half_warpSize )
					__syncthreads();

				if ( offset < bound ) {
					if ( half_warpSize )
						sum_val += LOAD_FROM_SM( vsums, (offset + bound) );	// vsums[1][tx]
					else
						sum_val += LOAD_FROM_SM(  sums, (offset + bound) );	//  sums[1][tx]
				}
			}

		} // switch

	} // Skips instantiated code that actually is never executed.

	#endif	/* __CUDA_ARCH__ */

	return sum_val;	// sums[ty][tx]

} // reduce_shmem_to_row

// -----------------------------------------------------------------------

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
 *
 * ext_grid:	'True' on a "extended" grid (i.e., gridDim.y > 1).
 *		If set to 'false', forces blockIdx.y = 0 and gridDim.y = 1.
 *		Otherwise, forces "single_Blk" to 'false'.
 *
 * single_Blk:	'True' if there is only a single block (i.e., gridDim.x == gridDim.y == 1  and  blockIdx.y == blockIdx.x == 0).
 *		It forces blockIdx.y = blockIdx.x = 0, and gridDim.x = gridDim.y = 1.
 *		It also forces "ext_grid" to 'false'.
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
 * Required size of shared memory:
 *	block_height != 1: block_width * block_height * sizeof(real) bytes.
 *	Else:
 *		multi-blocks mode:   sizeof(bool) bytes.
 *		Else (single block): 0 bytes.
 *
 * If both parameters "ext_grid" and "single_Blk" are set to 'true', they are ignored and overridden to 'false'.
 *
 * WARNING:
 *	- On Compute Capability < 1.2, the resulting rows in d_Tmp[] will NOT be reduced. Therefore, a second
 *	  call must be performed on d_Tmp[], with ext_grid = false, and single_Blk = true, to perform this action.
 *	  Such call should be something like this:
 *		index_t new_size = gridDim.y * gridDim.x * blockDim.x;
 *		kernel_reduce_to_row< false, true, ... ><<< 1, ... >>>( d_Tmp, new_size, NULL, d_C );
 *
 *	- On Compute Capability 1.x:
 *		In a non-"extended" grid (i.e., with gridDim.y == 1):
 *			matrix_size <= (gridDim.x * items_per_thread * bs) must be < 2**24
 *		In any case, (gridDim.y * gridDim.x) must be < 2**24
 *
 *	- No code is compiled on the following condition(s), since it would never be executed:
 *		On Compute Capability 3.0 - 3.5:
 *			(sizeof(index_t) == sizeof(int)) && (ext_grid == true).
 *				Since (maxGridSizeX == INT_MAX), then (IDX_MAX / maxGridSizeX) <= 2
 *				Therefore, (gridDim.x * bs) > IDX_MAX, and there is no need to set gridDim.y > 1.
 *
 *		On Compute Capabilities 1.0 - 3.5:
 *			block_height > 32
 *
 *		On Compute Capabilities 2.0 - 3.5:
 *			block_width  < 32
 *			(block_width * block_height) > 1024
 *
 *		On Compute Capability 1.x:
 *			(block_width * block_height) > 512
 */
template < bool ext_grid = false, bool single_Blk = false, index_t block_width = 0, index_t block_height = 0,
	index_t items_per_thread = REDUCE_TO_ROW__ITEMS_PER_THREAD >
__global__ static void kernel_reduce_to_row(real const *__restrict__ d_A,index_t matrix_size,real *__restrict__ d_Tmp,real *__restrict__ d_C)
{

	#if __CUDA_ARCH__

	// Skips instantiated code that actually is never executed.
	if (
		#if __CUDA_ARCH__ <= 350
			(block_height <= 32)
			#if __CUDA_ARCH__ >= 300	/* Compute Capability 3.0 - 3.5 */
			&& ( (sizeof(index_t) != sizeof(int)) || (! ext_grid) )
			#endif
		#endif
		#if __CUDA_ARCH__ < 200		/* Compute Capability 1.x */
			&& ((block_width * block_height) <=  512)
		#else
			&& ((! block_width) || (block_width >= 32))
			#if __CUDA_ARCH__ <= 350
				&& ((block_width * block_height) <= 1024)
			#endif
		#endif
	   )
	{

		// Grid "extension"
		bool const grid_extension = ext_grid && ( ! single_Blk );

		// Single Block
		bool const single_block = single_Blk && ( ! ext_grid );

		// Number of loads from global memory performed by each thread at a time.
		index_t const num_items = items_per_thread;

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
		index_t const bs = IMUL( bdy, bdx );

		// Offset from current block: threadIdx.y * blockDim.x + threadIdx.x
		index_t const offset = IMUL( ty, bdx ) + tx;

		// ------------------------------------------------

		// Reduces a portion of d_A[] to a row. That is, sum(d_A[...][tx]).
		real sum_val = reduce_gmem_to_row< grid_extension, single_block, num_items >( d_A, matrix_size, bs, offset );

		/* All threads store their results (i.e., "sum_val") in a shared-memory block,
		 * which is then reduced to a row.
		 *
		 * The returned value correspond to: sums[ty...(bdy-1)][tx]
		 */
		if ( bdy > 1 )
			sum_val = reduce_shmem_to_row< block_width, block_height >( offset, sum_val );

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
				index_t idx = IMUL( by, gdx ) + bx;
				if ( grid_extension )
					idx *= bdx;			// Unconditional 32-bit product for all Compute Capabilities.
				else
					idx = IMUL( idx, bdx );		// 24-bit product on Compute Capability 1.x. 32-bit op., otherwise.
				idx += tx;

				if ( (block_height == 1) || (offset < bdx) )	// if ( ty == 0 )
					d_Tmp[ idx ] = sum_val;			//	d_Tmp[by*gdx+bx][tx] = shmem[0][tx]
			}

			// ------------------------------------------------

			/* On Compute Capability >= 1.2:
			 *	The last block to finish, reduces the resulting rows in d_Tmp[ gdy*gdx ][ bdx ].
			 *	Please, see the threadfenceReduction example in CUDA SDK for details.
			 *
			 * On Compute Capability < 1.2, a second call must be performed on d_Tmp[], with grid_extension = false
			 * and single_block = true, to perform this action. Such call should be something like this:
			 *	index_t new_size = gridDim.y * gridDim.x * blockDim.x;
			 *	kernel_reduce_to_row< false, true, ... ><<< 1, ... >>>( d_Tmp, new_size, NULL, d_C );
			 *
			 * Actually, devices of Compute Capability 1.1 are able to execute this code, but not at a 100% of occupancy,
			 * due to the required number of registers.
			 */

			#if __CUDA_ARCH__ >= 120

				// Grid dimensions
				index_t const gdy = ( grid_extension ? gridDim.y : 1 );

				// This block is the last to finish the partial reduction of d_A[].
				extern __shared__ real smem[];
				bool *__restrict__ const p_isLastBlock = (bool *) &smem[0];

				// ------------------------------------------------

				// Thread ID is (0,0).
				bool const t0 = ! (tx + ty);

				// Total number of blocks
				index_t const num_blocks = IMUL( gdy, gdx );

				// Size of current partial sum in d_Tmp.
				index_t d_Tmp_size;
				if ( grid_extension )
					d_Tmp_size = num_blocks * bdx;			// > bdx
				else
					d_Tmp_size = IMUL( num_blocks, bdx );		// > bdx

				// ------------------------------------------------

				/* Waits until previous writes to global memory are finished.
				 *
				 * NOTE:
				 *	On Compute Capability 1.x, this functions takes about
				 *	20 instructions to complete.
				 */
				__threadfence();

				// Thread (0,0) from each block takes a ticket.
				if ( t0 ) {

					unsigned int const ticket = atomicInc( &retirement_count, num_blocks );

					// If the ticket ID is equal to the number of blocks, then this is the last block.
					*p_isLastBlock = ( ticket == ( num_blocks - 1 ) );
				}

				__syncthreads();

				// The last block sums the results of all other blocks.
				if ( *p_isLastBlock ) {

					// d_Tmp[ gdy*gdx ][ bdx ] is reduced to a row using a single block.

					// Reduces a portion of d_Tmp[] to a row. That is, sum(d_Tmp[...][tx]).
					sum_val = reduce_gmem_to_row< false, true, num_items >( d_Tmp, d_Tmp_size, bs, offset );

					/* All threads store their results (i.e., "sum_val") in a shared-memory block,
					 * which is then reduced to a row.
					 *
					 * The returned value correspond to: sums[ty...(bdy-1)][tx]
					 */
					if ( bdy > 1 )
						sum_val = reduce_shmem_to_row< block_width, block_height >( offset, sum_val );

					// The resulting value is stored in d_C[0][tx]
					if ( (bdy == 1) || (offset < bdx) ) {		// if ( ty == 0 )
						d_C[ tx ] = sum_val;			//	d_C[0][tx] = shmem[0][tx]

						// In addition, any of threads reset the counter (do not care which).
						retirement_count = 0;
					}

				} // if last_block

			#endif /* __CUDA_ARCH__ >= 110 */

		// "Single-block" mode
		} else {
			// The result is directly stored in d_C[0][tx].
			if ( (bdy == 1) || (offset < bdx) )		// if ( ty == 0 )
				d_C[ tx ] = sum_val;			//	d_C[0][tx] = shmem[0][tx]

		} // if ( gdx > 1 )

	} // Condition to skip instantiated code that actually is never executed.

	#endif /* if __CUDA_ARCH__ > 0 */

} // kernel_reduce_to_row

// -----------------------------------------------------------------------

/*
 * d_accum_A[j] = SUM( d_A[...][j] )
 *
 * matrix_size <= (grid_extension * grid_length * REDUCE_TO_ROW__ITEMS_PER_THREAD * block_height) * pitch
 * d_Tmp: Temporary storage. Ignored if grid_length == 1.
 * size_of( d_Tmp ) >= grid_extension * grid_length * pitch
 * length( d_accum_A ) >= pitch
 *
 * block_height <= (maxThreadsPerBlock / pitch), and must be a power of 2.
 * grid_length  <= maxGridSizeX
 * grid_extension <= MIN( grid_length, maxGridSizeY )
 * "pitch" must be a multiple of 'memory_alignment'.
 *
 * WARNING:
 *	- height > 1
 *	- On Compute Capability 1.x:
 *		In a non-"extended" grid (i.e., with grid_extension == 1):
 *			matrix_size <= (grid_length * REDUCE_TO_ROW__ITEMS_PER_THREAD * bs) must be < 2**24
 *		In any case, (grid_extension * grid_length) must be < 2**24
 */
__host__ void reduce_to_row( real const *__restrict__ d_A, index_t pitch, real *__restrict__ d_Tmp, index_t block_height, index_t grid_extension,
				index_t grid_length, index_t matrix_size, cudaStream_t stream_AccA, real *__restrict__ d_accum_A )
{

	dim3 const dimGrid( grid_length, grid_extension );

	// Number of loads from global memory performed by each thread at a time.
	index_t const num_items = REDUCE_TO_ROW__ITEMS_PER_THREAD;

	// --------------------------------

	// Launches the kernel according to block dimensions.

	if ( pitch > 32 ) {

		/* NOTE:
		 *	Since "pitch" is not known at compile time, sets the corresponding
		 *	template parameter to 0, so the kernel uses the "blockDim.x"
		 *	built-in variable.
		 */

		if ( grid_length > 1 )	// multi-blocks grid.

			switch ( block_height ) {

				// NOTE: More than 256 threads.
				case 8: {
					dim3 const dimBlock( pitch, 8 );
					size_t shmem_size = pitch * 8 * sizeof(real);	// Size of shared-memory block, in bytes.
					if ( grid_extension > 1 )
						kernel_reduce_to_row< true, false, 0, 8, num_items >
								<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
					else
						kernel_reduce_to_row< false, false, 0, 8, num_items >
								<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
				} break;

				case 4: {
					dim3 const dimBlock( pitch, 4 );
					size_t shmem_size = pitch * 4 * sizeof(real);	// Size of shared-memory block, in bytes.
					if ( grid_extension > 1 )
						kernel_reduce_to_row< true, false, 0, 4, num_items >
								<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
					else
						kernel_reduce_to_row< false, false, 0, 4, num_items >
								<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
				} break;

				case 2: {
					dim3 const dimBlock( pitch, 2 );
					size_t shmem_size = pitch * 2 * sizeof(real);	// Size of shared-memory block, in bytes.
					if ( grid_extension > 1 )
						kernel_reduce_to_row< true, false, 0, 2, num_items >
								<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
					else
						kernel_reduce_to_row< false, false, 0, 2, num_items >
								<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
				} break;

				/* NOTE:
				 *	block_height == 1 is referred as "'single-row' mode".
				 *	In multi-blocks mode (i.e., grid_length > 1), shared memory is allocated for
				 *	just a single boolean value.
				 *	Otherwise (i.e., in "single-block mode"), no shared memory is used,
				 */
				case 1: {
					if ( grid_extension > 1 )
						kernel_reduce_to_row< true, false, 0, 1, num_items >
								<<< dimGrid, pitch, sizeof(bool), stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
					else
						kernel_reduce_to_row< false, false, 0, 1, num_items >
								<<< dimGrid, pitch, sizeof(bool), stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
				} break;

				/* NOTE:
				 *	For other values, the template parameter is set to 0,
				 *	so the kernel uses the "blockDim.y" built-in variable.
				 */
				default: {
					dim3 const dimBlock( pitch, block_height );
					size_t shmem_size = pitch * block_height * sizeof(real); // Size of shared-memory block, in bytes.
					if ( grid_extension > 1 )
						kernel_reduce_to_row< true, false, 0, 0, num_items >
								<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
					else
						kernel_reduce_to_row< false, false, 0, 0, num_items >
								<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
				} break;

			} // switch( block_height )

		// Just a single block
		else {
			dim3 const dimBlock( pitch, block_height );
			size_t shmem_size = pitch * block_height * sizeof(real);	// Size of shared-memory block, in bytes.
			kernel_reduce_to_row< false, true, 0, 0, num_items >
					<<< 1, dimBlock, shmem_size, stream_AccA >>>
							( d_A, matrix_size, d_Tmp, d_accum_A );
		}

	} // pitch > 32

	// pitch == 32. On Compute Capability <= 3.5, it is equal to the warp size.
	else if ( pitch == 32 ) {

		if ( grid_length > 1 )	// multi-blocks grid.

			switch ( block_height ) {

				// NOTE: 512 threads.
				case 16: {
					dim3 const dimBlock( 32, 16 );
					size_t const shmem_size = 32 * 16 * sizeof(real);	// Size of shared-memory block, in bytes.
					if ( grid_extension > 1 )
						kernel_reduce_to_row< true, false, 32, 16, num_items >
								<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
					else
						kernel_reduce_to_row< false, false, 32, 16, num_items >
								<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
				} break;

				case 8: {
					dim3 const dimBlock( 32, 8 );
					size_t const shmem_size = 32 * 8 * sizeof(real);	// Size of shared-memory block, in bytes.
					if ( grid_extension > 1 )
						kernel_reduce_to_row< true, false, 32, 8, num_items >
								<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
					else
						kernel_reduce_to_row< false, false, 32, 8, num_items >
								<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
				} break;

				/* NOTE:
				 *	For other values, the template parameter is set to 0,
				 *	so the kernel uses the "blockDim.y" built-in variable.
				 */
				default: {
					dim3 const dimBlock( 32, block_height );
					size_t const shmem_size = 32 * block_height * sizeof(real); // Size of shared-memory block, in bytes.
					if ( grid_extension > 1 )
						kernel_reduce_to_row< true, false, 32, 0, num_items >
								<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, NULL, d_accum_A );
					else
						kernel_reduce_to_row< false, false, 32, 0, num_items >
								<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
				} break;

			} // switch( block_height )

		// Just a single block
		else {
			dim3 const dimBlock( 32, block_height );
			size_t const shmem_size = 32 * block_height * sizeof(real);	// Size of shared-memory block, in bytes.
			kernel_reduce_to_row< false, true, 32, 0, num_items >
					<<< 1, dimBlock, shmem_size, stream_AccA >>>
							( d_A, matrix_size, d_Tmp, d_accum_A );
		}

	} // pitch == 32

	/* pitch == 16.
	 * NOTE: Intended for Compute Capability 1.x ONLY, where it is referred as "'half-warp-size' mode".
	 */
	else {

		if ( grid_length > 1 )	// multi-blocks grid.

			switch ( block_height ) {

				case 16: {
					dim3 const dimBlock( 16, 16 );
					size_t const shmem_size = 16 * 16 * sizeof(real);	// Size of shared-memory block, in bytes.
					if ( grid_extension > 1 )
						kernel_reduce_to_row< true, false, 16, 16, num_items >
								<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
					else
						kernel_reduce_to_row< false, false, 16, 16, num_items >
								<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
				} break;

				/* NOTE:
				 *	For other values, the template parameter is set to 0,
				 *	so the kernel uses the "blockDim.y" built-in variable.
				 */
				default: {
					dim3 const dimBlock( 16, block_height );
					size_t const shmem_size = 16 * block_height * sizeof(real); // Size of shared-memory block, in bytes.
					if ( grid_extension > 1 )
						kernel_reduce_to_row< true, false, 16, 0, num_items >
								<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
					else
						kernel_reduce_to_row< false, false, 16, 0, num_items >
								<<< dimGrid, dimBlock, shmem_size, stream_AccA >>>
										( d_A, matrix_size, d_Tmp, d_accum_A );
				} break;

			} // switch( block_height )

		// Just a single block
		else {
			dim3 const dimBlock( 16, block_height );
			size_t const shmem_size = 16 * block_height * sizeof(real); // Size of shared-memory block, in bytes.
			kernel_reduce_to_row< false, true, 16, 0, num_items >
					<<< 1, dimBlock, shmem_size, stream_AccA >>>
							( d_A, matrix_size, NULL, d_accum_A );
		}

	} // pitch == 16

} // reduce_to_row

// =======================================================================

/*
 * d_A = d_B <op> d_A
 *
 * <op> is "./" or "-"
 *
 * matrix_size <= gridDim.y * gridDim.x * items_per_thread * blockDim.x
 * blockDim.y == 1  &  threadIdx.y == 0
 *
 * gridDim.y <= gridDim.x
 *
 * grid_extension: 'True' on a "extended" grid (i.e., gridDim.y > 1).
 *		   If set to 'false', forces blockIdx.y = 0 and gridDim.y = 1.
 *
 * items_per_thread: Number of loads from global memory performed by each thread at a time.
 *
 * div_operand: 'True' if operation to perform is a floating-point division.
 *		Otherwise, a subtraction is performed.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		In a non-"extended" grid (i.e., with gridDim.y == 1):
 *			matrix_size <= (gridDim.x * items_per_thread * blockDim.x) must be < 2**24
 *		In any case, (gridDim.y * gridDim.x) must be < 2**24
 *
 *	- No code is compiled on the following condition(s), since it would never be executed:
 *		On Compute Capability 3.0 - 3.5:
 *			(sizeof(index_t) == sizeof(int)) && (grid_extension == true).
 *				Since (maxGridSizeX == INT_MAX), then (IDX_MAX / maxGridSizeX) <= 2
 *				Therefore, (gridDim.x * bs) > IDX_MAX, and there is no need to set gridDim.y > 1.
 */
template < bool grid_extension = false, index_t items_per_thread = DIV_SUB__ITEMS_PER_THREAD, bool div_operand = false >
__global__ void kernel_div_sub( real *__restrict__ d_A, real const *__restrict__ d_B, index_t matrix_size )
{

	#if __CUDA_ARCH__

	// Skips instantiated code that actually is never executed.
	#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ <= 350)
	if ( (sizeof(index_t) != sizeof(int)) || (! grid_extension) )
	#endif
	{

		// Number of items read from global memory by each thread.
		index_t const num_items = items_per_thread;

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
		index_t elemIdx[ num_items ];

		// Block size
		index_t const bs = bdx;

		// Offset from current block: threadIdx.x
		index_t const offset = tx;

		// "Active" block size.
		index_t const act_bs = IMUL( num_items, bs );	// num_items * blockDim.x

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
		elemIdx[ 0 ] = IMUL( by, gdx ) + bx;
		if ( grid_extension )
			elemIdx[ 0 ] *= act_bs;				// Unconditional 32-bit product for all Compute Capabilities.
		else
			elemIdx[ 0 ] = IMUL( elemIdx[ 0 ], act_bs );	// 24-bit product on Compute Capability 1.x; 32-bit operation, otherwise.
		elemIdx[ 0 ] += offset;

		// Rest of elements.
		#pragma unroll
		for ( index_t i = 1 ; i < num_items ; i++ )
			elemIdx[ i ] = elemIdx[ i-1 ] + bs;

		// ------------------------------------------------

		// Each threads processes (up to) <num_items> elements from global memory.

		if ( elemIdx[ (num_items-1) ] < matrix_size ) {

			/* Compute Capability 1.0 - 1.1 */
			#if __CUDA_ARCH__ < 120

				if ( div_operand ) {
					#pragma unroll
					for ( index_t i = 0 ; i < num_items ; i++ )
						d_A[ elemIdx[ i ] ] = FDIV( d_B[ elemIdx[ i ] ], d_A[ elemIdx[ i ] ] );	// A = B / A
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

				if ( div_operand ) {
					#pragma unroll
					for ( index_t i = 0 ; i < num_items ; i++ )
						value_A[ i ] = FDIV( value_B[ i ], value_A[ i ] );	// A = B / A
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
			if ( div_operand ) {
				#pragma unroll
				for ( index_t i = 0 ; i < (num_items-1) ; i++ ) {
					if ( elemIdx[ i ] < matrix_size ) {
						real value_A = d_A[ elemIdx[ i ] ];
						real const value_B = d_B[ elemIdx[ i ] ];
						value_A = FDIV( value_B, value_A );	// A = B / A
						d_A[ elemIdx[ i ] ] = value_A;
					}
				}
			} else {
				#pragma unroll
				for ( index_t i = 0 ; i < (num_items-1) ; i++ ) {
					if ( elemIdx[ i ] < matrix_size ) {
						real value_A = d_A[ elemIdx[ i ] ];
						real const value_B = d_B[ elemIdx[ i ] ];
						value_A = ( value_B - value_A );	// A = B - A
						value_A = FDIV( value_B, value_A );	// A = B / A
						d_A[ elemIdx[ i ] ] = value_A;
					}
				}
			}
		}

	} // Skips instantiated code that actually is never executed.

	#endif /* if __CUDA_ARCH__ > 0 */

} // kernel_div

// -----------------------------------------------------------------------

/*
 * d_A = d_B <op> d_A
 *
 * <op> is "./" or "-"
 *
 * matrix_size <= (grid_extension * grid_length * DIV_SUB__ITEMS_PER_THREAD * block_size)
 * block_size <= maxThreadsPerBlock
 * grid_length  <= maxGridSizeX
 * grid_extension <= MIN( grid_length, maxGridSizeY )
 *
 * div_operand: 'True' if operation to perform is a floating-point division.
 *		Otherwise, a subtraction is performed.
 * WARNING:
 *	- On Compute Capability 1.x:
 *		In a non-"extended" grid (i.e., with grid_extension == gridDim.y == 1):
 *			matrix_size <= (gridDim.x * DIV_SUB__ITEMS_PER_THREAD * block_size) must be < 2**24
 *		In any case, (gridDim.y * gridDim.x) must be < 2**24
 */
__host__ void div_sub( real *__restrict__ d_A, real const *__restrict__ d_B, index_t matrix_size, index_t block_size,
			index_t grid_extension, index_t grid_length, bool div_operand, cudaStream_t stream_A )
{

	// Number of loads from global memory performed by each thread at a time.
	index_t const num_items = DIV_SUB__ITEMS_PER_THREAD;

	// --------------------------------

	dim3 const dimBlock( block_size, 1 );	// 1-D blocks

	// "Extended" grid.
	if ( grid_extension > 1 ) {

		dim3 const dimGrid( grid_length, grid_extension );

		bool const extended_grid = true;

		// Division or subtraction operand:
		if ( div_operand )
			kernel_div_sub< extended_grid, num_items, true  ><<< dimGrid, dimBlock, 0, stream_A >>>( d_A, d_B, matrix_size );
		else
			kernel_div_sub< extended_grid, num_items, false ><<< dimGrid, dimBlock, 0, stream_A >>>( d_A, d_B, matrix_size );

	// No "grid extension" required
	} else {

		dim3 const dimGrid( grid_length, 1 );

		bool const extended_grid = false;

		// Division or subtraction operand:
		if ( div_operand )
			kernel_div_sub< extended_grid, num_items, true  ><<< dimGrid, dimBlock, 0, stream_A >>>( d_A, d_B, matrix_size );
		else
			kernel_div_sub< extended_grid, num_items, false ><<< dimGrid, dimBlock, 0, stream_A >>>( d_A, d_B, matrix_size );

	} // grid extension

} // div_sub

// =======================================================================

/*
 * d_A[i][j] = d_A[i][j] .* d_Aux[i][j] ./ d_accum_b[j]
 *
 * blockDim.x == pitch
 * matrix_size <= (gridDim.y * gridDim.x * items_per_thread * blockDim.y) * blockDim.x
 * Size_of(d_accum_b) >= blockDim.x
 * gridDim.y <= gridDim.x
 *
 * grid_extension: 'True' on a "extended" grid (i.e., gridDim.y > 1).
 *		   If set to 'false', forces blockIdx.y = 0 and gridDim.y = 1.
 *
 * single_row:	'True' for single-row blocks (i.e., blockDim.y == 1 and threadIdx.y == 0).
 *		It forces threadIdx.y = 0 and blockDim.y = 1.
 *		In addition, no shared memory is used.
 *
 * items_per_thread: Number of loads from global memory performed by each thread at a time.
 *
 * Required size of shared memory:
 *	Multi-rows blocks:	  blockDim.x * sizeof(real) bytes.
 *	Else (single-row blocks): 0 bytes.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		In a non-"extended" grid (i.e., with gridDim.y == 1):
 *			matrix_size <= (gridDim.x * items_per_thread * blockDim.y * blockDim.x) must be < 2**24
 *		In any case, (gridDim.y * gridDim.x) must be < 2**24
 *
 *	- No code is compiled on the following condition(s), since it would never be executed:
 *		On Compute Capability 3.0 - 3.5:
 *			(sizeof(index_t) == sizeof(int)) && (grid_extension == true).
 *				Since (maxGridSizeX == INT_MAX), then (IDX_MAX / maxGridSizeX) <= 2
 *				Therefore, (gridDim.x * bs) > IDX_MAX, and there is no need to set gridDim.y > 1.
 */
template <bool grid_extension = false, bool single_row = false, index_t items_per_thread = MUL_DIV__ITEMS_PER_THREAD >
__global__ static void kernel_mul_div( real *__restrict__ d_A, real const *__restrict__ d_Aux, real const *__restrict__ d_accum_b,
					index_t matrix_size )
{

	#if __CUDA_ARCH__

	// Skips instantiated code that actually is never executed.
	#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ <= 350)
	if ( (sizeof(index_t) != sizeof(int)) || (! grid_extension) )
	#endif
	{

		// Number of items read from global memory by each thread.
		index_t const num_items = items_per_thread;

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
		DECLARE_POINTER_SM_VAR( acc, , const, smem, 0, bdx );

		if ( ! ( single_row || ty ) )
			STORE_IN_SM( acc, tx, d_accum_b[ tx ] );

		// ------------------------------------------------

		/* Each threads reads <num_items> elements from d_A[]
		 * with a distance equal to the block size ("bs").
		 */

		// Index to elements.
		index_t elemIdx[ num_items ];

		// Block size
		index_t const bs = IMUL( bdy, bdx );

		// Offset from current block: threadIdx.y * blockDim.x + threadIdx.x
		index_t const offset = IMUL( ty, bdx ) + tx;

		// "Active" block size.
		index_t const act_bs = IMUL( num_items, bs );	// num_items * (blockDim.y * blockDim.x)

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
		elemIdx[ 0 ] = IMUL( by, gdx ) + bx;
		if ( grid_extension )
			elemIdx[ 0 ] *= act_bs;				// Unconditional 32-bit product for all Compute Capabilities.
		else
			elemIdx[ 0 ] = IMUL( elemIdx[ 0 ], act_bs );	// 24-bit product on Compute Capability 1.x; 32-bit operation, otherwise.
		elemIdx[ 0 ] += offset;

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
			val_accum = LOAD_FROM_SM( acc, tx );
		}

		// ------------------------------------------------

		// Each threads processes (up to) <num_items> elements from global memory.

		if ( elemIdx[ (num_items-1) ] < matrix_size ) {

			/* Compute Capability 1.0 - 1.1 */
			#if __CUDA_ARCH__ < 120

				#pragma unroll
				for ( index_t i = 0 ; i < num_items ; i++ )
					d_A[ elemIdx[ i ] ] = FDIV( ( d_A[ elemIdx[ i ] ] * d_Aux[ elemIdx[ i ] ] ), val_accum );


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
					value_A[ i ] = FDIV( value_A[ i ], val_accum );

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
					value_A = FDIV( (value_A * value_Aux), val_accum );
					d_A[ elemIdx[ i ] ] = value_A;
				}
		}

	} // Skips instantiated code that actually is never executed.

	#endif /* if __CUDA_ARCH__ > 0 */

} // kernel_mul_div

// -----------------------------------------------------------------------

/*
 * d_A[i][j] = d_A[i][j] .* d_Aux[i][j] ./ d_accum_b[j]
 *
 * matrix_size <= (grid_extension * grid_length * MUL_DIV__ITEMS_PER_THREAD * block_height) * pitch
 * Size_of(d_accum_b) >= pitch
 * block_height <= (maxThreadsPerBlock / pitch)
 * grid_length  <= maxGridSizeX
 * grid_extension <= MIN( grid_length, maxGridSizeY )
 * "pitch" must be a multiple of 'memory_alignment'.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		In a non-"extended" grid (i.e., with grid_extension == gridDim.y == 1):
 *			matrix_size <= (gridDim.x * MUL_DIV__ITEMS_PER_THREAD * block_height * pitch) must be < 2**24
 *		In any case, (gridDim.y * gridDim.x) must be < 2**24
 */
__host__ void mul_div( real *__restrict__ d_A, real const *__restrict__ d_Aux, real const *__restrict__ d_accum_b, index_t pitch,
			index_t matrix_size, index_t block_height, index_t grid_extension, index_t grid_length, cudaStream_t stream_A )
{

	// Number of loads from global memory performed by each thread at a time.
	index_t const num_items = MUL_DIV__ITEMS_PER_THREAD;

	// --------------------------------

	// Multi-row blocks
	if ( block_height > 1 ) {

		bool const single_row = false;

		dim3 const dimBlock( pitch, block_height );

		size_t const shmem_size = pitch * sizeof(real);

		// "Extended" grid.
		if ( grid_extension > 1 ) {

			dim3 const dimGrid( grid_length, grid_extension );

			kernel_mul_div< true, single_row, num_items >
						<<< dimGrid, dimBlock, shmem_size, stream_A >>>( d_A, d_Aux, d_accum_b, matrix_size );

		// No "grid extension" required
		} else {

			dim3 const dimGrid( grid_length, 1 );

			kernel_mul_div< false, single_row, num_items >
						<<< dimGrid, dimBlock, shmem_size, stream_A >>>( d_A, d_Aux, d_accum_b, matrix_size );

		} // grid extension


	// Single-row blocks
	} else {

		bool const single_row = true;

		dim3 const dimBlock( pitch, 1 );

		size_t const shmem_size = 0;

		// "Extended" grid.
		if ( grid_extension > 1 ) {

			dim3 const dimGrid( grid_length, grid_extension );

			kernel_mul_div< true, single_row, num_items >
						<<< dimGrid, dimBlock, shmem_size, stream_A >>>( d_A, d_Aux, d_accum_b, matrix_size );

		// No "grid extension" required
		} else {

			dim3 const dimGrid( grid_length, 1 );

			kernel_mul_div< false, single_row, num_items >
						<<< dimGrid, dimBlock, shmem_size, stream_A >>>( d_A, d_Aux, d_accum_b, matrix_size );

		} // grid extension

	} // Multi- or single-row blocks

} // mul_div

// =======================================================================

/*
 * d_A = MAX( d_A, R_MIN )
 *
 * blockDim.x == pitch
 * matrix_size <= (gridDim.y * gridDim.x * items_per_thread * blockDim.y) * blockDim.x
 * Size_of(d_accum_b) >= blockDim.x
 * gridDim.y <= gridDim.x
 *
 * grid_extension: 'True' on a "extended" grid (i.e., gridDim.y > 1).
 *		   If set to 'false', forces blockIdx.y = 0 and gridDim.y = 1.
 *
 * items_per_thread: Number of loads from global memory performed by each thread at a time.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		In a non-"extended" grid (i.e., with gridDim.y == 1):
 *			matrix_size <= (gridDim.x * items_per_thread * blockDim.y * blockDim.x) must be < 2**24
 *		In any case, (gridDim.y * gridDim.x) must be < 2**24
 *
 *	- No code is compiled on the following condition(s), since it would never be executed:
 *		On Compute Capability 3.0 - 3.5:
 *			(sizeof(index_t) == sizeof(int)) && (grid_extension == true).
 *				Since (maxGridSizeX == INT_MAX), then (IDX_MAX / maxGridSizeX) <= 2
 *				Therefore, (gridDim.x * bs) > IDX_MAX, and there is no need to set gridDim.y > 1.
 */
template < bool grid_extension = false, index_t items_per_thread = ADJUST__ITEMS_PER_THREAD >
__global__ static void kernel_adjust( real *__restrict__ d_A, index_t matrix_size )
{

	#if __CUDA_ARCH__

	// Skips instantiated code that actually is never executed.
	#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ <= 350)
	if ( (sizeof(index_t) != sizeof(int)) || (! grid_extension) )
	#endif
	{

		// Number of items read from global memory by each thread.
		index_t const num_items = items_per_thread;

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
		index_t elemIdx[ num_items ];

		// Block size
		index_t const bs = IMUL( bdy, bdx );

		// Offset from current block: threadIdx.y * blockDim.x + threadIdx.x
		index_t const offset = IMUL( ty, bdx ) + tx;

		// "Active" block size.
		index_t const act_bs = IMUL( num_items, bs );	// num_items * (blockDim.y * blockDim.x)

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
		elemIdx[ 0 ] = IMUL( by, gdx ) + bx;
		if ( grid_extension )
			elemIdx[ 0 ] *= act_bs;				// Unconditional 32-bit product for all Compute Capabilities.
		else
			elemIdx[ 0 ] = IMUL( elemIdx[ 0 ], act_bs );	// 24-bit product on Compute Capability 1.x; 32-bit operation, otherwise.
		elemIdx[ 0 ] += offset;

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

					real const value = FMAX( d_A[ elemIdx[ i ] ], R_MIN );	// If d_A[] is 'NaN', returns 'R_MIN'

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
					value[ i ] = FMAX( value[ i ], R_MIN );		// If d_A[] is 'NaN', returns 'R_MIN'

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
					value = FMAX( value, R_MIN );		// If d_A[] is 'NaN', returns 'R_MIN'

					// if ( isnan( value ) || (value < R_MIN) )
					if ( value == R_MIN )
						d_A[ elemIdx[ i ] ] = value;
				}
			}
		}

	} // Skips instantiated code that actually is never executed.

	#endif /* if __CUDA_ARCH__ > 0 */

} // kernel_adjust

// -----------------------------------------------------------------------

/*
 * d_A = MAX( d_A , R_MIN )
 *
 * Adjusts d_A[ height ][ pitch ] to avoid underflow.
 *
 * matrix_size <= (grid_extension * grid_length * ADJUST__ITEMS_PER_THREAD * block_height) * pitch
 * block_height <= (maxThreadsPerBlock / pitch)
 * grid_length  <= maxGridSizeX
 * grid_extension <= MIN( grid_length, maxGridSizeY )
 * "pitch" must be a multiple of 'memory_alignment'.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		In a non-"extended" grid (i.e., with grid_extension == gridDim.y == 1):
 *			matrix_size <= (gridDim.x * ADJUST__ITEMS_PER_THREAD * block_height * pitch) must be < 2**24
 *		In any case, (gridDim.y * gridDim.x) must be < 2**24
 */
__host__ void adjust( real *__restrict__ d_A, index_t pitch, index_t matrix_size, index_t block_height, index_t grid_extension,
			index_t grid_length, cudaStream_t stream_A )
{

	// Number of loads from global memory performed by each thread at a time.
	index_t const num_items = ADJUST__ITEMS_PER_THREAD;

	// --------------------------------

	dim3 const dimBlock( pitch, block_height );

	// "Extended" grid.
	if ( grid_extension > 1 ) {

		dim3 const dimGrid( grid_length, grid_extension );

		kernel_adjust< true, num_items ><<< dimGrid, dimBlock, 0, stream_A >>>( d_A, matrix_size );

	// No "grid extension" required
	} else {

		dim3 const dimGrid( grid_length, 1 );

		kernel_adjust< false, num_items ><<< dimGrid, dimBlock, 0, stream_A >>>( d_A, matrix_size );

	} // grid extension

} // adjust

// =======================================================================

/*
 * Helper method of kernel_idx_max().
 *
 * Computes the maximum value in a portion of d_A and its column index.
 * That is, returns "max_val_idx", and "max_val" such that:
 *	max_val == d_A[i][max_val_idx] == max( d_A[i][...] ),
 * where
 *	0 <= max_val_idx <= width <= pitch
 *
 * blockDim.x <= width <= pitch
 * matrix_size <= (gridDim.y * gridDim.x * blockDim.y) * pitch
 * gridDim.y <= gridDim.x
 *
 * grid_extension: 'True' on a "extended" grid (i.e., gridDim.y > 1).
 *		   If set to 'false', forces blockIdx.y = 0 and gridDim.y = 1.
 *
 * block_width: Overrides blockDim.x.
 *		Disabled/ignored if set to IDX_MAX.
 *
 * items_per_thread: Number of loads from global memory performed by each thread at a time.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		In a non-"extended" grid (i.e., with gridDim.y == 1):
 *			matrix size <= (gridDim.x * blockDim.y * pitch) must be < 2**24
 *		In any case, (gridDim.y * gridDim.x) must be < 2**24
 *
 *	- No code is compiled on the following condition(s), since it would never be executed:
 *
 *		On Compute Capability 3.0 - 3.5:
 *			(sizeof(index_t) == sizeof(int)) && (ext_grid == true).
 *				Since (maxGridSizeX == INT_MAX), then (IDX_MAX / maxGridSizeX) <= 2
 *				Therefore, (gridDim.x * blockDim.y) > IDX_MAX, and there is no need to set gridDim.y > 1.
 *
 *		On Compute Capabilities 2.0 - 3.5:
 *			block_width == 16 (used on Compute Capability 1.x only).
 */
template < bool grid_extension = false, index_t block_width = IDX_MAX, index_t items_per_thread = IDX_MAX__ITEMS_PER_THREAD >
__device__ static void idx_max_gmem( real const *__restrict__ d_A, index_t width, index_t pitch, index_t matrix_size,
					real *__restrict__ max_val, index_t *__restrict__ max_val_idx )
{

	#if __CUDA_ARCH__

	#if (__CUDA_ARCH__ >= 200) && (__CUDA_ARCH__ <= 350)
	if ( ( block_width >= 32 )
		#if (__CUDA_ARCH__ >= 300)
			&& ( (sizeof(index_t) != sizeof(int)) || (! grid_extension) )
		#endif
	)
	#endif
	{
		// Number of items read from global memory by each thread.
		index_t const num_items = items_per_thread;

		// ------------------------------------------------

		// Thread index
		index_t const tx = threadIdx.x;
		index_t const ty = threadIdx.y;

		// Block index
		index_t const bx = blockIdx.x;
		index_t const by = ( grid_extension ? blockIdx.y : 0 );

		// Block dimensions
		index_t const bdx = ( (block_width < IDX_MAX) ? block_width : blockDim.x );
		index_t const bdy = blockDim.y;

		// Grid dimensions
		index_t const gdx = gridDim.x;

		// ------------------------------------------------

		/* Each threads processes <num_items> elements from d_A[]
		 * which have a distance equal to the block width ("bdx").
		 */

		// "Active" block size: blockDim.y * pitch
		index_t const act_bs = IMUL( bdy, pitch );

		// Offset from current block to current row: threadIdx.y * pitch
		index_t const offset = IMUL( ty, pitch );

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
		 * Offset to current row:   row_offset	 = block_offset + offset
		 */

		index_t const block_index = IMUL( by, gdx ) + bx;

		index_t block_offset;
		if ( grid_extension )
			block_offset = block_index * act_bs;	    // Unconditional 32-bit product for all Compute Capabilities.
		else
			block_offset = IMUL( block_index, act_bs ); // 24-bit product on Compute Capability 1.x; 32-bit operation, otherwise.

		index_t const row_offset = block_offset + offset;

		// Column index of elements.
		index_t colIdx[ num_items ];
		colIdx[ 0 ] = tx;

		#pragma unroll
		for ( index_t i = 1 ; i < num_items ; i++ )
			colIdx[ i ] = colIdx[ i-1 ] + bdx;

		// ------------------------------------------------

		// Each threads processes (up to) <num_items> elements from global memory.

		if ( row_offset < matrix_size ) {

			// Maximum value and its column index
			real l_max_val = REAL_C( 0.0 );
			index_t l_max_val_idx = 0;

			if ( colIdx[ (num_items-1) ] < width ) {

				// First value
				l_max_val = d_A[ row_offset + colIdx[ 0 ] ];
				l_max_val_idx = colIdx[ 0 ];

				// Rest of values
				real value[ (num_items-1) ];

				#pragma unroll
				for ( index_t i = 0 ; i < (num_items-1) ; i++ )
					value[ i ] = d_A[ row_offset + colIdx[ (i+1) ] ];

				// TODO: Parallel algorithm
				#pragma unroll
				for ( index_t i = 0 ; i < (num_items-1) ; i++ )
					if ( value[ i ] > l_max_val ) {
						l_max_val = value[ i ];
						l_max_val_idx = colIdx[ (i+1) ];
					}

			// Matrix height is not multiple of <num_items>
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

			// Returns selected values
			*max_val = l_max_val;
			*max_val_idx = l_max_val_idx;

		} // if ( row_offset < matrix_size )

	} // Skips instantiated code that actually is never executed.

	#endif /* if __CUDA_ARCH__ > 0 */

} // reduce_gmem_to_row

// -----------------------------------------------------------------------

/*
 * Helper method of kernel_idx_max().
 *
 * Computes the maximum value stored in a shared-memory block.
 * It also returns the corresponding column index also stored in the shared-memory block.
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
 *		Disabled/ignored if set to IDX_MAX.
 *
 * WARNING:
 *	- No code is compiled on the following condition(s), since it would never be executed:
 *
 *		On Compute Capabilities 2.0 - 3.5:
 *			block_width == 16 (used on Compute Capability 1.x only).
 *
 * TODO: Replace shared memory with warp-vote functions for Compute Capability >= 3.5.
 */
template < index_t block_width = IDX_MAX >
__device__ static void idx_max_shmem( real *__restrict__ max_val, index_t *__restrict__ max_val_idx )
{

	#if __CUDA_ARCH__

	// Skips instantiated code that actually is never executed.
	#if (__CUDA_ARCH__ >= 200) && (__CUDA_ARCH__ <= 350)
		if ( block_width >= 32 )
	#endif
	{
		// ------------------------------------------------

		/* Uses the "Warp-size mode" if block_width <= warpSize
		 * In such case, no synchronization is required to access to shared memory.
		 * Instead, uses volatile pointers.
		 */

		#if __CUDA_ARCH__ <= 350
			bool const warpSize_mode = ( block_width <= 32 );
		#else
			bool const warpSize_mode = ( block_width <= warpSize );
			// 'warpSize' might not be recognized as a constant when processing C++ templates,
			// so the comparison is NOT resolved at compile time.
		#endif

		// ------------------------------------------------

		// Thread index
		index_t const tx = threadIdx.x;
		index_t const ty = threadIdx.y;

		// Block dimensions
		index_t const bdx = ( block_width  ? block_width  : blockDim.x );
		index_t const bdy = blockDim.y;

		// ------------------------------------------------

		// Current Maximum values and its column index.
		real l_max_val = *max_val;
		index_t l_idx = *max_val_idx;

		// Block size: blockDim.y * blockDim.x
		index_t const bs = IMUL( bdy, bdx );

		// Offset from current block: threadIdx.y * blockDim.x + threadIdx.x
		index_t const offset = IMUL( ty, bdx ) + tx;

		// ------------------------------------------------

		/* Shared memory:
		 *	max_values: maximum values. Size: bs.
		 *	idx_max_values: column index of maximum values. Size: bs.
		 *
		 * NOTE:
		 *	idx_max_values[] is an array of INTEGER values.
		 *	When threads belong to the same warp, no synchronization is required,
		 *	and "volatile" pointers are used instead.
		 */
		extern __shared__ real smem[];
		index_t *__restrict__ const valIdx = (index_t *) smem;
		DECLARE_POINTER_SM_VAR( maxVal, , const, valIdx, bs, bs );

		/* No synchronization is required below the warp size.
		 * Instead, a "volatile" pointer to shared memory is used.
		 */
		volatile index_t *v_valIdx = valIdx;
		DECLARE_SM_POINTER( v_maxVal, volatile, const, maxVal, 0 );

		// ------------------------------------------------

		/* All threads initialize the shared memory block:
		 *	maxVal[ty][tx] and valIdx[ty][tx]
		 */

		if ( warpSize_mode ) {	// Uses volatile pointers
			STORE_IN_SM( v_maxVal, offset, l_max_val );
			v_valIdx[ offset ] = l_idx;

		} else {
			STORE_IN_SM( maxVal, offset, l_max_val );
			valIdx[ offset ] = l_idx;
		}

		// ------------------------------------------------

		/* Computes the maximum value of shared memory.
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
		 *		valIdx[ty][tx] = valIdx[ty][tx+half];
		 *	}
		 * }
		 *
		 * Please note that position [ty][tx] is referenced with: offset == ty*bdx + tx.
		 */

		// half = bdx/2 ... 2*warpSize
		for ( index_t half = (bdx >> 1) ; half > warpSize ; half >>= 1 ) {	// bdx is a power of 2.

			__syncthreads();

			// l_max_val = LOAD_FROM_SM( maxVal, offset );	// maxVal[ty][tx] is already in l_max_val
			// l_idx = valIdx[ offset ];			// valIdx[ty][tx] is already in l_idx

			if ( tx < half ) {

				real const val = LOAD_FROM_SM( maxVal, (offset + half) ); // maxVal[ty][tx+half]
				index_t const idx = valIdx[ offset + half ];		  // valIdx[ty][tx+half]

				// Stores new values in maxVal[ty][tx] and valIdx[ty][tx]
				if ( l_max_val < val ) {
					STORE_IN_SM( maxVal, offset, val );
					valIdx[ offset ] = idx;
					l_max_val = val;
					l_idx = idx;
				}
			}

		} // half = (bdx/2) ... (2*warpSize)


		if ( ! warpSize_mode )
			__syncthreads();


		// half = warpSize ... 64. It is never executed con Compute Capability <= 3.5.
		#if __CUDA_ARCH__ > 350

			#pragma unroll
			for ( index_t half = warpSize ; half > 32 ; half >>= 1 ) {	// bdx is a power of 2.

				/* No synchronization is required below the warp size.
				 * Instead, it uses "volatile" pointers to shared memory.
				 */

				// l_max_val = LOAD_FROM_SM( v_maxVal, offset ); // maxVal[ty][tx] is already in l_max_val
				// l_idx = v_valIdx[ offset ];			 // valIdx[ty][tx] is already in l_idx

				if ( tx < half ) {

					real const val = LOAD_FROM_SM( v_maxVal , (offset + half) );	// maxVal[ty][tx+half]
					index_t const idx = v_valIdx[ offset + half ];			// valIdx[ty][tx+half]

					// Stores new values in maxVal[ty][tx] and valIdx[ty][tx]
					if ( l_max_val < val ) {
						STORE_IN_SM( v_maxVal, offset, val );
						v_valIdx[ offset ] = idx;
						l_max_val = val;
						l_idx = idx;
					}
				}

			} // half = warpSize ... 64

		#endif /* __CUDA_ARCH__ > 350 */


		// half = 32 ... 2 (i.e., 5 times)
		#pragma unroll
		for ( index_t i = 0, bw = 64 ; i < 5 ; i++, bw >>= 1 ) {

			if ( block_width >= bw ) {

				int const half = ( bw >> 1 );

				/* No synchronization is required below the warp size.
				 * Instead, it uses "volatile" pointers to shared memory.
				 */

				// l_max_val = LOAD_FROM_SM( v_maxVal, offset ); // maxVal[ty][tx] is already in l_max_val
				// l_idx = v_valIdx[ offset ];			 // valIdx[ty][tx] is already in l_idx

				if ( tx < half ) {

					real const val = LOAD_FROM_SM( v_maxVal , (offset + half) );	// maxVal[ty][tx+half]
					index_t const idx = v_valIdx[ offset + half ];			// valIdx[ty][tx+half]

					// Stores new values in maxVal[ty][tx] and valIdx[ty][tx]
					if ( l_max_val < val ) {
						STORE_IN_SM( v_maxVal, offset, val );
						v_valIdx[ offset ] = idx;
						l_max_val = val;
						l_idx = idx;
					}
				}
			}
		} // for half = 32 ... 2

		// half == 1
		{

			int const half = 1;

			// l_max_val = LOAD_FROM_SM( v_maxVal, offset ); // maxVal[ty][tx] is already in l_max_val
			// l_idx = v_valIdx[ offset ];			 // valIdx[ty][tx] is already in l_idx

			if ( tx < half ) {

				real const val = LOAD_FROM_SM( v_maxVal , (offset + half) );	// maxVal[ty][tx+half]
				index_t const idx = v_valIdx[ offset + half ];			// valIdx[ty][tx+half]

				// Stores new values in maxVal[ty][tx] and valIdx[ty][tx]
				if ( l_max_val < val ) {
					// STORE_IN_SM( v_maxVal, offset, val );	// Not necessary
					// v_valIdx[ offset ] = idx;			// Not necessary
					l_max_val = val;
					l_idx = idx;
				}
			}
		}

		// Returns the selected values
		*max_val = l_max_val;
		*max_val_idx = l_idx;

	} // Skips instantiated code that actually is never executed.

	#endif /* __CUDA_ARCH__ */

} // idx_max_shmem

// -----------------------------------------------------------------------

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
 * grid_extension: 'True' on a "extended" grid (i.e., gridDim.y > 1).
 *		   If set to 'false', forces blockIdx.y = 0 and gridDim.y = 1.
 *
 * block_width: Overrides blockDim.x.
 *		Disabled/ignored if set to IDX_MAX.
 *
 * items_per_thread: Number of loads from global memory performed by each thread at a time.
 *
 * Size of shared-memory block: bs * ( sizeof(index_t) + sizeof(real) ),
 * where bs = (blockDim.x * blockDim.y).
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		In a non-"extended" grid (i.e., with gridDim.y == 1):
 *			matrix size <= (gridDim.x * blockDim.y * pitch) must be < 2**24
 *		In any case, (gridDim.y * gridDim.x) must be < 2**24
 *
 *	- No code is compiled on the following condition(s), since it would never be executed:
 *
 *		On Compute Capability 3.0 - 3.5:
 *			(sizeof(index_t) == sizeof(int)) && (ext_grid == true).
 *				Since (maxGridSizeX == INT_MAX), then (IDX_MAX / maxGridSizeX) <= 2
 *				Therefore, (gridDim.x * blockDim.y) > IDX_MAX, and there is no need to set gridDim.y > 1.
 *
 *		On Compute Capabilities 2.0 - 3.5:
 *			block_width == 16 (used on Compute Capability 1.x only).
 *
 */
template < bool grid_extension = false, index_t block_width = IDX_MAX, index_t items_per_thread = IDX_MAX__ITEMS_PER_THREAD >
__global__ static void kernel_idx_max( real const *__restrict__ d_A, index_t width, index_t pitch, index_t matrix_size,
					index_t *__restrict__ d_Idx )
{

	#if __CUDA_ARCH__

	// Skips instantiated code that actually is never executed.
	#if (__CUDA_ARCH__ >= 200) && (__CUDA_ARCH__ <= 350)
	if ( ( block_width >= 32 )
		#if (__CUDA_ARCH__ >= 300)
			&& ( (sizeof(index_t) != sizeof(int)) || (! grid_extension) )
		#endif
	)
	#endif
	{
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
		real max_val = REAL_C( 0.0 );
		index_t max_val_idx = 0;

		// Each threads computes the maximum value of a portion from d_A[].
		idx_max_gmem< grid_extension, block_width, items_per_thread >( d_A, width, pitch, matrix_size, &max_val, &max_val_idx );

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
		index_t rowIdx = IMUL( by, gdx ) + bx;

		if ( grid_extension )
			rowIdx *= bdy;			// Unconditional 32-bit product for all Compute Capabilities.
		else
			rowIdx = IMUL( rowIdx, bdy );	// 24-bit product on Compute Capability 1.x; 32-bit operation, otherwise.

		rowIdx += ty;

		if ( ! tx )
			d_Idx[ rowIdx ] = max_val_idx;

	} // Skips instantiated code that actually is never executed.

	#endif /* if __CUDA_ARCH__ > 0 */

} // kernel_idx_max

// -----------------------------------------------------------------------

/*
 * Computes the maximum value of each row in d_A[] and stores its column index in d_Idx[].
 * That is, returns d_Idx[i], such that:
 *	d_A[i][ d_Idx[i] ] == max( d_A[i][...] ).
 * where
 *	0 <= max_val_idx <= width <= pitch
 *
 * matrix_size <= (grid_extension * grid_length * block_height) * pitch
 * size_of( d_Idx ) >= grid_extension * grid_length * block_height
 *
 * block_height <= (maxThreadsPerBlock / pitch), and must be a power of 2.
 * grid_length  <= maxGridSizeX
 * grid_extension <= MIN( grid_length, maxGridSizeY )
 * "pitch" must be a multiple of 'memory_alignment'.
 *
 * WARNING:
 *	- On Compute Capability 1.x:
 *		In a non-"extended" grid (i.e., with grid_extension == 1):
 *			matrix_size <= (grid_length * block_height * pitch) must be < 2**24
 *		In any case, (grid_extension * grid_length) must be < 2**24
 */
__host__ void idx_max( real const *__restrict__ d_A, index_t width, index_t pitch, index_t matrix_size, index_t block_width,
			index_t block_height, index_t grid_extension, index_t grid_length, cudaStream_t stream_A, index_t *__restrict__ d_Idx )
{

	dim3 const dimGrid( grid_length, grid_extension );

	// Number of loads from global memory performed by each thread at a time.
	index_t const num_items = IDX_MAX__ITEMS_PER_THREAD;

	// --------------------------------

	// Launches the kernel according to the grid and block configuration.

	// "Extended" grid.
	if ( grid_extension > 1 ) {

		dim3 const dimGrid( grid_length, grid_extension );

		bool const extended_grid = true;

		switch ( block_width ) {

			case 16: { // On Compute Capability 1.x, this is the half of the warp size.
				index_t const bw = 16;
				dim3 const dimBlock( bw, block_height );
				kernel_idx_max< extended_grid, bw, num_items >
							<<< dimGrid, dimBlock, 0, stream_A >>>
								( d_A, width, pitch, matrix_size, d_Idx );
			} break;

			case 32: { // Warp size on Compute Capability <= 3.5
				index_t const bw = 32;
				dim3 const dimBlock( bw, block_height );
				kernel_idx_max< extended_grid, bw, num_items >
							<<< dimGrid, dimBlock, 0, stream_A >>>
								( d_A, width, pitch, matrix_size, d_Idx );
			} break;

			/* NOTE:
			 *	For other values, the template parameter is set to IDX_MAX,
			 *	so the kernel uses the "blockDim.x" built-in variable.
			 */
			default: {
				index_t const bw = IDX_MAX;
				dim3 const dimBlock( block_width, block_height );
				kernel_idx_max< extended_grid, bw, num_items >
							<<< dimGrid, dimBlock, 0, stream_A >>>
								( d_A, width, pitch, matrix_size, d_Idx );
			} break;

		} // switch

	// No "grid extension" required
	} else {

		dim3 const dimGrid( grid_length, 1 );

		bool const extended_grid = true;

		switch ( block_width ) {

			case 16: { // On Compute Capability 1.x, this is the half of the warp size.
				index_t const bw = 16;
				dim3 const dimBlock( bw, block_height );
				kernel_idx_max< extended_grid, bw, num_items >
							<<< dimGrid, dimBlock, 0, stream_A >>>
								( d_A, width, pitch, matrix_size, d_Idx );
			} break;

			case 32: { // Warp size on Compute Capability <= 3.5
				index_t const bw = 32;
				dim3 const dimBlock( bw, block_height );
				kernel_idx_max< extended_grid, bw, num_items >
							<<< dimGrid, dimBlock, 0, stream_A >>>
								( d_A, width, pitch, matrix_size, d_Idx );
			} break;

			/* NOTE:
			 *	For other values, the template parameter is set to IDX_MAX,
			 *	so the kernel uses the "blockDim.x" built-in variable.
			 */
			default: {
				index_t const bw = IDX_MAX;
				dim3 const dimBlock( block_width, block_height );
				kernel_idx_max< extended_grid, bw, num_items >
							<<< dimGrid, dimBlock, 0, stream_A >>>
								( d_A, width, pitch, matrix_size, d_Idx );
			} break;

		} // switch

	} // grid extension

} // idx_max

// =======================================================================
// =======================================================================
