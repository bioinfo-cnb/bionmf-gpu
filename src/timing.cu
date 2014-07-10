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
 * timing.cu
 *	Routines for timing.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Additional information:
 *		NMFGPU_VERBOSE_2: Shows the parameters in some routine calls.
 *
 *	GPU timing (WARNING: They PREVENT asynchronous operations. The CPU thread is blocked on synchronization):
 *		NMFGPU_PROFILING_TRANSF: Compute timing of data transfers. Shows additional information.
 *		NMFGPU_PROFILING_KERNELS: Compute timing of CUDA kernels. Shows additional information.
 *
 **********************************************************/

// Required by <inttypes.h>
#ifndef __STDC_FORMAT_MACROS
	#define __STDC_FORMAT_MACROS (1)
#endif

#include "timing.cuh"
#include "common.h"
#include "real_type.h"

#include <math.h>	/* isfinite() */
#include <inttypes.h>	/* PRIuFAST32, uintptr_t */
#include <stdio.h>
#include <stdlib.h>

////////////////////////////////////////////////
////////////////////////////////////////////////

/* Global variables */

#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
	// CUDA Events for timing.
	cudaEvent_t timing_events[ NUM_TIMING_EVENTS ];
#endif

#if NMFGPU_PROFILING_KERNELS
	// Timing on kernels
	timing_data_t reduce_timing[4], div_timing[2], mul_div_timing[2], adjust_timing[2], idx_max_timing[2], sub_timing[2];
#endif

#if NMFGPU_PROFILING_TRANSF
	// Timing on data transfers
	timing_data_t upload_Vrow_timing, upload_Vcol_timing, upload_H_timing, upload_W_timing, download_H_timing, download_W_timing,
			download_classf_timing;
#endif

////////////////////////////////////////////////
////////////////////////////////////////////////

/*
 * Returns a timing_data_t structure with the given arguments.
 *
 * nitems: Accumulated number of items processed in all counted "events".
 * counter: Number of "events" counted.
 * time: Accumulated elapsed time.
 */
timing_data_t new_timing_data( uintmax_t nitems, uint_fast32_t counter, float time )
{

	timing_data_t td;
	td.nitems = nitems;
	td.counter = counter;
	td.time = time;

	return td;

} // new_timing_data

////////////////////////////////////////////////

/*
 * Initializes kernel timers.
 */
void init_kernel_timers( void )
{

	#if NMFGPU_PROFILING_KERNELS

		for( index_t i=0 ; i<4 ; i++ )
			reduce_timing[i] = new_empty_timing_data();

		for( index_t i=0 ; i<2 ; i++ )
			div_timing[i] = new_empty_timing_data();

		for( index_t i=0 ; i<2 ; i++ )
			mul_div_timing[i] = new_empty_timing_data();

		for( index_t i=0 ; i<2 ; i++ )
			adjust_timing[i] = new_empty_timing_data();

		for( index_t i=0 ; i<2 ; i++ )
			idx_max_timing[i] = new_empty_timing_data();

		for( index_t i=0 ; i<2 ; i++ )
			sub_timing[i] = new_empty_timing_data();

	#endif	/* if defined( NMFGPU_PROFILING_KERNELS ) */

} // init_kernel_timers

////////////////////////////////////////////////

/*
 * Initializes timers for data-transfers.
 */
void init_transfer_timers( void )
{

	#if NMFGPU_PROFILING_TRANSF

		upload_Vrow_timing = new_empty_timing_data();

		upload_Vcol_timing = new_empty_timing_data();

		upload_H_timing = new_empty_timing_data();

		upload_W_timing = new_empty_timing_data();

		download_H_timing = new_empty_timing_data();

		download_W_timing = new_empty_timing_data();

		download_classf_timing = new_empty_timing_data();

	#endif	/* if defined( NMFGPU_PROFILING_TRANSF ) */

} // init_transfer_timers

////////////////////////////////////////////////

/*
 * Initializes the array of CUDA Events for timing.
 *
 * Returns EXIT_SUCCESS OR EXIT_FAILURE.
 */
int init_timing_events( void )
{

	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS

		bool const shown_by_all = true; // Messages shown by all processes.

		#if NMFGPU_VERBOSE_2
			print_message( shown_by_all, "Initializing array of CUDA Events for timing (number of events: %" PRI_IDX ")...\n",
					NUM_TIMING_EVENTS );
		#endif

		cudaError_t cuda_status = cudaSuccess;

		// ----------------------------

		// Start timer
		cuda_status = cudaEventCreateWithFlags( &timing_events[ START_EVENT ], cudaEventBlockingSync );
		if ( cuda_status != cudaSuccess ) {
			print_error( shown_by_all, "Error creating CUDA event for timing (timer start): %s\n",
					cudaGetErrorString(cuda_status) );
			return EXIT_FAILURE;
		}

		// Stop timer
		cuda_status = cudaEventCreateWithFlags( &timing_events[ STOP_EVENT ], cudaEventBlockingSync );
		if ( cuda_status != cudaSuccess ) {
			print_error( shown_by_all, "Error creating CUDA event for timing (timer stop): %s\n",
					cudaGetErrorString(cuda_status) );
			cudaEventDestroy( timing_events[ START_EVENT ] );
			return EXIT_FAILURE;
		}

		// ----------------------------

		#if NMFGPU_VERBOSE_2
			print_message( shown_by_all, "Initializing array of CUDA Events for timing (number of events: %"
					PRI_IDX ")... Done.\n", NUM_TIMING_EVENTS );
		#endif

	#endif /* if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS */

	return EXIT_SUCCESS;

} // init_timing_events

////////////////////////////////////////////////

/*
 * Finalizes all CUDA Events for timing.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int destroy_timing_events( void )
{

	int status = EXIT_SUCCESS;

	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS

		bool const shown_by_all = true; // Messages shown by all processes.

		#if NMFGPU_VERBOSE_2
			print_message( shown_by_all, "Finalizing CUDA Events for timing (%" PRI_IDX " objects)...\n", NUM_TIMING_EVENTS );
		#endif

		cudaError_t cuda_status = cudaSuccess;

		// ----------------------------

		// Stop timer
		cuda_status = cudaEventDestroy( timing_events[ STOP_EVENT ] );
		if ( cuda_status != cudaSuccess ) {
			print_error( shown_by_all, "Error destroying CUDA event for timing (timer stop): %s\n",
					cudaGetErrorString(cuda_status) );
			status = EXIT_FAILURE;
		}

		// Start timer
		cuda_status = cudaEventDestroy( timing_events[ START_EVENT ] );
		if ( cuda_status != cudaSuccess ) {
			print_error( shown_by_all, "Error destroying CUDA event for timing (timer start): %s\n",
					cudaGetErrorString(cuda_status) );
			status = EXIT_FAILURE;
		}

		// ----------------------------

		#if NMFGPU_VERBOSE_2
			print_message( shown_by_all, "Finalizing CUDA Events for timing (%" PRI_IDX " objects)... Done.\n",
					NUM_TIMING_EVENTS );
		#endif

	#endif /* if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS */

	return status;

} // destroy_timing_events

////////////////////////////////////////////////

/*
 * Starts the CUDA timer for the given CUDA event.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int start_cuda_timer_ev( cudaEvent_t timing_event )
{

	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS

		bool const shown_by_all = true; // Messages shown by all processes.

		cudaError_t cuda_status = cudaSuccess;

		// ----------------------

		/* Waits for *ALL* operations.
		 * NOTE: The CPU thread will block or spin according to flags
		 *	 specified in init_GPU().
		 */
		cuda_status = cudaDeviceSynchronize();
		if ( cuda_status != cudaSuccess ) {
			print_error( shown_by_all, "CUDA Error detected: %s\n", cudaGetErrorString(cuda_status) );
			return EXIT_FAILURE;
		}

		// Registers the current "timestamp".
		cuda_status = cudaEventRecord( timing_event, 0 );
		if ( cuda_status != cudaSuccess ) {
			print_error( shown_by_all, "Error recording a CUDA event: %s\n", cudaGetErrorString(cuda_status) );
			return EXIT_FAILURE;
		}

	#endif	/* if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS */

	return EXIT_SUCCESS;

} // start_cuda_timer_ev

////////////////////////////////////////////////

/*
 * Starts the CUDA timer using the timing_events[ START_EVENT ] CUDA event.
 *
 * It is equivalent to: start_cuda_timer_ev( timing_events[ START_EVENT ] );
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int start_cuda_timer( void )
{

	int status = EXIT_SUCCESS;

	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS

		status = start_cuda_timer_ev( timing_events[ START_EVENT ] );

	#endif

	return status;

} // start_cuda_timer

////////////////////////////////////////////////

/*
 * Stops the CUDA timer started using the given CUDA event.
 *
 * Returns the elapsed time (in ms) or a negative value on error.
 */
float stop_cuda_timer_ev( cudaEvent_t start_timing_event )
{

	float elapsed_time = 0.0f;

	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS

		bool const shown_by_all = true; // Messages shown by all processes.

		cudaError_t cuda_status = cudaSuccess;

		cudaEvent_t stop_timing_event = timing_events[ STOP_EVENT ];

		// ----------------------

		// Records the current "timestamp" for ALL previous operations.
		cuda_status = cudaEventRecord( stop_timing_event, 0 );
		if ( cuda_status != cudaSuccess ) {
			print_error( shown_by_all, "CUDA Error detected: %s\n", cudaGetErrorString(cuda_status) );
			return -1.0f;
		}

		/* Waits for the registered operations (all).
		 * NOTE: The CPU thread will block or spin according to flags
		 *	 specified in init_timing_events().
		 */
		cuda_status = cudaEventSynchronize( stop_timing_event );
		if ( cuda_status != cudaSuccess ) {
			print_error( shown_by_all, "CUDA Error detected: %s\n", cudaGetErrorString(cuda_status) );
			return -1.0f;
		}

		cuda_status = cudaEventElapsedTime( &elapsed_time, start_timing_event, stop_timing_event );
		if ( cuda_status != cudaSuccess ) {
			print_error( shown_by_all, "Error retrieving elapsed time: %s\n", cudaGetErrorString(cuda_status) );
			return -1.0f;
		}

		if ( ! isfinite( elapsed_time ) ) {
			print_error( shown_by_all, "Invalid elapsed time: %g\n", elapsed_time );
			return -1.0f;
		}

	#endif	/* if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS */

	return elapsed_time;

} // stop_cuda_timer_ev

////////////////////////////////////////////////

/*
 * Stops the CUDA timer started using the timing_events[ START_EVENT ] CUDA event.
 *
 * It is equivalent to: stop_cuda_timer_ev( timing_events[ START_EVENT ] );
 *
 * Returns the elapsed time (in ms) or a negative value on error.
 */
float stop_cuda_timer( void )
{

	float elapsed_time = 0.0f;

	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS

		elapsed_time = stop_cuda_timer_ev( timing_events[ START_EVENT ] );

	#endif

	return elapsed_time;

} // stop_cuda_timer

////////////////////////////////////////////////

/*
 * Stops the CUDA timer started using the given CUDA event.
 *
 * Updates the given timing data with the elapsed time (in ms).
 *
 * nitems: Number of items processed upon completion.
 * counter: Number of times that <nitems> items were processed.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int stop_cuda_timer_cnt_ev( cudaEvent_t start_timing_event, timing_data_t *__restrict__ td, gpu_size_t nitems, index_t counter )
{

	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS

		float const elapsed_time = stop_cuda_timer_ev( start_timing_event );	// It is never a NaN.
		if ( elapsed_time < 0.0f )
			return EXIT_FAILURE;

		if ( td ) {
			td->nitems  += (uintmax_t) nitems;
			td->counter += (uint_fast32_t) counter;
			td->time += elapsed_time;
		}

	#endif	/* if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS */

	return EXIT_SUCCESS;

} // stop_cuda_timer_cnt_ev

////////////////////////////////////////////////

/*
 * Stops the CUDA timer started using the timing_events[ START_EVENT ] CUDA event.
 *
 * Updates the given timing data with the elapsed time (in ms).
 *
 * nitems: Number of items processed upon completion.
 * counter: Number of times that <nitems> items were processed.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int stop_cuda_timer_cnt( timing_data_t *__restrict__ td, gpu_size_t nitems, index_t counter )
{

	int status = EXIT_SUCCESS;

	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS

		status = stop_cuda_timer_cnt_ev( timing_events[ START_EVENT ], td, nitems, counter );

	#endif

	return status;

} // stop_cuda_timer_cnt

////////////////////////////////////////////////

#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS

/*
 * Prints the following information for the given operation "<op>":
 *	- Total elapsed time, measured in milliseconds (but shown in seconds if show_secs is 'true').
 *	- Number of times the operation was performed and the average time, in milliseconds.
 *	- Bandwidth, in Gigabytes per second.
 *
 * size_of_data: Size in bytes of processed data.
 */
static void print_elapsed_time( char const *__restrict__ const op, timing_data_t *__restrict__ td, size_t size_of_data, bool show_secs,
				bool shown_by_all )
{

	// if ( op != NULL ) && ( td != NULL ) && ( size_of_data > 0 )
	if ( (uintptr_t) op * (uintptr_t) td * (uintptr_t) size_of_data ) {

		/* Bandwidth (GB/sec):
		 *	( (td->nitems * size_of_data) bytes / (2**30 bytes/GB) )  /  ( td->time (ms) / (1000 ms/sec) )
		 *
		 * Note that (size_of_data * ( 1000 / 2**30 )) is calculated at compile time.
		 */

		print_message( shown_by_all, "%s: %g %s (%" PRIuFAST32 " time(s), avg: %g ms), %g GiB/s\n", op,
				( show_secs ? (td->time / 1000.0f) : td->time ), ( show_secs ? "sec(s)" : "ms" ), td->counter,
				(td->time / td->counter), ( ( td->nitems * size_of_data * (1000.0f/((1<<30) + 0.0f)) ) / td->time ) );
	}

} // print_elapsed_time

#endif	/* NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS */

// ---------------------------------------------

/*
 * Shows time elapsed on some kernels.
 */
void show_kernel_times( void )
{

	#if NMFGPU_PROFILING_KERNELS

		bool const shown_by_all = false; // Messages shown by process 0 only.

		bool const show_secs = false;	// Shows elapsed time in milliseconds, not in seconds.

		print_message( shown_by_all, "\tDevice Kernels:\n" );

		// --------------------

		// reduce (sum)
		{
			float total_time = 0.0f;
			index_t num_kernels = INDEX_C( 0 );
			if ( reduce_timing[0].counter ) {
				print_elapsed_time("\t\tGPU matrix_to_row", &reduce_timing[0], sizeof(real), show_secs, shown_by_all );
				total_time = reduce_timing[0].time;
				num_kernels = INDEX_C( 1 );
			}
			if ( reduce_timing[1].counter ) {
				print_elapsed_time("\t\tGPU matrix_to_row (extended grid)", &reduce_timing[1], sizeof(real),
							show_secs, shown_by_all );
				total_time += reduce_timing[1].time;
				num_kernels++;
			}
			if ( reduce_timing[2].counter ) {
				print_elapsed_time("\t\tGPU matrix_to_row (single block)", &reduce_timing[2], sizeof(real),
							show_secs, shown_by_all );
				total_time += reduce_timing[2].time;
				num_kernels++;
			}
			if ( reduce_timing[3].counter ) {
				print_elapsed_time("\t\tGPU matrix_to_row (copy)", &reduce_timing[3], sizeof(real), show_secs, shown_by_all );
				total_time += reduce_timing[3].time;
				num_kernels++;
			}
			if ( num_kernels > INDEX_C( 1 ) )
				print_message( shown_by_all, "\t\t\tTotal matrix_to_row time: %g ms\n", total_time );
		}

		// --------------------

		// div
		if ( div_timing[0].counter )
			print_elapsed_time("\t\tGPU div", &div_timing[0], sizeof(real), show_secs, shown_by_all );

		if ( div_timing[1].counter )
			print_elapsed_time("\t\tGPU div (extended grid)", &div_timing[1], sizeof(real), show_secs, shown_by_all );

		if ( div_timing[0].counter * div_timing[1].counter )
			print_message( shown_by_all, "\t\t\tTotal div time: %g ms.\n", div_timing[0].time + div_timing[1].time );

		// ------------------

		// mul_div
		if ( mul_div_timing[0].counter )
			print_elapsed_time("\t\tGPU mul_div_time", &mul_div_timing[0], sizeof(real), show_secs, shown_by_all );

		if ( mul_div_timing[1].counter )
			print_elapsed_time("\t\tGPU mul_div_time (extended grid)", &mul_div_timing[1], sizeof(real),
						show_secs, shown_by_all );

		if ( mul_div_timing[0].counter * mul_div_timing[1].counter )
			print_message( shown_by_all, "\t\t\tTotal mul_div time: %g ms.\n", mul_div_timing[0].time + mul_div_timing[1].time );


		// --------------------

		// Adjust
		if ( adjust_timing[0].counter )
			print_elapsed_time( "\t\tGPU adjust", &adjust_timing[0], sizeof(real), show_secs, shown_by_all );

		if ( adjust_timing[1].counter )
			print_elapsed_time( "\t\tGPU adjust (extended grid)", &adjust_timing[1], sizeof(real),
						show_secs, shown_by_all );

		if ( adjust_timing[0].counter * adjust_timing[1].counter )
			print_message( shown_by_all, "\t\t\tTotal adjust time: %g ms.\n", adjust_timing[0].time + adjust_timing[1].time );

		// -------------------

		// Column index of maximum value.
		if ( idx_max_timing[0].counter )
			print_elapsed_time("\t\tGPU matrix_idx_max", &idx_max_timing[0], sizeof(index_t), show_secs, shown_by_all );

		if ( idx_max_timing[1].counter )
			print_elapsed_time("\t\tGPU matrix_idx_max (extended grid)", &idx_max_timing[1], sizeof(index_t),
						show_secs, shown_by_all );

		if ( idx_max_timing[0].counter * idx_max_timing[1].counter )
			print_message( shown_by_all, "\t\t\tTotal matrix_idx_max time: %g ms.\n",
					idx_max_timing[0].time + idx_max_timing[1].time );

		// --------------------

		// sub
		if ( sub_timing[0].counter )
			print_elapsed_time("\t\tGPU sub", &sub_timing[0], sizeof(real), show_secs, shown_by_all );

		if ( sub_timing[1].counter )
			print_elapsed_time("\t\tGPU sub (extended grid)", &sub_timing[1], sizeof(real), show_secs, shown_by_all );

		if ( sub_timing[0].counter * sub_timing[1].counter )
			print_message( shown_by_all, "\t\t\tTotal sub time: %g ms.\n", sub_timing[0].time + sub_timing[1].time );

	#endif	/* if defined( NMFGPU_PROFILING_KERNELS ) */

} // show_kernel_times

////////////////////////////////////////////////////////////////

/*
 * Shows time elapsed on data transfers.
 */
void show_transfer_times( void )
{

	#if NMFGPU_PROFILING_TRANSF

		bool const shown_by_all = false; // Messages shown by process 0 only.

		bool const show_secs = true;	// Shows elapsed time in seconds.

		print_message( shown_by_all, "\tData Transfers:\n" );

		// --------------------

		print_elapsed_time( "\t\tSend V (rows)", &upload_Vrow_timing, sizeof(real), show_secs, shown_by_all );

		if ( upload_Vcol_timing.counter )
			print_elapsed_time( "\t\tSend V (columns)", &upload_Vcol_timing, sizeof(real), show_secs, shown_by_all );

		if ( upload_W_timing.counter )
			print_elapsed_time( "\t\tSend W", &upload_W_timing, sizeof(real), show_secs, shown_by_all );

		if ( upload_H_timing.counter )
			print_elapsed_time( "\t\tSend H", &upload_H_timing, sizeof(real), show_secs, shown_by_all );

		print_elapsed_time( "\t\tGet W", &download_W_timing, sizeof(real), show_secs, shown_by_all );

		print_elapsed_time( "\t\tGet H", &download_H_timing, sizeof(real), show_secs, shown_by_all );

		// Transfer of classification vector (test of convergence).
		print_elapsed_time( "\t\tGet Classification vector", &download_classf_timing, sizeof(index_t), show_secs, shown_by_all );

		float const total_data_transf = upload_Vrow_timing.time + upload_Vcol_timing.time +
							upload_W_timing.time + upload_H_timing.time +
							download_W_timing.time + download_H_timing.time +
							download_classf_timing.time;

		print_message( shown_by_all, "\tTotal data-transfers time: %g ms\n\n", total_data_transf );

	#endif /* defined( NMFGPU_PROFILING_TRANSF ) */

} // show_transfer_times

////////////////////////////////////////////////
////////////////////////////////////////////////
