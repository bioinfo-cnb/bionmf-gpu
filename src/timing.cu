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
 * timing.cu
 *	Routines for timing and profiling.
 *
 * NOTE: The following macro constants can be defined to modify the
 *	behavior of routines, as well as some constant and data-type definitions.
 *
 *	Additional information:
 *		NMFGPU_VERBOSE_2: Shows the parameters in some routine calls.
 *
 *	Timing (WARNING: They PREVENT asynchronous operations):
 *		NMFGPU_PROFILING_CONV: Compute timing of convergence test.
 *		NMFGPU_PROFILING_TRANSF: Compute timing of data transfers (should be used with NMFGPU_SYNC_TRANSF).
 *		NMFGPU_PROFILING_KERNELS: Compute timing of CUDA kernels.
 *
 **********************************************************/

#include <stdlib.h>
#include <stdio.h>
#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_CONV
	#include <math.h> /* isless */
#endif

#include "timing.cuh"

// --------------------------------------
// --------------------------------------

#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_CONV
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

#if NMFGPU_PROFILING_CONV
	// Time spent performing the test of convergence.
	timing_data_t classf_timing;
#endif

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

/*
 * Initializes kernel timers.
 */
void init_kernel_timers()
{

	#if NMFGPU_PROFILING_KERNELS

		for( index_t i=0 ; i<4 ; i++ ) {
			reduce_timing[i].time = 0.0;
			reduce_timing[i].counter = 0;
			reduce_timing[i].nitems = 0;
		}

		for( index_t i=0 ; i<2 ; i++ ) {
			div_timing[i].time = 0.0;
			div_timing[i].counter = 0;
			div_timing[i].nitems = 0;
		}

		for( index_t i=0 ; i<2 ; i++ ) {
			mul_div_timing[i].time = 0.0;
			mul_div_timing[i].counter = 0;
			mul_div_timing[i].nitems = 0;
		}

		for( index_t i=0 ; i<2 ; i++ ) {
			adjust_timing.time = 0.0;
			adjust_timing.counter = 0;
			adjust_timing[i].nitems = 0;
		}

		for( i=0 ; i<2 ; i++ ) {
			idx_max_timing[i].time = 0.0;
			idx_max_timing[i].counter = 0;
			idx_max_timing[i].nitems = 0;
		}

		for( index_t i=0 ; i<2 ; i++ ) {
			sub_timing[i].time = 0.0;
			sub_timing[i].counter = 0;
			sub_timing[i].nitems = 0;
		}

	#endif	/* if defined( NMFGPU_PROFILING_KERNELS ) */

} // init_kernel_timers

/////////////////////////////////////////////////////////////////////

/*
 * Initializes timers for data-transfers.
 */
void init_transfer_timers()
{

	#if NMFGPU_PROFILING_TRANSF

		upload_Vrow_timing.time = 0.0;
		upload_Vrow_timing.counter = 0;
		upload_Vrow_timing.nitems = 0;

		upload_Vcol_timing.time = 0.0;
		upload_Vcol_timing.counter = 0;
		upload_Vcol_timing.nitems = 0;

		upload_H_timing.time = 0.0;
		upload_H_timing.counter = 0;
		upload_H_timing.nitems = 0;

		upload_W_timing.time = 0.0;
		upload_W_timing.counter = 0;
		upload_W_timing.nitems = 0;

		download_H_timing.time = 0.0;
		download_H_timing.counter = 0;
		download_H_timing.nitems = 0;

		download_W_timing.time = 0.0;
		download_W_timing.counter = 0;
		download_W_timing.nitems = 0;

		download_classf_timing.time = 0.0;
		download_classf_timing.counter = 0;
		download_classf_timing.nitems = 0;

	#endif	/* if defined( NMFGPU_PROFILING_TRANSF ) */

} // init_transfer_timers

/////////////////////////////////////////////////////////////////////

/*
 * Initializes structure for timing of convergence test.
 */
void init_conv_timer()
{

	#if NMFGPU_PROFILING_CONV
		classf_timing.time = 0.0;
		classf_timing.counter = 0;
		classf_timing.nitems = 0;
	#endif

} // init_conv_timer

/////////////////////////////////////////////////////////////////////

/*
 * Initializes the array of CUDA Events for timing.
 *
 * Returns EXIT_SUCCESS OR EXIT_FAILURE.
 */
int init_timing_events( index_t device_id )
{

	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_CONV

		#if NMFGPU_VERBOSE_2
			printf("\n[GPU%" PRI_IDX "] Initializing array of CUDA Events for timing (number of events: %" PRI_IDX ")...\n",
				device_id, NUM_TIMING_EVENTS );
		#endif

		cudaError_t cuda_status = cudaSuccess;

		// ----------------------------

		// Start timer
		cuda_status = cudaEventCreateWithFlags( &timing_events[ START_EVENT ], cudaEventBlockingSync );
		if ( cuda_status != cudaSuccess ) {
			fflush(stdout);
			fprintf( stderr, "\n[GPU%" PRI_IDX "] Error creating CUDA event for timing (timer start): %s\n",
				device_id, cudaGetErrorString(cuda_status) );
			return EXIT_FAILURE;
		}

		#if NMFGPU_PROFILING_CONV
			// Start outer timer
			cuda_status = cudaEventCreateWithFlags( &timing_events[ START_OUTER_EVENT ], cudaEventBlockingSync);
			if ( cuda_status != cudaSuccess ) {
				fflush(stdout);
				fprintf( stderr, "\n[GPU%" PRI_IDX "] Error creating CUDA event for timing (outer timer start): %s\n",
					device_id, cudaGetErrorString(cuda_status) );
				cudaEventDestroy( timing_events[ START_EVENT ] );
				return EXIT_FAILURE;
			}
		#endif

		// Stop timer
		cuda_status = cudaEventCreateWithFlags( &timing_events[ STOP_EVENT ], cudaEventBlockingSync );
		if ( cuda_status != cudaSuccess ) {
			fflush(stdout);
			fprintf( stderr, "\n[GPU%" PRI_IDX "] Error creating CUDA event for timing (timer stop): %s\n",
				device_id, cudaGetErrorString(cuda_status) );
			#if NMFGPU_PROFILING_CONV
				cudaEventDestroy( timing_events[ START_OUTER_EVENT ] );
			#endif
			cudaEventDestroy( timing_events[ START_EVENT ] );
			return EXIT_FAILURE;
		}

		// ----------------------------

		#if NMFGPU_VERBOSE_2
			printf("\n[GPU%" PRI_IDX "] Initializing array of CUDA Events for timing (number of events: %" PRI_IDX ")... Done.\n",
				device_id, NUM_TIMING_EVENTS );
		#endif

	#endif /* if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_CONV */

	return EXIT_SUCCESS;

} // init_timing_events

/////////////////////////////////////////////////////////////////////

/*
 * Finalizes all CUDA Events for timing.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int destroy_timing_events( index_t device_id )
{

	int status = EXIT_SUCCESS;

	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_CONV

		#if NMFGPU_VERBOSE_2
			printf("\n[GPU%" PRI_IDX "] Finalizing CUDA Events for timing (%" PRI_IDX " objects)...\n",
				device_id, NUM_TIMING_EVENTS );
		#endif

		cudaError_t cuda_status = cudaSuccess;

		// ----------------------------

		// Stop timer
		cuda_status = cudaEventDestroy( timing_events[ STOP_EVENT ] );
		if ( cuda_status != cudaSuccess ) {
			fflush(stdout);
			fprintf( stderr, "\n[GPU%" PRI_IDX "] Error destroying CUDA event for timing (timer stop): %s\n",
				device_id, cudaGetErrorString(cuda_status) );
			status = EXIT_FAILURE;
		}

		#if NMFGPU_PROFILING_CONV
			// Start outer timer
			cuda_status = cudaEventDestroy( timing_events[ START_OUTER_EVENT ] );
			if ( cuda_status != cudaSuccess ) {
				fflush(stdout);
				fprintf( stderr, "\n[GPU%" PRI_IDX "] Error destroying CUDA event for timing (outer timer start): %s\n",
					device_id, cudaGetErrorString(cuda_status) );
				status = EXIT_FAILURE;
			}
		#endif

		// Start timer
		cuda_status = cudaEventDestroy( timing_events[ START_EVENT ] );
		if ( cuda_status != cudaSuccess ) {
			fflush(stdout);
			fprintf( stderr, "\n[GPU%" PRI_IDX "] Error destroying CUDA event for timing (timer start): %s\n",
				device_id, cudaGetErrorString(cuda_status) );
			status = EXIT_FAILURE;
		}

		// ----------------------------

		#if NMFGPU_VERBOSE_2
			printf("\n[GPU%" PRI_IDX "] Finalizing CUDA Events for timing (%" PRI_IDX " objects)... Done.\n",
				device_id, NUM_TIMING_EVENTS );
		#endif

	#endif /* if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_CONV */

	return status;

} // destroy_timing_events

/////////////////////////////////////////////////////////////////////

/*
 * Starts the CUDA timer for the given CUDA event.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int start_cuda_timer_ev( cudaEvent_t timing_event, index_t device_id )
{

	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_CONV

		cudaError_t cuda_status = cudaSuccess;

		cuda_status = cudaDeviceSynchronize();
		if ( cuda_status != cudaSuccess ) {
			fflush(stdout);
			fprintf( stderr, "\n[GPU%" PRI_IDX "] CUDA Error detected: %s\n", device_id, cudaGetErrorString(cuda_status) );
			return EXIT_FAILURE;
		}

		cuda_status = cudaEventRecord( timing_event, 0 );	// NULL stream. Waits for all operations.
		if ( cuda_status != cudaSuccess ) {
			fflush(stdout);
			fprintf( stderr, "\n[GPU%" PRI_IDX "] Error recording a CUDA event: %s\n", device_id, cudaGetErrorString(cuda_status) );
			return EXIT_FAILURE;
		}

	#endif	/* if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_CONV */

	return EXIT_SUCCESS;

} // start_cuda_timer_ev

/////////////////////////////////////////////////////////////////////

/*
 * Starts the CUDA timer using the timing_events[ START_EVENT ] CUDA event.
 *
 * It is equivalent to: start_cuda_timer_ev( timing_events[ START_EVENT ], device_id );
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int start_cuda_timer( index_t device_id )
{

	int status = EXIT_SUCCESS;

	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_CONV

		status = start_cuda_timer_ev( timing_events[ START_EVENT ], device_id );

	#endif

	return status;

} // start_cuda_timer

/////////////////////////////////////////////////////////////////////

/*
 * Stops the CUDA timer started using the given CUDA event.
 *
 * Returns the elapsed time (in ms) or a negative value on error.
 */
float stop_cuda_timer_ev( cudaEvent_t start_timing_event, index_t device_id )
{

	float elapsed_time = 0.0f;

	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_CONV

		cudaError_t cuda_status = cudaSuccess;

		cudaEvent_t stop_timing_event = timing_events[ STOP_EVENT ];

		// ----------------------

		cuda_status = cudaEventRecord( stop_timing_event, 0 );
		if ( cuda_status != cudaSuccess ) {
			fflush(stdout);
			fprintf( stderr, "\n[GPU%" PRI_IDX "] CUDA Error detected: %s\n", device_id, cudaGetErrorString(cuda_status) );
			return -1.0f;
		}

		cuda_status = cudaEventSynchronize( stop_timing_event );
		if ( cuda_status != cudaSuccess ) {
			fflush(stdout);
			fprintf( stderr, "\n[GPU%" PRI_IDX "] CUDA Error detected: %s\n", device_id, cudaGetErrorString(cuda_status) );
			return -1.0f;
		}

		cuda_status = cudaEventElapsedTime( &elapsed_time, start_timing_event, stop_timing_event );
		if ( cuda_status != cudaSuccess ) {
			fflush(stdout);
			fprintf( stderr, "\n[GPU%" PRI_IDX "] Error retrieving elapsed time: %s\n", device_id, cudaGetErrorString(cuda_status) );
			return -1.0f;
		}

	#endif	/* if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_CONV */

	return elapsed_time;

} // stop_cuda_timer_ev

/////////////////////////////////////////////////////////////////////

/*
 * Stops the CUDA timer started using the timing_events[ START_EVENT ] CUDA event.
 *
 * It is equivalent to: stop_cuda_timer_ev( timing_events[ START_EVENT ], device_id );
 *
 * Returns the elapsed time (in ms) or a negative value on error.
 */
float stop_cuda_timer( index_t device_id )
{

	float elapsed_time = 0.0f;

	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_CONV

		elapsed_time = stop_cuda_timer_ev( timing_events[ START_EVENT ], device_id );

	#endif

	return elapsed_time;

} // stop_cuda_timer

/////////////////////////////////////////////////////////////////////

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
int stop_cuda_timer_cnt_ev( cudaEvent_t start_timing_event, timing_data_t *__restrict__ td, index_t nitems, index_t counter, index_t device_id )
{

	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_CONV


		float elapsed_time = stop_cuda_timer_ev( start_timing_event, device_id );
		if ( isless( elapsed_time, 0.0f ) )
			return EXIT_FAILURE;

		if ( td ) {
			td->counter += (uintmax_t) counter;
			td->nitems  += (uintmax_t) nitems;
			td->time += (long double) elapsed_time;
		}

	#endif	/* if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_CONV */

	return EXIT_SUCCESS;

} // stop_cuda_timer_cnt_ev

/////////////////////////////////////////////////////////////////////

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
int stop_cuda_timer_cnt( timing_data_t *__restrict__ td, index_t nitems, index_t counter, index_t device_id )
{

	#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_CONV

		int status = stop_cuda_timer_cnt( timing_events[ START_EVENT ], td, nitems, counter, device_id );
		if ( status == EXIT_FAILURE )
			return EXIT_FAILURE;

	#endif	/* if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_CONV */

	return EXIT_SUCCESS;

} // stop_cuda_timer_cnt

/////////////////////////////////////////////////////////////////////

/*
 * Prints the following information for the given operation "<op>":
 *	- Total elapsed time, measured in milliseconds but shown in seconds.
 *	- Number of times the operation was performed and the average time, in milliseconds.
 *	- Throughput, in Gigabytes per second.
 *
 * size_of_data: Size, in bytes, of data processed.
 */
void print_elapsed_time( char const *__restrict__ const op, timing_data_t *__restrict__ td, size_t size_of_data )
{

	// if ( op != NULL ) && ( td != NULL ) && ( size_of_data > 0 )
	if ( ((size_t) op) * ((size_t) td) * size_of_data ) {

		/* Throughput(GB/sec):
		 *	( (td->nitems * size_of_data) bytes / (2**30 bytes/GB) )  /  ( td->time (ms) / (1000 ms/sec) )
		 *
		 * Note that (size_of_data * ( 1000 / 2**30 )) is calculated at compile time.
		 */
		printf( "%s: %.3Lg sec (%" PRIuMAX " time(s), avg: %.6Lg ms), %.3Lg GB/s\n", op, td->time / 1000, td->counter,
			td->time / td->counter, ( ( td->nitems * size_of_data * (1000.0/(1<<30)) ) / td->time ) );
	}

} // print_elapsed_time

/////////////////////////////////////////////////////////////////////

/*
 * Shows time elapsed on some kernels.
 */
void show_kernel_times()
{

	#if NMFGPU_PROFILING_KERNELS

		printf("\n\tDevice Kernels:\n");

		// --------------------

		// reduce (sum)
		{
			long double total_time = 0.0;
			index_t num_kernels = 0;
			if ( reduce_timing[0].counter ) {
				print_elapsed_time("\t\tGPU matrix_to_row", &reduce_timing[0], sizeof(real) );
				total_time = reduce_timing[0].time;
				num_kernels = 1;
			}
			if ( reduce_timing[1].counter ) {
				print_elapsed_time("\t\tGPU matrix_to_row (extended grid)", &reduce_timing[1], sizeof(real) );
				total_time += reduce_timing[1].time;
				num_kernels++;
			}
			if ( reduce_timing[2].counter ) {
				print_elapsed_time("\t\tGPU matrix_to_row (single block)", &reduce_timing[2], sizeof(real) );
				total_time += reduce_timing[2].time;
				num_kernels++;
			}
			if ( reduce_timing[3].counter ) {
				print_elapsed_time("\t\tGPU matrix_to_row (copy)", &reduce_timing[3], sizeof(real) );
				total_time += reduce_timing[3].time;
				num_kernels++;
			}
			if ( num_kernels > 1 )
				printf("\t\t\tTotal matrix_to_row time: %.6Lg ms\n", total_time );
		}

		// --------------------

		// div
		if ( div_timing[0].counter )
			print_elapsed_time("\t\tGPU div", &div_timing[0], sizeof(real) );

		if ( div_timing[1].counter )
			print_elapsed_time("\t\tGPU div (extended grid)", &div_timing[1], sizeof(real) );

		if ( div_timing[0].counter * div_timing[1].counter )
			printf("\t\t\tTotal div time: %.6Lg ms.\n", div_timing[0].time + div_timing[1].time );

		// ------------------

		// mul_div
		if ( mul_div_timing[0].counter )
			print_elapsed_time("\t\tGPU mul_div_time", &mul_div_timing[0], sizeof(real) );

		if ( mul_div_timing[1].counter )
			print_elapsed_time("\t\tGPU mul_div_time (extended grid)", &mul_div_timing[1], sizeof(real) );

		if ( mul_div_timing[0].counter * mul_div_timing[1].counter )
			printf("\t\t\tTotal mul_div time: %.6Lg ms.\n", mul_div_timing[0].time + mul_div_timing[1].time );


		// --------------------

		// Adjust
		if ( adjust_timing[0].counter )
			print_elapsed_time( "\t\tGPU adjust", &adjust_timing[0], sizeof(real) );

		if ( adjust_timing[1].counter )
			print_elapsed_time( "\t\tGPU adjust (extended grid)", &adjust_timing[1], sizeof(real) );

		if ( adjust_timing[0].counter * adjust_timing[1].counter )
			printf("\t\t\tTotal adjust time: %.6Lg ms.\n", adjust_timing[0].time + adjust_timing[1].time );

		// -------------------

		// Column index of maximum value.
		if ( idx_max_timing[0].counter )
			print_elapsed_time("\t\tGPU matrix_idx_max", &idx_max_timing[0], sizeof(index_t) );

		if ( idx_max_timing[1].counter )
			print_elapsed_time("\t\tGPU matrix_idx_max (extended grid)", &idx_max_timing[1], sizeof(index_t) );

		if ( idx_max_timing[0].counter * idx_max_timing[1].counter )
			printf("\t\t\tTotal matrix_idx_max time: %.6Lg ms.\n", idx_max_timing[0].time + idx_max_timing[1].time );

		// --------------------

		// sub
		if ( sub_timing[0].counter )
			print_elapsed_time("\t\tGPU sub", &sub_timing[0], sizeof(real) );

		if ( sub_timing[1].counter )
			print_elapsed_time("\t\tGPU sub (extended grid)", &sub_timing[1], sizeof(real) );

		if ( sub_timing[0].counter * sub_timing[1].counter )
			printf("\t\t\tTotal sub time: %.6Lg ms.\n", sub_timing[0].time + sub_timing[1].time );

	#endif	/* if defined( NMFGPU_PROFILING_KERNELS ) */

} // show_kernel_times

////////////////////////////////////////////////////////////////

/*
 * Shows time elapsed on data transfers.
 */
void show_transfer_times()
{

	#if NMFGPU_PROFILING_TRANSF

		printf("\n\tData Transfers:\n");

		print_elapsed_time( "\t\tSend V (rows)", &upload_Vrow_timing, sizeof(real) );

		if ( upload_Vcol_timing.counter )
			print_elapsed_time( "\t\tSend V (columns)", &upload_Vcol_timing, sizeof(real) );

		print_elapsed_time( "\t\tSend W", &upload_W_timing, sizeof(real) );

		print_elapsed_time( "\t\tSend H", &upload_H_timing, sizeof(real) );

		print_elapsed_time( "\t\tGet W", &download_W_timing, sizeof(real) );

		print_elapsed_time( "\t\tGet H", &download_H_timing, sizeof(real) );

		// Transfer of classification vector (test of convergence).
		print_elapsed_time( "\t\tGet Classification", &download_classf_timing, sizeof(index_t) );

		long double const total_data_transf = upload_Vrow_timing.time + upload_Vcol_timing.time +
							upload_W_timing.time + upload_H_timing.time +
							download_W_timing.time + download_H_timing.time +
							download_classf_timing.time;

		printf( "\tTotal data-transfers time: %.10Lg ms\n\n", total_data_transf );

	#endif /* defined( NMFGPU_PROFILING_TRANSF ) */

} // show_transfer_times

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
