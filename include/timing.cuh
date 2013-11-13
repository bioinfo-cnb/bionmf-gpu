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
 * timing.cuh
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

#if ! NMFGPU_TIMING_CUH
#define NMFGPU_TIMING_CUH (1)

///////////////////////////////////////////////////////

#include <stdint.h>	/* [u]intmax_t */

#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_CONV
	#include <cuda_runtime_api.h>
#endif

#include "index_type.h"

// --------------------------------------
// --------------------------------------

/* Data types */


#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS || NMFGPU_PROFILING_CONV
	// Indexes for timing_events[], the array of CUDA Events for timing.
	enum timing_event_idx {
		START_EVENT,			// Start timer.
		#if NMFGPU_PROFILING_CONV
			START_OUTER_EVENT,	// Start outer timer.
		#endif
		STOP_EVENT,			// Stop timer.
		NUM_TIMING_EVENTS		// Number of timing events (ie. length of array timing_events[]).
	}
#endif

// Information for timing.
typedef struct timing_data {
	long double time;	// Accumulated elapsed time.
	uintmax_t counter;	// Number of "events" counted.
	uintmax_t nitems;	// Accumulated number of items processed in all counted "events".
} timing_data_t;

// --------------------------------------


#if defined(NMFGPU_PROFILING_TRANSF) || defined(NMFGPU_PROFILING_KERNELS) || defined(NMFGPU_PROFILING_CONV)
	// CUDA Event(s) for timing.
	extern cudaEvent_t *__restrict__ const timing_events;	// Length: NUM_TIMING_EVENTS
#endif

#ifdef NMFGPU_PROFILING_KERNELS
	extern timing_data_t	*__restrict__ const reduce_timing, *__restrict__ const div_timing, *__restrict__ const mul_div_timing,
				*__restrict__ const adjust_timing, *__restrict__ const idx_max_timing, *__restrict__ const sub_timing;
#endif

#if NMFGPU_PROFILING_TRANSF
	// Timing on data transfers
	extern timing_data_t upload_Vrow_timing, upload_Vcol_timing, upload_H_timing, upload_W_timing, download_H_timing, download_W_timing,
			download_classf_timing;
#endif

#if NMFGPU_PROFILING_CONV
	// Time spent performing the test of convergence.
	extern timing_data_t classf_timing;
#endif

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////

/*
 * Initializes kernel timers.
 */
void init_kernel_timers();

// ------------------------------------------

/*
 * Initializes timers for data-transfers.
 */
void init_transfer_timers();

// ------------------------------------------

/*
 * Initializes structure for timing of convergence test.
 */
void init_conv_timer();

// ------------------------------------------

/*
 * Initializes the array of CUDA Events for timing.
 *
 * Returns EXIT_SUCCESS OR EXIT_FAILURE.
 */
int init_timing_events( index_t device_id );

// ------------------------------------------

/*
 * Finalizes all CUDA Events for timing.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int destroy_timing_events( index_t device_id );

// ------------------------------------------

/*
* Starts the CUDA timer for the given CUDA event.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int start_cuda_timer_ev( cudaEvent_t timing_event, index_t device_id );

// ------------------------------------------

/*
 * Starts the CUDA timer using the timing_events[ START_EVENT ] CUDA event.
 *
 * It is equivalent to: start_cuda_timer_ev( timing_events[ START_EVENT ], device_id );
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int start_cuda_timer( index_t device_id );

// ------------------------------------------

/*
 * Stops the CUDA timer started using the given CUDA event.
 *
 * Returns the elapsed time (in ms) or a negative value on error.
 */
float stop_cuda_timer_ev( cudaEvent_t start_timing_event, index_t device_id );

// ------------------------------------------

/*
 * Stops the CUDA timer started using the timing_events[ START_EVENT ] CUDA event.
 *
 * It is equivalent to: stop_cuda_timer_ev( timing_events[ START_EVENT ], device_id );
 *
 * Returns the elapsed time (in ms) or a negative value on error.
 */
float stop_cuda_timer( index_t device_id );

// ------------------------------------------

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
int stop_cuda_timer_cnt_ev( cudaEvent_t start_timing_event, timing_data_t *__restrict__ td, index_t nitems, index_t counter, index_t device_id );

// ------------------------------------------

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
int stop_cuda_timer_cnt( timing_data_t *__restrict__ td, index_t nitems, index_t counter, index_t device_id );

// ------------------------------------------

/*
 * Prints the following information for the given operation "<op>":
 *	- Total elapsed time, measured in milliseconds but shown in seconds.
 *	- Number of times the operation was performed and the average time, in milliseconds.
 *	- Throughput, in Gigabytes per second.
 *
 * size_of_data: Size, in bytes, of data processed.
 */
void print_elapsed_time( char const *__restrict__ const op, timing_data_t *__restrict__ td, size_t size_of_data );

// ------------------------------------------

/*
 * Shows time elapsed on some kernels.
 */
void show_kernel_times();

// ------------------------------------------

/*
 * Shows time elapsed on data transfers.
 */
void show_transfer_times();

///////////////////////////////////////////////////////

#endif	/* NMFGPU_TIMING_CUH */
