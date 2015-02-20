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
 * timing.h
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

#if ! NMFGPU_TIMING_CUH
#define NMFGPU_TIMING_CUH (1)

#include "index_type.h"

#include <cuda_runtime_api.h>

#include <stdint.h>	/* uint_fast32_t, uintmax_t */
#include <stdbool.h>

////////////////////////////////////////////////
////////////////////////////////////////////////

/* Data types */

#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
	// Indexes for timing_events[], the array of CUDA Events for timing.
	enum timing_event_idx {
		START_EVENT,			// Start timer.
		STOP_EVENT,			// Stop timer.
		NUM_TIMING_EVENTS		// Number of timing events (i.e., length of array timing_events[]).
	};
#endif

// Information for timing.
typedef struct timing_data {
	uint_fast32_t counter;	// Number of "events" counted.
	float time;		// Accumulated elapsed time.
	uintmax_t nitems;	// Accumulated number of items processed in all counted "events".
} timing_data_t;

// --------------------------------------
// --------------------------------------

/* Global variables */

#if NMFGPU_PROFILING_TRANSF || NMFGPU_PROFILING_KERNELS
	// CUDA Event(s) for timing.
	extern cudaEvent_t timing_events[ NUM_TIMING_EVENTS ];	// Length: NUM_TIMING_EVENTS
#endif

#ifdef NMFGPU_PROFILING_KERNELS
	// Timing on GPU kernels
	extern timing_data_t reduce_timing[4];
	extern timing_data_t div_timing[2];
	extern timing_data_t mul_div_timing[2];
	extern timing_data_t adjust_timing[2];
	extern timing_data_t idx_max_timing[2];
	extern timing_data_t sub_timing[2];
#endif

#if NMFGPU_PROFILING_TRANSF
	// Timing on data transfers
	extern timing_data_t upload_Vrow_timing;
	extern timing_data_t upload_Vcol_timing;
	extern timing_data_t upload_H_timing;
	extern timing_data_t upload_W_timing;
	extern timing_data_t download_H_timing;
	extern timing_data_t download_W_timing;
	extern timing_data_t download_classf_timing;
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
timing_data_t new_timing_data( uintmax_t nitems, uint_fast32_t counter, float time );

////////////////////////////////////////////////

/*
 * Returns an empty timing_data_t structure.
 */
timing_data_t new_empty_timing_data( void );

////////////////////////////////////////////////

/*
 * Initializes kernel timers.
 */
void init_kernel_timers( void );

////////////////////////////////////////////////

/*
 * Initializes timers for data-transfers.
 */
void init_transfer_timers( void );

////////////////////////////////////////////////

/*
 * Initializes the array of CUDA Events for timing.
 *
 * Returns EXIT_SUCCESS OR EXIT_FAILURE.
 */
int init_timing_events( void );

////////////////////////////////////////////////

/*
 * Finalizes all CUDA Events for timing.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int destroy_timing_events( void );

////////////////////////////////////////////////

/*
 * Starts the CUDA timer for the given CUDA event.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int start_cuda_timer_ev( cudaEvent_t timing_event );

////////////////////////////////////////////////

/*
 * Starts the CUDA timer using the timing_events[ START_EVENT ] CUDA event.
 *
 * It is equivalent to: start_cuda_timer_ev( timing_events[ START_EVENT ] );
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int start_cuda_timer( void );

////////////////////////////////////////////////

/*
 * Stops the CUDA timer started using the given CUDA event.
 *
 * Returns the elapsed time (in ms) or a negative value on error.
 */
float stop_cuda_timer_ev( cudaEvent_t start_timing_event );

////////////////////////////////////////////////

/*
 * Stops the CUDA timer started using the timing_events[ START_EVENT ] CUDA event.
 *
 * It is equivalent to: stop_cuda_timer_ev( timing_events[ START_EVENT ] );
 *
 * Returns the elapsed time (in ms) or a negative value on error.
 */
float stop_cuda_timer( void );

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
int stop_cuda_timer_cnt_ev( cudaEvent_t start_timing_event, timing_data_t *restrict td, size_t nitems, index_t counter );

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
int stop_cuda_timer_cnt( timing_data_t *restrict td, size_t nitems, index_t counter );

////////////////////////////////////////////////

/*
 * Prints the following information for the given operation "<op>":
 *	- Total elapsed time, measured in milliseconds (but shown in seconds if show_secs is 'true').
 *	- Number of times the operation was performed and the average time, in milliseconds.
 *	- Bandwidth, in Gigabytes per second.
 *
 * data_size: Size in bytes of the processed data type.
 */
void print_elapsed_time( char const *restrict const op, timing_data_t *restrict td, size_t data_size, bool show_secs, bool all_processes );

////////////////////////////////////////////////

/*
 * Shows time elapsed on some kernels.
 */
void show_kernel_times( void );

////////////////////////////////////////////////

/*
 * Shows time elapsed on data transfers.
 */
void show_transfer_times( void );

////////////////////////////////////////////////
////////////////////////////////////////////////

#endif	/* NMFGPU_TIMING_CUH */
