/****************************
 *
 * common.h
 *	Common headers
 *
 * MAYOR CHANGES FROM PREVIOUS VERSION:
 *	- d_Waux, d_Haux  ---->  d_Aux
 *	- d_accum_h, d_accum_w ----> d_accum
 *	- BLN, BLM ----> N, M (or NnP, MnP)
 *	- K_padding ----> Kp
 *	- BLMnP_padding -----> MnPp
 *	- Added cudaStreams and cudaEvents
 *	- Added argument "my_rank" to all functions.
 *	- Asynchronous transfers.
 *	- W and Htras (CPU matrices) have padding.
 *	- bNv, NnPv: All bN and NnP*Kp values for all processes. Used only if (N % nProcs != 0)
 *	- bMv, MnPv: All bM and MnP*Kp values for all processes. Used only if (M % nProcs != 0)
 *
 * bN == (my_rank*N)/nProcs	(i.e.,Starting row)
 * bM == (my_rank*M)/nProcs	(i.e.,Starting column)
 *
 * NnP == ((my_rank+1)*N/nProcs) - bN	(i.e.,Number of rows for this process).
 * MnP == ((my_rank+1)*M/nProcs) - bM	(i.e.,Number of columns for this process).
 *
 * MnPp == padding( MnP )
 * Kp == padding( Kp )
 *
 **********************/

#ifndef _COMMON_GPU_H_
#define _COMMON_GPU_H_

///////////////////////////////////////////////////////

/* Data types */


/* Block configuration (block from matrix V loaded into device memory, not block of CUDA Threads)
 * Matrix V is processed in blocks of BLN rows and BLM columns.
 * There is one for each dimension "BLD" (D == N,M)
 */
typedef struct {
	int BL[2];		// Number of rows/columns for this dimension (i.e.,BLD)
	int BLp[2];		// Padding for BLD (if any).
	int num_steps[2];	// Number of blocks of size BL[] to be processed.
} block_t;


// --------------------------------

/* Other macros */


	// Shows transfer times.
	#ifdef _PROFILING_2
		#define _PRINT_TRANSFER_IT(op,td,size_bytes,iter) printf("%s: %.10g ms (%u time(s), avg: %.10g ms, %g GB/s)\n", \
			(op), (td)->time, (td)->counter, ((td)->time)/((td)->counter), \
			((iter)*(size_bytes)*1000)/(((td)->time)*1073741824) );

		#define _PRINT_TRANSFER(op,td,size_bytes) printf("%s: %.10g ms (%g GB/s)\n", \
			(op), (td)->time, (size_bytes*1000)/(((td)->time)*1073741824) );
	#endif


	// Timing
	#if defined(_PROFILING_2) || defined(_PROFILING_3) || defined(_PROFILING_CONV)

		#if defined(_DEBUG_NMF) || defined(_DEBUG_NMF_STREAMS_2)

			#define RECORD_CUDA_EVENT(e,s) \
			{ \
				cudaError_t cudaError=cudaThreadSynchronize(); \
				if ((cudaError!=cudaSuccess)||((cudaError=cudaGetLastError())!=cudaSuccess)) { \
					fflush(NULL); int my_dev; cudaGetDevice( &my_dev ); \
					fprintf(stderr,"Device %i: Error in CUDA: %s.\n",my_dev,cudaGetErrorString(cudaError)); \
				} \
				cudaEventRecord((e),(s));\
			}

			#define RECORD_CUDA_EVENT_SYNC(e,s)\
			{ \
				RECORD_CUDA_EVENT((e),(s)); cudaError_t cudaError=cudaEventSynchronize((e)); \
				if ((cudaError!=cudaSuccess)||((cudaError=cudaGetLastError())!=cudaSuccess)) { \
					fflush(NULL); int my_dev; cudaGetDevice( &my_dev ); \
					fprintf(stderr,"Device %i: Error waiting for event: %s\n",my_dev,cudaGetErrorString(cudaError)); \
				}\
			}

		#else

			#define RECORD_CUDA_EVENT(e,s) { cudaEventRecord((e),(s)); }

			#define RECORD_CUDA_EVENT_SYNC(e,s) { RECORD_CUDA_EVENT((e),(s)); cudaEventSynchronize((e)); }

		#endif

		// -----------------

		// Start timer:

		#define START_CUDA_TIMER_I(ev,ev_idx,s) { cudaThreadSynchronize(); RECORD_CUDA_EVENT(ev[ev_idx],(s)) }

		#define START_CUDA_TIMER(ev,s) { START_CUDA_TIMER_I((ev),START_EVENT,(s)) }

		// -----------------

		// Stop timer:

		#define STOP_CUDA_TIMER_I(ev,ev_idx,s,acc) { RECORD_CUDA_EVENT_SYNC((ev)[STOP_EVENT],(s)) ; \
				{float et=0; cudaEventElapsedTime(&et,ev[ev_idx],ev[STOP_EVENT]); (acc) += (double) et;} }

		#define STOP_CUDA_TIMER(ev,s,acc) { STOP_CUDA_TIMER_I((ev),START_EVENT,s,acc) }



		#define STOP_CUDA_TIMER_CNT_I(ev,ev_idx,s,td) { STOP_CUDA_TIMER_I((ev),(ev_idx),(s),(td)->time) ; (td)->counter++; }

		#define STOP_CUDA_TIMER_CNT(ev,s,td) { STOP_CUDA_TIMER_CNT_I((ev),START_EVENT,(s),(td)) }

	#endif

///////////////////////////////////////////////////////

#endif /* _COMMON_GPU_H_ */
