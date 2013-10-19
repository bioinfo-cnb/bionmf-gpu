BioNMF - GPU
==========

**BioNMF-GPU** is an efficient implementation of the ***Non-negative Matrix Factorization*** (***NMF***) algorithm based on a programmable ***Graphics-Processing Unit*** (***GPU***). Since numerous linear-algebra operations are required to draw images on the screen, GPUs are devices specially designed to perform such tasks much faster than any conventional processor. Nevertheless, their architecture has evolved into a general-purpose programmable system able to execute **non-graphics-related applications**, just like a *co-processor*. GPUs represent a *cost-effective* alternative to conventional multi-processor clusters, since they are already present on almost any modern PC or laptop as a *graphics card*.

This implementation is based on the NVIDIA’s programming model: ***CUDA*** (***Compute Unified Device Architecture***). Large input matrices are blockwise transferred and processed. Systems with multiple GPUs can be then synchronized through ***MPI*** (***Message Passing Interface***).
