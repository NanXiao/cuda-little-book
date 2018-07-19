# Kernel execution configuration  

This part is modified from [stackoverflow](https://stackoverflow.com/a/26774770/2106207) and [CUDA programming manual](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration):  

> The execution configuration (of a global function call) is specified by inserting an expression of the form <<<Dg,Db,Ns,S>>>, where:
	
> Dg (dim3) specifies the dimension and size of the grid.  
> Db (dim3) specifies the dimension and size of each block.    
> Ns (size_t) specifies the number of bytes in shared memory that is dynamically allocated per block for this call in addition to the statically allocated memory. This is an optional parameter which defaults to 0.  
>S (cudaStream_t) specifies the associated stream, is an optional parameter which defaults to 0.

So the first parameter defines how many blocks, and second specifies how many threads in each block.  

BTW, there is a cheat sheet for CUDA Thread Indexing.