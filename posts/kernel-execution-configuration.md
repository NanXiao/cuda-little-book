# Kernel execution configuration  

This part is modified from [stackoverflow](https://stackoverflow.com/a/26774770/2106207) and [CUDA programming manual](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration):  

> The execution configuration (of a global function call) is specified by inserting an expression of the form <<<Dg,Db,Ns,S>>>, where:
	
> Dg (dim3) specifies the dimension and size of the grid.  
> Db (dim3) specifies the dimension and size of each block.    
> Ns (size_t) specifies the number of bytes in shared memory that is dynamically allocated per block for this call in addition to the statically allocated memory. This is an optional parameter which defaults to 0.  
>S (cudaStream_t) specifies the associated stream, is an optional parameter which defaults to 0.

So the first parameter defines how many blocks, and second specifies how many threads in each block.  

BTW, there is a [cheat sheet](https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf) for CUDA Thread Indexing.  

Following is extracted from [CUDA for Engineers](http://cudaforengineers.com/):  
> Note that choosing the specific execution configuration that will produce the best performance on a given system involves both art and science. For now, just
point out that choosing the number of threads in a block to be some multiple of 32 is reasonable since that matches up with the number of CUDA cores in an SM. There are also limits on the sizes supported for both blocks and grids. One particularly relevant limit is that a single block cannot contain more than 1,024 threads. Since grids may have total thread counts well over 1,024, you
should expect your kernel launches to include lots of blocks, and plan on doing some execution configuration experiments to see what works best for your app running on your hardware. For such larger problems, reasonable values to test for the number of threads per block include 128, 256, and 512. 
