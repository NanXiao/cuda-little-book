# Check CUDA runtime API result

In CUDA example program, check the result of CUDA runtime API is always a good practice:  

	#define CUDA_SAFE_CALL(call)  \                                             
	do {\                                                                  
	    cudaError_t err = call;\                                           
	    if (cudaSuccess != err) {\                                         
	        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.",\ 
	                 __FILE__, __LINE__, cudaGetErrorString(err) );\       
	        exit(EXIT_FAILURE);\                                          
	    }\                                                                
	} while (0)

This can let we know the CUDA wrong state as early as possible.  

References:  
[Is cudaSafeCall no longer needed?](https://devtalk.nvidia.com/default/topic/525246/is-cudasafecall-no-longer-needed-/). 