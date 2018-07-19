# Grid-Stride Loops

In summary, grid-stride loops don't assume that the thread grid is large enough to cover the entire data array, this kernel loops over the data array one grid-size at a time. So this method has better scalability:  

	__global__
	void add(int n, float *x, float *y)
	{
	  int index = blockIdx.x * blockDim.x + threadIdx.x;
	  int stride = blockDim.x * gridDim.x;
	  for (int i = index; i < n; i += stride)
	    y[i] = x[i] + y[i];
	}

或  

	__global__
	void saxpy(int n, float a, float *x, float *y)
	{
	    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
	         i < n; 
	         i += blockDim.x * gridDim.x) 
	      {
	          y[i] = a * x[i] + y[i];
	      }
	}

References:  
[An Even Easier Introduction to CUDA](https://devblogs.nvidia.com/even-easier-introduction-cuda/);  
[CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops](https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/).