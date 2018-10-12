# Built-in variables

`CUDA` provides `5` built-in variables: `gridDim` denotes the dimension of `grid`, and `blockDim` denotes the dimension of `block`; their types are `dim3`. `blockIdx` and `threadIdx` identify the block index within the grid and thread index within the block respectively, and their types are `uint3`. `warpSize` is `int` type, and identifies the warp size in threads, and it should be `32` for all compute capabilities.  

References:  
[CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).
