# Pinned Host Memory

The following image is from [How to Optimize Data Transfers in CUDA C/C++](https://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc/):  
![image](https://raw.githubusercontent.com/NanXiao/cuda-little-book/master/images/pinned-host-memory.jpg)   

By default, host allocates pageable memory, and the content will be transited by pinned memory to device memory. Use `cudaMallocHost` or `cudaHostAlloc` can allocate pinned memory directly and improvement performance. But since allocating  pinned memory will cause physical memory available to the operating system and other programs reduced, so be careful of using it.  

References:  
[How to Optimize Data Transfers in CUDA C/C++](https://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc/).


