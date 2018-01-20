# CUDA stream
According to [How to Overlap Data Transfers in CUDA C/C++](https://devblogs.nvidia.com/how-overlap-data-transfers-cuda-cc/):  

> A stream in CUDA is a sequence of operations that execute on the device in the order in which they are issued by the host code. While operations within a stream are guaranteed to execute in the prescribed order, operations in different streams can be interleaved and, when possible, they can even run concurrently.  
> All device operations (kernels and data transfers) in CUDA run in a stream. When no stream is specified, the default stream (also called the “null stream”) is used. The default stream is different from other streams because it is a synchronizing stream with respect to operations on the device: no operation in the default stream will begin until all previously issued operations in any stream on the device have completed, and an operation in the default stream must complete before any other operation (in any stream on the device) will begin.

>Please note that CUDA 7, released in 2015, introduced a new option to use a separate default stream per host thread, and to treat per-thread default streams as regular streams (i.e. they don’t synchronize with operations in other streams).

To use non-default stream, we need to use `cudaMemcpyAsync` function to transfer data between host and device. Be aware that `cudaMemcpyAsync` need operate on pinned host memory.

References:  
[How to Overlap Data Transfers in CUDA C/C++](https://devblogs.nvidia.com/how-overlap-data-transfers-cuda-cc/);  
[CUDA C/C++ Streams and Concurrency](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf).