# CUDA stream
According to [How to Overlap Data Transfers in CUDA C/C++](https://devblogs.nvidia.com/how-overlap-data-transfers-cuda-cc/):  

> A stream in CUDA is a sequence of operations that execute on the device in the order in which they are issued by the host code. While operations within a stream are guaranteed to execute in the prescribed order, operations in different streams can be interleaved and, when possible, they can even run concurrently.  
> All device operations (kernels and data transfers) in CUDA run in a stream. When no stream is specified, the default stream (also called the “null stream”) is used. The default stream is different from other streams because it is a synchronizing stream with respect to operations on the device: no operation in the default stream will begin until all previously issued operations in any stream on the device have completed, and an operation in the default stream must complete before any other operation (in any stream on the device) will begin.

>Please note that CUDA 7, released in 2015, introduced a new option to use a separate default stream per host thread, and to treat per-thread default streams as regular streams (i.e. they don’t synchronize with operations in other streams).

To use non-default stream, we need to use `cudaMemcpyAsync` function to transfer data between host and device. Be aware that `cudaMemcpyAsync` need operate on pinned host memory.  

The following snippet is extracted from [GPU Pro Tip: CUDA 7 Streams Simplify Concurrency](https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency/):  

> The default stream is useful where concurrency is not crucial to performance. Before CUDA 7, each device has a single default stream used for all host threads, which causes implicit synchronization. As the section “Implicit Synchronization” in the CUDA C Programming Guide explains, two commands from different streams cannot run concurrently if the host thread issues any CUDA command to the default stream between them.

> CUDA 7 introduces a new option, the per-thread default stream, that has two effects. First, it gives each host thread its own default stream. This means that commands issued to the default stream by different host threads can run concurrently. Second, these default streams are regular streams. This means that commands in the default stream may run concurrently with commands in non-default streams.

> To enable per-thread default streams in CUDA 7 and later, you can either compile with the nvcc command-line option --default-stream per-thread, or #define the CUDA_API_PER_THREAD_DEFAULT_STREAM preprocessor macro before including CUDA headers (cuda.h or cuda_runtime.h). It is important to note: you cannot use #define CUDA_API_PER_THREAD_DEFAULT_STREAM to enable this behavior in a .cu file when the code is compiled by nvcc because nvcc implicitly includes cuda_runtime.h at the top of the translation unit.

In summary, before CUDA 7, all host threads share one default stream, so this will impact performance drastically. Since CUDA 7, every thread can have one unique default stream, so threads can issue commands concurrently in different default streams.

References:  
[GPU Pro Tip: CUDA 7 Streams Simplify Concurrency](https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency/)  
[How to Overlap Data Transfers in CUDA C/C++](https://devblogs.nvidia.com/how-overlap-data-transfers-cuda-cc/);  
[CUDA C/C++ Streams and Concurrency](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf).