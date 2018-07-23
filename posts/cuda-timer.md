# CUDA timer

The principle of Using CUDA Events to measure time is for every CUDA stream, the operations are executed in order. So when calling `cudaEventRecord` in one stream, a timestamp was noted. `cudaEventSynchronize` will wait the event was recorded, and `cudaEventElapsedTime` will return the time span between two events.  

The difference of creating event with `cudaEventBlockingSync` set or not is: using `cudaEventBlockingSync` will cause host thread to relinquish CPU when waiting for the event completes, while by default the CPU will do a busy-wait for event to complete.

References:  
[CUDA blocking flags](https://stackoverflow.com/questions/4822809/cuda-blocking-flags);  
[How to Implement Performance Metrics in CUDA C/C++](https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/);  
[cudaTimer](https://github.com/NanXiao/cudaTimer).