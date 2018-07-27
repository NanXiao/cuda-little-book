# Warp

 Groups of threads with consecutive thread indexes are bundled into warps. At runtime, a thread block is divided into a number of warps for execution on the cores of an SM. The warp size is `32` for all kinds of devices.  Alternating between normal threads on a CPU requires a context switch to the OS and many cycles to store the state of the CPU (registers, etc) to memory somewhere. With CUDA, all the registers, shared memory and local memory for the entire block of threads are reserved at the start of the block. "Switching" between warps on every instruction requires no expensive memory operations, but does need some hardware in the SM to manage the list of active and stalled warps. This is why defining "concurrency" in the context of a SM is not straightforward. All active warps are getting executed concurrently, but each warp has an independent instruction pointer.

Threads within a warp must follow the same execution trajectory. All threads must execute the same instruction at the same time. In other words, threads cannot diverge. So if thread contains statement like `if-then-else`, the CUDA platform will instruct the warp to execute the then part first, and then proceed to the else part. While executing the then part, all threads that evaluated to false (e.g. the else threads) are effectively deactivated. When execution proceeds to the else condition, the situation is reversed. As you can see, the then and else parts are not executed in parallel, but in serial. This serialization can result in a significant performance loss.  

References:  
[SIMT and Warp](https://cvw.cac.cornell.edu/gpu/simt_warp);  
[Thread Divergence](https://cvw.cac.cornell.edu/gpu/thread_div);  
[How do CUDA cores on a SM execute warps concurrently?](https://devtalk.nvidia.com/default/topic/486556/cuda-programming-and-performance/how-do-cuda-cores-on-a-sm-execute-warps-concurrently-/post/3491408/).
