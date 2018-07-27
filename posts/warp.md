# Warp

Refer [SIMT and Warp](https://cvw.cac.cornell.edu/gpu/simt_warp) and [Thread Divergence](https://cvw.cac.cornell.edu/gpu/thread_div):  

> Groups of threads with consecutive thread indexes are bundled into warps; one full warp is executed on a single CUDA core. At runtime, a thread block is divided into a number of warps for execution on the cores of an SM. The warp size is `32` for all kinds of devices.  

> Threads within a warp must follow the same execution trajectory. All threads must execute the same instruction at the same time. In other words, threads cannot diverge. So if thread contains statement like `if-then-else`, the CUDA platform will instruct the warp to execute the then part first, and then proceed to the else part. While executing the then part, all threads that evaluated to false (e.g. the else threads) are effectively deactivated. When execution proceeds to the else condition, the situation is reversed. As you can see, the then and else parts are not executed in parallel, but in serial. This serialization can result in a significant performance loss.
