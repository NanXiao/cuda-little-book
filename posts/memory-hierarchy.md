# Memory hierarchy

Each thread has private local memory. Each thread block has shared memory visible to all threads of the block and with the same lifetime as the block (the `3rd` parameter of launching kernel identifies the amount of shared memory, please refer [Kernel execution configuration](posts/kernel-execution-configuration.md)). All threads have access to the same global memory. There are also two additional read-only memory spaces accessible by all threads: the constant and texture memory spaces. The global, constant, and texture memory spaces are persistent across kernel launches by the same application, so the lifetime of these three memory spaces is same as the application.  

The following pictures describe the relations among these memories:  
![image](https://raw.githubusercontent.com/NanXiao/cuda-little-book/master/images/memory-hierarchy.png)   
![image](https://raw.githubusercontent.com/NanXiao/cuda-little-book/master/images/memory-architecture.jpg)   

Reference:  
[CUDA C PROGRAMMING GUIDE](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).