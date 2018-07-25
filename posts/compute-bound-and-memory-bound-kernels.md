# "compute-bound" & "memory-bound" kernels

"compute-bound" kernel spends most of its time in calculating, not accessing memory.  

"memory-bound" kernels is divided into two kinds:  
a) "bandwidth-bound", the transfer between device and global memory nearly reaches the limitation;  
b) "latency-bound", fetching from the memory is the bottleneck.  

Please refer following diagram:  
![image](https://raw.githubusercontent.com/NanXiao/cuda-little-book/master/images/compute-bound-memory-bound-kernels.jpg)   

References:  
[stackoverflow](https://stackoverflow.com/questions/23278304/cuda-memory-bound-vs-latency-bound-vs-bandwidth-bound-vs-compute-bound);  
[What the profiler is telling you: optimizing gpu kernels](http://on-demand.gputechconf.com/gtc/2017/presentation/s7444-Cristoph-Angerer-Optimizing-GPU.pdf).