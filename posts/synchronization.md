# Synchronization  

Launching kernel is asynchronous to host, so it means maybe you need to explicit synchronization (e.g., `cudaDeviceSynchronize`) or implicit (e.g., `cudaMemcpy`). You need to take care of the synchronization of `Memcpy` and `Memset` related functions, and for details, you can refer [API synchronization behavior](http://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html). Or you can refer below image:  
![image](https://raw.githubusercontent.com/NanXiao/cuda-little-book/master/images/cuda-apis-synchronizaiton.jpg)