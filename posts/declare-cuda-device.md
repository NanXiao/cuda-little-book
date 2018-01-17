# Declare CUDA device

In CUDA programming, if you don't set device explicitly, it will use default device `0`. Because CUDA runtime API is thread-safe, which means it maintains per-thread state about the current device, you must use `cudaSetDevice` to specify wanted device in every host thread. Otherwise the thread will use default device.  

My machine has `4` GPUs.  
(1) Check following code which doesn't call `cudaSetDevice`:  

	$ cat parallel.cu
	#include <stdio.h>
	#include <omp.h>
	
	int main(void)
	{
	        #pragma omp parallel for
	        for (int i = 0; i < 4; i++)
	        {
	                int device = 0;
	                cudaGetDevice(&device);
	                printf("Thread %d is running in device %d\n", omp_get_thread_num(), device);
	        }
	
	        return 0;
	}

Build and run it:  

	$ nvcc -Xcompiler -fopenmp parallel.cu -o parallel
	$ ./parallel
	Thread 0 is running in device 0
	Thread 2 is running in device 0
	Thread 3 is running in device 0
	Thread 1 is running in device 0

We can see all threads use device `0`.  

(2) Change code and let every thread sets its own working device:  

	#include <stdio.h>
	#include <omp.h>
	
	int main(void)
	{
	        #pragma omp parallel for
	        for (int i = 0; i < 4; i++)
	        {
	                int device = 0;
					cudaSetDevice(i);
	                cudaGetDevice(&device);
	                printf("Thread %d is running in device %d\n", omp_get_thread_num(), device);
	        }
	
	        return 0;
	}
Build and run it again:  

	$  nvcc -Xcompiler -fopenmp parallel.cu -o parallel
	$ ./parallel
	Thread 0 is running in device 0
	Thread 1 is running in device 1
	Thread 3 is running in device 3
	Thread 2 is running in device 2

Now every thread will use different device. All the operations of that thread will occur in the device until the thread calls `cudaSetDevice` to change device again.  

References:  
[CUDA Pro Tip: Always Set the Current Device to Avoid Multithreading Bugs](https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/).
