# Profile CUDA programs

`nvprof` can be used to profile `CUDA` programs,  and it works in summary mode by default:  

	$ nvprof ./simpleP2P
	......
	==95710== Profiling application: ./simpleP2P
	==95710== Profiling result:
	            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
	 GPU activities:   96.63%  681.92ms       100  6.8192ms  6.6804ms  7.1572ms  [CUDA memcpy PtoP]
	                    1.87%  13.224ms         2  6.6122ms  6.6098ms  6.6146ms  SimpleKernel(float*, float*)
	                    0.77%  5.4603ms         1  5.4603ms  5.4603ms  5.4603ms  [CUDA memcpy HtoD]
	                    0.72%  5.1016ms         1  5.1016ms  5.1016ms  5.1016ms  [CUDA memcpy DtoH]
	      API calls:   60.20%  681.57ms         1  681.57ms  681.57ms  681.57ms  cudaEventSynchronize
	                   32.06%  362.98ms         2  181.49ms  140.17us  362.84ms  cudaDeviceEnablePeerAccess
	                    2.85%  32.304ms         1  32.304ms  32.304ms  32.304ms  cudaHostAlloc
	                    1.29%  14.564ms         1  14.564ms  14.564ms  14.564ms  cudaFreeHost
	                    1.17%  13.232ms         2  6.6162ms  6.6159ms  6.6165ms  cudaDeviceSynchronize
	                    1.09%  12.289ms       102  120.48us  11.209us  5.5161ms  cudaMemcpy
	......

You can save output to log file (`%p` identifies the process ID and `%h` stands for host machine name) :  

	$ nvprof --log-file ~/nvprof.%p.%h.txt ./simpleP2P


GPU-Trace mode provides a timeline of all activities taking place on the GPU in chronological order:  

	$ nvprof --print-gpu-trace ./simpleP2P
	......
	==95120== NVPROF is profiling process 95120, command: ./simpleP2P
	==95120== Profiling application: ./simpleP2P
	==95120== Profiling result:
	   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream          Src Dev   Src Ctx          Dst Dev   Dst Ctx  Name
	645.76ms  6.7108ms                    -               -         -         -         -  64.000MB  9.3133GB/s      Device      Device  Tesla P100-PCIE         1        11  Tesla P100-PCIE         1  Tesla P100-PCIE         2  [CUDA memcpy PtoP]
	652.49ms  6.6872ms                    -               -         -         -         -  64.000MB  9.3462GB/s      Device      Device  Tesla P100-PCIE         2        18  Tesla P100-PCIE         2  Tesla P100-PCIE         1  [CUDA memcpy PtoP]
	659.19ms  6.6855ms                    -               -         -         -         -  64.000MB  9.3485GB/s      Device      Device  Tesla P100-PCIE         1        11  Tesla P100-PCIE         1  Tesla P100-PCIE         2  [CUDA memcpy PtoP]
	665.89ms  6.6874ms                    -               -         -         -         -  64.000MB  9.3459GB/s      Device      Device  Tesla P100-PCIE         2        18  Tesla P100-PCIE         2  Tesla P100-PCIE         1  [CUDA memcpy PtoP]
	672.59ms  6.6851ms                    -               -         -         -         -  64.000MB  9.3492GB/s      Device      Device  Tesla P100-PCIE         1        11  Tesla P100-PCIE         1  Tesla P100-PCIE         2  [CUDA memcpy PtoP]
	......
	
API-trace mode shows the timeline of all CUDA runtime and driver API calls invoked
on the host in chronological order:  

	$ nvprof --print-api-trace ./simpleP2P
	......
	==96043== NVPROF is profiling process 96043, command: ./simpleP2P
	==96043== Profiling application: ./simpleP2P
	==96043== Profiling result:
	   Start  Duration  Name
	143.37ms  5.3750us  cuDeviceGetPCIBusId
	214.11ms  4.8330us  cuDeviceGetPCIBusId
	219.79ms  4.0830us  cuDeviceGetPCIBusId
	225.49ms  3.5830us  cuDeviceGetPCIBusId
	231.28ms  4.0010us  cuDeviceGetCount
	231.29ms  3.0830us  cuDeviceGetCount
	......
Usually, we only care about runtime API calls: `--profile-api-trace runtime`.