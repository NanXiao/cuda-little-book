#  Compute Capability
Refer [CUDA C programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html):   

> The compute capability of a device is represented by a version number, also sometimes called its "SM version". This version number identifies the features supported by the GPU hardware and is used by applications at runtime to determine which hardware features and/or instructions are available on the present GPU.  
> The compute capability comprises a major revision number X and a minor revision
number Y and is denoted by X.Y.  

> Devices with the same major revision number are of the same core architecture. The major revision number is 7 for devices based on the Volta architecture, 6 for devices based on the Pascal architecture, 5 for devices based on the Maxwell architecture, 3 for devices based on the Kepler architecture, 2 for devices based on the Fermi architecture, and 1 for devices based on the Tesla architecture.  
 
> The minor revision number corresponds to an incremental improvement to the core
architecture, possibly including new features.  

`Volta`, `Maxwell`, etc are [microarchitectures](https://en.wikipedia.org/wiki/Category:Nvidia_microarchitectures) names.

