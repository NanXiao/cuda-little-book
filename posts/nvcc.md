# nvcc

`nvcc` is the main wrapper for the NVIDIA CUDA Compiler suite and used to compile and link both host and GPU code. For `nvcc`'s compiler support, check [this](https://gist.github.com/ax3l/9489132), compute ability support, check [this](https://stackoverflow.com/questions/28932864/cuda-compute-capability-requirements).  

The process of compiling is divided into two steps: `nvcc`/`clang` generate virtual GPU architecture code, i.e., `PTX`; then `ptxas` (the `PTX` optimizing assembler) will compile `PTX` into the `SASS`, the actual GPU machine code. `-arch compute_xx` specifies the what type of `PTX` code will be generated while `-code sm_xx` specifies the what type of `SASS` code will be generated. Since GPU driver supports `JIT-Compile`, it can compile `PTX`into `SASS` during running.  

References:  
[stackoverflow](https://stackoverflow.com/questions/17599189/what-is-the-purpose-of-using-multiple-arch-flags-in-nvidias-nvcc-compiler);  
[NVIDIA CUDA Compiler Driver NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/).