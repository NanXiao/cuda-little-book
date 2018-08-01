# nvcc

`nvcc` is the main wrapper for the NVIDIA CUDA Compiler suite and used to compile and link both host and GPU code. For `nvcc`'s compiler support, check [this](https://gist.github.com/ax3l/9489132), compute ability support, check [this](https://stackoverflow.com/questions/28932864/cuda-compute-capability-requirements).  

The process of compiling is divided into two steps: `nvcc`/`clang` generate virtual GPU architecture code, i.e., `PTX`; then `ptxas` (the `PTX` optimizing assembler) will assemble `PTX` into the GPU machine code.  