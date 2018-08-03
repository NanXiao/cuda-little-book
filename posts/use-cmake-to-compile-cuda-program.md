# Use CMake to compile CUDA program

Refer [Building Cross-Platform CUDA Applications with CMake](https://devblogs.nvidia.com/building-cuda-applications-cmake/) and [code-samples](https://github.com/robertmaynard/code-samples/tree/master/posts/cmake), now `CMakeLists.txt` can be as simple as this:  

	cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
	project(lscuda LANGUAGES CXX CUDA)
	
	add_executable(lscuda lscuda.cu)