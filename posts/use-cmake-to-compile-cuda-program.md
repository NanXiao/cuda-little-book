# Use CMake to compile CUDA program

Refer [Building Cross-Platform CUDA Applications with CMake](https://devblogs.nvidia.com/building-cuda-applications-cmake/) and [code-samples](https://github.com/robertmaynard/code-samples/tree/master/posts/cmake), now `CMakeLists.txt` can be as simple as this:  

	cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
	project(lscuda LANGUAGES CXX CUDA)
	
	add_executable(lscuda lscuda.cu)

If compiler is not in `$PATH`, there are two methods to set it:  

(1) Add following line in `CMakeLists.txt`:  

	SET(CMAKE_CUDA_COMPILER /usr/local/cuda-9.0/bin/nvcc)

(2) Set `CUDACXX` environmental variable:  

	$ CUDACXX=/usr/local/cuda-9.0/bin/nvcc cmake ..


Use [OpenMP](https://www.openmp.org/):  

	FIND_PACKAGE(OpenMP REQUIRED)
	IF (OPENMP_FOUND) 
		SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}") 
		SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}") 
	ENDIF()  

