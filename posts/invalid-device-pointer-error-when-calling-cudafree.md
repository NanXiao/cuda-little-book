# "invalid device pointer" error when calling cudaFree

Recently, I bumped into an error when calling `cudaFree`:  

	invalid device pointer

After some debugging, I find the reason is program calls [cudaDeviceReset](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gef69dd5c6d0206c2b8d099abac61f217) before `cudaFree`. Since `cudaDeviceReset` will release all resources, invoking `cudaFree` will again generate error.