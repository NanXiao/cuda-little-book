# Warp

Refer [SIMT and Warp](https://cvw.cac.cornell.edu/gpu/simt_warp):  

> Groups of threads with consecutive thread indexes are bundled into warps; one full warp is executed on a single CUDA core. At runtime, a thread block is divided into a number of warps for execution on the cores of an SM. The warp size is `32` for all kinds of devices.
