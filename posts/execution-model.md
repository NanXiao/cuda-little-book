# Execution model

One GPU contains multiple Streaming Multiprocessors (SM), and one SM consists of multiple cores. One block can only be scheduled to run on one SM, and there can be multiple blocks running on one SM simultaneously.  
![image](https://raw.githubusercontent.com/NanXiao/cuda-little-book/master/images/execution-model.jpg) 