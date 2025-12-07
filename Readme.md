# Compare Runtime

This is a simple project that compares the runtime of addition operations repeated many times between CPU and cuda (GPU).

The aim is to demonstrate the performance difference between CPU and GPU for a large number of simple operations.

## Methodology

The program performs addition operations on two arrays of floating-point numbers. The size of the arrays and the number of repetitions of the addition operation can be configured.

Here we used 1 million, 10 million, and 100 million elements in the arrays where we applied 1 million, 10 million, and 100 million addition operations respectively, then we measured the time taken to complete these operations on both CPU and GPU.

Check the requirements when you run and make sure that the platform is X64 from the configuration manager.

## Results 

The results are written in `cuda_results.csv` file. Generally the GPU shows that it can finish the same operation in 1% of the runtime compared to using CPU.

## Requirements
This project was developed and tested with the following environment:

- OS: Windows 11 Home
- GPU: NVIDIA GeForce RTX 4060 (Compute Capability 8.9)
- CUDA Toolkit: 13.0
- Microsoft Visual C++ 2022
- Nvidia Driver: 581.57 