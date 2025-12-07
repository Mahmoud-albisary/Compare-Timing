#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>

using namespace std;

__global__ void addVectors(const float* a, const float* b, float* c, int N) {
	// Kernel code here
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < N) {
		c[idx] = a[idx] + b[idx];
	}
}
int main() {
	vector<int> nums = { 1000000, 10000000, 100000000 };
	int deviceCount = 0;
	int threadsPerBlock = 256;
	for (int N : nums) {
		int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
		vector<float> x(N), y(N), z(N);
		int size = N * sizeof(float);

		for (int i = 0; i < N; i++) {
			x[i] = 2.0;
			y[i] = 3.0;
		}
		float* a, * b, * c;

		cudaMalloc(&a, size);
		cudaMalloc(&b, size);
		cudaMalloc(&c, size);

		cudaMemcpy(a, x.data(), size, cudaMemcpyHostToDevice);
		cudaMemcpy(b, y.data(), size, cudaMemcpyHostToDevice);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		addVectors << <blocksPerGrid, threadsPerBlock >> > (a, b, c, N);

		cudaDeviceSynchronize();


		cudaEventRecord(start);

		addVectors << <blocksPerGrid, threadsPerBlock >> > (a, b, c, N);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cout << "z[0]: " << z[0] << endl;
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		cout << "Time taken: " << milliseconds << " ms" << endl;

		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		cudaError_t err = cudaGetDeviceCount(&deviceCount);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
			return -1;
		}
		cudaMemcpy(z.data(), c, size, cudaMemcpyDeviceToHost);

		cudaFree(a);
		cudaFree(b);
		cudaFree(c);
	}
}