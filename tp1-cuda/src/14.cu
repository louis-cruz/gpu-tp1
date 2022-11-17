
#include <omp.h>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;


__global__ void deviceCudaReduction(float *threadSums, float *reducedSum) {

    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = threadSums[i];

    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0) 
      reducedSum[blockIdx.x] = sdata[0];
}

__global__ void deviceCudaThreadSum(float *threadSums, double *step){

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        double x = (index-0.5)* *step;
        // sum of this thread
        float threadSum = 4.0/(1.0+x*x);
        
        threadSums[threadIdx.x] = threadSum;

        __syncthreads();
}

double calculatePiReduction(int num_steps, double step, int threads){

  int blocks = num_steps / threads;

	double *dev_step = &step;

  float threadSums[blocks] = {0};
  float blockSum[threads] = {0};
  cudaMalloc((void **) blockSum, sizeof(float)*threads);
  cudaMalloc((void **) threadSums, sizeof(float)*blocks);
  cudaMalloc((void **) &dev_step, sizeof(double));

    float dev_input[threads];
    cudaMalloc((void **) dev_input, sizeof(float)*threads);
    
    float sum = 0;
    deviceCudaThreadSum<<<blocks, threads, threads*sizeof(double)>>>(dev_input, dev_step);
    for(int j=0;j<threads;j++){
      sum += dev_input[j];
    threadSums[j] = sum;}
    cudaFree(dev_input);
  

  deviceCudaReduction<<<blocks, threads, threads*sizeof(double)>>>(threadSums, blockSum);

	return step * blockSum[0];
}
