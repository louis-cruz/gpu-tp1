
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



__global__ void deviceSumCuda(float *pi, double *step){


        __shared__ float blockSum;
        if(threadIdx.x == 0)
          blockSum = 0;
      
        __syncthreads();

        int index = blockIdx.x * blockDim.x + threadIdx.x;

        double x = (index-0.5)* *step;
        float sum = 4.0/(1.0+x*x);

        atomicAdd(&blockSum, sum);

        __syncthreads();

        if(threadIdx.x == 0)
          atomicAdd(pi, blockSum);
}

double calculatePiAtomic(int num_steps, double step){

  float *dev_pi, tmpPi = 0;
	double *dev_step;

	cudaMalloc((void **) &dev_pi, sizeof(float));
	cudaMalloc((void **) &dev_step, sizeof(double));

	cudaMemcpy(dev_pi, &tmpPi, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_step, &step, sizeof(double), cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = num_steps / threads;

	deviceSumCuda<<<blocks, threads>>>(dev_pi, dev_step);
  
	cudaDeviceSynchronize();
  
  cudaMemcpy(&tmpPi,dev_pi,sizeof(float),cudaMemcpyDeviceToHost);

  cudaFree(dev_pi);
  cudaFree(dev_step);

	return step * tmpPi;
}
