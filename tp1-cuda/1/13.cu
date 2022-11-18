
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



__global__ void deviceSumCuda(float *pi, double *step, int *stepsPerThread){


        __shared__ float blockSum;
        if(threadIdx.x == 0)
          blockSum = 0;
      
        __syncthreads();

        int index = blockIdx.x * blockDim.x + threadIdx.x;

        float subSum = 0;
        for(int i = index * *stepsPerThread; i < (index + 1) * *stepsPerThread;i++){
          double x = (i-0.5)* *step;
          float sum = 4.0/(1.0+x*x);
          subSum += sum;
        }
        atomicAdd(&blockSum, subSum);


        __syncthreads();

        if(threadIdx.x == 0)
          atomicAdd(pi, blockSum);
}

double calculatePiAtomic(int num_steps, double step, int threads, int stepsPerThread){

  float *dev_pi, tmpPi = 0;
	double *dev_step;
  int * dev_stepsPerThread;

	cudaMalloc((void **) &dev_pi, sizeof(float));
	cudaMalloc((void **) &dev_step, sizeof(double));
	cudaMalloc((void **) &dev_stepsPerThread, sizeof(int));

	cudaMemcpy(dev_pi, &tmpPi, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_step, &step, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_stepsPerThread, &stepsPerThread, sizeof(int), cudaMemcpyHostToDevice);

  int blocks = num_steps / threads;
  blocks /= stepsPerThread;

	deviceSumCuda<<<blocks, threads>>>(dev_pi, dev_step, dev_stepsPerThread);
  
	cudaDeviceSynchronize();
  
  cudaMemcpy(&tmpPi,dev_pi,sizeof(float),cudaMemcpyDeviceToHost);

  cudaFree(dev_pi);
  cudaFree(dev_step);

	return step * tmpPi;
}
