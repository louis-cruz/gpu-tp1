
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


__global__ void deviceCudaReduction(float *blockSums, float *reducedSum) {

    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = blockSums[i];

    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        /*if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }*/

        int index = 2 * s * threadIdx.x;;

        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }

        __syncthreads();
    }

    if (tid == 0) {
      //reducedSum[blockIdx.x] = sdata[0];
      atomicAdd(reducedSum, sdata[0]);
      }
}

__global__ void deviceCudaBlockSum(float *blockSums, double *step){


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
          blockSums[blockIdx.x] = blockSum;

        __syncthreads();
}

double calculatePiReduction(int num_steps, double step, int threads){

    // variables used for both stages
    int blocks = num_steps / threads;
    float *dev_blockSums;
    float host_blockSums[blocks] = {0};




    //------------------ stage1 ---------------------------------------//
    // define host and device variables
    double *dev_step;
    // gpu memory allocation and copy
    cudaMalloc((void **) &dev_step, sizeof(double));
    cudaMalloc((void **) &dev_blockSums, sizeof(float)*blocks);
	  cudaMemcpy(dev_blockSums, host_blockSums, sizeof(float)*blocks, cudaMemcpyHostToDevice);
	  cudaMemcpy(dev_step, &step, sizeof(double), cudaMemcpyHostToDevice);
    // run kernel to compute individual block sums
    deviceCudaBlockSum<<<blocks, threads>>>(dev_blockSums, dev_step);
    // transfer data from gpu to main memeory
	  cudaMemcpy(host_blockSums,dev_blockSums, sizeof(float)*blocks, cudaMemcpyDeviceToHost);
    // liberate memory
    cudaFree(dev_blockSums);

    //----------------- stage 2 ----------------------------------------//
    // define host and device variables
    //float host_reduced[blocks] = {0};
    float *dev_reduced;
    float temp = 0;
    float *host_reduced =  &temp;
    // gpu memory allocation and copy
    cudaMalloc((void **) &dev_reduced, sizeof(float));
    cudaMalloc((void **) &dev_blockSums, sizeof(float)*blocks);
	  cudaMemcpy(dev_blockSums, host_blockSums, sizeof(float)*blocks, cudaMemcpyHostToDevice);
	  cudaMemcpy(dev_reduced, host_reduced, sizeof(float), cudaMemcpyHostToDevice);
    // run kernel to compute total block sum
    blocks = blocks/threads;
    deviceCudaReduction<<<blocks, threads, threads*sizeof(float)>>>(dev_blockSums, dev_reduced);
    // transfer data from gpu to main memeory
	  cudaMemcpy(host_reduced, dev_reduced, sizeof(float), cudaMemcpyDeviceToHost);
    // liberate memory
    cudaFree(dev_blockSums);
    cudaFree(dev_reduced);



    cout<< *host_reduced * step<< endl;
    //--------------------------------------------------------------//

    return step * *host_reduced;
}

