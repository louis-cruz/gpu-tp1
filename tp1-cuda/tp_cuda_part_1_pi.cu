
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

static long num_steps = 10000000;
double step;
string variation = "";


__global__ void deviceSumCuda(double *pi, int steps){
	
      double x, sum = 0.0;
          for (int i=1;i <= steps; i++){
              x = (i-0.5)*step;
              sum = sum + 4.0/(1.0+x*x);
          }
      *pi = sum;
}


double calculatePi(){
	variation = "cuda"
	double *pi;
	cudaMallocManaged(&pi);
	deviceSumCuda<<<1,256>>>(pi, num_steps);
	cudaDeviceSynchronize();
	return step * *pi();
}

int main (int argc, char** argv)
{

      int rounds = 10;


      step = 1.0/(double) num_steps;

      // Timer products.
      struct timeval begin, end;

      double totalTime = 0;

      std::ofstream myfile;
      myfile.open ("result_log.csv",ios_base::app);

      for(int i=0;i<rounds;i++){
        gettimeofday( &begin, NULL );
        double pi = calculatePi();
        gettimeofday( &end, NULL );

        // Calculate time.
        double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                  1.0e-6 * ( end.tv_usec - begin.tv_usec );
        totalTime += time;

        myfile << (variation + ", "+ to_string(0) + " ," + to_string(num_steps) + " ," + to_string(time) + "\n");

      }
  myfile.close();
}
