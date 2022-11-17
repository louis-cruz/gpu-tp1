
#include <omp.h>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string>
#include "13.cu"
#include "14.cu"

using namespace std;

void launch(double (*func)(int, double, int), string variation, long num_steps, int threads, int rounds){
  
    double step;
    step = 1.0/(double) num_steps;
    struct timeval begin, end;
    double totalTime = 0;
    ofstream myfile;
    myfile.open ("result_log.csv",ios_base::app);

    for(int i=0;i<rounds;i++){
      gettimeofday( &begin, NULL );
      printf("PI: %f\n",func(num_steps, step, threads));
      func(num_steps, step, threads);
      gettimeofday( &end, NULL );

      double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );
      totalTime += time;
      myfile << variation + ", " << 0 << " ," << num_steps << " ," << time << "\n";
    }
    myfile.close();
}

void evaluate(){

  int threads_per_block[] = {1, 32, 64, 128, 256};
  //int steps_per_thread[] = {1, 64, 256, 1024};
  
  int rounds = 1;
  for(int num_steps = 1000000; num_steps <= 1000000000000; num_steps*=100){
    for(int i = 0; i <= 4;i++){
      launch(&calculatePiAtomic, "atomic", num_steps, threads_per_block[i], rounds);
      launch(&calculatePiReduction, "reduction", num_steps, threads_per_block[i], rounds);
      }
    }
}

int main (int argc, char** argv)
{
      //evaluate();
      //launch(&calculatePiAtomic, "atomic", 1000000, 256, 1);
      launch(&calculatePiReduction, "reduction", 1000000, 256, 5);
}
