
#include <omp.h>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string>
#include "11-13.cu"
#include "14.cu"

using namespace std;

static long num_steps;
double step;

void launch(double (*func)(int, double), string variation){
  
    int rounds = 5;
      step = 1.0/(double) num_steps;
      struct timeval begin, end;
      double totalTime = 0;
      ofstream myfile;
      myfile.open ("result_log.csv",ios_base::app);

      for(int i=0;i<rounds;i++){
        gettimeofday( &begin, NULL );
        //printf("PI: %f\n",func(num_steps, step));
        func(num_steps, step);
        gettimeofday( &end, NULL );

        double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                  1.0e-6 * ( end.tv_usec - begin.tv_usec );
        totalTime += time;
        myfile << (variation + ", "+ to_string(0) + " ," + to_string(num_steps) + " ," + to_string(time) + "\n");
      }
      myfile.close();
}

int main (int argc, char** argv)
{
    num_steps = 10000000;
    launch(&calculatePiAtomic, "atomic");
    launch(&calculatePiReduction, "reduction");
}
