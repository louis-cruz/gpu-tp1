/*

This program will numerically compute the integral of

                  4/(1+x*x) 
				  
from 0 to 1.  The value of this integral is pi -- which 
is great since it gives us an easy way to check the answer.

History: Written by Tim Mattson, 11/1999.
         Modified/extended by Jonathan Rouzaud-Cornabas, 10/2022
*/

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

double sumSplitArray(){
      variation = "reduction";
      double sums[num_steps] = {0};
      int cores = 8;

        double x, sum = 0.0;
      for(int j=1;j<=cores;j++){
        int sectionBegin = num_steps/cores*(j-1) +1;
        int sectionSize = num_steps/cores;
        /*
        # pragma omp parallel for private(x) reduction (+:sum)
        for (int i=sectionBegin;i<= sectionBegin+sectionSize; i++){
            x = (i-0.5)*step;
            sum = sum + 4.0/(1.0+x*x);
        }*/
        cout << num_steps/cores*(j-1) << " " << num_steps/8*j << endl;

      }
      return sum;


}

double sumReduction(){
      variation = "reduction";
      double x, sum = 0.0;
      # pragma omp parallel for private(x) reduction (+:sum)
      for (int i=1;i<= num_steps; i++){
          x = (i-0.5)*step;
          sum = sum + 4.0/(1.0+x*x);
      }
      return sum;
}

double sumCritical(){
  variation = "critical";
  double x, sum = 0.0;
  # pragma omp parallel for private(x) shared (sum)

          for (int i=1;i<= num_steps; i++){

                x = (i-0.5)*step;
            # pragma omp critical
                sum = sum + 4.0/(1.0+x*x);


          }

      return sum;
}

double sumAtomic(){
  variation = "atomic";
      double x, sum = 0.0;
    #pragma omp parallel for private (x) shared (sum)
    for (int i=1;i<= num_steps; i++){
		  x = (i-0.5)*step;
          # pragma omp atomic
          sum = sum + 4.0/(1.0+x*x);
    }
    return sum;

}

int sumStandard(){
  variation = "standard";
      double x, sum = 0.0;
          for (int i=1;i<= num_steps; i++){
              x = (i-0.5)*step;
              sum = sum + 4.0/(1.0+x*x);
          }
      return sum;
}

double calculatePi(){
  //return step * sumStandard();
  //return step * sumAtomic();
  //return step * sumCritical();
  // return step * sumReduction();
  return step * sumSplitArray();
}

int main (int argc, char** argv)
{

      // Read command line arguments.
      for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-num_steps" ) == 0 ) ) {
            num_steps = atol( argv[ ++i ] );
            printf( "  User num_steps is %ld\n", num_steps );
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Pi Options:\n" );
            printf( "  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }

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

        //writeToCSV(("%i, %lf, %s",i,time, variation));
        myfile << (variation + ", "+ to_string(0) + " ," + to_string(num_steps) + " ," + to_string(time) + "\n");

      }
  myfile.close();
        //printf("\n Average: %lf \n ",totalTime/rounds);
}
