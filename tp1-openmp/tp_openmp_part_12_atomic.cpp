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
static int nb_core = 1;
double step;
string variation = "";

double calculatePi(){
  variation = "atomic";
      double x, sum = 0.0;
    #pragma omp parallel for private (x) shared (sum)
    for (int i=1;i<= num_steps; i++){
		  x = (i-0.5)*step;
          # pragma omp atomic
          sum = sum + 4.0/(1.0+x*x);
    }
    return step*sum;
}

int main (int argc, char** argv)
{

      // Read command line arguments.
      for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-num_steps" ) == 0 ) ) {
            num_steps = atol( argv[ ++i ] );
            printf( "  User num_steps is %ld\n", num_steps );
        } else if ( ( strcmp( argv[ i ], "-C" ) == 0 ) || ( strcmp( argv[ i ], "-nb_core" ) == 0 ) ) {
            nb_core = atol( argv[ ++i ] );
            printf( "  User nb_core is %ld\n", nb_core );
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Pi Options:\n" );
            printf( "  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }
      omp_set_num_threads(nb_core);
      step = 1.0/(double) num_steps;

      // Timer products.
      struct timeval begin, end;

      double totalTime = 0;

      std::ofstream myfile;
      myfile.open ("stats_pi.csv",ios_base::app);


        gettimeofday( &begin, NULL );
        double pi = calculatePi();
        gettimeofday( &end, NULL );

        // Calculate time.
        double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                  1.0e-6 * ( end.tv_usec - begin.tv_usec );
        totalTime += time;

        //writeToCSV(("%i, %lf, %s",i,time, variation));
        myfile << (variation + ", "+ to_string(nb_core) + " ," + to_string(num_steps) + " ," + to_string(time) + "\n");

  myfile.close();
        //printf("\n Average: %lf \n ",totalTime/rounds);
}
