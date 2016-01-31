
#include "omp.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
static long num_steps = 1000000;
double step;
#define NUM_THREADS 8
int main(int argc, char *argv[])
{
  double sum = 0.0;
  step = 1.0/(double)num_steps;
  double firstTime = omp_get_wtime();
printf("max threads: %d \n", omp_get_max_threads());
printf("number of procssors: %d \n", omp_num_procs());

  omp_set_num_threads(NUM_THREADS);
double pi = 0.0;
int i =0;
  #pragma omp parallel
  {
        #pragma omp for reduction(+:pi)
        for (i=0;i<num_steps;i++)
        {
        	double x = (i+0.5)*step;
        	pi += 4.0/(1.0+x*x);
        }

}
sum += pi * step;
double secondTime = (omp_get_wtime() - firstTime) * 1000;
  printf("time: %f \n", secondTime);
      printf("pi: (%f)\n", sum);

      return 0;
}
