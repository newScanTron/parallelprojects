
#include "omp.h"
#include <stdio.h>
static long num_steps = 100000;
double step;
#define NUM_THREADS 2
int main(int argc, char *argv[])
{
  int  nthreads;
  double  pi;
  double sum[NUM_THREADS];
  step = 1.0/(double)num_steps;


  omp_set_num_threads(NUM_THREADS);
  #pragma omp parallel
  {
//printf("num threads: (%d) \n", omp_get_num_threads());
          int numThreads = omp_get_num_threads();
        //  printf("num threads now: (%d) \n", numThreads);
          int threadNum = omp_get_thread_num();
          int threadStep = num_steps/numThreads;
          int localMax = threadStep + (threadNum * threadStep);
          int start = threadStep * threadNum;
          int i;
          double disSum;
          //printf("initial: %d\n going to: %d\n", start, localMax);
          if (threadNum == 0)
          {
            nthreads = numThreads;
            printf("num of threads: %d %d\n", nthreads, threadNum);
          }
        for (i=start;i<localMax;i++)
        {
        	double x = (i+0.5)*step;
        	disSum += 4.0/(1.0+x*x);
          //printf("summ[]: %f \n", sum[threadNum]);
        }
        #pragma omp critical
        {
          sum[threadNum] = disSum;
            printf("time: %f \n", omp_get_wtime());
        }
}
      int i;
      for (i=0;i<nthreads;i++)
      {
        pi += step * sum[i];
      }
      printf("pi: (%f)\n", pi);
      printf("time: %f \n", omp_get_wtime());
      return 0;
}
