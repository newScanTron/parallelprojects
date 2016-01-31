
#include "omp.h"
#include <stdio.h>
#include <time.h>
static long num_steps = 100000;
double step;
#define NUM_THREADS 12
int main(int argc, char *argv[])
{
  int  nthreads;
  double sum;
  step = 1.0/(double)num_steps;
  double firstTime = omp_get_wtime();

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
          //  printf("num of threads: %d %d\n", nthreads, threadNum);
          }
          #pragma omp for schedule(staic , 100)
        // for (i=start;i<localMax;i++)
        for (i=0;i<num_steps;i++)
        {
        	double x = (i+0.5)*step;
        	disSum += 4.0/(1.0+x*x);
        }
        #pragma omp atomic
          sum += disSum * step;


}
double secondTime = omp_get_wtime() - firstTime;
  printf("time: %f \n", secondTime);
      printf("pi: (%f)\n", sum);

      return 0;
}
