#include <stdlib.h>
#include <stdio.h>
#include "omp.h"


int main(int argc, char *argv[]) {

  int n = 1000;
  int numThreads;
  int breakDown;
  int *a = malloc(n*sizeof(int));
  a[0] = 0;
  int *b = malloc(n*sizeof(int));
  b[0]=0;
  int i;

  double start_t = omp_get_wtime();
  #pragma omp parallel
  {
    #pragma omp master
    {
      printf("num thread: %d\n", omp_get_num_threads());

      numThreads = omp_get_num_threads();
      breakDown = n/numThreads;
      printf("breakDown: %d", breakDown);
    }
  int threadNum = omp_get_thread_num();
      #pragma omp for
      for (i=breakDown*threadNum; i<(threadNum*breakDown)+breakDown; i++)
      {
        printf("i: %d, n: %d\n", breakDown*numThreads, (threadNum*breakDown)+breakDown );
        b[i] = b[i-1] + i;
      }
        #pragma omp for
        for(i=breakDown*numThreads; i<(threadNum*breakDown)+breakDown; i++)
        {
          #pragma omp critical
          a[i] += b[i];
        }


  }
  double end_t = omp_get_wtime();
  printf("time parallel: %f\n", end_t - start_t);

    printf("parallel a[%d] = %d\n", n-1, a[n-1]);

   start_t = omp_get_wtime();
    for (i=0; i<n; i++)
    {
      a[i] = a[i-1] + i;
    }
   end_t = omp_get_wtime();
  printf("serial a[%d] = %d\n", n-1, a[n-1]);
  printf("time serial: %f\n", end_t - start_t);
  free(a);
  printf("after free a[%d] = %d\n", n-1, a[n-1]);


     return 0;
}
