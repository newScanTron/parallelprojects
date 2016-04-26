// Author: Wes Kendall
// Copyright 2013 www.mpitutorial.com
// This code is provided freely with the tutorials on mpitutorial.com. Feel
// free to modify it for your own use. Any distribution of the code must
// either provide a link to www.mpitutorial.com or keep this header intact.
//
// Program that computes the average of an array of elements in parallel using
// MPI_Reduce.
//
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// Creates an array of random numbers. Each number has a value from 0 - 1
int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "Usage: you done it wrong. date file name, number of foat nodes, number of integer nodes.\n");
    exit(1);
  }

      FILE * fp;
      char * line = NULL;
      size_t len = 0;
      ssize_t read;
      int numLines = 0;
      int numFloatNodes = atoi(argv[2]);
      int numIntNodes = atoi(argv[3]);
      fp = fopen(argv[1], "r");
      if (fp == NULL)
          exit(EXIT_FAILURE);
//c is just a weter languane; two loops are needed.
      while ((read = getline(&line, &len, fp)) != -1) {
          numLines++;
      }
      //going to create an array of doubles to process and convert into ints.
      double doubs[numLines];
      int ints[numLines];
//clean up the file. and then re-open it cus i just can't read about better ways.
        fclose(fp);
        fp = fopen(argv[1], "r");
        if (fp == NULL)
            exit(EXIT_FAILURE);
            numLines = 0;
      while ((read = getline(&line, &len, fp)) != -1) {
           doubs[numLines] = atof(line);
           ints[numLines] = (int)floor(doubs[numLines]);
           numLines++;
      }
      fclose(fp);
      //some sweet cleanup, not sure the line check is nessisary but whatever its in all the examples.s
      	 if (line)
                free(line);

  MPI_Init(&argc, &argv);
//fprintf(stderr, "this should print\n");

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//fprintf(stderr, "this should print %d\n",world_rank);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//fprintf(stderr, "this should print\n");

    int eachDoubleSize = (int)(numLines/numFloatNodes);
    int eachIntSize = (int)(numLines/numIntNodes);

  double local_sum = 0;
  int local_int_sum = 0;
  int u;
//this little bit uses each node to cacluale their part of the array like you would in cuda.  Im going to leave it in  just for kicks.
if (world_rank < numFloatNodes)
{
  int localDoubleStart = world_rank * eachDoubleSize;
  int localDoubleEnd = localDoubleStart + eachDoubleSize;
  for (u = localDoubleStart; u < localDoubleEnd; u++) {
    local_sum += doubs[u];

  }
  // Print the random numbers on each process
  printf("Local double sum for process %d : %lf, avg = %lf\n",
         world_rank, local_sum, local_sum / eachDoubleSize);
}
int j;
if (world_rank >= numFloatNodes && world_rank < (numFloatNodes + numIntNodes))
{
  int localIntStart = (world_rank - numFloatNodes) * eachIntSize;
  int localIntEnd = localIntStart + eachIntSize;
  printf("local int start: %d. end : %d", localIntStart, localIntEnd);
  for (j = localIntStart; j < localIntEnd; j++)
  {
        local_int_sum += ints[j];
  }

  printf("Local int sum for process %d : %d, avg = %d\n",
         world_rank, local_int_sum, (int)(local_int_sum / eachIntSize));

}

  // Reduce all of the local sums into the global sum
  double global_sum;
  int global_int_sum;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&local_int_sum, &global_int_sum, 1, MPI_INT, MPI_SUM, 0,
                        MPI_COMM_WORLD);

  // Print the result
  if (world_rank == 0) {
    printf("\nTotal sum = %lf, avg = %lf\n", global_sum,
           global_sum / (world_size * eachDoubleSize));
    printf("Total int sum = %d, avg = %d\n\n", global_int_sum,
                  (int)global_int_sum / (world_size * eachDoubleSize));
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
