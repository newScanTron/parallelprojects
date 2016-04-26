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
  if (argc != 2) {
    fprintf(stderr, "Usage: filename\n");
    exit(1);
  }

      FILE * fp;
      char * line = NULL;
      size_t len = 0;
      ssize_t read;
      int numLines = 0;
      fp = fopen(argv[1], "r");
      if (fp == NULL)
          exit(EXIT_FAILURE);

      while ((read = getline(&line, &len, fp)) != -1) {
          // printf("Retrieved line of length %zu :\n", read);
          // printf("%s", line);
          numLines++;
      }
      double doubs[numLines];
        fclose(fp);
        fp = fopen(argv[1], "r");
        if (fp == NULL)
            exit(EXIT_FAILURE);
            numLines = 0;
      while ((read = getline(&line, &len, fp)) != -1) {
          // printf("Retrieved line of length %zu :\n", read);
           printf("wee %s", line);
           doubs[numLines] = atof(line);
           numLines++;
      }
      fclose(fp);
      //some sweet cleanup, not sure the line check is nessisary but whatever its in all the examples.s
      	 if (line)
                free(line);
//    char** tokens;
//    tokens = str_split(line, ',');

int i = 0;

int intNums[numLines];
    //  int arrayLength = (sizeof(doubs)/sizeof(double));
  //  printf("array Length: %d", arrayLength);
//	printf(" num of tokens %d\n " , (int)(sizeof(char) * tokens));
	 for (i = 0; i < numLines; i++)
        {
          printf("doubs; %0.4lf, \n" , doubs[i]);
          intNums[i] = (int)floor(doubs[i]);
	        }

  MPI_Init(&argc, &argv);
//fprintf(stderr, "this should print\n");

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//fprintf(stderr, "this should print %d\n",world_rank);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//fprintf(stderr, "this should print\n");

    int eachSize = (int)(numLines/world_size);

  // Create a random array of elements on all processes.
  srand(time(NULL)*world_rank);   // Seed the random number generator to get different results each time for each processor
  double *rand_nums = NULL;
 // rand_nums = create_rand_nums(num_elements_per_proc);
	rand_nums = doubs;
 //fprintf(stderr, "we got right before the first for lopp %d\n", world_size);
  // Sum the numbers locally
  double local_sum = 0;
  int local_int_sum = 0;
  int u;

  int localStart = world_rank * eachSize;
  int localEnd = localStart + eachSize;
  for (u = localStart; u < localEnd; u++) {
    local_sum += rand_nums[u];
    local_int_sum += intNums[u];
  }

  // Print the random numbers on each process
  printf("Local sum for process %d - %lf, avg = %f\n",
         world_rank, local_sum, local_sum / eachSize);

  // Reduce all of the local sums into the global sum
  double global_sum;
  int global_int_sum;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&local_int_sum, &global_int_sum, 1, MPI_INT, MPI_SUM, 0,
                        MPI_COMM_WORLD);

  // Print the result
  if (world_rank == 0) {
    printf("Total sum = %lf, avg = %lf\n", global_sum,
           global_sum / (world_size * eachSize));
    printf("Total int sum = %d, avg = %d\n", global_int_sum,
                  global_int_sum / (world_size * eachSize));
  }



  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
