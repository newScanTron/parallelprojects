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
char** str_split(char* a_str, const char a_delim)
{
    char** result    = 0;
    size_t count     = 0;
    char* tmp        = a_str;
    char* last_comma = 0;
    char delim[2];
    delim[0] = a_delim;
    delim[1] = 0;

    /* Count how many elements will be extracted. */
    while (*tmp)
    {
        if (a_delim == *tmp)
        {
            count++;
            last_comma = tmp;
        }
        tmp++;
    }

    /* Add space for trailing token. */
    count += last_comma < (a_str + strlen(a_str) - 1);

    /* Add space for terminating null string so caller
 *        knows where the list of returned strings ends. */
    count++;
    result = malloc(sizeof(char*) * count);

    if (result)
    {
        size_t idx  = 0;
        char* token = strtok(a_str, delim);

        while (token)
        {
            assert(idx < count);
            *(result + idx++) = strdup(token);
            token = strtok(0, delim);
        }
        assert(idx == count - 1);
        *(result + idx) = 0;
    }

    return result;
}
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
	    float * nums;
	    int * intNums;
      int arraySize;
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
printf ( "dope brow: %lf", doubs[0]);
    char** tokens;
    tokens = str_split(line, ',');

    if (tokens)
    {
        int i;
      float  chocFloat;
//	printf(" num of tokens %d\n " , (int)(sizeof(char) * tokens));
	 for (i = 0; *(tokens + i); i++)
        {
          //  printf("month=[%s]\n", *(tokens + i));
        	chocFloat = atof(*(tokens + i));
	        }
        printf("tokens %d\n",i);

	int j;
	nums = malloc(i * sizeof(float));
	intNums = malloc(i * sizeof(int));
  arraySize = i;

//array to fill each float as a float also going to do out conversion to ints
	for (j = 0; j < i; j++)
	{
		chocFloat = atof(*(tokens + j));
		nums[j] = chocFloat;

	 	intNums[j] = floor(chocFloat);
		free(*(tokens + j));

	}
	free(tokens);

    }


//some sweet cleanup
	 if (line)
          free(line);


//fprintf(stderr, "this should print\n");

//fprintf(stderr, "this should print also\n");

  MPI_Init(&argc, &argv);
//fprintf(stderr, "this should print\n");

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//fprintf(stderr, "this should print %d\n",world_rank);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//fprintf(stderr, "this should print\n");

    int eachSize = (int)(arraySize/world_size);

  // Create a random array of elements on all processes.
  srand(time(NULL)*world_rank);   // Seed the random number generator to get different results each time for each processor
  float *rand_nums = NULL;
 // rand_nums = create_rand_nums(num_elements_per_proc);
	rand_nums = nums;
 //fprintf(stderr, "we got right before the first for lopp %d\n", world_size);
  // Sum the numbers locally
  float local_sum = 0;
  int local_int_sum = 0;
  int u;

  int localStart = world_rank * eachSize;
  int localEnd = localStart + eachSize;
  for (u = localStart; u < localEnd; u++) {
    local_sum += rand_nums[u];
    local_int_sum += intNums[u];
  }

  // Print the random numbers on each process
  printf("Local sum for process %d - %f, avg = %f\n",
         world_rank, local_sum, local_sum / eachSize);

  // Reduce all of the local sums into the global sum
  float global_sum;
  int global_int_sum;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&local_int_sum, &global_int_sum, 1, MPI_INT, MPI_SUM, 0,
                        MPI_COMM_WORLD);

  // Print the result
  if (world_rank == 0) {
    printf("Total sum = %f, avg = %f\n", global_sum,
           global_sum / (world_size * eachSize));
    printf("Total int sum = %d, avg = %d\n", global_int_sum,
                  global_int_sum / (world_size * eachSize));
  }

  // Clean up
  free(rand_nums);


  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
