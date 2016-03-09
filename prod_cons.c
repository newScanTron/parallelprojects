#include <omp.h>
#ifdef APPLE
#include <stdlib.h>
#endif
#include <stdio.h>

#define N        10000

/* Some random number constants from numerical recipies */
#define SEED       2531
#define RAND_MULT  1366
#define RAND_ADD   150889
#define RAND_MOD   714025
int randy = SEED;
char ch;

void Tokenize(
    char * lines[],
    int line_count,
    int thread_count)
    {
      int my_rank, i, j;
      char *my_token;

      #pragma omp parallel num_threads(thread_count) default(none) private(my_rank, i, j, my_token) shared(lines, line_count)
      {
        my_rank = omp_get_thread_num();
        #pragma omp for schedule(static, 1)
        for (i = 0; i < line_count; i++)
        {
          printf("thread %d > line %d = %s", my_rank, i, lines[i]);
          j = 0;
          my_token = strtok_(lines[i], " ");
          while (my_token != NULL)
          {
            printf("thread %d > token %d = %s\n", my_rank, j, my_token);
            my_token = strtok_r(NULL, " ");
            j++;
          }
        }

      }
    }


char read_file(char name)
{



  char *chAr = malloc(100*sizeof(char));
   FILE *fp;
   printf( " %s\n", name );
   fp = fopen(&name,"r"); // read mode

   if( fp == NULL )
   {
      perror("Error while opening the file.\n");
      exit(EXIT_FAILURE);
   }
   char *temp;
   char buff[255];

fscanf(fp, "%s", buff);
   char *token;
   char s[2] = " ";
   /* get the first token */
   token = strtok(buff, s);

   /* walk through other tokens */
   while( token != NULL )
   {
      printf( " %s\n", token );

      token = strtok(NULL, s);
   }

   printf("The contents of %s file are :\n", temp);
   int i = 0;
   fscanf(fp, "%s", buff);
   printf("buff: %s", buff);

  //  while( ( ch = fgetc(fp) ) != EOF )
  //   {
  //      chAr[i] = ch;
  //     printf("%c",ch);
  //     i++;
  //   }


  fclose(fp);
  return *chAr;

}

/* function to sum the elements of an array */
double Sum_array(int length, double *a)
{
   int i;  double sum = 0.0;
   for (i=0;i<length;i++)  sum += *(a+i);
   return sum;
}

int main(int argc, char *argv[])
{
  double *A, sum, runtime;
  int flag = 0;
  char *chaAr = malloc(100*sizeof(char));
  A = (double *)malloc(N*sizeof(double));

  runtime = omp_get_wtime();
  *chaAr = read_file(*argv[1]);
  int i = 0;
  // for (i=0;i<100;i++)
  // {
    printf("%c", chaAr[1]);
  //}
  // while( ( ch = fgetc(fp) ) != EOF )
  //    printf("%c",ch);
//  fill_rand(N, A);        // Producer: fill an array of data

//  sum = Sum_array(N, A);  // Consumer: sum the array

  runtime = omp_get_wtime() - runtime;

  printf(" In %f seconds, The sum is %f \n",runtime,sum);

  return 0;
}
