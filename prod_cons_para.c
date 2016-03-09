#include <omp.h>
#include <stdlib.h>
#include <stdio.h>


#ifndef N
#define N 5
#endif
#ifndef FS
#define FS 38
#endif

struct node {
   char *data malloc(100*sizeof(char));
   int fibdata;
   struct node* next;
};

struct node* init_list(struct node* p);
void processwork(struct node* p);
int fib(int n);

int fib(int n)
{
   int x, y;
   if (n < 2) {
      return (n);
   } else {
      x = fib(n - 1);
      y = fib(n - 2);
          return (x + y);
   }
}

void processwork(struct node* p)
{
   int n, temp;
   n = p->data;
   temp = fib(n);

   p->fibdata = temp;

}
struct node* init_list(struct node* p)
{
    int i;
    struct node* head = NULL;
    struct node* temp = NULL;

    head = malloc(sizeof(struct node));
    p = head;
    p->data = FS;
    p->fibdata = 0;
    for (i=0; i< N; i++) {
       temp  = malloc(sizeof(struct node));
       p->next = temp;
       p = temp;
       p->data = FS + i + 1;
       p->fibdata = i+1;
    }
    p->next = NULL;
    return head;
}
int main(int argc, char * argv[])
{
  if ( argc != 2 ) /* argc should be 2 for correct execution */
  {
      /* We print argv[0] assuming it is the program name */
      printf( "usage: %s filename", argv[0] );
  }
  else
  {
      // We assume argv[1] is a filename to open
      FILE *file = fopen( argv[1], "r" );

      /* fopen returns 0, the NULL pointer, on failure */
      if ( file == 0 )
      {
          printf( "Could not open file\n" );
      }
      else
      {
          int x;
          /* read one character at a time from file, stopping at EOF, which
             indicates the end of the file.  Note that the idiom of "assign
             to a variable, check the value" used below works because
             the assignment statement evaluates to the value assigned. */
          while  ( ( x = fgetc( file ) ) != EOF )
          {
              printf( "%c", x );
          }
          fclose( file );
      }
  }


     double start, end;
     struct node *p=NULL;
     struct node *temp=NULL;
     struct node *head=NULL;

     printf("Process linked list\n");
     printf("  Each linked list node will be processed by function 'processwork()'\n");
     printf("  Each ll node will compute %d fibonacci numbers beginning with %d\n",N,FS);

     p = init_list(p);
     head = p;

     start = omp_get_wtime();

        #pragma omp parallel
        {
            #pragma omp master
                  printf("Threads:      %d\n", omp_get_num_threads());

                #pragma omp single
                {
                        p=head;
                        while (p) {
                                #pragma omp task firstprivate(p) //first private is required
                                {
                                        processwork(p);
                                }
                          p = p->next;
                   }
                }
        }
        end = omp_get_wtime();
p = head;
    while (p != NULL) {
   printf("%d : %d\n",p->data, p->fibdata);
   temp = p->next;
   free (p);
   p = temp;
}
    free (p);

printf("Compute Time: %f seconds\n", end - start);

return 0;
}
