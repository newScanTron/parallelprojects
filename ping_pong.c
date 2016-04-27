
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  int MESSAGE_COUNT = 1000;
  // Initialize the MPI environment
  MPI_Init(&argc, &argv);
  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // We are assuming at least 2 processes for this task
  if (world_size != 2) {
    fprintf(stderr, "World size must be two for %s\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int ping_pong_count = 0;
  //int partner_rank = (world_rank + 1) % 2;

    if (world_rank == 0) {
      // Increment the ping pong count before you send it
      double startTime = MPI_Wtime();
      printf("startTime = %lf\n", startTime);

      int i = 0;
      for (i = 0; i < MESSAGE_COUNT; i++)
      {
        ping_pong_count++;
        MPI_Send(&ping_pong_count, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&ping_pong_count, 1, MPI_INT, 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      }
      double endTime = MPI_Wtime();
      double elapsedTime = endTime - startTime;
    double averageTime = elapsedTime /(2*MESSAGE_COUNT);
      printf("endTime: %lf, total time: %lf, average message time: %lf\n", endTime, elapsedTime , averageTime);


    } else if (world_rank == 1){
      int i = 0;
      for (i = 0; i < MESSAGE_COUNT; i++)
      {
      MPI_Recv(&ping_pong_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Send(&ping_pong_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

      }
    }

  MPI_Finalize();
}
