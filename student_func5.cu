/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include "reference.cpp"
#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
    __shared__ unsigned int shared_bin[1024];
    int main_id = threadIdx.x + blockDim.x * blockIdx.x;
    int start_main = main_id;

      shared_bin[threadIdx.x] = 0;
      __syncthreads();

        int offset = blockDim.x * gridDim.x;
        if(main_id >= numVals)
            return;
        while (main_id < numVals)
        {
            atomicAdd(&shared_bin[vals[main_id]], 1);
            main_id += offset;
        }
        __syncthreads();

          atomicAdd(&histo[threadIdx.x] ,shared_bin[threadIdx.x]);

}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
int THREADS = 1024;
  int BLOCKS =  (int)ceil( (float)numElems/(float)THREADS);
  printf("Blocks: %d\n", BLOCKS);
  dim3 thread_dim(THREADS);
  dim3 block_dim(BLOCKS);

  const unsigned int shared_mem_size = sizeof(unsigned int) * numBins;


// cudaDeviceProp prop;
//  checkCudaErrors( cudaGetDeviceProperties( &prop, 0 ) );
//  int blocks = prop.multiProcessorCount;
//  std::cout << blocks << std::endl;

  dim3 other_block_dim(14*2);
    //TODO Launch the yourHisto kernel
    yourHisto<<<other_block_dim, thread_dim>>>(d_vals, d_histo, numElems);
    //if you want to use/launch more than one kernel,
    //feel free

    unsigned int * h_histo = new unsigned int[numBins];

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(h_histo, d_histo, sizeof(numBins),cudaMemcpyDeviceToHost));


    delete[] h_histo;
}
