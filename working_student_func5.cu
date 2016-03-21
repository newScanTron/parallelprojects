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

    // int main_id = threadIdx.x + blockDim.x * blockIdx.x;
    // if(main_id >= numVals)
    //     return;
    // else
    //     histo[vals[main_id]]++;
        int main_id = threadIdx.x + blockDim.x * blockIdx.x;
        if(main_id >= numVals)
            return;

        atomicAdd(&histo[vals[main_id]], 1);


    //Although we provide only one kernel skeleton,
    //feel free to use more if it will help you
    //write faster code
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
int THREADS = 1024;
  int BLOCKS =  (int)ceil( (float)numElems/(float)THREADS) + 1;
  dim3 thread_dim(THREADS);
  dim3 block_dim(BLOCKS);
    //TODO Launch the yourHisto kernel
    yourHisto<<<block_dim, thread_dim>>>(d_vals, d_histo, numElems);
    //if you want to use/launch more than one kernel,
    //feel free
    unsigned int * h_vals = new unsigned int[numElems];
    unsigned int * h_histo = new unsigned int[numBins];
    unsigned int * your_histo = new unsigned int[numBins];
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(h_histo, d_histo, sizeof(numBins),cudaMemcpyDeviceToHost));

    // for (int c = 0; c < numBins; c++)
    // {
    //   printf("bin %f: %f", c, h_histo[c] );
    // }
    delete[] h_vals;
    delete[] h_histo;
    delete[] your_histo;
}
