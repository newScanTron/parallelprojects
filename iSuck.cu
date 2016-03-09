

#include <float.h>
#include <math.h>
#include <stdio.h>

#include "utils.h"


/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.
   Note: ascending order == smallest to largest
   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.
   Implementing Parallel Radix Sort with CUDA
   ==========================================
   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.
   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there
   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.
 */

__global__
void histogram_kernel(unsigned int pass,
                      unsigned int * d_bins,
                      unsigned int* const d_input,
                      const int size) {
    int mid = threadIdx.x + blockDim.x * blockIdx.x;
    if(mid >= size)
        return;
    unsigned int one = 1;
    int bin = ((d_input[mid] & (one<<pass)) == (one<<pass)) ? 1 : 0;
    if(bin)
         atomicAdd(&d_bins[1], 1);
    else
         atomicAdd(&d_bins[0], 1);
}

// we will run 1 exclusive scan, but then when we
// do the move, for zero vals, we iwll take mid - val of scan there
__global__
void exclusive_scan_kernel(unsigned int pass,
                    unsigned int const * d_inputVals,
                    unsigned int * d_output,
                    const int size,
                    unsigned int base,
                    unsigned int threadSize) {
    int mid = threadIdx.x + threadSize * base;
   
        unsigned int one = 1;

    if(mid >= size)
        return;
      unsigned int val = 0;
    if(mid > 0)
        val = ((d_inputVals[mid-1] & (one<<pass))  == (one<<pass)) ? 1 : 0;
    else
        val = 0;

    d_output[mid] = val;

    __syncthreads();

    for(int s = 1; s <= threadSize; s *= 2) {
        int spot = mid - s;

        if(spot >= 0 && spot >=  threadSize*base)
             val = d_output[spot];
        __syncthreads();
        if(spot >= 0 && spot >= threadSize*base)
            d_output[mid] += val;
        __syncthreads();
    }
    if(base > 0)
        d_output[mid] += d_output[base*threadSize - 1];

}

__global__
void move_kernel(
    unsigned int pass,
    unsigned int* const d_inputVals,
    unsigned int* const d_inputPos,
    unsigned int* d_outputVals,
    unsigned int* d_outputPos,
    unsigned int* d_outputMove,
    unsigned int* const d_scanned,
    unsigned int  one_pos,
    const size_t numElems) {

    int mid = threadIdx.x + blockDim.x * blockIdx.x;
    if(mid >= numElems)
        return;

    unsigned int scan=0;
    unsigned int base=0;
    unsigned int one= 1;
    if( ( d_inputVals[mid] & (one<<pass)) == (1<<pass)) {
        scan = d_scanned[mid];
        base = one_pos;
    } else {
        scan = (mid) - d_scanned[mid];
        base = 0;
    }

    d_outputMove[mid] = base+scan;
    d_outputPos[base+scan]  = d_inputPos[mid];//d_inputPos[0];
    d_outputVals[base+scan] = d_inputVals[mid];//base+scan;//d_inputVals[0];

}




int get_max_size(int n, int d) {
    return (int)ceil( (float)n/(float)d ) + 1;
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{

    unsigned int* d_bins;
    unsigned int  h_bins[2];
    unsigned int* d_scanned;
    unsigned int* d_moved;
    const size_t histo_size = sizeof(unsigned int) * 2;
    const size_t input_size   = sizeof(unsigned int) * numElems;

    checkCudaErrors(cudaMalloc(&d_bins, histo_size));
    checkCudaErrors(cudaMalloc(&d_scanned, input_size));
    checkCudaErrors(cudaMalloc(&d_moved, input_size));
    // just keep thread dim at 1024
    int THREADS = 1024;
    //make blocks so we are evenly distrubuting the threads
    int BLOCK = (int)ceil(numElems/THREADS)+1;
    //(int)ceil( (float)n/(float)d ) + 1;
    printf("BLOCK size: %d\n", BLOCK);
    dim3 thread_dim(THREADS);
    dim3 histo_block_dim(BLOCK);
    for(unsigned int pass = 0; pass < 32; pass++) {
        unsigned int one = 1;
//set each array to 0 to make sure we are dealing with fresh data.
        checkCudaErrors(cudaMemset(d_bins, 0, histo_size));
        checkCudaErrors(cudaMemset(d_scanned, 0, input_size));
        checkCudaErrors(cudaMemset(d_outputVals, 0, input_size));
        checkCudaErrors(cudaMemset(d_outputPos, 0, input_size));

        histogram_kernel<<<histo_block_dim, thread_dim>>>(pass, d_bins, d_inputVals, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        // copy the histogram data to host
        checkCudaErrors(cudaMemcpy(&h_bins, d_bins, histo_size, cudaMemcpyDeviceToHost));

        printf("0: %d, 1: %d, %d %d %d \n", h_bins[0], h_bins[1], h_bins[0]+h_bins[1], numElems, (one<<i));

        // now we have 0, and 1 start position..
        // get the scan of 1's

        for(int i = 0; i < get_max_size(numElems, thread_dim.x); i++) {
            exclusive_scan_kernel<<<dim3(1), thread_dim>>>(
                   pass,
                   d_inputVals,
                   d_scanned,
                   numElems,
                   i,
                   thread_dim.x
            );
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        }
        // calculate the move positions
        move_kernel<<<histo_block_dim, thread_dim>>>(
            pass,
            d_inputVals,
            d_inputPos,
            d_outputVals,
            d_outputPos,
            d_moved,
            d_scanned,
            h_bins[0],
            numElems
        );
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        //finall
         // copy the histogram data to input
        checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, input_size, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_inputPos, d_outputPos, input_size, cudaMemcpyDeviceToDevice));
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }
    checkCudaErrors(cudaFree(d_moved));
    checkCudaErrors(cudaFree(d_scanned));
    checkCudaErrors(cudaFree(d_bins));
}
