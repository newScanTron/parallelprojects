

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
//fairly straight forward kernel that is called once for all 32 bits of
//a unsigned int collecting the number of 1 and 0s.
__global__
void get_histogram(unsigned int iter_num,
                      unsigned int * d_bins,
                      unsigned int* const d_input,
                      const int size) {
    int mid = threadIdx.x + (blockDim.x * blockIdx.x);
    if(mid >= size)
        return;
    unsigned int one = 1;
    int bin = ((d_input[mid] & (one<<iter_num)) == (one<<iter_num)) ? 1 : 0;
    if(bin)
         atomicAdd(&d_bins[1], 1);
    else
         atomicAdd(&d_bins[0], 1);
}

//this exclusive scan is based on the example http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
//it will only work on one block of threads working on a singel SM.
//the look up into d_inputVals on line 72 would case race conditions
//when we went to look up the last element of the previous block.
__global__
void exclusive_scan(unsigned int iter_num,
                    unsigned int const * d_inputVals,
                    unsigned int * d_output,
                    const int size,
                    unsigned int base) {
    int main_id = threadIdx.x + blockDim.x * base;
//make sure we are in the image and if main_id is zero just set val = 0;
    if(main_id >= size)
        return;
    //this is the meat of the scan checking the bits at the location of the
    //main_id-1 then setting the d_output[main_id] to that
    unsigned int val = 0;
    if(main_id > 0)
        val = ((d_inputVals[main_id-1] & (1u<<iter_num))  == (1u<<iter_num)) ? 1 : 0;
    else
        val = 0;
    //write to the global array then sync the threads on the one block we are
    //working with
    d_output[main_id] = val;
    __syncthreads();

    for(int s = 1; s <= blockDim.x ; s *= 2)
    {
        int spot = main_id - s;
//syncthreads is needed here because we are reading from and writing to d_output
//atomicAdd would be better for the second but seems not to be implemented for
//usigned int.
        if(spot >= 0 && spot >=  blockDim.x *base)
             val = d_output[spot];
        __syncthreads();
        if(spot >= 0 && spot >= blockDim.x *base)
            d_output[main_id] += val;
        __syncthreads();
    }
    if(base > 0)
    //don't actually need this atomicAdd because this kernel
    //will only run on one block of theads at a time i was trying to
    //make it
        atomicAdd(&d_output[main_id] , d_output[base*blockDim.x  - 1]);

}

__global__
void combine(
    unsigned int iter_num,
    unsigned int* const d_inputVals,
    unsigned int* const d_inputPos,
    unsigned int* d_outputVals,
    unsigned int* d_outputPos,
    unsigned int* const d_scanned,
    unsigned int  ones_pos,
    const size_t numElems) {
//standard geting of main_id based on blcok and thread ids
    int main_id = threadIdx.x + blockDim.x * blockIdx.x;
    if(main_id >= numElems)
        return;
    //
    unsigned int index=0;
    unsigned int base=0;
    if( ( d_inputVals[main_id] & (1u<<iter_num)) == (1u<<iter_num))
    {
        index = d_scanned[main_id];
        base = ones_pos;
    } else {
        index = (main_id) - d_scanned[main_id];
    }

    d_outputPos[base+index]  = d_inputPos[main_id];
    d_outputVals[base+index] = d_inputVals[main_id];

}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    //alocate space for the needed histogram arrays
    unsigned int* d_bins;
    unsigned int  h_bins[2];
    //an array for the combine kernel to use
    unsigned int* d_scanned;
    //size constents for allocation
    const size_t histo_size = sizeof(unsigned int) * 2;
    const size_t input_size   = sizeof(unsigned int) * numElems;

    checkCudaErrors(cudaMalloc(&d_bins, histo_size));
    checkCudaErrors(cudaMalloc(&d_scanned, input_size));
    //Set thread and block size based on the number of threads
    int THREADS = 1024;
    //make blocks so we are evenly distrubuting the threads
    int BLOCK = (int)ceil(numElems/THREADS)+1;
    dim3 thread_dim(THREADS);
    dim3 histo_block_dim(BLOCK);
    for(unsigned int iter_num = 0; iter_num < 32; iter_num++) {
        //set each array to 0 to make sure we are dealing with fresh data.
        checkCudaErrors(cudaMemset(d_bins, 0, histo_size));
        checkCudaErrors(cudaMemset(d_scanned, 0, input_size));
        checkCudaErrors(cudaMemset(d_outputVals, 0, input_size));
        checkCudaErrors(cudaMemset(d_outputPos, 0, input_size));
        //make the histogram of bits
        get_histogram<<<histo_block_dim, thread_dim>>>(iter_num, d_bins, d_inputVals, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        // copy the histogram data to host
        checkCudaErrors(cudaMemcpy(&h_bins, d_bins, histo_size, cudaMemcpyDeviceToHost));
        //I tried to do something better here but was unable, so
        //iterating over one block at a time it is
        for(int i = 0; i < BLOCK; i++) {
            exclusive_scan<<<dim3(1), thread_dim>>>(
                   iter_num,
                   d_inputVals,
                   d_scanned,
                   numElems,
                   i
            );
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        }
        // calculate the move positions based on histogram and scaned
        combine<<<histo_block_dim, thread_dim>>>(
            iter_num,
            d_inputVals,
            d_inputPos,
            d_outputVals,
            d_outputPos,
            d_scanned,
            h_bins[0],
            numElems
        );
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        //finally copy the the output data to the input data
        checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, input_size, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_inputPos, d_outputPos, input_size, cudaMemcpyDeviceToDevice));
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }
    //free the stuff whe cudaMalloced
    checkCudaErrors(cudaFree(d_scanned));
    checkCudaErrors(cudaFree(d_bins));
}
