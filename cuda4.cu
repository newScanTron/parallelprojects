//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>
#include <math.h>
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
 // Forward declaration of partition_by_bit(), called by radix_sort()
 __device__ void partition_by_bit(unsigned int *values, unsigned int bit);

 __global__
 void get_histogram(unsigned int shift,
                       unsigned int * d_bins,
                       unsigned int* const d_input,
                       const int size) {
     int index = threadIdx.x + blockDim.x * blockIdx.x;
     //check to see the index determined above is larger than size of
     if(index >= size)
         return;

     int bin_value = ((d_input[index] & (1u<<shift)) == (1u<<shift)) ? 1 : 0;
//this does really only checks if 0 or not but does the trick
     if(bin_value)
          atomicAdd(&d_bins[1], 1);
     else
          atomicAdd(&d_bins[0], 1);
 }
 //radix sort kernel
 __global__ void radix_sort(unsigned int *values)
 {
     int  bit;
     for( bit = 0; bit < 32; ++bit )
     {
         partition_by_bit(values, bit);
         __syncthreads();
     }
 }
 //scan used by partition_by_bit() kernel
 //also just learned about how to when and why to use __device__
 template<class T>
 __device__ T plus_scan(T *x)
 {
     unsigned int i = threadIdx.x; // id of thread executing this instance
     unsigned int n = blockDim.x;  // total number of threads in this block
     unsigned int offset;          // distance between elements to be added

     for( offset = 1; offset < n; offset *= 2) {
         T t;

         if ( i >= offset )
             t = x[i-offset];

         __syncthreads();

         if ( i >= offset )
             x[i] = t + x[i];      // i.e., x[i] = x[i] + x[i-1]

         __syncthreads();
     }
     return x[i];
 }
 __device__ void partition_by_bit(unsigned int *values, unsigned int bit)
 {
     unsigned int i = threadIdx.x;
     unsigned int size = blockDim.x;
     unsigned int x_i = values[i];          // value of integer at position i
     unsigned int p_i = (x_i >> bit) & 1;   // value of bit at position bit
     unsigned int * x;
     // Replace values array so that values[i] is the value of bit bit in
     // element i.
     values[i] = p_i;

     // Wait for all threads to finish this.
     __syncthreads();

     unsigned int T_before = plus_scan(values);

     unsigned int T_total  = values[size-1];

     unsigned int F_total  = size - T_total;

     __syncthreads();

     if ( p_i )
         values[T_before-1 + F_total] = x_i;
     else
         values[i - T_before] = x_i;

 }

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  //setting up the variables i will need to have a histogram space on both
  //the device and host.
    unsigned int* d_bins;
    unsigned int h_bins[2];
    unsigned int* d_scanned;
    unsigned int* d_shifted;
    const size_t histo_size = sizeof(unsigned int) * 2;
    const size_t input_size = sizeof(unsigned int) * numElems;
    checkCudaErrors(cudaMalloc(&d_bins, histo_size));
    checkCudaErrors(cudaMalloc(&d_scanned, input_size));
    checkCudaErrors(cudaMalloc(&d_shifted, input_size));
    int THREADS = 1024;
    //make blocks so we are evenly distrubuting the threads
    int BLOCK = (int)ceil(numElems/THREADS)+1;
    //(int)ceil( (float)n/(float)d ) + 1;
    printf("BLOCK size: %d\n", BLOCK);
    dim3 thread_dim(THREADS);
    dim3 histo_block_dim(BLOCK);

    unsigned int h_arr[numElems];
        checkCudaErrors(cudaMemcpy(&h_arr, d_inputVals, input_size, cudaMemcpyDeviceToHost));
      printf("element %d\n", numElems);
      //we know that unsigned ints are 32 bits in length so to iterate over each bit we need to call the get_histogram
      //kernel 32 times or "passes".
for (unsigned  int i = 0; i < 32; i ++)
{
    unsigned int one = 1;
    checkCudaErrors(cudaMemset(d_bins, 0, histo_size));
  //  checkCudaErrors(cudaMemset(d_scanned, 0, input_size));
    get_histogram<<<histo_block_dim, thread_dim>>>(i, d_bins, d_inputVals, numElems);
     cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

     checkCudaErrors(cudaMemcpy(&h_bins, d_bins, histo_size, cudaMemcpyDeviceToHost));
     printf("0: %d, 1: %d, %d %d %d \n", h_bins[0], h_bins[1], h_bins[0]+h_bins[1], numElems, (one<<i));



}

radix_sort<<<dim3(1), thread_dim>>>(d_inputVals);
   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
   checkCudaErrors(cudaMemcpy(&h_arr, d_inputVals, input_size, cudaMemcpyDeviceToHost));

for (int c = 0; c < 50; c ++)
{
  printf("sorted? %d", h_arr[c]);
}


}
