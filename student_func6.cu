//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.

      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly -
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>
#include "reference_calc.cpp"
#include <float.h>
#include <math.h>
#include <stdio.h>
//pre-compute the values of g, which depend only the source image
//and aren't iteration dependent.
void compute_G(const unsigned char* const channel,
              float* const g,
              const size_t numColsSource,
              const std::vector<uint2>& interiorPixelList)
{
  for (size_t i = 0; i < interiorPixelList.size(); ++i) {
    uint2 coord = interiorPixelList[i];
    unsigned int offset = coord.x * numColsSource + coord.y;

    float sum = 4.f * channel[offset];

    sum -= (float)channel[offset - 1] + (float)channel[offset + 1];
    sum -= (float)channel[offset + numColsSource] + (float)channel[offset - numColsSource];

    g[offset] = sum;
  }
}
// __global__
// void comp_G(const unsigned char* const channel,
//               float* const g,
//               const size_t numColsSource,
//               int size,
//               const std::vector<uint2> interiorPixelList,
//               const size_t numRows)
// {
//   int main_x = threadIdx.x + blockDim.x * blockIdx.x;
//   int main_y = threadIdx.y + blockDim.y * blockIdx.y;
//   int i = main_x + main_y * numColsSource;
//    if (i < size)
//    {
//      uint2 coord = interiorPixelList[i];
//      unsigned int offset = coord.x * numColsSource + coord.y;
//
//      float sum = 4.f * channel[offset];
//
//      sum -= (float)channel[offset - 1] + (float)channel[offset + 1];
//      sum -= (float)channel[offset + numColsSource] + (float)channel[offset - numColsSource];
//
//      g[offset] = sum;
//    }
// }
__global__
void addToBlended(float * blendedValsRed_1,
                  float * blendedValsRed_2,
                  float * blendedValsBlue_1,
                  float * blendedValsBlue_2,
                  float * blendedValsGreen_1,
                  float * blendedValsGreen_2,
                  unsigned char* d_red_src,
                  unsigned char* d_blue_src,
                  unsigned char* d_green_src,
                  const size_t numCols,
                  const size_t numRows)
{
  const size_t srcSize = numCols * numRows;
  int main_x = threadIdx.x + blockDim.x * blockIdx.x;
  int main_y = threadIdx.y + blockDim.y * blockIdx.y;
  int i = main_x + main_y * numCols;


  if (i < srcSize)
  {
    blendedValsRed_1[i] = d_red_src[i];
    blendedValsRed_2[i] = d_red_src[i];
    blendedValsBlue_1[i] = d_blue_src[i];
    blendedValsBlue_2[i] = d_blue_src[i];
    blendedValsGreen_1[i] = d_green_src[i];
    blendedValsGreen_2[i] = d_green_src[i];
  }

}

__device__
bool isMasked(uchar4 val) {
	return (val.x < 255 || val.y < 255 || val.z < 255);
}
__global__
void getMask(unsigned char * d_mask,
             uchar4 * d_sourceImg,
            const size_t numCols,
            const size_t numRows
             )
{
      const size_t srcSize = numCols * numRows;
      int main_x = threadIdx.x + blockDim.x * blockIdx.x;
      int main_y = threadIdx.y + blockDim.y * blockIdx.y;
      int main_id = main_x + main_y * numCols;


      if (main_id >= srcSize)
          return;
       d_mask[main_id] = (d_sourceImg[main_id].x + d_sourceImg[main_id].y + d_sourceImg[main_id].z < 3 * 255) ? 1 : 0;



}
__global__
void findBorderPixels(unsigned char * d_mask,
             unsigned char * d_borderPixels,
             unsigned char * d_strictInteriorPixels,
            const size_t numCols)
            {
              int main_x = threadIdx.x + blockDim.x * blockIdx.x;
              int main_y = threadIdx.y + blockDim.y * blockIdx.y;
              int main_id = main_x + main_y * numCols;
              int right = (main_x + 1) + main_y * numCols;
              int left = (main_x - 1) + main_y * numCols;
              int up = main_x + (main_y + 1) * numCols;
              int down = main_x + (main_y - 1) * numCols;


              //  __syncthreads();
                //now we need to check the four pixels north south east west to see if they are in the mask or Not
                if (d_mask[main_id] ==1)
                {
                  int isInside = 0;
                  if (d_mask[left] ==1)
                    isInside++;
                  if (d_mask[right] ==1)
                    isInside++;
                  if (d_mask[up] ==1)
                    isInside++;
                  if (d_mask[down] ==1)
                    isInside++;

                  if (isInside == 4)
                  {
                    d_strictInteriorPixels[main_id] = 1;
                  } else if (isInside > 0)
                  {
                    d_borderPixels[main_id] = 1;
                  }
                }
            }

__global__
void seperateRGB( uchar4 * d_sourceImg,
                  uchar4 * d_destImg,
                  unsigned char * red_src,
                  unsigned char* blue_src,
                  unsigned char* green_src,
                  unsigned char* red_dst,
                  unsigned char* blue_dst,
                  unsigned char* green_dst,
                  const size_t numCols,
                  const size_t numRows)
{
  int main_x = threadIdx.x + blockDim.x * blockIdx.x;
  int main_y = threadIdx.y + blockDim.y * blockIdx.y;
  int main_id = main_x + main_y * numCols;
  red_src[main_id]   = d_sourceImg[main_id].x;
  blue_src[main_id]  = d_sourceImg[main_id].y;
  green_src[main_id] = d_sourceImg[main_id].z;
  red_dst[main_id]   = d_destImg[main_id].x;
  blue_dst[main_id]  = d_destImg[main_id].y;
  green_dst[main_id] = d_destImg[main_id].z;

}
//jocobi kernel
__global__
void jacobi( unsigned char* const dstImg,
                       unsigned char* const strictInteriorPixels,
                       unsigned char* const borderPixels,
                      uint2 * interiorPixelList,
                       const size_t numColsSource,
                       float*  f,
                       float*  g,
                     float* const f_next,
                     int listSize
                    )

{



  int i = threadIdx.x + blockDim.x * blockIdx.x;
int zero = 0;
  unsigned int off = interiorPixelList[zero].x * numColsSource + interiorPixelList[zero].y;



  if (i < listSize)
  {
    float blendedSum = 0.f;
    float borderSum  = 0.f;

    uint2 coord = interiorPixelList[i];

    unsigned int offset = coord.x * numColsSource + coord.y;

    //process all 4 neighbor pixels
    //for each pixel if it is an interior pixel
    //then we add the previous f, otherwise if it is a
    //border pixel then we add the value of the destination
    //image at the border.  These border values are our boundary
    //conditions.
    if (strictInteriorPixels[offset - 1]) {
      blendedSum += f[offset - 1];
    }
    else {
      borderSum += dstImg[offset - 1];
    }

    if (strictInteriorPixels[offset + 1]) {
      blendedSum += f[offset + 1];
    }
    else {
      borderSum += dstImg[offset + 1];
    }

    if (strictInteriorPixels[offset - numColsSource]) {
      blendedSum += f[offset - numColsSource];
    }
    else {
      borderSum += dstImg[offset - numColsSource];
    }

    if (strictInteriorPixels[offset + numColsSource]) {
      blendedSum += f[offset + numColsSource];
    }
    else {
      borderSum += dstImg[offset + numColsSource];
    }

    float f_next_val = (blendedSum + borderSum + g[offset]) / 4.f;
__syncthreads();
    f_next[offset] = min(255.f, max(0.f, f_next_val)); //clip to [0, 255]
  }
}

//Performs one iteration of the solver
void compute_Iteration(const unsigned char* const dstImg,
                      const unsigned char* const strictInteriorPixels,
                      const unsigned char* const borderPixels,
                      const std::vector<uint2>& interiorPixelList,
                      const size_t numColsSource,
                      const float* const f,
                      const float* const g,
                      float* const f_next)
{
  unsigned int off = interiorPixelList[0].x * numColsSource + interiorPixelList[0].y;

  for (size_t i = 0; i < interiorPixelList.size(); ++i) {
    float blendedSum = 0.f;
    float borderSum  = 0.f;

    uint2 coord = interiorPixelList[i];

    unsigned int offset = coord.x * numColsSource + coord.y;

    //process all 4 neighbor pixels
    //for each pixel if it is an interior pixel
    //then we add the previous f, otherwise if it is a
    //border pixel then we add the value of the destination
    //image at the border.  These border values are our boundary
    //conditions.
    if (strictInteriorPixels[offset - 1]) {
      blendedSum += f[offset - 1];
    }
    else {
      borderSum += dstImg[offset - 1];
    }

    if (strictInteriorPixels[offset + 1]) {
      blendedSum += f[offset + 1];
    }
    else {
      borderSum += dstImg[offset + 1];
    }

    if (strictInteriorPixels[offset - numColsSource]) {
      blendedSum += f[offset - numColsSource];
    }
    else {
      borderSum += dstImg[offset - numColsSource];
    }

    if (strictInteriorPixels[offset + numColsSource]) {
      blendedSum += f[offset + numColsSource];
    }
    else {
      borderSum += dstImg[offset + numColsSource];
    }

    float f_next_val = (blendedSum + borderSum + g[offset]) / 4.f;

    f_next[offset] = min(255.f, max(0.f, f_next_val)); //clip to [0, 255]
  }

}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{



//  uchar4 * h_reference = new uchar4[numRowsSource*numColsSource];
size_t srcSize = numRowsSource * numColsSource;
//cuaMalloc all a mask array and aray of boarder and interior items
unsigned char * d_mask;
unsigned char * d_borderPixels;
unsigned char * d_strictInteriorPixels;

//some test host variables
unsigned char test_mask[srcSize];
unsigned char test_strinct_interior[srcSize];
unsigned char test_borderpixel[srcSize];

uchar4 * d_sourceImg;
uchar4 * d_destImg;
uchar4 * d_blendedImg;
checkCudaErrors(cudaMalloc(&d_mask, srcSize * sizeof(unsigned char)));
checkCudaErrors(cudaMalloc(&d_borderPixels, srcSize * sizeof(unsigned char)));
checkCudaErrors(cudaMalloc(&d_strictInteriorPixels, srcSize * sizeof(unsigned char)));
checkCudaErrors(cudaMemset(d_borderPixels, 0, srcSize * sizeof(unsigned char)));
checkCudaErrors(cudaMemset(d_strictInteriorPixels, 0, srcSize * sizeof(unsigned char)));
checkCudaErrors(cudaMalloc(&d_sourceImg, srcSize * sizeof(uchar4)));
checkCudaErrors(cudaMalloc(&d_destImg, srcSize * sizeof(uchar4)));
checkCudaErrors(cudaMalloc(&d_blendedImg, srcSize * sizeof(uchar4)));

checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, (sizeof(uchar4) * srcSize), cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, (sizeof(uchar4) * srcSize), cudaMemcpyHostToDevice));
int BLOCKS = 32;

dim3 block_dim(BLOCKS, BLOCKS);
dim3 thread_dim(ceil(numColsSource/block_dim.x)+1, ceil(numRowsSource/block_dim.y)+1);
getMask<<<block_dim, thread_dim>>>(d_mask,  d_sourceImg, numColsSource, numRowsSource);
cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
/*
TODO testing memcpy
*/
size_t cpySize = sizeof(unsigned char) * srcSize;
checkCudaErrors(cudaMemcpy(&test_mask, d_mask, cpySize, cudaMemcpyDeviceToHost));

findBorderPixels<<<block_dim, thread_dim>>>(d_mask, d_borderPixels, d_strictInteriorPixels, numColsSource);
cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
checkCudaErrors(cudaMemcpy(&test_borderpixel, d_borderPixels, cpySize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(&test_strinct_interior, d_strictInteriorPixels, cpySize, cudaMemcpyDeviceToHost));



//this whole bit is still needed for a later part of the serial implemnintaion.
std::vector<uint2> interiorPixelList;




//the source region in the homework isn't near an image boundary, so we can
//simplify the conditionals a little...
for (size_t r = 1; r < numRowsSource - 1; ++r) {
  for (size_t c = 1; c < numColsSource - 1; ++c) {
    if (test_mask[r * numColsSource + c]) {
      if (test_mask[(r -1) * numColsSource + c] && test_mask[(r + 1) * numColsSource + c] &&
          test_mask[r * numColsSource + c - 1] && test_mask[r * numColsSource + c + 1]) {
        interiorPixelList.push_back(make_uint2(r, c));
  }
}

}}
int listSize = interiorPixelList.size();
 uint2 transferList[listSize];
for (size_t i = 0; i < interiorPixelList.size(); ++i) {
  transferList[i] = interiorPixelList[i];
}


uint2 * d_interiorPixelList;
checkCudaErrors(cudaMalloc(&d_interiorPixelList, (listSize * sizeof(uint2))));
checkCudaErrors(cudaMemcpy(d_interiorPixelList, transferList, (listSize * sizeof(uint2)), cudaMemcpyHostToDevice));

//serial get mask for loop
//split the source and destination images into their respective
//channels
unsigned char t_red_src[srcSize];
unsigned char t_blue_src[srcSize];
unsigned char t_green_src[srcSize];
unsigned char t_red_dst[srcSize];
unsigned char t_blue_dst[srcSize];
unsigned char t_green_dst[srcSize];

unsigned char* d_red_src;
unsigned char* d_blue_src;
unsigned char* d_green_src;
unsigned char* d_red_dst;
unsigned char* d_blue_dst;
unsigned char* d_green_dst;
checkCudaErrors(cudaMalloc(&d_red_src, srcSize * sizeof(unsigned char)));
checkCudaErrors(cudaMalloc(&d_blue_src, srcSize * sizeof(unsigned char)));
checkCudaErrors(cudaMalloc(&d_green_src, srcSize * sizeof(unsigned char)));
checkCudaErrors(cudaMalloc(&d_red_dst, srcSize * sizeof(unsigned char)));
checkCudaErrors(cudaMalloc(&d_blue_dst, srcSize * sizeof(unsigned char)));
checkCudaErrors(cudaMalloc(&d_green_dst, srcSize * sizeof(unsigned char)));

seperateRGB<<<block_dim, thread_dim>>>(d_sourceImg, d_destImg, d_red_src, d_blue_src, d_green_src, d_red_dst, d_blue_dst, d_green_dst, numColsSource, numRowsSource);
cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


checkCudaErrors(cudaMemcpy(&t_red_src, d_red_src, cpySize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(&t_blue_src, d_blue_src, cpySize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(&t_green_src, d_green_src, cpySize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(&t_red_dst, d_red_dst, cpySize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(&t_blue_dst, d_blue_dst, cpySize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(&t_green_dst, d_green_dst, cpySize, cudaMemcpyDeviceToHost));
//next we'll precompute the g term - it never changes, no need to recompute every iteration
float *g_red   = new float[srcSize];
float *g_blue  = new float[srcSize];
float *g_green = new float[srcSize];

memset(g_red,   0, srcSize * sizeof(float));
memset(g_blue,  0, srcSize * sizeof(float));
memset(g_green, 0, srcSize * sizeof(float));

// checkCudaErrors(cudaMalloc(&t_g_red, srcSize * sizeof(unsigned char)));
// checkCudaErrors(cudaMalloc(&t_g_blue, srcSize * sizeof(unsigned char)));
// checkCudaErrors(cudaMalloc(&t_g_green, srcSize * sizeof(unsigned char)));
//
// checkCudaErrors(cudaMemcpy(&t_g_red, g_red, cpySize, cudaMemcpyHostToDevice));
// checkCudaErrors(cudaMemcpy(&t_g_blue, g_blue, cpySize, cudaMemcpyHostToDevice));
// checkCudaErrors(cudaMemcpy(&t_g_green, g_green, cpySize, cudaMemcpyHostToDevice));

// comp_G<<<block_dim, thread_dim>>>(d_red_src, t_g_red, numColsSource, interiorPixelList.size(), interiorPixelList, numRowsSource);
// cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


compute_G(t_red_src,   g_red,   numColsSource, interiorPixelList);
compute_G(t_blue_src,  g_blue,  numColsSource, interiorPixelList);
compute_G(t_green_src, g_green, numColsSource, interiorPixelList);

//for each color channel we'll need two buffers and we'll ping-pong between them
float *blendedValsRed_1 = new float[srcSize];
float *blendedValsRed_2 = new float[srcSize];

float *blendedValsBlue_1 = new float[srcSize];
float *blendedValsBlue_2 = new float[srcSize];

float *blendedValsGreen_1 = new float[srcSize];
float *blendedValsGreen_2 = new float[srcSize];
//test stuff
float *t_blendedValsRed_1 = new float[srcSize];
float *t_blendedValsRed_2 = new float[srcSize];

float *t_blendedValsBlue_1 = new float[srcSize];
float *t_blendedValsBlue_2 = new float[srcSize];

float *t_blendedValsGreen_1 = new float[srcSize];
float *t_blendedValsGreen_2 = new float[srcSize];


float *d_blendedValsRed_1;
float *d_blendedValsRed_2;

float *d_blendedValsBlue_1;
float *d_blendedValsBlue_2;

float *d_blendedValsGreen_1;
float *d_blendedValsGreen_2;

float *d_blendedValsTemp;
float *td_blendedRed_1;
float *td_blendedRed_2;
size_t floatSize = sizeof(float)*srcSize;



checkCudaErrors(cudaMalloc(&d_blendedValsRed_1, floatSize));
checkCudaErrors(cudaMalloc(&d_blendedValsRed_2, floatSize));
checkCudaErrors(cudaMalloc(&d_blendedValsBlue_1, floatSize));
checkCudaErrors(cudaMalloc(&d_blendedValsBlue_2, floatSize));
checkCudaErrors(cudaMalloc(&d_blendedValsGreen_1, floatSize));
checkCudaErrors(cudaMalloc(&d_blendedValsGreen_2, floatSize));
checkCudaErrors(cudaMalloc(&d_blendedValsTemp, floatSize));

checkCudaErrors(cudaMalloc(&td_blendedRed_1, floatSize));
checkCudaErrors(cudaMalloc(&td_blendedRed_2, floatSize));

addToBlended<<<block_dim, thread_dim>>>(d_blendedValsRed_1,
                                       d_blendedValsRed_2,
                                       d_blendedValsBlue_1,
                                       d_blendedValsBlue_2,
                                       d_blendedValsGreen_1,
                                       d_blendedValsGreen_2,
                                       d_red_src,
                                       d_blue_src,
                                       d_green_src,
                                       numColsSource,
                                       numRowsSource);

checkCudaErrors(cudaMemcpy(t_blendedValsRed_1, d_blendedValsRed_1, floatSize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(t_blendedValsRed_2, d_blendedValsRed_2, floatSize, cudaMemcpyDeviceToHost));

checkCudaErrors(cudaMemcpy(t_blendedValsBlue_1, d_blendedValsBlue_1,  floatSize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(t_blendedValsBlue_2, d_blendedValsBlue_2, floatSize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(t_blendedValsGreen_1, d_blendedValsGreen_1,  floatSize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(t_blendedValsGreen_2, d_blendedValsGreen_2, floatSize, cudaMemcpyDeviceToHost));

float *d_g_red;
float *d_g_blue;
float *d_g_green;
checkCudaErrors(cudaMalloc(&d_g_red, floatSize));
checkCudaErrors(cudaMalloc(&d_g_blue, floatSize));
checkCudaErrors(cudaMalloc(&d_g_green, floatSize));
checkCudaErrors(cudaMemcpy(d_g_red, g_red, floatSize, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(d_g_blue, g_blue, floatSize, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(d_g_green, g_green, floatSize, cudaMemcpyHostToDevice));


//Perform the solve on each color channel
const size_t numIterations = 800;
for (size_t i = 0; i < numIterations; ++i) {
  compute_Iteration(t_red_dst, test_strinct_interior, test_borderpixel,
                   interiorPixelList, numColsSource, t_blendedValsRed_1, g_red,
                   t_blendedValsRed_2);
  std::swap(t_blendedValsRed_1, t_blendedValsRed_2);


//not sure why this was three loops, clerity i suppose
  compute_Iteration(t_blue_dst, test_strinct_interior, test_borderpixel,
                   interiorPixelList, numColsSource, t_blendedValsBlue_1, g_blue,
                   t_blendedValsBlue_2);
  std::swap(t_blendedValsBlue_1, t_blendedValsBlue_2);




  compute_Iteration(t_green_dst, test_strinct_interior, test_borderpixel,
                   interiorPixelList, numColsSource, t_blendedValsGreen_1, g_green,
                   t_blendedValsGreen_2);
  std::swap(t_blendedValsGreen_1, t_blendedValsGreen_2);
}
//my Iterations
int eightHun = 800;
dim3 jacobiBlock(28);
dim3 jacobiThread(ceil(listSize/28)+1);
std::cout << listSize/28 << " threads: list -> " << listSize << std::endl;
for (int i = 0; i < eightHun; i++)
{
  //kernel launch for red channel
  jacobi<<<jacobiBlock, jacobiThread>>>(d_red_dst,
                                    d_strictInteriorPixels,
                                    d_borderPixels,
                                   d_interiorPixelList,
                                   numColsSource,
                                   d_blendedValsRed_1,
                                   d_g_red,
                                   d_blendedValsRed_2,
                                   listSize);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  std::swap(d_blendedValsRed_1, d_blendedValsRed_2);

  //kernel launch for red channel
  jacobi<<<jacobiBlock, jacobiThread>>>(d_blue_dst,
                                    d_strictInteriorPixels,
                                    d_borderPixels,
                                   d_interiorPixelList,
                                   numColsSource,
                                   d_blendedValsBlue_1,
                                   d_g_blue,
                                   d_blendedValsBlue_2,
                                   listSize);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  std::swap(d_blendedValsBlue_1, d_blendedValsBlue_2);

    // checkCudaErrors(cudaMemcpy(d_blendedValsTemp, d_blendedValsRed_1, (srcSize * sizeof(float)), cudaMemcpyDeviceToDevice));
    // checkCudaErrors(cudaMemcpy(d_blendedValsRed_1, d_blendedValsRed_2, (srcSize * sizeof(float)), cudaMemcpyDeviceToDevice));
    // checkCudaErrors(cudaMemcpy(d_blendedValsRed_2, d_blendedValsTemp, (srcSize * sizeof(float)), cudaMemcpyDeviceToDevice));
    jacobi<<<jacobiBlock, jacobiThread>>>(d_green_dst,
                                      d_strictInteriorPixels,
                                      d_borderPixels,
                                     d_interiorPixelList,
                                     numColsSource,
                                     d_blendedValsGreen_1,
                                     d_g_green,
                                     d_blendedValsGreen_2,
                                     listSize);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    std::swap(d_blendedValsGreen_1, d_blendedValsGreen_2);



}

//copy stuff over and perform the final swap not going to save anyting but kinda clever
checkCudaErrors(cudaMemcpy(t_blendedValsRed_1, d_blendedValsRed_2, floatSize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(t_blendedValsRed_2, d_blendedValsRed_1, floatSize, cudaMemcpyDeviceToHost));

checkCudaErrors(cudaMemcpy(t_blendedValsBlue_1, d_blendedValsBlue_2, floatSize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(t_blendedValsBlue_2, d_blendedValsBlue_1, floatSize, cudaMemcpyDeviceToHost));

checkCudaErrors(cudaMemcpy(t_blendedValsGreen_1, d_blendedValsGreen_2, floatSize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(t_blendedValsGreen_2, d_blendedValsGreen_1, floatSize, cudaMemcpyDeviceToHost));

// std::swap(t_blendedValsRed_1,   t_blendedValsRed_2);   //put output into _2
// std::swap(t_blendedValsBlue_1,  t_blendedValsBlue_2);  //put output into _2
// std::swap(t_blendedValsGreen_1, t_blendedValsGreen_2); //put output into _2

//copy the destination image to the output
memcpy(h_blendedImg, h_destImg, sizeof(uchar4) * srcSize);

//copy computed values for the interior into the output
for (size_t i = 0; i < interiorPixelList.size(); ++i) {
  uint2 coord = interiorPixelList[i];

  unsigned int offset = coord.x * numColsSource + coord.y;

  h_blendedImg[offset].x = t_blendedValsRed_2[offset];
  h_blendedImg[offset].y = t_blendedValsBlue_2[offset];
  h_blendedImg[offset].z = t_blendedValsGreen_2[offset];
}

//wow, we allocated a lot of memory!
delete[] blendedValsRed_1;
delete[] blendedValsRed_2;
delete[] blendedValsBlue_1;
delete[] blendedValsBlue_2;
delete[] blendedValsGreen_1;
delete[] blendedValsGreen_2;
delete[] g_red;
delete[] g_blue;
delete[] g_green;
// delete[] red_src;
// delete[] red_dst;
// delete[] blue_src;
// delete[] blue_dst;
// delete[] green_src;
// delete[] green_dst;


//checkResultsEps((unsigned char *)h_reference, (unsigned char *)h_blendedImg, 4 * numRowsSource * numColsSource, 2, .01);
//delete[] h_reference;
  /* To Recap here are the steps you need to implement

     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */



  /* The reference calculation is provided below, feel free to use it
     for debugging purposes.
   */

  /*
    uchar4* h_reference = new uchar4[srcSize];
    reference_calc(h_sourceImg, numRowsSource, numColsSource,
                   h_destImg, h_reference);

    checkResultsEps((unsigned char *)h_reference, (unsigned char *)h_blendedImg, 4 * srcSize, 2, .01);
    delete[] h_reference; */
    //Udacity HW 6
//Poisson Blending Reference Calculation




}
