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

__global__
void getMask(unsigned char * d_mask,
             unsigned char * d_borderPixels,
             unsigned char * d_strictInteriorPixels,
             uchar4 * d_sourceImg
             )
{
      int main_id = threadIdx.x + blockDim.x * blockIdx.x;
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
unsigned char* mask = new unsigned char[srcSize];

//cudaMalloc all a mask array and aray of boarder and interior items
unsigned char * d_mask;
unsigned char * d_borderPixels;
unsigned char * d_strictInteriorPixels;
const uchar4 * d_sourceImg;
uchar4 * d_destImg;
uchar4 * d_blendedImg;
checkCudaErrors(cudaMalloc(&d_mask, srcSize * sizeof(unsigned char)));
checkCudaErrors(cudaMalloc(&d_borderPixels, srcSize * sizeof(unsigned char)));
checkCudaErrors(cudaMalloc(&d_strictInteriorPixels, srcSize * sizeof(unsigned char)));
checkCudaErrors(cudaMalloc(&d_sourceImg, srcSize * sizeof(uchar4)));
checkCudaErrors(cudaMalloc(&d_destImg, srcSize * sizeof(uchar4)));
checkCudaErrors(cudaMalloc(&d_blendedImg, srcSize * sizeof(uchar4)));

checkCudaErrors(cudaMemcpy(&d_sourceImg, h_sourceImg, sizeof(unsigned char)* srcSize, cudaMemcpyHostToDevice));


for (int i = 0; i < srcSize; ++i) {
  mask[i] = (h_sourceImg[i].x + h_sourceImg[i].y + h_sourceImg[i].z < 3 * 255) ? 1 : 0;
}

//next compute strictly interior pixels and border pixels
unsigned char *borderPixels = new unsigned char[srcSize];
unsigned char *strictInteriorPixels = new unsigned char[srcSize];

std::vector<uint2> interiorPixelList;

//the source region in the homework isn't near an image boundary, so we can
//simplify the conditionals a little...
for (size_t r = 1; r < numRowsSource - 1; ++r) {
  for (size_t c = 1; c < numColsSource - 1; ++c) {
    if (mask[r * numColsSource + c]) {
      if (mask[(r -1) * numColsSource + c] && mask[(r + 1) * numColsSource + c] &&
          mask[r * numColsSource + c - 1] && mask[r * numColsSource + c + 1]) {
        strictInteriorPixels[r * numColsSource + c] = 1;
        borderPixels[r * numColsSource + c] = 0;
        interiorPixelList.push_back(make_uint2(r, c));
      }
      else {
        strictInteriorPixels[r * numColsSource + c] = 0;
        borderPixels[r * numColsSource + c] = 1;
      }
    }
    else {
        strictInteriorPixels[r * numColsSource + c] = 0;
        borderPixels[r * numColsSource + c] = 0;

    }
  }
}

//split the source and destination images into their respective
//channels
unsigned char* red_src   = new unsigned char[srcSize];
unsigned char* blue_src  = new unsigned char[srcSize];
unsigned char* green_src = new unsigned char[srcSize];

for (int i = 0; i < srcSize; ++i) {
  red_src[i]   = h_sourceImg[i].x;
  blue_src[i]  = h_sourceImg[i].y;
  green_src[i] = h_sourceImg[i].z;
}

unsigned char* red_dst   = new unsigned char[srcSize];
unsigned char* blue_dst  = new unsigned char[srcSize];
unsigned char* green_dst = new unsigned char[srcSize];

for (int i = 0; i < srcSize; ++i) {
  red_dst[i]   = h_destImg[i].x;
  blue_dst[i]  = h_destImg[i].y;
  green_dst[i] = h_destImg[i].z;
}

//next we'll precompute the g term - it never changes, no need to recompute every iteration
float *g_red   = new float[srcSize];
float *g_blue  = new float[srcSize];
float *g_green = new float[srcSize];

memset(g_red,   0, srcSize * sizeof(float));
memset(g_blue,  0, srcSize * sizeof(float));
memset(g_green, 0, srcSize * sizeof(float));

compute_G(red_src,   g_red,   numColsSource, interiorPixelList);
compute_G(blue_src,  g_blue,  numColsSource, interiorPixelList);
compute_G(green_src, g_green, numColsSource, interiorPixelList);

//for each color channel we'll need two buffers and we'll ping-pong between them
float *blendedValsRed_1 = new float[srcSize];
float *blendedValsRed_2 = new float[srcSize];

float *blendedValsBlue_1 = new float[srcSize];
float *blendedValsBlue_2 = new float[srcSize];

float *blendedValsGreen_1 = new float[srcSize];
float *blendedValsGreen_2 = new float[srcSize];

//IC is the source image, copy over
for (size_t i = 0; i < srcSize; ++i) {
  blendedValsRed_1[i] = red_src[i];
  blendedValsRed_2[i] = red_src[i];
  blendedValsBlue_1[i] = blue_src[i];
  blendedValsBlue_2[i] = blue_src[i];
  blendedValsGreen_1[i] = green_src[i];
  blendedValsGreen_2[i] = green_src[i];
}

//Perform the solve on each color channel
const size_t numIterations = 800;
for (size_t i = 0; i < numIterations; ++i) {
  compute_Iteration(red_dst, strictInteriorPixels, borderPixels,
                   interiorPixelList, numColsSource, blendedValsRed_1, g_red,
                   blendedValsRed_2);

  std::swap(blendedValsRed_1, blendedValsRed_2);
}

for (size_t i = 0; i < numIterations; ++i) {
  compute_Iteration(blue_dst, strictInteriorPixels, borderPixels,
                   interiorPixelList, numColsSource, blendedValsBlue_1, g_blue,
                   blendedValsBlue_2);

  std::swap(blendedValsBlue_1, blendedValsBlue_2);
}

for (size_t i = 0; i < numIterations; ++i) {
  compute_Iteration(green_dst, strictInteriorPixels, borderPixels,
                   interiorPixelList, numColsSource, blendedValsGreen_1, g_green,
                   blendedValsGreen_2);

  std::swap(blendedValsGreen_1, blendedValsGreen_2);
}
std::swap(blendedValsRed_1,   blendedValsRed_2);   //put output into _2
std::swap(blendedValsBlue_1,  blendedValsBlue_2);  //put output into _2
std::swap(blendedValsGreen_1, blendedValsGreen_2); //put output into _2

//copy the destination image to the output
memcpy(h_blendedImg, h_destImg, sizeof(uchar4) * srcSize);

//copy computed values for the interior into the output
for (size_t i = 0; i < interiorPixelList.size(); ++i) {
  uint2 coord = interiorPixelList[i];

  unsigned int offset = coord.x * numColsSource + coord.y;

  h_blendedImg[offset].x = blendedValsRed_2[offset];
  h_blendedImg[offset].y = blendedValsBlue_2[offset];
  h_blendedImg[offset].z = blendedValsGreen_2[offset];
}

//wow, we allocated a lot of memory!
delete[] mask;
delete[] blendedValsRed_1;
delete[] blendedValsRed_2;
delete[] blendedValsBlue_1;
delete[] blendedValsBlue_2;
delete[] blendedValsGreen_1;
delete[] blendedValsGreen_2;
delete[] g_red;
delete[] g_blue;
delete[] g_green;
delete[] red_src;
delete[] red_dst;
delete[] blue_src;
delete[] blue_dst;
delete[] green_src;
delete[] green_dst;
delete[] borderPixels;
delete[] strictInteriorPixels;
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
