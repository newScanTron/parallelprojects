//Udacity HW 6
//Poisson Blending

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
                     int listSize)
{

  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int zero = 0;

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
    f_next[offset] = min(255.f, max(0.f, f_next_val)); //clip to [0, 255]
  }
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
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
//cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

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
//cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

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

size_t floatSize = sizeof(float)*srcSize;

float *d_g_red;
float *d_g_blue;
float *d_g_green;
checkCudaErrors(cudaMalloc(&d_g_red, floatSize));
checkCudaErrors(cudaMalloc(&d_g_blue, floatSize));
checkCudaErrors(cudaMalloc(&d_g_green, floatSize));
checkCudaErrors(cudaMemcpy(d_g_red, g_red, floatSize, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(d_g_blue, g_blue, floatSize, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(d_g_green, g_green, floatSize, cudaMemcpyHostToDevice));
int blockSize = 28;
dim3 jacobiBlock(blockSize);
dim3 jacobiThread(ceil(listSize/blockSize)+1);
// comp_G<<<block_dim, thread_dim>>>(d_red_src, d_g_red, numColsSource, listSize, transferList);
//  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


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

checkCudaErrors(cudaMalloc(&d_blendedValsRed_1, floatSize));
checkCudaErrors(cudaMalloc(&d_blendedValsRed_2, floatSize));
checkCudaErrors(cudaMalloc(&d_blendedValsBlue_1, floatSize));
checkCudaErrors(cudaMalloc(&d_blendedValsBlue_2, floatSize));
checkCudaErrors(cudaMalloc(&d_blendedValsGreen_1, floatSize));
checkCudaErrors(cudaMalloc(&d_blendedValsGreen_2, floatSize));

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
//cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

checkCudaErrors(cudaMemcpy(t_blendedValsRed_1, d_blendedValsRed_1, floatSize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(t_blendedValsRed_2, d_blendedValsRed_2, floatSize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(t_blendedValsBlue_1, d_blendedValsBlue_1,  floatSize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(t_blendedValsBlue_2, d_blendedValsBlue_2, floatSize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(t_blendedValsGreen_1, d_blendedValsGreen_1,  floatSize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(t_blendedValsGreen_2, d_blendedValsGreen_2, floatSize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(d_g_red, g_red, floatSize, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(d_g_blue, g_blue, floatSize, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(d_g_green, g_green, floatSize, cudaMemcpyHostToDevice));

int eightHun = 800;

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
//    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
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
  //  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  std::swap(d_blendedValsBlue_1, d_blendedValsBlue_2);

    jacobi<<<jacobiBlock, jacobiThread>>>(d_green_dst,
                                      d_strictInteriorPixels,
                                      d_borderPixels,
                                     d_interiorPixelList,
                                     numColsSource,
                                     d_blendedValsGreen_1,
                                     d_g_green,
                                     d_blendedValsGreen_2,
                                     listSize);
  //    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    std::swap(d_blendedValsGreen_1, d_blendedValsGreen_2);
}

//copy stuff over and perform the final swap not going to save anyting but kinda clever
checkCudaErrors(cudaMemcpy(t_blendedValsRed_1, d_blendedValsRed_2, floatSize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(t_blendedValsRed_2, d_blendedValsRed_1, floatSize, cudaMemcpyDeviceToHost));

checkCudaErrors(cudaMemcpy(t_blendedValsBlue_1, d_blendedValsBlue_2, floatSize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(t_blendedValsBlue_2, d_blendedValsBlue_1, floatSize, cudaMemcpyDeviceToHost));

checkCudaErrors(cudaMemcpy(t_blendedValsGreen_1, d_blendedValsGreen_2, floatSize, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(t_blendedValsGreen_2, d_blendedValsGreen_1, floatSize, cudaMemcpyDeviceToHost));

memcpy(h_blendedImg, h_destImg, sizeof(uchar4) * srcSize);

checkCudaErrors(cudaMemcpy(d_blendedImg, d_destImg, sizeof(uchar4) * srcSize , cudaMemcpyDeviceToDevice));
//copy computed values for the interior into the output

for (size_t i = 0; i < interiorPixelList.size(); ++i) {
  uint2 coord = interiorPixelList[i];

  unsigned int offset = coord.x * numColsSource + coord.y;

  h_blendedImg[offset].x = t_blendedValsRed_2[offset];
  h_blendedImg[offset].y = t_blendedValsBlue_2[offset];
  h_blendedImg[offset].z = t_blendedValsGreen_2[offset];
}

delete[] g_red;
delete[] g_blue;
delete[] g_green;

//test stuff
delete[] t_blendedValsRed_1;
delete[] t_blendedValsRed_2;

delete[] t_blendedValsBlue_1 ;
delete[] t_blendedValsBlue_2 ;

delete[] t_blendedValsGreen_1 ;
delete[] t_blendedValsGreen_2 ;

}
