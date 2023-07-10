#ifndef uchar
	typedef unsigned char uchar;
#endif

#include <math.h>
#include "SkinKernel.cuh"
#include <stdio.h>

__global__ void getSkinMap(uchar* image, int cols, int rows, float* inverseCovariance, float* mean, float* threshold)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if( x >= cols || y >= rows )
		return;

	int64_t idx = (x + y * cols) * 3;

  float B, G, R, ng, nr;
  B = (float)image[idx]; 
  G = (float)image[idx+1];
  R = (float)image[idx+2];

  // float nb = B /(B+G+R);
  ng = (G/(B+G+R)) - mean[1];
  nr = (R/(B+G+R)) - mean[0];

  float gate = exp( -0.5f *  (( (nr*nr)*inverseCovariance[0])   
                                +(2*inverseCovariance[1]*(nr*ng)) 
                                +((ng*ng)*inverseCovariance[3]))
                   );

  if (gate <= threshold[0]){
      image[idx] = 0;
      image[idx+1] = 0;
      image[idx+2] = 0;
  }
}

__global__ void getSkinMask(uchar* image, uchar* output, int cols, int rows, float* inverseCovariance, float* mean, float* threshold)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if( x >= cols || y >= rows )
		return;

	int64_t idx = (x + y * cols) * 3;

  float B, G, R, ng, nr;
  B = (float)image[idx]; 
  G = (float)image[idx+1];
  R = (float)image[idx+2];

  // float nb = B /(B+G+R);
  ng = (G/(B+G+R)) - mean[1];
  nr = (R/(B+G+R)) - mean[0];

  float gate = exp( -0.5f *  (( (nr*nr)*inverseCovariance[0])   
                                +(2*inverseCovariance[1]*(nr*ng)) 
                                +((ng*ng)*inverseCovariance[3]))
                   );

  int maskIdx = (x + y * cols);
  if (gate >= threshold[0]){
      output[maskIdx] = 255;
  }
}