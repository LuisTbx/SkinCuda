#ifndef uchar
	typedef unsigned char uchar;
#endif


#include <math.h>
#include "SkinKernel.cuh"

__global__ void getSkinMap(uchar* image, float* inverseCovariance, float* mean, float* threshold)
{
  int x = blockIdx.x;
	int y = blockIdx.y;
	int idx = (x + y * gridDim.x) * 3;
  
  float B = (float)image[idx]; 
  float G = (float)image[idx+1];
  float R = (float)image[idx+2];

  // float nb = B /(B+G+R);
  float ng = (G/(B+G+R)) - mean[1];
  float nr = (R/(B+G+R)) - mean[0];

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

__global__ void getSkinMask(uchar* image, uchar* output, float* inverseCovariance, float* mean, float* threshold)
{
  int x = blockIdx.x;
	int y = blockIdx.y;
	int idx = (x + y * gridDim.x) * 3;
  
  float B = (float)image[idx]; 
  float G = (float)image[idx+1];
  float R = (float)image[idx+2];

  // float nb = B /(B+G+R);
  float ng = (G/(B+G+R)) - mean[1];
  float nr = (R/(B+G+R)) - mean[0];

  float gate = exp( -0.5f *  (( (nr*nr)*inverseCovariance[0])   
                                +(2*inverseCovariance[1]*(nr*ng)) 
                                +((ng*ng)*inverseCovariance[3]))
                   );

  int maskIdx = (x + y * gridDim.x);
  if (gate >= threshold[0]){
      output[maskIdx] = 255;
  }
}