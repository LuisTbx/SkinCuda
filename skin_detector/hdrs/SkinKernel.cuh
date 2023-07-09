#ifndef SKINKERNEL_CUH
#define SKINKERNEL_CUH

#ifndef uchar
	typedef unsigned char uchar;
#endif

#pragma once

#include <math.h>


__global__ void getSkinMap(uchar* image, float* inverseCovariance, float* mean, float* threshold);

__global__ void getSkinMask(uchar* image, uchar* output, float* inverseCovariance, float* mean, float* threshold);


#endif