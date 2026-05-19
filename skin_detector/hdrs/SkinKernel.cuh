#pragma once
#ifndef SKINKERNEL_CUH
#define SKINKERNEL_CUH

#ifndef uchar
    typedef unsigned char uchar;
#endif

__global__ void getSkinMap(uchar* __restrict__ image, int cols, int rows,
                           const float* __restrict__ inverseCovariance,
                           const float* __restrict__ mean,
                           const float* __restrict__ threshold);

__global__ void getSkinMask(const uchar* __restrict__ image, uchar* __restrict__ output,
                            int cols, int rows,
                            const float* __restrict__ inverseCovariance,
                            const float* __restrict__ mean,
                            const float* __restrict__ threshold);

#endif
