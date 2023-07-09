#ifndef SKINDETECTOR_H
#define SKINDETECTOR_H

#pragma once

#include <iostream>
#include <math.h>
#include "SkinKernel.cuh"


#ifndef uchar
	typedef unsigned char uchar;
#endif

class SkinDetector
{
public:
    SkinDetector();
    SkinDetector(float* mInverseCovDev, float* mMean, float mThreshold, int mCols, int mRows);
    ~SkinDetector();

    void skinMap(uchar* image);
    void skinMask(uchar* image, uchar* output);

private:
    // Algorithm pointers
    float* meanDev;
    float* inverseCovDev;
    float* threshDev;

    // Image data pointers
    uchar* devInput;
    uchar* devOutput;

    // Algo sizes
    int channels;
    int rows;
    int cols;

};

#endif