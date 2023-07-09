#include "SkinDetector.h"

#ifndef uchar
	typedef unsigned char uchar;
#endif

SkinDetector::SkinDetector()
{
}

/**
 * @brief Construct a new Skin Detector:: Skin Detector object based on Normalized RGB clustering, pixels are asigned to Skin or not skin.
 *        Assigment uses the Mahalanobis distance and a threshold.
 * 
 * @param mInverseCovDev : Inverse covariance matrix (we use the inverse to speed up calculations)
 * @param mMean : mean values for the normalized R and G components
 * @param mThreshold : Threshold value for assignment
 * @param mCols : Number of Colons on the images to process
 * @param mRows : Number of Rows on the images to process.
 */
SkinDetector::SkinDetector(float* mInverseCovDev, float* mMean, float mThreshold, int mCols, int mRows)
{
    meanDev = NULL;
    inverseCovDev = NULL;
    threshDev = NULL;
    devInput = NULL;
    rows = mRows;
    cols = mCols;
    channels = 3;

    float threshold[1]={mThreshold};

    // Hardcoded sizes... no bueno!
    // TODO: GET SIZES FROM ARRAYS
    cudaMallocManaged(&meanDev, 2*sizeof(float));
    cudaMallocManaged(&inverseCovDev, 4*sizeof(float));
    cudaMallocManaged(&threshDev, 1*sizeof(float));

    // Copy algorithm arrays to GPU
    cudaMemcpy(meanDev, mMean, 2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inverseCovDev, mInverseCovDev, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(threshDev, threshold, 1*sizeof(float), cudaMemcpyHostToDevice);

    // Allocate Images
    cudaMallocManaged(&devInput, rows*cols*channels);
    cudaMallocManaged(&devOutput, rows*cols);
}

SkinDetector::~SkinDetector()
{
    // Clean the GPU Memory
    cudaFree(devInput);
    cudaFree(devOutput);
    cudaFree(meanDev);
    cudaFree(inverseCovDev);
    cudaFree(threshDev);
}

/**
 * @brief Calcualtes the skin mask and applies it to the original image.
 *        By default the image is overwritten by the algorithm, non skin pixels are zeroed.
 *        If you would like to not modify the image, or would like to refine/use the mask, please use skinMask function.
 * 
 * @param image : Pointer to image data.
 */
void SkinDetector::skinMap(uchar* image)
{
    // Copy image data to GPU
    cudaMemcpy(devInput, image, rows*cols*channels, cudaMemcpyHostToDevice);
    // Prepare and launch the kernel
    dim3 gridImage(cols, rows);
    getSkinMap << <gridImage, 1 >> >(devInput, inverseCovDev, meanDev, threshDev);
    
    // Copy back the image from GPU
    cudaMemcpy(image, devInput, rows*cols*channels, cudaMemcpyDeviceToHost);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
}

/**
 * @brief Calcualtes the skin mask and return a binary image with 0 values for non skin regions and 255 for skin regions.
 * 
 * @param image Pointer to image data.
 * @param output Pointer to output image data.
 */
void SkinDetector::skinMask(uchar* image, uchar* output)
{
    // Copy image data to GPU
    cudaMemcpy(devInput, image, rows*cols*channels, cudaMemcpyHostToDevice);
    // Make mask a zero filled array
    cudaMemset(devOutput, 0, rows*cols);

    // Prepare and launch the kernel
    dim3 gridImage(cols, rows);
    getSkinMask << <gridImage, 1 >> >(devInput, devOutput, inverseCovDev, meanDev, threshDev);
    
    // Copy back the image from GPU
    cudaMemcpy(output, devOutput, rows*cols, cudaMemcpyDeviceToHost);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
}
