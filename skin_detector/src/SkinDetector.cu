#include "SkinDetector.h"

#ifndef uchar
    typedef unsigned char uchar;
#endif

SkinDetector::SkinDetector() {}

/**
 * @brief Construct a SkinDetector.
 *
 * Uploads the algorithm parameters to dedicated device memory and allocates
 * a double-buffered pair of input / output device images so that two CUDA
 * streams can operate concurrently without sharing buffers.
 */
SkinDetector::SkinDetector(float* mInverseCovDev, float* mMean, float mThreshold,
                           int mCols, int mRows)
{
    rows     = mRows;
    cols     = mCols;
    channels = 3;

    // 16×16 = 256 threads/block.  Compared with 32×32 (1024 threads/block),
    // the smaller block allows the scheduler to place more blocks per SM,
    // improving latency hiding on Blackwell (sm_120) and Ampere/Ada GPUs.
    blockDim = dim3(16, 16);
    gridDim  = dim3((mCols + blockDim.x - 1) / blockDim.x,
                    (mRows + blockDim.y - 1) / blockDim.y);

    // Upload small algorithm parameters to ordinary device memory.
    // They are read by every thread but the values fit in L1 / constant cache
    // after the first warp accesses them, so no special allocation is needed.
    cudaMalloc(&meanDev,       2 * sizeof(float));
    cudaMalloc(&inverseCovDev, 4 * sizeof(float));
    cudaMalloc(&threshDev,     1 * sizeof(float));

    cudaMemcpy(meanDev,       mMean,          2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inverseCovDev, mInverseCovDev, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(threshDev,     &mThreshold,    1 * sizeof(float), cudaMemcpyHostToDevice);

    // Double-buffered device images: slot 0 is used by synchronous calls;
    // slots 0 and 1 are alternated by the async video pipeline.
    const size_t inSz  = (size_t)rows * cols * channels;
    const size_t outSz = (size_t)rows * cols;
    for (int i = 0; i < 2; i++) {
        cudaMalloc(&devInput[i],  inSz);
        cudaMalloc(&devOutput[i], outSz);
    }
}

SkinDetector::~SkinDetector()
{
    for (int i = 0; i < 2; i++) {
        cudaFree(devInput[i]);
        cudaFree(devOutput[i]);
    }
    cudaFree(meanDev);
    cudaFree(inverseCovDev);
    cudaFree(threshDev);
}

// ── Synchronous API ───────────────────────────────────────────────────────────

void SkinDetector::skinMap(uchar* image)
{
    const size_t sz   = (size_t)rows * cols * channels;
    const int    step = cols * channels;
    cudaMemcpy(devInput[0], image, sz, cudaMemcpyHostToDevice);
    getSkinMap<<<gridDim, blockDim>>>(devInput[0], cols, rows, step,
                                      inverseCovDev, meanDev, threshDev);
    cudaMemcpy(image, devInput[0], sz, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

void SkinDetector::skinMask(uchar* image, uchar* output)
{
    const size_t inSz  = (size_t)rows * cols * channels;
    const size_t outSz = (size_t)rows * cols;
    const int    step  = cols * channels;
    cudaMemcpy(devInput[0], image, inSz, cudaMemcpyHostToDevice);
    getSkinMask<<<gridDim, blockDim>>>(devInput[0], devOutput[0], cols, rows, step,
                                       inverseCovDev, meanDev, threshDev);
    cudaMemcpy(output, devOutput[0], outSz, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

// ── Asynchronous API ──────────────────────────────────────────────────────────

void SkinDetector::skinMapAsync(uchar* pinnedFrame, int slot, cudaStream_t stream)
{
    const size_t sz   = (size_t)rows * cols * channels;
    const int    step = cols * channels;
    cudaMemcpyAsync(devInput[slot], pinnedFrame, sz, cudaMemcpyHostToDevice, stream);
    getSkinMap<<<gridDim, blockDim, 0, stream>>>(devInput[slot], cols, rows, step,
                                                  inverseCovDev, meanDev, threshDev);
    cudaMemcpyAsync(pinnedFrame, devInput[slot], sz, cudaMemcpyDeviceToHost, stream);
}

void SkinDetector::skinMaskAsync(const uchar* pinnedIn, uchar* pinnedOut,
                                  int slot, cudaStream_t stream)
{
    const size_t inSz  = (size_t)rows * cols * channels;
    const size_t outSz = (size_t)rows * cols;
    const int    step  = cols * channels;
    cudaMemcpyAsync(devInput[slot],  pinnedIn,  inSz,  cudaMemcpyHostToDevice, stream);
    getSkinMask<<<gridDim, blockDim, 0, stream>>>(devInput[slot], devOutput[slot], cols, rows, step,
                                                   inverseCovDev, meanDev, threshDev);
    cudaMemcpyAsync(pinnedOut, devOutput[slot], outSz, cudaMemcpyDeviceToHost, stream);
}

// ── GPU-resident API ──────────────────────────────────────────────────────────

void SkinDetector::skinMapInPlace(uchar* devFrame, cudaStream_t stream)
{
    getSkinMap<<<gridDim, blockDim, 0, stream>>>(devFrame, cols, rows, cols * channels,
                                                  inverseCovDev, meanDev, threshDev);
}

void SkinDetector::skinMapInPlace(uchar* devFrame, int frameCols, int frameRows,
                                   int frameStep, cudaStream_t stream)
{
    dim3 grid((frameCols + blockDim.x - 1) / blockDim.x,
              (frameRows + blockDim.y - 1) / blockDim.y);
    getSkinMap<<<grid, blockDim, 0, stream>>>(devFrame, frameCols, frameRows, frameStep,
                                               inverseCovDev, meanDev, threshDev);
}

void SkinDetector::skinMaskInPlace(const uchar* devFrame, uchar* devMask, cudaStream_t stream)
{
    getSkinMask<<<gridDim, blockDim, 0, stream>>>(devFrame, devMask, cols, rows, cols * channels,
                                                   inverseCovDev, meanDev, threshDev);
}
