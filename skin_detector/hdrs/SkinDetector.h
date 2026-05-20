#pragma once
#ifndef SKINDETECTOR_H
#define SKINDETECTOR_H

#include <cuda_runtime.h>
#include "SkinKernel.cuh"

#ifndef uchar
    typedef unsigned char uchar;
#endif

class SkinDetector
{
public:
    SkinDetector();

    /**
     * @brief Construct a SkinDetector for images of a fixed size.
     *
     * @param mInverseCovDev  Row-major 2×2 inverse covariance matrix (4 floats).
     * @param mMean           Mean normalised-R and normalised-G values (2 floats).
     * @param mThreshold      Gate threshold in [0, 1]; raise to detect less skin.
     * @param mCols           Frame width in pixels.
     * @param mRows           Frame height in pixels.
     */
    SkinDetector(float* mInverseCovDev, float* mMean, float mThreshold,
                 int mCols, int mRows);
    ~SkinDetector();

    // ── Synchronous API ───────────────────────────────────────────────────────
    // These functions block the calling thread until the GPU has finished.

    /** Zero non-skin pixels directly in 'image' (in-place, 3-channel BGR). */
    void skinMap(uchar* image);

    /** Write a binary mask into 'output' (1-channel: 255 = skin, 0 = background). */
    void skinMask(uchar* image, uchar* output);

    // ── Asynchronous API ──────────────────────────────────────────────────────
    // Work is queued into 'stream' and the call returns immediately.
    // Use cudaEventRecord / cudaEventSynchronize to know when a frame is ready.
    //
    // IMPORTANT: pinnedFrame / pinnedIn / pinnedOut must be allocated with
    // cudaHostAlloc so that cudaMemcpyAsync can transfer them without stalling.
    //
    // 'slot' selects which internal double-buffer pair (0 or 1) to use.
    // Alternate slots across frames to overlap GPU work with CPU frame capture.

    /** Async in-place skin map on a pinned host frame. */
    void skinMapAsync(uchar* pinnedFrame, int slot, cudaStream_t stream);

    /** Async skin mask: reads from pinnedIn, writes binary mask to pinnedOut. */
    void skinMaskAsync(const uchar* pinnedIn, uchar* pinnedOut,
                       int slot, cudaStream_t stream);

    // ── GPU-resident API ──────────────────────────────────────────────────────
    // Use these when the frame is already in device memory (e.g. decoded by
    // NVDEC via cv::cudacodec::VideoReader).  No host↔device copy is performed.

    /** In-place skin map on a frame that is already on the GPU.
     *  devFrame must point to a BGR image of size (cols * rows * 3) bytes. */
    void skinMapInPlace(uchar* devFrame, cudaStream_t stream = 0);

    /** Overload that uses explicit dimensions and row pitch instead of the stored ones.
     *  Required for GpuMat frames: CUDA aligns each row to a pitch boundary so
     *  gpuFrame.step is typically larger than gpuFrame.cols * 3.
     *  frameStep is gpuFrame.step in bytes. */
    void skinMapInPlace(uchar* devFrame, int frameCols, int frameRows,
                        int frameStep, cudaStream_t stream = 0);

    /** Write a binary mask from a frame already on the GPU.
     *  devMask must point to a pre-allocated device buffer of (cols * rows) bytes. */
    void skinMaskInPlace(const uchar* devFrame, uchar* devMask, cudaStream_t stream = 0);

private:
    float* meanDev;
    float* inverseCovDev;
    float* threshDev;

    // Double-buffered device images — slot 0 is used by synchronous calls;
    // slots 0 and 1 alternate in the async double-buffered video pipeline.
    uchar* devInput[2];
    uchar* devOutput[2];

    int   channels;
    int   rows;
    int   cols;
    dim3  gridDim;
    dim3  blockDim;
};

#endif
