#include "SkinKernel.cuh"
#include <math.h>

#ifndef uchar
    typedef unsigned char uchar;
#endif

// Computes the Gaussian gate value from BGR pixel components.
// Returns a value in [0, 1]; higher means more skin-like colour.
// Implements: gate = exp(-0.5 * [nr, ng] * Sigma^-1 * [nr, ng]^T)
// where nr, ng are the normalised-RGB deviations from the skin mean.
__device__ __forceinline__ float computeGate(
    float B, float G, float R,
    const float* __restrict__ inverseCov,
    const float* __restrict__ mean)
{
    const float sum = B + G + R;
    if (sum < 1e-6f) return 0.0f;   // black / near-black pixel — not skin

    const float inv = 1.0f / sum;
    const float nr  = R * inv - mean[0];
    const float ng  = G * inv - mean[1];

    // Quadratic form of the 2×2 symmetric inverse covariance matrix, using
    // fused multiply-add (fmaf) to reduce rounding and improve throughput.
    const float exponent = fmaf(nr * nr,              inverseCov[0],
                            fmaf(2.0f * inverseCov[1], nr * ng,
                                 ng * ng *             inverseCov[3]));
    return __expf(-0.5f * exponent);   // fast single-precision exp intrinsic
}

// Applies the skin map in-place: zeroes pixels whose gate value is below the
// threshold.  Every thread handles exactly one pixel.
__global__ void getSkinMap(uchar* __restrict__ image, int cols, int rows,
                           const float* __restrict__ inverseCovariance,
                           const float* __restrict__ mean,
                           const float* __restrict__ threshold)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    const int   idx = (x + y * cols) * 3;
    const float B   = (float)image[idx];
    const float G   = (float)image[idx + 1];
    const float R   = (float)image[idx + 2];

    if (computeGate(B, G, R, inverseCovariance, mean) <= threshold[0]) {
        image[idx]     = 0;
        image[idx + 1] = 0;
        image[idx + 2] = 0;
    }
}

// Writes a binary mask: 255 for skin pixels, 0 for non-skin.
// Output is written unconditionally (branchless) to avoid warp divergence.
__global__ void getSkinMask(const uchar* __restrict__ image, uchar* __restrict__ output,
                            int cols, int rows,
                            const float* __restrict__ inverseCovariance,
                            const float* __restrict__ mean,
                            const float* __restrict__ threshold)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    const int   idx = (x + y * cols) * 3;
    const float B   = (float)image[idx];
    const float G   = (float)image[idx + 1];
    const float R   = (float)image[idx + 2];

    output[x + y * cols] =
        (computeGate(B, G, R, inverseCovariance, mean) >= threshold[0]) ? 255 : 0;
}
