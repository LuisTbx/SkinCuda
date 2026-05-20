// Losely based on https://github.com/abubakr-shafique/Image_Inversion_CUDA_CPP/blob/master/kernel.cu
#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "SkinDetector.h"

#ifdef HAVE_CUDACODEC
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudawarping.hpp>   // cuda::resize (for display downscale if needed)
#include <opencv2/cudaimgproc.hpp>   // cuda::cvtColor
#endif


// ── Helpers ───────────────────────────────────────────────────────────────────

static bool isWebcam(const std::string& src)
{
    // Treat single-digit strings and empty as webcam indices.
    if (src.empty()) return true;
    for (char c : src) if (!std::isdigit(c)) return false;
    return true;
}


// ── GPU decode path (NVDEC via cv::cudacodec) ─────────────────────────────────
// Frames are decoded directly into device memory by the hardware video decoder.
// The skin kernel runs on the GPU-resident frame — no host↔device copy at all.
// Requires OpenCV built with WITH_NVCUVID=ON (or WITH_CUDA + Video Codec SDK).
// Supported containers / codecs: H.264, H.265, VP8, VP9, MJPEG (file-based).

#ifdef HAVE_CUDACODEC
static int runGpuDecode(const std::string& filename, SkinDetector& det)
{
    cv::Ptr<cv::cudacodec::VideoReader> reader;
    try {
        cv::cudacodec::VideoReaderInitParams params;
        reader = cv::cudacodec::createVideoReader(filename, {}, params);
        reader->set(cv::cudacodec::ColorFormat::BGR);
    } catch (const cv::Exception& e) {
        std::cerr << "[cudacodec] Cannot open '" << filename << "': " << e.what()
                  << "\n[cudacodec] Falling back to CPU capture.\n";
        return -1;   // signal caller to fall back
    }

    cv::cuda::GpuMat gpuFrame;
    cv::Mat display;

    std::cout << "[cudacodec] GPU decode active — zero H2D copy per frame\n";

    for (;;) {
        if (!reader->nextFrame(gpuFrame) || gpuFrame.empty()) break;

        // Run skin detection in-place directly on the device buffer.
        det.skinMapInPlace(gpuFrame.ptr<uchar>());

        // Download only for display (unavoidable; remove if a GPU display path
        // such as OpenGL interop is available).
        gpuFrame.download(display);
        cv::imshow("SKINMAP [GPU decode]", display);
        if (cv::waitKey(1) == 27) break;
    }

    cv::destroyAllWindows();
    return 0;
}
#endif   // HAVE_CUDACODEC


// ── CPU capture path with double-buffered async transfer ──────────────────────
// Used for webcams and as a fallback when cudacodec is unavailable.
// Two pinned host buffers alternate: while the GPU processes frame N, the CPU
// reads frame N+1.  PCIe transfers use cudaMemcpyAsync into pinned memory.

static int runCpuCapture(const std::string& capture_name, SkinDetector& det,
                         int cols, int rows)
{
    cv::VideoCapture cap(capture_name, cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open video source: " << capture_name << std::endl;
        return 1;
    }

    const size_t frameBytes = (size_t)cols * rows * 3;

    uchar* pinnedFrame[2];
    cudaHostAlloc(&pinnedFrame[0], frameBytes, cudaHostAllocDefault);
    cudaHostAlloc(&pinnedFrame[1], frameBytes, cudaHostAllocDefault);

    cudaStream_t streams[2];
    cudaEvent_t  events[2];
    for (int i = 0; i < 2; i++) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
    }

    // Prime pipeline with the first frame.
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        std::cerr << "Error: source produced no frames\n";
        return 1;
    }
    std::memcpy(pinnedFrame[0], frame.data, frameBytes);
    det.skinMapAsync(pinnedFrame[0], 0, streams[0]);
    cudaEventRecord(events[0], streams[0]);

    int slot = 1;
    for (;;) {
        const bool gotFrame = cap.read(frame) && !frame.empty();

        if (gotFrame) {
            std::memcpy(pinnedFrame[slot], frame.data, frameBytes);
            det.skinMapAsync(pinnedFrame[slot], slot, streams[slot]);
            cudaEventRecord(events[slot], streams[slot]);
        }

        const int prev = 1 - slot;
        cudaEventSynchronize(events[prev]);
        cv::imshow("SKINMAP", cv::Mat(rows, cols, CV_8UC3, pinnedFrame[prev]));

        if (cv::waitKey(1) == 27 || !gotFrame) break;
        slot ^= 1;
    }

    for (int i = 0; i < 2; i++) cudaStreamSynchronize(streams[i]);

    for (int i = 0; i < 2; i++) {
        cudaEventDestroy(events[i]);
        cudaStreamDestroy(streams[i]);
        cudaFreeHost(pinnedFrame[i]);
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}


// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    const std::string capture_name = (argc >= 2) ? argv[1] : "0";

    // Probe frame dimensions regardless of which path we take.
    // For webcam we need the actual capture size; for files we can read it here.
    cv::VideoCapture probe(capture_name, cv::CAP_ANY);
    if (!probe.isOpened()) {
        std::cerr << "Error: cannot open video source: " << capture_name << std::endl;
        return 1;
    }
    const int cols = (int)probe.get(cv::CAP_PROP_FRAME_WIDTH);
    const int rows = (int)probe.get(cv::CAP_PROP_FRAME_HEIGHT);
    probe.release();

    std::cout << "Source: " << capture_name
              << "  resolution: " << cols << "x" << rows << "\n";

    // Gaussian skin model in normalised-RGB colour space.
    cv::Mat covariance = (cv::Mat_<float>(2, 2) << 0.0038f, -0.0009f, -0.0009f, 0.0009f);
    cv::Mat meanCV     = (cv::Mat_<float>(2, 1) << 0.4404f, 0.3111f);
    cv::Mat inverseCov;
    cv::invert(covariance, inverseCov);
    const float threshold = 0.33f;

    SkinDetector det((float*)inverseCov.data, (float*)meanCV.data, threshold, cols, rows);

#ifdef HAVE_CUDACODEC
    // For video files, try the zero-copy NVDEC path first.
    // Webcam streams (integer index) are not supported by cudacodec; go straight
    // to the double-buffered CPU path for those.
    if (!isWebcam(capture_name)) {
        int ret = runGpuDecode(capture_name, det);
        if (ret == 0) return 0;   // success — no fallback needed
        // ret == -1 means cudacodec could not open the source; fall through.
    } else {
        std::cout << "[cudacodec] Webcam source detected — using CPU capture path\n";
    }
#endif

    return runCpuCapture(capture_name, det, cols, rows);
}
