#include <iostream>
#include <cstring>
#include <chrono>
#include <cstdio>
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


// ── Per-frame timing stats (exponential moving averages) ─────────────────────

struct Stats {
    float fps      = 0.0f;
    float frameMs  = 0.0f;
    float kernelMs = 0.0f;
    float h2dMs    = 0.0f;   // CPU capture path
    float d2hMs    = 0.0f;   // CPU capture path
    float decodeMs = 0.0f;   // GPU decode path
    float dlMs     = 0.0f;   // GPU decode path (GpuMat download)
};

// Draws a small timing overlay in the top-left corner.
// gpuDecode selects which pair of transfer labels to show.
static void drawStats(cv::Mat& frame, const Stats& s, bool gpuDecode)
{
    const int kLH   = 17;    // line height (px)
    const int kPad  = 8;
    const int kW    = 172;
    const int kH    = kPad * 2 + 5 * kLH;
    cv::Rect  roi(kPad, kPad, kW, kH);
    roi &= cv::Rect(0, 0, frame.cols, frame.rows);
    cv::rectangle(frame, roi, cv::Scalar(18, 18, 18), cv::FILLED);

    char buf[48];
    int  y = kPad + kLH;
    auto put = [&](cv::Scalar col) {
        cv::putText(frame, buf, {kPad + 5, y},
                    cv::FONT_HERSHEY_SIMPLEX, 0.48, col, 1);
        y += kLH;
    };

    snprintf(buf, sizeof(buf), "FPS    %6.1f",    s.fps);      put({50,  220, 80});
    snprintf(buf, sizeof(buf), "Frame  %5.2f ms", s.frameMs);  put({200, 200, 200});
    snprintf(buf, sizeof(buf), "Kernel %5.3f ms", s.kernelMs); put({200, 200, 200});
    if (gpuDecode) {
        snprintf(buf, sizeof(buf), "Decode %5.2f ms", s.decodeMs); put({80, 200, 255});
        snprintf(buf, sizeof(buf), "Dwnld  %5.2f ms", s.dlMs);     put({80, 200, 255});
    } else {
        snprintf(buf, sizeof(buf), "H2D    %5.3f ms", s.h2dMs);    put({80, 200, 255});
        snprintf(buf, sizeof(buf), "D2H    %5.3f ms", s.d2hMs);    put({80, 200, 255});
    }
}

// Update EMA: first call initialises; subsequent calls blend with alpha=0.15.
static void updateEma(Stats& s, bool& init,
                      float fps, float frameMs, float kernelMs,
                      float h2dMs, float d2hMs,
                      float decodeMs = 0.0f, float dlMs = 0.0f)
{
    if (!init) {
        s    = {fps, frameMs, kernelMs, h2dMs, d2hMs, decodeMs, dlMs};
        init = true;
        return;
    }
    constexpr float a = 0.15f, b = 1.0f - a;
    s.fps      = b * s.fps      + a * fps;
    s.frameMs  = b * s.frameMs  + a * frameMs;
    s.kernelMs = b * s.kernelMs + a * kernelMs;
    s.h2dMs    = b * s.h2dMs    + a * h2dMs;
    s.d2hMs    = b * s.d2hMs    + a * d2hMs;
    s.decodeMs = b * s.decodeMs + a * decodeMs;
    s.dlMs     = b * s.dlMs     + a * dlMs;
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

    cudaEvent_t evKernelStart, evKernelEnd;
    cudaEventCreate(&evKernelStart);
    cudaEventCreate(&evKernelEnd);

    using Clock = std::chrono::steady_clock;
    using Ms    = std::chrono::duration<float, std::milli>;

    Stats s, snap;
    bool  emaInit  = false;
    auto  tSnap    = Clock::now();
    auto  tPrev    = Clock::now();

    std::cout << "[cudacodec] GPU decode active — zero H2D copy per frame\n";

    for (;;) {
        auto tDecode = Clock::now();
        if (!reader->nextFrame(gpuFrame) || gpuFrame.empty()) break;
        float decodeMs = Ms(Clock::now() - tDecode).count();

        // Pass actual decoded dimensions and row pitch — NVDEC pads frame rows to
        // a CUDA alignment boundary, so gpuFrame.step >= gpuFrame.cols * 3.
        cudaEventRecord(evKernelStart, 0);
        det.skinMapInPlace(gpuFrame.ptr<uchar>(), gpuFrame.cols, gpuFrame.rows,
                           (int)gpuFrame.step);
        cudaEventRecord(evKernelEnd, 0);

        // Download only for display (unavoidable; remove if a GPU display path
        // such as OpenGL interop is available).
        auto tDl = Clock::now();
        gpuFrame.download(display);   // synchronises default stream
        float dlMs = Ms(Clock::now() - tDl).count();

        float kernelMs = 0.0f;
        cudaEventElapsedTime(&kernelMs, evKernelStart, evKernelEnd);

        auto  tNow    = Clock::now();
        float frameMs = Ms(tNow - tPrev).count();
        tPrev = tNow;
        float fps = frameMs > 0.001f ? 1000.0f / frameMs : s.fps;

        updateEma(s, emaInit, fps, frameMs, kernelMs, 0.0f, 0.0f, decodeMs, dlMs);
        if (Ms(tNow - tSnap).count() >= 500.0f) { snap = s; tSnap = tNow; }
        drawStats(display, snap, /*gpuDecode=*/true);

        cv::imshow("SKINMAP [GPU decode]", display);
        if (cv::waitKey(1) == 27) break;
    }

    cudaEventDestroy(evKernelStart);
    cudaEventDestroy(evKernelEnd);
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

    using Clock = std::chrono::steady_clock;
    using Ms    = std::chrono::duration<float, std::milli>;

    Stats s, snap;
    bool  emaInit = false;
    auto  tSnap   = Clock::now();
    auto  tPrev   = Clock::now();
    int   slot    = 1;

    for (;;) {
        const bool gotFrame = cap.read(frame) && !frame.empty();

        if (gotFrame) {
            std::memcpy(pinnedFrame[slot], frame.data, frameBytes);
            det.skinMapAsync(pinnedFrame[slot], slot, streams[slot]);
            cudaEventRecord(events[slot], streams[slot]);
        }

        const int prev = 1 - slot;
        cudaEventSynchronize(events[prev]);

        // All timing events for slot 'prev' are completed — query them now.
        float h2d    = det.getH2Dms(prev);
        float kernel = det.getKernelMs(prev);
        float d2h    = det.getD2Hms(prev);

        auto  tNow    = Clock::now();
        float frameMs = Ms(tNow - tPrev).count();
        tPrev = tNow;
        float fps = frameMs > 0.001f ? 1000.0f / frameMs : s.fps;

        updateEma(s, emaInit, fps, frameMs, kernel, h2d, d2h);
        if (Ms(tNow - tSnap).count() >= 500.0f) { snap = s; tSnap = tNow; }

        // Draw overlay directly on the pinned buffer — safe because memcpy
        // overwrites it with fresh frame data before the GPU reads it next.
        cv::Mat view(rows, cols, CV_8UC3, pinnedFrame[prev]);
        drawStats(view, snap, /*gpuDecode=*/false);
        cv::imshow("SKINMAP", view);

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
