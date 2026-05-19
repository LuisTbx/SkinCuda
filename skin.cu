// Losely based on https://github.com/abubakr-shafique/Image_Inversion_CUDA_CPP/blob/master/kernel.cu
#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "SkinDetector.h"


int main(int argc, char* argv[])
{
    const std::string capture_name = (argc >= 2) ? argv[1] : "0";
    std::cout << "Opening: " << capture_name << std::endl;

    cv::VideoCapture cap(capture_name, cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open video source: " << capture_name << std::endl;
        return 1;
    }

    // CAP_PROP_FRAME_WIDTH is the column count and FRAME_HEIGHT is the row count.
    const int cols = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int rows = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Resolution: " << cols << " x " << rows << std::endl;

    // Gaussian skin model in normalised-RGB colour space.
    cv::Mat covariance = (cv::Mat_<float>(2, 2) << 0.0038f, -0.0009f, -0.0009f, 0.0009f);
    cv::Mat meanCV     = (cv::Mat_<float>(2, 1) << 0.4404f, 0.3111f);
    cv::Mat inverseCov;
    cv::invert(covariance, inverseCov);
    const float threshold = 0.33f;

    SkinDetector det((float*)inverseCov.data, (float*)meanCV.data, threshold, cols, rows);

    // ── Double-buffered async pipeline ────────────────────────────────────────
    // Two pinned host buffers let cudaMemcpyAsync transfer without stalling.
    // Two CUDA streams allow the GPU to process frame N while the CPU reads N+1.
    // cudaEventDisableTiming skips timestamp hardware, making events cheaper.

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

    // Prime the pipeline: submit the very first frame on slot 0.
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        std::cerr << "Error: source produced no frames" << std::endl;
        return 1;
    }
    std::memcpy(pinnedFrame[0], frame.data, frameBytes);
    det.skinMapAsync(pinnedFrame[0], 0, streams[0]);
    cudaEventRecord(events[0], streams[0]);

    int slot = 1;
    for (;;) {
        // Read the next frame on the CPU while the GPU processes the previous slot.
        const bool gotFrame = cap.read(frame) && !frame.empty();

        if (gotFrame) {
            std::memcpy(pinnedFrame[slot], frame.data, frameBytes);
            det.skinMapAsync(pinnedFrame[slot], slot, streams[slot]);
            cudaEventRecord(events[slot], streams[slot]);
        }

        // Wait for the previous slot to finish, then display its result.
        const int prev = 1 - slot;
        cudaEventSynchronize(events[prev]);
        cv::imshow("SKINMAP", cv::Mat(rows, cols, CV_8UC3, pinnedFrame[prev]));

        if (cv::waitKey(1) == 27 || !gotFrame) break;
        slot ^= 1;
    }

    // Drain any in-flight GPU work before releasing resources.
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
