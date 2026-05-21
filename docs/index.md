---
layout: default
title: Real-Time Skin Detection on the GPU
description: A walkthrough of SkinCuda — a CUDA-accelerated skin detector with a double-buffered async video pipeline, hardware video decode, and live performance profiling.
---

# Real-Time Skin Detection on the GPU

*Design notes for [SkinCuda](https://github.com/LuisTbx/SkinCuda) — a from-scratch CUDA pipeline for skin pixel classification.*

---

## Why GPU acceleration for something this simple?

Skin detection sounds trivial: look at a pixel, decide if it looks like skin. The algorithm behind SkinCuda is indeed a single mathematical formula per pixel. The challenge is **scale**. A 1080p frame contains about 2 million pixels. At 60 frames per second that is 120 million pixel evaluations every second — and each one involves a matrix multiplication, a division, and an exponential. On a single CPU core, this saturates. On a GPU with thousands of cores running in parallel, it completes in under a millisecond.

This article walks through the key ideas in the implementation: the detection algorithm, the GPU kernel design, the memory architecture, the video pipeline, and the profiling system built to measure all of it.

---

## The detection algorithm

The detector classifies each pixel using the **Mahalanobis distance** in normalised-RGB colour space.

Instead of using raw red, green, and blue values — which change wildly with lighting — the pixel is first projected into a lighting-invariant space. For a pixel with components R, G, B, the normalised values are:

```
nR = R / (R + G + B)
nG = G / (R + G + B)
```

Because `nR + nG + nB = 1`, only two dimensions carry information. Human skin in this space clusters tightly around a known mean regardless of scene illumination, which is what makes the approach robust.

The classification test is a **Gaussian gate**:

```
gate = exp( −0.5 · [nR−μR, nG−μG] · Σ⁻¹ · [nR−μR, nG−μG]ᵀ )
```

`Σ⁻¹` is the 2×2 inverse covariance matrix of the skin cluster, and `[μR, μG]` is its mean. The gate value is 1 at the centre of the skin distribution and decays toward zero for non-skin colours. A pixel is labelled skin when `gate > threshold`.

The default parameters were tuned on a diverse skin dataset:

| Parameter | Value |
|---|---|
| Mean (nR, nG) | 0.4404, 0.3111 |
| Covariance Σ | \[\[0.0038, −0.0009\], \[−0.0009, 0.0009\]\] |
| Threshold | 0.33 |

This single threshold can be raised to reduce false positives or lowered for higher recall, making it easy to tune for a specific application.

---

## Kernel design

Every pixel is independent — one thread handles one pixel. The GPU launches a 2D grid of 16×16-thread blocks covering the entire image, processing all pixels simultaneously.

Inside the kernel, a few implementation choices matter for throughput:

**Fast math intrinsics.** The exponential is computed with `__expf` instead of `expf`. The intrinsic trades a small amount of precision (irrelevant here, since we are comparing against a threshold) for hardware-native speed. Similarly, `fmaf(a, b, c)` evaluates `a·b + c` as a fused multiply-add in a single instruction, which is relevant for the matrix multiplication in the gate formula.

**Block size of 16×16 instead of 32×32.** A 32×32 block uses 1024 threads — the hardware maximum. Filling a Streaming Multiprocessor with 1024-thread blocks means fewer blocks per SM. Smaller blocks allow the scheduler to place more blocks on each SM, hiding memory latency by switching between them while others wait for data. On Blackwell and Ampere GPUs, 16×16 (256 threads) produces measurably better occupancy.

**Branchless mask write.** The skin mask kernel writes `255` for skin and `0` for background. Rather than branching on the gate comparison, the result is computed as `(uchar)(gate > threshold) * 255`. The branch predictor handles poorly-predictable per-pixel decisions — eliminating the branch entirely is faster.

**`__restrict__` and `const` qualifiers.** Marking pointer parameters `__restrict__` tells the compiler that no aliasing exists between the input and output buffers. This enables load/store optimisations that would be illegal to perform in the presence of aliasing. Marking them `const` where applicable allows the compiler to use read-only cache paths (texture cache on older architectures, uniform cache on newer ones).

---

## Memory architecture

Getting data to and from the GPU quickly is often the real bottleneck in a vision pipeline, not the computation itself.

**Pinned host memory.** Ordinary `malloc` allocates pageable memory that the OS can swap to disk at any time. Before the PCIe DMA engine can transfer it, the CUDA driver must first copy it to a locked page — a hidden extra copy. Allocating with `cudaHostAlloc` instead produces memory that is permanently page-locked, letting the DMA engine transfer it directly. In practice this roughly doubles effective PCIe bandwidth.

**Double-buffered device memory.** The detector maintains two sets of device buffers (`devInput[0/1]`, `devOutput[0/1]`), one per pipeline slot. This is what allows two CUDA streams to operate concurrently without interfering with each other.

**Device-resident parameters.** The inverse covariance matrix, mean, and threshold are small (7 floats total) but accessed by every thread. They are uploaded to device memory once at startup and effectively live in L1 / constant cache throughout execution, so the per-thread read cost is negligible.

---

## The video pipeline

### CPU-capture path: double-buffered async

For webcams and as a fallback when hardware decode is unavailable, the pipeline uses two alternating CUDA streams to overlap CPU work with GPU work:

```
         CPU                            GPU stream 0          GPU stream 1
read frame 0 → pinnedBuf[0]  ──H→D──▶  kernel ──D→H──▶  display
read frame 1 → pinnedBuf[1]  ─────────────────── H→D──▶  kernel ──D→H──▶  display
read frame 2 → pinnedBuf[0]  ──H→D──▶  kernel ──D→H──▶  display
```

While the GPU is processing frame N, the CPU reads frame N+1 into the other pinned buffer. By the time the CPU is done reading, the GPU has finished with its slot and is ready for the next transfer. The two streams never touch the same buffer at the same time.

This is the key insight: without double buffering, the CPU would sit idle waiting for the GPU, and then the GPU would sit idle waiting for the CPU. With double buffering, both run continuously.

### GPU-decode path: zero host↔device copy

When OpenCV is built with NVDEC support (the NVIDIA hardware video decoder), compressed video files can be decoded directly into GPU memory. The skin kernel then operates on that device-resident frame in place. The only PCIe traffic is the display download — one direction instead of two:

```
Compressed video
  ──NVDEC──▶  GpuMat (device)  ──kernel (in-place)──▶  GpuMat  ──D→H──▶  display
```

For a 1080p H.264 stream at 30 fps, this eliminates roughly 370 MB/s of PCIe traffic (the H→D upload of decoded raw frames), leaving only the D→H download for display.

The pipeline detects at startup whether `opencv_cudacodec` is available and falls back gracefully to the CPU-capture path if not.

---

## Performance profiling

Understanding where time is spent is essential for optimising a GPU pipeline. Naive approaches — wrapping the whole frame in `std::chrono` — collapse all stages into one number and hide where the real cost is.

SkinCuda instruments the pipeline at four points using **stream-ordered CUDA events**. Four `cudaEvent_t` markers are recorded inside `skinMapAsync`, bracketing each stage:

```
[evStart] ── H2D copy ── [evAfterH2D] ── kernel ── [evAfterKernel] ── D2H copy ── [evEnd]
```

Events are recorded on the same CUDA stream as the work they bracket, so their timestamps are causally ordered with the operations. After the existing `cudaEventSynchronize` call (which the pipeline already needed for correctness), `cudaEventElapsedTime` is called to extract the three intervals. **No extra synchronisation is added** — the timing information comes for free by querying events that are already guaranteed complete.

An **exponential moving average** (α = 0.15) smooths each measurement across frames, preventing the display from reflecting single-frame outliers. The display itself only refreshes every 500 ms so the numbers are stable enough to read.

The live overlay shows:

| Field | What it measures |
|---|---|
| FPS | Wall-clock frame delivery rate |
| Frame ms | End-to-end time per displayed frame |
| Kernel ms | GPU skin detection kernel alone |
| H2D ms | Host→device PCIe transfer |
| D2H ms | Device→host PCIe transfer |

Separating H2D, kernel, and D2H makes it immediately visible whether a performance issue is transfer-bound (large H2D/D2H relative to kernel) or compute-bound (kernel dominates).

---

## Key takeaways

- **Per-pixel independent work is a perfect fit for the GPU.** The algorithm touches no neighbours, making the kernel trivially parallel and free of synchronisation overhead.
- **Memory access patterns matter as much as compute.** Pinned memory and double buffering collectively remove the two main pipeline stalls: OS-level copies and CPU/GPU idle time while the other side works.
- **Measure first, optimise second.** Adding stream-ordered CUDA event timing cost four `cudaEventRecord` calls per frame and zero extra synchronisation. That investment immediately revealed the real shape of each frame's time budget.
- **Hardware decoders are underutilised.** NVDEC can decode H.264 at hundreds of frames per second without touching the CUDA cores or the PCIe bus for the upload. For any pipeline that processes compressed video, routing decode through the hardware unit is almost always worth the setup cost.

---

*Source code: [github.com/LuisTbx/SkinCuda](https://github.com/LuisTbx/SkinCuda)*
