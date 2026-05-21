# CUDA Skin Detector

A CUDA-accelerated skin detector and a worked example of a CMake CUDA project.

The detector classifies pixels using the **Mahalanobis distance in normalised-RGB space**.
Each pixel's normalised R and G components are compared against a Gaussian skin model
(mean + inverse covariance matrix). A pixel is labelled _skin_ when its gate value —
`exp(–0.5 · [nr, ng] · Σ⁻¹ · [nr, ng]ᵀ)` — exceeds a threshold.

---

## Features

| | |
|---|---|
| Kernel | `__expf` fast intrinsic, `fmaf` fused multiply-add, `__restrict__` / `const` qualifiers, division-by-zero guard, branchless mask write |
| Block size | 16 × 16 threads (256 / block) — better SM occupancy than 32 × 32 on Blackwell / Ampere |
| Memory | Explicit `cudaMalloc` device buffers; pinned host memory (`cudaHostAlloc`) for fast PCIe transfers |
| Video pipeline | Double-buffered: two CUDA streams alternate so the GPU processes frame N while the CPU reads frame N+1 |
| GPU decode | When OpenCV is built with NVDEC support, compressed video files are decoded directly into GPU memory — **zero H2D copy** |
| Architecture | Targets **sm_120 (Blackwell / RTX 50xx)** by default; one variable to change in `CMakeLists.txt` |

---

## Building

### Prerequisites

- CMake ≥ 3.26
- CUDA Toolkit (tested with CUDA 12.x)
- OpenCV 4.x (core + videoio + highgui)
- _(optional)_ OpenCV built with `WITH_NVCUVID=ON` to enable GPU video decode

### Quick start

```bash
# Clone
git clone https://github.com/LuisTbx/SkinCuda.git
cd SkinCuda/skin_detector

# Prepare test images (optional)
mkdir test_images
cd test_images
wget -O test_image.jpg https://assets.weforum.org/article/image/XaHpf_z51huQS_JPHs-jkPhBp0dLlxFJwt-sPLpGJB0.jpg
cd ..

# Build
mkdir build && cd build
cmake ..
make -j$(nproc)
```

CMake prints whether GPU video decode is enabled:

```
-- opencv_cudacodec found — GPU video decode path enabled
```

or

```
-- opencv_cudacodec NOT found — GPU video decode path disabled (CPU fallback active)
```

### GPU architecture

The default target is **sm_120** (Blackwell, RTX 50xx series).
Change one line in `CMakeLists.txt` to target a different GPU:

```cmake
set(CUDA_TARGET_ARCH "120")   # change to your GPU's compute capability
```

| GPU family | Compute capability |
|---|---|
| Pascal (GTX 10xx) | 61 |
| Turing (RTX 20xx) | 75 |
| Ampere (RTX 30xx) | 86 |
| Ada Lovelace (RTX 40xx) | 89 |
| Blackwell (RTX 50xx) | 120 |

For a portable binary that runs on multiple GPU generations use a semicolon-separated list:
```cmake
set(CUDA_TARGET_ARCH "61;75;86;89;120")
```

### Enabling GPU video decode (optional)

GPU video decode routes compressed video directly through the NVIDIA hardware decoder
(NVDEC) into GPU memory, eliminating the host↔device copy entirely.

**Requirement**: OpenCV must be compiled with NVDEC support.  On Linux:
```bash
cmake -DWITH_CUDA=ON -DWITH_NVCUVID=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ..
```

On Windows, the same flags apply when building OpenCV with the `opencv_contrib` repository.
Pre-built binaries from opencv.org do **not** include this module for licensing reasons.

**Supported formats** (hardware-dependent): H.264, H.265/HEVC, VP8, VP9, MJPEG.

---

## Usage

After building, three executables are available in `build/`.

```bash
./skin                          # webcam (CPU capture path)
./skin test_images/video.avi    # video file (GPU decode if available, else CPU)
./skin rtsp://camera/stream     # RTSP stream (GPU decode if available)

./skinMap  test_images/test_image.jpg   # → skinMap.jpg  (non-skin pixels zeroed)
./skinMask test_images/test_image.jpg   # → skinMask.jpg (binary mask: 255 = skin)
```

Press **ESC** to exit the live viewer.

---

## Pipeline overview

### Synchronous (single image)

```
Host image  ──cudaMemcpy H→D──▶  devInput  ──kernel──▶  devInput / devOutput
            ◀─cudaMemcpy D→H──
```

### CPU-capture path: double-buffered async (webcam / no NVDEC)

```
          CPU                              GPU stream 0          GPU stream 1
read frame 0 → pinnedBuf[0]  ──H→D──▶  kernel ──D→H──▶  display
read frame 1 → pinnedBuf[1]  ──────────────────── H→D──▶  kernel ──D→H──▶  display
read frame 2 → pinnedBuf[0]  ──H→D──▶  kernel ──D→H──▶  display
```

CPU frame capture overlaps with GPU computation.  Pinned host memory removes the
OS page-lock copy step, approximately doubling effective PCIe bandwidth.

### GPU-decode path: zero H2D copy (NVDEC + cudacodec)

```
Compressed video
    ──NVDEC──▶  GpuMat (device)  ──kernel (in-place)──▶  GpuMat  ──D2H──▶  display
```

The hardware decoder writes directly to device memory.  The skin kernel operates
on that buffer without any host involvement.  Only the display download (D2H) touches
the PCIe bus.

---

## SkinDetector API

```cpp
// Synchronous (blocks until GPU finishes)
void skinMap(uchar* image);                        // in-place, host memory
void skinMask(uchar* image, uchar* output);        // host memory

// Async double-buffered (returns immediately; use cudaEvent to sync)
void skinMapAsync(uchar* pinnedFrame, int slot, cudaStream_t stream);
void skinMaskAsync(const uchar* pinnedIn, uchar* pinnedOut, int slot, cudaStream_t stream);

// GPU-resident (frame already on the device — no copy at all)
void skinMapInPlace(uchar* devFrame, cudaStream_t stream = 0);
void skinMaskInPlace(const uchar* devFrame, uchar* devMask, cudaStream_t stream = 0);
```

---

## Algorithm parameters

Default values (tuned on a diverse skin dataset):

| Parameter | Value |
|---|---|
| Covariance Σ | `[[0.0038, -0.0009], [-0.0009, 0.0009]]` |
| Mean (nR, nG) | `[0.4404, 0.3111]` |
| Threshold | `0.33` |

Raise the threshold to detect less skin (fewer false positives); lower it for higher recall.

---

## Project structure

```
├── CMakeLists.txt           Root build — arch variable, optional cudacodec detection
├── skin.cu                  Live video/webcam: GPU decode path + CPU fallback
├── skinMap.cu               Single-image skin map
├── skinMask.cu              Single-image binary mask
└── skin_detector/
    ├── CMakeLists.txt       Library build
    ├── hdrs/
    │   ├── SkinDetector.h   Public class: sync / async / GPU-resident API
    │   └── SkinKernel.cuh   Kernel declarations
    └── src/
        ├── SkinDetector.cu  Detector implementation, stream management
        └── SkinKernel.cu    CUDA kernels (getSkinMap, getSkinMask)
```

---

If you have suggestions or run into issues, open an issue or PR on GitHub.

Aknowledgement: The first implementation of this repo started following the video tutorial from:  [https://github.com/abubakr-shafique/Image_Inversion_CUDA_CPP]
