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
| Memory | Explicit `cudaMalloc` device buffers; application uses `cudaHostAlloc` pinned memory for fast PCIe transfers |
| Video pipeline | Double-buffered: two CUDA streams alternate so the GPU processes frame N while the CPU reads frame N+1 |
| Architecture | Targets **sm_120 (Blackwell / RTX 50xx)** by default; one variable to change in `CMakeLists.txt` |

---

## Building

### Prerequisites

- CMake ≥ 3.26
- CUDA Toolkit (tested with CUDA 12.x)
- OpenCV (any recent 4.x build)

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

### GPU architecture

The default target is **sm_120** (Blackwell, RTX 50xx series).
To target a different GPU, change one line in `CMakeLists.txt`:

```cmake
# Root CMakeLists.txt
set(CUDA_TARGET_ARCH "120")   # change to your GPU's compute capability
```

| GPU family | Compute capability |
|---|---|
| Pascal (GTX 10xx) | 61 |
| Turing (RTX 20xx) | 75 |
| Ampere (RTX 30xx) | 86 |
| Ada Lovelace (RTX 40xx) | 89 |
| Blackwell (RTX 50xx) | 120 |

For a portable binary that runs on multiple GPU generations, use a semicolon-separated list:
```cmake
set(CUDA_TARGET_ARCH "61;75;86;89;120")
```

---

## Usage

After building, three executables are available in `build/`.

```bash
./skin                          # webcam (default)
./skin test_images/video.avi    # video file

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

### Asynchronous double-buffered (video)

```
           CPU                              GPU (stream 0)        GPU (stream 1)
 read frame 0 → pinnedFrame[0]  ──H→D──▶  kernel ──D→H──▶  display frame 0
 read frame 1 → pinnedFrame[1]  ──────────────────── H→D──▶  kernel ──D→H──▶  display frame 1
 read frame 2 → pinnedFrame[0]  ──H→D──▶  kernel ──D→H──▶  display frame 2
 ...
```

CPU frame capture overlaps with GPU computation.  Pinned host memory removes the
extra copy through the OS page-locked buffer, approximately doubling PCIe throughput.

---

## Algorithm parameters

Default values (tuned on a diverse skin dataset):

| Parameter | Value |
|---|---|
| Covariance Σ | `[[0.0038, -0.0009], [-0.0009, 0.0009]]` |
| Mean (nR, nG) | `[0.4404, 0.3111]` |
| Threshold | `0.33` |

To adjust sensitivity, raise the threshold (fewer skin detections) or lower it
(more detections, more false positives).

---

## Project structure

```
├── CMakeLists.txt           Root build — three executables, one CUDA_TARGET_ARCH variable
├── skin.cu                  Live video / webcam with double-buffered GPU pipeline
├── skinMap.cu               Single-image skin map
├── skinMask.cu              Single-image binary mask
└── skin_detector/
    ├── CMakeLists.txt       Library build
    ├── hdrs/
    │   ├── SkinDetector.h   Public class: synchronous + async API
    │   └── SkinKernel.cuh   Kernel declarations
    └── src/
        ├── SkinDetector.cu  Detector implementation, stream management
        └── SkinKernel.cu    CUDA kernels (getSkinMap, getSkinMask)
```

---

If you have suggestions or run into issues, open an issue or PR on GitHub.
