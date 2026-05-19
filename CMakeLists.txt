# Use CMake 3.26 to take advantage of automatic CUDA language support.
cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

# Declare the project as a CUDA project.
project(skin LANGUAGES CXX CUDA)

enable_language(CUDA)

# Look for OpenCV.
find_package(OpenCV REQUIRED)

# Add the skin_detector library.
add_subdirectory(skin_detector)

# ── GPU architecture ───────────────────────────────────────────────────────────
# Set to match the target GPU's compute capability.
#   Pascal  10xx : 61   |  Turing  20xx : 75  |  Ampere  30xx : 86
#   Ada     40xx : 89   |  Blackwell 50xx : 120
# For multi-arch PTX (portable, slower compile):
#   set(CUDA_TARGET_ARCH "61;75;86;89;120")
set(CUDA_TARGET_ARCH "120")   # RTX 5060 Ti (Blackwell)

# ── Executables ───────────────────────────────────────────────────────────────

# skin: real-time webcam / video processing with double-buffered GPU pipeline.
add_executable(${PROJECT_NAME} skin.cu)
include_directories(${PROJECT_NAME} ${OpenCV_INCLUDE_DIRS} skin_detector_INCLUDE_DIRS)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} skin_detector)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES ${CUDA_TARGET_ARCH})

# skinMask: generate a binary skin mask from a single image.
set(SUBPROJECT skinMask)
add_executable(${SUBPROJECT} skinMask.cu)
include_directories(${SUBPROJECT} ${OpenCV_INCLUDE_DIRS} skin_detector_INCLUDE_DIRS)
target_link_libraries(${SUBPROJECT} ${OpenCV_LIBRARIES} skin_detector)
set_target_properties(${SUBPROJECT} PROPERTIES CUDA_ARCHITECTURES ${CUDA_TARGET_ARCH})

# skinMap: apply the skin map to a single image (zeros non-skin pixels).
set(SUBPROJECTMAP skinMap)
add_executable(${SUBPROJECTMAP} skinMap.cu)
include_directories(${SUBPROJECTMAP} ${OpenCV_INCLUDE_DIRS} skin_detector_INCLUDE_DIRS)
target_link_libraries(${SUBPROJECTMAP} ${OpenCV_LIBRARIES} skin_detector)
set_target_properties(${SUBPROJECTMAP} PROPERTIES CUDA_ARCHITECTURES ${CUDA_TARGET_ARCH})

# Copy test images to the build directory.
add_custom_target(copy-test-images ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/test_images ${CMAKE_BINARY_DIR}/test_images
    DEPENDS ${PROJECT_NAME}
)
