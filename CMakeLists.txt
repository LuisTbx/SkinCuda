# Use CMake 3.26 to take advange of automatic CUDA language 
cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

# Declare the project as a CUDA project
project(skin LANGUAGES CXX CUDA)

enable_language(CUDA)

# Look for OpenCv 
find_package(OpenCV REQUIRED)

# Add the skin_detector library
add_subdirectory(skin_detector)

# Add an executables
add_executable(${PROJECT_NAME} 
                skin.cu 
)

# Indicate include directories from opencv and skin_detector
include_directories(${PROJECT_NAME} 
                    ${OpenCV_INCLUDE_DIRS} 
                    skin_detector_INCLUDE_DIRS
)

# Include the libraries from skin_detector
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} skin_detector)

# Set the architecture, for NVIDIA series 10XX use 61
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "61")

set(SUBPROJECT skinMask)
add_executable(${SUBPROJECT} 
               skinMask.cu 
)

include_directories(${SUBPROJECT} 
                    ${OpenCV_INCLUDE_DIRS} 
                    skin_detector_INCLUDE_DIRS
)

target_link_libraries(${SUBPROJECT} ${OpenCV_LIBRARIES} skin_detector)
set_target_properties(${SUBPROJECT} PROPERTIES CUDA_ARCHITECTURES "61")

# Add custom command to copy images files
add_custom_target(copy-test-images ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/test_images ${CMAKE_BINARY_DIR}/test_images
    DEPENDS ${PROJECT_NAME}
)