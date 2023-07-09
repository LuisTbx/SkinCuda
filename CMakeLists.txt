cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(skin LANGUAGES CXX CUDA)

enable_language(CUDA)

find_package(OpenCV REQUIRED)

add_subdirectory(skin_detector)

add_executable(${PROJECT_NAME} 
                skin.cu 
)

include_directories(${PROJECT_NAME} 
                    ${OpenCV_INCLUDE_DIRS} 
                    skin_detector_INCLUDE_DIRS
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} skin_detector)
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

add_custom_target(copy-test-images ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/test_images ${CMAKE_BINARY_DIR}/test_images
    DEPENDS ${PROJECT_NAME}
)