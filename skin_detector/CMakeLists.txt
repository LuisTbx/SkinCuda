project(skin_detector)

set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/hdrs)

enable_language(CUDA)

file(GLOB skin_src
    ${SOURCES}/*.cpp
    ${SOURCES}/*.cu
)

file(GLOB skin_hdrs
    ${HEADERS}/*.h
    ${HEADERS}/*.cuh
)

# Build the skin_detector shared library.
add_library(${PROJECT_NAME} ${skin_src})

# Generate the Windows DLL export header automatically.
include(GenerateExportHeader)
GENERATE_EXPORT_HEADER(${PROJECT_NAME}
    BASE_NAME        ${PROJECT_NAME}
    EXPORT_MACRO_NAME ${PROJECT_NAME}_EXPORTS
    EXPORT_FILE_NAME  ${PROJECT_NAME}_EXPORTS.h
    STATIC_DEFINE     ${PROJECT_NAME}_EXPORTS_BUILT_AS_STATIC
)

# Include directories.
include_directories(${PROJECT_NAME}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_NAME}
    ${HEADERS}
)

target_include_directories(${PROJECT_NAME} PUBLIC ${HEADERS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

target_link_directories(${PROJECT_NAME} PUBLIC ${HEADERS})

# ── GPU architecture ───────────────────────────────────────────────────────────
# Must match the CUDA_TARGET_ARCH set in the root CMakeLists.txt.
#   Pascal  10xx : 61   |  Turing  20xx : 75  |  Ampere  30xx : 86
#   Ada     40xx : 89   |  Blackwell 50xx : 120
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "120")
