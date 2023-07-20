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

# Add cuda library
add_library(${PROJECT_NAME} ${skin_src})

# Make the project compatible with windows
include (GenerateExportHeader)          
GENERATE_EXPORT_HEADER(${PROJECT_NAME}           # generates the export header ${PROJECT_NAME}_EXPORTS.h automatically
    BASE_NAME ${PROJECT_NAME}
    EXPORT_MACRO_NAME ${PROJECT_NAME}_EXPORTS
    EXPORT_FILE_NAME ${PROJECT_NAME}_EXPORTS.h
    STATIC_DEFINE ${PROJECT_NAME}_EXPORTS_BUILT_AS_STATIC
)


# Add include directory
include_directories(${PROJECT_NAME} 
                    ${OpenCV_INCLUDE_DIRS} 
                    ${PROJECT_NAME}  
                    ${HEADERS}
)

# Make public the header directory and headers
target_include_directories(${PROJECT_NAME} 
                           PUBLIC 
                           ${HEADERS}
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

target_link_directories(${PROJECT_NAME} 
                        PUBLIC 
                        ${HEADERS}
)


set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "61")
# set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CUDA)