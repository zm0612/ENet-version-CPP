cmake_minimum_required(VERSION 2.8)

project(segmentation)

add_definitions(-std=c++11)

find_package(OpenCV 3.4 REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
)


add_library(${PROJECT_NAME} SHARED
        src/SemanticSeg.cc
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/example)
add_executable(segment example/segmentation.cc)
target_link_libraries(segment ${OpenCV_LIBS} ${PROJECT_NAME})