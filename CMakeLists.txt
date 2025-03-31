cmake_minimum_required(VERSION 3.5)
project(superpointglue)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "release")
add_definitions(-w)

add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/tensorrtbuffer)


find_package(OpenCV 4.5 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories("/root/TensorRT-8.5.1.7/include/")
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)

add_library(${PROJECT_NAME}_lib SHARED
  src/super_point.cpp
  src/super_glue.cpp
)

target_link_libraries(${PROJECT_NAME}_lib
  /root/TensorRT-8.5.1.7/lib/libnvinfer.so
  /root/TensorRT-8.5.1.7/lib/libnvonnxparser.so
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  yaml-cpp
  tensorrtbuffer
)

add_executable(feature_matching feature_matching.cpp)
add_executable(${PROJECT_NAME}_image inference_image.cpp)
add_executable(${PROJECT_NAME}_sequence inference_sequence.cpp)

link_directories("/root/TensorRT-8.5.1.7/lib/")
target_link_libraries(${PROJECT_NAME}_image ${PROJECT_NAME}_lib)
target_link_libraries(${PROJECT_NAME}_sequence  ${PROJECT_NAME}_lib)
target_link_libraries(feature_matching ${OpenCV_LIBS})