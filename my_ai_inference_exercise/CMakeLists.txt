cmake_minimum_required(VERSION 3.26)

# check CUDA
include(CheckLanguage)
check_language(CUDA)

# all use standard 17
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")
# set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(maie LANGUAGES CUDA CXX)

set(ONNXRUNTIME_DIR /workspace/opensource/onnxruntime)
set(TensorRT_DIR /workspace/opensource/TensorRT)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(TensorRT_INCLUDE_DIRS ${TensorRT_DIR}/include ${TensorRT_DIR}/parsers/onnx)
set(TensorRT_LIB_DIRS ${TensorRT_DIR}/out)
set(onnxruntime_INCLUDE_DIRS ${ONNXRUNTIME_DIR}/include ${ONNXRUNTIME_DIR}/include/onnxruntime/core/session)
set(onnxruntime_LIB_DIRS ${ONNXRUNTIME_DIR}/build/Linux/RelWithDebInfo)

aux_source_directory(./source MAIE_SRC_DIR)


add_library(maie SHARED ${MAIE_SRC_DIR})
target_include_directories(maie PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${onnxruntime_INCLUDE_DIRS})
target_link_directories(maie PUBLIC ${onnxruntime_LIB_DIRS} ${TensorRT_LIB_DIRS})
target_link_libraries(maie PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} onnxruntime nvinfer nvonnxparser cudart)


add_executable(detectresnet50 ./examples/imagedetect_resnet50/main.cxx)
target_include_directories(detectresnet50 PUBLIC ./third_party/nlohmann/include ./inlude)
target_link_directories(detectresnet50 PUBLIC ./build)
target_link_libraries(detectresnet50 PUBLIC maie)


add_executable(detectyolov8 ./examples/imagedetect_yolov8/main.cxx)
target_include_directories(detectyolov8 PUBLIC ./inlude)
target_link_directories(detectyolov8 PUBLIC ./build)
target_link_libraries(detectyolov8 PUBLIC maie)
