cmake_minimum_required(VERSION 3.12)
project(elite_crop LANGUAGES CXX CUDA)

# RTX 4080용 아키텍쳐
cmake_policy(SET CMP0104 NEW)  
set(CMAKE_CUDA_ARCHITECTURES 89) 

# === [Pybind11 설정] ===
execute_process(
  COMMAND python3 -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

# === [Include 디렉토리 설정] ===
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/elite/cuda/crop)
include_directories(${CMAKE_SOURCE_DIR}/pybind)
include_directories(${CMAKE_SOURCE_DIR}/src/utils/crop)


# miniconda용 Python 헤더도 포함
include_directories(
  /home/hjkwon/miniconda3/include/python3.13
  /home/hjkwon/miniconda3/lib/python3.13/site-packages/pybind11/include
)

# === [CUDA 소스파일 강제 지정] ===
set_source_files_properties(src/cuda/crop_aabb.cu PROPERTIES LANGUAGE CUDA)

# === [라이브러리 추가] ===
add_library(crop_utils STATIC
    src/utils/crop/crop_utils.cpp
    src/utils/crop/crop_aabb.cu
)

# === [Pybind 모듈 추가] ===
pybind11_add_module(elite_crop pybind/bindings.cpp)
target_link_libraries(elite_crop PRIVATE crop_utils)
