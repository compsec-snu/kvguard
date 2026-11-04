cmake_minimum_required(VERSION 3.20)
project(kvg_interposer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_path(CUDA_INCLUDE_DIR cuda.h HINTS $ENV{CUDA_HOME}/include /usr/local/cuda/include)
if(NOT CUDA_INCLUDE_DIR)
  message(FATAL_ERROR "cuda.h not found; set CUDA_HOME")
endif()

add_library(kvg_interposer SHARED src/kvg_interposer.cpp)
target_include_directories(kvg_interposer PRIVATE ${CUDA_INCLUDE_DIR})
target_compile_options(kvg_interposer PRIVATE -Wall -Wextra -fvisibility=hidden)
target_link_libraries(kvg_interposer PRIVATE ${CMAKE_DL_LIBS})
