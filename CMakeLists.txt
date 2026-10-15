cmake_minimum_required(VERSION 3.18)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the CUDA compiler path or name

project(llama_infer CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LLAMA3_SUPPORT "Build the LLAMA3 model variant" OFF)
if (LLAMA3_SUPPORT)
  message(STATUS "LLAMA3 SUPPORT")
  add_definitions(-DLLAMA3_SUPPORT)
endif()
option(QWEN2_SUPPORT "Build the QWEN2 model variant" OFF)
if (QWEN2_SUPPORT)
  message(STATUS "QWEN2 SUPPORT")
  add_definitions(-DQWEN2_SUPPORT)
endif()

option(QWEN3_SUPPORT "Build the QWEN3 model variant" OFF)
if (QWEN3_SUPPORT)
  message(STATUS "QWEN3 SUPPORT")
  add_definitions(-DQWEN3_SUPPORT)
endif()

set(model_variant_count 0)
foreach(flag LLAMA3_SUPPORT QWEN2_SUPPORT QWEN3_SUPPORT)
  if(${flag})
    math(EXPR model_variant_count "${model_variant_count} + 1")
  endif()
endforeach()
if(model_variant_count GREATER 1)
  message(FATAL_ERROR "Select only one model family per build directory")
endif()
# ---- Add dependencies via FetchContent ----
include(FetchContent)

# Google Test
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.0.zip
)
FetchContent_MakeAvailable(googletest)

# glog
set(BUILD_TESTING OFF CACHE BOOL "Build testing" FORCE)
set(WITH_GFLAGS OFF CACHE BOOL "Build with gflags" FORCE)
FetchContent_Declare(
  glog
  URL https://github.com/google/glog/archive/refs/tags/v0.7.1.zip
)
FetchContent_MakeAvailable(glog)

# Armadillo
FetchContent_Declare(
  armadillo
  GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
  GIT_TAG 14.0.1
)
FetchContent_MakeAvailable(armadillo)

# sentencepiece
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_Declare(
  sentencepiece
  URL https://github.com/google/sentencepiece/archive/refs/tags/v0.2.0.zip
)
FetchContent_MakeAvailable(sentencepiece)
unset(CMAKE_POLICY_VERSION_MINIMUM)

if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
  # absl
  set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "Propagate CXX std" FORCE)
  set(ABSL_ENABLE_INSTALL ON CACHE BOOL "Enable install" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "Build testing" FORCE)
  FetchContent_Declare(
    absl
    URL https://github.com/abseil/abseil-cpp/archive/refs/tags/20240722.0.zip
  )
  FetchContent_MakeAvailable(absl)

  # re2
  FetchContent_Declare(
    re2
    URL https://github.com/google/re2/archive/refs/tags/2024-07-02.zip
  )
  FetchContent_MakeAvailable(re2)

  # nlohmann_json
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# ---------------------------------
# Removed find_package calls as we are building from source now

aux_source_directory(source/model DIR_MODEL)
aux_source_directory(source/sampler DIR_SAMPLER)
aux_source_directory(source/runtime DIR_RUNTIME)
aux_source_directory(source/base/ DIR_BASE)
aux_source_directory(source/tensor DIR_TENSOR)
aux_source_directory(source/op  DIR_OP)
aux_source_directory(source/op/kernels/cpu DIR_KERNEL_CPU)
aux_source_directory(source/op/kernels/cuda DIR_KERNEL_CUDA)
aux_source_directory(source/op/kernels/ DIR_KERNEL)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
add_library(llama SHARED ${DIR_BASE} ${DIR_TENSOR} ${DIR_OP} ${DIR_KERNEL_CPU} ${DIR_KERNEL_CUDA} ${DIR_KERNEL} ${DIR_MODEL} ${DIR_SAMPLER} ${DIR_RUNTIME})

# Link libraries
target_link_libraries(llama
  PUBLIC
  sentencepiece
  glog::glog
  pthread
  CUDA::cudart
  CUDA::cublas
  armadillo
)

if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
  target_link_libraries(llama PUBLIC absl::base re2::re2 nlohmann_json::nlohmann_json)
endif()

target_link_directories(llama PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

# Include directories
# Some libraries might not expose include dirs properly via interface, so we add them if needed.
# glog and gtest usually do.
# Armadillo and sentencepiece might need help if not fully modern-cmake compliant in their older versions or specific configs.

if(glog_SOURCE_DIR)
  target_include_directories(llama PUBLIC ${glog_SOURCE_DIR}/src) # Fallback if needed, though glog::glog should handle it
endif()

if(Armadillo_SOURCE_DIR)
    target_include_directories(llama PUBLIC ${Armadillo_SOURCE_DIR}/include)
endif()

if(sentencepiece_SOURCE_DIR)
  target_include_directories(llama PUBLIC ${sentencepiece_SOURCE_DIR}/src)
endif()

target_include_directories(llama PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(llama PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
enable_testing()
add_subdirectory(test)
add_subdirectory(demo)
