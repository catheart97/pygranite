cmake_minimum_required(VERSION 3.12)
project(pygranite LANGUAGES CXX)

find_package(CUDA 11.2 REQUIRED)

set(CUDA_STANDARD 17)
set(CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CUDA_PROPAGATE_HOST_FLAGS ON)

set(CMAKE_VERBOSE_MAKEFILE ON)
 
add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
find_package(PythonInterp 3.6 REQUIRED) 
find_package(PythonLibs 3.6 REQUIRED)

option (CODEGEN_SHORTEN "Reduce compilation time by limiting the number of kernels! Only for build system tests!" OFF) 

if (CODEGEN_SHORTEN)
    set (CODEGEN_ARG "1")
else ()
    set (CODEGEN_ARG "0")
endif()

set (CODEGEN_SCRIPT "source/granite/TrajectoryIntegrator.gen.py")
execute_process(
                  COMMAND ${PYTHON_EXECUTABLE} ${CODEGEN_SCRIPT} ${CODEGEN_ARG}
                  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                  RESULT_VARIABLE CODEGEN_RES
               )
message(STATUS "Code Generation Result: ${CODEGEN_RES}")

set(CMAKE_BUILD_TYPE Release)

add_subdirectory(third_party/pybind11)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "--std=c++17 -O3 -Xptxas=--suppress-stack-size-warning" )
if (WIN32)
    set(CMAKE_CXX_FLAGS "/MD /permissive /std:c++17 /O2 /EHsc /Ob2 /bigobj /Zc:__cplusplus /MP8")
    set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS})
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-Xcompiler /MD -Xcompiler /permissive -Xcompiler /MP8" )
else ()
    set(CMAKE_CXX_FLAGS "-O3 -fvisibility=hidden")
    set(CMAKE_MAKE_PROGRAM "${CMAKE_MAKE_PROGRAM} -j${MULTITHREADED_BUILD}")
endif()

include_directories(
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${PYTHON_INCLUDE_DIRS}
    third_party/pybind11/include
    third_party/cpp-peglib
    third_party/catch2/extras
    source
)

set(HEADERS
    source/granite/Language.hpp
    source/granite/SimulationData.hpp
    source/granite/Constants.hpp
    source/granite/IntegratorSettings.hpp
    source/granite/TrajectoryIntegrator.hpp
    source/granite/TrajectoryIntegratorCompute.hpp
    source/granite/TrajectorySet.hpp
    source/granite/DataLoader.hpp
    source/granite/Info.hpp
    source/granite/TextureInfo.hpp
    source/my/math/LinearAlgebra.hpp
    source/my/math/Constants.hpp
    source/my/math/Arithmetic.hpp
    source/my/util/Config.hpp
    source/my/util/Util.cuh
    source/my/util/SwapChain.hpp
    source/my/cuda/ToolKernel.cuh
    source/my/cuda/ToolKernel.tpp
    source/pygranite.hpp
)

file(GLOB TRAJECTORY_INTEGRATOR_COMPUTE_CU_FILES 
     ${PROJECT_SOURCE_DIR}/source/granite/TrajectoryIntegratorCompute/*.cu)

set(SOURCES
    source/pygranite.cpp
    source/granite/TrajectoryIntegrator.cpp
    source/granite/DataLoader.cpp
    source/granite/TrajectorySet.cpp
    source/granite/TrajectoryIntegratorCompute.cpp
    ${TRAJECTORY_INTEGRATOR_COMPUTE_CU_FILES}
)

cuda_add_library(pygranite SHARED ${SOURCES})


set_property(TARGET pygranite PROPERTY CUDA_ARCHITECTURES 61 62 72 75)

# ! theres a bug in pybind11 on Windows which causes compilation to fail
#   if debug python libraries and non debug libraries are available
target_link_libraries(pygranite pybind11::module)

set_target_properties(pygranite PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
    CUDA_SEPARABLE_COMPILATION ON
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
