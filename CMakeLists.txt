cmake_minimum_required(VERSION 3.18)
project(dualmesh VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(DUALMESH_BUILD_PYTHON "Build the Python extension module" ON)
option(DUALMESH_BUILD_TESTS "Build the C++ unit tests" ON)
option(DUALMESH_ENABLE_OPENMP "Thread the assembly loops with OpenMP" ON)
option(DUALMESH_ENABLE_MPI "Build the distributed (MPI) solver" OFF)
option(DUALMESH_ENABLE_METIS "Use METIS for mesh partitioning when available" ON)
# Automatic differentiation seeds one derivative slot per element node and
# variable, so the default of 48 covers, for example, Quad9 with five fields,
# Hex8 with six and Hex27 with one.  Raise it for a multi-field problem on a
# Hex27 mesh; every ADReal grows with it, so raising it costs memory and
# assembly time everywhere.
set(DUALMESH_MAX_AD_DERIVATIVES 48 CACHE STRING
    "Maximum seeded derivatives per element (nodes per element x variables)")

# ---- dependencies ---------------------------------------------------------
# Eigen is header-only.  A system installation is used when there is one
# (macOS: `brew install eigen`; Debian/Ubuntu: `apt install libeigen3-dev`);
# otherwise the headers are downloaded into the build tree.
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
  set(DUALMESH_EIGEN_VERSION 3.4.0)
  set(DUALMESH_EIGEN_DIR ${CMAKE_BINARY_DIR}/_deps/eigen-${DUALMESH_EIGEN_VERSION})
  if(NOT EXISTS ${DUALMESH_EIGEN_DIR}/Eigen/Dense)
    set(DUALMESH_EIGEN_ARCHIVE ${CMAKE_BINARY_DIR}/_deps/eigen.tar.gz)
    message(STATUS "Eigen not found; downloading Eigen ${DUALMESH_EIGEN_VERSION}")
    file(DOWNLOAD
      https://gitlab.com/libeigen/eigen/-/archive/${DUALMESH_EIGEN_VERSION}/eigen-${DUALMESH_EIGEN_VERSION}.tar.gz
      ${DUALMESH_EIGEN_ARCHIVE}
      EXPECTED_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
      STATUS DUALMESH_EIGEN_DOWNLOAD)
    list(GET DUALMESH_EIGEN_DOWNLOAD 0 DUALMESH_EIGEN_DOWNLOAD_CODE)
    if(NOT DUALMESH_EIGEN_DOWNLOAD_CODE EQUAL 0)
      list(GET DUALMESH_EIGEN_DOWNLOAD 1 DUALMESH_EIGEN_DOWNLOAD_MESSAGE)
      message(FATAL_ERROR
        "Could not download Eigen (${DUALMESH_EIGEN_DOWNLOAD_MESSAGE}).\n"
        "Install it instead — 'brew install eigen' on macOS, "
        "'apt install libeigen3-dev' on Debian or Ubuntu — or point CMake at an "
        "existing copy with -DEigen3_DIR=<path to the Eigen3 CMake package>.")
    endif()
    file(ARCHIVE_EXTRACT INPUT ${DUALMESH_EIGEN_ARCHIVE}
         DESTINATION ${CMAKE_BINARY_DIR}/_deps)
  endif()
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${DUALMESH_EIGEN_DIR})
endif()

# ---- optional parallel dependencies ------------------------------------------
# OpenMP threads the element assembly loop; it is optional, and without it the
# library builds and runs exactly as before, on one thread.
if(DUALMESH_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX QUIET)
  if(NOT OpenMP_CXX_FOUND)
    message(STATUS "OpenMP not found; the assembly will run on one thread")
  endif()
endif()

# METIS gives better partitions than the built-in geometric partitioner for
# unstructured meshes.  It is what libMesh, and therefore MOOSE, uses.
if(DUALMESH_ENABLE_METIS)
  find_path(METIS_INCLUDE_DIR metis.h)
  find_library(METIS_LIBRARY NAMES metis)
  if(METIS_INCLUDE_DIR AND METIS_LIBRARY)
    message(STATUS "METIS found: ${METIS_LIBRARY}")
  else()
    message(STATUS "METIS not found; using the built-in mesh partitioners")
  endif()
endif()

if(DUALMESH_ENABLE_MPI)
  find_package(MPI COMPONENTS CXX REQUIRED)
endif()

# ---- core library ------------------------------------------------------------
file(GLOB_RECURSE DUALMESH_SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_library(dualmesh_core STATIC ${DUALMESH_SOURCES})
target_include_directories(dualmesh_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(dualmesh_core PUBLIC Eigen3::Eigen)
target_compile_definitions(dualmesh_core
                           PUBLIC DUALMESH_MAX_AD_DERIVATIVES=${DUALMESH_MAX_AD_DERIVATIVES})
if(MSVC)
  target_compile_options(dualmesh_core PRIVATE /W3 /bigobj)
  target_compile_definitions(dualmesh_core PUBLIC _USE_MATH_DEFINES)
else()
  target_compile_options(dualmesh_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
if(DUALMESH_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(dualmesh_core PUBLIC OpenMP::OpenMP_CXX)
endif()
if(DUALMESH_ENABLE_METIS AND METIS_INCLUDE_DIR AND METIS_LIBRARY)
  target_include_directories(dualmesh_core PRIVATE ${METIS_INCLUDE_DIR})
  target_link_libraries(dualmesh_core PRIVATE ${METIS_LIBRARY})
  target_compile_definitions(dualmesh_core PRIVATE DUALMESH_HAVE_METIS)
endif()
if(DUALMESH_ENABLE_MPI)
  target_link_libraries(dualmesh_core PUBLIC MPI::MPI_CXX)
  target_compile_definitions(dualmesh_core PUBLIC DUALMESH_HAVE_MPI)
endif()

# ---- command-line driver --------------------------------------------------------
# (the Python package provides the full input-file driver: `dualmesh run input.yaml`)

# ---- tests ---------------------------------------------------------------------
if(DUALMESH_BUILD_TESTS)
  enable_testing()
  add_executable(dualmesh_unit_tests tests/cpp/unit_tests.cpp)
  target_link_libraries(dualmesh_unit_tests PRIVATE dualmesh_core)
  add_test(NAME unit_tests COMMAND dualmesh_unit_tests)
  add_executable(dualmesh_benchmark tests/cpp/benchmark.cpp)
  target_link_libraries(dualmesh_benchmark PRIVATE dualmesh_core)
  add_executable(dualmesh_parallel_tests tests/cpp/parallel_tests.cpp)
  target_link_libraries(dualmesh_parallel_tests PRIVATE dualmesh_core)
  add_test(NAME parallel_tests COMMAND dualmesh_parallel_tests)
endif()

# ---- Python module ---------------------------------------------------------------
if(DUALMESH_BUILD_PYTHON)
  find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
                    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_package(pybind11 CONFIG REQUIRED)
  endif()
  pybind11_add_module(_core MODULE python/bindings/module.cpp)
  target_link_libraries(_core PRIVATE dualmesh_core)
  if(SKBUILD)
    install(TARGETS _core LIBRARY DESTINATION dualmesh)
  else()
    set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/dualmesh)
  endif()
endif()
