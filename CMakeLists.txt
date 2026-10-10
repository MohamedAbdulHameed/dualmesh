cmake_minimum_required(VERSION 3.18)
project(dualmesh VERSION 0.2.0 LANGUAGES CXX)

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
option(DUALMESH_ENABLE_PETSC "Use PETSc for linear solves (linear_solver='petsc'); always on with MPI" OFF)
option(DUALMESH_ENABLE_GSLIB "With MPI, build gslib (Nek5000's gather-scatter) as a second gather-scatter next to PETSc's" ON)
# Automatic differentiation seeds one derivative slot per local degree of
# freedom of an element (one per node and variable; the corners only for a
# first-order variable), so the default of 96 covers, for example, Quad9 with
# ten fields, Hex8 with twelve, Hex27 with three, and the Taylor-Hood element
# on Hex27 (three quadratic velocity components and a linear pressure, 89
# slots).  An ADReal copies only the slots in use, so the size of the array
# costs neither assembly time nor memory in practice: the 20 x 20 x 20 Hex8
# elasticity benchmark (tests/cpp/benchmark.cpp) runs in the same time and
# peak memory with 48 and with 96.
set(DUALMESH_MAX_AD_DERIVATIVES 96 CACHE STRING
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


if(DUALMESH_ENABLE_MPI)
  find_package(MPI COMPONENTS CXX REQUIRED)
  if(DUALMESH_ENABLE_GSLIB)
    enable_language(C)
    find_package(MPI COMPONENTS C REQUIRED)
  endif()
  # The distributed solver communicates through PETSc (its star forests) and solves with PETSc's Krylov methods and preconditioners, so an MPI build needs PETSc.
  set(DUALMESH_ENABLE_PETSC ON CACHE BOOL "" FORCE)
endif()

# PETSc is found with pkg-config, which every PETSc installation provides
# (PETSC_DIR/PETSC_ARCH/lib/pkgconfig for a source build; set PKG_CONFIG_PATH
# to it).  PETSc is itself built on MPI, and must use the same MPI as the
# distributed solver.
if(DUALMESH_ENABLE_PETSC)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)
  message(STATUS "PETSc found: ${PETSC_VERSION}")
endif()

# gslib, the gather-scatter library of Nek5000 (BSD license with the DOE notice), is downloaded at a pinned release and built with its own Makefile.
# It is compiled as C99 and reached through the C file src/parallel/GslibShim.c, because its headers redefine inline and restrict outside C99.
if(DUALMESH_ENABLE_MPI AND DUALMESH_ENABLE_GSLIB)
  include(ExternalProject)
  set(GSLIB_PREFIX ${CMAKE_BINARY_DIR}/_deps/gslib)
  ExternalProject_Add(
    gslib_external
    URL https://github.com/Nek5000/gslib/archive/refs/tags/v1.0.9.tar.gz
    URL_HASH SHA256=572c78b1ca4ddda75ec85c6299ee0597459f4dbf5170b9faa6de4c435dcdc80b
    PREFIX ${GSLIB_PREFIX}
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make CC=${MPI_C_COMPILER} MPI=1 NBC=1 "CFLAGS=-O2 -fPIC" DESTDIR=${GSLIB_PREFIX}/install install
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${GSLIB_PREFIX}/install/lib/libgs.a)
  file(MAKE_DIRECTORY ${GSLIB_PREFIX}/install/include)
  add_library(gslib STATIC IMPORTED)
  set_target_properties(gslib PROPERTIES IMPORTED_LOCATION ${GSLIB_PREFIX}/install/lib/libgs.a
                                         INTERFACE_INCLUDE_DIRECTORIES ${GSLIB_PREFIX}/install/include)
  add_dependencies(gslib gslib_external)
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
if(DUALMESH_ENABLE_MPI)
  target_link_libraries(dualmesh_core PUBLIC MPI::MPI_CXX)
  target_compile_definitions(dualmesh_core PUBLIC DUALMESH_HAVE_MPI)
endif()
if(DUALMESH_ENABLE_PETSC)
  target_link_libraries(dualmesh_core PUBLIC PkgConfig::PETSC)
  target_compile_definitions(dualmesh_core PUBLIC DUALMESH_HAVE_PETSC)
endif()
if(DUALMESH_ENABLE_MPI AND DUALMESH_ENABLE_GSLIB)
  target_sources(dualmesh_core PRIVATE ${PROJECT_SOURCE_DIR}/src/parallel/GslibShim.c)
  set_source_files_properties(${PROJECT_SOURCE_DIR}/src/parallel/GslibShim.c PROPERTIES LANGUAGE C)
  target_link_libraries(dualmesh_core PRIVATE gslib MPI::MPI_C)
  target_compile_definitions(dualmesh_core PRIVATE DUALMESH_HAVE_GSLIB)
endif()

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
