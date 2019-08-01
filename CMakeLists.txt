cmake_minimum_required(VERSION 3.11.0)
project(cat4py)
find_package(PythonExtensions REQUIRED)

# For building C-Blosc2
set(BUILD_STATIC ON CACHE BOOL "Build a static version of the blosc2 library.")
set(BUILD_SHARED OFF CACHE BOOL "Build a shared library version of the blosc2 library.")
set(BUILD_TESTS OFF CACHE BOOL "Build test programs form the blosc2 compression library")
set(BUILD_BENCHMARKS OFF CACHE BOOL "Build benchmark programs form the blosc2 compression library")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(c-blosc2)

# For building Caterva
include_directories(c-blosc2/blosc)
set(BLOSC_LIB blosc_static) # required for caterva tests
add_subdirectory(Caterva)

# For building extension
include_directories(Caterva/caterva)
add_library(container_ext MODULE cat4py/container_ext.c)
target_link_libraries(container_ext blosc_static)
python_extension_module(container_ext)
install(TARGETS container_ext LIBRARY DESTINATION cat4py)
