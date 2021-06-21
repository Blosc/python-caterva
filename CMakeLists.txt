cmake_minimum_required(VERSION 3.11.0)
project(caterva)
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)

add_subdirectory(caterva)
