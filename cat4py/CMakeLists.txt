include_directories(${NumPy_INCLUDE_DIRS})

set(STATIC_LIB ON CACHE BOOL "Build a static version of the blosc library.")
set(SHARED_LIB OFF CACHE BOOL "Build a shared library version of the blosc
    library.")
set(CATERVA_BUILD_TESTS OFF CACHE BOOL "Build Caterva tests")
set(CATERVA_BUILD_EXAMPLES OFF CACHE BOOL "Build Caterva examples")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(Caterva)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/Caterva/caterva")


add_cython_target(container_ext container_ext.pyx)
add_library(container_ext MODULE ${container_ext})

target_link_libraries(container_ext caterva_static)

python_extension_module(container_ext)

add_custom_command(
    TARGET container_ext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:container_ext> ${CMAKE_SOURCE_DIR}/cat4py
)

install(TARGETS container_ext LIBRARY DESTINATION cat4py)
