set(STATIC_LIB ON CACHE BOOL "Build a static version of the blosc library.")
set(SHARED_LIB ON CACHE BOOL "Build a shared library version of the blosc
    library.")
set(CATERVA_BUILD_TESTS OFF CACHE BOOL "Build Caterva tests")
set(CATERVA_BUILD_EXAMPLES OFF CACHE BOOL "Build Caterva examples")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(caterva)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/caterva/caterva")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/caterva/contribs/c-blosc2/include")


add_cython_target(caterva_ext caterva_ext.pyx)
add_library(caterva_ext MODULE ${caterva_ext})

target_link_libraries(caterva_ext caterva_static)

python_extension_module(caterva_ext)

add_custom_command(
    TARGET caterva_ext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:caterva_ext> ${CMAKE_SOURCE_DIR}/caterva
)

install(TARGETS caterva_ext LIBRARY DESTINATION caterva)
