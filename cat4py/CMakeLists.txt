include_directories(${NumPy_INCLUDE_DIRS})
include_directories("${CMAKE_SOURCE_DIR}/Caterva/caterva")
include_directories("${CMAKE_SOURCE_DIR}/c-blosc2/blosc")

add_cython_target(container_ext CXX)
add_library(container_ext MODULE ${container_ext})

target_link_libraries(container_ext caterva_static)

python_extension_module(container_ext)

install(TARGETS container_ext LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/cat4py)
