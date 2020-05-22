include_directories(${NumPy_INCLUDE_DIRS})
include_directories("${CMAKE_SOURCE_DIR}/Caterva/caterva")
include_directories("${CMAKE_SOURCE_DIR}/c-blosc2/blosc")

add_cython_target(container_ext container_ext.pyx)
add_library(container_ext MODULE ${container_ext})

target_link_libraries(container_ext caterva_static)

python_extension_module(container_ext)

add_custom_command(
    TARGET container_ext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:container_ext> ${CMAKE_SOURCE_DIR}/cat4py
)

install(TARGETS container_ext LIBRARY DESTINATION cat4py)
