file(GLOB SOURCES "*.cpp" "*.h" "*.hpp" "*.cu" "*.cuh")

add_library(fluid_common_lib)

target_sources(fluid_common_lib PRIVATE ${SOURCES})

target_include_directories(fluid_common_lib PRIVATE ${FLUID_INCLUDE_DIR})
