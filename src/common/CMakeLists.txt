file(GLOB SOURCES "*.cpp" "*.h" "*.hpp" "*.cu" "*.cuh")

add_library(fluid_common_lib)

target_sources(fluid_common_lib PRIVATE ${SOURCES})
