file(GLOB SOURCES "*.cpp" "*.h" "*.hpp" "*.cu" "*.cuh")

add_library(fluid_apic_lib)

target_sources(fluid_apic_lib PRIVATE ${SOURCES})

target_include_directories(fluid_apic_lib PRIVATE ${FLUID_INCLUDE_DIR})
