file(GLOB SOURCES "*.cpp" "*.h" "*.hpp" "*.cu" "*.cuh")

add_executable(fluid_gui)

target_sources(fluid_gui PRIVATE ${SOURCES})
