file(GLOB SOURCES "*.cpp" "*.h" "*.hpp" "*.cu" "*.cuh")

add_executable(fluid_gui)

target_sources(fluid_gui PRIVATE ${SOURCES})

target_include_directories(fluid_gui PRIVATE ${FLUID_INCLUDE_DIR})

target_link_libraries(fluid_gui PRIVATE GameX)

#PACK_SHADER_CODE(fluid_gui)
