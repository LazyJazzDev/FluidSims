file(GLOB SOURCES "*.cpp" "*.h" "*.hpp" "*.cu" "*.cuh")

add_library(fluid_core)

target_sources(fluid_core PRIVATE ${SOURCES})

target_include_directories(fluid_core PRIVATE ${FLUID_INCLUDE_DIR})

find_package(glm CONFIG REQUIRED)
target_link_libraries(fluid_core PRIVATE glm::glm)

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(fluid_core PRIVATE Eigen3::Eigen)
