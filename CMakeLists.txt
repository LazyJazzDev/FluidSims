cmake_minimum_required(VERSION 3.22)

set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/external/GameX/external/grassland/external/vcpkg/scripts/buildsystems/vcpkg.cmake)

project(FluidSims LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_subdirectory(external/GameX)

add_subdirectory(src)
