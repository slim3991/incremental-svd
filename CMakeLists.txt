cmake_minimum_required(VERSION 3.15)

project(incremental_svd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

add_subdirectory(extern/pybind11)

# Build the Python extension module
pybind11_add_module(incremental_svd_lib
    src/wrapper.cpp
)

# Include directories
target_include_directories(incremental_svd_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/extern
)

# Optional compiler optimizations
if(MSVC)
    target_compile_options(incremental_svd_lib PRIVATE /O2)
else()
    target_compile_options(incremental_svd_lib PRIVATE -O3)
endif()

# Install extension into Python package
install(TARGETS incremental_svd_lib LIBRARY DESTINATION isvd)
