cmake_minimum_required(VERSION 3.15)
project(isvd_project LANGUAGES CXX)

# Instead of add_subdirectory, find the version installed via pip
find_package(pybind11 CONFIG REQUIRED)

# Setup Eigen (Keep this as you have it)
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/Eigen")

# Define the module
pybind11_add_module(incremental_svd_lib src/wrapper.cpp)

target_include_directories(incremental_svd_lib PRIVATE 
    include 
    extern
)
