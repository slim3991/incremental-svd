cmake_minimum_required(VERSION 3.15)
project(isvd_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Subdirectories
add_subdirectory(extern/pybind11)

# 2. Define the module
pybind11_add_module(incremental_svd_lib src/wrapper.cpp)

# 3. Include paths
target_include_directories(incremental_svd_lib PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/extern # Point directly to Eigen
)

# 4. Install rule
# This will move the .so file from /build into the /isvd folder
install(TARGETS incremental_svd_lib DESTINATION .)
