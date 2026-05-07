cmake_minimum_required(VERSION 3.14) # FetchContent works best with 3.14+
project(isvd)

# 1. Boilerplate: Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Setup FetchContent
include(FetchContent)

# --- Download Eigen ---
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0 # Stable version
)
FetchContent_MakeAvailable(eigen)

# --- Download pybind11 ---
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.0 # Stable version
)
FetchContent_MakeAvailable(pybind11)

# 3. Create the Python Module
# The module name must match the name in your wrapper.cpp (PYBIND11_MODULE)
pybind11_add_module(isvd_cpp src/wrapper.cpp)

# 4. Tell the compiler where to find YOUR headers and the libraries
target_include_directories(isvd_cpp PRIVATE include)

# Link against Eigen and pybind11
# Eigen3::Eigen is the standard "target" name provided by the Eigen maintainers
target_link_libraries(isvd_cpp PRIVATE Eigen3::Eigen)
