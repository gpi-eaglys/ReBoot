cmake_minimum_required(VERSION 3.28)
project(ReBoot)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Installed binaries (e.g. reboot_py copied into site-packages) need an RPATH
# pointing at OpenFHE's shared libs, same as the build tree already gets via
# CMake's automatic build-RPATH. There's no install-time equivalent unless
# this is enabled explicitly.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

option(BUILD_PYTHON_BINDINGS
        "Build the reboot_py Python extension module" ON)

option(BUILD_TESTS
        "Build tests for reboot extension module" ON)

add_subdirectory(cpp/core)

if(BUILD_TESTS)
    message(STATUS "BUILD_TESTS: enabled - building tests for reboot extension module")
    enable_testing()
    add_subdirectory(cpp/tests)
endif()

if(BUILD_PYTHON_BINDINGS)
    message(STATUS "BUILD_PYTHON_BINDINGS: enabled - building Python bindings for reboot extension module")
    add_subdirectory(cpp/python)
endif()


