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

# If OpenFHE isn't already installed on the system, build it from the
# extern/openfhe-development submodule and point find_package() at that
# install instead - so this project builds standalone with no pre-existing
# OpenFHE install required.
find_package(OpenFHE 1.2.0 QUIET)
if(NOT OpenFHE_FOUND)
    set(_openfhe_src "${CMAKE_SOURCE_DIR}/extern/openfhe-development")
    set(_openfhe_build "${CMAKE_SOURCE_DIR}/build/extern/openfhe-development")
    set(_openfhe_install "${CMAKE_SOURCE_DIR}/build/extern/openfhe-development-install")

    if(NOT EXISTS "${_openfhe_src}/CMakeLists.txt")
        message(FATAL_ERROR "OpenFHE not found on the system, and extern/openfhe-development is empty - run 'git submodule update --init --recursive'")
    endif()

    if(NOT EXISTS "${_openfhe_install}/lib/OpenFHE/OpenFHEConfig.cmake")
        message(STATUS "OpenFHE not found on the system - building the vendored copy in extern/openfhe-development (first time only, this can take a while)")
        execute_process(
                COMMAND ${CMAKE_COMMAND} -S "${_openfhe_src}" -B "${_openfhe_build}"
                -DCMAKE_INSTALL_PREFIX=${_openfhe_install}
                -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
                -DCMAKE_BUILD_TYPE=Release
                RESULT_VARIABLE _openfhe_cfg_result)
        if(NOT _openfhe_cfg_result EQUAL 0)
            message(FATAL_ERROR "Failed to configure vendored OpenFHE")
        endif()

        include(ProcessorCount)
        ProcessorCount(_openfhe_nproc)
        execute_process(
                COMMAND ${CMAKE_COMMAND} --build "${_openfhe_build}" --target install -j ${_openfhe_nproc}
                RESULT_VARIABLE _openfhe_build_result)
        if(NOT _openfhe_build_result EQUAL 0)
            message(FATAL_ERROR "Failed to build vendored OpenFHE")
        endif()
    endif()

    set(OpenFHE_DIR "${_openfhe_install}/lib/OpenFHE" CACHE PATH "Vendored OpenFHE install" FORCE)
endif()

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


