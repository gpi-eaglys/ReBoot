cmake_minimum_required(VERSION 3.5.1)
project(reboot_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS})

set(PYBIND11_FINDPYTHON ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY /workspaces/ReBoot)

find_package(OpenFHE 1.2.0 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(pybind11 REQUIRED)

set(C_COMPILE_FLAGS "-Wall -Werror -Ofast -fopenmp -march=native")
set(CXX_COMPILE_FLAGS "-Wall -Werror -Ofast -fopenmp -march=native")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-O1 -Wl,--as-needed")

include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})

if(BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS"${OpenFHE_EXE_LINKER_FLAGS}-static")
    link_libraries(${OpenFHE_STATIC_LIBRARIES})
else()
    set(CMAKE_EXE_LINKER_FLAGS${OpenFHE_EXE_LINKER_FLAGS})
    link_libraries(${OpenFHE_SHARED_LIBRARIES})
endif()

include_directories(/workspaces/ReBoot/cpp/include)
include_directories(/workspaces/ReBoot/cpp/src)

set(SOURCES
    lib/encrypted_value.cpp
    lib/utils.cpp
    lib/nn.cpp
    lib/packing.cpp
)

pybind11_add_module(reboot_cpp
    bindings.cpp
    ${SOURCES}
)