find_package(OpenFHE 1.2.0 REQUIRED)
find_package(OpenMP REQUIRED)


add_library(reboot_core STATIC
        src/encrypted_value.cpp
        src/utils.cpp
        src/nn.cpp
        src/packing.cpp
)

target_include_directories(reboot_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include

        ${OpenFHE_INCLUDE}
        ${OpenFHE_INCLUDE}/third-party/include
        ${OpenFHE_INCLUDE}/core
        ${OpenFHE_INCLUDE}/pke
        ${OpenFHE_INCLUDE}/binfhe
)

target_link_libraries(reboot_core PUBLIC
        OpenMP::OpenMP_CXX
        $<IF:$<BOOL:${BUILD_STATIC}>,${OpenFHE_STATIC_LIBRARIES},${OpenFHE_SHARED_LIBRARIES}>
)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reboot_core PRIVATE -Wall -Werror -Ofast)
endif()

set_target_properties(reboot_core PROPERTIES BUILD_RPATH "${OpenFHE_LIBDIR}")

# Propagated to whatever links reboot_core (the test executable, the Python module).
target_link_options(reboot_core INTERFACE -Wl,--disable-new-dtags)