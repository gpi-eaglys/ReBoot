include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

add_executable(reboot_core_tests
        test_scaffold.cpp
        test_matrix_add.cpp
        test_matrix_mul.cpp
)

target_link_libraries(reboot_core_tests
        PRIVATE
        reboot_core
        GTest::gtest_main
)
set_target_properties(reboot_core_tests
        PROPERTIES BUILD_RPATH "${OpenFHE_LIBDIR}")
target_link_options(reboot_core_tests PRIVATE -Wl,-O1 -Wl,--as-needed)

include(GoogleTest)
gtest_discover_tests(reboot_core_tests)
