# --- reboot_py: the Python binding module ---

set(PYBIND11_FINDPYTHON ON)

# Prefer the project's uv-managed venv over the system interpreter, unless the
# caller already pinned one (-DPython3_EXECUTABLE=..., -DPython3_ROOT_DIR=...,
# or an activated VIRTUAL_ENV).
if(NOT Python3_EXECUTABLE AND NOT Python3_ROOT_DIR AND NOT DEFINED ENV{VIRTUAL_ENV})
    foreach(_venv_candidate "${CMAKE_SOURCE_DIR}/.venv" "${CMAKE_SOURCE_DIR}/../.venv")
        if(EXISTS "${_venv_candidate}/bin/python3")
            set(Python3_ROOT_DIR "${_venv_candidate}")
            break()
        endif()
    endforeach()
endif()
# Provide hint for PyBind11 where to find package
# cf. : https://cmake.org/cmake/help/latest/module/FindPython3.html#hints
set(Python3_FIND_VIRTUALENV FIRST)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

if(NOT pybind11_DIR)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

find_package(pybind11 REQUIRED)

pybind11_add_module(reboot_py
    bindings.cpp
)

target_link_libraries(reboot_py PRIVATE reboot_core)

install(TARGETS reboot_py DESTINATION .)
