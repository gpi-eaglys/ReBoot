OPENFHE_DIR ?=
OPENFHE_DIR_FLAG = $(if $(OPENFHE_DIR),-DOpenFHE_DIR=$(OPENFHE_DIR),)
NPROC := $(shell nproc)
BUILD_TYPE ?= Debug
BUILD_DIR := $(CURDIR)/build/cmake-build-$(shell echo $(BUILD_TYPE) | tr '[:upper:]' '[:lower:]')
GENERATOR ?= Ninja

.PHONY: all lib build-tests tests tidy clean dep_openfhe py py-openfhe

all: tests

# Ensures the OpenFHE/OpenFHE-Python submodules are checked out.
# If OPENFHE_DIR isn't set, CMake builds OpenFHE from extern/openfhe-development
# itself (falling back to it only if no system OpenFHE is found) - this target
# just guarantees the submodule source is present for that to work.
dep_openfhe:
	@echo "[INFO] Ensures that OpenFHE/OpenFHE-Python submodules are checked out"
	git submodule update --init --recursive extern/openfhe-development extern/openfhe-python

lib: dep_openfhe
	cmake $(OPENFHE_DIR_FLAG) -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -G "$(GENERATOR)" -S . -B "$(BUILD_DIR)"
	cmake --build "$(BUILD_DIR)" -j$(NPROC) --target reboot_core

build-tests: dep_openfhe
	cmake $(OPENFHE_DIR_FLAG) \
		  -DBUILD_TESTS=ON \
          -DBUILD_PYTHON_BINDINGS=OFF \
          -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -G "$(GENERATOR)" -S . -B "$(BUILD_DIR)"
	cmake --build "$(BUILD_DIR)" -j$(NPROC) --target reboot_core_tests

tests: build-tests
	ctest --test-dir "$(BUILD_DIR)" --output-on-failure

tidy: dep_openfhe
	cmake $(OPENFHE_DIR_FLAG) -DBUILD_PYTHON_BINDINGS=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G "$(GENERATOR)" -S . -B "$(BUILD_DIR)"
	run-clang-tidy -p "$(BUILD_DIR)" -j$(NPROC) -quiet 'cpp/(core|tests)/.*'

py:
	@echo "No Python target! Call: 'uv pip install -e .'"

# Builds & installs the official openfhe Python bindings (extern/openfhe-python)
# against the vendored OpenFHE C++ install, into the project's uv venv.
py-openfhe: lib
	@echo "[INFO] Build & install the official openFHE Python bindings to: extern/openfhe-python"
	@test -x .venv/bin/python || { echo "error: create the venv first (uv venv)" >&2; exit 1; }
	$(eval PYBIND11_DIR := $(shell .venv/bin/python -m pybind11 --cmakedir))
	CMAKE_PREFIX_PATH="$(CURDIR)/build/extern/openfhe-development-install/lib/OpenFHE:$(PYBIND11_DIR)" \
	  uv pip install -e ./extern/openfhe-python --python .venv/bin/python

clean:
	@echo "[INFO]  Cleaning up build directories (not external dependencies)"
	rm -rf "$(CURDIR)/build"
#	rm -rf "$(CURDIR)/build/cmake-build-"*

clean-all:
	@echo "[INFO]  Cleaning up build directories and external dependencies"
	rm -rf "$(CURDIR)/build"
	rm -rf "$(CURDIR)/extern/"
