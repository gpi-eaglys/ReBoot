OPENFHE_DIR ?= /opt/openfhe/v1.2.1/install/lib/OpenFHE
NPROC := $(shell nproc)
BUILD_TYPE ?= Debug
BUILD_DIR := $(CURDIR)/build/cmake-build-$(shell echo $(BUILD_TYPE) | tr '[:upper:]' '[:lower:]')
GENERATOR ?= Ninja

.PHONY: all lib build-tests tests tidy clean dep_openfhe py

all: tests

dep_openfhe:
	@test -d "$(OPENFHE_DIR)" || { echo "error: OPENFHE_DIR not found: $(OPENFHE_DIR) (install OpenFHE or override with 'make OPENFHE_DIR=...')" >&2; exit 1; }

lib: dep_openfhe
	cmake -DOpenFHE_DIR="$(OPENFHE_DIR)" -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -G "$(GENERATOR)" -S . -B "$(BUILD_DIR)"
	cmake --build "$(BUILD_DIR)" -j$(NPROC) --target reboot_core

build-tests: dep_openfhe
	cmake -DOpenFHE_DIR="$(OPENFHE_DIR)" \
		  -DBUILD_TESTS=ON \
          -DBUILD_PYTHON_BINDINGS=OFF \
          -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -G "$(GENERATOR)" -S . -B "$(BUILD_DIR)"
	cmake --build "$(BUILD_DIR)" -j$(NPROC) --target reboot_core_tests

tests: build-tests
	ctest --test-dir "$(BUILD_DIR)" --output-on-failure

tidy: dep_openfhe
	cmake -DOpenFHE_DIR="$(OPENFHE_DIR)" -DBUILD_PYTHON_BINDINGS=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G "$(GENERATOR)" -S . -B "$(BUILD_DIR)"
	run-clang-tidy -p "$(BUILD_DIR)" -j$(NPROC) -quiet 'cpp/(core|tests)/.*'

py:
	@echo "No Python target! Call: 'uv pip install -e .'"

clean:
	rm -rf "$(CURDIR)/build/cmake-build-"*

