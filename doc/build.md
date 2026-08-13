# How does build works?


## Building C++ 
* simple CMake project 






## Building Python package 
* install with `uv` and `pip`
```
uv pip install -e .
```
* processing steps
    * uv parses [pyproject.toml](pyproject.toml)
    * pip/build calls scikit_build_core's PEP 517 hooks
    * **scikit-build-core**:
      * configures a CMake build tree
      * builds ReBoot and Python bindings via cmake: 
      ```
        cmake -S . -B build ...
        cmake --build build ...
        cmake --install build ...
      ```
    * **scikit-build-core** 
      *  collects the built shared library and artifacts (relying on ReBoot's CMake `install()` settings ) 
      *  packs them into a wheel

## Building OpenFHE-Python

`reboot.cryptocontext` does `import openfhe`, the official OpenFHE Python bindings. This is **not** a `pyproject.toml` dependency — the public PyPI `openfhe` wheels bundle their own separate copy of the OpenFHE C++ runtime, which wouldn't be ABI/instance-compatible with `reboot_py` (passing a `CryptoContext` between the two would break). It must be built from source against the same OpenFHE C++ install `reboot_py` uses.

```
git clone --depth 1 --branch v0.8.9 https://github.com/openfheorg/openfhe-python.git
PYBIND11_DIR=$(.venv/bin/python -m pybind11 --cmakedir)
CMAKE_PREFIX_PATH="/opt/openfhe/v1.2.1/install:$PYBIND11_DIR" \
  uv pip install -e ./openfhe-python --python .venv/bin/python
```

- `openfhe-python`'s `setup.py` calls `find_package(pybind11 REQUIRED)` with no hints and doesn't forward a `CMAKE_ARGS` env var, so `pybind11_DIR` can't be passed as a normal `-D` flag here — instead it's appended to `CMAKE_PREFIX_PATH`, which CMake also checks directly (not just with suffixes appended) for a `pybind11Config.cmake`.
- Adjust the `v0.8.9` tag and `/opt/openfhe/v1.2.1/install` path if using a different OpenFHE/OpenFHE-Python version — they must match (see `README.md` Requirements).
- `openfhe-python/` is gitignored; it's a vendored build dependency, not repo source.