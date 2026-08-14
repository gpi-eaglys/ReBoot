# How does build works?


## Building C++ 
* simple CMake project 
* `reboot_core` requires OpenFHE via `find_package(OpenFHE 1.2.0 REQUIRED)` (see `cpp/core/CMakeLists.txt`). The root `CMakeLists.txt` probes for a system install first; if none is found, it builds OpenFHE from the `extern/openfhe-development` submodule into `build/extern/openfhe-development-install` and points `find_package` at that instead — so a fresh checkout with no local OpenFHE install still builds standalone. This only happens once; subsequent configures reuse the existing install.
* run `git submodule update --init --recursive` once after cloning to check out `extern/openfhe-development` and `extern/openfhe-python`.



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

It's vendored as the `extern/openfhe-python` submodule (pinned at `v0.8.9`, matching `README.md`'s Requirements). The `py-openfhe` Makefile target builds it against the vendored OpenFHE install (building that first via `make lib` if it isn't there yet), then installs it into the project's venv:

```
make py-openfhe
```

Which is equivalent to:

```
PYBIND11_DIR=$(.venv/bin/python -m pybind11 --cmakedir)
CMAKE_PREFIX_PATH="build/extern/openfhe-development-install/lib/OpenFHE:$PYBIND11_DIR" \
  uv pip install -e ./extern/openfhe-python --python .venv/bin/python
```

- `openfhe-python`'s `setup.py` calls `find_package(pybind11 REQUIRED)` with no hints and doesn't forward a `CMAKE_ARGS` env var, so `pybind11_DIR` can't be passed as a normal `-D` flag here — instead it's appended to `CMAKE_PREFIX_PATH`, which CMake also checks directly (not just with suffixes appended) for a `pybind11Config.cmake`.
- If pointing this at a different (e.g. system-installed) OpenFHE, adjust `CMAKE_PREFIX_PATH` accordingly — the OpenFHE/OpenFHE-Python versions must match (see `README.md` Requirements).
- `build/` is gitignored; it holds every build artifact for this repo (CMake build trees, the vendored OpenFHE install) — none of it is repo source. The submodule source itself lives in `extern/`, which is real repo state (tracked via `.gitmodules`).