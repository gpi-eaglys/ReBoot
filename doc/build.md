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