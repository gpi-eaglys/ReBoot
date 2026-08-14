# ReBoot Digest
* this is the digested version of the original ReBoot github project
* changes 
  * refactored project
  * added Python packaging 
  * added explanations


## Build 
* pull in openFHE and build it
``` 
git submodule update --init --recursive
```

* build the Python package (see [build doc](doc/build.md) for details)
  ```
  uv venv
  uv sync
  uv pip install -e .
  ```


``` 
make py-openfhe
```

* test if `reboot` import works (must be in virtual environment)
  ```
  source .venv/bin/activate
  
  python -c "import reboot"
  ```


* run sample MNIST experiment
  ```
  experiments/1_training_plain/run-sample-exp.sh
  ```


## Project structure 

``` 
ReBoot
├── build        # build artifacts - not checked into git
├── cpp          # all CPP code
├── doc          # documentation
├── experiments  # experiments for the publication
├── extern       # external dependencies/submodules 
└── py           # Python source code
```




# ReBoot (original) 
ReBoot is the first framework to enable fully encrypted and non-interactive training of Multi-Layer Perceptrons (MLPs) using CKKS bootstrapping.
ReBoot has been introduced in the paper: ["ReBoot: Encrypted Training of Deep Neural Networks with CKKS Bootstrapping"](https://arxiv.org/abs/2506.19693), published in the '40th Annual AAAI Conference on Artificial Intelligence'.

## Requirements

ReBoot was developed and tested with:

- **Python:** 3.10.12
- **OpenFHE:** 1.2.1
- **OpenFHE-Python:** 0.8.9

OpenFHE and OpenFHE-Python are vendored as git submodules under `extern/` and
built automatically if no system install is found (see
[doc/build.md](doc/build.md)), so no local OpenFHE install is required to
build this repo. Alternatively, use the provided `.devcontainer` files to spin
up a VSCode DevContainer with the library pre-installed system-wide.

## Installing

Check out the vendored submodules once:

```
git submodule update --init --recursive
```

Then, with a `uv`-managed venv active in this repo:

```
uv pip install -e .
```

This builds `reboot_core`/`reboot_py` via CMake and installs the `reboot` Python package plus the compiled `reboot_py` extension into `.venv`, both importable from anywhere the venv is active. If no system OpenFHE is found, CMake builds it from `extern/openfhe-development` first (only once — this can take a while the first time).

If OpenFHE is (or should be) installed elsewhere, forward it via `CMAKE_ARGS`, which `scikit-build-core` (the build backend) picks up automatically:

```
CMAKE_ARGS="-DOpenFHE_DIR=/path/to/OpenFHE" uv pip install -e .
```

For pure C++ development (building `reboot_core`/tests without going through the Python packaging), see the `Makefile` (`make lib`, `make tests`).

The official `openfhe` Python bindings (used by `reboot.cryptocontext`) are a separate build, from the `extern/openfhe-python` submodule — see [doc/build.md](doc/build.md) or run `make py-openfhe`.

## Multiplicative depth

This table summarizes the multiplicative depth required to perform a training step with different MLP architectures.
The worst-case multiplicative depth is represented.

| Architecture     | Forward | Backward | Weights |  Additional depth per step |
|------------------|:-------:|:--------:|:-------:|:--------------------------:|
| No hidden layers | 1       | 1        | 3       | 3                          |
| 1 hidden layer   | 3       | 5        | 7       | 7                          |
| 2 hidden layers  | 5       | 7        | 9       | 7                          |
| 3 hidden layers  | 7       | 9        | 11      | 7                          |

**Remark:** the use of **weight decay** or **momentum** in the optimizer does not increase the depth.

## Authors and Contacts
If you have questions, suggestions or problems, feel free to open an Issue.
You can contact us at:
- [Alberto Pirillo](https://github.com/albertopirillo): alberto.pirillo@mail.polimi.it
- [Luca Colombo](https://github.com/lucacolombo97): luca2.colombo@polimi.it
