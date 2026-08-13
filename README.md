# ReBoot Digest
* this is the digested version of the original ReBoot github project
* changes 
  * refactored project
  * added Python packaging 
  * added explanations


## TL;DR 
* in the repository dir
* build the package 
  ```
  uv pip install -e .
  ```

* test import 
  ```
  python -c "import reboot"
  ```







# ReBoot (original) 
ReBoot is the first framework to enable fully encrypted and non-interactive training of Multi-Layer Perceptrons (MLPs) using CKKS bootstrapping.
ReBoot has been introduced in the paper: ["ReBoot: Encrypted Training of Deep Neural Networks with CKKS Bootstrapping"](https://arxiv.org/abs/2506.19693), published in the '40th Annual AAAI Conference on Artificial Intelligence'.

## Requirements

ReBoot was developed and tested with:

- **Python:** 3.10.12
- **OpenFHE:** 1.2.1
- **OpenFHE-Python:** 0.8.9

Use the provided `.devcontainer` files to spin up a VSCode DevContainer with the library correctly installed.

It will install the OpenFHE and OpenFHE-Python libraries, along with the necessary dependencies to run ReBoot.

## Installing

With OpenFHE and OpenFHE-Python already installed (via the devcontainer, or manually — see [doc/build.md](doc/build.md) for building OpenFHE-Python against a `uv` venv) and a `uv`-managed venv active in this repo:

```
uv pip install -e .
```

This builds `reboot_core`/`reboot_py` via CMake and installs the `reboot` Python package plus the compiled `reboot_py` extension into `.venv`, both importable from anywhere the venv is active.

If OpenFHE isn't installed at the default location, forward it via `CMAKE_ARGS`, which `scikit-build-core` (the build backend) picks up automatically:

```
CMAKE_ARGS="-DOpenFHE_DIR=/path/to/OpenFHE" uv pip install -e .
```

For pure C++ development (building `reboot_core`/tests without going through the Python packaging), see the `Makefile` (`make lib`, `make tests`).

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
