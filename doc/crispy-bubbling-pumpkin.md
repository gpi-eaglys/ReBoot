# Package `reboot` for `uv`-native install (scikit-build-core)

## Context

The Python side of this repo (`py/reboot/`) and its C++ extension (`reboot_py`, built via `cpp/python/CMakeLists.txt`) currently have no install story at all — no `pyproject.toml` exists. Usage today depends on running scripts from the repo root (so the implicit `py` namespace resolves) and manually placing the compiled `.so` at repo root or fiddling with `PYTHONPATH`. Since this project is meant to be used from a `uv`-managed venv in this repo, the natural fix is a real package install: `uv pip install -e .` should build the CMake project and drop both the pure-Python `reboot` package and the compiled `reboot_py` extension straight into `.venv/lib/python3.12/site-packages`, importable from anywhere without path tricks.

Chosen mechanism: **scikit-build-core** as the PEP 517 build backend — it drives our existing CMakeLists.txt files directly, so no parallel build system needs inventing.

Two scoping decisions already confirmed with the user:
- `openfhe` (the official OpenFHE Python bindings) is built from a local source checkout against the exact OpenFHE C++ install (see `.devcontainer/after_creation.sh`) — it will **not** be listed as a pyproject dependency, only documented as an external prerequisite, to avoid `uv` ever trying to fetch/overwrite it from PyPI.
- The Makefile's `py11`/`bindings` target is retired now that `uv pip install -e .` is the one Python build path. `lib`, `build-tests`, `tests`, `tidy`, `clean`, `dep_openfhe` stay untouched (pure C++ dev loop).

## Required import-path fixes (blocking, not cosmetic)

Once installed via `wheel.packages = ["py/reboot"]`, the top-level importable package name becomes **`reboot`** (scikit-build-core uses the source directory's own basename — `py/` is just the src-layout container, dropped on install). Right now imports are inconsistent:

- 14 files under `py/reboot/**` and several under `experiments/**` use `from py.reboot...` / `import py.reboot...` — these must become `from reboot...` / `import reboot...` (drop the `py.` prefix). Representative files: `py/reboot/cryptocontext.py`, `py/reboot/layers/linear.py`, `experiments/1_training_plain/backprop_plain.py`.
- `py/reboot/models/sequential.py` already uses bare `reboot.*` — no change needed there.
- `experiments/2_logreg_comparison/cross_training.py:12` has a stale, *also wrong-path* import: `from lib.utils import get_parser_args` → `get_parser_args` actually lives in `py/reboot/parser.py`, so this becomes `from reboot.parser import get_parser_args`.
- `test.ipynb` (repo root) has 5 stale `from lib...` cells to rename 1:1 to `from reboot...` (all target modules confirmed to exist: `reboot.utils.enums`, `reboot.models.backprop_models`, `reboot.parser`, `reboot.utils.data`, `reboot.utils.nn`, `reboot.utils.train`).
- `import reboot_py` (the compiled extension) stays a bare top-level import everywhere — no changes needed, it'll sit alongside the `reboot` package in site-packages.

This is a mechanical find/replace (`py.reboot` → `reboot`), not a redesign — apply it across `py/`, `experiments/`, and `test.ipynb`.

## File changes

1. **`py/reboot/{blocks,layers,models,optim,utils}/__init__.py`** (new, empty) — these subpackages currently have no `__init__.py` (implicit namespace packages), unlike `py/reboot/__init__.py` which exists (empty). Add matching empty files for consistency and to avoid editable-install namespace-package edge cases.

2. **`cpp/python/CMakeLists.txt`** — add an `install()` rule; there is none today, so `cmake --install` (which scikit-build-core runs internally) currently has nothing to package:
   ```cmake
   install(TARGETS reboot_py DESTINATION .)
   ```
   `DESTINATION .` places the `.so` at the top level of site-packages, matching the existing bare `import reboot_py`.

3. **New `pyproject.toml`** at repo root:
   - `[build-system]`: `requires = ["scikit-build-core>=0.10", "pybind11>=2.12"]`, `build-backend = "scikit_build_core.build"`.
   - `[project]`: `name = "reboot"`, `requires-python = ">=3.10"` (README says 3.10.12, current `.venv` is 3.12.3 — stay permissive). `dependencies` carries over the "Core" block from `requirements.txt` (numpy==2.0.0, pyyaml==6.0.1, scikit-learn==1.5.2, ucimlrepo==0.0.7, imbalanced-learn==0.12.4 — note corrected hyphenated name) plus `pandas` and `torch`/`torchvision` (all four are unconditionally imported by core modules — not optional — see `utils/data.py`, `optim/schedulers.py`, `models/backprop_models.py`). `openfhe` deliberately omitted per the decision above.
   - `[project.optional-dependencies].dev`: the "Dev" block from `requirements.txt` (jupyter, ipykernel, matplotlib, seaborn, wandb, palettable).
   - `[tool.uv.sources]` / `[[tool.uv.index]]`: pin `torch`/`torchvision` to the CPU wheel index (`https://download.pytorch.org/whl/cpu`, `explicit = true`), matching what `after_creation.sh` already does manually.
   - `[tool.scikit-build]`: `wheel.packages = ["py/reboot"]`; `cmake.define.BUILD_TESTS = "OFF"` (skip fetching/building googletest for a Python install — tests stay reachable via `make tests`); `OpenFHE_DIR` forwarded via the standard `CMAKE_ARGS` env var scikit-build-core already respects (documented in README, e.g. `CMAKE_ARGS="-DOpenFHE_DIR=/opt/openfhe/v1.2.1/install/lib/OpenFHE" uv pip install -e .`).

4. **`requirements.txt`** — delete. It's superseded by `pyproject.toml`, and it was already stale (UTF-16 encoded, missing torch/torchvision/pandas, wrong pip name for imbalanced-learn).

5. **`Makefile`** — remove the `py11` target and its dependency in `all` (`all: py11 tests` → `all: tests`). `PYBIND11_DIR` variable becomes unused and can go too.

6. **`README.md`** — replace the devcontainer-only install blurb with: prerequisites (OpenFHE C++ + openfhe-python, as today) still installed via the devcontainer script, then `uv pip install -e .` (with the `CMAKE_ARGS` example for a custom `OpenFHE_DIR`) as the one-command way to get `reboot` + `reboot_py` importable in `.venv`.

## Verification

1. `uv pip install scikit-build-core pybind11 --python .venv/bin/python` is *not* needed manually (build-system requires handles it in an isolated env), but run `uv pip install -e . --python .venv/bin/python` for real and confirm it succeeds end-to-end (CMake configure + build + install step logs should show `reboot_core`, `reboot_py` building, then wheel/editable install).
2. `uv pip list --python .venv/bin/python` should now show `reboot` (editable) plus `reboot_py`'s presence confirmed via import, not listing (compiled extensions don't show as a "package" the same way).
3. From repo root **and** from `/tmp` (to prove it's not cwd-dependent anymore): `python3 -c "import reboot_py; import reboot.parser; import reboot.types; print('ok')"`.
4. Confirm `cryptocontext.py`'s `import openfhe` is the only expected failure point in this sandbox (no openfhe-python installed here) — this is a pre-existing, documented external prerequisite, not a regression from this change.
5. `make tests` still passes (build-tests/tests targets untouched, BUILD_TESTS default unaffected at the Makefile layer).
6. Spot check that the import-path find/replace didn't miss anything: `grep -rn "py\.reboot" py/ experiments/ test.ipynb` returns nothing.
