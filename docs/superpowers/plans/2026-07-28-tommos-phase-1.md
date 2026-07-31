# `tommos` Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task by
> task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the existing flat Python modules into an installable
distribution and import package named `tommos`, using `src/tommos/` and an
editable Pixi installation while retaining the existing external C++/oneMKL
build workflow.

**Architecture:** Move only Python modules into `src/tommos/`, convert internal
imports to package-relative imports, and use setuptools with PEP 621 metadata.
Keep C++ sources at `src/cpp/` and keep native compilation external to the
wheel. Replace source-file command execution with `python -m tommos.<module>`.

**Tech Stack:** Python 3.11+, setuptools, PEP 517/621, Pixi, pytest, Ruff,
JAX, NumPy, SciPy, PyAMG, MeshPy, Matplotlib.

## Global Constraints

- The distribution name and import namespace are exactly `tommos`.
- Python source must live under `src/tommos/`.
- The supported command form is `python -m tommos.<module>`; do not retain
  `src/*.py` compatibility wrappers.
- Pixi must install the local project editably during Phase 1.
- Keep `src/cpp/`, `compile_local.sh`, `activate_build.sh`, and the existing
  external native compilation model.
- Do not compile or bundle the C++/oneMKL library through the Python build
  backend in Phase 1.
- Do not change numerical algorithms, CLI options, defaults, output formats, or
  the public functions/classes already defined by the modules.
- Do not add console-script names in Phase 1.
- Do not invent license, copyright, maintainer, or project-URL metadata.
- Every modified Python file must pass `py_compile` before any modified Python
  module or test is executed.
- Use Google-style docstrings and explicit Python type hints for any new
  functions. Phase 1 should not require new production functions.
- Do not perform Git writes. `git status`, `git diff`, `git show`, and other
  read-only Git operations are allowed; `git add`, `git commit`, branch,
  checkout, reset, clean, and worktree operations are forbidden.
- Existing user changes must be preserved.
- Pure documentation and declarative-configuration edits do not require
  artificial source-text tests. Validate them with their parsers, executable
  module smoke tests, and focused acceptance searches.

---

### Task 1: Establish the `tommos` source package

**Files:**

- Create: `tests/test_package_imports.py`
- Create: `src/tommos/__init__.py`
- Move: `src/add_shell.py` → `src/tommos/add_shell.py`
- Move: `src/amg_utils.py` → `src/tommos/amg_utils.py`
- Move: `src/cpp_minimizer.py` → `src/tommos/cpp_minimizer.py`
- Move: `src/energy_kernels.py` → `src/tommos/energy_kernels.py`
- Move: `src/extract_nucleation.py` → `src/tommos/extract_nucleation.py`
- Move: `src/fem_utils.py` → `src/tommos/fem_utils.py`
- Move: `src/hysteresis_loop.py` → `src/tommos/hysteresis_loop.py`
- Move: `src/io_utils.py` → `src/tommos/io_utils.py`
- Move: `src/loop.py` → `src/tommos/loop.py`
- Move: `src/make_krn.py` → `src/tommos/make_krn.py`
- Move: `src/mesh.py` → `src/tommos/mesh.py`
- Move: `src/mesh_convert.py` → `src/tommos/mesh_convert.py`
- Move: `src/minimizers.py` → `src/tommos/minimizers.py`
- Move: `src/plot_hysteresis.py` → `src/tommos/plot_hysteresis.py`
- Move: `src/poisson_solve.py` → `src/tommos/poisson_solve.py`
- Move: `src/reorder_mesh.py` → `src/tommos/reorder_mesh.py`
- Move: `src/salomeMeshToNpz.py` → `src/tommos/salomeMeshToNpz.py`
- Preserve unchanged: `src/cpp/CMakeLists.txt`
- Preserve unchanged: `src/cpp/cpp_mkl_minimizer.cpp`

**Interfaces:**

- Produces: regular import package `tommos`.
- Produces: importable modules `tommos.<existing_module_name>`.
- Produces: module execution for CLIs that already have `main()` and
  `if __name__ == "__main__"` guards.
- Preserves: all existing Python function and class signatures.
- Preserves: repository-level native library output at `lib/`.

- [ ] **Step 1: Write the failing namespace and module-execution tests**

Create `tests/test_package_imports.py`:

```python
import importlib
import subprocess
import sys

import pytest


IMPORTABLE_MODULES = (
    "add_shell",
    "amg_utils",
    "cpp_minimizer",
    "energy_kernels",
    "extract_nucleation",
    "fem_utils",
    "hysteresis_loop",
    "io_utils",
    "loop",
    "make_krn",
    "mesh",
    "mesh_convert",
    "minimizers",
    "plot_hysteresis",
    "poisson_solve",
    "reorder_mesh",
    "salomeMeshToNpz",
)

HELP_MODULES = (
    "tommos.add_shell",
    "tommos.loop",
    "tommos.mesh",
)


def test_tommos_namespace_is_importable() -> None:
    module = importlib.import_module("tommos")
    assert module.__name__ == "tommos"


@pytest.mark.parametrize("module_name", IMPORTABLE_MODULES)
def test_modules_import_through_tommos_namespace(module_name: str) -> None:
    module = importlib.import_module(f"tommos.{module_name}")
    assert module.__package__ == "tommos"


@pytest.mark.parametrize("module_name", HELP_MODULES)
def test_primary_cli_modules_expose_help(module_name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
```

The production mutation caught by these tests is removal or misplacement of
the `tommos` package, failure to use package-correct internal imports, or loss
of module execution for a primary CLI.

- [ ] **Step 2: Check test syntax before RED execution**

Run:

```bash
pixi run python -m py_compile tests/test_package_imports.py
```

Expected: exit code `0`.

- [ ] **Step 3: Run the namespace test and verify RED**

Run:

```bash
PYTHONPATH=src pixi run pytest tests/test_package_imports.py::test_tommos_namespace_is_importable -v
```

Expected: failure caused by `ModuleNotFoundError: No module named 'tommos'`.

- [ ] **Step 4: Move the Python modules and add the package initializer**

Create `src/tommos/__init__.py`:

```python
"""Matrix-free micromagnetic simulations with JAX."""
```

Move the listed Python files without altering unrelated content. Keep
`src/cpp/` in place.

- [ ] **Step 5: Convert every internal import to a package-relative import**

Apply these exact mapping rules in `src/tommos/`:

```text
import add_shell
    -> from . import add_shell

from add_shell import ...
    -> from .add_shell import ...

from amg_utils import ...
    -> from .amg_utils import ...

from cpp_minimizer import ...
    -> from .cpp_minimizer import ...

from energy_kernels import ...
    -> from .energy_kernels import ...

from fem_utils import ...
    -> from .fem_utils import ...

from hysteresis_loop import ...
    -> from .hysteresis_loop import ...

from io_utils import ...
    -> from .io_utils import ...

from minimizers import ...
    -> from .minimizers import ...

from poisson_solve import ...
    -> from .poisson_solve import ...
```

Apply the mappings to imports at module scope and inside functions. Do not
change third-party or standard-library imports.

- [ ] **Step 6: Preserve repository-native library lookup after the move**

In `src/tommos/cpp_minimizer.py` and `src/tommos/amg_utils.py`, change only
the repository fallback from one parent directory to two:

```python
os.path.join(os.path.dirname(__file__), "../../lib/libcpp_mkl_minimizer.so")
```

Keep the existing Slurm and `MUMAG_LIB_OUT` precedence unchanged.

- [ ] **Step 7: Check all modified Python syntax**

Run:

```bash
pixi run python -m py_compile src/tommos/*.py tests/test_package_imports.py
```

Expected: exit code `0`.

- [ ] **Step 8: Run focused package tests and verify GREEN**

Run:

```bash
PYTHONPATH=src pixi run pytest tests/test_package_imports.py -v
```

Expected: all package-import and primary CLI-help cases pass.

- [ ] **Step 9: Inspect the task without writing Git state**

Run:

```bash
git status --short
git diff --stat
```

Confirm that `src/cpp/` remains in place and no unrelated source file changed.
Do not stage or commit.

---

### Task 2: Add distribution metadata and editable Pixi installation

**Files:**

- Create: `tests/test_distribution_metadata.py`
- Modify: `pyproject.toml`
- Modify: `pixi.toml`
- Modify: `.gitignore`

**Interfaces:**

- Consumes: package `tommos` from Task 1.
- Produces: distribution metadata with name `tommos`, version `0.1.0`, and
  Python requirement `>=3.11`.
- Produces: setuptools discovery of only `src/tommos*`.
- Produces: editable Pixi dependency on the repository root.
- Produces: Pixi task `build-package` running `python -m build`.

- [ ] **Step 1: Write the failing distribution-metadata test**

Create `tests/test_distribution_metadata.py`:

```python
from importlib.metadata import metadata, version


def test_distribution_metadata_matches_import_package() -> None:
    project_metadata = metadata("tommos")
    assert project_metadata["Name"] == "tommos"
    assert version("tommos") == "0.1.0"
    assert project_metadata["Requires-Python"] == ">=3.11"
```

The production mutation caught by this test is missing or inconsistent
installed distribution metadata.

- [ ] **Step 2: Check test syntax before RED execution**

Run:

```bash
pixi run python -m py_compile tests/test_distribution_metadata.py
```

Expected: exit code `0`.

- [ ] **Step 3: Run the metadata test and verify RED**

Run:

```bash
PYTHONPATH=src pixi run pytest tests/test_distribution_metadata.py -v
```

Expected: failure caused by
`importlib.metadata.PackageNotFoundError: No package metadata was found for tommos`.

- [ ] **Step 4: Add build and project metadata**

Add these tables to `pyproject.toml` without removing the existing Ruff
configuration:

```toml
[build-system]
requires = ["setuptools >= 77.0.3"]
build-backend = "setuptools.build_meta"

[project]
name = "tommos"
version = "0.1.0"
description = "Matrix-free micromagnetic simulations with JAX"
readme = "README.md"
requires-python = ">=3.11"
authors = [
  { name = "MaMMoS-project" },
]
dependencies = [
  "jax",
  "matplotlib",
  "meshpy",
  "numpy",
  "pyamg",
  "scipy",
]

[project.optional-dependencies]
io = [
  "mammos-entity",
  "meshio",
]

[tool.setuptools.packages.find]
where = ["src"]
include = ["tommos*"]
namespaces = false
```

Change the Ruff local-package setting to:

```toml
isort.known-local-folder = [ "tommos" ]
```

Do not add license, maintainer, URL, console-script, or native-library
metadata.

- [ ] **Step 5: Add the editable project and build frontend to Pixi**

Under the existing `[pypi-dependencies]` table in `pixi.toml`, add:

```toml
build = "*"
tommos = { path = ".", editable = true }
```

Do not rename the Pixi workspace or change its CPU/CUDA features.

Under `[tasks]`, add:

```toml
build-package = "python -m build"
```

- [ ] **Step 6: Ignore standard local build artifacts**

Add to `.gitignore`:

```gitignore
dist/
*.egg-info/
```

- [ ] **Step 7: Validate TOML before package execution**

Run:

```bash
pixi run python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb')); tomllib.load(open('pixi.toml', 'rb'))"
```

Expected: exit code `0`.

- [ ] **Step 8: Materialize the editable installation**

Run:

```bash
pixi install
```

Expected: the environment resolves and installs local distribution `tommos`
editably.

- [ ] **Step 9: Check all modified Python syntax**

Run:

```bash
pixi run python -m py_compile src/tommos/*.py tests/test_package_imports.py tests/test_distribution_metadata.py
```

Expected: exit code `0`.

- [ ] **Step 10: Run metadata and import tests and verify GREEN**

Run:

```bash
pixi run pytest tests/test_distribution_metadata.py tests/test_package_imports.py -v
```

Expected: all tests pass without `PYTHONPATH`.

- [ ] **Step 11: Build and inspect both distribution formats**

Run:

```bash
pixi run build-package
pixi run python -m zipfile -l dist/tommos-0.1.0-py3-none-any.whl
```

Expected: `dist/` contains the `tommos` wheel and source distribution; the
wheel contains `tommos/__init__.py` and the moved Python modules, and does not
contain `src/cpp/` or `libcpp_mkl_minimizer.so`.

- [ ] **Step 12: Test the built wheel outside the checkout**

Create a temporary install target and install without resolving dependencies:

```bash
package_target="$(mktemp -d /tmp/tommos-wheel-install.XXXXXX)"
pixi run python -m pip install --no-deps --target "$package_target" dist/tommos-0.1.0-py3-none-any.whl
cd /tmp
PYTHONPATH="$package_target" /workspace/.pixi/envs/default/bin/python -c "import tommos; import tommos.fem_utils"
```

Expected: imports succeed from `/tmp`, not from the repository checkout.

- [ ] **Step 13: Inspect the task without writing Git state**

Run:

```bash
git status --short
git diff -- pyproject.toml pixi.toml .gitignore
```

Do not stage or commit.

---

### Task 3: Migrate tests, benchmarks, and development imports

**Files:**

- Modify: `tests/test_gradients.py`
- Modify: `tests/test_energy.py`
- Modify: `tests/test_cpp_minimizer.py`
- Modify: `tests/test_compare_kernels.py`
- Modify: `tests/test_py_minimizer.py`
- Modify: `tests/test_mkl_jit.py`
- Modify: `benchmarking/test_poisson_convergence.py`
- Modify: `benchmarking/profile_energy_jax.py`
- Modify: `benchmarking/profile_energy.py`
- Modify: `benchmarking/profile_compilation.py`
- Modify: `benchmarking/generate_test_mesh.py`
- Modify: `benchmarking/compare_minimizers.py`
- Modify: `develop/strain/test_stoner_wohlfarth_me.py`

**Interfaces:**

- Consumes: editable `tommos` installation from Task 2.
- Produces: all repository Python consumers import `tommos.*` without adding
  `src/` to `sys.path`.
- Preserves: test behavior and benchmark algorithms.

- [ ] **Step 1: Verify the old flat-import consumers are RED**

Run:

```bash
pixi run pytest tests/test_energy.py --collect-only -q
```

Expected: collection fails because flat modules such as `add_shell` are no
longer importable from `src/`.

- [ ] **Step 2: Remove source-path injection**

Remove only the `sys.path.append(...)` calls whose purpose is to add repository
`src/`. Remove now-unused `sys`, `os`, or `Path` imports only when they have no
other use in the same file.

- [ ] **Step 3: Convert consumer imports**

Use these exact mappings:

```text
import add_shell
    -> from tommos import add_shell

import mesh
    -> from tommos import mesh

from <local_module> import <names>
    -> from tommos.<local_module> import <names>
```

Here, `<local_module>` is one of `amg_utils`, `cpp_minimizer`,
`energy_kernels`, `fem_utils`, `hysteresis_loop`, `io_utils`, `loop`,
`minimizers`, or `poisson_solve`.

Do not change the pre-existing `curvilinear_bb_minimizer` import in
`benchmarking/compare_minimizers.py`; that absent module is outside Phase 1.

- [ ] **Step 4: Check modified Python syntax**

Run:

```bash
pixi run python -m py_compile tests/*.py benchmarking/*.py develop/strain/test_stoner_wohlfarth_me.py
```

Expected: exit code `0`.

- [ ] **Step 5: Verify test collection is GREEN**

Run:

```bash
pixi run pytest tests/ --collect-only -q
```

Expected: repository tests collect without errors caused by flat local imports.

- [ ] **Step 6: Run focused portable tests**

Run:

```bash
pixi run pytest tests/test_energy.py tests/test_gradients.py tests/test_py_minimizer.py -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Inspect the task without writing Git state**

Run:

```bash
git status --short
git diff -- tests benchmarking develop/strain/test_stoner_wohlfarth_me.py
```

Confirm that changes are limited to import and obsolete path-injection lines.
Do not stage or commit.

---

### Task 4: Migrate commands, examples, samples, and documentation

**Files:**

- Modify: `README.md`
- Modify: `samples/run_cube.sh`
- Modify: `samples/sphereR10nm/run_sphereR10nm.sh`
- Modify: `samples/sphereR20nm/run_sphereR20nm.sh`
- Modify: `benchmarking/run_comparison_benchmark.sh`
- Modify: `benchmarking/install_cpp.md`
- Modify: `examples/evaluate_materials/generate_structures.py`
- Modify: `examples/evaluate_materials/compute_evaluations.py`
- Modify: `examples/benchmark_1/benchmark1_workflow.py`

**Interfaces:**

- Consumes: installed module entry points from Task 1.
- Produces: repository-supported invocations use
  `python -m tommos.<module>`.
- Preserves: all CLI arguments and workflow ordering.

- [ ] **Step 1: Replace shell and documentation commands**

Use these exact command mappings:

```text
python src/loop.py
python3 src/loop.py
python ../src/loop.py
python ../../src/loop.py
    -> python -m tommos.loop

python src/mesh.py
python3 src/mesh.py
python ../src/mesh.py
python ../../src/mesh.py
    -> python -m tommos.mesh

python3 ../src/add_shell.py
    -> python -m tommos.add_shell
```

Preserve existing `pixi run`, environment-selection flags, shell variables,
line continuations, and CLI arguments around the replaced command.

Update prose references from source file paths to module names where the text
describes how users invoke the installed package. Historical documents under
`doc/` are not part of this migration.

- [ ] **Step 2: Replace Python subprocess source paths**

In the listed example scripts, replace command construction of this form:

```python
script = (base / "src/mesh.py").resolve()
cmd = [sys.executable, str(script), *arguments]
```

with:

```python
cmd = [sys.executable, "-m", "tommos.mesh", *arguments]
```

Apply the corresponding exact module names `tommos.make_krn` and
`tommos.loop`. Remove path variables only when no longer used elsewhere.
Preserve subprocess arguments, working directories, logging, and error
handling.

- [ ] **Step 3: Check modified Python and shell syntax**

Run:

```bash
pixi run python -m py_compile examples/evaluate_materials/generate_structures.py examples/evaluate_materials/compute_evaluations.py examples/benchmark_1/benchmark1_workflow.py
bash -n samples/run_cube.sh samples/sphereR10nm/run_sphereR10nm.sh samples/sphereR20nm/run_sphereR20nm.sh benchmarking/run_comparison_benchmark.sh
```

Expected: both commands exit with code `0`.

- [ ] **Step 4: Verify the installed module commands**

Run:

```bash
pixi run python -m tommos.loop --help
pixi run python -m tommos.mesh --help
pixi run python -m tommos.add_shell --help
```

Expected: each command exits with code `0` and prints its argument-parser
usage.

- [ ] **Step 5: Verify migration acceptance**

Run:

```bash
rg -n 'python3? +((\\.\\./)+)?src/(loop|mesh|add_shell|make_krn)\\.py' README.md samples benchmarking examples
rg -n '(base / "src/(loop|mesh|make_krn)\\.py")' examples --glob '*.py'
```

Expected: no matches in the files in this task. Any match outside those files
must be reported rather than changed without approval.

- [ ] **Step 6: Inspect the task without writing Git state**

Run:

```bash
git status --short
git diff -- README.md samples benchmarking examples
```

Confirm that commands changed but their arguments and workflow sequencing did
not. Do not stage or commit.

---

### Task 5: Add package verification to CI and run the full validation

**Files:**

- Modify: `.github/workflows/test.yml`

**Interfaces:**

- Consumes: Pixi task `build-package` from Task 2.
- Produces: CI evidence that distributions build before the existing test and
  sample jobs.
- Preserves: existing Ubuntu and macOS Pixi test/sample coverage.

- [ ] **Step 1: Add the package-build CI step**

After `Setup Pixi` and before `Run tests`, add:

```yaml
    - name: Build Python distributions
      run: pixi run build-package
```

Do not change the OS matrix or remove existing test, sample, or artifact steps.

- [ ] **Step 2: Validate YAML**

Run:

```bash
pixi run pre-commit run check-yaml --files .github/workflows/test.yml
```

Expected: the YAML hook passes.

- [ ] **Step 3: Check all Phase 1 Python syntax before execution**

Run:

```bash
pixi run python -m py_compile src/tommos/*.py tests/*.py benchmarking/*.py examples/evaluate_materials/*.py examples/benchmark_1/benchmark1_workflow.py develop/strain/test_stoner_wohlfarth_me.py
```

Expected: exit code `0`.

- [ ] **Step 4: Run lint and formatting checks**

Run:

```bash
pixi run ruff check src/tommos tests
pixi run ruff format --check src/tommos tests
```

Expected: both commands pass without modifying files.

- [ ] **Step 5: Build distributions from the final source state**

Run:

```bash
pixi run build-package
```

Expected: wheel and source distribution build successfully.

- [ ] **Step 6: Run the complete repository test task**

Run:

```bash
pixi run test
```

Expected: compilation succeeds and all tests under `tests/` pass.

- [ ] **Step 7: Run the existing sample workflow**

Run:

```bash
pixi run sample
```

Expected: the sample completes through its existing platform-specific task.

- [ ] **Step 8: Verify the final wheel outside the checkout**

Run:

```bash
final_target="$(mktemp -d /tmp/tommos-final-install.XXXXXX)"
pixi run python -m pip install --no-deps --target "$final_target" dist/tommos-0.1.0-py3-none-any.whl
cd /tmp
PYTHONPATH="$final_target" /workspace/.pixi/envs/default/bin/python -c "import tommos; import tommos.loop; import tommos.mesh"
```

Expected: all imports succeed outside the repository.

- [ ] **Step 9: Perform final read-only Git inspection**

Run:

```bash
git status --short
git diff --check
git diff --stat
git diff
```

Confirm that:

- the staging design and implementation plan remain uncommitted as instructed;
- all Python modules are under `src/tommos/`;
- `src/cpp/` remains present;
- no source-file CLI wrappers were added;
- no native binary is tracked or included in the wheel;
- no unrelated files changed.

Do not stage or commit.
