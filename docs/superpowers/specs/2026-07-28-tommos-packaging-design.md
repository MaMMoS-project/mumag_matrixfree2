# `tommos` Packaging and Native-Build Staging Design

**Date:** 2026-07-28  
**Status:** Proposed implementation staging; the package layout and module-execution decisions are confirmed.

## 1. Goal

Convert the repository into an installable Python distribution named `tommos`
whose import namespace is also `tommos`, while keeping the first implementation
small, preserving existing numerical behavior, and making every change easy to
review.

Native C++/Intel oneMKL compilation will remain available through the existing
Pixi workflow during the initial package conversion. Integrating native
compilation into Python wheel builds and removing the top-level `pixi.toml` are
separate later stages.

## 2. Confirmed Decisions

1. The distribution name and import namespace are both `tommos`.
2. Python modules will use the conventional `src/tommos/` layout.
3. Existing direct invocations such as `python src/loop.py` will be replaced by
   installed-module invocations such as `python -m tommos.loop`.
4. Development through Pixi will use an editable installation of the local
   `tommos` project.
5. Native-wheel integration may be delayed. The initial packaging change will
   not claim that a wheel contains or installs the C++/oneMKL backend.

The Python Packaging User Guide documents both the `src/<package>/` layout and
its requirement for an installation, normally an editable installation during
development: [PyPA source-layout guidance](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).
Pixi documents editable local Python dependencies:
[Pixi package specifications](https://pixi.prefix.dev/latest/concepts/package_specifications/).

## 3. Current Repository Constraints

- [`pyproject.toml`](../../../pyproject.toml) currently contains Ruff
  configuration only. It does not declare a build backend, project metadata, or
  package discovery.
- Python modules are flat files under [`src/`](../../../src/) and use unqualified
  imports. Representative examples are
  [`src/loop.py`](../../../src/loop.py),
  [`src/hysteresis_loop.py`](../../../src/hysteresis_loop.py), and
  [`src/poisson_solve.py`](../../../src/poisson_solve.py).
- Tests and benchmarks add `src/` to `sys.path`, for example
  [`tests/test_gradients.py`](../../../tests/test_gradients.py) and
  [`benchmarking/profile_energy.py`](../../../benchmarking/profile_energy.py).
- The native library is an ordinary C++ shared library with an `extern "C"`
  interface in
  [`src/cpp/cpp_mkl_minimizer.cpp`](../../../src/cpp/cpp_mkl_minimizer.cpp).
  Python loads and binds it through `ctypes` in
  [`src/cpp_minimizer.py`](../../../src/cpp_minimizer.py).
- The current native build is coupled to Pixi/Conda paths, writes outside the
  Python package, and enables `-march=native`; see
  [`src/cpp/CMakeLists.txt`](../../../src/cpp/CMakeLists.txt) and
  [`compile_local.sh`](../../../compile_local.sh).
- There is no repository-level license file. A distribution license must not be
  inferred from comments in individual source files.

## 4. Rejected Layout Alternatives

### 4.1 Map the existing `src/` directory directly to `tommos`

Setuptools can map an import-package name to an arbitrarily named directory, but
its documentation advises keeping the source hierarchy identical to the
installed hierarchy where possible:
[setuptools `package_dir` documentation](https://setuptools.pypa.io/en/latest/references/keywords.html).
This option would avoid file moves but would leave the repository layout
different from the installed package layout. It was rejected in favor of an
explicit `src/tommos/` directory.

### 4.2 Place `tommos/` at the repository root

A root-level package would be directly importable from the checkout, but it
would abandon the existing source-layout direction. It was rejected because the
confirmed design uses `src/tommos/` and an editable installation.

## 5. Stage 1: Minimal Python Package Conversion

### 5.1 Intended source tree

```text
pyproject.toml
pixi.toml
src/
├── tommos/
│   ├── __init__.py
│   ├── add_shell.py
│   ├── amg_utils.py
│   ├── cpp_minimizer.py
│   ├── energy_kernels.py
│   ├── extract_nucleation.py
│   ├── fem_utils.py
│   ├── hysteresis_loop.py
│   ├── io_utils.py
│   ├── loop.py
│   ├── make_krn.py
│   ├── mesh.py
│   ├── mesh_convert.py
│   ├── minimizers.py
│   ├── plot_hysteresis.py
│   ├── poisson_solve.py
│   ├── reorder_mesh.py
│   └── salomeMeshToNpz.py
└── cpp/
    ├── CMakeLists.txt
    └── cpp_mkl_minimizer.cpp
```

The C++ source remains at `src/cpp/` in this stage so that packaging does not
unnecessarily alter the existing compilation workflow.

### 5.2 Packaging metadata

Extend `pyproject.toml` with:

- a PEP 517 build-system declaration;
- PEP 621 project metadata for `tommos`;
- Python requirement `>=3.11`, matching `pixi.toml`;
- package discovery restricted to `src/tommos`;
- runtime dependencies derived from imports used by the installed package,
  rather than copying every development and platform dependency from Pixi.

Setuptools is sufficient for this pure-Python stage. PyPA documents the
`[build-system]` and `[project]` tables in
[Writing `pyproject.toml`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).

The authoritative license, copyright holder, maintainers, project URLs, and
publication target require explicit project-owner decisions before publication.
They are not prerequisites for a private editable installation, but the design
will not invent them.

### 5.3 Imports and execution

- Convert internal imports to explicit package-relative imports.
- Convert tests, benchmarks, and development scripts from source-path injection
  and flat imports to `tommos.*` imports.
- Replace documented and scripted `python src/<module>.py` invocations with
  `python -m tommos.<module>`.
- Do not add duplicate top-level compatibility modules. Loading the same source
  under both a flat name and `tommos.<module>` could create distinct module and
  class identities.
- Do not create new console-script names in this stage unless separately
  approved. Module execution provides the requested interface without adding a
  second public naming scheme.

### 5.4 Pixi integration

Keep `pixi.toml` during this stage and add the local project as an editable
Python dependency. Existing environment-specific CPU, CUDA, MKL, compiler, and
sample tasks remain under Pixi until their replacements have been designed and
tested.

The current C++ compilation remains external to the Python wheel. Only path
adjustments required by moving Python modules into `src/tommos/` belong in this
stage. The existing Slurm temporary-build behavior and `MUMAG_LIB_OUT` override
remain supported.

### 5.5 Stage 1 exclusions

The following are explicitly outside the initial package conversion:

- compiling C++ through the Python build backend;
- bundling `libcpp_mkl_minimizer.so` or oneMKL libraries in a wheel;
- changing the C ABI or replacing `ctypes`;
- promising native support on macOS or Windows;
- removing `pixi.toml`, `compile_local.sh`, or `activate_build.sh`;
- unrelated refactoring, public-API redesign, or numerical-method changes;
- adding or guessing license and maintainer metadata.

## 6. Stage 2: Native Compilation and Native Wheels

Stage 2 begins only after Stage 1 is installed and tested successfully and after
the native distribution policy is approved.

### 6.1 Build backend

Replace the pure-Python build backend with `scikit-build-core`, which documents
the specific case of packaging a CMake-built shared library for `ctypes`:
[scikit-build-core `ctypes` guide](https://scikit-build-core.readthedocs.io/en/latest/guide/ctypes.html).

Retain the existing C ABI initially. This avoids combining package-build work
with a binding-layer rewrite and allows a platform wheel to be independent of a
specific CPython minor ABI when the shared library does not call the Python C
API, as documented by scikit-build-core.

### 6.2 CMake integration

Revise CMake so that it:

1. discovers oneMKL through its CMake package rather than hard-coded
   `CONDA_PREFIX` paths;
2. links through the imported `MKL::MKL` target;
3. retains OpenMP as an explicit dependency;
4. installs the generated shared library into `tommos/_native/` in the wheel
   staging tree;
5. separates portable release flags from explicitly requested local/HPC
   optimization flags;
6. does not write build products into the repository.

Intel documents `MKLConfig.cmake`, `find_package(MKL CONFIG REQUIRED)`, and the
`MKL::MKL` imported target:
[Intel oneMKL CMake configuration](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2025-0/cmake-config-for-onemkl.html).

### 6.3 Native loader

Create one private, lazy loader for all uses of the native library. Its lookup
order will be:

1. an explicit absolute path supplied through `TOMMOS_NATIVE_LIBRARY`;
2. a developer/HPC directory override;
3. a library packaged under `tommos/_native/`;
4. deprecated compatibility paths for `MUMAG_LIB_OUT` and the existing Slurm
   temporary build.

The loader will:

- use `importlib.resources` for packaged-library discovery;
- keep resource extraction alive for the lifetime of the loaded library;
- centralize all `ctypes` argument and return-type declarations;
- report the selected candidate and preserve the original loader error;
- load only when a native backend is requested or probed;
- distinguish automatic fallback from an explicit native request.

For automatic backend selection, an unavailable native library permits the
existing portable backend. An explicit request for the C++/MKL backend fails
early with a diagnostic instead of silently changing algorithms.

### 6.4 ABI contract

Add and test a small exported native ABI-version function. Python will verify
that value before binding the remaining symbols. Existing exported symbol names,
array dtypes, CSR index types, and native-handle lifetime behavior remain
unchanged until a separately approved ABI migration.

### 6.5 oneMKL delivery decision

Before releasing native wheels, choose and legally verify exactly one initial
oneMKL delivery model:

1. dynamically depend on a separately installed oneMKL runtime;
2. bundle permitted runtime libraries into repaired wheels;
3. statically link permitted components;
4. keep oneMKL external and publish only source/HPC native-build instructions.

Intel lists distinct `mkl`, `mkl-devel`, and `mkl-static` packages:
[Intel oneMKL installation options](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html).
The repository does not contain sufficient licensing evidence to select a
redistribution model without review.

### 6.6 Initial native support boundary

The first native-wheel target should be limited to Linux x86-64 because that is
the only native path implemented by the current build scripts and CMake
configuration. Additional operating systems and architectures require their own
implemented library naming, compiler, OpenMP, oneMKL, and test strategy before
support is claimed.

## 7. Stage 3: Remove the Top-Level Pixi Manifest

Remove `pixi.toml` only after all of its active responsibilities have verified
replacements:

- runtime dependencies are declared in `pyproject.toml`;
- development and test dependencies have a documented installation group;
- CPU and CUDA JAX selection has a documented installation procedure;
- native build prerequisites are supplied by the wheel build or documented
  source-build contract;
- Python tests, native tests, samples, linting, and formatting have replacement
  commands;
- CI exercises those commands without Pixi;
- Slurm/HPC build and runtime behavior has a documented replacement.

The removal must be its own reviewable change. It must not be combined with the
initial source move or with the first native-wheel implementation.

## 8. Verification Gates

### 8.1 Stage 1

1. Check syntax for every modified Python file before executing Python modules.
2. Validate `pyproject.toml`.
3. Build both a wheel and source distribution.
4. Inspect their contents for missing modules and unintended files.
5. Install the wheel in a clean environment outside the repository.
6. Verify `import tommos` and representative `tommos.*` imports.
7. Verify the approved `python -m tommos.<module> --help` interfaces.
8. Run the portable unit tests against the installed package.
9. Run the existing Pixi C++/MKL and sample workflows separately.
10. Run Ruff and the repository's configured checks.

### 8.2 Stage 2

Use `cibuildwheel` or an equivalently reviewed wheel pipeline to build and test
each supported platform artifact. `cibuildwheel` documents installed-wheel
testing and native-library repair tools:
[official cibuildwheel documentation](https://github.com/pypa/cibuildwheel).

For every native wheel:

1. inspect the final wheel and its native dependencies;
2. install it outside the checkout with repository and cluster library
   overrides unset;
3. load the packaged native library;
4. run a deterministic native computation;
5. verify explicitly that the native path, not a fallback, executed;
6. repeat on the oldest supported CPU baseline and a clean compatible operating
   system image;
7. build a wheel from the published source distribution and repeat the native
   smoke test.

Linux repair requires an explicit check because `auditwheel` documents that
dependencies reached through runtime `ctypes`/`dlopen` loading can escape static
detection:
[auditwheel limitations](https://pypi.org/project/auditwheel/).

## 9. Review and Commit Boundaries

Implementation must be split into independently reviewable and tested commits.
At minimum:

1. package structure and import migration;
2. packaging metadata and editable Pixi installation;
3. command, test, benchmark, sample, and documentation migration;
4. native source-workflow compatibility adjustments;
5. build/install verification and CI coverage.

The exact commit split will be finalized in the implementation plan. Each code
commit must pass syntax checks and its relevant tests before the next task
begins.

## 10. Deferred Decisions

The following decisions require explicit approval before the corresponding
implementation:

- authoritative project license and copyright ownership;
- maintainer and project URL metadata;
- public package publication target;
- stable console-script names, if any;
- base and optional Python dependency groups;
- oneMKL acquisition and redistribution policy;
- native wheel platforms and CPU instruction baseline;
- timing and replacement tooling for removing Pixi.
