# `tommos` Packaging and Native-Build Staging Design

**Date:** 2026-07-28  
**Status:** Stage 1 is complete and verified. Stages 2 and 3 are revised from
the evidence gathered during Stage 1.

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
3. Existing direct invocations such as `python src/loop.py` were replaced by
   installed-module invocations such as `python -m tommos.loop`.
4. Development through Pixi uses an editable installation of the local
   `tommos` project.
5. Native-wheel integration may be delayed. The initial packaging change will
   not claim that a wheel contains or installs the C++/oneMKL backend.
6. Native compilation and MKL-accelerated execution are supported only on
   Linux x86-64. Linux installations obtain oneMKL from Intel's PyPI packages;
   an externally installed oneMKL is not a supported alternative.
7. macOS uses the existing portable SciPy/JAX code paths without oneMKL.
   Windows is not a target because the complete project dependency set is not
   available there.

The Python Packaging User Guide documents both the `src/<package>/` layout and
its requirement for an installation, normally an editable installation during
development: [PyPA source-layout guidance](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).
Pixi documents editable local Python dependencies:
[Pixi package specifications](https://pixi.prefix.dev/latest/concepts/package_specifications/).

## 3. Repository State After Stage 1

- [`pyproject.toml`](../../../pyproject.toml) declares the pure-Python
  distribution, while [`pixi.toml`](../../../pixi.toml) installs it editably
  and continues to provide platform-specific development and native
  dependencies.
- Python modules are under [`src/tommos/`](../../../src/tommos/) and use
  package-relative internal imports. Tests, benchmarks, examples, and samples
  consume the editable installation instead of adding `src/` to `sys.path`.
- The native library is an ordinary C++ shared library with an `extern "C"`
  interface in
  [`src/cpp/cpp_mkl_minimizer.cpp`](../../../src/cpp/cpp_mkl_minimizer.cpp).
  Python loads and binds it through `ctypes` in
  [`src/tommos/cpp_minimizer.py`](../../../src/tommos/cpp_minimizer.py).
- Native-library discovery and `ctypes` declarations are duplicated between
  [`src/tommos/cpp_minimizer.py`](../../../src/tommos/cpp_minimizer.py) and
  [`src/tommos/amg_utils.py`](../../../src/tommos/amg_utils.py). This
  duplication must be removed before changing the build backend.
- There are two distinct MKL-dependent capabilities: the repository-built
  `libcpp_mkl_minimizer` library and the `sparse_dot_mkl` Python wrapper used by
  the default Linux sparse-matrix backend. `sparse-dot-mkl` itself requires a
  separately loadable oneMKL runtime:
  [sparse-dot-mkl requirements](https://pypi.org/project/sparse-dot-mkl/).
- The current native build is coupled to Pixi/Conda paths, writes outside the
  Python package, and enables `-march=native`; see
  [`src/cpp/CMakeLists.txt`](../../../src/cpp/CMakeLists.txt) and
  [`compile_local.sh`](../../../compile_local.sh).
- The sample workflow writes results into the checkout. Replacement CI commands
  must run samples in an isolated output directory and verify that the source
  tree remains unchanged.
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

### 5.1 Resulting source tree

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

Stage 1 extended `pyproject.toml` with:

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

Stage 1 kept `pixi.toml` and added the local project as an editable Python
dependency. Existing environment-specific CPU, CUDA, MKL, compiler, and sample
tasks remain under Pixi until their replacements have been designed and tested.

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

## 6. Stage 2: Native Build Integration

Stage 2 is divided into three milestones. The supported native deliverable is a
Linux x86-64 build whose build and runtime oneMKL dependencies come from
Intel's PyPI packages. macOS remains a portable-only target, and Windows is out
of scope. A publishable native wheel is a later milestone and must not be
implied by merely changing the build backend.

### 6.1 Capability boundaries

The implementation and tests must distinguish four runtime capabilities:

| Capability | Current provider | Required absence behavior |
| --- | --- | --- |
| Portable Python execution | JAX and SciPy backends | Remains usable without MKL |
| MKL sparse operations | `sparse_dot_mkl` plus a loadable oneMKL runtime | Automatic selection falls back; an explicit MKL request fails clearly |
| C++ minimizer | `libcpp_mkl_minimizer` minimizer symbols plus oneMKL and OpenMP | Automatic selection uses the Python minimizer; an explicit C++ request fails clearly |
| PARDISO Poisson solver | `libcpp_mkl_minimizer` PARDISO symbols plus oneMKL and OpenMP | Automatic selection uses the JAX Poisson solver; an explicit PARDISO request fails clearly |

Availability of one native capability must not be used as evidence that another
is available. Capability probes must return both availability and provenance so
that tests can assert which implementation executed.

The C++ minimizer and PARDISO share one library file but require separate symbol
and ABI probes. Automatic C++ minimizer selection records either `cpp_mkl` or
`python` as provenance. Automatic Poisson selection records either `pardiso` or
`jax`. Explicit `--cpp-mkl` and `--poisson-solver pardiso` requests fail before
simulation setup when their respective probe fails; explicit portable choices
do not probe the native operation.

The CPU sparse-backend default should become capability-based `auto` selection
after numerical parity tests exist. On a CPU, `auto` selects `persistent_mkl`
only when importing `sparse_dot_mkl` and loading its oneMKL runtime both
succeed; otherwise it selects `scipy`. The selection result records
`persistent_mkl` or `scipy` as provenance. Explicit backend names retain strict
behavior and never fall back silently. GPU sparse execution retains its
existing JAX path and is not inferred from the CPU capability probe.

### 6.2 Stage 2A: Central native loader and ABI contract

Complete this milestone while the existing Pixi compilation workflow remains
authoritative. Do not change the Python build backend yet.

Create one private lazy loader used by both `cpp_minimizer` and `amg_utils`.
Its lookup order will be:

1. the exact absolute file supplied through `TOMMOS_NATIVE_LIBRARY`;
2. the existing Slurm temporary-build location;
3. the existing `MUMAG_LIB_OUT` directory;
4. a library packaged under `tommos/_native/`;
5. the repository-level `lib/` compatibility location.

This preserves the current Slurm-before-`MUMAG_LIB_OUT` precedence and allows
both existing HPC mechanisms to override an installed package library until
their Stage 3 migration is complete. `TOMMOS_NATIVE_LIBRARY` is the only new
higher-priority explicit override.

The loader will:

- use `importlib.resources` for packaged-library discovery and keep any
  extraction context alive for the lifetime of the loaded library;
- load lazily rather than at module import;
- centralize `ctypes` loading, argument types, return types, and native-handle
  ownership;
- expose the selected path and selection reason for diagnostics and tests;
- preserve the original loader exception when all candidates fail;
- separate probing from an explicit request so that only automatic selection
  may fall back.

Add a small exported ABI-version function to the existing C library. Python
will verify the value before binding other symbols. Existing symbol names,
array dtypes, CSR index types, and handle lifetimes remain unchanged.

Tests for this milestone must cover lookup precedence, missing and incompatible
libraries, lazy loading, explicit-request failures, automatic fallback, ABI
mismatch, and separate probing of `sparse_dot_mkl`.

### 6.3 Stage 2B: Reproducible PyPI-oneMKL build

After Stage 2A is green, replace setuptools with `scikit-build-core`, which
documents packaging CMake-built libraries consumed through `ctypes`:
[scikit-build-core `ctypes` guide](https://scikit-build-core.readthedocs.io/en/latest/guide/ctypes.html).

Modernize CMake so that it:

1. discovers the PyPI-installed `mkl-devel` distribution rather than using
   hard-coded `CONDA_PREFIX` paths or searching for an external oneMKL;
2. links the versioned dynamic oneMKL runtime exposed by that distribution,
   using its CMake configuration where available;
3. retains OpenMP as an explicit dependency;
4. uses `install(TARGETS ...)` to place the library under `tommos/_native/`;
5. disables `-march=native`, `-ffast-math`, and similar host-specific flags by
   default;
6. keeps all build products outside the source tree.

Intel documents `MKLConfig.cmake`, `find_package(MKL CONFIG REQUIRED)`, and the
`MKL::MKL` imported target:
[Intel oneMKL CMake configuration](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2025-0/cmake-config-for-onemkl.html).

The build contract must declare `mkl-devel` as a Linux x86-64 build dependency
and discover its installed files without requiring `MKLROOT`. Intel states that
PyPI installation does not set `MKLROOT` and that the Linux `mkl-devel` package
does not provide unversioned dynamic-library symlinks:
[Intel oneMKL PyPI guidance](https://www.intel.com/content/www/us/en/docs/onemkl/get-started-guide/2024-1/overview.html).
Before the CMake change, inspect the installed distribution files and shared
object names in a clean build environment. Use that evidence to select the
exact versioned library and to make the CMake lookup deterministic.

The existing Pixi path remains available during this milestone as a tested
compatibility environment, but CMake must no longer depend on Pixi-specific
directory layouts. An externally managed oneMKL installation is not part of the
build contract.

Artifacts produced here are for local installation and CI validation. They
must not be published as generally usable native wheels until Stage 2C passes.

### 6.4 Stage 2C: Native-wheel delivery

Publish a Linux x86-64 native wheel that declares platform-conditioned runtime
dependencies on Intel's `mkl` wheel and `sparse-dot-mkl`. Do not copy oneMKL
libraries into the `tommos` wheel, statically link oneMKL, or add an
external-runtime mode. Intel documents distinct `mkl`, `mkl-devel`, and
`mkl-include` PyPI packages:
[Intel oneMKL installation options](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html).

The native wheel target is Linux x86-64 with a conservative, explicit CPU
baseline. The macOS wheel remains portable and excludes the Linux-only MKL
dependencies. Windows and Linux ARM wheels are not targets. CUDA remains a
separate Python dependency variant and does not change the CPU native-library
platform scope.

The Linux wheel must address both MKL-dependent capabilities: packaging
`libcpp_mkl_minimizer` alone does not make the default accelerated sparse path
available. The Linux dependency set therefore includes both `mkl` and
`sparse-dot-mkl`; macOS includes neither.

Do not embed an absolute build-environment path in the native library. First
inspect where the `mkl` wheel installs its versioned runtime and its ELF
`SONAME`. Then either use a stable `$ORIGIN`-relative `RUNPATH`, if the installed
layout supports one, or locate and preload the installed runtime before loading
`libcpp_mkl_minimizer`. Validate the selected mechanism from a clean virtual
environment without `MKLROOT` or `LD_LIBRARY_PATH`.

Use `cibuildwheel` or an equivalently reviewed pipeline only after the
PyPI-oneMKL loading mechanism and CPU baseline are reviewed:
[official cibuildwheel documentation](https://github.com/pypa/cibuildwheel).

## 7. Stage 3: Replace Pixi Responsibilities and Remove `pixi.toml`

Stage 3 replaces Pixi one responsibility at a time. Deleting `pixi.toml` is the
last action, not the mechanism used to discover missing responsibilities.

### 7.1 Replacement matrix

| Pixi responsibility | Required replacement |
| --- | --- |
| Install `tommos` and runtime Python dependencies | `pyproject.toml` project dependencies and reviewed project extras, including the existing `io` extra |
| Development, test, lint, and build dependencies | Standard `[dependency-groups]` entries in `pyproject.toml` |
| Cross-platform environment resolution and locking | Versioned lock artifacts for every supported portable environment, plus documented regeneration and frozen-install checks |
| CPU and CUDA JAX variants | Separate documented installation and CI procedures with mutually exclusive validation |
| CMake, compiler, OpenMP, and oneMKL | The Stage 2 Linux build contract using PyPI `mkl-devel`, plus a pinned native CI environment |
| `mkl` and `sparse_dot_mkl` | Linux-only project dependencies plus the portable macOS fallback |
| Neper, Gmsh, and other external tools | Documented system prerequisites and preflight diagnostics |
| Pixi tasks | Direct `python -m`, CMake, and shell commands; do not add a task runner unless repetition justifies one |
| Linux and macOS environments | Virtual-environment CI on both platforms, with native MKL tests restricted to Linux |
| Slurm behavior | Maintained Linux build/run documentation and scripts using the same PyPI-managed oneMKL contract |
| Sample execution and cleanup | Isolated temporary output directories and a clean-worktree assertion |

Dependency Groups are intended for internal activities such as testing and
linting without becoming wheel metadata:
[PyPA Dependency Groups specification](https://packaging.python.org/en/latest/specifications/dependency-groups/).

### 7.2 Migration order

1. Keep runtime features such as `io` under `[project.optional-dependencies]`
   and add dependency groups only for internal build, test, and lint workflows.
2. Approve a lock format/tool and produce versioned, reproducible locks for the
   supported portable environments.
3. Document direct commands for creating portable CPU and CUDA environments.
4. Add Linux native and macOS portable CI jobs that install the package,
   extras, and dependency groups without Pixi.
5. Add the pinned Stage 2 Linux native-wheel build job without Pixi.
6. Make sample and benchmark smoke tests write only to temporary directories,
   covering the maintained Linux and macOS invocations.
7. Replace Pixi-dependent Slurm activation and compilation behavior with the
   same PyPI-managed oneMKL build used outside Slurm.
8. Run the Pixi and replacement workflows in parallel until their required
   outputs and tests agree.
9. Remove `pixi.toml` and obsolete activation code in a final isolated change.

The replacement should prefer standardized package metadata and direct tool
commands. A new environment manager or task runner requires a concrete missing
capability and separate approval.

### 7.3 Removal gate

`pixi.toml` may be removed only when:

- clean locked Linux and macOS environments can build, install, test, lint, and
  run their maintained sample commands without Pixi;
- the selected lock tool can regenerate the versioned artifacts, and frozen
  installations reproduce the tested dependency sets;
- a clean pinned native environment can exercise MKL sparse operations, the C++
  minimizer, and PARDISO without Pixi;
- the CPU and CUDA procedures are independently validated on their supported
  platforms;
- CI contains no Pixi setup or command;
- repository documentation and maintained scripts contain no active Pixi
  instructions;
- Slurm users have an explicit compiler, PyPI-oneMKL, and build-directory
  contract;
- sample and benchmark checks leave no new or modified files in the checkout;
- a read-only repository search finds no remaining active dependency on
  `PIXI_PROJECT_ROOT` or `.pixi/`.

## 8. Verification Gates

### 8.1 Stage 1

Stage 1 is complete in the current working tree. The requirements and commands
are recorded in
[`docs/superpowers/plans/2026-07-28-tommos-phase-1.md`](../plans/2026-07-28-tommos-phase-1.md).
Local verification on 2026-07-28 produced the following evidence:

- `pixi run build-package` built `tommos-0.1.0.tar.gz` and
  `tommos-0.1.0-py3-none-any.whl`;
- the wheel contained 18 Python modules and no C++ source or shared library;
- `pixi run test` compiled the existing native library and reported
  `33 passed, 14 warnings in 337.46s`;
- `pixi run sample` exited successfully after writing the compatibility output;
- Ruff check and format validation, YAML validation, shell syntax checks, and
  `git diff --check` passed;
- an isolated installation under `/tmp` imported `tommos`, `tommos.loop`, and
  `tommos.mesh` from the installed wheel rather than the checkout.

These are local verification results, not a remote CI run. The Stage 1 workflow
now includes distribution building so that its first external run can provide
durable CI evidence.

### 8.2 Stage 2A

1. Run loader and capability tests without compiling a new library.
2. Compile through the existing Pixi workflow and repeat the tests.
3. Assert exact lookup precedence and loaded-library provenance.
4. Run deterministic C++ minimizer and PARDISO computations.
5. Verify explicit requests fail and `auto` selections fall back as specified.
6. Inject an incompatible ABI value and verify rejection before symbol binding.

### 8.3 Stage 2B

1. Build from the repository and from the generated source distribution using
   `mkl-devel` in a clean isolated Linux x86-64 build environment.
2. Install outside the checkout with repository and Slurm overrides unset.
3. Assert that the installed library path is under the installed `tommos`
   package.
4. Run deterministic computations through the C++ minimizer, PARDISO, and
   `sparse_dot_mkl` paths and compare them with portable references.
5. Build with default flags on a conservative CI CPU and inspect the compiler
   command to reject `-march=native` and undocumented fast-math flags.
6. Run the portable test suite with all native libraries deliberately absent.
7. Confirm that the build does not search for or link an externally installed
   oneMKL.

### 8.4 Stage 2C

For every native wheel:

1. assert that it has platform tags rather than `py3-none-any`;
2. inspect the final wheel and all native dependencies;
3. install it in a clean compatible operating-system image;
4. unset repository, Slurm, and cluster library overrides;
5. assert the exact packaged native path and ABI loaded;
6. run deterministic native computations with fallback prohibited;
7. exercise `sparse_dot_mkl` separately from `libcpp_mkl_minimizer`;
8. inspect ELF `NEEDED`, `RUNPATH`, and resolved library paths, with `MKLROOT`
   and `LD_LIBRARY_PATH` unset;
9. repeat on the oldest supported CPU baseline;
10. build a wheel from the published source distribution and repeat the checks.

Linux repair requires an explicit check because `auditwheel` documents that
dependencies reached through runtime `ctypes`/`dlopen` loading can escape
static detection:
[auditwheel limitations](https://pypi.org/project/auditwheel/).

### 8.5 Stage 3

1. Validate project extras and each dependency group in clean environments.
2. Regenerate every lock and reproduce each environment through a frozen
   installation.
3. Run Linux native, macOS portable, and supported CUDA CI without Pixi.
4. Build against PyPI `mkl-devel` and run the native Linux tests without Pixi.
5. Run samples from temporary directories and assert a clean source tree.
6. Build both distribution formats and repeat installed-artifact tests.
7. Search maintained files for active Pixi paths and commands.
8. Remove `pixi.toml`, then repeat every replacement workflow before declaring
   Stage 3 complete.

## 9. Review Boundaries

The remaining work must be split into these independently reviewable units:

1. capability-selection tests and the central native loader;
2. ABI-version export and validation;
3. capability-based `auto` sparse-backend selection;
4. scikit-build-core and portable-default CMake integration;
5. PyPI-oneMKL Linux build documentation and CI;
6. native-wheel policy and the first approved wheel pipeline;
7. project extras and development dependency groups;
8. cross-platform lock selection, generation, and frozen-install validation;
9. direct non-Pixi commands and Linux/macOS CI;
10. CUDA, native, and hermetic sample CI replacements;
11. Slurm migration to the same PyPI-oneMKL build contract;
12. final removal of `pixi.toml` and obsolete activation code.

Each unit must pass its focused syntax, tests, artifact inspection, and
installed-package checks before the next unit begins. Native-wheel delivery and
Pixi removal must not be combined.

## 10. Settled and Deferred Decisions

The revised staging settles these decisions:

- Linux x86-64 builds use PyPI `mkl-devel` at build time and the PyPI `mkl`
  runtime; externally installed oneMKL is unsupported;
- macOS uses portable SciPy/JAX paths without MKL, and Windows is out of scope;
- portable Python, MKL sparse operations, the C++ minimizer, and PARDISO are
  separately probed runtime capabilities even though the latter two share a
  library file;
- automatic selection may fall back, while explicit native requests may not;
- no native wheel is publishable merely because scikit-build-core produces it;
- `pixi.toml` removal is gated by verified replacements rather than a date.

The following still require explicit approval before implementation:

- authoritative project license and copyright ownership;
- maintainer and project URL metadata;
- public package publication target;
- stable console-script names, if any;
- final runtime dependencies and project optional extras;
- final internal development dependency-group contents;
- native-wheel CPU baseline and platform expansion;
- the exact lock format and tool, which must be approved in Stage 3 step 2 and
  cannot remain unresolved at the Pixi-removal gate.
