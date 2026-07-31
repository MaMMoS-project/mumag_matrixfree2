# Tommos Packaging Cleanup Design

**Date:** 2026-07-30

## Objective

Reduce the staged Phase 2 work to the smallest change set required to build and
install the existing Linux C++/oneMKL library as part of the `tommos` Python
package. Preserve the Phase 1 numerical algorithms, native source, and public
interfaces.

Development uses Pixi. Linux native support is limited to conventional Pixi
and Python virtual environments. macOS continues to use the existing portable
Python paths. Windows remains unsupported.

## Scope

The retained work is limited to:

1. Python packaging metadata and a minimal Pixi development workspace.
2. CMake configuration that builds the existing C++ source against the PyPI
   oneMKL development package.
3. Installation of `libcpp_mkl_minimizer.so` under `tommos/_native`.
4. Minimal package-resource lookup at the two existing Python call sites.
5. Linux wheel construction, repair, and clean-install verification.
6. Portable macOS packaging verification.
7. Concise user and developer documentation.

The work must not intentionally change numerical algorithms, backend defaults,
CLI choices, public APIs, configuration precedence, native error contracts, or
resource ownership. Numerical preservation does not mean bitwise-identical
floating-point results across different compiler, toolchain, or compiler-flag
configurations.

## Pixi Development Workflow

Delete the standalone `pixi.toml`. Configure Pixi in `pyproject.toml` using:

- `[tool.pixi.workspace]` for `linux-64`, `osx-arm64`, and `osx-64`;
- `[tool.pixi.pypi-dependencies]` for an editable dependency on `tommos`;
- pixi environments to define dependencies for test, lint, and package-build tools;
- CPU and CUDA development environments without duplicating base runtime
  requirements;
- Pixi tasks for testing, linting, package building, and samples.

Pixi reads `project.requires-python` and `project.dependencies`; these
requirements must not be duplicated as Conda dependencies. In particular,
there is no explicit Pixi `libblas`, oneMKL, compiler, CMake, Make, or pip
dependency.

Pure Python changes are visible immediately through the editable installation.
A `ctypes` library does not trigger scikit-build-core's experimental automatic
rebuild-on-import behavior. Keep PEP 517 build isolation and do not add a manual
editable-loader rebuild task: such a rebuild would require the build backend,
CMake, compiler, and oneMKL development files to remain in the Pixi
environment, duplicating the isolated build requirements.

Changes to C++, CMake, dependency metadata, or package/build files require
`pixi reinstall -e <environment> tommos`. This is documented as a development
operation, not hidden in an activation script.

## Legacy Compilation and Slurm

Delete `compile_local.sh` and `activate_build.sh`. Remove the old `compile` and
`compile-global` tasks.

Remove all staged Slurm-specific compilation and runtime lookup behavior,
including `SLURM_JOB_ID`, `MUMAG_LIB_OUT`, per-job `/tmp` copies, repository
fallbacks, and activation-time compilation. Cluster jobs use the same
environment-installed library as other Linux executions.

Any future node-local deployment optimization is a separate feature and is
not part of Python packaging.

The removed per-job native build did not configure JAX's persistent
compilation cache. Slurm documentation therefore treats native-library
installation and JAX cache policy as separate concerns: the package is
prepared once before job submission, while an optional
`JAX_COMPILATION_CACHE_DIR` remains a cluster-specific job configuration.
`README.md` links to a focused `docs/slurm.md` transition guide rather than
embedding this detail in the general installation section.

## CMake and oneMKL Discovery

Delete `src/cpp/find_mkl.py`.

The Linux `mkl-devel` wheel installs `MKLConfig.cmake` below the active Python
environment prefix, but it does not export a scikit-build-core CMake search
entry point. CMake therefore:

1. finds the PEP 517 build interpreter;
2. asks that interpreter for `sys.prefix`;
3. uses that prefix as the exclusive `find_package(MKL CONFIG REQUIRED)`
   search root;
4. links the existing target to the oneMKL single dynamic library and GNU
   OpenMP using CMake targets, leaving oneMKL SDL threading at its runtime
   default while GNU OpenMP applies to Tommos's own pragmas;
5. installs the shared library to `tommos/_native`.

The native build is Linux x86-64 only. It uses C++17 and conservative
distribution compiler flags; it does not use `-march=native`.

`src/cpp/cpp_mkl_minimizer.cpp` remains byte-for-byte at the Phase 1 baseline.
That source invariant means there is no intentional numerical-algorithm or
native-API change. It does not claim bitwise-identical results across different
compiler configurations.

## Runtime Linking

Use one runtime strategy only: an install-relative ELF RUNPATH.

For the supported Linux Pixi/virtual-environment layout:

- the `mkl` wheel installs `libmkl_rt.so.3` below `<prefix>/lib`;
- the Tommos library is installed below
  `<prefix>/lib/pythonX.Y/site-packages/tommos/_native`;
- `$ORIGIN/../../../..` resolves to `<prefix>/lib`.

The package performs no Python metadata scan or explicit oneMKL preload.
`ctypes` loads only the packaged Tommos library. The ELF loader resolves
`libmkl_rt.so.3` through the install-relative RUNPATH.

Wheel repair excludes `libmkl_rt.so.3`, because it is supplied by the declared
`mkl` dependency, and bundles non-MKL compiler runtimes such as `libgomp`.
After repair, the Tommos library retains both the wheel-local compiler-runtime
path and the environment-relative oneMKL path. The repair helper and tests
verify the required path components, dependencies, and absence of bundled
oneMKL libraries.

Nonstandard layouts such as `pip --target`, embedded interpreters, and system
prefix installs are outside the supported native-installation contract.

## Python Changes

Delete `src/tommos/_native_loader.py`.

Restore the Phase 1 versions of:

- `src/tommos/loop.py`;
- `src/tommos/hysteresis_loop.py`;
- `src/tommos/poisson_solve.py`;
- `src/cpp/cpp_mkl_minimizer.cpp`.

Restore `src/tommos/amg_utils.py` and `src/tommos/cpp_minimizer.py`, then make
only the minimal changes needed for their existing `ctypes` bindings to locate
`tommos/_native/libcpp_mkl_minimizer.so` through `importlib.resources`.

The following staged behavior changes are removed:

- native capability probes and automatic selection;
- CLI choice/default changes and additional `.p2` keys;
- ABI-version negotiation;
- PARDISO ownership, locking, and close protocols;
- native status-propagation changes;
- sparse-operator lifecycle changes;
- diagnostic APIs;
- Slurm, environment-variable, and repository library candidates.

The pre-existing `--poisson-solver` option remains exactly as it was after
Phase 1.

## Tests and Continuous Integration

Remove tests introduced solely for the out-of-scope runtime redesign:

- native selection;
- custom MKL discovery;
- ABI negotiation;
- PARDISO concurrency and ownership;
- changed C++ error contracts.

Retain focused packaging tests that verify:

- distribution metadata and platform markers;
- wheel and sdist contents;
- Linux native library installation under `tommos/_native`;
- no bundled oneMKL libraries;
- the expected ELF `NEEDED` entries and RUNPATH components;
- loading and executing the existing native library from a clean conventional
  virtual environment outside the checkout;
- a portable macOS wheel without native or oneMKL contents.

Existing application and numerical tests return to their Phase 1 form except
where imports must reference the installed package.

## Documentation

In `README.md`, modify only the currently staged hunks. Replace the detailed
design narrative with:

- Pixi installation and task commands;
- the supported Linux and macOS behavior;
- the explicit `pixi reinstall` development operation for native and packaging
  changes.

Do not document internal RUNPATH values, wheel-repair implementation,
diagnostic internals, historical design decisions, or removed compatibility
scripts.

Detailed packaging rationale remains in this design document and the
implementation plan.

## Repository and Review Constraints

- Do not create commits.
- Do not stage, unstage, restore, reset, or otherwise write Git state.
- Preserve unrelated user changes.
- Keep each implementation step small and inspectable.
- Syntax-check modified Python before executing it.
- Verify the final combined working tree relative to `HEAD`, because the index
  contains the earlier staged Phase 2 patch.
