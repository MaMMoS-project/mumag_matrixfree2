# Tommos Packaging Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the staged Phase 2 patch to a human-reviewable Python
packaging change that builds the existing Linux C++/oneMKL library into
`tommos/_native` without intentionally changing application algorithms or
public APIs.

**Architecture:** `pyproject.toml` is the sole Python/Pixi manifest.
scikit-build-core invokes a Linux-only CMake build in PEP 517 isolation, while
macOS produces a portable pure-Python wheel. The two existing `ctypes` call
sites locate the installed shared library with `importlib.resources`; ELF
RUNPATH resolves the separately installed PyPI oneMKL runtime.

**Tech Stack:** Python 3.11+, Pixi, scikit-build-core, CMake, C++17, Intel
oneMKL 2026, GNU OpenMP, auditwheel, patchelf, pytest, Ruff.

## Global Constraints

- The import package and distribution name are exactly `tommos`, with source
  below `src/tommos`.
- Existing commands run from an editable installation as
  `python -m tommos.<module>`.
- Linux x86-64 is the only native-build target; macOS uses the existing
  portable Python paths; Windows is unsupported.
- `src/cpp/cpp_mkl_minimizer.cpp` must be byte-for-byte identical to the Phase
  1 `HEAD` baseline.
- Numerical algorithms, backend defaults, CLI choices, public APIs,
  configuration precedence, native error contracts, and resource ownership
  must not intentionally change. This does not promise bitwise-identical
  floating-point results across different compiler, toolchain, or
  compiler-flag configurations.
- oneMKL is a Linux-only PyPI dependency. The Tommos wheel must not bundle
  `libmkl_rt.so.3`.
- The native library installs as
  `tommos/_native/libcpp_mkl_minimizer.so`.
- Runtime discovery uses only `importlib.resources` and install-relative ELF
  RUNPATH. Do not add metadata scanning, preloading, environment variables,
  repository fallbacks, or Slurm handling.
- `pyproject.toml` contains Pixi features and environments for test, lint,
  package-build, CPU, CUDA, and sample workflows. Do not use PEP dependency
  groups for development tools.
- Do not duplicate project runtime dependencies as Pixi Conda dependencies.
  Do not add explicit Pixi `libblas`, oneMKL, compiler, CMake, Make, or pip
  dependencies.
- Native, CMake, dependency-metadata, and package-file changes are rebuilt with
  `pixi reinstall -e <environment> tommos`; there is no compile or
  `native-rebuild` task.
- Delete `pixi.toml`, `compile_local.sh`, `activate_build.sh`,
  `src/cpp/find_mkl.py`, and `src/tommos/_native_loader.py`.
- Modify only README hunks already changed in the staged Phase 2 patch.
- Do not create commits or write Git state. Git commands are read-only.
- Preserve unrelated user changes. Use `apply_patch` for repository edits.
- Syntax-check every modified Python file before executing Python tests.
- Verify the combined working tree against `HEAD`, because the index contains
  the earlier staged Phase 2 patch.

---

## File Map

- `pyproject.toml`: build metadata, runtime dependencies, scikit-build-core
  configuration, and all Pixi features/environments/tasks.
- `src/cpp/CMakeLists.txt`: Linux x86-64 native build, oneMKL discovery, target
  linkage, install location, and install RUNPATH.
- `src/tommos/amg_utils.py`: existing PARDISO binding with only its shared
  library path changed to a package resource.
- `src/tommos/cpp_minimizer.py`: existing minimizer binding with only its
  shared library path changed to a package resource.
- `.github/scripts/repair_linux_wheel.sh`: auditwheel repair excluding oneMKL
  and preservation of both required RUNPATH entries.
- `.github/workflows/test.yml`: Pixi test/build verification on Linux and
  macOS.
- `.github/workflows/wheels.yml`: manylinux build, clean-install test, and ELF
  artifact checks.
- `tests/test_distribution_metadata.py`: distribution requirements and platform
  marker tests.
- `tests/test_wheel_install.py`: sdist, wheel, native artifact, RUNPATH, and
  clean-install behavior.
- `README.md`: concise installation and development commands, restricted to
  already staged hunks.

### Task 1: Restore the Phase 1 application baseline

**Files:**

- Restore: `src/cpp/cpp_mkl_minimizer.cpp`
- Restore: `src/tommos/loop.py`
- Restore: `src/tommos/hysteresis_loop.py`
- Restore: `src/tommos/poisson_solve.py`
- Restore: `src/tommos/amg_utils.py`
- Restore: `src/tommos/cpp_minimizer.py`
- Restore: `tests/test_compare_kernels.py`
- Restore: `tests/test_cpp_minimizer.py`
- Restore: `tests/test_mkl.py`
- Restore: `tests/test_mkl_jit.py`
- Restore: `tests/test_package_imports.py`
- Restore: `tests/test_py_minimizer.py`
- Delete: `src/tommos/_native_loader.py`
- Delete: `tests/test_cpp_minimizer_contract.py`
- Delete: `tests/test_find_mkl.py`
- Delete: `tests/test_native_loader.py`
- Delete: `tests/test_native_selection.py`

**Interfaces:**

- Consumes: Phase 1 file contents from `git show HEAD:<path>`.
- Produces: the original application algorithms and public APIs, ready for the
  narrowly scoped resource-path edits in Task 3.

- [ ] **Step 1: Record the files whose staged changes exceed packaging scope**

  Run `git diff --name-status HEAD` and compare the listed application/runtime
  files to the Global Constraints.

- [ ] **Step 2: Restore tracked files through patches generated from read-only
  `git show HEAD:<path>` output**

  The resulting files must exactly match `HEAD` before Task 3 changes
  `amg_utils.py` and `cpp_minimizer.py`.

- [ ] **Step 3: Delete the staged runtime-redesign modules and tests**

  Remove only the five paths listed under **Delete**.

- [ ] **Step 4: Verify the restoration**

  Run:

  ```bash
  git diff --quiet HEAD -- \
    src/cpp/cpp_mkl_minimizer.cpp \
    src/tommos/loop.py \
    src/tommos/hysteresis_loop.py \
    src/tommos/poisson_solve.py \
    tests/test_compare_kernels.py \
    tests/test_cpp_minimizer.py \
    tests/test_mkl.py \
    tests/test_mkl_jit.py \
    tests/test_package_imports.py \
    tests/test_py_minimizer.py
  ```

  Expected: exit status 0.

  Confirm the five deleted paths are absent. Do not require the broad removal
  search to be clean yet: the Phase 1 versions of `amg_utils.py` and
  `cpp_minimizer.py` intentionally retain their old path logic until Task 3,
  and `test_wheel_install.py` is reduced in Task 4.

### Task 2: Consolidate packaging and Pixi configuration

**Files:**

- Modify: `pyproject.toml`
- Modify: `tests/test_distribution_metadata.py`
- Delete: `pixi.toml`
- Delete: `compile_local.sh`
- Delete: `activate_build.sh`

**Interfaces:**

- Consumes: the existing `[project]` runtime requirements and optional `io`
  extra.
- Produces: a single manifest with an editable `tommos` dependency; Pixi
  environments named `default`, `test`, `lint`, `build`, `sample`, and `cuda`;
  tasks named `test`, `lint`, `build-package`, `sample`, `sample-gpu`, and
  `clean-samples`.

- [ ] **Step 1: Rewrite metadata tests to express the retained packaging
  contract**

  Tests must assert:

  - the distribution is named `tommos`;
  - Python requires at least 3.11;
  - Linux x86-64 markers guard both `mkl` and `sparse-dot-mkl`;
  - no Windows native dependency is present;
  - `src/cpp/find_mkl.py` is absent from `sdist.include`;
  - Pixi platforms are `linux-64`, `osx-arm64`, and `osx-64`;
  - Pixi defines separate test/lint/build environments without a
    `[dependency-groups]` table.

- [ ] **Step 2: Syntax-check the modified metadata test**

  Run `pixi run python -m py_compile tests/test_distribution_metadata.py`.

- [ ] **Step 3: Run the metadata test and verify the old manifest fails the new
  contract**

  Run `pixi run pytest -q tests/test_distribution_metadata.py`.

  Expected before implementation: failure because the configuration is still
  split across `pixi.toml` and `pyproject.toml`.

- [ ] **Step 4: Update `pyproject.toml`**

  Keep the project runtime dependencies in `[project.dependencies]`. Keep
  `scikit-build-core` and Linux `mkl-devel` in `[build-system.requires]`.
  Configure:

  ```toml
  [tool.pixi.workspace]
  channels = ["conda-forge"]
  platforms = ["linux-64", "osx-arm64", "osx-64"]

  [tool.pixi.pypi-dependencies]
  tommos = { path = ".", editable = true }

  [tool.pixi.feature.test.pypi-dependencies]
  pytest = "*"

  [tool.pixi.feature.lint.dependencies]
  ruff = "*"
  pre-commit = ">=4.6.0,<5"

  [tool.pixi.feature.build.pypi-dependencies]
  build = "*"
  cibuildwheel = "==4.1.1"
  ```

  Put sample-only packages in a `sample` feature. Keep the CUDA JAX extra in a
  `cuda` feature and do not repeat the base CPU JAX requirement.

  Define separate Pixi environments using those features and common solve
  groups. Define test/lint/build/sample tasks inside the corresponding features.
  On macOS, override the sample command with the existing portable arguments.
  Do not define `compile`, `compile-global`, or `native-rebuild`.

- [ ] **Step 5: Delete the standalone manifest and legacy compilation scripts**

  Delete exactly `pixi.toml`, `compile_local.sh`, and `activate_build.sh`.

- [ ] **Step 6: Validate and test the consolidated manifest**

  Run:

  ```bash
  pixi project export conda-environment --environment test >/dev/null
  pixi run -e test pytest -q tests/test_distribution_metadata.py
  ```

  Expected: both commands exit 0.

### Task 3: Implement the minimal native build and resource lookup

**Files:**

- Modify: `src/cpp/CMakeLists.txt`
- Modify: `src/tommos/amg_utils.py`
- Modify: `src/tommos/cpp_minimizer.py`
- Delete: `src/cpp/find_mkl.py`
- Test: `tests/test_compare_kernels.py`
- Test: `tests/test_cpp_minimizer.py`
- Test: `tests/test_mkl.py`

**Interfaces:**

- Consumes: the Linux `mkl-devel` build requirement and installed package
  resource `tommos/_native/libcpp_mkl_minimizer.so`.
- Produces: existing `ctypes.CDLL` bindings loaded from
  `importlib.resources.files("tommos").joinpath("_native",
  "libcpp_mkl_minimizer.so")`.

- [ ] **Step 1: Add a focused installed-resource assertion before changing the
  bindings**

  Add this Linux-only test to `tests/test_cpp_minimizer.py`:

  ```python
  def test_native_library_is_installed_as_package_resource() -> None:
      """Load the installed native library through the package resource."""
      native_library = files("tommos").joinpath(
          "_native", "libcpp_mkl_minimizer.so"
      )

      assert native_library.is_file()
      ctypes.CDLL(str(native_library))
  ```

  Import `files` from `importlib.resources`. Do not add native selection,
  status, ABI, ownership, or lifecycle behavior.

- [ ] **Step 2: Syntax-check the modified tests**

  Run:

  ```bash
  pixi run -e test python -m py_compile \
    tests/test_cpp_minimizer.py tests/test_mkl.py
  ```

- [ ] **Step 3: Run the focused test and verify it fails while bindings still
  use repository paths**

  Run:

  ```bash
  pixi run -e test pytest -q tests/test_cpp_minimizer.py tests/test_mkl.py
  ```

  Expected before implementation: the installed package-resource assertion
  fails.

- [ ] **Step 4: Replace `CMakeLists.txt` with the minimal Linux build**

  The CMake file must:

  - require CMake 3.22 and C++17;
  - reject non-Linux and non-x86-64 native builds;
  - find the build Python interpreter;
  - obtain its `sys.prefix`;
  - set `MKL_ROOT` and `MKL_DIR` to that prefix's installed oneMKL paths;
  - set `MKL_LINK` to `sdl` and `MKL_INTERFACE` to `lp64`, leaving
    `MKL_THREADING` unset so oneMKL SDL uses its runtime default;
  - call `find_package(MKL CONFIG REQUIRED NO_DEFAULT_PATH)`;
  - link `MKL::mkl_rt` and `OpenMP::OpenMP_CXX`, with GNU OpenMP applying to
    Tommos's own pragmas rather than selecting oneMKL's SDL threading layer;
  - compile with `-O3` but not `-march=native`;
  - install `libcpp_mkl_minimizer.so` below `tommos/_native`;
  - set install RUNPATH to `$ORIGIN/../../../..`;
  - disable link-path-derived install RUNPATH additions.

- [ ] **Step 5: Delete `src/cpp/find_mkl.py`**

  CMake must contain the complete supported discovery path; no Python discovery
  helper remains.

- [ ] **Step 6: Change only the two existing production library paths**

  Add `from importlib.resources import files` to each binding module and replace
  the old Slurm/environment/repository path logic with:

  ```python
  library_path = files("tommos").joinpath(
      "_native", "libcpp_mkl_minimizer.so"
  )
  ```

  Pass `str(library_path)` to the existing `ctypes.CDLL` calls. Preserve all
  surrounding function signatures, error behavior, and ownership logic from
  `HEAD`.

- [ ] **Step 7: Update the numerical comparison test's obsolete fixture paths**

  Preserve the complete numerical setup, native calls, and assertions in
  `tests/test_compare_kernels.py`. Change only its repository-local native
  library lookup to the installed `tommos/_native` resource and its
  unversioned oneMKL lookup to `<sys.prefix>/lib/libmkl_rt.so.3`.

- [ ] **Step 8: Syntax-check Python before execution**

  Run:

  ```bash
  pixi run -e test python -m py_compile \
    src/tommos/amg_utils.py src/tommos/cpp_minimizer.py \
    tests/test_compare_kernels.py tests/test_cpp_minimizer.py tests/test_mkl.py
  ```

- [ ] **Step 9: Reinstall and run focused native tests**

  Run:

  ```bash
  pixi reinstall -e test tommos
  pixi run -e test pytest -q \
    tests/test_compare_kernels.py tests/test_cpp_minimizer.py tests/test_mkl.py
  ```

  Expected: build and tests exit 0.

### Task 4: Repair and verify Linux wheels

**Files:**

- Modify: `.github/scripts/repair_linux_wheel.sh`
- Modify: `.github/workflows/wheels.yml`
- Modify: `tests/test_wheel_install.py`

**Interfaces:**

- Consumes: an unrepaired Linux wheel containing
  `tommos/_native/libcpp_mkl_minimizer.so`.
- Produces: a manylinux wheel that bundles its renamed GNU OpenMP runtime,
  excludes oneMKL, and retains RUNPATH components for both `tommos.libs` and
  the environment prefix.

- [ ] **Step 1: Reduce `test_wheel_install.py` to artifact and clean-install
  behavior**

  Retain tests for:

  - one wheel and one sdist built from the project;
  - native Linux wheel path and absence of bundled oneMKL;
  - portable macOS wheel without native/oneMKL contents;
  - clean installation outside the checkout;
  - `ctypes` loading and existing native execution;
  - ELF `NEEDED` entries and both required RUNPATH components.

  Remove tests for the deleted loader, diagnostic APIs, capability selection,
  ABI negotiation, and environment/Slurm candidates.

- [ ] **Step 2: Syntax-check the wheel test**

  Run `pixi run -e test python -m py_compile tests/test_wheel_install.py`.

- [ ] **Step 3: Run the source-distribution test before changing the repair
  helper**

  Run the artifact subset available on the host:

  ```bash
  pixi run -e build python -m build --sdist
  pixi run -e test pytest -q tests/test_wheel_install.py -k sdist
  ```

- [ ] **Step 4: Simplify the repair helper**

  Keep `auditwheel repair --exclude libmkl_rt.so.3`. After repair, add
  `$ORIGIN/../../../..` with patchelf without `--force-rpath`, verify path
  components rather than exact ordering, and repack the wheel.

- [ ] **Step 5: Simplify manylinux CI**

  Build CPython 3.11 through 3.14 x86-64 manylinux wheels. Clean-install tests
  must unset only generic loader variables (`MKLROOT`, `LD_LIBRARY_PATH`, and
  `PYTHONPATH`), run outside the checkout, and execute the retained wheel/native
  tests. Artifact inspection must assert:

  - `libcpp_mkl_minimizer.so` exists;
  - a renamed `libgomp` exists under `tommos.libs`;
  - no `libmkl*.so*` exists in the wheel;
  - ELF `NEEDED` includes `libmkl_rt.so.3` and renamed `libgomp`;
  - ELF `RUNPATH` contains both `$ORIGIN/../../tommos.libs` and
    `$ORIGIN/../../../..`.

- [ ] **Step 6: Build and inspect a Linux wheel locally**

  Run:

  ```bash
  pixi run -e build python -m build --wheel
  ```

  Then run the wheel artifact tests that do not require a repaired manylinux
  wheel. If the container supports auditwheel repair, run the repair helper and
  the complete Linux artifact subset.

### Task 5: Clean CI and documentation

**Files:**

- Modify: `.github/workflows/test.yml`
- Modify: `README.md`
- Modify: `PACKAGING_REVIEW_NOTES.md`
- Create: `docs/slurm.md`
- Restore: `docs/superpowers/specs/2026-07-28-tommos-packaging-design.md`
- Delete: `docs/superpowers/plans/2026-07-29-tommos-phase-2.md`
- Retain: `docs/superpowers/specs/2026-07-30-tommos-packaging-cleanup-design.md`
- Retain: `docs/superpowers/plans/2026-07-30-tommos-packaging-cleanup.md`

**Interfaces:**

- Consumes: Pixi environment/task names from Task 2 and wheel tests from Task 4.
- Produces: concise current development instructions, a focused Slurm
  transition guide, and portable Linux/macOS CI with no operational references
  to removed runtime/compilation machinery.

- [ ] **Step 1: Restore the obsolete revised design and remove its obsolete
  Phase 2 plan**

  Restore `docs/superpowers/specs/2026-07-28-tommos-packaging-design.md` exactly
  to `HEAD`. Delete `docs/superpowers/plans/2026-07-29-tommos-phase-2.md`.

- [ ] **Step 2: Simplify `test.yml`**

  Run the build task in the build environment and tests in the test
  environment. Keep the macOS clean-wheel verification, but remove references
  to `TOMMOS_NATIVE_LIBRARY`, `MUMAG_LIB_OUT`, and `SLURM_JOB_ID`.

- [ ] **Step 3: Replace staged README installation hunks**

  Document:

  ```bash
  pixi install
  pixi run test
  pixi run lint
  pixi run build-package
  pixi reinstall -e test tommos
  ```

  State only that Linux x86-64 installs the native oneMKL backend and macOS uses
  portable paths. Explain that reinstall is needed after C++, CMake,
  dependency-metadata, or package-file changes. Restore every other staged
  README hunk to `HEAD`, except for the stale Slurm block explicitly approved
  for correction after final review.

- [ ] **Step 4: Replace the stale Slurm build instructions**

  In the README Slurm example:

  - replace the per-job compilation workflow description with preparation of
    the shared Pixi environment before job submission;
  - remove `pixi run compile` and
    `rm -rf /tmp/mumag_build_${SLURM_JOB_ID}`;
  - run the simulation with `pixi run -e cpu python -m tommos.loop`;
  - link to `docs/slurm.md`.

  In `docs/slurm.md`, document:

  - `pixi install -e cpu` and `pixi reinstall -e cpu tommos` as environment
    preparation outside concurrently running jobs;
  - the transition from per-job `/tmp/mumag_build_${SLURM_JOB_ID}` C++ builds
    to the environment-installed package resource;
  - that the removed Slurm logic never configured JAX's persistent compilation
    cache;
  - that `--benchmark` performs an in-process warm-up;
  - optional node-local and trusted shared `JAX_COMPILATION_CACHE_DIR`
    strategies, linked to the official JAX persistent-cache guide;
  - that shared cache policy is cluster-specific and separate from Tommos
    packaging.

- [ ] **Step 5: Verify removed concepts are absent from retained modified
  files**

  Before verification, add one short review-note bullet explaining that local
  auditwheel repair requires a compatible manylinux build environment: this
  host-built glibc 2.34 wheel cannot be repaired to manylinux_2_28, so the
  genuine repair remains a CI integration check. Do not describe the synthetic
  temporary ELF check as a successful wheel repair.

  Run:

  ```bash
  rg -n \
    "compile_local|activate_build|native-rebuild|_native_loader|find_mkl|SLURM_JOB_ID|MUMAG_LIB_OUT|TOMMOS_NATIVE_LIBRARY" \
    pyproject.toml README.md src tests .github
  ```

  Expected: no operational references outside the transition guide, where the
  removed names may appear only to explain the migration.

### Task 6: Full verification and scope audit

**Files:**

- Verify all files changed by Tasks 1 through 5.

**Interfaces:**

- Consumes: the complete packaging cleanup.
- Produces: fresh evidence for syntax, metadata, build, installation, runtime,
  and scope claims.

- [ ] **Step 1: Syntax-check every modified Python file**

  Obtain the combined modified Python list with read-only Git commands, then
  run `python -m py_compile` inside the test environment for each existing
  file.

- [ ] **Step 2: Validate Pixi environments and task discovery**

  Run:

  ```bash
  pixi info
  pixi task list
  pixi run -e lint ruff check src/tommos tests
  ```

- [ ] **Step 3: Reinstall from the corrected isolated build**

  Run:

  ```bash
  pixi reinstall -e test tommos
  ```

  Expected: the editable native installation builds successfully on Linux.

- [ ] **Step 4: Run focused packaging and native tests**

  Run:

  ```bash
  pixi run -e test pytest -q \
    tests/test_distribution_metadata.py \
    tests/test_package_imports.py \
    tests/test_cpp_minimizer.py \
    tests/test_mkl.py \
    tests/test_wheel_install.py
  ```

- [ ] **Step 5: Run the complete test suite**

  Run `pixi run -e test pytest -q tests`.

- [ ] **Step 6: Build distributions**

  Run:

  ```bash
  pixi run -e build python -m build
  ```

  Inspect wheel and sdist archives with `unzip`, `tar`, and `readelf`.

- [ ] **Step 7: Audit the combined diff**

  Run:

  ```bash
  git status --short
  git diff --stat HEAD
  git diff --check HEAD
  git diff HEAD
  ```

  Confirm line by line that every remaining change belongs to packaging,
  minimal resource lookup, tests/CI, or the approved documentation. Confirm
  `cpp_mkl_minimizer.cpp` and all restored application modules are unchanged
  relative to `HEAD`.

- [ ] **Step 8: Independent final review**

  Give a reviewer the approved design, this plan, the complete combined diff,
  and verification outputs. Resolve every important packaging, behavior,
  portability, or scope finding before reporting completion.
