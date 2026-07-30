# `tommos` Phase 2 Native Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or `superpowers:executing-plans`
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Build and install the existing C++/oneMKL backend as part of the
`tommos` package on Linux x86-64 while keeping macOS on the portable Python
implementation.

**Architecture:** One private Python module owns native-library discovery,
oneMKL runtime loading, ABI validation, symbol binding, capability probes, and
PARDISO handle lifetime. `scikit-build-core` invokes CMake only on Linux
x86-64; the build environment obtains headers from PyPI `mkl-devel`, and the
installed package obtains its runtime from PyPI `mkl`. macOS builds skip CMake
and produce a pure wheel.

**Tech Stack:** Python 3.11+, `ctypes`, C++17, CMake 3.22+,
`scikit-build-core`, Intel oneMKL PyPI packages, pytest, Ruff, cibuildwheel, and
auditwheel.

## Global Constraints

- The distribution and import namespace remain `tommos` under `src/tommos/`.
- Native compilation, `sparse-dot-mkl`, the C++ minimizer, and PARDISO are
  supported only on Linux x86-64.
- Linux build dependencies come from PyPI `mkl-devel`; Linux runtime
  dependencies come from PyPI `mkl` and `sparse-dot-mkl`.
- Do not support, search for, or document an externally installed oneMKL.
- Do not copy or statically link oneMKL into the `tommos` wheel.
- macOS uses SciPy/JAX without MKL and builds a `py3-none-any` wheel.
- Windows and Linux ARM are unsupported.
- Preserve the current C ABI, `np.float64` values, `np.int32` CSR
  indices/indptr, and existing numerical behavior.
- Native capabilities are probed independently. Automatic choices may fall
  back; explicit native choices fail with the original cause.
- Use Google-style docstrings and explicit parameter and return type hints for
  every new Python function.
- Write focused failing tests before production behavior changes.
- Keep changes reviewable and avoid unrelated refactoring.
- Do not make commits, create branches/worktrees, stage files, or perform any
  other Git write operation.

---

### Task 1: Central Lazy Native Loader and ABI Contract

**Files:**

- Create: `src/tommos/_native_loader.py`
- Create: `tests/test_native_loader.py`
- Modify: `src/cpp/cpp_mkl_minimizer.cpp`

**Interfaces:**

- Produces:
  - `NATIVE_ABI_VERSION: int = 1`
  - `CapabilityProbe(name: str, available: bool, path: Path | None,
    source: str | None, error: BaseException | None)`
  - `probe_cpp_minimizer() -> CapabilityProbe`
  - `require_cpp_minimizer() -> Any`
  - `probe_pardiso() -> CapabilityProbe`
  - `require_pardiso() -> PardisoBindings`
  - `probe_sparse_dot_mkl() -> CapabilityProbe`
  - `native_diagnostics() -> dict[str, CapabilityProbe]`
  - C symbol `int tommos_native_abi_version(void)` returning `1`

- [ ] **Step 1: Add failing lookup and laziness tests**

  Test that importing `_native_loader` calls neither `ctypes.CDLL` nor
  `importlib.import_module`. Test candidate precedence:

  1. absolute `TOMMOS_NATIVE_LIBRARY`;
  2. `/tmp/mumag_build_${SLURM_JOB_ID}/libcpp_mkl_minimizer.so`;
  3. `${MUMAG_LIB_OUT}/libcpp_mkl_minimizer.so`;
  4. `tommos/_native/libcpp_mkl_minimizer.so` through
     `importlib.resources`;
  5. repository `lib/libcpp_mkl_minimizer.so`.

  An explicitly configured `TOMMOS_NATIVE_LIBRARY` must be absolute and must
  fail immediately when invalid; automatic candidates continue in order.

- [ ] **Step 2: Verify the tests fail for missing `_native_loader` APIs**

  Run:

  ```bash
  pixi run pytest -q tests/test_native_loader.py
  ```

  Expected: failure because the new module or requested interfaces do not
  exist.

- [ ] **Step 3: Implement discovery and process-lifetime resource handling**

  Use `importlib.resources.files("tommos")`, `as_file()`, and a module-owned
  `ExitStack`. Cache the selected absolute path, source label, `CDLL`, and
  original failure. Do not load at import time.

- [ ] **Step 4: Add failing ABI and independent-symbol tests**

  Fake libraries must cover ABI `1`, an incompatible ABI, absent ABI symbol,
  absent minimizer symbol, and absent PARDISO symbols. Assert that ABI
  validation happens before any operation symbol is read and that one missing
  capability does not invalidate another.

- [ ] **Step 5: Implement ABI validation and per-capability bindings**

  Bind `tommos_native_abi_version` first. Bind
  `run_cpp_pcohen_hs_minimization` only for the minimizer capability. Bind
  `init_pardiso`, `pardiso_solve_direct`, and `free_pardiso` only for PARDISO.
  Retain the existing `ctypes` signatures exactly.

- [ ] **Step 6: Export the additive C ABI version function**

  Add inside the existing `extern "C"` block:

  ```cpp
  int tommos_native_abi_version() {
      return 1;
  }
  ```

- [ ] **Step 7: Add and implement the sparse wrapper probe**

  The failing test must distinguish package import from successful access to
  `sparse_dot_mkl._mkl_interface.MKL`. Implement the probe without treating
  either the native C++ library or raw MKL runtime loading as evidence that
  `sparse-dot-mkl` works.

- [ ] **Step 8: Verify Task 1**

  Run:

  ```bash
  pixi run pytest -q tests/test_native_loader.py
  pixi run python -m py_compile src/tommos/_native_loader.py
  pixi run ruff check src/tommos/_native_loader.py tests/test_native_loader.py
  ```

---

### Task 2: Native Consumers, Capability Selection, and Provenance

**Files:**

- Create: `tests/test_native_selection.py`
- Modify: `src/tommos/_native_loader.py`
- Modify: `src/tommos/cpp_minimizer.py`
- Modify: `src/tommos/amg_utils.py`
- Modify: `src/tommos/loop.py`
- Modify: `src/tommos/hysteresis_loop.py`
- Modify: `src/tommos/poisson_solve.py`
- Modify: `tests/test_cpp_minimizer.py`
- Modify: `tests/test_compare_kernels.py`
- Modify: `tests/test_mkl.py`
- Modify: `tests/test_mkl_jit.py`
- Modify: `tests/test_py_minimizer.py`

**Interfaces:**

- Consumes all Task 1 probes and bindings.
- Produces:
  - `NativeSelections(cpp_minimizer: str, poisson_solver: str,
    cpu_spmv_backend: str)`
  - `resolve_native_selections(cpp_mkl: bool | None, poisson_solver: str,
    cpu_spmv_backend: str, has_gpu: bool) -> NativeSelections`
  - Result provenance values `cpp_mkl`/`python`, `pardiso`/`jax`, and
    `persistent_mkl`/`scipy`.
  - Additive result keys `cpp_minimizer`, `poisson_solver`, and
    `cpu_spmv_backend`.

- [ ] **Step 1: Add failing selection tests**

  Cover successful and unavailable independent probes. Explicit
  `--no-cpp-mkl`, `--poisson-solver jax`, and `--cpu-spmv-backend scipy` must
  not probe native code. Explicit native requests must raise before mesh or
  simulation setup. Automatic CPU selection must resolve each capability
  independently; GPU selection must keep the current JAX sparse path. The C++
  minimizer probe remains independent, but selecting that implementation also
  requires a selected/available PARDISO solver because its current native
  Poisson path consumes a PARDISO handle.

- [ ] **Step 2: Verify selection tests fail for the missing resolver**

  Run:

  ```bash
  pixi run pytest -q tests/test_native_selection.py
  ```

- [ ] **Step 3: Implement selection and CLI integration**

  Add `auto` to `--cpu-spmv-backend` and make it the CLI/default API choice.
  Replace the single `ctypes.CDLL("libmkl_rt.so")` check with the three
  independent Task 1 probes. Resolve all choices before assembled matrices are
  laid out. Automatic GPU choices do not probe native capabilities; explicit
  native choices remain strict. Explicit `dot_product_mkl`, `jax_default`, and
  `custom_jax` remain strict/pass-through choices. Remove the already
  unsupported `mkl_ffi` and `jax_mkl` CLI choices.

- [ ] **Step 4: Refactor the minimizer consumer**

  Remove import-time path selection and `CDLL` loading from
  `cpp_minimizer.py`. Obtain the already ABI-validated minimizer function
  inside `cpp_minimize()`.

- [ ] **Step 5: Refactor the PARDISO consumer**

  Remove duplicate native path discovery and declarations from `amg_utils.py`.
  Put the handle/factory beside `PardisoBindings` in `_native_loader.py`. Use a
  typed, idempotently closable handle that retains its contiguous `float64`
  values, contiguous `int32` indices/indptr, bindings, and library reference.
  Serialize solve and close on each handle. Keep finalization only as a backup
  for explicit `close()` after successful loop output synchronization.

- [ ] **Step 6: Keep sparse wrapper loading isolated**

  Make `PersistentMKLOperator` use the independently probed
  `sparse-dot-mkl` wrapper for handle creation, execution, and destruction;
  remove `ctypes.util.find_library()`, raw `LoadLibrary`, and the direct
  optimize call. Remove any inference that the C++/PARDISO library proves
  sparse-wrapper availability. Defer the direct oneMKL runtime handle and
  inspector optimization to Task 4, where the installed PyPI `mkl` metadata
  and versioned runtime are available. Propagate idempotent close ownership
  through callback-backed sparse operators and close successful simulation
  resources after output synchronization.

- [ ] **Step 7: Expose provenance**

  Add the resolved backend names to the diagnostics/result returned by the
  loop. Preserve all existing result keys.

- [ ] **Step 8: Update native integration tests**

  Skip a native test only when its corresponding probe is unavailable. Replace
  direct loading of `tests/../lib/libcpp_mkl_minimizer.so` with
  loader-resolved bindings. Exercise sparse operations through the independently
  probed wrapper, retain deterministic numerical comparisons against SciPy/JAX,
  and move raw `libmkl_rt.so` optimize/hint coverage to Task 4.

- [ ] **Step 9: Verify Task 2**

  First syntax-check every modified Python file, then run:

  ```bash
  pixi run pytest -q tests/test_native_loader.py tests/test_native_selection.py
  pixi run pytest -q tests/test_mkl.py tests/test_mkl_jit.py
  pixi run ruff check src/tommos tests
  ```

---

### Task 3: `scikit-build-core` and PyPI oneMKL Build Integration

**Files:**

- Create: `src/cpp/find_mkl.py`
- Create: `tests/test_find_mkl.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_distribution_metadata.py`
- Modify: `src/cpp/CMakeLists.txt`
- Modify: `compile_local.sh`
- Modify: `activate_build.sh`
- Modify: `pixi.toml`

**Interfaces:**

- `python src/cpp/find_mkl.py --cmake-dir` prints the parent directory
  containing the `MKLConfig.cmake` owned by `mkl-devel`.
- `python src/cpp/find_mkl.py --runtime` prints the versioned
  `libmkl_rt.so.*` owned by `mkl`.
- CMake installs `libcpp_mkl_minimizer.so` to
  `tommos/_native/libcpp_mkl_minimizer.so`.

- [ ] **Step 1: Inspect and record the actual PyPI wheel layout**

  In an isolated directory under `/tmp`, download/install `mkl-devel`, `mkl`,
  and `sparse-dot-mkl`. Record distribution metadata, `MKLConfig.cmake`,
  `mkl.h`, `libmkl_rt.so.*`, its `SONAME`, and its transitive dependencies.
  Use only files owned by those Python distributions.

- [ ] **Step 2: Add failing deterministic-discovery tests**

  Fake `importlib.metadata.Distribution.files` entries for a valid unique
  CMake configuration/runtime, no match, and ambiguous matches. Assert exact
  absolute output and actionable failures.

- [ ] **Step 3: Implement the discovery helper**

  Use `importlib.metadata.distribution(...).files` and `locate_file()`; never
  read `CONDA_PREFIX`, `MKLROOT`, `LD_LIBRARY_PATH`, or generic system paths.

- [ ] **Step 4: Verify the helper tests**

  Run:

  ```bash
  pixi run pytest -q tests/test_find_mkl.py
  pixi run python -m py_compile src/cpp/find_mkl.py
  ```

- [ ] **Step 5: Replace the Python build backend**

  Use:

  ```toml
  [build-system]
  requires = [
    "scikit-build-core>=0.12",
    "mkl-devel>=2026.0,<2027; sys_platform == 'linux' and platform_machine == 'x86_64'",
  ]
  build-backend = "scikit_build_core.build"

  [tool.scikit-build]
  minimum-version = "build-system.requires"
  cmake.source-dir = "src/cpp"
  wheel.packages = ["src/tommos"]
  wheel.py-api = "py3"
  sdist.include = ["src/cpp/CMakeLists.txt", "src/cpp/cpp_mkl_minimizer.cpp", "src/cpp/find_mkl.py"]

  [[tool.scikit-build.overrides]]
  if.platform-system = "^darwin"
  wheel.cmake = false
  wheel.platlib = false
  ```

- [ ] **Step 6: Add Linux runtime metadata**

  Add the two exact Linux x86-64 project dependencies:

  ```text
  mkl>=2026.0,<2027; sys_platform == "linux" and platform_machine == "x86_64"
  sparse-dot-mkl>=0.9.10; sys_platform == "linux" and platform_machine == "x86_64"
  ```

  Extend the distribution-metadata tests before changing the metadata. Keep
  both dependencies absent on macOS and unsupported architectures.

- [ ] **Step 7: Modernize CMake**

  Fail configuration unless the platform is Linux x86-64. Set
  `MKL_LINK=sdl`, `MKL_INTERFACE=lp64`, and `MKL_THREADING=gnu_thread` to
  retain the existing single-dynamic-library and 32-bit CSR-index ABI. Invoke
  `find_mkl.py --cmake-dir`, derive and force `MKL_ROOT` from that
  distribution-owned configuration directory so Intel's config cannot consult
  `MKLROOT`, then call
  `find_package(MKL CONFIG REQUIRED PATHS <observed-directory>
  NO_DEFAULT_PATH)`, use `MKL_INCLUDE`, and link `MKL::mkl_rt` and
  `OpenMP::OpenMP_CXX`. Do not link the aggregate `MKL::MKL` target because
  its PyPI configuration injects the build environment's absolute MKL RPATH.
  Remove `CONDA_PREFIX` and source-tree output handling, retain portable `-O3`,
  remove `-march=native`, `-ffast-math`, and `-funroll-loops`, and add:

  ```cmake
  install(
      TARGETS cpp_mkl_minimizer
      LIBRARY DESTINATION tommos/_native
  )
  ```

- [ ] **Step 8: Give the installed native library a relocatable runtime path**

  Set only an `$ORIGIN`-relative RUNPATH derived from the observed wheel layout.
  Set `SKIP_BUILD_RPATH`, disable link-path-derived install RPATHs, and do not
  embed any build or installation prefix. Confirm the final ELF `NEEDED` entry
  uses the observed versioned oneMKL `SONAME`.

- [ ] **Step 9: Move Pixi Linux MKL responsibility to PyPI**

  Remove the Conda `mkl`, `mkl-include`, `libblas=*mkl`,
  `sparse_dot_mkl`, and MKL-specific OpenMP-mutex entries. Keep compiler,
  CMake, and Make development dependencies. Let the editable `tommos`
  dependency install the Linux-only PyPI runtime dependencies added in Step 6.
  Remove the activation-time auto-build hook because editable installation now
  builds the native package. Make the explicit legacy compile task rebuild the
  editable installation through isolated PEP 517 instead of invoking the
  removed Conda-prefix CMake contract; when `SLURM_JOB_ID` is set, preserve the
  existing `/tmp/mumag_build_${SLURM_JOB_ID}` compatibility artifact by copying
  the just-built package library there.

- [ ] **Step 10: Verify Task 3**

  Syntax-check shell and Python first, then run:

  ```bash
  bash -n compile_local.sh activate_build.sh
  pixi run python -m build
  pixi run pytest -q tests/test_find_mkl.py tests/test_distribution_metadata.py
  ```

  Inspect the built native library and assert exactly the observed versioned
  oneMKL `NEEDED` entry, RUNPATH `$ORIGIN/../../../..`, no absolute build
  prefix, and no bundled `libmkl*.so*`.

---

### Task 4: Linux Runtime Metadata, Clean Wheel Loading, and macOS Fallback

**Files:**

- Create: `tests/test_wheel_install.py`
- Modify: `src/tommos/_native_loader.py`
- Modify: `tests/test_distribution_metadata.py`
- Modify: `tests/test_package_imports.py`

**Interfaces:**

- The Linux x86-64 wheel declares:

  ```text
  mkl>=2026.0,<2027; sys_platform == "linux" and platform_machine == "x86_64"
  sparse-dot-mkl>=0.9.10; sys_platform == "linux" and platform_machine == "x86_64"
  ```

- `load_mkl_runtime() -> Any` locates the versioned runtime through
  `importlib.metadata.distribution("mkl")`, loads it with
  `ctypes.RTLD_GLOBAL`, caches the handle, and never searches external paths.

- [ ] **Step 1: Add failing metadata and runtime-loading tests**

  Assert the two exact Linux markers, their absence on simulated macOS,
  versioned runtime selection from distribution-owned files, cached
  `RTLD_GLOBAL` loading, clear errors for absent or ambiguous runtimes, and
  loader-owned inspector optimize/hint bindings used by
  `PersistentMKLOperator`.

- [ ] **Step 2: Verify the focused tests fail**

  Run:

  ```bash
  pixi run pytest -q tests/test_distribution_metadata.py tests/test_native_loader.py
  ```

- [ ] **Step 3: Add loader support**

  Implement distribution-owned, versioned oneMKL loading for the runtime
  dependencies already declared in Task 3. Preload oneMKL before loading the
  packaged C++ library, and make `PersistentMKLOperator` use the cached
  loader-owned runtime handle rather than `ctypes.util.find_library()` or the
  unversioned `libmkl_rt.so`. Preserve the original loader exception as the
  cause of an unavailable capability.

- [ ] **Step 4: Build and inspect Linux artifacts**

  Build wheel and sdist with `MKLROOT` and `LD_LIBRARY_PATH` unset. Assert:

  - the wheel has a Linux platform tag;
  - it contains `tommos/_native/libcpp_mkl_minimizer.so`;
  - it contains no `libmkl*.so*`;
  - the sdist contains all CMake/native sources;
  - compiler commands contain no host-specific or fast-math flags.

- [ ] **Step 5: Verify a clean Linux installation**

  In a new `/tmp` virtual environment, install the wheel with dependencies,
  outside the checkout and without `MKLROOT`/`LD_LIBRARY_PATH`. Run independent
  C++ minimizer, PARDISO, and `sparse-dot-mkl` probes plus deterministic native
  computations. Inspect `readelf -d`, `ldd`, and installed distribution
  metadata.

- [ ] **Step 6: Verify macOS through CI**

  Build with the Darwin scikit-build override, assert `py3-none-any`, assert
  no `_native` library and no MKL requirements, and run the portable tests with
  SciPy/JAX provenance.

---

### Task 5: Native Wheel Pipeline, Documentation, and Final Verification

**Files:**

- Create: `.github/workflows/wheels.yml`
- Create: `.github/scripts/repair_linux_wheel.sh`
- Modify: `.github/workflows/test.yml`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-28-tommos-packaging-design.md`
- Modify: `tests/test_wheel_install.py`

**Interfaces:**

- cibuildwheel builds only Linux x86-64 CPython 3.11+ wheels using a
  `manylinux_2_28` image.
- The pipeline tests the installed wheel, does not upload to PyPI, and retains
  wheel artifacts for review.

- [ ] **Step 1: Add the Linux wheel workflow**

  Use current official `cibuildwheel` guidance. Restrict builds to CPython
  versions allowed by `requires-python`, Linux x86-64, and manylinux
  (not musllinux). Configure auditwheel to leave the separately installed
  oneMKL runtime outside the `tommos` wheel while repairing permitted compiler
  runtime dependencies. After auditwheel grafts those dependencies, restore
  the install-relative PyPI oneMKL route and regenerate wheel `RECORD`.

- [ ] **Step 2: Add installed-wheel tests to the pipeline**

  Unset `MKLROOT` and `LD_LIBRARY_PATH`; assert native path, ABI, capability
  provenance, absence of bundled MKL, ELF dependencies, and deterministic
  native computations.

- [ ] **Step 3: Remove stale Windows workflow references**

  Keep Ubuntu and macOS in the package test matrix and remove the commented
  Windows target. Do not add a Windows wheel job.

- [ ] **Step 4: Update user and staging documentation**

  Document `pip install -e .`, Linux PyPI MKL behavior, macOS portable
  behavior, native diagnostics, clean rebuild commands, and the fact that
  external oneMKL and Windows are unsupported. Record the completed Phase 2
  verification evidence without claiming remote CI results.

- [ ] **Step 5: Run final local verification**

  Syntax-check modified Python before executing it, then run:

  ```bash
  pixi run python -m py_compile src/tommos/*.py src/cpp/find_mkl.py
  bash -n compile_local.sh activate_build.sh
  pixi run ruff check .
  pixi run ruff format --check .
  pixi run pytest -q tests
  pixi run python -m build
  git diff --check
  ```

  Inspect both distributions and repeat the clean Linux wheel installation
  from Task 4. No commit or other Git write operation follows verification.
