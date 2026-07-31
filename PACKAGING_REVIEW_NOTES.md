# Packaging Review Notes

This is the short reviewer checklist for the retained packaging cleanup.

- **Editable native rebuilds and isolation.** Pure Python edits are visible
  through the editable install, but accessing a packaged `ctypes` resource does
  not trigger scikit-build-core's automatic rebuild hook. After C++, CMake,
  dependency-metadata, or package-file changes, rebuild explicitly with
  `pixi reinstall -e <environment> tommos`; the PEP 517 build remains isolated.
  See the official
  [scikit-build-core `ctypes` guide](https://scikit-build-core.readthedocs.io/en/latest/guide/ctypes.html)
  and [Pixi reinstall reference](https://pixi.prefix.dev/latest/reference/cli/pixi/reinstall/).

- **Host compiler prerequisite.** Build isolation supplies the declared Python
  build requirements, including `mkl-devel`, but the Linux native build still
  requires a working host C++ compiler and OpenMP toolchain. The build host or
  container must provide them; they are not duplicated as Pixi project
  dependencies. See the official
  [GCC installation guidance](https://gcc.gnu.org/install/).

- **PyPI oneMKL runtime and resource lookup.** The isolated build uses Intel's
  CMake configuration from the build interpreter prefix. At runtime, the
  declared PyPI `mkl` package provides the versioned `libmkl_rt.so.3`; Tommos
  loads only `tommos/_native/libcpp_mkl_minimizer.so` through
  `importlib.resources`, and the installed ELF RUNPATH resolves oneMKL.
  Sparse-dot-mkl integration reuses its loaded MKL library identity rather than
  guessing an unversioned filename. See Intel's
  [oneMKL PyPI guidance](https://www.intel.com/content/www/us/en/docs/onemkl/get-started-guide/2024-1/overview.html)
  and [oneMKL CMake configuration](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2025-0/cmake-config-for-onemkl.html).

- **SDL threading.** CMake selects `MKL_LINK=sdl` and links `MKL::mkl_rt`
  without setting `MKL_THREADING`, so oneMKL uses the SDL runtime default.
  `OpenMP::OpenMP_CXX` remains linked for Tommos's own OpenMP pragmas; it does
  not select oneMKL's threading layer. See Intel's
  [oneMKL CMake configuration](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2025-0/cmake-config-for-onemkl.html).

- **Numerical compiler-flag boundary.** The C++ source, algorithms, and native
  API remain unchanged. The packaging build uses `-O3` without
  `-march=native`, `-ffast-math`, or `-funroll-loops`; this is not a promise of
  bitwise-identical floating-point results across different compilers,
  toolchains, or flag configurations. See GCC's official
  [optimization options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html).

- **Local manylinux limitation.** Local `auditwheel` repair requires a
  compatible manylinux build environment: this host-built glibc 2.34 wheel
  cannot be repaired to manylinux_2_28, so genuine repair remains a CI
  integration check.

- **Review baselines.** Commit `fc85acc` contains the superseded Phase 2 patch
  and is the base for reviewing this correction. Its parent `5e8ed76` remains
  the Phase 1 baseline for checking the complete transition and the preserved
  C++/application sources.

- **Legacy comparison test.** The Phase 1 comparison test hard-coded the
  removed repository `lib/` layout. Its numerical body and assertions remain
  unchanged, but it now loads the installed package resource and versioned
  oneMKL runtime. The complete Linux suite passes after this correction.

- **Slurm transition and JAX cache.** The removed Slurm mechanism isolated
  per-job CMake builds and native-library output under `/tmp`; it did not
  configure JAX's persistent compilation cache. Slurm jobs now use the
  environment-installed package, while optional JAX cache policy is documented
  separately in `docs/slurm.md`.

- **Deferred archive/CI follow-ups.** The source distribution is currently
  98 MiB because it includes large pre-existing tracked example data. Reducing
  it requires a separate decision about which examples belong in the source
  archive. The wheel workflow's output-count assertion can also be tightened
  separately; neither issue changes the retained runtime design.
