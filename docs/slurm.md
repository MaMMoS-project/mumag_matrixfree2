# Slurm Transition Guide

Tommos now uses the same environment-installed package for local and Slurm
execution. There is no Slurm-specific compilation or native-library lookup.

## Prepare the environment

Install the Pixi environments before submitting jobs:

```bash
pixi install -e cpu
```

After changing C++, CMake, dependency metadata, or packaged files, rebuild the
editable CPU installation before submitting jobs:

```bash
pixi reinstall -e cpu tommos
```

Jobs sharing one Pixi environment should treat it as read-only. Do not run
`pixi install` or `pixi reinstall` concurrently from those jobs.

The job itself runs the installed package:

```bash
pixi run -e cpu tommos loop <modelname> [options]
```

## What changed

| Concern | Previous behavior | Current behavior |
| --- | --- | --- |
| Native build | Phase 1 copied CMake and C++ sources into `/tmp/mumag_build_${SLURM_JOB_ID}` and compiled inside the job; the superseded Phase 2 patch instead rebuilt the editable installation and copied its `.so` there | Built by the editable package installation before job submission |
| Native lookup | Preferred the per-job path selected through `SLURM_JOB_ID`, then other repository or environment paths | Loads `tommos/_native/libcpp_mkl_minimizer.so` from the installed package |
| Cleanup | Removed the per-job build directory | No per-job native build directory exists |
| JAX cache | Not configured by the Slurm scripts or native loader | Still not configured by Tommos |

Both removed variants isolated a per-job native-library artifact from the
shared environment. The Phase 1 variant also isolated concurrent CMake build
directories from the shared repository. Neither variant isolated or
coordinated JAX compilation caches. The current package removes the native
build race by preparing the library once and having jobs load it read-only
from their environment.

## JAX compilation caching

The `--benchmark` option performs a warm-up call in the current process. It
does not select a persistent cache directory.

JAX provides a separate, optional persistent compilation cache. It is enabled
by setting `JAX_COMPILATION_CACHE_DIR` before the first compilation; see the
[official JAX persistent compilation cache guide](https://docs.jax.dev/en/latest/persistent_compilation_cache.html).
Tommos deliberately leaves this policy to the cluster configuration.

For isolated independent jobs, a job-local directory avoids shared-filesystem
writers:

```bash
export JAX_COMPILATION_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/tommos-jax-cache-${SLURM_JOB_ID}"
```

This provides isolation but normally prevents reuse between jobs. For one
distributed JAX job spanning multiple nodes, JAX documents that the processes
should use a trusted shared cache so that all ranks can read entries:

```bash
export JAX_COMPILATION_CACHE_DIR="/trusted/shared/path/tommos-jax-cache"
```

JAX notes that non-local cache filesystems may require `etils`, and warns that
a persistent cache must not be writable by untrusted users. For multiple
independent jobs, choose between job-local isolation and a shared cache
according to the cluster filesystem and reuse requirements; this is separate
from Tommos packaging.
