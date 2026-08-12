# Slurm Rewrite: Lost Behavior

The current packaging workflow builds Tommos before jobs are submitted and
expects jobs to treat the installed Pixi environment as read-only. This
replaces the former per-job build under
`/tmp/mumag_build_${SLURM_JOB_ID}`.

## What was lost

- **A job-local native-library snapshot.** Each job copied the C++ sources and
  built its own library when the job started. Later source changes or updates
  to the shared `lib/` directory did not replace that job's native artifact.
- **Automatic synchronization with C++ sources.** Starting a job rebuilt the
  library from the C++ sources visible at that moment. The current workflow
  requires `pixi reinstall -e <environment> tommos` before submission after
  relevant changes.
- **Isolation from shared-filesystem build races.** Concurrent jobs did not
  share CMake output or write the same native library. The current approach is
  safe only while the installed environment remains read-only.
- **Compute-node-specific compilation.** The former build used
  `-march=native`; the package build deliberately uses portable compiler
  settings.
- **Slurm and manual library overrides.** Runtime lookup through
  `SLURM_JOB_ID` and `MUMAG_LIB_OUT` has been removed.

## Modifying code while jobs are queued or running

Protecting active jobs from concurrent development was plausibly one reason
for the old design. It protected the compiled C++ library, but it was not a
complete source snapshot: Python still ran from the shared checkout, so files
imported later could come from a newer revision.

The current editable installation is likewise not immutable. Modifying the
checkout or reinstalling the shared environment while jobs are active can mix
revisions. Until stronger isolation is restored, use a separate checkout and
Pixi environment for each stable batch of jobs and do not reinstall that
environment while those jobs are queued or running.

## Possible future recovery

The cleanest recovery would be to build a wheel for each source revision and
install it into a revision-specific, read-only environment. A job-local
installation under `SLURM_TMPDIR` could recover stronger per-job isolation at
the cost of repeated installation and storage. If node-specific optimization
is important, a separate native artifact cache could be keyed by source
revision, toolchain, and CPU type.

The removed mechanism did not manage JAX's persistent compilation cache; that
remains a separate cluster policy. See [the Slurm transition
guide](docs/slurm.md) for the current workflow.
