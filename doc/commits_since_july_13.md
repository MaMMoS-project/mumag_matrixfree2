# Commits since Monday, July 13, 0:00

Here is the list of all committed versions in the `mumag_matrixfree2` repository since Monday, July 13, 0:00, along with their commit data (Hash, Author, Date, and Message):

| Commit Hash | Author | Date | Message |
|-------------|--------|------|---------|
| `bcc126a` | Schrefl | 2026-07-20 17:24:38 +0200 | add slurm scripts and ignore .out files |
| `6dcda62` | Schrefl | 2026-07-20 17:17:16 +0200 | docs: sync slurm scripts and cli arguments in README |
| `692fa32` | Schrefl | 2026-07-20 01:02:35 +0200 | Pass large scaling arrays (M_rel, inv_M_rel, inv_M_prec) dynamically via sparse_ops dict to fix XLA GPU compilation OOM, without altering preconditioner functionality |
| `238a8e2` | Schrefl | 2026-07-17 17:45:42 +0200 | pass arrays like M_inv_vol as dynamic arrays, ad salomeMessToNpz.py |
| `762e32f` | Schrefl | 2026-07-14 18:11:03 +0200 | Optimize airbox generation with auto-scaling shell defaults |
| `292d42c` | Schrefl | 2026-07-14 16:53:27 +0200 | Merge mkl_ffi.cpp PARDISO helpers directly into cpp_mkl_minimizer.cpp and delete mkl_ffi.cpp |
| `977e6f6` | Schrefl | 2026-07-14 16:45:12 +0200 | Remove sphere sample from samples directory |
| `a2c58b5` | Schrefl | 2026-07-14 16:42:08 +0200 | Move compiled library libcpp_mkl_minimizer.so to lib/ directory, cleanup unused curvilinear_bb_minimizer.py, and finalize build system simplification |
| `6e5bf08` | Schrefl | 2026-07-14 15:49:11 +0200 | Move single_solid.npz into tests directory, update test paths, and add it to the repository |
| `a5610e5` | Schrefl | 2026-07-14 15:08:53 +0200 | Refactor C++ extension build system to CMake, add convex hull shell variant, and clean up unused scripts |
| `0ff57f9` | HoWilgh | 2026-07-14 13:17:47 +0200 | example benchmark_1 adopted for legacy outputs of matrixfree2<br>new file: examples/benchmark_1/benchmark1_workflow.py<br>new file: examples/benchmark_1/isotrop_down/isotrop.krn<br>new file: examples/benchmark_1/isotrop_down/isotrop.p2<br>new file: examples/benchmark_1/isotrop_up/isotrop.krn<br>new file: examples/benchmark_1/isotrop_up/isotrop.p2 |
| `2503bf1` | AdamsMP | 2026-07-13 23:39:34 +0200 | Merge pull request #41 from MaMMoS-project/pre-commit-ci-update-config |
| `324dc1c` | pre-commit-ci[bot] | 2026-07-13 20:58:18 +0000 | [pre-commit.ci] pre-commit autoupdate |
| `fe60a84` | Schrefl | 2026-07-13 17:10:18 +0200 | Add --data-parallel CLI argument skeleton |
| `8c08214` | Schrefl | 2026-07-13 16:52:32 +0200 | Add multi-GPU support for TR and PTR minimizers |
| `49b54f4` | Schrefl | 2026-07-13 13:55:32 +0200 | Rename SALOME text output to FEMME input files in README |
| `60fd715` | Schrefl | 2026-07-13 13:53:44 +0200 | Add mesh conversion scripts to Required Input in README |
| `00a7418` | Schrefl | 2026-07-13 13:16:54 +0200 | Add AMG preconditioner info to jax solver description in README |
| `6a18d18` | Schrefl | 2026-07-13 13:15:50 +0200 | Update cg_tol documentation in README to explain capping |
| `dd4cf1e` | Schrefl | 2026-07-13 13:12:09 +0200 | Update minimizer examples in table |
| `11feb2b` | Schrefl | 2026-07-13 13:07:43 +0200 | Clean up minimizer descriptions in README |
| `75d32b6` | Schrefl | 2026-07-13 13:02:33 +0200 | Update minimizer descriptions in README |
| `4816b8d` | Schrefl | 2026-07-13 12:53:52 +0200 | Add appendix for .p2 configuration parameters |
| `1d31c26` | Schrefl | 2026-07-13 12:48:24 +0200 | Add exhaustive CLI appendix to README.md |
| `afccf75` | Schrefl | 2026-07-13 12:43:34 +0200 | Update README.md with Multi-GPU slurm examples |
| `0049958` | Schrefl | 2026-07-13 01:51:28 +0200 | Implement multi-GPU support for sparse operators and loop orchestrator |
