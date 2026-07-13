# MaMMoS-MuMag: Matrix-Free Micromagnetics with JAX

MaMMoS-MuMag is a high-performance micromagnetic simulation package built on **JAX** and **C++**. It utilizes a **matrix-free FEM** approach for the Poisson equation (demagnetization field) to enable large-scale simulations on both CPU and GPU architectures without the severe memory overhead of storing global stiffness matrices. The architecture elegantly bridges Python's expressiveness (via JAX) for rapid GPU development with a highly optimized, dynamically compiled C++ and Intel MKL backend for uncompromising bare-metal CPU performance on large clusters.

## 1. Prerequisites and Installation

The project uses **Pixi** for cross-platform dependency management, environment isolation, and automated C++ compilation.

### Prerequisites
1. **Pixi**: Install Pixi if you haven't already:
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```
2. **GPU Drivers** (Optional): If running on an NVIDIA GPU, ensure you have appropriate NVIDIA drivers installed. The CUDA toolkit is managed directly by Pixi.

### Installation
Clone the repository and let Pixi handle everything. The environments will automatically build themselves upon first execution.
```bash
git clone git@github.com:MaMMoS-project/mumag_matrixfree2.git
cd mumag_matrixfree2
```
*Note: For Linux users running on CPU, Pixi will automatically compile the highly optimized `libcpp_mkl_minimizer.so` shared libraries using the Intel MKL in the background when the environment activates.*

## 2. How to Run the Software Locally

All operations are executed through `pixi run`. The default environment is `cpu`.

### Running Locally on CPU
To run simulations or meshes on the CPU (uses Intel MKL and C++ backend by default on Linux):
```bash
pixi run python3 src/loop.py <modelname> [options]
# Or explicitly targeting the CPU environment:
pixi run -e cpu python3 src/loop.py <modelname> [options]
```

### Running Locally on GPU
To run purely on the GPU (bypasses C++ MKL in favor of JAX XLA compilation):
```bash
pixi run -e cuda python3 src/loop.py <modelname> [options]
```

## 3. How to Submit Jobs with Slurm

### Slurm Job for CPU
To run a multi-core CPU job (e.g., on a `Gd` node), use the `cpu` environment. The C++ shared library is compiled automatically if it doesn't exist, and the CPU threading variables are strictly managed for performance.

Create a file `run_cpu.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=mumag_cpu
#SBATCH --partition=dissSims
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

export JAX_ENABLE_X64=True
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# The environment handles compilation automatically.
pixi run -e cpu python3 ../src/loop.py my_model --add-shell --verbose
```
Submit with: `sbatch run_cpu.slurm`

### Slurm Job for GPU
To run a GPU job (e.g., on an `A100` node), explicitly target the `cuda` environment.

Create a file `run_gpu.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=mumag_gpu
#SBATCH --partition=dissSims
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

# Ensure the -e cuda flag is present!
pixi run -e cuda python3 ../src/loop.py my_model --add-shell --verbose
```
Submit with: `sbatch run_gpu.slurm`

### Slurm Job for Multi-GPU
To run a multi-GPU job (e.g., on a node with multiple `L40S` or `A100` GPUs), explicitly target the `cuda` environment and request more than one GPU in the SLURM headers. The codebase automatically detects the number of available GPUs and dynamically partitions the massive sparse matrix operators (exchange, demag, preconditioner) across them to prevent Out-Of-Memory errors on massive meshes.

Create a file `run_multigpu.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=mumag_multigpu
#SBATCH --partition=dissSims
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

export JAX_ENABLE_X64=True
# Limit memory fraction per GPU to avoid overallocation
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

# Run the simulation. The code automatically distributes operators across all available GPUs!
pixi run -e cuda python3 ../src/loop.py my_model --add-shell --verbose
```
Submit with: `sbatch run_multigpu.slurm`

## 4. Required Input

A simulation requires three primary input files, usually sharing the same `<modelname>` prefix:

1. **Mesh File (`<modelname>.npz`)**: A numpy archive containing the tetrahedral mesh nodes (`knt`) and elements (`ijk`). Generated using `src/mesh.py`.
2. **Parameters File (`<modelname>.p2`)**: An INI-formatted configuration file defining the physical environment, field sweeps, and solver tolerances.
3. **Materials File (`<modelname>.krn`)**: A 6-column space-separated text file mapping material IDs to their intrinsic magnetic properties (theta, phi, K1, -, Js, A).

### The `.p2` Parameter File
```ini
[mesh]
size = 1e-9         ; Size of one mesh unit in meters (1e-9 = nm)

[initial state]
mx = 0.0            ; Initial magnetization vector (x, y, z)
my = 0.0
mz = 1.0

[field]
hstart = 2.0        ; Starting applied field (Tesla)
hfinal = -4.0       ; End field (Tesla)
hstep = -0.05       ; Field sweep step size (Tesla)
hx = 0.0            ; Field direction vector
hy = 0.0
hz = 1.0

[minimizer]
method = pcohen_hs  ; Energy minimization algorithm
tol_fun = 1e-8      ; Relative energy convergence tolerance
eps_a = 1e-12       ; Absolute gradient tolerance
```

## 5. Output

The simulation saves results into the directory specified by `--out-dir` (default: `hyst_<modelname>`).

1. **`params.log`**: A Markdown table detailing all resolved configuration variables and their origin (CLI, `.p2`, or Defaults).
2. **`hysteresis.csv`**: Global results tracking external field `B_ext_T`, parallel magnetization `J_par_T`, total energy `E`, and gradient norms across the sweep.
3. **`*.mh`**: Legacy compatibility text file with space-separated columns of the magnetization history.
4. **`*.vtu`**: ParaView-compatible XML visualization snapshots of the vector state. E.g., `state_cfg00001_B+1.5000e+00T.vtu`.
5. **`simulation.log`**: The captured standard output if explicitly tee'd in the run script.

## 6. Basic Usage Examples

**1. Create a 20nm Cube Mesh:**
```bash
pixi run python3 src/mesh.py --geom box --extent 20,20,20 --h 2.0 --out-name cube_20nm
```

**2. Run a CPU Simulation with an Auto-Generated Airbox:**
```bash
pixi run python3 src/loop.py cube_20nm --add-shell --layers 4
```

**3. Run a Full Pipeline Example (Provided):**
```bash
pixi run sample
```
*(This automatically meshes a cube, runs a full hysteresis loop, and outputs the results).*

## 7. Numerical Methods & Algorithms

### Energy Minimization Methods (`--method`)
The package employs Curvilinear Search Methods to strictly enforce the $|m|=1$ constraint at every node.
- **`pcohen_hs` (Default)**: Preconditioned Cohen Conjugate Gradient with Hestenes-Stiefel update. Highly robust for micromagnetics.
- **`pcohen`**: Preconditioned Cohen CG with Polak-Ribière update.
- **`lbfgs` / `plbfgs`**: Standard and Preconditioned Limited-memory BFGS.
- **`tn`**: Truncated Newton-CG.
- **`pbb`**: Preconditioned Barzilai-Borwein.

### Poisson Solvers (`--poisson-solver`)
- **`auto` (Default)**: Intelligently selects the solver based on hardware. Uses `pardiso` if Intel MKL/CPU is detected, and `jax` if a GPU is detected.
- **`pardiso`**: Direct sparse solver utilizing the C++ Intel MKL backend. Vastly superior for CPU nodes.
- **`jax`**: Iterative Preconditioned Conjugate Gradient (PCG) matrix-free solver. Highly parallelized for massive GPU execution.

## 8. CLI Features & Arguments

### `src/loop.py` (Main Driver)
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `modelname` | string | **Positional**. Base name. Looks for `<modelname>.npz`, `.krn`, and `.p2`. |
| `--mesh` | string | Explicit path to the input mesh. Extension `.npz` is automatically resolved if omitted. |
| `--add-shell` | flag | Automatically add a graded airbox shell around the core mesh. |
| `--layers` | int | Number of graded shell layers (default: 4). |
| `--K` | float | Geometric growth factor for shell layer thickness (default: 1.3). |
| `--cpp-mkl` / `--no-cpp-mkl` | flag | Toggle the high-performance C++ backend. Defaults to True on CPU, False on GPU. |
| `--poisson-solver` | choice | `auto` (default), `jax`, or `pardiso`. |
| `--method` | choice | Energy minimizer algorithm (default: `pcohen_hs`). |
| `--operator-mode` | choice | Mode for execution: `matrix-free` or `assembled`. |
| `--pc-iters` | int | Inner iterations for preconditioning (default: 10). |
| `--out-dir` | path | Directory for results (default: `hyst_<modelname>`). |
| `--verbose` | flag | Print detailed minimizer iterations. |

### `src/mesh.py` (Meshing Tool)
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--geom` | choice | Geometry type: `box` (default), `ellipsoid`, `eye`, `poly`, `poly_gb`, etc. |
| `--extent` | CSV | Full dimensions `Lx,Ly,Lz` of the core mesh. |
| `--h` | float | Target characteristic edge length. |
| `--gb-thickness` | float | (Poly_GB only) Total thickness of the grain boundary phase. |
| `--backend` | choice | Meshing engine: `meshpy` (TetGen) or `grid`. |
| `--out-name` | string | Base name for output files. |
| `--no-vis` | flag | Skip writing the `.vtu` file for the mesh geometry. |

## 9. Appendix: Complete CLI Parameters for `src/loop.py`

Below is an exhaustive list of all command-line arguments accepted by the main driver script `src/loop.py`, categorized by function.

### Positional Arguments
| Parameter | Description | Default |
| :--- | :--- | :--- |
| `modelname` | Base name of the model (for compatibility). Looks for `<modelname>.npz`, `.krn`, and `.p2`. | N/A |

### General I/O & Execution
| Parameter | Description | Default |
| :--- | :--- | :--- |
| `--mesh` | Path to the input NPZ mesh file (`knt`, `ijk`). | Extracted from `<modelname>.npz` |
| `--materials` | Path to a `.krn` file mapping material IDs to intrinsic properties. | Extracted from `<modelname>.krn` |
| `--out-dir` | Directory where results and snapshots are saved. | `hyst_out` |
| `--snapshot-every` | Save VTU snapshots of the vector state every N steps (0 to disable). | `1` |
| `--verbose` | Print detailed inner minimizer iterations at each field step. | `False` |
| `--benchmark` | Run a dummy warmup step before the main loop to compile JIT functions ahead of time. | `False` |

### Field Sweep & Initialization
| Parameter | Description | Default |
| :--- | :--- | :--- |
| `--h-dir` | Applied field direction as a unit vector `"hx,hy,hz"`. | `"0,0,1"` |
| `--B-start` | Starting magnitude of the applied field (in Tesla). | `-1.0` |
| `--B-end` | Final magnitude of the applied field (in Tesla). | `1.0` |
| `--dB` | Field step size magnitude (in Tesla). | `0.05` |
| `--m0-dir` | Initial uniform magnetization direction `"mx,my,mz"`. | Same as field direction |
| `--bias-type` | Type of symmetry-breaking field for mode initialization (`circular` or `random`). | `None` |
| `--bias-strength` | Strength of the bias field relative to saturation (e.g., 0.01). | `0.0` |

### Shell Generation (Airbox)
| Parameter | Description | Default |
| :--- | :--- | :--- |
| `--add-shell` | Automatically add a graded airbox shell around the core mesh. | `False` |
| `--layers` | Number of graded shell layers to generate ($\ge 1$). | `4` |
| `--K` | Geometric growth factor for the shell layer thickness ($> 1$). | `1.3` |
| `--beta` | Mesh-size/geometry coupling exponent (1.0 for linear scaling). | `1.0` |
| `--center` | Ray origin for shell expansion as `"cx,cy,cz"` (in mesh units). | `"0,0,0"` |
| `--h0` | Target edge length near the body surface (in mesh units). | `None` |
| `--hmax` | Target edge length at the outermost shell boundary (in mesh units). | `None` |
| `--minratio` | TetGen quality `minratio` (`-q`) for the shell tetrahedra. | `1.4` |
| `--max-steiner` | Limit the number of Steiner points added by TetGen. | `None` |
| `--no-exact` | Suppress exact arithmetic in TetGen (`-X`). | `False` |
| `--shell-verbose` | Enable verbose standard output from the shell generation pipeline. | `False` |

### Solver Backend & Parallelization
| Parameter | Description | Default |
| :--- | :--- | :--- |
| `--operator-mode` | SpMV execution mode: `matrix_free` (recompute on-the-fly) or `assembled` (sparse matrix format). | `assembled` |
| `--poisson-solver` | Solver for the magnetostatic Poisson problem (`auto`, `jax`, `pardiso`, `jax_mkl`). | `auto` |
| `--cpu-spmv-backend`| Backend for SpMV when running on CPU in assembled mode (`persistent_mkl`, `dot_product_mkl`, `scipy`, `jax_default`, `custom_jax`, `mkl_ffi`). | `persistent_mkl` |
| `--cpp-mkl` / `--no-cpp-mkl` | Force use of the pure C++ MKL minimizer backend. | True on CPU, False on GPU |
| `--chunk-elems` | Number of elements processed per chunk to control peak GPU memory. | `200000` |
| `--geom-backend` | Strategy for providing shape gradients: `stored_JinvT`, `stored_grad_phi`, or `on_the_fly`. | `stored_JinvT` |

### Energy Minimizer Configuration
| Parameter | Description | Default |
| :--- | :--- | :--- |
| `--method` | Energy minimization algorithm (e.g. `pcohen_hs`, `lbfgs`, `pnag`, `tn_split`). | `pcohen_hs` |
| `--max-iter` | Maximum inner iterations for the energy minimizer per field step. | `2000` |
| `--tau-f` | Relative energy convergence tolerance for the minimizer. | `1e-8` |
| `--eps-a` | Absolute tangent gradient norm tolerance for the minimizer. | `1e-12` |
| `--tau0` | Initial step size guess for the minimizer line search. | `0.01` |
| `--tau-min` | Minimum allowed step size (mainly for the BB minimizer). | `1e-6` |
| `--tau-max` | Maximum allowed step size (mainly for the BB minimizer). | `1.0` |
| `--L` | Restart frequency for conjugate gradient methods. | Number of nodes |
| `--memory` | History size ($m$) for L-BFGS and Anderson acceleration. | `5` |
| `--tn-iters` | Maximum inner iterations for Truncated Newton-CG solvers. | `5` |
| `--lr` | Learning rate for Nesterov accelerated gradient methods. | `0.1` |
| `--mu` | Momentum factor for Nesterov accelerated gradient methods. | `0.9` |
| `--wg-gamma` | Number of steps in convex region before switching to BB (WG method). | `5` |
| `--wg-threshold` | Convexity threshold for the WG algorithm. | `1e-6` |

### Preconditioning Configuration
| Parameter | Description | Default |
| :--- | :--- | :--- |
| `--precond-type` | Poisson solver preconditioner: `amgcl`, `jacobi`, `chebyshev`, or `amg`. | `amgcl` |
| `--pc-iters` | Inner iteration limit for preconditioning solvers. | `10` |
| `--pc-auto` / `--pc-no-auto`| Enable automated tuning of preconditioning accuracy. | `True` |
| `--pc-force-eta` | Base forcing parameter for adaptive preconditioning. | `0.5` |
| `--pc-force-alpha`| Exponent forcing parameter for adaptive preconditioning. | `0.5` |
| `--pc-stagnation-nu`| Relative threshold for detecting quadratic model stagnation. | `0.01` |
| `--pc-reg` | Diagonal regularization shift for the preconditioner matrix. | `0.0` |
| `--cg-maxiter` | Maximum iterations for the Poisson PCG solver (demagnetization field). | `2000` |
| `--cg-tol` | Relative residual tolerance for the Poisson PCG solver. | `1e-8` |
| `--poisson-reg` | Tikhonov regularization constant for the Poisson operator diagonal. | `1e-12` |
| `--phi-extrapolate` / `--no-phi-extrapolate` | Use linear extrapolation of scalar potential for faster iterative Poisson solves. | `True` |
