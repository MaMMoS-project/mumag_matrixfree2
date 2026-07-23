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

*Note for Mac Users (Apple Silicon / ARM64): Intel MKL is not available on Mac. You must bypass the MKL backends by appending `--cpu-spmv-backend scipy --no-cpp-mkl` to your `loop.py` simulation commands.*

## 2. How to Run the Software Locally

> [!WARNING]
> **Experimental Feature:** The `--operator-mode matrix_free` flag is highly experimental and is **not recommended** for use. It incurs excessively long JAX JIT compilation times on GPUs. Please use the default `--operator-mode assembled` for all workloads.

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

### Included Slurm Pipeline Examples

The repository includes complete end-to-end pipeline examples located in the `../test_matrixfree2/Nd0.5Fe0.5/` directory (assuming your simulation workspaces are structured side-by-side). These scripts automatically handle the execution and performance environment tuning for different hardware profiles.

**1. `test_cpu.slurm`** (High-Performance CPU)
- **Hardware**: Reserves CPUs on the `dissSims` partition (e.g., `Gd` node). Sets critical thread pinning (OpenMP) environment variables to guarantee optimal bare-metal CPU performance (MKL variables are now securely handled automatically in Python).
- **Workflow**: 
  - Triggers a secure, isolated local compilation of the C++ MKL backend directly into `/tmp/` to avoid network filesystem race conditions.
  - Runs the demagnetization simulation using the generated cubic mesh.
  - Safely deletes the `/tmp/` build artifacts upon completion to leave the node clean.

**2. `test_a100.slurm`** (Single GPU)
- **Hardware**: Reserves 1 A100 GPU and 2 CPU cores, configuring XLA memory allocation safely to prevent out-of-memory errors (`XLA_PYTHON_CLIENT_MEM_FRACTION`).
- **Workflow**:
  - Sweeps the external field entirely on the extremely fast GPU backend (no C++ compilation required).

**3. `test_multi_gpu.slurm`** (Multi-GPU)
- **Hardware**: Reserves 4 L40s GPUs. 
- **Workflow**:
  - Automatically detects all available GPUs and dynamically partitions the massive sparse matrix operators (exchange, demag, preconditioner) across them to prevent Out-Of-Memory errors on massive meshes.
  - **Crucial Setting**: Includes `export JAX_DISABLE_P2P=1`. When running on multi-GPU nodes that lack NVLink bridges (such as standard PCIe nodes with strict Access Control Services routing), direct GPU-to-GPU memory copies may hang indefinitely or silently fail. This forces JAX to route cross-device memory transfers safely through host RAM.

To submit any job, simply `cd` into the test directory and use `sbatch`:
```bash
cd ../test_matrixfree2/Nd0.5Fe0.5
sbatch test_cpu.slurm
sbatch test_a100.slurm
sbatch test_multi_gpu.slurm
```

## 4. Required Input

A simulation requires three primary input files, usually sharing the same `<modelname>` prefix:

1. **Mesh File (`<modelname>.npz`)**: A numpy archive containing the tetrahedral mesh nodes (`knt`) and elements (`ijk`). It can be generated using `src/mesh.py`. Alternatively, you can convert existing meshes using provided scripts:
   - `src/mesh_convert.py`: Converts a VTK UnstructuredGrid (`.vtu`) mesh into the required `.npz` format (and vice versa).
   - `src/salomeMeshToNpz.py`: Converts FEMME input files (`.knt` for nodes, `.ijk` for connectivity) into the `.npz` format.
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
3. **`mammos_hysteresis.csv`**: An ontology-compliant version of the hysteresis data mapped to the [MagMo/MAMMOS EMMO ontology](https://emmo-repo.github.io/domain-magnetic-materials/magnetic-materials.html). It automatically scales reduced units into strictly typed physical entities (e.g., `EnergyDensity` in $J/m^3$, `MagneticFluxDensity` in Tesla, and `Index` for configuration numbers).
4. **`*.mh`**: Legacy compatibility text file with space-separated columns of the magnetization history.
5. **`*.vtu`**: ParaView-compatible XML visualization snapshots of the vector state. E.g., `state_cfg00001_B+1.5000e+00T.vtu`.
6. **`simulation.log`**: The captured standard output if explicitly tee'd in the run script.

## 6. Basic Usage Examples

**1. Create a 20nm Cube Mesh:**
```bash
pixi run python3 src/mesh.py --geom box --extent 20,20,20 --h 2.0 --out-name cube_20nm
```

**2. Create a 20nm Cube Mesh with an Auto-Generated Airbox:**
```bash
pixi run python3 src/mesh.py --geom box --extent 20,20,20 --h 2.0 --out-name cube_20nm_with_shell --add-shell
```

**3. Run a CPU Simulation (Adding an Airbox On-the-Fly):**
```bash
pixi run python3 src/loop.py cube_20nm --add-shell
```

**4. Run a Full Pipeline Example (Provided):**
```bash
pixi run sample
```
*(This automatically meshes a cube, runs a full hysteresis loop, and outputs the results).*

**5. Run the Pipeline on Mac (Apple Silicon / ARM64):**
Because the `pixi run sample` shortcut relies on hardcoded Linux commands, Mac users must execute the simulation step explicitly to append the MKL bypass flags:
```bash
# Generate the mesh
pixi run python3 src/mesh.py --geom box --extent 20,20,20 --h 2.0 --backend grid --out-name cube_20nm --no-vis
# Run the simulation without MKL
pixi run python3 src/loop.py cube_20nm --add-shell --cpu-spmv-backend scipy --no-cpp-mkl
```

## 7. Numerical Methods & Algorithms

### Energy Minimization Methods (`--method`)
The package employs Curvilinear Search Methods to strictly enforce the $|m|=1$ constraint at every node.
- **`pcohen_hs` (Default)**: Preconditioned Cohen Conjugate Gradient with Hestenes-Stiefel update. This is the most successful and robust minimizer for micromagnetics across our benchmarks.
- **`pcohen`**: Preconditioned Cohen CG with Polak-Ribière update.
- **`tn`**: Truncated Newton-CG.

### Poisson Solvers (`--poisson-solver`)
- **`auto` (Default)**: Intelligently selects the solver based on hardware. Uses `pardiso` if Intel MKL/CPU is detected, and `jax` if a GPU is detected.
- **`pardiso`**: Direct sparse solver utilizing the C++ Intel MKL backend. Vastly superior for CPU nodes.
- **`jax`**: Iterative Preconditioned Conjugate Gradient (PCG) matrix-free solver. It uses an algebraic multigrid method for preconditioning the conjugate gradient. Highly parallelized for massive GPU execution.

## 8. CLI Features & Arguments

### `src/loop.py` (Main Driver)
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `modelname` | string | **Positional**. Base name. Looks for `<modelname>.npz`, `.krn`, and `.p2`. |
| `--mesh` | string | Explicit path to the input mesh. Extension `.npz` is automatically resolved if omitted. |
| `--add-shell` | flag | Automatically add a graded airbox shell around the core mesh (default creates a high-quality, optimal element-count convex hull). |
| `--KL` | float | Total outermost geometric scale relative to body (default: 10.0). |
| `--K` | float | Geometric growth factor for shell layer thickness (default: 1.5). |
| `--hmax` | float | Target edge length at the outermost shell boundary (default: auto scales with magnet size). |
| `--shell-type`| choice | Outer boundary: `triangles` or `hull` (default: `hull`). |
| `--cpp-mkl` / `--no-cpp-mkl` | flag | Toggle the high-performance C++ backend. Defaults to True on CPU, False on GPU. |
| `--poisson-solver` | choice | `auto` (default), `jax`, or `pardiso`. |
| `--method` | choice | Energy minimizer algorithm (default: `pcohen_hs`). |
| `--operator-mode` | choice | Mode for execution: `assembled` (default, recommended) or `matrix_free` (experimental, do not use). |
| `--pc-iters` | int | Inner iterations for preconditioning (default: 10). |
| `--out-dir` | path | Directory for results (default: `hyst_<modelname>`). |
| `--verbose` | flag | Print detailed minimizer iterations. |

### `src/mesh.py` (Meshing Tool)
| Parameter | Type | Description | Deafult |
| :--- | :--- | :--- | :--- |
| `--geom` | choice | Geometry type: `box`, `ellipsoid`, `eye`, `poly`, `poly_gb`, etc. | `box` |
| `--extent` | CSV | Full dimensions `Lx,Ly,Lz` of the core mesh. | `60.0,60.0,60.0` |
| `--h` | float | Target characteristic edge length. | `2.0` |
| `--n` | int | (Poly / Poly_GB only) Number of grains for the polycrystalline generation. | `10` |
| `--neper-timeout` | int | Timeout in seconds for the Neper Voronoi tessellation. | `None` |
| `--gb-thickness` | float | (Poly_GB only) Total thickness of the grain boundary phase. | `1.0` |
| `--backend` | choice | Meshing engine: `meshpy` (TetGen) or `grid`. | `meshpy` |
| `--out-name` | string | Base name for output files. | `single_solid` |
| `--no-vis` | flag | Skip writing the `.vtu` file for the mesh geometry. | `True` |

### `src/add_shell.py` (Standalone Airbox Tool)
The airbox tool can be run independently to add a far-field vacuum region to an existing mesh. It is highly optimized to minimize the number of tetrahedrons using a convex hull.
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--in` | string | **Required**. Input `.npz` mesh containing the core body. |
| `--out-npz` | string | Optional path to save the merged mesh as an `.npz` file. |
| `--out-vtu` | string | Optional path to save the merged mesh as a `.vtu` file for visualization. |
| `--shell-type`| choice | Outer boundary geometry: `triangles` or `hull` (default: `hull`). The `hull` mode dramatically reduces element counts. |
| `--KL` | float | Total outermost expansion distance relative to the core body (default: `10.0`). |
| `--K` | float | Geometric growth factor for the thickness of each subsequent shell layer (default: `1.5`). |
| `--hmax` | float | Target edge length at the outermost boundary. Defaults to `None` (intelligently auto-scales to 20% of the expanded airbox bounds to prevent element explosions on large models). |
| `--auto-layers`| flag | Automatically compute the number of layers required to reach `KL` using growth rate `K`. This is intrinsically enabled if `--layers` is omitted. |

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
| `--shell-type`| Outer shell boundary type: copy original 'triangles' or use convex 'hull'. | `hull` |
| `--KL` | Total outermost geometric scale relative to body ($> 1$). | `10.0` |
| `--K` | Geometric growth factor for the shell layer thickness ($> 1$). | `1.5` |
| `--layers` | Number of graded shell layers to generate (if omitted, auto derived from KL and K). | `None` |
| `--auto-layers`| Automatically compute the number of layers given --KL and --K (enabled by default if --layers is None). | `True` |
| `--beta` | Mesh-size/geometry coupling exponent (1.0 for linear scaling). | `1.0` |
| `--center` | Ray origin for shell expansion as `"cx,cy,cz"` (in mesh units). | `"0,0,0"` |
| `--h0` | Target edge length near the body surface (in mesh units). | `None` |
| `--hmax` | Target edge length at the outermost shell boundary (in mesh units). | `None` (auto scales) |
| `--minratio` | TetGen quality `minratio` (`-q`) for the shell tetrahedra. | `1.4` |
| `--max-steiner` | Limit the number of Steiner points added by TetGen. | `None` |
| `--no-exact` | Suppress exact arithmetic in TetGen (`-X`). | `False` |
| `--shell-verbose` | Enable verbose standard output from the shell generation pipeline. | `False` |

### Solver Backend & Parallelization
| Parameter | Description | Default |
| :--- | :--- | :--- |
| `--operator-mode` | SpMV execution mode: `assembled` (default, sparse matrix) or `matrix_free` (experimental, do not use). | `assembled` |
| `--poisson-solver` | Solver for the magnetostatic Poisson problem (`auto`, `jax`, `pardiso`). | `auto` |
| `--cpu-spmv-backend`| Backend for SpMV when running on CPU in assembled mode (`persistent_mkl`, `dot_product_mkl`, `scipy`, `jax_default`, `custom_jax`). | `persistent_mkl` |
| `--cpp-mkl` / `--no-cpp-mkl` | Force use of the pure C++ MKL minimizer backend. | True on CPU, False on GPU |
| `--chunk-elems` | Number of elements processed per chunk to control peak GPU memory. | `200000` |
| `--geom-backend` | Strategy for providing shape gradients: `stored_JinvT`, `stored_grad_phi`, or `on_the_fly`. | `stored_JinvT` |

### Energy Minimizer Configuration
| Parameter | Description | Default |
| :--- | :--- | :--- |
| `--method` | Energy minimization algorithm (e.g. `pcohen_hs`, `pcohen`, `tn`). | `pcohen_hs` |
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
| `--cg-tol` | Relative residual tolerance for the Poisson PCG solver. The actual value passed to the solver is dynamically capped to be at least an order of magnitude tighter than the minimizer's relative energy tolerance (`min(cg_tol, tau_f * 0.1)`). | `1e-8` |
| `--poisson-reg` | Tikhonov regularization constant for the Poisson operator diagonal. | `1e-12` |
| `--phi-extrapolate` / `--no-phi-extrapolate` | Use linear extrapolation of scalar potential for faster iterative Poisson solves. | `True` |

## 10. Appendix: `.p2` Configuration Parameters and Overrides

In addition to CLI arguments, you can define simulation parameters permanently using a `<modelname>.p2` INI file. 

### Parameter Resolution Priority
The code dynamically merges parameters with the following strict priority:
1. **Explicit CLI Flags** (Highest priority, completely overwrites `.p2` and defaults)
2. **`.p2` Parameter File** (Overwrites built-in defaults)
3. **Built-in Defaults** (Lowest priority)

This means you can set a baseline in your `.p2` file and easily override a specific value for a single run using the CLI (e.g., `pixi run python3 src/loop.py my_model --method lbfgs`).

### Supported `.p2` Parameters

#### `[mesh]`
| Parameter | Description | Default | CLI Equivalent |
| :--- | :--- | :--- | :--- |
| `size` | Size of one mesh unit in meters (e.g., `1e-9` for nm). | `1e-9` | N/A |

#### `[initial state]`
| Parameter | Description | Default | CLI Equivalent |
| :--- | :--- | :--- | :--- |
| `mx`, `my`, `mz` | Uniform initial magnetization vector components. | Field direction | `--m0-dir` |

#### `[field]`
| Parameter | Description | Default | CLI Equivalent |
| :--- | :--- | :--- | :--- |
| `h` (or `hx`,`hy`,`hz`) | Applied field direction vector. | `0,0,1` | `--h-dir` |
| `hstart` | Starting magnitude of the applied field (Tesla). | `-1.0` | `--B-start` |
| `hfinal` | Final magnitude of the applied field (Tesla). | `1.0` | `--B-end` |
| `hstep` | Field sweep step size magnitude (Tesla). | `0.05` | `--dB` |
| `mstep` | Magnetization change threshold for saving `.vtu` snapshots. Setting this to a very large value (e.g. `10000`) prevents VTU outputs to save disk space. | `None` | N/A |
| `bias_type` | Symmetry-breaking initialization field (`circular` or `random`). | `None` | `--bias-type` |
| `bias_strength` | Strength of the bias field relative to saturation. | `0.0` | `--bias-strength` |

#### `[minimizer]`
| Parameter | Description | Default | CLI Equivalent |
| :--- | :--- | :--- | :--- |
| `method` | Energy minimization algorithm. | `pcohen_hs` | `--method` |
| `max_iter` | Maximum inner iterations per field step. | `2000` | `--max-iter` |
| `tol_fun` | Relative energy convergence tolerance. | `1e-8` | `--tau-f` |
| `eps_a` | Absolute tangent gradient norm tolerance. | `1e-12` | `--eps-a` |
| `tau0` | Initial step size guess. | `0.01` | `--tau0` |
| `tau_min` | Minimum allowed step size. | `1e-6` | `--tau-min` |
| `tau_max` | Maximum allowed step size. | `1.0` | `--tau-max` |
| `pc_iters` | Inner iteration limit for preconditioning solvers. | `10` | `--pc-iters` |
| `pc_auto` | Enable automated tuning of preconditioning. | `True` | `--pc-auto` |
| `pc_force_eta` | Base forcing parameter for adaptive preconditioning. | `0.5` | `--pc-force-eta` |
| `pc_force_alpha`| Exponent forcing parameter for adaptive preconditioning. | `0.5` | `--pc-force-alpha` |
| `pc_stagnation_nu`| Stagnation threshold for quadratic models. | `0.01` | `--pc-stagnation-nu` |
| `memory` | History size for L-BFGS and Anderson acceleration. | `5` | `--memory` |
| `tn_iters` | Maximum inner iterations for Truncated Newton-CG. | `5` | `--tn-iters` |
| `lr` | Learning rate for Nesterov accelerated gradient methods. | `0.1` | `--lr` |
| `mu` | Momentum factor for Nesterov accelerated gradient methods. | `0.9` | `--mu` |
| `wg_gamma` | Steps in convex region before switching to BB. | `5` | `--wg-gamma` |
| `wg_threshold` | Convexity threshold for the WG algorithm. | `1e-6` | `--wg-threshold` |

#### `[poisson]`
| Parameter | Description | Default | CLI Equivalent |
| :--- | :--- | :--- | :--- |
| `cg_maxiter` | Maximum iterations for Poisson PCG solver. | `2000` | `--cg-maxiter` |
| `cg_tol` | Relative residual tolerance for Poisson PCG solver. The actual value passed to the solver is dynamically capped to be at least an order of magnitude tighter than the minimizer's relative energy tolerance (`min(cg_tol, tau_f * 0.1)`). | `1e-8` | `--cg-tol` |
| `reg` | Tikhonov regularization for the Poisson operator. | `1e-12`| `--poisson-reg` |

## 11. Evaluate Materials Workflow

The `examples/evaluate_materials` directory provides an automated pipeline for generating fixed sets of granular structures and evaluating different intrinsic magnetic properties across them.

**1. Generate Base Structures:**
```bash
cd examples/evaluate_materials
pixi run python generate_structures.py --extent 80,80,80 --grains 8 --num-structures 10
```
*(Alternatively, submit `sbatch run_generate_structures.slurm` to a cluster).*
This generates 10 fixed meshes with their random easy-axis orientations permanently seeded in `isotrop.krn` under the `base_structures/` folder.

**2. Evaluate Magnetic Properties:**
```bash
pixi run -e cuda python evaluate_properties.py --K1 700000 --Js 0.8 --A 7.6e-11
```
*(Alternatively, submit `sbatch run_evaluate_properties.slurm` to a cluster).*
This wrapper script orchestrates two distinct phases for the specified intrinsic properties across all 10 structures:
- **Compute Phase:** Overwrites `K1`, `Js`, and `A` while preserving the fixed easy-axes, and runs `loop.py` sequentially for each structure.
- **Analyze Phase:** Analyzes the simulation outputs, automatically handling demagnetization field shearing and detecting overskewed coercivities.

Results are written to the `evaluations/` directory, including a visual plot (`*_demag_curves.png`) and three strictly typed, ontology-mapped CSV files:
- `evaluation_results_average.csv`: The full average demagnetization loop curve.
- `evaluation_results_individual.csv`: The discrete $H_c$ and $J_r$ properties for each individual mesh.
- `evaluation_summary_scalars.csv`: A single-row summary of the geometry and intrinsic/extrinsic properties fully typed using the MaMMoS EMMO ontology.

**3. Clean Up Data:**
```bash
./clean_evaluations.sh
```
Safely deletes the heavy simulation output directories inside `evaluations/` while preserving your evaluation CSV results and plots.
