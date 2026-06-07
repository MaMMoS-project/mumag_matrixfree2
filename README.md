# MaMMoS-MuMag: Matrix-Free Micromagnetics with JAX

MaMMoS-MuMag is a high-performance micromagnetic simulation package built on **JAX**. It utilizes a **matrix-free FEM** approach for the Poisson equation (demagnetization field) to enable large-scale simulations on both CPU and GPU architectures without the memory overhead of storing global stiffness matrices.

## 1. Installation

The project uses **Pixi** for dependency management and environment isolation.

### Prerequisites
Install Pixi if you haven't already:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Option (i): CPU-only
To install and run on a system without a compatible GPU:
```bash
pixi run -e cpu <task_name>
```
(The default environment points to `cpu` if no `-e` is specified).

### Option (ii): CPU + GPU (CUDA)
To utilize NVIDIA GPU acceleration, ensure you have the appropriate drivers and CUDA toolkit installed, then run:
```bash
pixi run -e cuda <task_name>
```

### Dependencies
The following core dependencies are managed automatically by Pixi:
- **JAX & JAXOpt**: High-performance numerical computing and optimization.
- **PyAMG**: Algebraic Multigrid solvers (used for preconditioning on CPU).
- **MeshPy**: Interface to TetGen for shell generation.
- **Neper & POV-Ray**: Polyhedral tessellation and visualization.
- **Meshio & VTK**: Mesh conversion and output.

## 2. Methodology & Background

### Energy Minimization
The package implements a **Curvilinear Search Method** specifically tailored for micromagnetics to maintain the unit-length constraint $|m|=1$ at every node. 

The default optimizer is **Preconditioned Cohen CG (Strict Auto)**, which uses physics-based preconditioning for the exchange interaction and adaptive accuracy tuning.

#### Choice of Optimizers
- **`pcohen` (Default)**: Preconditioned Cohen Conjugate Gradient with Polak-Ribière update. Superior for most problems.
- **`pcohen_hs`**: Preconditioned Cohen CG with Hestenes-Stiefel update.
- **`bb`**: Standard Barzilai-Borwein with curvilinear line-search fallback.
- **`cohen`**: Cohen CG without preconditioning.
- **`pcg`**: Preconditioned Nonlinear CG using Hestenes-Stiefel (Exl 2019).
- **`lbfgs`**: Standard Limited-memory BFGS.
- **`plbfgs`**: Preconditioned L-BFGS using the local Hessian for initial scaling.
- **`dplbfgs`**: Damped Preconditioned L-BFGS with Powell's damping for improved stability.
- **`rplbfgs`**: Riemannian PL-BFGS using vector transport (tangent space projection).
- **`tn` / `tn_split`**: Truncated Newton-CG (Full or local-only Hessian).
- **`pbb`**: Preconditioned Barzilai-Borwein.
- **`pbbs`**: Preconditioned Barzilai-Borwein with Steihaug engine.
- **`pcohen_lbfgs`**: LBFGS-Preconditioned Cohen CG Hybrid. Superior for complex landscapes.
- **`tr`**: Trust-Region Newton-CG (Steihaug-Toint).
- **`aapg`**: Anderson Accelerated Preconditioned Gradient.
- **`pnag`**: Preconditioned Nesterov Accelerated Gradient.
- **`wg`**: Algorithm 2 of Wen and Goldfarb (2009) - Non-monotone Curvilinear Search with BB steps.

#### Minimizer Parameters
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `method` | `pcohen` | Minimizer algorithm. |
| `pc_iters` | `6` | Maximum inner iterations for the preconditioner. |
| `pc_auto` | `True` | **Adaptive Forcing**: Enable dynamic tuning of preconditioning accuracy based on the **Eisenstat-Walker** formula: `pc_tol = min(eta, |g|^alpha) * |g|`. |
| `pc_force_eta` | `0.1` | Forcing parameter $\eta_{base}$: limits maximum preconditioning laziness. |
| `pc_force_alpha` | `1.0` | Forcing parameter $\alpha$: controls how fast accuracy tightens as $|g| \to 0$. |
| `phi_extrapolate`| `True` | Enable linear extrapolation of scalar potential for faster Poisson solves. |
| `memory` | `5` | History size for L-BFGS and Anderson acceleration. |
| `tn_iters` | `5` | Inner iterations for Newton-CG solvers. |
| `lr` | `0.1` | Learning rate for Nesterov acceleration. |
| `mu` | `0.9` | Momentum factor for Nesterov acceleration. |
| `pc_reg` | `0.0` | Diagonal regularization for the preconditioner. |

#### Preconditioner Strategy: Steihaug-Toint Detection
All preconditioned methods (`pcohen`, `pcg`, `plbfgs`, etc.) utilize a **Steihaug-style exit** strategy:
- The internal linear CG solver monitors the curvature $p^T A p$.
- If **negative or zero curvature** is detected (indicating an unstable switching region), the solver immediately exits and returns the best descent direction found so far.
- This ensures the minimizer remains robust and fast even during violent magnetization reversals.

#### Preconditioner Accuracy & Early Stopping
The internal preconditioning solve ($Py = g$) is optimized for speed using a multi-layered stopping criteria:
1.  **Eisenstat-Walker Forcing**: The target precision is dynamically calculated as `pc_tol = min(eta, |g|^alpha) * |g|`. This ensures high speed when far from equilibrium and high precision near the minimum.
2.  **Hard Iteration Cap (`pc_iters`)**: Regardless of the accuracy target, the solver will always terminate and return the best available direction once `pc_iters` is reached.
3.  **Stagnation Detection**: The solver monitors the reduction in the internal quadratic model. If a step reduces the model by less than a fraction $\nu$ of the total reduction achieved so far (`pc_stagnation_nu`), the solver exits early to avoid wasting effort on negligible improvements.
4.  **Physical Robustness**: If the resulting direction is not a descent direction (e.g., due to extreme ill-conditioning), the solver automatically falls back to the projected raw gradient.

### Stopping Criteria
Convergence is determined by the criteria established by *Gill, Murray, and Wright* in "Practical Optimization" (1981):
- **U1 (Energy)**: Relative change in energy is below `tau_f` (default: 1e-8): $(E_{prev} - E) < \tau_f (1 + |E|)$.
- **U2 (Magnetization)**: Magnitude of the update step is below $\sqrt{\tau_f}$: $|m_{new} - m| < \sqrt{\tau_f} (1 + |m|)$.
- **U3 (Gradient)**: The infinity norm of the projected tangent gradient is below a threshold: $|g_{tan}|_\infty \leq \tau_f^{1/3} (1 + |E|)$.
- **U4 (Absolute)**: The tangent gradient norm is below the absolute tolerance `eps_a` (default: 1e-12).

The simulation terminates if either **(U1 AND U2 AND U3)** is satisfied, or if **U4** is reached. The absolute criterion (U4) is essential as a safety exit when numerical noise floor prevents the more stringent relative criteria from being simultaneously met.

### Matrix-Free Poisson Solver
To solve the magnetic scalar potential $U$, we use a **Preconditioned Conjugate Gradient (PCG)** method. Instead of assembling a global stiffness matrix $A$, the operator-vector product $A u$ is computed element-wise in JAX.

### Polyhedral Meshing with Grain Boundary Phase
The package supports generating polycrystalline geometries with an explicit grain boundary (GB) phase of thickness $t$.
1. **Tessellation**: Uses **Neper** to generate a Voronoi tessellation of the domain.
2. **Geometric Shrinking**: Each grain is mathematically shrunk by $t/2$ using a plane-offsetting algorithm. Instead of moving vertices, the faces (planes) are pushed inward, and new vertices are reconstructed via half-space intersection. This ensures that faces remain perfectly flat and parallel to the original interfaces.
3. **Conformal Meshing**: A single Piecewise Linear Complex (PLC) is constructed containing all shrunken grain surfaces and the outer bounding box.
4. **Multi-Material Tetrahedralization**: **TetGen** (via MeshPy) meshes the entire volume in a single pass. Grains are assigned material IDs $1 \dots N$, and the interstitial grain boundary phase is assigned ID $N+1$. The resulting mesh is perfectly conformal at all interfaces.
5. **Refinement Control**: The parameter `--h` controls the target element size **inside each grain**, while `--gb-h` specifically controls the refinement **within the thin grain boundary phase**. Providing smaller values for these parameters results in a finer mesh. Additionally, reducing the `--minratio` parameter (e.g., from 1.4 to 1.2) improves the shape quality of the tetrahedra and can further increase the mesh density by forcing more Steiner points.

### GPU Memory & Batching
To handle meshes with millions of elements on GPUs with limited memory, element-wise operations are **batched (chunked)**. 
- The parameter `--chunk-elems` (default: 200,000) controls how many elements are processed in a single JAX loop iteration. 
- Smaller values reduce peak GPU memory usage but may slightly increase overhead.
- This allows simulations to scale far beyond the memory limits of traditional matrix-assembly FEM codes.

## 3. Usage & Examples

### Running the Sample
The primary example is a demagnetization curve of a 20nm NdFeB-like cube.
```bash
pixi run sample
```
This script will:
1. Generate a 20nm cube mesh with 2nm resolution.
2. Run a field sweep from 2.0 T down to -8.0 T.
3. Save snapshots in `bench_demag_curve/` and magnetization data in `.mh` format.

### Meshing Examples
Generate polycrystalline core meshes using the `src/mesh.py` tool.

**1. Basic 12-grain cube with GB phase:**
```bash
pixi run python src/mesh.py --geom poly_gb --extent 10,10,10 --n 12 --h 2.0 --gb-thickness 0.5 --gb-h 1.0 --out-name poly_sample
```

**2. Fine mesh inside grains:**
```bash
pixi run python src/mesh.py --geom poly_gb --extent 10,10,10 --n 12 --h 0.5 --gb-thickness 0.5 --gb-h 0.25 --out-name poly_fine
```

**3. High-quality tetrahedra (refined shape):**
```bash
pixi run python src/mesh.py --geom poly_gb --extent 10,10,10 --n 12 --h 1.0 --gb-thickness 0.5 --gb-h 0.5 --minratio 1.2 --out-name poly_high_qual
```

### Configuration (.p2 file)
Simulations are primarily controlled via `.p2` files (INI format). These act as the **base settings** for a model.
**Parameter Priority**: Explicit CLI Arguments > `.p2` Configuration File > Hardcoded Defaults.
Example `cube_20nm.p2`:
```ini
[mesh]
size = 1e-9         ; Length of one mesh unit in meters (default 1e-9 for nm)

[initial state]
mx = 0.0            ; Initial magnetization x-component
my = 0.0            ; Initial magnetization y-component
mz = 1.0            ; Initial magnetization z-component (Easy axis)

[field]
hx = 0.0            ; Field direction x
hy = 0.0            ; Field direction y
hz = 1.0            ; Field direction z
hstart = 2.0        ; Start field (Tesla)
hfinal = -8.0       ; End field (Tesla)
hstep = -0.5        ; Step size magnitude (Tesla). Sign is adjusted automatically.
loop = false        ; If true, runs a full hysteresis cycle (default: false)
mstep = 0.1         ; Save snapshot if |J_par - J_last| > 0.1 T
mfinal = 0.0        ; Stop sweep if J_par <= 0.0 T
bias_type = circular ; Symmetry-breaking field ('circular' or 'random')
bias_strength = 0.01 ; Magnitude relative to saturation

[minimizer]
method = pcohen     ; Algorithm: pcohen, bb, lbfgs, etc.
tol_fun = 1e-8      ; Energy convergence tolerance (tau_f)
eps_a = 1e-12       ; Absolute tangent gradient tolerance
max_iter = 2000     ; Max iterations per field step
pc_iters = 15       ; Inner preconditioning iterations
pc_auto = true      ; Enable adaptive forcing sequence
pc_force_eta = 0.5  ; Forcing base (Eisenstat-Walker)
pc_force_alpha = 0.5; Forcing exponent (Eisenstat-Walker)
pc_stagnation_nu = 0.01; Stagnation detection threshold
phi_extrapolate = true; Enable potential extrapolation
memory = 5          ; BFGS history

[poisson]
cg_maxiter = 2000   ; Poisson solver max iterations
cg_tol = 1e-8       ; Poisson solver relative tolerance
reg = 1e-12         ; Poisson regularization (poisson_reg)
```

### Material Properties (.krn file)

The `.krn` file defines the intrinsic magnetic properties for each material group in the mesh. **Each line in the file corresponds to a material ID (Line 1 = ID 1, Line 2 = ID 2, etc.)**. Headers starting with `#` are supported.

The file expects 6 columns (Classic format):
1. **theta** (rad): Polar angle of the uniaxial easy axis.
2. **phi** (rad): Azimuthal angle of the uniaxial easy axis.
3. **K1** (J/m³): Uniaxial anisotropy constant.
4. **not used** (0.0): Placeholder for future features.
5. **Js** (Tesla): Saturation magnetic polarization.
6. **A** (J/m): Exchange stiffness constant.

### Key Parameters: mfinal, mstep, and tau limits
- **`mfinal` (Tesla)**: The threshold for early termination of the field sweep. If the volume-averaged magnetization component parallel to the field ($J_{par}$) drops to or below this value, the simulation stops. Default: None (no early stopping).
- **`mstep` (Tesla)**: The threshold for saving state snapshots. A new `.vtu` file and `config` index are generated only when the change in $J_{par}$ since the last snapshot exceeds this value. Default: None (falls back to `--snapshot-every`).
- **`Adaptive Poisson Tolerance`**: To ensure stable energy minimization, the Poisson solver's target precision (`phi_tol`) is automatically adjusted to be one order of magnitude stricter than the energy convergence target: `phi_tol = min(cg_tol, 0.1 * tau_f)`. This prevents numerical noise in the demagnetization field from stalling the minimizer.
- **`bias_type` & `bias_strength`**: (In `[field]` section) Used for symmetry breaking to trigger specific reversal modes (e.g., curling in spheres). `bias_type` can be `circular` or `random`. `bias_strength` is the magnitude relative to saturation (e.g., 0.01).
- **`tau_min` & `tau_max`**: Bounds for the Barzilai-Borwein step size. `tau_max` is particularly important as it limits the maximum rotation angle allowed in a single iteration. Default: `1e-6` to `1.0`.

## 4. Output Files

The simulation results are saved in the directory specified by `--out-dir` (default: `hyst_out`).

### Parameter Log (`params.log`)
A Markdown-formatted table generated at the start of the simulation. It lists all active configuration parameters, their final resolved values, and their source (e.g., `default`, `.p2`, or `cli`), providing a complete record of the simulation settings.

### Hysteresis Data (`hysteresis.csv`)
A CSV file containing the global results of the field sweep. Columns include:
- **`config`**: The configuration index. This increments only when a new snapshot (VTU) is saved (via `mstep` or `--snapshot-every`).
- **`B_ext_T`**: The external magnetic field in Tesla.
- **`J_par_T`**: The volume-averaged magnetic polarization parallel to the field direction in Tesla.
- **`E`**: The total dimensionless energy.
- **`gnorm`**: The norm of the energy gradient (convergence indicator).

### Compatibility Data (`*.mh`)
A space-separated text file compatible with other MaMMoS tools. It records the magnetization history with columns: `B_ext [T]`, `J_parallel [T]`, `mx`, `my`, `mz`, and `E [J/m³]`.

### Visualization Files (`*.vtu`)
Snapshots of the magnetic state in VTK format, viewable in ParaView.
**Filename Convention**: `state_cfgXXXXX_B+Y.YYYYe+00T.vtu`
- **`cfgXXXXX`**: The configuration index (5 digits). The first file is always `cfg00000` (the initial state at $B_{start}$).
- **`B+Y.YYYYe+00T`**: The external field value in Tesla at which the snapshot was taken.

## 5. CLI Reference

### `src/loop.py` (Main Driver)
The primary entry point for running hysteresis loop simulations.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `modelname` | string | **Positional**. Base name. Automatically looks for `[modelname].npz`, `[modelname].krn`, and `[modelname].p2`. |
| `--mesh` | path | Explicit path to the input NPZ mesh (knt, ijk). |
| `--add-shell` | flag | Automatically add an airbox shell around the core mesh. |
| `--layers` | int | Number of graded shell layers (default: 4). |
| `--K` | float | Geometric growth factor for shell layer thickness (default: 1.3). |
| `--beta` | float | Mesh-size/geometry coupling exponent (default: 1.0). |
| `--center` | CSV | Ray origin for shell expansion as "cx,cy,cz" (default: 0,0,0). |
| `--h0` | float | Target edge length near the body surface (mesh units). |
| `--hmax` | float | Target edge length at the outermost shell boundary (mesh units). |
| `--minratio` | float | TetGen quality minratio (-q) for shell tetrahedra (default: 1.4). |
| `--max-steiner` | int | Limit the number of Steiner points added by TetGen. |
| `--no-exact` | flag | Suppress exact arithmetic in TetGen (-X). |
| `--materials` | path | Path to a .krn file with intrinsic properties (theta, phi, K1, K2, Js, A, ...). |
| `--precond-type` | choice | Poisson preconditioner: `amgcl` (default), `jacobi`, `chebyshev`, or `amg`. |
| `--geom-backend` | choice | Gradient info strategy: `stored_JinvT` (default), `stored_grad_phi`, or `on_the_fly`. |
| `--chunk-elems` | int | Elements processed per loop iteration (default: 200,000). Controls GPU memory. |
| `--cg-maxiter` | int | Maximum iterations for the Poisson PCG solver (default: 2000). |
| `--cg-tol` | float | Relative residual tolerance for the Poisson solver (default: 1e-8). |
| `--poisson-reg` | float | Tikhonov regularization constant for Poisson diagonal (default: 1e-12). |
| `--h-dir` | CSV | Applied field direction as unit vector "hx,hy,hz" (default: 0,0,1). |
| `--B-start` | float | Starting magnitude of the applied field (Tesla, default: -1.0). |
| `--B-end` | float | Final magnitude of the applied field (Tesla, default: 1.0). |
| `--dB` | float | Field step size magnitude (Tesla, default: 0.05). |
| `--max-iter` | int | Maximum iterations for the energy minimizer per field step (default: 2000). |
| `--tau-f` | float | Relative energy convergence tolerance for the minimizer (default: 1e-8). |
| `--eps-a` | float | Absolute tangent gradient norm tolerance (default: 1e-12). |
| `--method` | choice | Energy minimizer algorithm: `pcohen` (default), `bb`, `lbfgs`, etc. |
| `--pc-iters` | int | Inner iterations for preconditioning (default: 15). |
| `--pc-auto` | flag | Enable automated tuning of preconditioning accuracy (default: True). |
| `--pc-no-auto`| flag | Disable automated tuning of preconditioning accuracy. |
| `--pc-force-eta`| float | Base forcing parameter for adaptive preconditioning (default: 0.5). |
| `--pc-force-alpha`| float | Exponent forcing parameter for adaptive preconditioning (default: 0.5). |
| `--pc-tol` | float | Absolute tolerance target for preconditioning (default: 0.0). |
| `--pc-rel-tol` | float | Relative tolerance reduction goal for preconditioning (default: 0.0). |
| `--pc-stagnation-nu` | float | Relative threshold $\nu$ for quadratic model stagnation detection (default: 0.01). |
| `--phi-extrapolate`| flag | Enable linear extrapolation of scalar potential for faster Poisson solves (default: True). |
| `--no-phi-extrapolate`| flag | Disable linear extrapolation of scalar potential. |
| `--memory` | int | History size for L-BFGS and Anderson (default: 5). |
| `--tn-iters` | int | Inner iterations for Newton-CG solvers (default: 5). |
| `--lr` | float | Learning rate for Nesterov acceleration (default: 0.1). |
| `--mu` | float | Momentum factor for Nesterov acceleration (default: 0.9). |
| `--tau-min` | float | Minimum step size allowed for the BB minimizer (default: 1e-6). |
| `--tau-max` | float | Maximum step size allowed for the BB minimizer (default: 1.0). |
| `--bias-type` | choice | Symmetry-breaking field type: `circular` or `random` (default: None). |
| `--bias-strength` | float | Strength of the bias field relative to saturation (default: 0.0). |
| `--out-dir` | path | Directory for results and snapshots (default: hyst_out). |
| `--snapshot-every` | int | Save VTU snapshots every N steps (0 to disable, default: 1). |
| `--m0-dir` | CSV | Initial magnetization direction "mx,my,mz". Defaults to field direction. |
| `--verbose` | flag | Print detailed minimizer iterations at each step. |

### `src/mesh.py` (Meshing Tool)
A versatile mesher for core magnetic bodies.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--geom` | choice | Geometry type: `box` (default), `ellipsoid`, `eye`, `elliptic_cylinder`, `poly`, or `poly_gb`. |
| `--extent` | CSV | Full dimensions Lx,Ly,Lz of the core mesh (default: 60,60,60). |
| `--h` | float | Target characteristic edge length for grains (default: 2.0). |
| `--minratio` | float | TetGen quality minratio (-q) for refinement (default: 1.4). |
| `--backend` | choice | Meshing engine: `meshpy` (TetGen, default) or `grid` (regular Freudenthal). |
| `--dir-x` | CSV | Target direction for the local x-axis (default: 1,0,0). |
| `--dir-y` | CSV | Initial direction for the local y-axis (default: 0,1,0). |
| `--dir-z` | CSV | Initial direction for the local z-axis (default: 0,0,1). |
| `--ell-subdiv` | mixed | (Ellipsoid only) Icosphere subdivision level: integer or `auto` (default). |
| `--n` | int | (Poly/Poly_GB only) Number of grains for Voronoi tessellation (default: 10). |
| `--id` | int | (Poly/Poly_GB only) Random seed for tessellation (default: 1). |
| `--gb-thickness` | float | (Poly_GB only) Total thickness of the grain boundary phase (default: 1.0). |
| `--gb-h` | float | (Poly_GB only) Target element size within the GB phase (default: 1.0). |
| `--out-name` | string | Base name for output files (default: single_solid). |
| `--no-vis` | flag | Skip writing the .vtu visualization file. |
| `--verbose` | flag | Enable verbose logging during meshing. |

### `src/add_shell.py` (Airbox Tool)
Manually add graded exterior tetrahedral layers.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--in` | path | **Required**. Input NPZ mesh containing core body 'knt' and 'ijk'. |
| `--layers` | int | Number of graded shell layers L. |
| `--K` | float | Geometric scale factor (> 1) for layer thickness. |
| `--KL` | float | Total outermost geometric scale relative to body. |
| `--auto-layers` | flag | Compute L given KL and K. |
| `--auto-K` | flag | Compute K given KL and layers. |
| `--beta` | float | Mesh-size/geometry coupling exponent (default: 1.0). |
| `--same-scaling` | flag | Shortcut for beta=1.0 and automatic hmax. |
| `--center` | CSV | Ray origin for homothetic expansion (default: 0,0,0). |
| `--h0` | float | Target edge length at the first shell layer. |
| `--hmax` | float | Target edge length at the outermost boundary. |
| `--body-h` | float | Characteristic size of the input body mesh (optional). |
| `--out-npz` | path | Optional path to save the merged mesh as an NPZ. |
| `--out-vtu` | path | Optional path to save the merged mesh as a VTU. |

### `src/mesh_convert.py` (Interoperability)
Convert between JAX-friendly NPZ and standard formats.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--in` | path | **Required**. Input mesh (.npz or .vtu). |
| `--out` | path | **Required**. Output mesh (.vtu or .npz). |

## 5. Benchmarking & Tests
- **Unit Tests**: Run `pixi run test` to verify the physics (energies, gradients, and switching fields).
- **Performance**: Run `pixi run benchmark` to compare the JAX Poisson solver against the native C++ implementation.
