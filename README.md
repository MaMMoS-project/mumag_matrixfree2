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
The package implements a **Curvilinear Search Method** for p-Harmonic flows on spheres, specifically tailored for micromagnetics to maintain the unit-length constraint $|m|=1$ at every node.

The algorithm follows **Algorithm 2** from *Goldfarb et al. (2009)* (DOI: [10.1137/080726926](https://doi.org/10.1137/080726926)):
1. **Torque Projection**: The raw energy gradient $g$ is used to form a skew-symmetric matrix (or torque field $H = m \times g$) that generates a descent direction in the tangent space.
2. **Cayley Transform**: Magnetization is updated along a "curvilinear path" using a Cayley transform:
   $m(\tau) = (I + \frac{\tau}{2} W)^{-1} (I - \frac{\tau}{2} W) m(0)$
   where $W$ is a skew-symmetric matrix. This is a rotation-preserving map that keeps the vectors on the unit sphere exactly, avoiding the cumulative errors of simple normalization.
3. **Barzilai-Borwein (BB) Step**: Dynamic step sizes are calculated using the BB method (alternating spectral steps) to provide quasi-Newton acceleration without Hessian storage.

### Stopping Criteria
Convergence is determined by the criteria established by *Gill, Murray, and Wright* in "Practical Optimization" (1981):
- **U1 (Energy)**: Relative change in energy is below `tau_f`: $(E_{prev} - E) < \tau_f (1 + |E|)$.
- **U2 (Magnetization)**: Magnitude of the update step is below $\sqrt{\tau_f}$: $|m_{new} - m| < \sqrt{\tau_f} (1 + |m|)$.
- **U3 (Gradient)**: The infinity norm of the projected tangent gradient is below a threshold: $|g_{tan}|_\infty \leq \tau_f^{1/3} (1 + |E|)$.
- **U4 (Absolute)**: The tangent gradient norm is below the absolute tolerance `eps_a`.

### Matrix-Free Poisson Solver
To solve the magnetic scalar potential $U$, we use a **Preconditioned Conjugate Gradient (PCG)** method. Instead of assembling a global stiffness matrix $A$, the operator-vector product $A u$ is computed element-wise in JAX.

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

### Configuration (.p2 file)
Simulations are controlled via `.p2` files (INI format). Example `cube_20nm.p2`:
```ini
[mesh]
size = 1e-9    ; Length of one mesh unit in meters (default 1e-9 for nm)

[field]
hx = 0.0            ; Field x-component
hy = 0.0            ; Field y-component
hz = 1.0            ; Field z-component
hstart = 2.0        ; Start field (Tesla)
hfinal = -8.0       ; End field (Tesla)
hstep = -0.5        ; Step size (Tesla)
loop = false        ; If true, runs a full hysteresis cycle back to hstart

[minimizer]
tol_fun = 1e-6      ; tau_f tolerance
precond_iter = 400  ; Poisson solver max iterations
```

## 4. CLI Reference

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
| `--cg-maxiter` | int | Maximum iterations for the Poisson PCG solver (default: 400). |
| `--cg-tol` | float | Relative residual tolerance for the Poisson solver (default: 1e-8). |
| `--poisson-reg` | float | Tikhonov regularization constant for Poisson diagonal (default: 1e-12). |
| `--h-dir` | CSV | Applied field direction as unit vector "hx,hy,hz" (default: 0,0,1). |
| `--B-start` | float | Starting magnitude of the applied field (Tesla, default: -1.0). |
| `--B-end` | float | Final magnitude of the applied field (Tesla, default: 1.0). |
| `--dB` | float | Field step size magnitude (Tesla, default: 0.05). |
| `--tau-f` | float | Relative energy convergence tolerance for the minimizer (default: 1e-6). |
| `--eps-a` | float | Absolute tangent gradient norm tolerance (default: 1e-10). |
| `--out-dir` | path | Directory for results and snapshots (default: hyst_out). |
| `--snapshot-every` | int | Save VTU snapshots every N steps (0 to disable, default: 1). |
| `--m0-dir` | CSV | Initial magnetization direction "mx,my,mz". Defaults to field direction. |
| `--verbose` | flag | Print detailed minimizer iterations at each step. |

### `src/mesh.py` (Meshing Tool)
A versatile mesher for core magnetic bodies.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--geom` | choice | Geometry type: `box` (default), `ellipsoid`, `eye`, `elliptic_cylinder`, or `poly`. |
| `--extent` | CSV | Full dimensions Lx,Ly,Lz of the core mesh (default: 60,60,60). |
| `--h` | float | Target characteristic edge length (default: 2.0). |
| `--minratio` | float | TetGen quality minratio (-q) for refinement (default: 1.4). |
| `--backend` | choice | Meshing engine: `meshpy` (TetGen, default) or `grid` (regular Freudenthal). |
| `--dir-x` | CSV | Target direction for the local x-axis (default: 1,0,0). |
| `--dir-y` | CSV | Initial direction for the local y-axis (default: 0,1,0). |
| `--dir-z` | CSV | Initial direction for the local z-axis (default: 0,0,1). |
| `--ell-subdiv` | mixed | (Ellipsoid only) Icosphere subdivision level: integer or `auto` (default). |
| `--n` | int | (Poly only) Number of grains for Voronoi tessellation (default: 10). |
| `--id` | int | (Poly only) Random seed for tessellation (default: 1). |
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
