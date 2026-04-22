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

### CLI Options for `loop.py`
- `modelname`: (Positional) Base name. Automatically looks for `[modelname].npz`, `[modelname].krn`, and `[modelname].p2`.
- `--mesh`: Explicit path to the core mesh file.
- `--materials`: Path to a `.krn` file with magnetic properties.
- `--add-shell`: Programmatically adds an airbox shell (see below).
- `--chunk-elems`: Number of elements per GPU processing batch (controls memory usage).
- `--snapshot-every`: VTU output frequency (0=none, 1=every step).
- `--precond-type`: Poisson solver preconditioner (`amgcl`, `jacobi`, `chebyshev`).
- `--eps-a`: Absolute tolerance for the tangent gradient norm.

## 4. Meshing Tools

### `mesh.py`
A versatile mesher for core magnetic bodies. Supports:
- **Geometries**: `box`, `ellipsoid`, `eye` (Bézier arcs), `elliptic_cylinder`, and `poly` (Voronoi grains via Neper).
- **Backends**: 
    - `grid`: Regular brick grid split into tetrahedra (extremely fast, recommended for simple shapes).
    - `meshpy`: Unstructured refinement via TetGen (allows quality constraints).
- **Orientation**: Full 3D orientation using `--dir-x`, `--dir-y`, and `--dir-z`.

### `add_shell.py` (The Airbox)
**No airbox is needed in your input mesh.** The simulation adds one programmatically if `--add-shell` is used.
- **Layers**: Number of concentric tetrahedral layers added outside the body.
- **Grading (K)**: The layer thickness grows geometrically. Layer $l$ is scaled by $K^l$ relative to the body surface.
- **h0**: The target edge length at the body-air interface. Should match the body's internal mesh size.
- **hmax**: The target edge length at the outermost boundary.
- **Grading Logic**: Mesh size $h_l$ grows as $h_l = h_0 \cdot K^l$, effectively reducing the node count in the far field where potential gradients are low.

### `mesh_convert.py`
A utility to convert between JAX-friendly `.npz` files and standard `.vtu` (Paraview) or `.inp` (AVS UCD) formats.
```bash
pixi run python src/mesh_convert.py --in mesh.npz --out mesh.vtu
```

## 5. Benchmarking & Tests
- **Unit Tests**: Run `pixi run test` to verify the physics (energies, gradients, and switching fields).
- **Performance**: Run `pixi run benchmark` to compare the JAX Poisson solver against the native C++ (OpenCL/VexCL) implementation.
- **Profiling**: `benchmarking/profile_compilation.py` and `profile_energy_jax.py` provide detailed XLA trace analysis.
