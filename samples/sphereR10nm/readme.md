# sphereR10nm

Minimal `mumag_matrixfree2` example for a spherical Fe particle with diameter `20 nm` (`R = 10 nm`).

## Files
- `sphereR10nm.krn`: intrinsic material parameters
- `sphereR10nm.p2`: mesh scale, field sweep, and minimizer settings
- `run_sphereR10nm.sh`: generates the mesh and runs the simulation

## Geometry
The particle is meshed as an ellipsoid with equal extents:
- `extent = 20,20,20`
- `h = 2.0 nm`

This corresponds to a sphere with:
- diameter `D = 20 nm`
- radius `R = 10 nm`

## Material parameters
The `.krn` file uses:
- `K1 = 4.8e4 J/m^3`
- `Js = 2.13628 T`
- `A = 1.0e-11 J/m`

## Field protocol
The `.p2` file defines:
- field tilt of about `1 degree` toward `+x`
- sweep from `+1.0 T` to `-1.0 T`
- step size `-0.01 T`

## Run
Use the CUDA Pixi environment if GPU support is available:

```bash
pixi shell -e cuda
cd samples/sphereR10nm
./run_sphereR10nm.sh
