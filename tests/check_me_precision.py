import jax
import jax.numpy as jnp
import numpy as np
from fem_utils import TetGeom
from energy_kernels import make_energy_kernels

def test_precision():
    jax.config.update("jax_enable_x64", True)
    
    # Single element
    knt = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=float)
    conn = np.array([[0,1,2,3]], dtype=int)
    mat_id = np.array([1], dtype=int)
    
    # Geometry
    Ve = np.array([1/6])
    grad_phi = np.array([[[-1,-1,-1], [1,0,0], [0,1,0], [0,0,1]]])
    geom = TetGeom(conn=jnp.array(conn), volume=jnp.array(Ve), mat_id=jnp.array(mat_id), grad_phi=jnp.array(grad_phi))
    
    K1_tot = 4.3e6
    Js = 1.6
    A = 7.7e-12
    V_mag = 1/6
    M_nodal = jnp.array([1/24, 1/24, 1/24, 1/24]) * Js
    
    # Run 1: Standard
    e_and_g_std, _, _ = make_energy_kernels(
        geom, 
        A_lookup=jnp.array([A]), 
        K1_lookup=jnp.array([K1_tot]), 
        Js_lookup=jnp.array([Js]),
        axes_lookup=jnp.eye(3)[None, ...],
        V_mag=V_mag,
        M_nodal=M_nodal,
        K1p_lookup=jnp.array([0.0])
    )
    
    # Run 2: Split
    e_and_g_me, _, _ = make_energy_kernels(
        geom, 
        A_lookup=jnp.array([A]), 
        K1_lookup=jnp.array([K1_tot/2]), 
        Js_lookup=jnp.array([Js]),
        axes_lookup=jnp.eye(3)[None, ...],
        V_mag=V_mag,
        M_nodal=M_nodal,
        K1p_lookup=jnp.array([0.0]),
        k1me=jnp.array([K1_tot/2])
    )
    
    m = jnp.array([[0.1, 0.2, jnp.sqrt(1 - 0.1**2 - 0.2**2)]] * 4)
    B_ext = jnp.array([0, 0, 0])
    U = jnp.zeros(4)
    
    E1, G1 = e_and_g_std(m, U, B_ext)
    E2, G2 = e_and_g_me(m, U, B_ext)
    
    print(f"E_std: {E1}")
    print(f"E_me:  {E2}")
    print(f"Diff E: {E1 - E2}")
    print(f"Max Diff G: {jnp.max(jnp.abs(G1 - G2))}")

if __name__ == "__main__":
    test_precision()
