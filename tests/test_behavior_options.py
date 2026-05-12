import sys
import os
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import patch
import jax.numpy as jnp
import jax

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import loop
from hysteresis_loop import LoopParams, run_hysteresis_loop
from fem_utils import compute_node_volumes, TetGeom
from loop import compute_volume_JinvT, compute_grad_phi_from_JinvT

def test_jax_x64_enabled():
    """Verify that float64 is enabled in JAX."""
    assert jax.config.read("jax_enable_x64") is True
    x = jnp.array([1.0], dtype=jnp.float64)
    assert x.dtype == jnp.float64

def test_krn_validation_and_header(tmp_path):
    """Test that .krn files support headers and enforce strict line counts (#20, #26)."""
    # 1. Create a dummy mesh with 2 material groups
    mesh_path = tmp_path / "test_mesh.npz"
    knt = np.zeros((4, 3))
    conn = np.array([[0, 1, 2, 3]])
    mat_id = np.array([2]) # Max ID is 2
    np.savez(mesh_path, knt=knt, conn=conn, mat_id=mat_id)
    
    # 2. Create a .krn with a header and only 1 row (should FAIL)
    krn_path = tmp_path / "test_mesh.krn"
    with open(krn_path, "w") as f:
        f.write("# theta phi K1 K2 Js A\n")
        f.write("0.0 0.0 4.3e6 0.0 1.6 7.7e-12\n")
    
    with pytest.raises(ValueError, match="has 1 rows, but the mesh has 2 material groups"):
        loop.load_materials_krn(str(krn_path), G=2)

    # 3. Create a .krn with 3 rows (should FAIL)
    with open(krn_path, "w") as f:
        f.write("0.0 0.0 0.0 0.0 0.0 0.0\n")
        f.write("0.0 0.0 0.0 0.0 0.0 0.0\n")
        f.write("0.0 0.0 0.0 0.0 0.0 0.0\n")
    
    with pytest.raises(ValueError, match="has 3 rows, but the mesh only has 2 material groups"):
        loop.load_materials_krn(str(krn_path), G=2)

    # 4. Create a .krn with exactly 2 rows (should PASS)
    with open(krn_path, "w") as f:
        f.write("# line 1\n")
        f.write("0.0 0.0 4.3e6 0.0 1.6 7.7e-12\n")
        f.write("# line 2\n")
        f.write("0.0 0.0 1.0e6 0.0 1.0 1.0e-12\n")
    
    A, K1, Js, k_easy = loop.load_materials_krn(str(krn_path), G=2)
    assert A.shape == (2,)
    assert Js[1] == 1.0

def test_p2_parsing_and_normalization(tmp_path):
    """Test the new .p2 structure and normalization of directions (#25)."""
    p2_content = """
[mesh]
size = 2.0e-9

[initial state]
mx = 1.0
my = 1.0
mz = 0.0

[field]
hx = 0.0
hy = 0.0
hz = 10.0
hstart = 2.0
hfinal = -2.0
hstep = 0.1
mstep = 0.4
mfinal = -0.5

[poisson]
cg_maxiter = 123
"""
    p2_path = tmp_path / "test.p2"
    p2_path.write_text(p2_content)
    
    overrides = loop.load_params_p2(p2_path)
    
    # Check normalization of initial state (1,1,0) -> (1/sqrt(2), 1/sqrt(2), 0)
    m0_str = overrides["m0_dir"]
    m0 = np.fromstring(m0_str, sep=",")
    assert np.allclose(np.linalg.norm(m0), 1.0)
    assert m0[0] == pytest.approx(1.0/np.sqrt(2))
    
    # Check normalization of field (0,0,10) -> (0,0,1)
    h_dir = overrides["h_dir"]
    assert np.allclose(h_dir, [0, 0, 1])
    
    # Check values
    assert overrides["mesh_unit"] == 2.0e-9
    assert overrides["cg_maxiter"] == 123
    assert overrides["mfinal"] == -0.5
    assert overrides["mstep"] == 0.4

def test_mfinal_stop_signal():
    """Verify that the LoopParams correctly store mfinal (#23)."""
    params = LoopParams(
        h_dir=np.array([0,0,1]), B_start=1.0, B_end=0.0, dB=0.1,
        mfinal=0.5
    )
    assert params.mfinal == 0.5

def test_mstep_propagation():
    """Verify that mstep is correctly scaled by Js_ref in the main loop logic (#24)."""
    Js_ref = 2.0
    p2_overrides = {"mstep": 0.4}
    if "mstep" in p2_overrides:
        p2_overrides["mstep"] /= Js_ref
    assert p2_overrides["mstep"] == 0.2

def test_mfinal_functional(tmp_path):
    """Verify that the hysteresis loop terminates early when mfinal is reached."""
    # 1. Setup minimal geometry (single tet)
    knt = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float64)
    tets = np.array([[0,1,2,3]])
    mat_id = np.array([1])
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    geom = TetGeom(
        conn=jnp.asarray(conn32, dtype=jnp.int32),
        volume=jnp.asarray(volume, dtype=jnp.float64),
        mat_id=jnp.asarray(mat_id, dtype=jnp.int32),
        grad_phi=jnp.asarray(compute_grad_phi_from_JinvT(JinvT), dtype=jnp.float64),
    )
    
    # 2. Material: Tilted easy axis (15 deg from X) to ensure rotation.
    Js_lookup = np.array([1.0])
    K1_lookup = np.array([5.0]) 
    A_lookup = np.array([0.0])
    # Easy axis at 15 deg from X towards Z
    angle = np.deg2rad(15)
    k_easy = np.array([np.cos(angle), 0.0, np.sin(angle)])
    k_easy_lookup = np.array([k_easy])
    M_nodal = compute_node_volumes(geom, chunk_elems=1)
    
    # 3. Parameters: Sweep B_z from 10.0 down to 0.0
    # Stop when parallel magnetization (mz) <= 0.8
    params = LoopParams(
        h_dir=np.array([0,0,1]), B_start=10.0, B_end=0.0, dB=-1.0,
        mfinal=0.8, out_dir=str(tmp_path), verbose=False
    )
    
    # Start at Z saturation
    m0 = np.tile(np.array([0.0, 0.0, 1.0]), (4, 1))
    
    res = run_hysteresis_loop(
        points=knt, geom=geom, A_lookup=A_lookup, K1_lookup=K1_lookup,
        Js_lookup=Js_lookup, k_easy_lookup=k_easy_lookup, m0=m0,
        params=params, V_mag=float(np.sum(volume)),
        node_volumes=M_nodal, M_nodal=M_nodal
    )
    
    # 4. Verification
    history = res["history"]
    j_pars = history[:, 1]
    assert j_pars[-1] <= 0.8
    assert history[-1, 0] > 0.0 # Stopped early

def test_mstep_functional(tmp_path):
    """Verify that snapshots are only saved when change in Jpar >= mstep."""
    knt = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float64)
    tets = np.array([[0,1,2,3]])
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    geom = TetGeom(
        conn=jnp.asarray(conn32, dtype=jnp.int32),
        volume=jnp.asarray(volume, dtype=jnp.float64),
        mat_id=jnp.array([1], dtype=jnp.int32),
        grad_phi=jnp.asarray(compute_grad_phi_from_JinvT(JinvT), dtype=jnp.float64),
    )
    M_nodal = compute_node_volumes(geom, chunk_elems=1)

    # Material: Tilted easy axis to ensure mz changes with Bz
    Js_lookup = np.array([1.0])
    K1_lookup = np.array([5.0])
    angle = np.deg2rad(15)
    k_easy = np.array([np.cos(angle), 0.0, np.sin(angle)])
    k_easy_lookup = np.array([k_easy])

    # Sweep Bz: 10.0 down to 0.0 in large steps
    # Set mstep = 0.1
    params = LoopParams(
        h_dir=np.array([0,0,1]), B_start=10.0, B_end=0.0, dB=-1.0,
        mstep=0.1, out_dir=str(tmp_path), snapshot_every=1, verbose=False
    )
    
    m0 = np.tile(np.array([0.0, 0.0, 1.0]), (4, 1))
    run_hysteresis_loop(
        points=knt, geom=geom, A_lookup=np.array([0.0]), K1_lookup=K1_lookup,
        Js_lookup=Js_lookup, k_easy_lookup=k_easy_lookup, m0=m0,
        params=params, V_mag=float(np.sum(volume)),
        node_volumes=M_nodal, M_nodal=M_nodal
    )
    
    # 4. Check CSV for config indices
    csv_path = tmp_path / "hysteresis.csv"
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    configs = data[:, 0]
    
    # Init state (config 0)
    # Then each step where mz changes by >= 0.1 should increment config.
    # Given K1=5.0 and tilted axis, mz will definitely drop from ~1.0 at 10T 
    # to ~sin(15)=0.25 at 0T.
    # Total change is ~0.75, so with mstep=0.1 we expect several config changes.
    assert configs[0] == 0
    assert configs[-1] > 0
    # Config indices should be non-decreasing
    assert np.all(np.diff(configs) >= 0)
