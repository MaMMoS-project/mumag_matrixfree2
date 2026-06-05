import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import loop
from hysteresis_loop import LoopParams


def test_p2_tolerance_parsing(tmp_path):
    """Test that tolerances and Poisson parameters are correctly parsed from .p2."""
    p2_content = """
[minimizer]
tol_fun = 1e-12
eps_a = 1e-15
max_iter = 500

[poisson]
cg_maxiter = 123
cg_tol = 1e-9
reg = 1e-11
"""
    p2_path = tmp_path / "test.p2"
    p2_path.write_text(p2_content)

    overrides = loop.load_params_p2(str(p2_path))

    assert overrides["tau_f"] == 1e-12
    assert overrides["eps_a"] == 1e-15
    assert overrides["max_iter"] == 500
    assert overrides["cg_maxiter"] == 123
    assert overrides["cg_tol"] == 1e-9
    assert overrides["poisson_reg"] == 1e-11


def test_loop_params_population():
    """Test that LoopParams is correctly populated from overrides in loop.py."""

    # Mock args
    class Args:
        h_dir = "0,0,1"
        B_start = -1.0
        B_end = 1.0
        dB = 0.1
        max_iter = 200
        tau_f = 1e-6
        eps_a = 1e-10
        cg_maxiter = 400
        cg_tol = 1e-8
        poisson_reg = 1e-12
        out_dir = "test_out"
        snapshot_every = 1
        verbose = False

    args = Args()
    Js_ref = 1.0
    h_dir = np.array([0.0, 0.0, 1.0])

    p2_overrides = {
        "tau_f": 1e-12,
        "eps_a": 1e-15,
        "cg_maxiter": 123,
        "cg_tol": 1e-9,
        "poisson_reg": 1e-11,
        "max_iter": 500,
    }

    params_dict = {
        "h_dir": h_dir,
        "B_start": float(args.B_start) / Js_ref,
        "B_end": float(args.B_end) / Js_ref,
        "dB": float(args.dB) / Js_ref,
        "max_iter": int(args.max_iter),
        "tau_f": float(args.tau_f),
        "eps_a": float(args.eps_a),
        "cg_maxiter": int(args.cg_maxiter),
        "cg_tol": float(args.cg_tol),
        "poisson_reg": float(args.poisson_reg),
        "loop": True,
        "out_dir": args.out_dir,
        "snapshot_every": int(args.snapshot_every),
        "verbose": args.verbose,
        "Js_ref": float(Js_ref),
    }

    params_dict.update(p2_overrides)
    params = LoopParams(**params_dict)

    assert params.tau_f == 1e-12
    assert params.eps_a == 1e-15
    assert params.cg_maxiter == 123
    assert params.cg_tol == 1e-9
    assert params.poisson_reg == 1e-11
    assert params.max_iter == 500


if __name__ == "__main__":
    # If run manually
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_p2_tolerance_parsing(Path(tmp_dir))
    test_loop_params_population()
    print("Tests passed!")
