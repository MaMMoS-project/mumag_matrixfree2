"""Test relaxation of a system with no external field.

We expect the magnetization to relax in the direction of the anisotropy direction.
"""

import shlex
import subprocess
from pathlib import Path
from textwrap import dedent

import mammos_entity as me
import numpy as np
import pytest


def _write_p2_file(filename, theta, phi):
    mx = np.sin(theta) * np.cos(phi)
    my = np.sin(theta) * np.sin(phi)
    mz = np.cos(phi)
    Path(filename.with_suffix(".p2")).write_text(
        dedent(
            f"""\
            [mesh]
            size = 1e-9

            [initial_state]
            mx = {mx}
            my = {my}
            mz = {mz}

            [field]
            hx = 0
            hy = 0
            hz = 1
            hstart = 0.0
            hfinal = 0.0
            hstep = -0.5
            """
        )
    )


def _write_krn_file(filename, Js):
    K1 = 4.3e6
    A = 7.7e-12
    Path(filename.with_suffix(".krn")).write_text(
        dedent(
            f"""\
            # theta (rad) phi (rad) K1 (J/m3) not used Js (Tesla) A (J/m)
            0.0 0.0 {K1} 0.0 {Js} {A}
            """
        )
    )


@pytest.mark.parametrize("theta_deg", [15, 30, 45, 60, 75])
def test_stoner_wohlfarth_switching(loop_bin, mesh_bin, tmp_path, subtests, theta_deg):
    """Test switch in Stoner-Wohlfarth model."""
    for phi_deg in np.random.randint(0, 359, 5):
        system_name = f"sw_{theta_deg}_{phi_deg}"
        theta = np.deg2rad(theta_deg)
        phi = np.deg2rad(phi_deg)

        # generate mesh
        L = 20.0
        h = 4.0  # Coarse for speed
        cmd = shlex.split(f"{mesh_bin} --geom box --extent {L},{L},{L} --h {h} --out-name {system_name}")
        res = subprocess.run(cmd, cwd=tmp_path)
        res.check_returncode()

        # write input files
        Js = 1.6  # Tesla
        _write_p2_file(tmp_path / system_name, theta, phi)
        _write_krn_file(tmp_path / system_name, Js)

        # run hysteresis loop without demag
        cmd = shlex.split(f"{loop_bin} {system_name} --verbose")
        res = subprocess.run(cmd, cwd=tmp_path)
        res.check_returncode()

        # extract Bc from loop
        hystloop = me.from_csv(tmp_path / f"hyst_{system_name}" / "mammos_hysteresis.csv")
        with subtests.test(msg=f"phi_deg={phi_deg}"):
            assert np.all(hystloop.J_par_T.value >= 0.9 * Js)
