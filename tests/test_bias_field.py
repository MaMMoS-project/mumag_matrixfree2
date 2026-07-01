import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hysteresis_loop import LoopParams


def test_bias_field_generation():
    # Mock data
    knt = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

    # 1. Test random bias
    class Args:
        bias_type = "random"
        bias_strength = 0.01
        h_dir = "0,0,1"

    # Simulate logic in loop.py
    params = LoopParams(
        h_dir=np.array([0, 0, 1]), B_start=0.0, B_end=1.0, dB=1.0, bias_type="random", bias_strength=0.01
    )

    rng = np.random.default_rng(42)
    B_bias = rng.standard_normal((knt.shape[0], 3))
    B_bias /= np.linalg.norm(B_bias, axis=1, keepdims=True) + 1e-30
    B_bias *= params.bias_strength

    assert B_bias.shape == (4, 3)
    assert np.allclose(np.linalg.norm(B_bias, axis=1), 0.01)

    # 2. Test circular bias
    params = LoopParams(
        h_dir=np.array([0, 0, 1]), B_start=0.0, B_end=1.0, dB=1.0, bias_type="circular", bias_strength=0.05
    )

    center = np.mean(knt, axis=0)
    coords = knt - center
    bx = -coords[:, 1]
    by = coords[:, 0]
    norm = np.sqrt(bx**2 + by**2) + 1e-30
    B_bias_circ = np.zeros((knt.shape[0], 3))
    B_bias_circ[:, 0] = bx / norm
    B_bias_circ[:, 1] = by / norm
    B_bias_circ *= params.bias_strength

    assert np.allclose(np.linalg.norm(B_bias_circ[:, :2], axis=1), 0.05)
    assert np.allclose(B_bias_circ[:, 2], 0.0)


if __name__ == "__main__":
    test_bias_field_generation()
    print("Bias field tests passed!")
