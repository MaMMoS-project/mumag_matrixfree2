import argparse
from pathlib import Path

import numpy as np
import pytest

from tommos.initial_state import generate_named_initial_state, resolve_initial_state, uniform_state, vortex_state
from tommos.io_utils import read_vtu_magnetization, write_vtu_tetra
from tommos.loop import add_initial_state_arguments, load_params_p2

POINTS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
TETS = np.array([[0, 1, 2, 3]], dtype=np.int32)


def _write_snapshot(path: Path, points: np.ndarray, magnetization: np.ndarray | None) -> None:
    """Write a small VTU snapshot for loader tests.

    Args:
        path: Destination VTU path.
        points: Snapshot coordinates.
        magnetization: Optional point data named ``m``.
    """
    point_data = {} if magnetization is None else {"m": magnetization}
    write_vtu_tetra(path, points, TETS, point_data=point_data)


@pytest.mark.parametrize("flag", ["--ini", "--initial-state"])
def test_cli_initial_state_aliases_use_the_same_destination(flag: str) -> None:
    """Parse both initial-state aliases into one argument."""
    parser = argparse.ArgumentParser()
    add_initial_state_arguments(parser)
    args = parser.parse_args([flag, "vortex"])
    assert args.initial_state == "vortex"


def test_p2_parsing_reads_named_initial_state(tmp_path: Path) -> None:
    """Retain a named initial state from a p2 file."""
    p2_path = tmp_path / "model.p2"
    p2_path.write_text("[initial state]\nstate = vortex\nmx = 0\nmy = 0\nmz = 1\n", encoding="utf-8")
    params = load_params_p2(p2_path)
    assert params["initial_state"] == "vortex"


def test_p2_parsing_reads_vtu_path(tmp_path: Path) -> None:
    """Retain a VTU selector from a p2 file."""
    p2_path = tmp_path / "model.p2"
    p2_path.write_text(
        "[initial state]\nstate = states/previous_state.vtu\nmx = 0\nmy = 0\nmz = 1\n",
        encoding="utf-8",
    )
    params = load_params_p2(p2_path)
    assert params["initial_state"] == "states/previous_state.vtu"


def test_cli_initial_state_overrides_p2_state() -> None:
    """Give an explicit CLI state priority over the p2 state."""
    resolved = resolve_initial_state(
        POINTS,
        cli_state="vortex",
        cli_m0_dir=None,
        p2_state="flower",
        p2_m0_dir="0,0,1",
        applied_field_dir=(0.0, 0.0, 1.0),
    )
    assert resolved.value == "vortex"
    assert resolved.source == "CLI --ini / --initial-state"
    np.testing.assert_allclose(resolved.magnetization, vortex_state(POINTS))


def test_cli_m0_dir_overrides_p2_state_without_cli_initial_state() -> None:
    """Give an explicit CLI uniform direction priority over the p2 state."""
    resolved = resolve_initial_state(
        POINTS,
        cli_state=None,
        cli_m0_dir="1,2,0",
        p2_state="vortex",
        p2_m0_dir="0,0,1",
        applied_field_dir=(0.0, 0.0, 1.0),
    )
    assert resolved.value == "uniform"
    assert resolved.source == "CLI --m0-dir"
    np.testing.assert_allclose(resolved.magnetization, np.tile(np.array([1.0, 2.0, 0.0]) / np.sqrt(5.0), (6, 1)))


def test_uniform_state_uses_normalized_requested_direction() -> None:
    """Normalize the requested uniform direction at every node."""
    magnetization = uniform_state(POINTS, (0.0, 3.0, 4.0))
    np.testing.assert_allclose(magnetization, np.tile([0.0, 0.6, 0.8], (POINTS.shape[0], 1)))


def test_uniform_state_rejects_zero_direction() -> None:
    """Reject a zero uniform direction with a clear error."""
    with pytest.raises(ValueError, match="must be non-zero"):
        uniform_state(POINTS, (0.0, 0.0, 0.0))


def test_vortex_state_is_finite_normalized_nodal_field() -> None:
    """Generate a valid vortex, including at a node on its axis."""
    magnetization = vortex_state(POINTS)
    assert magnetization.shape == (POINTS.shape[0], 3)
    assert np.all(np.isfinite(magnetization))
    np.testing.assert_allclose(np.linalg.norm(magnetization, axis=1), 1.0)


@pytest.mark.parametrize("state", ["flower", "twisted", "random"])
def test_additional_named_states_are_finite_and_normalized(state: str) -> None:
    """Generate finite unit vectors for each optional legacy state."""
    magnetization = generate_named_initial_state(POINTS, state)
    assert np.all(np.isfinite(magnetization))
    np.testing.assert_allclose(np.linalg.norm(magnetization, axis=1), 1.0)


def test_random_state_is_deterministic() -> None:
    """Use a reproducible default seed for random initialization."""
    first = generate_named_initial_state(POINTS, "random")
    second = generate_named_initial_state(POINTS, "random")
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("identifier", "name"),
    [(0, "uniform"), (1, "flower"), (2, "vortex"), (3, "twisted"), (4, "random")],
)
def test_legacy_numeric_state_identifiers(identifier: int, name: str) -> None:
    """Map established numeric identifiers to their corresponding state names."""
    direction = (0.0, 0.0, 1.0) if identifier == 0 else None
    actual = generate_named_initial_state(POINTS, identifier, uniform_direction=direction)
    expected = generate_named_initial_state(POINTS, name, uniform_direction=direction)
    np.testing.assert_array_equal(actual, expected)


def test_unknown_state_has_helpful_error() -> None:
    """List supported names and VTU input in unknown-state errors."""
    with pytest.raises(ValueError, match=r"Unknown initial state.*uniform.*vortex.*\.vtu"):
        generate_named_initial_state(POINTS, "not-a-state")


def test_existing_uniform_p2_behavior_is_preserved(tmp_path: Path) -> None:
    """Resolve legacy p2 magnetization components as a normalized uniform field."""
    p2_path = tmp_path / "model.p2"
    p2_path.write_text("[initial state]\nmx = 0\nmy = 3\nmz = 4\n", encoding="utf-8")
    params = load_params_p2(p2_path)
    resolved = resolve_initial_state(
        POINTS,
        cli_state=None,
        cli_m0_dir=None,
        p2_state=params.get("initial_state"),
        p2_m0_dir=params.get("m0_dir"),
        applied_field_dir=(1.0, 0.0, 0.0),
        p2_path=p2_path,
    )
    assert resolved.source == ".p2 mx/my/mz"
    np.testing.assert_allclose(resolved.magnetization, np.tile([0.0, 0.6, 0.8], (POINTS.shape[0], 1)))


def test_vtu_round_trip_normalizes_magnetization(tmp_path: Path) -> None:
    """Read and normalize point data written by the existing VTU writer."""
    path = tmp_path / "state.vtu"
    magnetization = np.tile([0.0, 3.0, 4.0], (POINTS.shape[0], 1))
    _write_snapshot(path, POINTS, magnetization)
    loaded = read_vtu_magnetization(path, POINTS)
    np.testing.assert_allclose(loaded, np.tile([0.0, 0.6, 0.8], (POINTS.shape[0], 1)))


def test_p2_vtu_path_resolves_relative_to_p2_file(tmp_path: Path) -> None:
    """Resolve relative p2 snapshot paths from the parameter-file directory."""
    model_dir = tmp_path / "model"
    state_dir = model_dir / "states"
    state_dir.mkdir(parents=True)
    p2_path = model_dir / "cube.p2"
    p2_path.write_text("[initial state]\nstate = states/start.vtu\n", encoding="utf-8")
    snapshot_path = state_dir / "start.vtu"
    _write_snapshot(snapshot_path, POINTS, np.tile([1.0, 0.0, 0.0], (POINTS.shape[0], 1)))

    resolved = resolve_initial_state(
        POINTS,
        cli_state=None,
        cli_m0_dir=None,
        p2_state="states/start.vtu",
        p2_m0_dir=None,
        applied_field_dir=(0.0, 0.0, 1.0),
        p2_path=p2_path,
    )
    assert resolved.value == str(snapshot_path.resolve())
    np.testing.assert_allclose(resolved.magnetization, np.tile([1.0, 0.0, 0.0], (POINTS.shape[0], 1)))


def test_vtu_without_m_point_data_is_rejected(tmp_path: Path) -> None:
    """Require point data named m."""
    path = tmp_path / "missing_m.vtu"
    _write_snapshot(path, POINTS, None)
    with pytest.raises(ValueError, match="point data named 'm'"):
        read_vtu_magnetization(path, POINTS)


def test_vtu_with_wrong_m_shape_is_rejected(tmp_path: Path) -> None:
    """Require point data m to have three components per node."""
    path = tmp_path / "wrong_shape.vtu"
    _write_snapshot(path, POINTS, np.ones((POINTS.shape[0], 2)))
    with pytest.raises(ValueError, match=r"expected \(6, 3\), found \(6, 2\)"):
        read_vtu_magnetization(path, POINTS)


def test_vtu_with_wrong_node_count_is_rejected(tmp_path: Path) -> None:
    """Reject snapshots from meshes with a different node count."""
    path = tmp_path / "wrong_count.vtu"
    fewer_points = POINTS[:-1]
    _write_snapshot(path, fewer_points, np.tile([1.0, 0.0, 0.0], (fewer_points.shape[0], 1)))
    with pytest.raises(ValueError, match="node count mismatch.*expected 6, found 5"):
        read_vtu_magnetization(path, POINTS)


def test_vtu_with_different_coordinate_order_is_rejected(tmp_path: Path) -> None:
    """Reject equal-size meshes whose coordinate ordering differs."""
    path = tmp_path / "wrong_order.vtu"
    reordered_points = POINTS[[1, 0, 2, 3, 4, 5]]
    _write_snapshot(path, reordered_points, np.tile([1.0, 0.0, 0.0], (POINTS.shape[0], 1)))
    with pytest.raises(ValueError, match="coordinates or node ordering do not match"):
        read_vtu_magnetization(path, POINTS)


def test_vtu_with_zero_vector_is_rejected(tmp_path: Path) -> None:
    """Reject zero-length magnetization vectors."""
    path = tmp_path / "zero_vector.vtu"
    magnetization = np.tile([1.0, 0.0, 0.0], (POINTS.shape[0], 1))
    magnetization[2] = 0.0
    _write_snapshot(path, POINTS, magnetization)
    with pytest.raises(ValueError, match="zero-length vectors at node indices \[2\]"):
        read_vtu_magnetization(path, POINTS)


def test_vtu_with_non_finite_value_is_rejected(tmp_path: Path) -> None:
    """Reject non-finite magnetization components."""
    path = tmp_path / "non_finite.vtu"
    magnetization = np.tile([1.0, 0.0, 0.0], (POINTS.shape[0], 1))
    magnetization[3, 1] = np.nan
    _write_snapshot(path, POINTS, magnetization)
    with pytest.raises(ValueError, match="contains 1 non-finite value"):
        read_vtu_magnetization(path, POINTS)
