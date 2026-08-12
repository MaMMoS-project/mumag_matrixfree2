"""Initial magnetization generation and resolution utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io_utils import read_vtu_magnetization

SUPPORTED_STATE_NAMES = ("uniform", "flower", "vortex", "twisted", "random")
LEGACY_STATE_NAMES = {
    "0": "uniform",
    "1": "flower",
    "2": "vortex",
    "3": "twisted",
    "4": "random",
}


@dataclass(frozen=True)
class ResolvedInitialState:
    """Resolved nodal magnetization and provenance information."""

    magnetization: np.ndarray
    value: str
    source: str
    direction: np.ndarray | None = None
    direction_source: str | None = None


def normalize_direction(direction: str | Sequence[float] | np.ndarray, *, label: str) -> np.ndarray:
    """Parse and normalize a three-component direction.

    Args:
        direction: Direction components or a comma-separated string.
        label: User-facing description included in validation errors.

    Returns:
        Normalized direction with shape ``(3,)``.

    Raises:
        ValueError: If the direction is malformed, non-finite, or zero.
    """
    if isinstance(direction, str):
        try:
            vector = np.asarray([float(value.strip()) for value in direction.split(",")], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"{label} must contain three comma-separated numbers; found {direction!r}.") from exc
    else:
        vector = np.asarray(direction, dtype=np.float64)

    if vector.shape != (3,):
        raise ValueError(f"{label} must have shape (3,); found {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain only finite values; found {vector.tolist()}.")

    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{label} must be non-zero; found {vector.tolist()}.")
    return vector / norm


def _validate_points(points: np.ndarray) -> np.ndarray:
    """Validate nodal coordinates used by generated states.

    Args:
        points: Simulation node coordinates.

    Returns:
        Coordinates as a double-precision array.

    Raises:
        ValueError: If coordinates do not have shape ``(N, 3)`` or are non-finite.
    """
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"Initial-state coordinates must have shape (N, 3); found {coordinates.shape}.")
    if coordinates.shape[0] == 0:
        raise ValueError("Initial-state coordinates must contain at least one node.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Initial-state coordinates must contain only finite values.")
    return coordinates


def _normalize_field(field: np.ndarray, *, state_name: str) -> np.ndarray:
    """Validate and normalize every vector in a generated field.

    Args:
        field: Nodal vectors to normalize.
        state_name: State name included in validation errors.

    Returns:
        Normalized double-precision nodal vectors.

    Raises:
        ValueError: If vectors are malformed, non-finite, or zero length.
    """
    vectors = np.asarray(field, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"Generated {state_name!r} state must have shape (N, 3); found {vectors.shape}.")
    if not np.all(np.isfinite(vectors)):
        raise ValueError(f"Generated {state_name!r} state contains non-finite values.")

    norms = np.linalg.norm(vectors, axis=1)
    zero_indices = np.flatnonzero(norms == 0.0)
    if zero_indices.size:
        preview = zero_indices[:5].tolist()
        raise ValueError(f"Generated {state_name!r} state contains zero-length vectors at node indices {preview}.")
    return vectors / norms[:, None]


def uniform_state(points: np.ndarray, direction: str | Sequence[float] | np.ndarray) -> np.ndarray:
    """Create a uniform nodal magnetization field.

    Args:
        points: Simulation node coordinates.
        direction: Requested uniform direction.

    Returns:
        Normalized magnetization with shape ``(N, 3)``.
    """
    coordinates = _validate_points(points)
    unit_direction = normalize_direction(direction, label="Uniform initial magnetization direction")
    return np.tile(unit_direction, (coordinates.shape[0], 1))


def vortex_state(points: np.ndarray) -> np.ndarray:
    """Create a vortex circulating in the y-z plane around the x-axis.

    Args:
        points: Simulation node coordinates.

    Returns:
        Normalized vortex magnetization with shape ``(N, 3)``.
    """
    coordinates = _validate_points(points)
    center = 0.5 * (np.min(coordinates, axis=0) + np.max(coordinates, axis=0))
    relative = coordinates - center
    dy = relative[:, 1]
    dz = relative[:, 2]
    radius = np.hypot(dy, dz)
    radius_max = float(np.max(radius))
    if radius_max == 0.0:
        return uniform_state(coordinates, (1.0, 0.0, 0.0))

    core_radius = 0.14 * radius_max
    core = np.exp(-2.0 * radius / core_radius)
    swirl_amplitude = np.sqrt(np.maximum(0.0, 1.0 - np.exp(-4.0 * radius**2 / core_radius**2)))
    inverse_radius = np.divide(1.0, radius, out=np.zeros_like(radius), where=radius > 0.0)
    field = np.column_stack(
        [
            core,
            -dz * inverse_radius * swirl_amplitude,
            dy * inverse_radius * swirl_amplitude,
        ]
    )
    return _normalize_field(field, state_name="vortex")


def flower_state(points: np.ndarray) -> np.ndarray:
    """Create the legacy flower-like initial magnetization pattern.

    Args:
        points: Simulation node coordinates.

    Returns:
        Normalized flower magnetization with shape ``(N, 3)``.
    """
    coordinates = _validate_points(points)
    center = 0.5 * (np.min(coordinates, axis=0) + np.max(coordinates, axis=0))
    relative = coordinates - center
    scale = float(np.max(np.abs(relative)))
    if scale == 0.0:
        return uniform_state(coordinates, (0.0, 0.0, 1.0))

    x, y, z = (relative[:, axis] / scale for axis in range(3))
    field = np.column_stack([x * z / 10.0, y * z / 10.0, np.ones_like(z)])
    return _normalize_field(field, state_name="flower")


def twisted_state(points: np.ndarray) -> np.ndarray:
    """Create the legacy twisted flower-like initial magnetization pattern.

    Args:
        points: Simulation node coordinates.

    Returns:
        Normalized twisted magnetization with shape ``(N, 3)``.
    """
    coordinates = _validate_points(points)
    center = 0.5 * (np.min(coordinates, axis=0) + np.max(coordinates, axis=0))
    relative = coordinates - center
    scale = float(np.max(np.abs(relative)))
    if scale == 0.0:
        return uniform_state(coordinates, (0.0, 0.0, 1.0))

    x, y, z = (relative[:, axis] / scale for axis in range(3))
    radial = np.hypot(x, y)
    inverse_radial = np.divide(1.0, radial, out=np.zeros_like(radial), where=radial > 0.0)
    twist = 4.0 * np.abs(z) * np.sign(z)
    field = np.column_stack(
        [
            x * z / 10.0 - twist * y * inverse_radial,
            y * z / 10.0 + twist * x * inverse_radial,
            np.ones_like(z),
        ]
    )
    return _normalize_field(field, state_name="twisted")


def random_state(points: np.ndarray, *, seed: int = 42) -> np.ndarray:
    """Create a deterministic random nodal magnetization field.

    Args:
        points: Simulation node coordinates.
        seed: Random-number generator seed.

    Returns:
        Normalized random magnetization with shape ``(N, 3)``.
    """
    coordinates = _validate_points(points)
    generator = np.random.default_rng(seed)
    field = generator.standard_normal((coordinates.shape[0], 3))
    return _normalize_field(field, state_name="random")


def generate_named_initial_state(
    points: np.ndarray,
    state: str | int,
    *,
    uniform_direction: str | Sequence[float] | np.ndarray | None = None,
    random_seed: int = 42,
) -> np.ndarray:
    """Generate a supported named or legacy-numbered initial state.

    Args:
        points: Simulation node coordinates.
        state: State name or legacy numeric identifier.
        uniform_direction: Direction used for the uniform state.
        random_seed: Seed used for the random state.

    Returns:
        Normalized magnetization with shape ``(N, 3)``.

    Raises:
        ValueError: If the state is unknown or uniform has no direction.
    """
    state_key = str(state).strip().lower()
    state_name = LEGACY_STATE_NAMES.get(state_key, state_key)
    if state_name == "uniform":
        if uniform_direction is None:
            raise ValueError("The uniform initial state requires a magnetization direction.")
        return uniform_state(points, uniform_direction)
    if state_name == "flower":
        return flower_state(points)
    if state_name == "vortex":
        return vortex_state(points)
    if state_name == "twisted":
        return twisted_state(points)
    if state_name == "random":
        return random_state(points, seed=random_seed)
    raise ValueError(_unknown_state_message(str(state)))


def _unknown_state_message(state: str) -> str:
    """Build the user-facing error for an unsupported state selector.

    Args:
        state: Unsupported selector supplied by the user.

    Returns:
        Helpful error message listing supported inputs.
    """
    names = ", ".join(SUPPORTED_STATE_NAMES)
    return (
        f"Unknown initial state {state!r}. Supported names are {names}; legacy identifiers 0 through 4 are also "
        "accepted. A path to a Tommos .vtu snapshot may be supplied instead."
    )


def resolve_initial_state(
    points: np.ndarray,
    *,
    cli_state: str | None,
    cli_m0_dir: str | None,
    p2_state: str | None,
    p2_m0_dir: str | None,
    applied_field_dir: str | Sequence[float] | np.ndarray,
    p2_path: str | Path | None = None,
) -> ResolvedInitialState:
    """Resolve and create the initial state using the documented priority.

    Args:
        points: Final simulation mesh coordinates.
        cli_state: Explicit ``--ini`` or ``--initial-state`` value.
        cli_m0_dir: Explicit ``--m0-dir`` value.
        p2_state: Value of ``[initial state] state``.
        p2_m0_dir: Uniform direction from ``mx``, ``my``, and ``mz``.
        applied_field_dir: Resolved applied-field direction.
        p2_path: Parameter-file path used to resolve relative VTU paths.

    Returns:
        Resolved magnetization and provenance information.

    Raises:
        ValueError: If a selector or generated field is invalid.
        FileNotFoundError: If a selected VTU file does not exist.
    """
    coordinates = _validate_points(points)
    if cli_state is not None:
        selector = cli_state
        source = "CLI --ini / --initial-state"
    elif cli_m0_dir is not None:
        selector = "uniform"
        source = "CLI --m0-dir"
    elif p2_state is not None:
        selector = p2_state
        source = ".p2 state"
    elif p2_m0_dir is not None:
        selector = "uniform"
        source = ".p2 mx/my/mz"
    else:
        selector = "uniform"
        source = "applied field direction"

    selector_text = str(selector).strip()
    state_key = selector_text.lower()
    state_name = LEGACY_STATE_NAMES.get(state_key, state_key)
    if state_name in SUPPORTED_STATE_NAMES:
        if state_name == "uniform":
            if cli_m0_dir is not None:
                direction_value: str | Sequence[float] | np.ndarray = cli_m0_dir
                direction_source = "CLI --m0-dir"
            elif p2_m0_dir is not None:
                direction_value = p2_m0_dir
                direction_source = ".p2 mx/my/mz"
            else:
                direction_value = applied_field_dir
                direction_source = "applied field direction"
            direction = normalize_direction(direction_value, label="Uniform initial magnetization direction")
            magnetization = uniform_state(coordinates, direction)
            return ResolvedInitialState(magnetization, state_name, source, direction, direction_source)

        magnetization = generate_named_initial_state(coordinates, state_name)
        return ResolvedInitialState(magnetization, state_name, source)

    candidate = Path(selector_text).expanduser()
    if candidate.suffix.lower() != ".vtu":
        raise ValueError(_unknown_state_message(selector_text))
    if not candidate.is_absolute() and source == ".p2 state" and p2_path is not None:
        candidate = Path(p2_path).resolve().parent / candidate
    candidate = candidate.resolve()
    magnetization = read_vtu_magnetization(candidate, coordinates)
    return ResolvedInitialState(magnetization, str(candidate), source)
