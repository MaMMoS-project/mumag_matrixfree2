#!/usr/bin/env python3

# NOTE: This script must be run from the main 'mammos-mumag-matrixfree' folder!
# Example:
#   $ python examples/sensor_loop/sensor_loop_step_by_step.py
# Running from any other directory may result in missing file errors.

# MaMMoS Benchmark 2 (Sensor) - Step-by-Step Workflow
# ---------------------------------------------------
# This script automates the full simulation workflow for the MaMMoS sensor benchmark (Deliverable 6.2, Chapter 3).
#
# Steps:
#   0. Mesh selection/generation and cleanup
#   1. Compute initial equilibrium magnetization state
#   2. Distribute initial state to all down-case directories
#   3-11. For each case (a: easy-axis, b: 45-degree, c: hard-axis):
#         - Run down-sweep (decreasing field)
#         - Transfer state to up-case
#         - Run up-sweep (increasing field)
#
# Usage: Run from the /examples/sensor_loop directory:
#   python sensor_loop_step_by_step.py [options]
# See README.md for detailed options and examples.


from pathlib import Path
import shutil
import subprocess
import sys
import os
import argparse

# Ensure unbuffered output for real-time logging, otherwise output may be delayed
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)


def generate_workspace(sensor_loop_dir: Path):
    """Dynamically generates all case directories, .p2 configs, and .krn files."""
    import textwrap
    print("\n[WORKSPACE] Generating case directories and configuration files...")
    krn_content = textwrap.dedent("""\
        # theta (rad)   phi (rad)   K1(J/m3)   not used   Js (T)   A (J/m)
        0.0             0.0         0.0        0.0        1.005     13.0e-12
        0.0 0.0 0.0 0.0 0.0 0.0
    """)

    configs = {
        "sensor_initial_state": {
            "hstart": 0.035, "hfinal": 0.0, "hstep": -0.001,
            "hx": 1.0, "hy": 0.0, "hz": 0.0, "ini": None
        },
        "sensor_case-a_precompute": {
            "hstart": 0.0, "hfinal": 0.0314159, "hstep": 0.0005,
            "hx": 1.0, "hy": 0.0, "hz": 0.0, "ini": "00040"
        },
        "sensor_case-a_down": {
            "hstart": 0.0314159, "hfinal": -0.0314159, "hstep": -0.0005,
            "hx": 1.0, "hy": 0.0, "hz": 0.0, "ini": "00040"
        },
        "sensor_case-a_up": {
            "hstart": -0.0314159, "hfinal": 0.0314159, "hstep": 0.0005,
            "hx": 1.0, "hy": 0.0, "hz": 0.0, "ini": "00040"
        },
        "sensor_case-b_precompute": {
            "hstart": 0.0, "hfinal": 0.0314159, "hstep": 0.0005,
            "hx": 1.414, "hy": 1.414, "hz": 0.0, "ini": "00040"
        },
        "sensor_case-b_down": {
            "hstart": 0.0314159, "hfinal": -0.0314159, "hstep": -0.0005,
            "hx": 1.414, "hy": 1.414, "hz": 0.0, "ini": "00040"
        },
        "sensor_case-b_up": {
            "hstart": -0.0314159, "hfinal": 0.0314159, "hstep": 0.0005,
            "hx": 1.414, "hy": 1.414, "hz": 0.0, "ini": "00040"
        },
        "sensor_case-c_precompute": {
            "hstart": 0.0, "hfinal": 0.0314159, "hstep": 0.0005,
            "hx": 0.0, "hy": 1.0, "hz": 0.0, "ini": "00040"
        },
        "sensor_case-c_down": {
            "hstart": 0.0314159, "hfinal": -0.0314159, "hstep": -0.0005,
            "hx": 0.0, "hy": 1.0, "hz": 0.0, "ini": "00040"
        },
        "sensor_case-c_up": {
            "hstart": -0.0314159, "hfinal": 0.0314159, "hstep": 0.0005,
            "hx": 0.0, "hy": 1.0, "hz": 0.0, "ini": "00040"
        }
    }

    for dname, cfg in configs.items():
        dpath = sensor_loop_dir / dname
        dpath.mkdir(parents=True, exist_ok=True)
        (dpath / "sensor.krn").write_text(krn_content)
        
        p2_lines = [
            "[mesh]",
            "size = 1e-9",
            "",
            "[initial state]",
            "mx = 0.",
            "my = 1.",
            "mz = 0."
        ]
        if cfg["ini"] is not None:
            p2_lines.append(f"ini = {cfg['ini']}")
        
        p2_lines.extend([
            "",
            "[field]",
            f"hstart = {cfg['hstart']}",
            f"hfinal = {cfg['hfinal']}",
            f"hstep = {cfg['hstep']}",
            "",
            f"hx = {cfg['hx']}",
            f"hy = {cfg['hy']}",
            f"hz = {cfg['hz']}",
            "",
            "mstep = 3.0",
            "",
            "[minimizer]",
            "tol_fun = 1e-12" if "case-c" in dname else "tol_fun = 1e-10"
        ])
        (dpath / "sensor.p2").write_text("\n".join(p2_lines) + "\n")
    print("[WORKSPACE] ✓ Workspace generation complete.")


def run_loop(loop_cmd: list[str], cwd: Path) -> None:
    """
    Run the loop.py script in the specified working directory.

    Args:
        loop_cmd: Base command list containing python, script path, and --mesh flag
        cwd: Working directory where the simulation will run

    Raises:
        FileNotFoundError: If the working directory doesn't exist
    """
    if not cwd.exists():
        raise FileNotFoundError(f"Working directory does not exist: {cwd}")

    # Create a fresh copy to avoid accumulating arguments across calls
    cmd = loop_cmd.copy()
    cmd.append("sensor")
    cmd.extend(["--mesh", str((cwd / "sensor.npz").resolve())])
    cmd.extend(["--out-dir", str(cwd.resolve())])
    print(f"  [CMD] {' '.join(cmd)}")

    subprocess.run(cmd, cwd=cwd, check=True)


def standardize_state_file_names(directory: Path, backup_name: str, simulation_name: str = "sensor") -> None:
    """
    Standardize state file names to match the simulation name prefix.

    For example, if simulation is called 'sensor', rename files like:
      sensor_backup.0050.state.npz → sensor.0050.state.npz
      other_prefix.0050.state.npz → sensor.0050.state.npz

    This ensures that state files loaded from external sources or backups have consistent naming for workflow compatibility.

    Args:
        directory: Directory containing state files
        backup_name: The specific backup file name to standardize (if provided)
        simulation_name: Expected simulation name prefix (default: "sensor")
    """
    import re
    
    if not directory.exists():
        return
    
    # Find all .state.npz files with any prefix
    pattern = r'^(.+)\.(\d+)\.state\.npz$'
    renamed_count = 0
    
    # If backup_name is specified, only standardize that specific file
    if backup_name:
        state_files = [directory / backup_name] if (directory / backup_name).exists() else []
    else:
        state_files = sorted(directory.glob("*.state.npz"))
    
    for state_file in state_files:
        match = re.match(pattern, state_file.name)
        if not match:
            continue
        
        current_prefix = match.group(1)
        step_number = match.group(2)
        expected_name = f"{simulation_name}.{step_number}.state.npz"
        
        # Only copy if prefix doesn't match the simulation name
        if current_prefix != simulation_name:
            new_path = directory / expected_name
            shutil.copy(state_file, new_path)
            print(f"  [COPY] {state_file.name} → {expected_name} (original preserved)")
            renamed_count += 1
    
    if renamed_count > 0:
        print(f"[COPY] ✓ Standardized {renamed_count} state file(s) to '{simulation_name}' prefix (originals preserved)")


def standardize_mesh_file_name(directory: Path, backup_mesh_name: str = None, simulation_name: str = "sensor") -> None:
    """
    Standardize mesh file name to match the simulation name.

    For example, if simulation is called 'sensor', rename:
      sensor_backup.npz → sensor.npz
      other_mesh.npz → sensor.npz

    Args:
        directory: Directory containing mesh files
        backup_mesh_name: The specific backup mesh file name (if provided)
        simulation_name: Expected simulation name (default: "sensor")
    """
    expected_name = f"{simulation_name}.npz"
    
    if not directory.exists():
        return
    
    # If backup_mesh_name is specified, use that; otherwise look for any .npz file that's not the expected name
    if backup_mesh_name:
        backup_mesh_path = directory / backup_mesh_name
        if backup_mesh_path.exists() and backup_mesh_name != expected_name:
            new_path = directory / expected_name
            # Remove existing mesh if present
            if new_path.exists():
                new_path.unlink()
            shutil.copy(backup_mesh_path, new_path)
            print(f"  [COPY] {backup_mesh_name} → {expected_name} (original preserved)")
            print(f"[COPY] ✓ Standardized mesh file to '{simulation_name}.npz' (original preserved)")


def find_last_state_file(directory: Path) -> str:
    """
    Find the last state file with format sensor.XXXX.state.npz and highest number XXXX.

    Args:
        directory: Directory to search for state files

    Returns:
        Filename of the last (highest numbered) state file

    Raises:
        FileNotFoundError: If no state files are found
    """
    state_files = list(directory.glob("state_cfg*.vtu"))
    if not state_files:
        raise FileNotFoundError(f"No state files found in directory: {directory}")
    state_files.sort()
    last_state_file = state_files[-1].name
    print(f"  [STATE] Found: {last_state_file} in {directory.name}")
    return last_state_file


def copy_state(
    src_dir: Path, src_name: str, dst_dir: Path, dst_name: str | None = None
) -> None:
    """
    Copy a state file between directories, optionally renaming it.

    Args:
        src_dir: Source directory containing the state file
        src_name: Name of the source state file
        dst_dir: Destination directory
        dst_name: Optional new name for the destination file (defaults to src_name)

    Raises:
        FileNotFoundError: If the source file doesn't exist
    """
    src = src_dir / src_name
    if not src.exists():
        raise FileNotFoundError(f"State file not found: {src}")
    dst = dst_dir / (dst_name or src_name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def set_p2_params(directory: Path, updates: dict[str, str]) -> None:
    """
    Force-set parameters in a sensor.p2 file within a directory.

    Ensures keys like 'ini' and 'hstep' are updated reliably regardless of spacing or order.

    Args:
        directory: Directory containing the sensor.p2 file
        updates: Mapping of parameter names to string values (without spaces)
    """
    p2_file = directory / "sensor.p2"
    if not p2_file.exists():
        print(f"  [WARNING] sensor.p2 not found in {directory.name}, cannot update {list(updates.keys())}")
        return

    with open(p2_file, "r") as f:
        lines = f.readlines()

    keys = set(updates.keys())
    new_lines: list[str] = []
    seen: set[str] = set()
    for line in lines:
        stripped = line.strip()
        replaced = False
        for k in keys:
            if stripped.startswith(f"{k} ="):
                new_lines.append(f"{k} = {updates[k]}\n")
                seen.add(k)
                replaced = True
                break
        if not replaced:
            new_lines.append(line)

    # Append any missing keys at the end to ensure presence
    for k in keys - seen:
        new_lines.append(f"{k} = {updates[k]}\n")

    with open(p2_file, "w") as f:
        f.writelines(new_lines)


def update_hstep_in_folders(directories: list[Path], new_hstep_abs: float) -> None:
    """
    Update the hstep value in sensor.p2 files across multiple directories.

    This function modifies the hstep parameter in sensor.p2 files while preserving
    the original sign (positive or negative). For example, if hstep = -0.00025 and
    new_hstep_abs = 0.003, the result will be hstep = -0.003.

    Args:
        directories: List of directory paths containing sensor.p2 files
        new_hstep_abs: New absolute value for hstep (sign will be preserved from original)

    Raises:
        FileNotFoundError: If a sensor.p2 file doesn't exist in a directory
    """
    import re
    
    print("\n" + "-" * 80)
    print("UPDATE HSTEP VALUES IN SENSOR.P2 FILES")
    print("-" * 80)
    
    updated_count = 0
    for directory in directories:
        p2_file = directory / "sensor.p2"
        
        if not p2_file.exists():
            print(f"  [WARNING] sensor.p2 not found in {directory.name}, skipping")
            continue
            
        with open(p2_file, "r") as f:
            lines = f.readlines()
        
        modified = False
        new_lines = []
        
        for line in lines:
            # Match lines like "hstep = -0.00025" or "hstep = 0.00025"
            match = re.match(r'^(\s*hstep\s*=\s*)([+-]?)(.+)$', line)
            if match:
                prefix = match.group(1)  # "hstep = "
                sign = match.group(2)     # "-" or "+" or ""
                old_value = match.group(3).strip()  # "0.00025"
                
                # Preserve the sign, update the magnitude
                new_line = f"{prefix}{sign}{new_hstep_abs}\n"
                new_lines.append(new_line)
                
                print(f"  [UPDATE] {directory.name}: hstep = {sign}{old_value} → {sign}{new_hstep_abs}")
                modified = True
                updated_count += 1
            else:
                new_lines.append(line)
        
        if modified:
            with open(p2_file, "w") as f:
                f.writelines(new_lines)
    
    print(f"[UPDATE] ✓ Updated hstep in {updated_count} file(s)")
    print("-" * 80)


def main() -> int:
    """
    Orchestrate the step-by-step sensor loop workflow for MaMMoS Benchmark 2 (Sensor).

    Steps:
      0. Mesh selection/generation and cleanup
      1. Compute initial equilibrium magnetization state
      2. Distribute initial state to all down-case directories
      3-11. For each case (a: easy-axis, b: 45-degree, c: hard-axis):
            - Run down-sweep (decreasing field)
            - Transfer state to up-case
            - Run up-sweep (increasing field)

    Returns:
        Exit code (0 for success)
    """
    # ============================================================================
    # COMMAND-LINE ARGUMENT PARSING
    # ============================================================================
    parser = argparse.ArgumentParser(
        description="Run step-by-step sensor loop simulations for MaMMoS Deliverable 6.2 benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full example with all cases (a, b, c)
  python sensor_loop_step_by_step.py
  
  # Run minimal example with all cases
  python sensor_loop_step_by_step.py --minimal
  
  # Run minimal example with only case a
  python sensor_loop_step_by_step.py --minimal --cases a
  
  # Run full example with cases a and b
  python sensor_loop_step_by_step.py --cases a b
  
  # Run with custom mesh sizes
  python sensor_loop_step_by_step.py --minimal --mesh-size-coarse 20.0
  python sensor_loop_step_by_step.py --mesh-size-fine 10.0

    # Directly specify mesh element size h (overrides coarse/fine)
    python sensor_loop_step_by_step.py --mesh-h 12.5
  
  # Update hstep in all sensor_case-*_* folders
  python sensor_loop_step_by_step.py --hstep 0.003
  
  # Load pre-computed initial state instead of computing
  python sensor_loop_step_by_step.py --load-initial-state
  
  # Load a specific initial state file by name
  python sensor_loop_step_by_step.py --initial-state-file sensor.0050.state.npz
  
    # Only compute or load the initial state and exit
    python sensor_loop_step_by_step.py --only-compute-initial-state
  
  # Load initial state with backup mesh file
  python sensor_loop_step_by_step.py --initial-state-file backup_sensor.0050.state.npz --initial-mesh-file sensor_backup.npz
        """,
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Use coarse mesh (faster) instead of fine mesh (default: False, use fine mesh)",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["a", "b", "c"],
        choices=["a", "b", "c"],
        metavar="CASE",
        help="Cases to run: a (easy-axis), b (45-degree), c (hard-axis) (default: a b c)",
    )
    parser.add_argument(
        "--no-mesh-regen",
        action="store_true",
        help="Use existing mesh file instead of regenerating (default: False, regenerate mesh)",
    )
    parser.add_argument(
        "--mesh-size-coarse",
        type=float,
        default=30.0,
        metavar="SIZE",
        help="Coarse mesh element size in mesh units (default: 30.0)",
    )
    parser.add_argument(
        "--mesh-size-fine",
        type=float,
        default=5.0,
        metavar="SIZE",
        help="Fine mesh element size in mesh units (default: 5.0)",
    )
    parser.add_argument(
        "--mesh-h",
        type=float,
        metavar="SIZE",
        help="Direct mesh element size in mesh units; overrides coarse/fine presets",
    )

    parser.add_argument(
        "--hstep",
        type=float,
        metavar="VALUE",
        help="Update hstep value in all sensor_case-*_* folders (preserves sign)",
    )
    parser.add_argument(
        "--only-hstep",
        action="store_true",
        help="Only update hstep across folders and exit (do not run simulations)",
    )
    parser.add_argument(
        "--load-initial-state",
        action="store_true",
        help="Load pre-computed initial state instead of computing it (skips Step 1)",
    )
    parser.add_argument(
        "--only-compute-initial-state",
        action="store_true",
        help="Only compute or load the initial state and exit (skip distribution and sweeps)",
    )
    parser.add_argument(
        "--initial-state-file",
        type=str,
        metavar="FILENAME",
        help="Specific initial state file to load (e.g., sensor.0050.state.npz); automatically enables --load-initial-state",
    )
    parser.add_argument(
        "--initial-mesh-file",
        nargs="?",
        const="backup_mesh_sensor.npz",
        type=str,
        metavar="FILENAME",
        default=None,
        help=(
            "Backup mesh file to use with initial state. If provided without a value, "
            "defaults to backup_mesh_sensor.npz; will be renamed to sensor.npz. "
            "The file is searched inside the sensor_initial_state directory."
        ),
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace directory to isolate runs.",
    )

    args = parser.parse_args()

    # If --initial-state-file is provided, automatically enable --load-initial-state
    if args.initial_state_file:
        args.load_initial_state = True

    # ============================================================================
    # USER CONFIGURATION FROM COMMAND-LINE ARGUMENTS
    # ============================================================================

    # Case selection: "a" = easy-axis, "b" = 45-degree, "c" = hard-axis
    # See "MaMMoS_Deliverable_6.2_Definition of benchmark.pdf", chapter 3
    cases = args.cases

    # Mesh configuration
    run_minimal_example = args.minimal
    use_existing_mesh = args.no_mesh_regen
    # Mesh size configuration for the eye sensor example
    mesh_size_coarse = args.mesh_size_coarse  # Coarse mesh element size
    mesh_size_fine = args.mesh_size_fine  # Fine mesh element size
    # Examples of mesh sizes and resulting element counts for the eye sensor example:
    # h = 30.0 creates      nodes=24727,    tets=77791
    # h = 10.0 creates      nodes=89761,    tets=292147
    # h = 5.0 creates     nodes=1044050,  tets=4454406

    # ============================================================================
    # END OF CONFIGURATION
    # ============================================================================

    print("\n" + "=" * 80)
    print("SENSOR LOOP STEP-BY-STEP WORKFLOW")
    print("=" * 80)
    print(
        "\nThis script performs hysteresis loop simulations for magnetic field sensors."
    )
    print("Workflow: mesh generation → initial state → down-sweep → up-sweep\n")

    # Resolve all paths relative to this script's directory to allow
    # running from any current working directory.
    run_dir = Path(__file__).resolve().parent
    base = run_dir.parent.parent.resolve()
    print("[PATH INFO]")
    print(f"  Base directory:        {base}")
    examples_dir = base.joinpath("examples")
    print(f"  Examples directory:    {examples_dir}")
    
    if args.workspace:
        sensor_loop_dir = Path(args.workspace).resolve()
        sensor_loop_dir.mkdir(parents=True, exist_ok=True)
    else:
        sensor_loop_dir = examples_dir.joinpath("sensor_loop")
        
    print(f"  Sensor loop directory: {sensor_loop_dir}")
    initial_dir = sensor_loop_dir.joinpath("sensor_initial_state")
    print(f"  Initial state dir:     {initial_dir}")

    # Generate workspace directories and config files dynamically
    generate_workspace(sensor_loop_dir)

    # If --hstep is provided, update all sensor_case-*_* folders before running simulations
    if args.hstep is not None:
        print("\n" + "=" * 80)
        print("HSTEP UPDATE MODE")
        print("=" * 80)
        print(f"[CONFIG] New hstep absolute value: {args.hstep}")
        
        # Find all sensor_case-*_* directories and include sensor_initial_state
        all_sensor_dirs = list(sensor_loop_dir.glob("sensor_case-*_*"))
        target_dirs = [initial_dir] + all_sensor_dirs
        
        if not all_sensor_dirs:
            print("[WARNING] No sensor_case-*_* folders found; updating sensor_initial_state only")
        else:
            print(f"[INFO] Found {len(all_sensor_dirs)} sensor_case-*_* folder(s); also updating sensor_initial_state")
            
        # Update hstep in initial state and all case folders
        update_hstep_in_folders(target_dirs, args.hstep)

        if args.only_hstep:
            print("[INFO] --only-hstep specified; exiting after hstep update.")
            return 0
        else:
            print("[INFO] Continuing with simulation workflow...\n")

    # Step0.1: select coarse, fine, or custom mesh h, newly generate mesh if needed
    # This step only creates the mesh for the eye shaped sensor, not for the surrounding air region
    print("\n" + "-" * 80)
    print("SENSOR-EXAMPLE, STEP 0.1: Mesh Selection and Generation")
    print("-" * 80)

    # Skip mesh generation if backup mesh will be provided for initial state
    if args.initial_mesh_file:
        print("[MESH] Skipping mesh generation (backup mesh will be used with initial state)")
        print(f"[MESH] Backup mesh file: {args.initial_mesh_file}")
        mesh_file_name = None  # Will not be used for distribution
    else:
        # Derive mesh type from user configuration or custom h
        custom_h = args.mesh_h
        if custom_h is not None:
            mesh_file_name = "sensor_custom_mesh.npz"
            mesh_type = "CUSTOM (h=" + str(custom_h) + ")"
        else:
            use_fine_mesh = not run_minimal_example
            if use_fine_mesh:
                mesh_file_name = "sensor_fine_mesh.npz"
                mesh_type = "FINE (h=" + str(mesh_size_fine) + ")"
            else:
                mesh_file_name = "sensor_coarse_mesh.npz"
                mesh_type = "COARSE (h=" + str(mesh_size_coarse) + ")"
        print(f"[MESH] Type: {mesh_type}")
        print(f"[MESH] File: {mesh_file_name}")
        print(
            f"[MESH] Mode: {'Generate new mesh' if not use_existing_mesh else 'Use existing mesh'}"
        )

        if not use_existing_mesh:
            mesh_script = (base / "src/mesh.py").resolve()
            mesh_gen_cmd = [
                sys.executable,
                str(mesh_script),
                "--geom",
                "eye",
                "--extent",
                "3500.0,1000.0,10.0",
                "--backend",
                "meshpy",
                "--out-name",
                mesh_file_name.replace(".npz", ""),
                "--add-shell",
            ]
            if custom_h is not None:
                mesh_gen_cmd.extend(["--h", str(custom_h)])
            else:
                if 'use_fine_mesh' in locals() and use_fine_mesh:
                    mesh_gen_cmd.extend(["--h", str(mesh_size_fine)])
                else:
                    mesh_gen_cmd.extend(["--h", str(mesh_size_coarse)])
            # mesh_            print("\n[MESH GENERATION] Starting mesh generation...")
            print(f"[COMMAND] {' '.join(mesh_gen_cmd)}")
            subprocess.run(mesh_gen_cmd, check=True, cwd=str(sensor_loop_dir))
            print("[MESH GENERATION] ✓ Mesh generated successfully")

    # Define case names for display purposes
    case_names = {"a": "easy-axis", "b": "45-degree", "c": "hard-axis"}
    print(
        f"\n[CASES] Running simulations for: {', '.join([f'{c} ({case_names[c]})' for c in cases])}"
    )

    # Dictionary of specific precompute / down / up folders for each case
    precompute_dirs = {s: sensor_loop_dir / f"sensor_case-{s}_precompute" for s in cases}
    down_dirs = {s: sensor_loop_dir / f"sensor_case-{s}_down" for s in cases}
    up_dirs = {s: sensor_loop_dir / f"sensor_case-{s}_up" for s in cases}

    # Symlink mesh file to the initial-state and all case directories and rename it to "sensor.npz"
    # Skip this step if backup mesh will be provided (only for initial_dir)
    if not args.initial_mesh_file:
        print("\n[MESH DISTRIBUTION] Symlinking mesh to all case directories...")
        for d in [initial_dir] + list(precompute_dirs.values()) + list(down_dirs.values()) + list(up_dirs.values()):
            mesh_dst = d / "sensor.npz"
            mesh_src = sensor_loop_dir / mesh_file_name
            if not mesh_src.exists():
                raise FileNotFoundError(f"Mesh file not found: {mesh_src}")
            mesh_dst.parent.mkdir(parents=True, exist_ok=True)
            print(f"  → {d.name}/sensor.npz")
            if mesh_dst.exists() or mesh_dst.is_symlink():
                mesh_dst.unlink()
            mesh_dst.symlink_to(Path("..") / mesh_file_name)
        print("[MESH DISTRIBUTION] ✓ Mesh copied to all directories")
    else:
        print("\n[MESH DISTRIBUTION] Copying mesh to case directories (excluding initial_dir)...")
        # For down and up dirs, we still need to copy from initial_dir after the backup mesh is standardized
        # This will be handled in Step 2 along with the state file
        print("[MESH DISTRIBUTION] Initial directory will use backup mesh; case directories will be populated after Step 1")

    loop_script = (base / "src/loop.py").resolve()

    loop_cmd_in_main: list[str] = [sys.executable, str(loop_script)]
    # Step0.2: remove previous output files like sensor.*.state.npz and sensor.mh in all case directories
    print("\n" + "-" * 80)
    print("SENSOR-EXAMPLE, STEP 0.2: Cleanup Previous Output Files")
    print("-" * 80)
    removed_count = 0
    for s in cases:
        pdir = precompute_dirs[s]
        ddir = down_dirs[s]
        udir = up_dirs[s]

        for d in [pdir, ddir, udir]:
            # Remove sensor.*.state.npz files
            state_files = list(d.glob("sensor.*.state.npz"))
            for f in state_files:
                print(f"  [CLEANUP] Removing {f.name} from {d.name}")
                f.unlink()
                removed_count += 1

            # Remove sensor.mh file
            sensor_mh = d / "sensor.mh"
            if sensor_mh.exists():
                print(f"  [CLEANUP] Removing sensor.mh from {d.name}")
                sensor_mh.unlink()
                removed_count += 1
    print(f"[CLEANUP] ✓ Removed {removed_count} previous output file(s)")

    # Step1: run "python ./../../../src/loop.py --mesh sensor" in subfolder "sensor_initial_state"
    # -> handled directly via run_loop with the initial-state directory
    print("\n" + "=" * 80)
    print("SENSOR-EXAMPLE, STEP 1: Initial Equilibrium Computation")
    print("=" * 80)
    
    if args.load_initial_state:
        print("[SIMULATION] Loading pre-computed initial state (--load-initial-state)...")
        if args.initial_state_file and not args.initial_mesh_file:
            print("[WARNING] ⚠️  MESH COMPATIBILITY CHECK REQUIRED:")
            print("[WARNING]     The mesh from the loaded initial state must EXACTLY match the current simulation mesh.")
            print("[WARNING]     If meshes differ (different resolution, geometry, etc.), the simulation will produce incorrect results.")
            print("[WARNING]     Provide --initial-mesh-file to copy the matching mesh alongside the state.")
        
        # Handle backup mesh file if provided
        if args.initial_mesh_file:
            backup_mesh_path = initial_dir / args.initial_mesh_file
            if backup_mesh_path.exists():
                print(f"  [MESH] Found backup mesh file: {args.initial_mesh_file}")
                print(f"  [MESH] Location searched: {backup_mesh_path}")
                print("[STANDARDIZE] Renaming backup mesh file...")
                standardize_mesh_file_name(initial_dir, backup_mesh_name=args.initial_mesh_file, simulation_name="sensor")
            else:
                print(f"[ERROR] Specified backup mesh file not found: {backup_mesh_path}")
                return 1
        
        # Expect initial state file to already exist in sensor_initial_state directory
        try:
            if args.initial_state_file:
                # Check if the specified file exists (before standardization)
                initial_state_path = initial_dir / args.initial_state_file
                if initial_state_path.exists():
                    print(f"  [STATE] Found specified file: {args.initial_state_file}")
                    # Standardize the file if needed
                    print("[STANDARDIZE] Checking for backup state files...")
                    standardize_state_file_names(initial_dir, backup_name=args.initial_state_file, simulation_name="sensor")
                    # After standardization, determine the actual filename to use
                    # Extract step number from original filename and construct standardized name
                    import re
                    match = re.match(r'^(.+)\.(\d+)\.state\.npz$', args.initial_state_file)
                    if match:
                        step_number = match.group(2)
                        initial_state_name = f"sensor.{step_number}.state.npz"
                        if initial_state_name != args.initial_state_file:
                            print(f"  [STANDARDIZED] {args.initial_state_file} → {initial_state_name}")
                    else:
                        # Couldn't parse; assume it's already correct
                        initial_state_name = args.initial_state_file
                else:
                    # File doesn't exist with the specified name
                    raise FileNotFoundError(f"Specified initial state file not found: {initial_state_path}")
            else:
                # No specific file requested; standardize any backup files and find the latest
                print("[STANDARDIZE] Checking for backup state files...")
                standardize_state_file_names(initial_dir, backup_name=None, simulation_name="sensor")
                initial_state_name = find_last_state_file(initial_dir)
            print(f"[RESULT] ✓ Loaded initial state: {initial_state_name}")
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            # If a specific file was requested, suggest checking the directory
            if args.initial_state_file:
                print("[SUGGESTION] Check that the file exists in the sensor_initial_state directory.")
            else:
                print("[ERROR] No pre-computed initial state found. Run without --load-initial-state to compute it.")
            return 1
    else:
        print("[SIMULATION] Computing initial magnetization state...")
        run_loop(loop_cmd_in_main, initial_dir)
        # Find the last state file from initial computation
        initial_state_name = find_last_state_file(initial_dir)
        print(f"[RESULT] ✓ Initial state saved as: {initial_state_name}")

    # Optional early exit: only compute or load the initial state
    if args.only_compute_initial_state:
        print("\n" + "=" * 80)
        print("ONLY-COMPUTE-INITIAL-STATE MODE")
        print("=" * 80)
        print(f"[INFO] Exiting after initial state. Directory: {initial_dir}")
        return 0

    # Step2: copy the last state file containing the information of the computed equilibrium
    # from "sensor_initial_state" to
    # "sensor_case-a_down" and
    # "sensor_case-b_down" and
    # "sensor_case-c_down"
    # -> `copy_state()` - generic function to copy state files between any directories
    print("\n" + "=" * 80)
    print("SENSOR-EXAMPLE, STEP 2: Distribute Initial State to Down-Cases")
    print("=" * 80)
    
    # If backup mesh was used, also copy the mesh from initial_dir to case directories
    if args.initial_mesh_file:
        print("[MESH DISTRIBUTION] Symlinking standardized mesh from initial_dir to case directories...")
        mesh_src = initial_dir / "sensor.npz"
        if not mesh_src.exists():
            print(f"[ERROR] Standardized mesh not found in initial_dir: {mesh_src}")
            return 1
        for d in list(down_dirs.values()) + list(up_dirs.values()):
            mesh_dst = d / "sensor.npz"
            mesh_dst.parent.mkdir(parents=True, exist_ok=True)
            print(f"  → {d.name}/sensor.npz")
            if mesh_dst.exists() or mesh_dst.is_symlink():
                mesh_dst.unlink()
            mesh_dst.symlink_to(Path("..") / "sensor_initial_state" / "sensor.npz")
        print("[MESH DISTRIBUTION] ✓ Mesh symlinked to all case directories")
    
    import re
    initial_ini_match = re.search(r"state_cfg(\d+)", initial_state_name)
    initial_ini = initial_ini_match.group(1) if initial_ini_match else "0"
    for s, pdir in precompute_dirs.items():
        print(f"[COPY] {initial_state_name} → case precompute-{s} ({case_names[s]})")
        copy_state(initial_dir, initial_state_name, pdir)
        set_p2_params(pdir, {"ini": initial_ini})
    print(f"[COPY] ✓ Initial state distributed to {len(cases)} precompute-case(s)")

    # Steps 3-11: run precompute, copy to down, run down, copy to up, run up for each case
    for idx, s in enumerate(cases, 1):
        pdir = precompute_dirs[s]
        ddir = down_dirs[s]
        udir = up_dirs[s]

        # Phase 1: Precompute sweep
        print("\n" + "=" * 80)
        print(
            f"SENSOR-EXAMPLE, PHASE 1: PRECOMPUTE - Case {s.upper()} ({case_names[s]})"
        )
        print("=" * 80)
        print("[SIMULATION] Running zero-to-saturating field sweep...")
        run_loop(loop_cmd_in_main, pdir)

        precompute_result_state = find_last_state_file(pdir)
        print(f"[RESULT] ✓ Precompute completed: {precompute_result_state}")

        # Transfer from precompute to down
        print("\n" + "-" * 80)
        print(f"SENSOR-EXAMPLE, TRANSFER STATE - Precompute → Down")
        print("-" * 80)
        print(f"[COPY] {precompute_result_state} → case down-{s}")
        copy_state(pdir, precompute_result_state, ddir)
        print("[COPY] ✓ State transferred to down-case")

        import re
        expected_ini_match = re.search(r"state_cfg(\d+)", precompute_result_state)
        expected_ini_value = expected_ini_match.group(1) if expected_ini_match else "0"
        print(f"  [UPDATE] Setting .p2 params: ini = {expected_ini_value}")
        set_p2_params(ddir, {"ini": expected_ini_value})

        # Phase 2: Down sweep
        print("\n" + "=" * 80)
        print(
            f"SENSOR-EXAMPLE, PHASE 2: DOWN-SWEEP - Case {s.upper()} ({case_names[s]})"
        )
        print("=" * 80)
        print("[SIMULATION] Running decreasing field sweep...")
        run_loop(loop_cmd_in_main, ddir)

        down_result_state = find_last_state_file(ddir)
        print(f"[RESULT] ✓ Down-sweep completed: {down_result_state}")

        # Transfer from down to up
        print("\n" + "-" * 80)
        print(f"SENSOR-EXAMPLE, TRANSFER STATE - Down → Up")
        print("-" * 80)
        print(f"[COPY] {down_result_state} → case up-{s}")
        copy_state(ddir, down_result_state, udir)
        print("[COPY] ✓ State transferred to up-case")

        expected_ini_match = re.search(r"state_cfg(\d+)", down_result_state)
        expected_ini_value = expected_ini_match.group(1) if expected_ini_match else "0"
        expected_hstep = str(args.hstep) if args.hstep is not None else "0.0005"
        print(
            f"  [UPDATE] Setting .p2 params: ini = {expected_ini_value}, hstep = {expected_hstep}"
        )
        set_p2_params(udir, {"ini": expected_ini_value, "hstep": expected_hstep})

        # Phase 3: Up sweep
        print("\n" + "=" * 80)
        print(
            f"SENSOR-EXAMPLE, PHASE 3: UP-SWEEP - Case {s.upper()} ({case_names[s]})"
        )
        print("=" * 80)
        print("[SIMULATION] Running increasing field sweep...")
        run_loop(loop_cmd_in_main, udir)
        print(f"[RESULT] ✓ Up-sweep completed for case {s}")

    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(
        f"[SUMMARY] Processed {len(cases)} case(s): {', '.join([f'{c} ({case_names[c]})' for c in cases])}"
    )
    print("[OUTPUT] Results saved in respective case directories")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
