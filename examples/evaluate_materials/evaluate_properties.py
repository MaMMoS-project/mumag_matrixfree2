"""Wrapper script to evaluate intrinsic properties."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """CLI entry point to evaluate material properties."""
    parser = argparse.ArgumentParser(description="Evaluate intrinsic properties (wrapper script).")
    parser.add_argument("--K1", type=float, required=True, help="Anisotropy constant K1 [J/m^3]")
    parser.add_argument("--Js", type=float, required=True, help="Saturation polarization Js [T]")
    parser.add_argument("--A", type=float, required=True, help="Exchange constant A [J/m]")
    parser.add_argument("--hstart", type=float, default=2.0, help="Start field [T] (default: 2.0)")
    parser.add_argument("--hfinal", type=float, default=-2.0, help="Final field [T] (default: -2.0)")
    parser.add_argument("--hstep", type=float, default=0.01, help="Field step [T] (default: 0.01)")
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parent
    compute_script = run_dir / "compute_evaluations.py"
    analyze_script = run_dir / "analyze_evaluations.py"

    if not compute_script.exists():
        print(f"Error: {compute_script} not found.")
        sys.exit(1)
    if not analyze_script.exists():
        print(f"Error: {analyze_script} not found.")
        sys.exit(1)

    cmd_args = [
        "--K1",
        str(args.K1),
        "--Js",
        str(args.Js),
        "--A",
        str(args.A),
        "--hstart",
        str(args.hstart),
        "--hfinal",
        str(args.hfinal),
        "--hstep",
        str(args.hstep),
    ]

    print("=== Step 1: Computing Evaluations ===")
    compute_cmd = [sys.executable, str(compute_script)] + cmd_args
    subprocess.run(compute_cmd, check=True)

    print("\n=== Step 2: Analyzing Results ===")
    analyze_cmd = [sys.executable, str(analyze_script)] + cmd_args
    subprocess.run(analyze_cmd, check=True)

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
