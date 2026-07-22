"""Generate base granular structures for evaluation."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    """CLI entry point to generate granular structures."""
    parser = argparse.ArgumentParser(description="Generate base granular structures.")
    parser.add_argument("--extent", default="80,80,80", help="Mesh extent (default: 80,80,80)")
    parser.add_argument("--grains", type=int, default=8, help="Number of grains (default: 8)")
    parser.add_argument("--num-structures", type=int, default=10, help="Number of structures (default: 10)")
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parent
    base_dir = run_dir.parent.parent.resolve()
    base_structures_dir = run_dir / "base_structures"
    base_structures_dir.mkdir(parents=True, exist_ok=True)

    mesh_script = base_dir / "src" / "mesh.py"
    make_krn_script = base_dir / "src" / "make_krn.py"

    # Save metadata so evaluate_properties.py can record it in CSVs
    with open(base_structures_dir / "metadata.txt", "w") as f:
        f.write(f"extent={args.extent}\n")
        f.write(f"grains={args.grains}\n")

    print(f"Generating {args.num_structures} base structures in {base_structures_dir}...")
    for i in range(1, args.num_structures + 1):
        struct_name = f"structure_{i:02d}"
        struct_dir = base_structures_dir / struct_name
        struct_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Processing {struct_name} ---")

        # 1. Generate Mesh
        mesh_cmd = [
            sys.executable,
            str(mesh_script),
            "--geom",
            "poly",
            "--n",
            str(args.grains),
            "--id",
            str(123 + i),  # Vary seed so each structure is unique
            "--extent",
            args.extent,
        ]
        print("Running mesh generation...")
        subprocess.run(mesh_cmd, cwd=struct_dir, check=True)
        shutil.move(struct_dir / "single_solid.npz", struct_dir / "isotrop.npz")

        # 2. Generate KRN with defaults just to set the fixed easy axes
        krn_cmd = [
            sys.executable,
            str(make_krn_script),
            "--mesh",
            "isotrop.npz",
            "--out",
            "isotrop.krn",
            "--tol",
            "0.05",  # Loose tol to avoid infinite loops with random seeds
            "--seed",
            str(1000 + i),
        ]
        print("Running easy-axis generation...")
        subprocess.run(krn_cmd, cwd=struct_dir, check=True)

    print("\n✓ Base structures generated successfully.")


if __name__ == "__main__":
    main()
