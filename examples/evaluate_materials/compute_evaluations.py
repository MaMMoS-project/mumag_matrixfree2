import argparse
import os
import subprocess
import sys
from pathlib import Path


def write_p2(path, hstart, hfinal, hstep):
    content = f"""[mesh]
format = "npz"

[material]
exchange = "A"
anisotropy = "K1"
magnetization = "Js"
krn_file = "isotrop.krn"

[initial state]
mx = 0.0
my = 0.0
mz = 1.0

[field]
direction = [0, 0, 1]
hstart = {hstart}
hfinal = {hfinal}
hstep = {hstep}
mstep = 10000

[minimizer]
method = pcohen_hs
tol_fun = 1e-10
eps_a = 1e-12
"""
    with open(path, "w") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Compute intrinsic properties evaluations.")
    parser.add_argument("--K1", type=float, required=True, help="Anisotropy constant K1 [J/m^3]")
    parser.add_argument("--Js", type=float, required=True, help="Saturation polarization Js [T]")
    parser.add_argument("--A", type=float, required=True, help="Exchange constant A [J/m]")
    parser.add_argument("--hstart", type=float, default=2.0, help="Start field [T] (default: 2.0)")
    parser.add_argument("--hfinal", type=float, default=-2.0, help="Final field [T] (default: -2.0)")
    parser.add_argument("--hstep", type=float, default=0.01, help="Field step [T] (default: 0.01)")
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parent
    base_dir = run_dir.parent.parent.resolve()
    base_structures_dir = run_dir / "base_structures"
    evaluations_dir = run_dir / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)

    loop_script = base_dir / "src" / "loop.py"

    struct_dirs = sorted(base_structures_dir.glob("structure_*"))
    if not struct_dirs:
        print("No base structures found. Run generate_structures.py first.")
        sys.exit(1)

    eval_name = f"eval_K1_{args.K1:g}_Js_{args.Js:g}_A_{args.A:g}"
    eval_run_dir = evaluations_dir / eval_name
    eval_run_dir.mkdir(parents=True, exist_ok=True)

    for struct_dir in struct_dirs:
        struct_name = struct_dir.name
        run_struct_dir = eval_run_dir / struct_name
        run_struct_dir.mkdir(parents=True, exist_ok=True)

        # Symlink mesh
        mesh_src = struct_dir / "isotrop.npz"
        mesh_dst = run_struct_dir / "isotrop.npz"
        if not mesh_dst.exists():
            os.symlink(mesh_src, mesh_dst)

        # Read and modify krn (replacing intrinsic properties but keeping theta/phi)
        krn_src = struct_dir / "isotrop.krn"
        krn_dst = run_struct_dir / "isotrop.krn"
        with open(krn_src) as f_in, open(krn_dst, "w") as f_out:
            for line in f_in:
                if line.startswith("#"):
                    f_out.write(line)
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    parts[2] = f"{args.K1:.6e}"
                    parts[4] = f"{args.Js:.6e}"
                    parts[5] = f"{args.A:.6e}"
                    f_out.write(" ".join(parts) + "\n")

        # Write p2
        p2_dst = run_struct_dir / "isotrop.p2"
        write_p2(p2_dst, args.hstart, args.hfinal, args.hstep)

        # Run loop.py
        print(f"\n--- Running loop.py for {struct_name} ---")
        loop_cmd = [sys.executable, str(loop_script), "isotrop", "--mesh", "isotrop.npz", "--add-shell"]
        subprocess.run(loop_cmd, cwd=run_struct_dir, check=True)


if __name__ == "__main__":
    main()
