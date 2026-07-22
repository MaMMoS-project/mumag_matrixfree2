import argparse
import subprocess
import sys
import csv
import numpy as np
from pathlib import Path
import os

try:
    import mammos_analysis
    import mammos_entity as me
    import mammos_units as u
    _MAMMOS_AVAILABLE = True
except ImportError:
    _MAMMOS_AVAILABLE = False

def compute_properties_from_arrays(Hext_T, J_T, demag=1.0/3.0):
    if not _MAMMOS_AVAILABLE:
        return None, None
    try:
        mu0 = 4 * np.pi * 1e-7
        H_A_per_m = Hext_T / mu0
        M_A_per_m = J_T / mu0
        H_entity = me.H(list(H_A_per_m))
        M_entity = me.M(list(M_A_per_m))
        extrinsic = mammos_analysis.hysteresis.extrinsic_properties(
            H=H_entity, M=M_entity, demagnetization_coefficient=demag
        )
        
        hc_val = None
        try: hc_val = float(extrinsic.Hc.q.to_value(u.A / u.m))
        except:
            try: hc_val = float(extrinsic.Hc.q.m)
            except: pass
        
        jr_val = None
        for attr in ["Mr", "Br", "Jr", "M_r", "B_r", "J_r", "remanence", "remanent_magnetization"]:
            if hasattr(extrinsic, attr):
                jr_attr = getattr(extrinsic, attr)
                try:
                    jr_val = float(jr_attr.q.to_value(u.A / u.m)) * mu0
                    break
                except:
                    try:
                        jr_val = float(jr_attr.q.m) * mu0
                        break
                    except: pass

        return hc_val, jr_val
    except Exception as e:
        print(f"Error computing properties: {e}")
        return None, None

def write_p2(path, hstart, hfinal, hstep):
    content = f"""[Mesh]
format = "npz"

[Material]
exchange = "A"
anisotropy = "K1"
magnetization = "Js"
krn_file = "isotrop.krn"

[Initial]
mx = 0.0
my = 0.0
mz = 1.0

[Field]
direction = [0, 0, 1]
hstart = {hstart}
hfinal = {hfinal}
hstep = {hstep}

[Minimizer]
tol_fun = 1e-10
tol_hmag_factor = 1.0
"""
    with open(path, "w") as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(description="Evaluate intrinsic properties.")
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

    indiv_csv = evaluations_dir / "evaluation_results_individual.csv"
    avg_csv = evaluations_dir / "evaluation_results_average.csv"

    # Initialize CSV files
    if not indiv_csv.exists():
        with open(indiv_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Structure", "Extent", "Grains", "K1", "Js", "A", "Hc [A/m]", "Jr [T]"])
    if not avg_csv.exists():
        with open(avg_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Extent", "Grains", "K1", "Js", "A", "Hc_avg [A/m]", "Jr_avg [T]"])

    all_data = []

    extent_str = "unknown"
    grains_val = "unknown"
    metadata_file = base_structures_dir / "metadata.txt"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            for line in f:
                if line.startswith("extent="): extent_str = line.split("=")[1].strip()
                if line.startswith("grains="): grains_val = line.split("=")[1].strip()

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
        with open(krn_src, "r") as f_in, open(krn_dst, "w") as f_out:
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
        loop_cmd = [sys.executable, str(loop_script), "isotrop", "--mesh", "isotrop.npz"]
        subprocess.run(loop_cmd, cwd=run_struct_dir, check=True)

        # Process output
        mh_file = run_struct_dir / "hyst_isotrop" / "isotrop.mh"
        if mh_file.exists():
            data = np.loadtxt(mh_file, skiprows=1)
            all_data.append(data)
            
            hc, jr = compute_properties_from_arrays(data[:,0], data[:,1])
            with open(indiv_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([struct_name, extent_str, grains_val, args.K1, args.Js, args.A, hc, jr])
        else:
            print(f"Warning: {mh_file} not found.")

    if all_data:
        # Check shapes and compute average
        min_len = min(d.shape[0] for d in all_data)
        truncated_data = [d[:min_len, :] for d in all_data]
        data_stack = np.stack(truncated_data, axis=0)
        data_avg = np.mean(data_stack, axis=0)
        
        hc_avg, jr_avg = compute_properties_from_arrays(data_avg[:,0], data_avg[:,1])
        with open(avg_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([extent_str, grains_val, args.K1, args.Js, args.A, hc_avg, jr_avg])
            
        print(f"\n✓ Completed evaluation for {eval_name}")
        if hc_avg is not None:
            print(f"Average Hc: {hc_avg:.4e} A/m")
        if jr_avg is not None:
            print(f"Average Jr: {jr_avg:.4e} T")
    else:
        print("No data collected for averaging.")

if __name__ == "__main__":
    main()
