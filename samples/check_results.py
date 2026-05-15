import numpy as np
import sys
from pathlib import Path

def load_mh(path):
    # Load .mh file skipping lines starting with #
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            data.append([float(x) for x in line.split()])
    return np.array(data)

def check_consistency():
    base = Path(__file__).parent
    files = {
        "std_demag": base / "bench_demag/cube_20nm.mh",
        "std_no_demag": base / "bench_no_demag/cube_20nm.mh",
        "me_demag": base / "bench_me_demag/cube_20nm.mh",
        "me_no_demag": base / "bench_me_no_demag/cube_20nm.mh"
    }
    
    # 1. Load data
    results = {}
    for name, path in files.items():
        if not path.exists():
            print(f"Error: {path} not found.")
            sys.exit(1)
        results[name] = load_mh(path)
        print(f"Loaded {name}: {len(results[name])} steps.")

    def compare(n1, n2):
        d1, d2 = results[n1], results[n2]
        if len(d1) != len(d2):
            print(f"FAILED: {n1} and {n2} have different number of steps ({len(d1)} vs {len(d2)})")
            return False
        
        # Check B_ext, J_par, and E (columns 0, 1, 5)
        cols = [0, 1, 5]
        labels = ["B_ext", "J_par", "Energy"]
        
        success = True
        for i, col in enumerate(cols):
            diff = np.abs(d1[:, col] - d2[:, col])
            if col == 5:
                # Relative check for Energy: |d1-d2| / (|d1|+1)
                denom = np.abs(d1[:, col]) + 1.0
                max_val = np.max(diff / denom)
                tol = 1e-7
            else:
                max_val = np.max(diff)
                tol = 1e-6
                
            if max_val > tol:
                print(f"FAILED: {n1} vs {n2} | Max difference in {labels[i]}: {max_val:.2e}")
                success = False
        return success

    print("\n--- Comparison: Standard vs Magnetoelastic ---")
    c1 = compare("std_demag", "me_demag")
    c2 = compare("std_no_demag", "me_no_demag")
    
    if c1 and c2:
        print("SUCCESS: Magnetoelastic runs match Standard runs exactly.")
    else:
        print("FAILURE: Inconsistency detected between Standard and ME runs.")
        sys.exit(1)

    print("\n--- Physical Check: Demag vs No-Demag ---")
    # Switching field is the B_ext at the last step
    b_sw_demag = results["std_demag"][-1, 0]
    b_sw_nodemag = results["std_no_demag"][-1, 0]
    
    print(f"Switching Field (Demag):    {b_sw_demag:.2f} T")
    print(f"Switching Field (No-Demag): {b_sw_nodemag:.2f} T")
    
    # In this sample, field starts at +2.0 and goes negative. 
    # Demag should help reversal -> switch at less negative field (e.g. -6.0 vs -6.5)
    if b_sw_nodemag < b_sw_demag:
        print("SUCCESS: No-Demag switches later (more negative field) than Demag.")
    else:
        print("FAILURE: Switching field logic is incorrect.")
        sys.exit(1)

    print("\n=== All Consistency Checks Passed ===")

if __name__ == "__main__":
    check_consistency()
