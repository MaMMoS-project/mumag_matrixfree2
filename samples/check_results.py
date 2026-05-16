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
        
        # Identify switching point: index where dJ/dB is maximum (coercive field region)
        # In this sample, field goes from 2.0 to negative.
        diff_j = np.abs(np.diff(d1[:, 1]))
        switch_idx = np.argmax(diff_j) + 1  # The step where the jump happened
        
        success = True
        for i, col in enumerate(cols):
            diff = np.abs(d1[:, col] - d2[:, col])
            
            # Use different tolerances based on switching
            for idx in range(len(d1)):
                is_switch = (idx == switch_idx)
                
                if col == 0: # B_ext
                    val = diff[idx]
                    tol = 1e-12
                elif col == 5: # Energy
                    # Relative check for Energy: |d1-d2| / (|d1|+1)
                    denom = np.abs(d1[idx, col]) + 1.0
                    val = diff[idx] / denom
                    tol = 1e-3 if is_switch else 1e-7
                else: # J_par
                    val = diff[idx]
                    tol = 1e-3 if is_switch else 1e-6
                
                if val > tol:
                    loc = "AT SWITCH" if is_switch else f"at step {idx}"
                    print(f"FAILED: {n1} vs {n2} | {labels[i]} {loc} | Diff: {val:.2e} | Tol: {tol:.2e}")
                    success = False
                    break # Only report first failure per column
                    
        return success

    print("\n--- Comparison: Standard vs Magnetoelastic ---")
    c1 = compare("std_demag", "me_demag")
    c2 = compare("std_no_demag", "me_no_demag")
    
    if c1 and c2:
        print("SUCCESS: Magnetoelastic runs match Standard runs (within switching tolerances).")
    else:
        print("FAILURE: Inconsistency detected between Standard and ME runs.")
        sys.exit(1)

    print("\n--- Physical Check: Demag vs No-Demag ---")
    # Switching field is the B_ext at the last step
    b_sw_demag = results["std_demag"][-1, 0]
    b_sw_nodemag = results["std_no_demag"][-1, 0]
    
    print(f"Switching Field (Demag):    {b_sw_demag:.2f} T")
    print(f"Switching Field (No-Demag): {b_sw_nodemag:.2f} T")
    
    if b_sw_nodemag < b_sw_demag:
        print("SUCCESS: No-Demag switches later (more negative field) than Demag.")
    else:
        print("FAILURE: Switching field logic is incorrect.")
        sys.exit(1)

    print("\n=== All Consistency Checks Passed ===")

if __name__ == "__main__":
    check_consistency()
