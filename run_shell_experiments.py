import sys
import os
import numpy as np

# Append project src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import add_shell

def compute_tet_quality(knt, tets):
    """
    Computes the normalized volume-to-edge ratio quality metric eta for each tetrahedron.
    eta = (6 * sqrt(2) * Vol) / (rms_edge_length ** 3)
    Ranges from 0.0 (degenerate/flat) to 1.0 (regular tetrahedron).
    """
    pts = knt[tets] # (E, 4, 3)
    a = pts[:, 0]
    b = pts[:, 1]
    c = pts[:, 2]
    d = pts[:, 3]

    ab = b - a
    ac = c - a
    ad = d - a
    bc = c - b
    bd = d - b
    cd = d - c

    # Volume = 1/6 * |ab . (ac x ad)|
    vols6 = np.abs(np.einsum('ij,ij->i', np.cross(ab, ac), ad))
    vols = vols6 / 6.0

    # Sum of square edge lengths
    l_ab = np.einsum('ij,ij->i', ab, ab)
    l_ac = np.einsum('ij,ij->i', ac, ac)
    l_ad = np.einsum('ij,ij->i', ad, ad)
    l_bc = np.einsum('ij,ij->i', bc, bc)
    l_bd = np.einsum('ij,ij->i', bd, bd)
    l_cd = np.einsum('ij,ij->i', cd, cd)

    S = l_ab + l_ac + l_ad + l_bc + l_bd + l_cd
    rms = np.sqrt(np.maximum(S, 1e-12) / 6.0)

    eta = (6.0 * np.sqrt(2.0) * vols) / (rms ** 3)
    # Clip to [0, 1] due to small numerical issues
    return np.clip(eta, 0.0, 1.0)

def run_experiment(name, in_mesh, config):
    print(f"\n==================================================")
    print(f"Running Experiment: {name}")
    print(f"Config: {config}")
    print(f"==================================================")
    
    # Run the shell addition pipeline in-memory
    knt_merged, ijk_merged = add_shell.run_add_shell_pipeline(
        in_npz=in_mesh,
        **config
    )
    
    # Identify core vs shell elements
    # Core elements are those with mat_id = 1 (and 2, 3 ... for poly, gb)
    # In run_add_shell_pipeline, the shell material is: shell_mat = body_mat + 1
    # Let's find the maximum material ID to identify the shell elements
    max_mat = int(ijk_merged[:, 4].max())
    shell_mask = ijk_merged[:, 4] == max_mat
    
    tets_shell = ijk_merged[shell_mask, :4].astype(np.int64)
    tets_core = ijk_merged[~shell_mask, :4].astype(np.int64)
    
    n_shell_elements = tets_shell.shape[0]
    n_core_elements = tets_core.shape[0]
    
    print(f"Core Elements: {n_core_elements}")
    print(f"Shell Elements: {n_shell_elements}")
    print(f"Total Elements: {ijk_merged.shape[0]}")
    print(f"Total Nodes: {knt_merged.shape[0]}")
    
    # Bounding box of the outer shell nodes
    # Let's find nodes used by the shell
    shell_nids = np.unique(tets_shell)
    shell_pts = knt_merged[shell_nids]
    
    mins = np.min(shell_pts, axis=0)
    maxs = np.max(shell_pts, axis=0)
    extents = maxs - mins
    vol_airbox = float(np.prod(extents))
    
    print(f"Airbox Bounds Min: {mins}")
    print(f"Airbox Bounds Max: {maxs}")
    print(f"Airbox Extents: {extents}")
    print(f"Airbox Volume: {vol_airbox:.6g}")
    
    # Quality metrics for shell elements
    if n_shell_elements > 0:
        q_shell = compute_tet_quality(knt_merged, tets_shell)
        q_mean = float(np.mean(q_shell))
        q_min = float(np.min(q_shell))
        print(f"Shell Tet Quality (eta): Mean = {q_mean:.4f}, Min = {q_min:.4f}")
    else:
        q_mean, q_min = 0.0, 0.0
        print("No shell elements generated.")
        
    return {
        "name": name,
        "n_shell": n_shell_elements,
        "n_nodes": knt_merged.shape[0],
        "extents": extents.tolist(),
        "vol": vol_airbox,
        "q_mean": q_mean,
        "q_min": q_min
    }

def main():
    in_mesh = os.path.abspath('poly_gb_10.npz')
    if not os.path.exists(in_mesh):
        print(f"Error: {in_mesh} does not exist. Please generate the core mesh first.")
        sys.exit(1)
        
    configs = {
        "Exp 1: Triangles (Default size)": {
            "shell_type": "triangles",
            "layers": 4,
            "K": 1.3,
            "hmax": None,
            "verbose": False
        },
        "Exp 2: Convex Hull (Default size)": {
            "shell_type": "hull",
            "layers": 4,
            "K": 1.3,
            "hmax": None,
            "verbose": False
        },
        "Exp 3: Convex Hull (hmax=5.0)": {
            "shell_type": "hull",
            "layers": 4,
            "K": 1.3,
            "hmax": 5.0,
            "verbose": False
        },
        "Exp 4: Convex Hull (K=1.5, layers=3, hmax=6.0)": {
            "shell_type": "hull",
            "layers": 3,
            "K": 1.5,
            "hmax": 6.0,
            "verbose": False
        }
    }
    
    results = []
    for name, cfg in configs.items():
        try:
            res = run_experiment(name, in_mesh, cfg)
            results.append(res)
        except Exception as e:
            print(f"Experiment {name} failed: {e}")
            
    # Print summary table
    print("\n\n" + "="*80)
    print(f"{'Experiment Name':<45} | {'Shell Tets':<10} | {'Airbox Vol':<12} | {'Mean Q':<6} | {'Min Q':<6}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<45} | {r['n_shell']:<10d} | {r['vol']:<12.1e} | {r['q_mean']:<6.4f} | {r['q_min']:<6.4f}")
    print("="*80)

if __name__ == '__main__':
    main()
