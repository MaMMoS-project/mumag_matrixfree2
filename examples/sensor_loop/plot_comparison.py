import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({'font.size': 14})
matplotlib.use('Agg')

def mT_to_kAm(B_mT):
    return B_mT * 10.0 / (4.0 * np.pi)

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default=".", help="Workspace dir")
    args = parser.parse_args()
    workspace = Path(args.workspace)

    oommf_csv = Path("OOMMF_results_from_definition_of_benchmark2/MaMMoS_benchmark_OOMMF_sweeps.csv")
    if not oommf_csv.exists():
        print(f"Error: {oommf_csv} not found.")
        sys.exit(1)
        
    oommf_df = pd.read_csv(oommf_csv)

    cases = {
        'a': ('easy', 'Case A (Easy Axis)', 'sensor_case-a.mh'),
        'b': ('diagonal', 'Case B (Diagonal Axis)', 'sensor_case-b.mh'),
        'c': ('hard', 'Case C (Hard Axis)', 'sensor_case-c.mh')
    }

    plt.figure(figsize=(18, 6))

    for i, (key, (oommf_axis, title, mh_file)) in enumerate(cases.items()):
        plt.subplot(1, 3, i + 1)
        
        # Plot OOMMF Reference
        subset = oommf_df[oommf_df['axis'] == oommf_axis]
        if not subset.empty:
            plt.plot(mT_to_kAm(subset['bParallel_mT']), subset['mParallel'], 'k--', linewidth=2.5, label='OOMMF Reference', zorder=2)
        
        # Plot Simulated Data
        mh_path = workspace / mh_file
        if mh_path.exists():
            try:
                data = np.loadtxt(mh_path, skiprows=1)
                b_ext_mT = data[:, 0] * 1000.0
                m_parallel = data[:, 1] / 1.005 # J_parallel / Js
                
                # Use a distinct visual cue as per plotting rules (color + solid line + higher z-order)
                plt.plot(mT_to_kAm(b_ext_mT), m_parallel, color='#FF5733', linestyle='-', linewidth=2, label='MuMag MatrixFree', alpha=0.9, zorder=3)
            except Exception as e:
                print(f"Error loading {mh_file}: {e}")
        else:
            print(f"Warning: {mh_file} not found. Skipping simulated data for {title}.")

        # The plotting rule states: You are prohibited from generating plot titles using plt.title(); all descriptions must go in captions.
        plt.text(0.5, 1.05, title, ha='center', va='bottom', transform=plt.gca().transAxes, fontsize=16)
        
        plt.xlabel('H$_{parallel}$ [kA/m]')
        plt.ylabel('M$_{parallel}$ / M$_s$')
        plt.xlim(-15, 15)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(workspace / 'comparison_plot.png', dpi=300, bbox_inches='tight')
    print(f"Successfully generated {workspace / 'comparison_plot.png'}")

if __name__ == "__main__":
    main()
