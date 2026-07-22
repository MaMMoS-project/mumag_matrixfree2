import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import mammos_entity as me
    import mammos_units as u

    _MAMMOS_AVAILABLE = True
except ImportError:
    _MAMMOS_AVAILABLE = False


def find_crossing(x, y, target_y=0.0):
    """Finds x where y crosses target_y by linear interpolation."""
    idx = np.argsort(y)
    x_sorted = x[idx]
    y_sorted = y[idx]
    return np.interp(target_y, y_sorted, x_sorted)


def compute_coercivity(H, M):
    cross_H = find_crossing(H, M, 0.0)
    mask = M >= 0
    if np.any(mask):
        H_left = np.min(H[mask])
        idx_left = np.argmin(H[mask])
        M_left = M[mask][idx_left]
    else:
        H_left = cross_H
        M_left = 0.0

    Hc = max(abs(cross_H), abs(H_left))
    if abs(cross_H) >= abs(H_left):
        H_left = cross_H
        M_left = 0.0

    return Hc, H_left, M_left


class StringEntity(list):
    def __init__(self, vals, label):
        super().__init__(vals)
        self.ontology_label = label
        try:
            from mammos_entity import Entity

            dummy = Entity(label, 1.0)
            self.ontology_iri = getattr(dummy, "ontology_iri", "")
        except Exception:
            self.ontology_iri = ""
        self.description = ""
        self.unit = ""


def get_all_properties(B_ext, J, Neff=1.0 / 3.0):
    mu0 = 4 * np.pi * 1e-7
    Jr_ext = find_crossing(J, B_ext, 0.0)
    Bc_ext, _, _ = compute_coercivity(B_ext, J)

    B_int = B_ext - Neff * J
    Jr_int = find_crossing(J, B_int, 0.0)
    Bc_int, _, _ = compute_coercivity(B_int, J)

    H_int = B_int / mu0
    M = J / mu0
    Mr_int = find_crossing(M, H_int, 0.0)
    Hc_int, _, _ = compute_coercivity(H_int, M)

    return Bc_ext, Jr_ext, Bc_int, Jr_int, Hc_int, Mr_int


def plot_demagnetization_curves(all_data, data_avg, props, out_dir, eval_name, Neff=1.0 / 3.0):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update(
        {"font.size": 14, "axes.labelsize": 16, "legend.fontsize": 12, "xtick.labelsize": 14, "ytick.labelsize": 14}
    )
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    mu0 = 4 * np.pi * 1e-7
    Bc_ext, Jr_ext, Bc_int, Jr_int, Hc_int, Mr_int = props

    for data in all_data:
        B_ext = data[:, 0]
        J = data[:, 1]
        axs[0].plot(B_ext, J, color="gray", linestyle="--", alpha=0.3)

    B_ext_avg = data_avg[:, 0]
    J_avg = data_avg[:, 1]
    B_int_avg = B_ext_avg - Neff * J_avg
    H_int_avg = B_int_avg / mu0
    M_avg = J_avg / mu0

    # Calculate coordinates for dashed lines
    _, B_ext_left, J_ext_left = compute_coercivity(B_ext_avg, J_avg)
    _, B_int_left, J_int_left = compute_coercivity(B_int_avg, J_avg)
    _, H_int_left, M_int_left = compute_coercivity(H_int_avg, M_avg)

    axs[0].plot(B_ext_avg, J_avg, color="black", linewidth=3, label="Average")
    axs[0].plot(-Bc_ext, 0, "rs", markersize=8, label=f"mu0 Hc = {Bc_ext:.3f} T")
    if J_ext_left > 0:
        axs[0].plot([-Bc_ext, B_ext_left], [0, J_ext_left], "r--")
    axs[0].plot(0, Jr_ext, "bo", markersize=8, label=f"Jr = {Jr_ext:.3f} T")
    axs[0].set_xlabel(r"$\mu_0 H_{ext}$ (T)")
    axs[0].set_ylabel("Polarization $J$ (T)")
    axs[0].grid(True, linestyle=":", alpha=0.7)
    axs[0].legend(loc="best")

    axs[1].plot(B_int_avg, J_avg, color="black", linewidth=3, label="Average")
    axs[1].plot(-Bc_int, 0, "rs", markersize=8, label=f"mu0 Hc = {Bc_int:.3f} T")
    if J_int_left > 0:
        axs[1].plot([-Bc_int, B_int_left], [0, J_int_left], "r--")
    axs[1].plot(0, Jr_int, "bo", markersize=8, label=f"Jr = {Jr_int:.3f} T")
    axs[1].set_xlabel(r"$\mu_0 H_{int}$ (T)")
    axs[1].set_ylabel("Polarization $J$ (T)")
    axs[1].grid(True, linestyle=":", alpha=0.7)
    axs[1].legend(loc="best")

    axs[2].plot(H_int_avg, M_avg, color="black", linewidth=3, label="Average")
    axs[2].plot(-Hc_int, 0, "rs", markersize=8, label=f"Hc = {Hc_int:.1e} A/m")
    if M_int_left > 0:
        axs[2].plot([-Hc_int, H_int_left], [0, M_int_left], "r--")
    axs[2].plot(0, Mr_int, "bo", markersize=8, label=f"Mr = {Mr_int:.1e} A/m")
    axs[2].set_xlabel(r"$H_{int}$ (A/m)")
    axs[2].set_ylabel("Magnetization $M$ (A/m)")
    axs[2].grid(True, linestyle=":", alpha=0.7)
    axs[2].legend(loc="best")

    plt.tight_layout()
    plot_path = out_dir / f"{eval_name}_demag_curves.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze intrinsic properties evaluations.")
    parser.add_argument("--K1", type=float, required=True, help="Anisotropy constant K1 [J/m^3]")
    parser.add_argument("--Js", type=float, required=True, help="Saturation polarization Js [T]")
    parser.add_argument("--A", type=float, required=True, help="Exchange constant A [J/m]")
    # These args are accepted for consistency but not strictly required for analysis,
    # except to reconstruct eval_name perfectly if we wanted to (though eval_name only uses K1, Js, A).
    parser.add_argument("--hstart", type=float, default=2.0)
    parser.add_argument("--hfinal", type=float, default=-2.0)
    parser.add_argument("--hstep", type=float, default=0.01)
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parent
    base_structures_dir = run_dir / "base_structures"
    evaluations_dir = run_dir / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)

    struct_dirs = sorted(base_structures_dir.glob("structure_*"))
    if not struct_dirs:
        print("No base structures found. Run generate_structures.py first.")
        sys.exit(1)

    eval_name = f"eval_K1_{args.K1:g}_Js_{args.Js:g}_A_{args.A:g}"
    eval_run_dir = evaluations_dir / eval_name

    if not eval_run_dir.exists():
        print(f"Evaluation directory {eval_run_dir} does not exist. Run compute_evaluations.py first.")
        sys.exit(1)

    indiv_csv = evaluations_dir / "evaluation_results_individual.csv"
    avg_csv = evaluations_dir / "evaluation_results_average.csv"
    summary_csv = evaluations_dir / "evaluation_summary_scalars.csv"

    def append_val(collection, key, val, constructor):
        if key in collection:
            item = collection[key]
            old = np.atleast_1d(item.value).tolist() if hasattr(item, "value") else np.atleast_1d(item).tolist()
        else:
            old = []

        try:
            return constructor(old + [val])
        except Exception:
            # Fallback for raw lists if the entity constructor fails
            return old + [val]

    # Load existing indiv/summary collections if they exist to append to them
    if indiv_csv.exists() and os.path.getsize(indiv_csv) > 0:
        try:
            c_indiv = me.from_csv(indiv_csv)
        except Exception:
            c_indiv = me.EntityCollection()
    else:
        c_indiv = me.EntityCollection()

    if summary_csv.exists() and os.path.getsize(summary_csv) > 0:
        try:
            c_summary = me.from_csv(summary_csv)
        except Exception:
            c_summary = me.EntityCollection()
    else:
        c_summary = me.EntityCollection()

    all_data = []

    extent_str = "unknown"
    grains_val = "unknown"
    metadata_file = base_structures_dir / "metadata.txt"
    if metadata_file.exists():
        with open(metadata_file) as f:
            for line in f:
                if line.startswith("extent="):
                    extent_str = line.split("=")[1].strip()
                if line.startswith("grains="):
                    grains_val = line.split("=")[1].strip()

    for struct_dir in struct_dirs:
        struct_name = struct_dir.name
        run_struct_dir = eval_run_dir / struct_name

        mh_file = run_struct_dir / "hyst_isotrop" / "isotrop.mh"
        if mh_file.exists():
            data = np.loadtxt(mh_file, skiprows=1)
            all_data.append(data)

            # Use get_all_properties to correctly calculate both internal and external properties
            props = get_all_properties(data[:, 0], data[:, 1])
            Bc_ext, Jr_ext, Bc_int, Jr_int, Hc_int, Mr_int = props

            mu0 = 4 * np.pi * 1e-7
            Hc_ext_Am = Bc_ext / mu0
            Mr_ext_Am = Jr_ext / mu0

            # Ontology mappings for individual loops (keep requested properties in matching order with Index first)
            c_indiv["Index"] = append_val(c_indiv, "Index", struct_name, list)
            c_indiv["CoercivityHcExternal"] = append_val(
                c_indiv, "CoercivityHcExternal", Hc_ext_Am, lambda v: me.Entity("CoercivityHcExternal", v, "A/m")
            )
            c_indiv["M(Hext=0)"] = append_val(c_indiv, "M(Hext=0)", Mr_ext_Am, lambda v: v * (u.A / u.m))
            c_indiv["mu0HcExternal"] = append_val(c_indiv, "mu0HcExternal", Bc_ext, lambda v: v * u.T)
            c_indiv["J(Hext=0)"] = append_val(c_indiv, "J(Hext=0)", Jr_ext, lambda v: v * u.T)
        else:
            print(f"Warning: {mh_file} not found.")

    if all_data:
        try:
            c_indiv.to_csv(indiv_csv)
        except Exception as e:
            print("Error writing indiv mammos csv:", e)

        min_len = min(d.shape[0] for d in all_data)
        truncated_data = [d[:min_len, :] for d in all_data]
        data_stack = np.stack(truncated_data, axis=0)
        data_avg = np.mean(data_stack, axis=0)

        props = get_all_properties(data_avg[:, 0], data_avg[:, 1])
        Bc_ext, Jr_ext, Bc_int, Jr_int, Hc_int, Mr_int = props

        try:
            # Set index to 0 for all points
            config_idx = [0] * data_avg.shape[0]
            B_ext_col = data_avg[:, 0].tolist()
            J_par_col = data_avg[:, 1].tolist()

            c_avg = me.EntityCollection(
                config=me.Entity("Index", config_idx),
                B_ext_T=me.Entity("MagneticFluxDensity", B_ext_col, "T"),
                J_par_T=me.Entity("MagneticPolarisation", J_par_col, "T"),
            )
            c_avg.to_csv(avg_csv)
            print(f"[ok] Saved averaged demagnetization curve to {avg_csv} using mammos_entity")
        except Exception as e:
            print("Error writing mammos csv for average curve:", e)

        try:
            c_summary["GeometricalSize"] = append_val(
                c_summary, "GeometricalSize", extent_str, lambda v: StringEntity(v, "GeometricalSize")
            )
            c_summary["Grains"] = append_val(c_summary, "Grains", grains_val, list)
            c_summary["K1"] = append_val(c_summary, "K1", args.K1, lambda v: me.Entity("K1", v, "J/m^3"))
            c_summary["Js"] = append_val(c_summary, "Js", args.Js, lambda v: me.Entity("Js", v, "T"))
            c_summary["A"] = append_val(c_summary, "A", args.A, lambda v: me.Entity("A", v, "J/m"))

            mu0 = 4 * np.pi * 1e-7
            Hc_ext_Am = Bc_ext / mu0
            Mr_ext_Am = Jr_ext / mu0

            c_summary["CoercivityHcExternal"] = append_val(
                c_summary, "CoercivityHcExternal", Hc_ext_Am, lambda v: me.Entity("CoercivityHcExternal", v, "A/m")
            )
            c_summary["M(Hext=0)"] = append_val(c_summary, "M(Hext=0)", Mr_ext_Am, lambda v: v * (u.A / u.m))

            # User requested mu0HcExternal and J(Hext=0) in T
            c_summary["mu0HcExternal"] = append_val(c_summary, "mu0HcExternal", Bc_ext, lambda v: v * u.T)
            c_summary["J(Hext=0)"] = append_val(c_summary, "J(Hext=0)", Jr_ext, lambda v: v * u.T)

            c_summary["CoercivityHc"] = append_val(
                c_summary, "CoercivityHc", Hc_int, lambda v: me.Entity("CoercivityHc", v, "A/m")
            )
            c_summary["Remanence"] = append_val(
                c_summary, "Remanence", Mr_int, lambda v: me.Entity("Remanence", v, "A/m")
            )

            c_summary.to_csv(summary_csv)
            print(f"[ok] Saved summary scalars to {summary_csv} using mammos_entity")
        except Exception as e:
            print("Error writing mammos csv for summary scalars:", e)

        print(f"\n✓ Completed evaluation analysis for {eval_name}")
        print(f"Average mu0_Hc (ext): {Bc_ext:.4f} T, Jr (ext): {Jr_ext:.4f} T")
        print(f"Average mu0_Hc (int): {Bc_int:.4f} T, Jr (int): {Jr_int:.4f} T")
        print(f"Average Hc (int): {Hc_int:.4e} A/m, Mr (int): {Mr_int:.4e} A/m")

        plot_demagnetization_curves(all_data, data_avg, props, evaluations_dir, eval_name)
    else:
        print("No data collected for averaging.")


if __name__ == "__main__":
    main()
