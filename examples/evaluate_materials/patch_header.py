import re

with open("evaluate_properties.py", "r") as f:
    content = f.read()

header_logic = """    # Generate headers using mammos_entity where possible, fallback to MagMo labels
    try:
        import mammos_entity as me
        k1_label = me.K1([0]).ontology_label
        js_label = me.Js([0]).ontology_label
        a_label = me.A([0]).ontology_label
        hc_ext_label = me.Hc([0]).ontology_label  # usually CoercivityHcExternal
        mr_label = me.Mr([0]).ontology_label      # usually Remanence
    except:
        k1_label = "MagnetocrystallineAnisotropyConstantK1"
        js_label = "SpontaneousMagneticPolarization"
        a_label = "ExchangeStiffnessConstant"
        hc_ext_label = "CoercivityHcExternal"
        mr_label = "Remanence"

    csv_header = [
        "Extent", "Grains", 
        f"{k1_label}", 
        f"{js_label}", 
        f"{a_label}", 
        "CoercivityBHcExternal",   # mu0_Hc_ext_T
        "RemanenceExternal",       # Jr_ext_T
        "CoercivityBHc",           # mu0_Hc_int_T
        f"{mr_label}",             # Jr_int_T
        "CoercivityHc",            # Hc_int_Am
        f"{mr_label}"              # Mr_int_Am
    ]
"""

old_init = """    # Initialize CSV files
    if not indiv_csv.exists():
        with open(indiv_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Structure", "Extent", "Grains", "K1", "Js", "A", "Hc [A/m]", "Jr [T]"])
    if not avg_csv.exists():
        with open(avg_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Extent", "Grains", "K1", "Js", "A", "mu0_Hc_ext_T", "Jr_ext_T", "mu0_Hc_int_T", "Jr_int_T", "Hc_int_Am", "Mr_int_Am"])"""

new_init = header_logic + """
    # Initialize CSV files
    if not indiv_csv.exists():
        with open(indiv_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Structure", "Extent", "Grains", "K1", "Js", "A", "Hc [A/m]", "Jr [T]"])
    if not avg_csv.exists():
        with open(avg_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)"""

content = content.replace(old_init, new_init)

with open("evaluate_properties.py", "w") as f:
    f.write(content)
