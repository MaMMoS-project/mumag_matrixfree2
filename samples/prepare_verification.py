import numpy as np

# 1. Create half-K1 .krn file
krn_content = """# theta (rad) phi (rad) K1 (J/m3) not used Js (Tesla) A (J/m)
0.0 0.0 2.15e6 0.0 1.6 7.7e-12 0.0 0.0
"""
with open("samples/cube_20nm_halfK1.krn", "w") as f:
    f.write(krn_content)
print("Wrote samples/cube_20nm_halfK1.krn")

# 2. Create .inp file from .npz with half-K1me
data = np.load("samples/cube_20nm.npz")
nodes = data["knt"]
ijk = data["ijk"]

num_nodes = len(nodes)
num_elems = len(ijk)

k1me_val = 2.15e6
k1me_p_val = 0.0

with open("samples/cube_20nm_halfK1me.inp", "w") as f:
    # Header
    f.write(f"{num_nodes} {num_elems} 0 2 0\n")
    # Nodes
    for i, (x, y, z) in enumerate(nodes):
        f.write(f"{i+1} {x} {y} {z}\n")
    # Elements
    for i, conn in enumerate(ijk):
        mat_id = int(conn[4]) if len(conn) > 4 else 1
        f.write(f"{i+1} {mat_id} tet {conn[0]+1} {conn[1]+1} {conn[2]+1} {conn[3]+1}\n")
    # Cell Data
    f.write("2\n")
    f.write("k1me, J/m3\n")
    f.write("k1me_p, J/m3\n")
    for i in range(num_elems):
        f.write(f"{i+1} {k1me_val} {k1me_p_val}\n")

print("Wrote samples/cube_20nm_halfK1me.inp")
