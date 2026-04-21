import sys
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))
import mesh
import add_shell

L_cube = 20.0
h = 2.0
knt0, ijk0, _, _ = mesh.run_single_solid_mesher(
    geom='box', extent=f"{L_cube},{L_cube},{L_cube}", h=h, 
    backend='grid', no_vis=True, return_arrays=True
)
tmp_path = "test_mesh.npz"
np.savez(tmp_path, knt=knt0, ijk=ijk0)

knt, ijk = add_shell.run_add_shell_pipeline(
    in_npz=tmp_path,
    layers=8,
    K=1.4,
    h0=h,
    verbose=False
)
np.savez("cube_20nm_shell.npz", knt=knt, ijk=ijk)
print("Generated cube_20nm_shell.npz")
