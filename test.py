import mammos_entity as me
import numpy as np
from pathlib import Path

def append_to_csv(csv_path, ext, k1):
    if csv_path.exists() and csv_path.stat().st_size > 0:
        c = me.from_csv(csv_path)
    else:
        c = me.EntityCollection()

    def append_val(collection, key, val, constructor):
        if key in collection:
            item = collection[key]
            if hasattr(item, "value"):
                old = np.atleast_1d(item.value).tolist()
            else:
                old = np.atleast_1d(item).tolist()
        else:
            old = []
        return constructor(old + [val])

    c.Extent = append_val(c, "Extent", ext, list)
    c.K1 = append_val(c, "K1", k1, lambda v: me.K1(v, "J/m^3"))
    c.to_csv(csv_path)

p = Path("test_col.csv")
if p.exists(): p.unlink()
append_to_csv(p, "100x100", 1e6)
append_to_csv(p, "200x200", 2e6)
c2 = me.from_csv(p)
print(list(c2["Extent"]))
print(list(c2["K1"].value))
