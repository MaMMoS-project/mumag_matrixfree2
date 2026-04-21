#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from typing import Dict, List, Tuple, Optional
import argparse
import numpy as np
from scipy.spatial import KDTree

def parse_lsdyna_mesh(path: str):
    """Parses nodes and elements from an LS-DYNA keyword file."""
    nodes: Dict[int, Tuple[float, float, float]] = {}
    elements: Dict[int, List[int]] = {}
    section = None

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('*'):
                u = line.upper()
                if u.startswith('*NODE'):
                    section = 'NODE'
                elif u.startswith('*ELEMENT_SOLID'):
                    section = 'ELEMENT_SOLID'
                else:
                    section = None
                continue
            
            if line.startswith('$'):
                continue

            if section == 'NODE':
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        nid = int(parts[0])
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        nodes[nid] = (x, y, z)
                    except ValueError:
                        continue
            elif section == 'ELEMENT_SOLID':
                parts = line.split()
                if len(parts) >= 10:
                    try:
                        eid = int(parts[0])
                        # Node IDs are parts[2] through parts[9]
                        nids = [int(p) for p in parts[2:10]]
                        elements[eid] = nids
                    except ValueError:
                        continue
    return nodes, elements

def apply_symmetry(nodes: Dict[int, Tuple[float, float, float]], 
                   elements: Dict[int, List[int]], 
                   element_data: Dict[str, Dict[int, float]]):
    """Applies 8-fold symmetry (reflections across XY, YZ, ZX planes) to create a full sphere."""
    full_nodes = {}
    full_elements = {}
    full_element_data = {name: {} for name in element_data}
    
    max_nid = max(nodes.keys()) if nodes else 0
    max_eid = max(elements.keys()) if elements else 0
    
    # Octants: (sx, sy, sz) where s is 1 or -1
    octants = [
        (1, 1, 1),   # 0: original (+x, +y, +z)
        (-1, 1, 1),  # 1: refl X    (-x, +y, +z)
        (1, -1, 1),  # 2: refl Y    (+x, -y, +z)
        (1, 1, -1),  # 3: refl Z    (+x, +y, -z)
        (-1, -1, 1), # 4: refl XY   (-x, -y, +z)
        (-1, 1, -1), # 5: refl XZ   (-x, +y, -z)
        (1, -1, -1), # 6: refl YZ   (+x, -y, -z)
        (-1, -1, -1) # 7: refl XYZ  (-x, -y, -z)
    ]
    
    for i, (sx, sy, sz) in enumerate(octants):
        node_offset = i * (max_nid + 1)
        elem_offset = i * (max_eid + 1)
        
        # If an odd number of reflections are applied, the element orientation is flipped.
        flip = (sx * sy * sz) < 0
        
        for nid, (x, y, z) in nodes.items():
            full_nodes[nid + node_offset] = (x * sx, y * sy, z * sz)
            
        for eid, nids in elements.items():
            new_nids = [nid + node_offset for nid in nids]
            if flip:
                # Swap pairs to fix orientation: 0-1, 3-2, 4-5, 7-6
                new_nids = [new_nids[1], new_nids[0], new_nids[3], new_nids[2],
                            new_nids[5], new_nids[4], new_nids[7], new_nids[6]]
            
            full_elements[eid + elem_offset] = new_nids
            
            for name, data_map in element_data.items():
                if eid in data_map:
                    full_element_data[name][eid + elem_offset] = data_map[eid]
                    
    return full_nodes, full_elements, full_element_data

def parse_element_results(path: str, result_name: str, state_no: int = 5) -> Dict[int, float]:
    """Parses element-wise results from the $SOLID_ELEMENT_RESULTS section."""
    results: Dict[int, float] = {}
    in_results_section = False
    current_state = None
    current_result_name = None

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('$'):
                su = line.upper()
                if su.startswith('$SOLID_ELEMENT_RESULTS'):
                    in_results_section = True
                elif su.startswith('$STATE_NO'):
                    try:
                        current_state = int(line.split('=')[1].strip())
                    except:
                        current_state = None
                elif su.startswith('$RESULT OF'):
                    current_result_name = line[10:].strip()
                continue
            
            if line.startswith('*'):
                in_results_section = False
                continue

            if in_results_section and current_state == state_no and current_result_name == result_name:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        eid = int(parts[0])
                        val = float(parts[1])
                        results[eid] = val
                    except ValueError:
                        continue
    return results

def parse_inp(path: str):
    """Parses an AVS UCD (.inp) file."""
    print(f"Parsing INP file: {path}")
    with open(path, 'r') as f:
        header_line = f.readline().split()
        if not header_line:
            return None, None, None
        num_nodes = int(header_line[0])
        num_elems = int(header_line[1])
        num_node_data = int(header_line[2])
        num_cell_data = int(header_line[3])
        
        nodes: Dict[int, Tuple[float, float, float]] = {}
        for _ in range(num_nodes):
            line = f.readline().split()
            if not line: break
            nid = int(line[0])
            coords = (float(line[1]), float(line[2]), float(line[3]))
            nodes[nid] = coords
            
        elements = []
        for _ in range(num_elems):
            line = f.readline().split()
            if not line: break
            eid = int(line[0])
            mat_id = int(line[1])
            etype = line[2]
            nids = [int(x) for x in line[3:]]
            elements.append({'eid': eid, 'mat_id': mat_id, 'type': etype, 'nids': nids})
            
        # We don't necessarily need existing data for this task, 
        # but we need to advance the file pointer if we were to read more.
    return nodes, elements, header_line

def write_inp(path: str, nodes: Dict[int, Tuple[float, float, float]], 
              elements: List[Dict], cell_data: Dict[str, List[float]]):
    """Writes an AVS UCD (.inp) file with cell data."""
    num_nodes = len(nodes)
    num_elems = len(elements)
    num_cell_data = len(cell_data)
    
    print(f"Writing INP file: {path}")
    with open(path, 'w') as f:
        f.write(f"{num_nodes} {num_elems} 0 {num_cell_data} 0\n")
        
        # Write nodes
        # Sort by ID to be safe
        for nid in sorted(nodes.keys()):
            x, y, z = nodes[nid]
            f.write(f"{nid} {x} {y} {z}\n")
            
        # Write elements
        for elem in elements:
            nids_str = " ".join(map(str, elem['nids']))
            f.write(f"{elem['eid']} {elem['mat_id']} {elem['type']} {nids_str}\n")
            
        if num_cell_data > 0:
            # Cell data header
            widths = " ".join(["1"] * num_cell_data)
            f.write(f"{num_cell_data} {widths}\n")
            # Labels
            labels = list(cell_data.keys())
            for label in labels:
                f.write(f"{label}, none\n")
            # Values
            for i in range(num_elems):
                eid = elements[i]['eid']
                vals = [str(cell_data[label][i]) for label in labels]
                f.write(f"{eid} {' '.join(vals)}\n")

def get_centroids(nodes, elements):
    centroids = []
    eids = []
    for eid, nids in elements.items():
        try:
            pts = [nodes[nid] for nid in nids]
            centroid = np.mean(pts, axis=0)
            centroids.append(centroid)
            eids.append(eid)
        except KeyError:
            continue
    return np.array(centroids), eids

def main():
    parser = argparse.ArgumentParser(description="Map LS-DYNA results to an unstructured INP grid.")
    parser.add_argument("--full", action="store_true", help="Apply 8-fold symmetry to LS-DYNA data.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Parse LS-DYNA data
    path_1x = os.path.join(script_dir, "1x")
    path_1y = os.path.join(script_dir, "1y")
    path_1z = os.path.join(script_dir, "1z")

    print(f"Reading LS-DYNA mesh geometry from {path_1x}...")
    ls_nodes, ls_elements = parse_lsdyna_mesh(path_1x)
    if not ls_nodes:
        print(f"Error: No nodes found in {path_1x}.")
        return

    print("Reading LS-DYNA strains...")
    strain_x = parse_element_results(path_1x, "X-strain-Infinitesimal")
    strain_y = parse_element_results(path_1y, "Y-strain-Infinitesimal")
    strain_z = parse_element_results(path_1z, "Z-strain-Infinitesimal")

    # 2. Compute k1me and k1me_p for LS-DYNA elements
    B11 = 75.0e6
    B12 = -13.0e6
    B13 = -81.1e6

    ls_element_data = {"k1me": {}, "k1me_p": {}}
    for eid in ls_elements:
        ex = strain_x.get(eid, 0.0)
        ey = strain_y.get(eid, 0.0)
        ez = strain_z.get(eid, 0.0)
        
        ls_element_data["k1me"][eid] = 0.5 * (B11 + B12) * (ex + ey) - B13 * ez
        ls_element_data["k1me_p"][eid] = 0.5 * (B11 - B12) * (ex - ey)

    # 3. Apply symmetry if requested
    if args.full:
        print("Applying 8-fold symmetry to LS-DYNA data...")
        ls_nodes, ls_elements, ls_element_data = apply_symmetry(ls_nodes, ls_elements, ls_element_data)

    # 4. Prepare KDTree for fast search
    print("Building KDTree for LS-DYNA centroids...")
    ls_centroids, ls_eids = get_centroids(ls_nodes, ls_elements)
    tree = KDTree(ls_centroids)

    # 5. Parse target ring mesh
    ring_inp_path = os.path.join(script_dir, "ring.converted.inp")
    if not os.path.exists(ring_inp_path):
        print(f"Error: {ring_inp_path} not found.")
        return
        
    ring_nodes, ring_elements, _ = parse_inp(ring_inp_path)
    if not ring_nodes:
        print(f"Error: Could not parse {ring_inp_path}")
        return

    # 6. Map data to ring mesh
    print("Mapping data to ring mesh...")
    mapped_k1me = []
    mapped_k1me_p = []
    
    # Calculate centroids for ring elements
    ring_centroids = []
    for elem in ring_elements:
        pts = [ring_nodes[nid] for nid in elem['nids']]
        ring_centroids.append(np.mean(pts, axis=0))
    
    ring_centroids = np.array(ring_centroids)
    
    # Query KDTree
    distances, indices = tree.query(ring_centroids)
    
    for idx in indices:
        matched_ls_eid = ls_eids[idx]
        mapped_k1me.append(ls_element_data["k1me"][matched_ls_eid])
        mapped_k1me_p.append(ls_element_data["k1me_p"][matched_ls_eid])

    # 7. Write output INP
    output_data = {
        "k1me": mapped_k1me,
        "k1me_p": mapped_k1me_p
    }
    output_path = os.path.join(script_dir, "ring.mappend.inp")
    write_inp(output_path, ring_nodes, ring_elements, output_data)
    print(f"Done. Saved to {output_path}")

if __name__ == "__main__":
    main()
