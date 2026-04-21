import sys
import re
import subprocess
import platform
from typing import Dict, Any

def get_hardware_info() -> Dict[str, str]:
    info = {}
    info['OS'] = f"{platform.system()} {platform.release()}"
    
    try:
        # CPU
        if platform.system() == "Linux":
            lscpu = subprocess.check_output("lscpu", shell=True).decode()
            model_name = re.search(r"Model name:\s+(.*)", lscpu)
            if model_name:
                info['CPU'] = model_name.group(1).strip()
        else:
            info['CPU'] = platform.processor()
            
        # GPU
        try:
            nvidia_smi = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True).decode()
            info['GPU'] = nvidia_smi.strip()
        except:
            info['GPU'] = "Unknown / Not NVIDIA"
    except:
        pass
    return info

def parse_output(text: str) -> Dict[str, Any]:
    results = {}
    
    # Mesh info
    mesh_match = re.search(r"Mesh:\s+(\d+)\s+nodes,\s+(\d+)\s+elements", text)
    if mesh_match:
        results['nodes'] = int(mesh_match.group(1))
        results['elements'] = int(mesh_match.group(2))
    elif "Total elements:" in text:
        nodes_match = re.search(r"nodes=(\d+)", text)
        elements_match = re.search(r"elements=(\d+)", text)
        if nodes_match: results['nodes'] = int(nodes_match.group(1))
        if elements_match: results['elements'] = int(elements_match.group(2))

    # Benchmark results
    # Matches: Jacobi      :  330 iterations, 0.528 s, rel_res: 9.69e-11
    # or:      Amgcl       :   34 iterations, 0.231 s, rel_res: 6.34e-11
    matches = re.findall(r"(\w+)\s+:\s+(\d+)\s+iterations,\s+([\d\.]+)\s+s,\s+rel_res:\s+([\d\.e\-\+]+)", text)
    for m in matches:
        ptype = m[0].strip()
        results[ptype] = {
            'iters': int(m[1]),
            'time': float(m[2]),
            'res': float(m[3])
        }
        
    return results

def safe_format(val: Any, fmt: str) -> str:
    if val is None or val == 'N/A': return 'N/A'
    return f"{val:{fmt}}"

def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: generate_report.py python_output.txt cpp_output.txt")
        return

    with open(sys.argv[1], 'r') as f:
        py_text = f.read()
    with open(sys.argv[2], 'r') as f:
        cpp_text = f.read()

    py_res = parse_output(py_text)
    cpp_res = parse_output(cpp_text)
    hw = get_hardware_info()

    table_rows = []
    # Add JAX results
    for pt in ["None", "Jacobi", "Chebyshev", "Amg", "Amgcl"]:
        if pt in py_res:
            r = py_res[pt]
            table_rows.append(f"| Python ({pt}) | {r['iters']} | {r['time']:.3f} | {r['res']:.2e} |")
    
    # Add C++ result
    if "Amg" in cpp_res:
        r = cpp_res["Amg"]
        table_rows.append(f"| **C++ (Native)** | **{r['iters']}** | **{r['time']:.3f}** | **{r['res']:.2e}** |")

    table_body = "\n".join(table_rows)

    report = f"""# Poisson Solver Convergence Benchmark Report

## Hardware Information
- **OS**: {hw.get('OS', 'Unknown')}
- **CPU**: {hw.get('CPU', 'Unknown')}
- **GPU**: {hw.get('GPU', 'Unknown')}

## Mesh Information
- **Nodes**: {cpp_res.get('nodes', py_res.get('nodes', 'Unknown')):,}
- **Elements**: {cpp_res.get('elements', py_res.get('elements', 'Unknown')):,}

## Performance Comparison (Tolerance 1e-10)

| Implementation | Iterations | Time (s) | Rel. Residual |
| :--- | :---: | :---: | :---: |
{table_body}

---
*Generated automatically by benchmark script.*
"""



    with open("BENCHMARK_REPORT.md", "w") as f:
        f.write(report)
    print("[ok] Benchmark report generated: BENCHMARK_REPORT.md")

if __name__ == "__main__":
    main()
