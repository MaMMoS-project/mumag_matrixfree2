"""analyze_trace.py

Utility for analyzing JAX/XLA execution traces (hp.trace.json.gz).
Aggregates durations by operation and identifies top GPU/assembly kernels.
"""

import json
import gzip
import collections
from pathlib import Path


def analyze_trace(trace_path: str | Path):
    """Analyze a JAX/XLA execution trace file.

    Parses the compressed JSON trace, aggregates event durations by name,
    and prints reports for top operations, GPU kernels, and categories.

    Args:
        trace_path (str | Path): Path to the `hp.trace.json.gz` file or a 
            directory containing one.

    Example:
        >>> analyze_trace("trace_dir/plugins/profile/hp.trace.json.gz")
    """
    trace_path = Path(trace_path)
    print(f"Analyzing {trace_path}...")
    with gzip.open(trace_path, 'rt') as f:
        trace = json.load(f)

    # trace_events is usually the key
    events = trace.get('traceEvents', [])
    
    # We want to group by name and sum durations
    # Only consider events with duration 'dur'
    stats = collections.defaultdict(float)
    counts = collections.defaultdict(int)
    
    total_dur = 0
    for ev in events:
        if 'dur' in ev and 'name' in ev:
            # Durations are in microseconds
            stats[ev['name']] += ev['dur']
            counts[ev['name']] += 1
            total_dur += ev['dur']

    # Sort by duration
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 20 Operations by Total Duration (All):")
    print(f"{'Operation':<60} | {'Count':<8} | {'Total (ms)':<12} | {'%':<5}")
    print("-" * 95)
    
    for name, dur_us in sorted_stats[:20]:
        dur_ms = dur_us / 1000.0
        percentage = (dur_us / total_dur) * 100 if total_dur > 0 else 0
        display_name = (name[:57] + '...') if len(name) > 60 else name
        print(f"{display_name:<60} | {counts[name]:<8} | {dur_ms:<12.2f} | {percentage:<5.1f}%")

    # GPU Kernel Analysis
    kernel_stats = collections.defaultdict(float)
    kernel_counts = collections.defaultdict(int)
    
    for ev in events:
        if 'dur' in ev and 'name' in ev:
            name = ev['name']
            # Heuristics for GPU kernels: 
            # 1. Contains 'void' (CUDA kernel signatures)
            # 2. Category is 'Kernel'
            # 3. Specific assembly keywords
            is_kernel = ('void' in name or 
                         ev.get('cat') == 'Kernel' or 
                         any(k in name.lower() for k in ['scatter', 'segment', 'reduce', 'gemm', 'solve']))
            
            if is_kernel:
                kernel_stats[name] += ev['dur']
                kernel_counts[name] += 1

    sorted_kernels = sorted(kernel_stats.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop GPU/Assembly Kernels:")
    print(f"{'Kernel':<80} | {'Count':<8} | {'Total (ms)':<12}")
    print("-" * 105)
    for name, dur_us in sorted_kernels[:20]:
        display_name = (name[:77] + '...') if len(name) > 80 else name
        print(f"{display_name:<80} | {kernel_counts[name]:<8} | {dur_us/1000.0:<12.2f}")

    # Explicitly look for the assembly kernels to compare scatter vs segment
    print("\nAssembly-Related Kernels (Scatter/Segment/Reduce):")
    found_assembly = False
    for name, dur_us in sorted_kernels:
        if any(k in name.lower() for k in ['scatter', 'segment', 'reduce']):
            print(f"{name[:77]:<80} | {kernel_counts[name]:<8} | {dur_us/1000.0:<12.2f} ms")
            found_assembly = True
    if not found_assembly:
        print("No explicit scatter/segment/reduce kernels found in trace.")

    # Group by category if possible
    cat_stats = collections.defaultdict(float)
    for ev in events:
        if 'dur' in ev and 'cat' in ev:
            cat_stats[ev['cat']] += ev['dur']
            
    print("\nDuration by Category:")
    cat_items = sorted(cat_stats.items(), key=lambda x: x[1], reverse=True)
    for cat, dur_us in cat_items:
        percentage = (dur_us / total_dur) * 100 if total_dur > 0 else 0
        print(f"{cat:<30}: {dur_us/1000.0:>12.2f} ms ({percentage:>5.1f}%)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        trace_path = Path(sys.argv[1])
        if trace_path.is_dir():
            trace_files = list(trace_path.rglob("hp.trace.json.gz"))
            if not trace_files:
                print(f"No trace files found in {trace_path}")
                sys.exit(1)
            trace_path = max(trace_files, key=lambda p: p.stat().st_mtime)
        analyze_trace(trace_path)
    else:
        # Find the most recent trace in current dir
        trace_files = list(Path(".").rglob("hp.trace.json.gz"))
        if not trace_files:
            print("No trace files found.")
        else:
            # Sort by mtime to get the latest
            latest_trace = max(trace_files, key=lambda p: p.stat().st_mtime)
            analyze_trace(latest_trace)
