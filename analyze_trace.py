import json
import gzip
from collections import defaultdict

trace_file = 'trace_dir/plugins/profile/2026_03_11_01_33_22/hp.trace.json.gz'

stats = defaultdict(lambda: {'count': 0, 'dur': 0.0})

with gzip.open(trace_file, 'rt') as f:
    data = json.load(f)
    for event in data.get('traceEvents', []):
        if event.get('ph') == 'X': # Complete event
            name = event.get('name', 'UNKNOWN')
            dur = event.get('dur', 0.0)
            hlo_op = event.get('args', {}).get('hlo_op', '')
            if hlo_op:
                name = f"{name} ({hlo_op})"
            stats[name]['count'] += 1
            stats[name]['dur'] += dur

# Sort by duration
sorted_stats = sorted(stats.items(), key=lambda x: x[1]['dur'], reverse=True)

print(f"{'Kernel Name':<80} | {'Count':>6} | {'Total Dur (ms)':>15}")
print("-" * 105)
for name, data in sorted_stats[:30]:
    print(f"{name[:80]:<80} | {data['count']:>6} | {data['dur']/1000:>15.2f}")
