#!/usr/bin/env python3
"""
Verify the units of processing times in the data
"""

import json

# Load the real production snapshot
with open('data/real_production_snapshot.json', 'r') as f:
    data = json.load(f)

# Analyze processing times
processing_times = []
for family_id, family in data['families'].items():
    for task in family['tasks']:
        pt = task.get('processing_time', 0)
        processing_times.append(pt)

print("Processing Time Analysis")
print("="*40)
print(f"Total tasks: {len(processing_times)}")
print(f"Min: {min(processing_times):.2f}")
print(f"Max: {max(processing_times):.2f}")
print(f"Average: {sum(processing_times)/len(processing_times):.2f}")
print(f"Total: {sum(processing_times):.1f}")

# Show distribution
print("\nDistribution:")
ranges = [(0, 1), (1, 10), (10, 50), (50, 100), (100, 200), (200, 1000)]
for low, high in ranges:
    count = sum(1 for pt in processing_times if low <= pt < high)
    print(f"  {low:3d}-{high:3d}: {count:3d} tasks ({count/len(processing_times)*100:5.1f}%)")

# If these are hours, what's the workload?
total_hours = sum(processing_times)
print(f"\nIf units are HOURS:")
print(f"  Total workload: {total_hours:.1f} hours")
print(f"  Days (24h): {total_hours/24:.1f} days")
print(f"  Weeks (40h): {total_hours/40:.1f} weeks")
print(f"  With 149 machines: {total_hours/149:.1f} hours/machine minimum")

# If these are minutes, what's the workload?
total_minutes = sum(processing_times)
print(f"\nIf units are MINUTES:")
print(f"  Total workload: {total_minutes/60:.1f} hours")
print(f"  With 149 machines: {total_minutes/60/149:.1f} hours/machine minimum")