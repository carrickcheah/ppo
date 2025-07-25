# Training Snapshot Validation Report

Generated: 2025-07-24 17:00:37

Total Snapshots: 9

## Summary Table

| Snapshot | Tasks | Machines | Multi-Machine % | Urgent % | Status |
|----------|-------|----------|-----------------|----------|--------|
| edge_case_cascading.json | 120 | 20 | 64.2% | 40.0% | ✅ Valid |
| edge_case_conflicts.json | 80 | 10 | 73.8% | 100.0% | ✅ Valid |
| edge_case_multi_complex.json | 90 | 25 | 100.0% | 40.0% | ✅ Valid |
| edge_case_same_machine.json | 48 | 5 | 0.0% | 56.2% | ✅ Valid |
| snapshot_bottleneck.json | 212 | 50 | 92.9% | 28.7% | ✅ Valid |
| snapshot_heavy.json | 504 | 145 | 96.4% | 28.9% | ✅ Valid |
| snapshot_multi_heavy.json | 295 | 145 | 97.6% | 29.5% | ✅ Valid |
| snapshot_normal.json | 295 | 145 | 95.9% | 29.5% | ✅ Valid |
| snapshot_rush.json | 295 | 145 | 95.9% | 86.4% | ✅ Valid |

## Detailed Results

### edge_case_cascading.json

**Description**: Edge case: Cascading deadline dependencies
**Type**: edge_cascading

**Basic Statistics:**
- Families: 60
- Tasks: 120
- Machines: 20
- Multi-machine tasks: 77 (64.2%)
- Important families: 20 (33.3%)

**Processing Times (hours):**
- Range: 0.5 - 2.0
- Mean: 1.2
- Total: 141.6

**Deadline Distribution:**
- Range: 3 - 14 days
- Urgent jobs (≤7 days): 24 (40.0%)

**Machine Utilization Preview:**
- Most used machines:
  - Machine 15: 17 tasks
  - Machine 10: 16 tasks
  - Machine 13: 15 tasks
  - Machine 6: 15 tasks
  - Machine 9: 14 tasks
---

### edge_case_conflicts.json

**Description**: Edge case: Conflicting priorities with impossible deadlines
**Type**: edge_conflicts

**Basic Statistics:**
- Families: 40
- Tasks: 80
- Machines: 10
- Multi-machine tasks: 59 (73.8%)
- Important families: 40 (100.0%)

**Processing Times (hours):**
- Range: 8.0 - 23.8
- Mean: 16.1
- Total: 1290.1

**Deadline Distribution:**
- Range: 1 - 3 days
- Urgent jobs (≤7 days): 40 (100.0%)

**Machine Utilization Preview:**
- Most used machines:
  - Machine 4: 26 tasks
  - Machine 8: 20 tasks
  - Machine 9: 19 tasks
  - Machine 7: 19 tasks
  - Machine 5: 18 tasks
---

### edge_case_multi_complex.json

**Description**: Edge case: Complex multi-machine requirements
**Type**: edge_multi_complex

**Basic Statistics:**
- Families: 30
- Tasks: 90
- Machines: 25
- Multi-machine tasks: 90 (100.0%)
- Important families: 13 (43.3%)

**Processing Times (hours):**
- Range: 2.0 - 8.0
- Mean: 4.9
- Total: 438.8

**Deadline Distribution:**
- Range: 3 - 14 days
- Urgent jobs (≤7 days): 12 (40.0%)

**Machine Utilization Preview:**
- Most used machines:
  - Machine 21: 25 tasks
  - Machine 1: 24 tasks
  - Machine 12: 23 tasks
  - Machine 5: 22 tasks
  - Machine 4: 21 tasks
---

### edge_case_same_machine.json

**Description**: Edge case: All jobs need same machine
**Type**: edge_same_machine

**Basic Statistics:**
- Families: 16
- Tasks: 48
- Machines: 5
- Multi-machine tasks: 0 (0.0%)
- Important families: 7 (43.8%)

**Processing Times (hours):**
- Range: 1.3 - 4.9
- Mean: 2.9
- Total: 138.5

**Deadline Distribution:**
- Range: 3 - 10 days
- Urgent jobs (≤7 days): 9 (56.2%)

**Machine Utilization Preview:**
- Most used machines:
  - Machine 1: 41 tasks
  - Machine 2: 3 tasks
  - Machine 3: 2 tasks
  - Machine 5: 2 tasks
- Unused machines: 1

---

### snapshot_bottleneck.json

**Description**: Machine bottleneck (50 machines)
**Type**: machine_bottleneck

**Basic Statistics:**
- Families: 87
- Tasks: 212
- Machines: 50
- Multi-machine tasks: 197 (92.9%)
- Important families: 25 (28.7%)

**Processing Times (hours):**
- Range: 0.0 - 170.0
- Mean: 29.9
- Total: 6341.2

**Deadline Distribution:**
- Range: 1 - 29 days
- Urgent jobs (≤7 days): 25 (28.7%)

**Machine Utilization Preview:**
- Most used machines:
  - Machine 128: 132 tasks
  - Machine 17: 132 tasks
  - Machine 34: 132 tasks
  - Machine 81: 132 tasks
  - Machine 82: 132 tasks
---

### snapshot_heavy.json

**Description**: Heavy load (149 families)
**Type**: heavy_load

**Basic Statistics:**
- Families: 149
- Tasks: 504
- Machines: 145
- Multi-machine tasks: 486 (96.4%)
- Important families: 46 (30.9%)

**Processing Times (hours):**
- Range: 0.0 - 187.8
- Mean: 37.5
- Total: 18896.6

**Deadline Distribution:**
- Range: 1 - 32 days
- Urgent jobs (≤7 days): 43 (28.9%)

**Machine Utilization Preview:**
- Most used machines:
  - Machine 17: 227 tasks
  - Machine 34: 227 tasks
  - Machine 81: 227 tasks
  - Machine 82: 227 tasks
  - Machine 83: 227 tasks
- Unused machines: 30

---

### snapshot_multi_heavy.json

**Description**: Multi-machine heavy (97.6% multi)
**Type**: multi_machine_heavy

**Basic Statistics:**
- Families: 88
- Tasks: 295
- Machines: 145
- Multi-machine tasks: 288 (97.6%)
- Important families: 26 (29.5%)

**Processing Times (hours):**
- Range: 0.0 - 170.0
- Mean: 36.8
- Total: 10851.4

**Deadline Distribution:**
- Range: 1 - 29 days
- Urgent jobs (≤7 days): 26 (29.5%)

**Machine Utilization Preview:**
- Most used machines:
  - Machine 81: 133 tasks
  - Machine 84: 133 tasks
  - Machine 85: 133 tasks
  - Machine 17: 132 tasks
  - Machine 34: 132 tasks
- Unused machines: 27

---

### snapshot_normal.json

**Description**: Normal production load
**Type**: normal

**Basic Statistics:**
- Families: 88
- Tasks: 295
- Machines: 145
- Multi-machine tasks: 283 (95.9%)
- Important families: 26 (29.5%)

**Processing Times (hours):**
- Range: 0.0 - 170.0
- Mean: 36.8
- Total: 10851.4

**Deadline Distribution:**
- Range: 1 - 29 days
- Urgent jobs (≤7 days): 26 (29.5%)

**Machine Utilization Preview:**
- Most used machines:
  - Machine 17: 132 tasks
  - Machine 34: 132 tasks
  - Machine 81: 132 tasks
  - Machine 82: 132 tasks
  - Machine 83: 132 tasks
- Unused machines: 30

---

### snapshot_rush.json

**Description**: Rush orders (80% urgent)
**Type**: rush_orders

**Basic Statistics:**
- Families: 88
- Tasks: 295
- Machines: 145
- Multi-machine tasks: 283 (95.9%)
- Important families: 54 (61.4%)

**Processing Times (hours):**
- Range: 0.0 - 170.0
- Mean: 36.8
- Total: 10851.4

**Deadline Distribution:**
- Range: 1 - 22 days
- Urgent jobs (≤7 days): 76 (86.4%)

**Machine Utilization Preview:**
- Most used machines:
  - Machine 17: 132 tasks
  - Machine 34: 132 tasks
  - Machine 81: 132 tasks
  - Machine 82: 132 tasks
  - Machine 83: 132 tasks
- Unused machines: 30

---

## Overall Statistics

- Total tasks across all snapshots: 1939
- Total multi-machine tasks: 1763
- Overall multi-machine percentage: 90.9%