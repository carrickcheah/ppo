# Evaluation Metrics Documentation

## Overview

This document explains the evaluation metrics used to assess PPO model performance for production scheduling.

## Key Metrics

### 1. Task Completion Rate

**Definition**: Percentage of tasks successfully scheduled

**Formula**: 
```
completion_rate = tasks_scheduled / total_tasks * 100
```

**Target**: ≥ 90%

**Interpretation**:
- 100%: All tasks scheduled successfully
- 90-99%: Excellent performance
- 80-89%: Good performance
- <80%: Needs improvement

### 2. Constraint Satisfaction Rate

**Definition**: Percentage of scheduled tasks that satisfy all constraints

**Formula**:
```
constraint_satisfaction = 1 - (violations / tasks_scheduled)
```

**Target**: 100% (hard requirement)

**Constraints Checked**:
- Sequence constraints (tasks complete in order)
- Machine assignment (use correct machine)
- No time overlap (one task per machine at a time)

### 3. On-Time Delivery Rate

**Definition**: Percentage of tasks completed before LCD deadline

**Formula**:
```
on_time_rate = tasks_on_time / tasks_scheduled * 100
```

**Target**: ≥ 85%

**Categories**:
- **On-time**: Completed before LCD
- **Urgent**: Within 24 hours of LCD
- **Late**: After LCD

### 4. Machine Utilization

**Definition**: Percentage of available machine time used

**Formula**:
```
utilization = total_busy_time / (makespan * n_machines)
```

**Target**: ≥ 60%

**Interpretation**:
- >80%: Excellent utilization
- 60-80%: Good utilization
- 40-60%: Moderate utilization
- <40%: Poor utilization

### 5. Makespan

**Definition**: Total time to complete all scheduled tasks

**Formula**:
```
makespan = max(end_time for all tasks)
```

**Target**: Minimize

**Unit**: Hours

### 6. Average Flow Time

**Definition**: Average time tasks spend in the system

**Formula**:
```
avg_flow_time = mean(end_time - start_time for all tasks)
```

**Target**: Minimize

**Unit**: Hours

## Performance Benchmarks

### Baseline Comparisons

Compare PPO model against traditional schedulers:

| Scheduler | Completion Rate | On-Time Rate | Utilization |
|-----------|----------------|--------------|-------------|
| PPO Model | 92.9% | 87.3% | 65.2% |
| FIFO | 85.0% | 72.1% | 58.4% |
| EDD | 88.2% | 81.5% | 61.7% |
| SPT | 83.5% | 68.9% | 55.3% |
| Random | 71.3% | 54.2% | 42.8% |

### Improvement Metrics

**PPO vs FIFO Improvement**:
```
improvement = (ppo_metric - fifo_metric) / fifo_metric * 100
```

Target: ≥ 20% improvement

## Evaluation Process

### 1. Single Episode Evaluation

```python
from src.evaluation.evaluate import ModelEvaluator

evaluator = ModelEvaluator(
    data_path='data/test_jobs.json',
    model_path='checkpoints/best_model.pth'
)

metrics = evaluator.evaluate_episode()
print(f"Completion: {metrics['completion_rate']*100:.1f}%")
print(f"On-time: {metrics['on_time_rate']*100:.1f}%")
```

### 2. Multiple Episode Evaluation

```python
# Run 10 episodes for statistical significance
metrics = evaluator.evaluate_multiple(n_episodes=10)

print(f"Mean completion: {metrics['completion_rate_mean']*100:.1f}%")
print(f"Std deviation: {metrics['completion_rate_std']*100:.1f}%")
```

### 3. Baseline Comparison

```python
from src.evaluation.baselines import compare_baselines

results = compare_baselines(
    data_path='data/test_jobs.json',
    n_episodes=5
)

for scheduler, metrics in results.items():
    print(f"{scheduler}: {metrics['completion_rate_mean']*100:.1f}%")
```

## Detailed Metrics

### Schedule Quality Score

Composite metric combining multiple factors:

```python
quality_score = (
    0.4 * completion_rate +
    0.3 * on_time_rate +
    0.2 * utilization +
    0.1 * (1 - normalized_makespan)
)
```

Range: 0-1 (higher is better)

### Tardiness Metrics

**Total Tardiness**:
```
total_tardiness = sum(max(0, end_time - lcd) for all tasks)
```

**Average Tardiness**:
```
avg_tardiness = total_tardiness / n_late_tasks
```

**Maximum Tardiness**:
```
max_tardiness = max(end_time - lcd for all late tasks)
```

### Sequence Metrics

**Sequence Completion Rate**:
```
sequence_completion = families_fully_scheduled / total_families
```

**Average Sequence Delay**:
```
avg_sequence_delay = mean(actual_gap - min_gap between sequences)
```

## Visualization

### 1. Performance Radar Chart

Shows multiple metrics on single chart:
- Completion Rate
- On-Time Rate
- Utilization
- Constraint Satisfaction
- Efficiency

### 2. Learning Curves

Track metrics over training episodes:
- Reward progression
- Completion rate improvement
- Loss convergence

### 3. Gantt Charts

Visualize scheduled tasks:
- Job view (tasks by family)
- Machine view (tasks by machine)
- Color coding by status

## Statistical Analysis

### Confidence Intervals

```python
import numpy as np
from scipy import stats

# 95% confidence interval
mean = np.mean(completion_rates)
std = np.std(completion_rates)
confidence = 0.95
n = len(completion_rates)

interval = stats.t.interval(
    confidence, n-1, 
    loc=mean, 
    scale=std/np.sqrt(n)
)
print(f"95% CI: [{interval[0]:.3f}, {interval[1]:.3f}]")
```

### Hypothesis Testing

Test if PPO significantly outperforms baseline:

```python
from scipy.stats import ttest_ind

# Two-sample t-test
t_stat, p_value = ttest_ind(ppo_results, fifo_results)

if p_value < 0.05:
    print("PPO significantly better than FIFO (p < 0.05)")
```

## Failure Analysis

### Common Failure Modes

1. **Incomplete Scheduling**
   - Cause: Insufficient episode steps
   - Solution: Increase max_steps

2. **Constraint Violations**
   - Cause: Model not learning constraints
   - Solution: Increase violation penalty

3. **Poor Utilization**
   - Cause: Conservative scheduling
   - Solution: Adjust action bonus

4. **Many Late Tasks**
   - Cause: Not prioritizing urgent jobs
   - Solution: Increase urgency weight

## Reporting

### Evaluation Report Template

```
==============================================
PPO MODEL EVALUATION REPORT
==============================================
Date: 2025-08-06
Model: checkpoints/best_model.pth
Data: data/test_jobs.json

PERFORMANCE METRICS
------------------
Completion Rate: 92.9% (118/127 tasks)
Constraint Satisfaction: 100.0%
On-Time Delivery: 87.3%
Machine Utilization: 65.2%
Makespan: 485.3 hours

COMPARISON WITH BASELINES
------------------------
vs FIFO: +9.3% completion, +21.1% on-time
vs EDD: +5.3% completion, +7.1% on-time
vs SPT: +11.3% completion, +26.6% on-time

STATISTICAL ANALYSIS
--------------------
Mean Completion (10 runs): 92.1% ± 2.3%
95% Confidence Interval: [90.5%, 93.7%]
Best Run: 95.3%
Worst Run: 88.2%

RECOMMENDATIONS
--------------
1. Model performs well on medium-scale problems
2. Consider fine-tuning for urgent job handling
3. Ready for production deployment
==============================================
```

## Best Practices

1. **Multiple Runs**: Always evaluate over multiple episodes
2. **Statistical Significance**: Use at least 10 runs for metrics
3. **Diverse Test Sets**: Test on different job patterns
4. **Compare Baselines**: Always include FIFO/EDD comparison
5. **Track Over Time**: Monitor performance degradation
6. **Document Results**: Keep detailed evaluation logs
7. **Visualize Results**: Use charts for stakeholder communication
8. **Consider Business Impact**: Weight metrics by business value
9. **Test Edge Cases**: Include stress tests and edge cases
10. **Continuous Monitoring**: Track production performance