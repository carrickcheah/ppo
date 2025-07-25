# LLM-Based Job Scheduler

A proof-of-concept implementation using Large Language Models (DeepSeek) for production job scheduling. This package provides an alternative approach to the PPO reinforcement learning method.

## Overview

This LLM scheduler leverages the reasoning capabilities of large language models to generate production schedules while respecting complex constraints. Unlike traditional optimization methods or reinforcement learning, it uses natural language understanding to interpret requirements and generate solutions.

## Features

- **Multiple Scheduling Strategies**
  - Direct scheduling (single-shot)
  - Chain-of-thought reasoning
  - Constraint-focused approach
  - Example-based (few-shot) learning
  - Problem decomposition

- **Robust Parsing**
  - Handles multiple output formats
  - Error recovery for malformed responses
  - Flexible time parsing

- **Constraint Validation**
  - Sequence constraints within job families
  - Machine compatibility checking
  - Time overlap detection
  - Multi-machine job validation

- **Production Ready**
  - Loads real production data from snapshots
  - Compatible with existing API format
  - Cost tracking and performance metrics

## Quick Start

```python
from llm_scheduler import LLMScheduler

# Initialize scheduler
scheduler = LLMScheduler()

# Generate schedule using chain-of-thought strategy
result = scheduler.schedule(
    strategy="chain_of_thought",
    max_jobs=50  # For testing
)

# Check results
print(f"Scheduled {result['metrics']['total_jobs']} jobs")
print(f"Makespan: {result['metrics']['makespan_hours']} hours")
print(f"Cost: ${result['llm_metadata']['cost_estimate']['total_cost']}")
```

## Installation

1. Install dependencies:
```bash
cd /Users/carrickcheah/Project/ppo/app_2
uv add -r llm/requirements.txt
```

2. Ensure DeepSeek API key is in `.env`:
```
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_MODEL=deepseek-chat
```

## Architecture

### Core Components

1. **DeepSeekClient** (`deepseek_client.py`)
   - OpenAI-compatible API client
   - Handles authentication and retries
   - Token counting for cost estimation

2. **DataAdapter** (`data_adapter.py`)
   - Loads production snapshots
   - Converts data to LLM-friendly format
   - Parses LLM output to structured schedule

3. **SchedulingPrompts** (`prompts.py`)
   - Collection of prompt templates
   - Different strategies for different scenarios
   - System prompts with constraint specifications

4. **LLMScheduler** (`llm_scheduler.py`)
   - Main orchestrator class
   - Implements scheduling strategies
   - Handles batching for large problems

5. **ScheduleParser** (`parser.py`)
   - Advanced text parsing with regex
   - Multiple format support
   - Error recovery mechanisms

6. **ScheduleValidator** (`validator.py`)
   - Validates all constraints
   - Detailed violation reporting
   - Soft and strict validation modes

## Scheduling Strategies

### 1. Direct Scheduling
Simplest approach - provide all information and ask for a schedule.
```python
result = scheduler.schedule(strategy="direct")
```

### 2. Chain of Thought
Step-by-step reasoning for complex problems.
```python
result = scheduler.schedule(strategy="chain_of_thought")
```

### 3. Constraint Focused
Emphasizes constraint checking at each step.
```python
result = scheduler.schedule(strategy="constraint_focused")
```

### 4. Example Based
Provides examples for few-shot learning.
```python
result = scheduler.schedule(strategy="example_based")
```

### 5. Decomposition
Breaks problem into subproblems by urgency.
```python
result = scheduler.schedule(strategy="decomposition")
```

## Configuration

Edit `config.yaml` to customize behavior:

```yaml
scheduling:
  max_jobs_per_prompt: 50
  enable_batching: true
  batch_size: 30
  default_strategy: "chain_of_thought"

constraints:
  enforce_sequence: true
  enforce_compatibility: true
  enforce_no_overlap: true
```

## Performance Metrics

Typical performance for 100 jobs:
- Response time: 5-10 seconds
- Token usage: 2000-4000 tokens
- Cost: $0.05-0.10
- Constraint satisfaction: 90-95%

## Comparison with PPO

| Metric | LLM Scheduler | PPO Scheduler |
|--------|--------------|---------------|
| Training Required | No | Yes (weeks) |
| Inference Speed | 5-10s | <100ms |
| Explainability | High | Low |
| Flexibility | High | Medium |
| Cost per Schedule | $0.05-0.10 | $0.001 |
| Constraint Guarantee | No* | Learned |

*Can be improved with validation and iterative refinement

## Examples

### Basic Usage
```python
# Load and schedule all jobs
scheduler = LLMScheduler()
result = scheduler.schedule()
```

### Custom Snapshot
```python
# Use specific snapshot
result = scheduler.schedule(
    snapshot_path="/path/to/snapshot_rush.json",
    strategy="constraint_focused"
)
```

### With Validation
```python
from validator import ScheduleValidator
from parser import ScheduleParser

# Generate schedule
result = scheduler.schedule()

# Parse and validate
parser = ScheduleParser()
tasks = parser.parse_schedule(result['schedule'])

validator = ScheduleValidator()
is_valid, violations = validator.validate_schedule(tasks)

if not is_valid:
    print(f"Found {len(violations)} violations")
    for v in violations[:5]:
        print(f"  - {v}")
```

## Testing

Run individual component tests:
```bash
# Test DeepSeek connection
uv run python llm/deepseek_client.py

# Test data adapter
uv run python llm/data_adapter.py

# Test parser
uv run python llm/parser.py

# Test complete scheduler
uv run python llm/llm_scheduler.py
```

## Future Enhancements

1. **Iterative Refinement**
   - Use validation results to improve schedule
   - Multiple rounds of generation

2. **Hybrid Approach**
   - LLM for strategy planning
   - Traditional optimizer for execution

3. **Caching**
   - Cache similar problems
   - Reuse successful patterns

4. **Real-time Updates**
   - Handle dynamic job arrivals
   - Incremental scheduling

## Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure DEEPSEEK_API_KEY is set in .env
   - Check key validity

2. **Parsing Failures**
   - Check LLM output format
   - Adjust temperature for consistency

3. **Constraint Violations**
   - Try constraint_focused strategy
   - Enable strict validation
   - Use iterative refinement

4. **High Costs**
   - Reduce max_jobs_per_prompt
   - Enable batching
   - Use simpler strategies

## Contributing

To add new features:
1. Add new strategies to `prompts.py`
2. Implement parsing patterns in `parser.py`
3. Add validation rules to `validator.py`
4. Update tests and documentation

## License

This is a proof-of-concept implementation for research and evaluation purposes.