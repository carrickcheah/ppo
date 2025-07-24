# PPO Scheduling System - Comprehensive Test Report

Generated: 2025-07-24 16:01:47

## Executive Summary

- **Total Tests:** 14
- **Passed:** 9
- **Failed:** 5
- **Success Rate:** 64.3%
- **Total Duration:** 0.32s

## Component Test Results

### Data Layer
- Tests: 2
- Passed: 1
- Failed: 1
- Success Rate: 50.0%
- Duration: 0.00s

**Failed Tests:**
- Production data loading: No jobs loaded from production snapshot

### Environment
- Tests: 4
- Passed: 2
- Failed: 2
- Success Rate: 50.0%
- Duration: 0.00s

**Failed Tests:**
- Rules engine validation: RulesEngine.is_action_valid() missing 4 required positional arguments: 'job_assignments', 'job_to_family', 'job_sequences', and 'all_jobs'
- Reward calculation: 'RewardFunction' object has no attribute 'calculate_job_reward'

### PPO Model
- Tests: 6
- Passed: 5
- Failed: 1
- Success Rate: 83.3%
- Duration: 0.31s

**Failed Tests:**
- Transformer policy: 'job_embeddings'

### Integration
- Tests: 2
- Passed: 1
- Failed: 1
- Success Rate: 50.0%
- Duration: 0.00s

**Failed Tests:**
- Data → Environment flow: No valid actions available


## System Readiness Assessment

❌ **SYSTEM NOT READY**

Significant issues detected. Fix failing tests before proceeding.

## Critical Component Status

❌ **Data Layer**: Data loading and processing
   - Success Rate: 50.0%
❌ **Environment**: Game rules and state management
   - Success Rate: 50.0%
❌ **PPO Model**: Neural network and training logic
   - Success Rate: 83.3%
❌ **Integration**: Component communication
   - Success Rate: 50.0%

## Detailed Error Log

### Data Layer - Production data loading
**Error:** No jobs loaded from production snapshot
```
Traceback (most recent call last):
  File "/Users/carrickcheah/Project/ppo/app_2/phase2/test_result/comprehensive_test.py", line 60, in run
    test_func()
  File "/Users/carrickcheah/Project/ppo/app_2/phase2/test_result/comprehensive_test.py", line 142, in test_production_data
    assert len(jobs) > 0, "No jobs loaded from production snapshot"
           ^^^^^^^^^^^^^
AssertionError: No jobs loaded from production snapshot
```

### Environment - Rules engine validation
**Error:** RulesEngine.is_action_valid() missing 4 required positional arguments: 'job_assignments', 'job_to_family', 'job_sequences', and 'all_jobs'
```
Traceback (most recent call last):
  File "/Users/carrickcheah/Project/ppo/app_2/phase2/test_result/comprehensive_test.py", line 60, in run
    test_func()
  File "/Users/carrickcheah/Project/ppo/app_2/phase2/test_result/comprehensive_test.py", line 219, in test_rules_engine
    valid, _ = rules.is_action_valid(job, machine, 0, state)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: RulesEngine.is_action_valid() missing 4 required positional arguments: 'job_assignments', 'job_to_family', 'job_sequences', and 'all_jobs'
```

### Environment - Reward calculation
**Error:** 'RewardFunction' object has no attribute 'calculate_job_reward'
```
Traceback (most recent call last):
  File "/Users/carrickcheah/Project/ppo/app_2/phase2/test_result/comprehensive_test.py", line 60, in run
    test_func()
  File "/Users/carrickcheah/Project/ppo/app_2/phase2/test_result/comprehensive_test.py", line 239, in test_reward_function
    reward = reward_fn.calculate_job_reward(job, 0, 2)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'RewardFunction' object has no attribute 'calculate_job_reward'. Did you mean: 'calculate_step_reward'?
```

### PPO Model - Transformer policy
**Error:** 'job_embeddings'
```
Traceback (most recent call last):
  File "/Users/carrickcheah/Project/ppo/app_2/phase2/test_result/comprehensive_test.py", line 60, in run
    test_func()
  File "/Users/carrickcheah/Project/ppo/app_2/phase2/test_result/comprehensive_test.py", line 303, in test_transformer_policy
    logits, values = policy(encoded_state, action_mask)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/carrickcheah/Project/ppo/app_2/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/carrickcheah/Project/ppo/app_2/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/carrickcheah/Project/ppo/app_2/phase2/transformer_policy.py", line 207, in forward
    job_embeddings = encoded_state['job_embeddings']
                     ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
KeyError: 'job_embeddings'
```

### Integration - Data → Environment flow
**Error:** No valid actions available
```
Traceback (most recent call last):
  File "/Users/carrickcheah/Project/ppo/app_2/phase2/test_result/comprehensive_test.py", line 60, in run
    test_func()
  File "/Users/carrickcheah/Project/ppo/app_2/phase2/test_result/comprehensive_test.py", line 405, in test_data_to_env
    assert np.any(mask), "No valid actions available"
           ^^^^^^^^^^^^
AssertionError: No valid actions available
```
