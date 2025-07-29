"""
Small Complex Environment
50 jobs, 25 machines - Tests handling of complex dependencies and multi-machine jobs
"""

import os
from typing import Dict, Optional, List, Set, Tuple
from .base_strategy_env import BaseStrategyEnvironment


class SmallComplexEnvironment(BaseStrategyEnvironment):
    """
    Complex scenario with job dependencies and multi-machine requirements.
    Tests handling of complex constraints and coordination.
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize Small Complex environment."""
        
        # Path to data file
        data_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data',
            'small_complex_data.json'
        )
        
        # Complex-optimized reward configuration
        reward_config = {
            'no_action_penalty': -1.0,
            'invalid_action_penalty': -15.0,  # Higher - complex constraints
            'valid_schedule_reward': 20.0,
            'sequence_completion': 40.0,      # Higher - sequences matter more
            'family_completion': 100.0,       # Very high - completing complex jobs
            'on_time_bonus': 25.0,
            'late_penalty_per_day': -3.0,
            'multi_machine_bonus': 15.0,      # Bonus for multi-machine coordination
            'dependency_bonus': 10.0          # Bonus for respecting dependencies
        }
        
        super().__init__(
            scenario_name='small_complex',
            data_file=data_file,
            reward_config=reward_config,
            max_steps=300,
            verbose=verbose
        )
        
        # Complex-specific tracking
        self.multi_machine_jobs_scheduled = 0
        self.dependency_violations = 0
        self.completed_families_with_deps = 0
        self._identify_complex_jobs()
    
    def _identify_complex_jobs(self):
        """Identify jobs with complex requirements."""
        self.multi_machine_tasks = set()
        self.families_with_dependencies = set()
        
        for fid, family in self.families.items():
            has_multi_machine = False
            
            for i, task in enumerate(family.get('tasks', [])):
                task_key = f"{fid}_seq{i+1}"
                
                # Check for multi-machine requirements
                if task.get('multi_machine', False) or len(task.get('required_machines', [])) > 1:
                    self.multi_machine_tasks.add(task_key)
                    has_multi_machine = True
                
                # Check for dependencies (more than 3 sequences)
                if family.get('total_sequences', 1) > 3:
                    self.families_with_dependencies.add(fid)
            
            if has_multi_machine:
                self.families_with_dependencies.add(fid)
    
    def _validate_complex_action(self, family_id: str, machine_id: int) -> Tuple[bool, str]:
        """Validate action considering complex constraints."""
        # First do basic validation
        is_valid, reason = self._validate_action(family_id, machine_id) if hasattr(self, '_validate_action') else (True, "")
        
        if not is_valid:
            return False, reason
        
        # Check multi-machine requirements
        family = self.families[family_id]
        progress = self.family_progress[family_id]
        next_seq = progress['next_sequence']
        
        if next_seq <= len(family.get('tasks', [])):
            task = family['tasks'][next_seq - 1]
            
            # Check if this is a multi-machine task
            if task.get('multi_machine', False):
                required_machines = task.get('required_machines', [])
                if required_machines and machine_id not in required_machines:
                    return False, "Machine not in required set for multi-machine job"
                
                # Check if all required machines are available
                current_time = self._get_machine_available_time(machine_id)
                for req_machine in required_machines:
                    if req_machine != machine_id:
                        req_time = self._get_machine_available_time(req_machine)
                        if abs(req_time - current_time) > 0.1:  # Not synchronized
                            return False, "Required machines not synchronized"
        
        return True, ""
    
    def _schedule_complex_job(self, family_id: str, machine_id: int, task: Dict) -> float:
        """Schedule a complex job, handling multi-machine requirements."""
        base_reward = self.reward_config['valid_schedule_reward']
        
        # Multi-machine scheduling
        if task.get('multi_machine', False):
            required_machines = task.get('required_machines', [machine_id])
            
            # Schedule on all required machines
            start_time = max(
                self._get_machine_available_time(m) 
                for m in required_machines
            )
            
            processing_time = task['processing_time']
            end_time = start_time + processing_time
            
            job_key = f"{family_id}_seq{task['sequence']}"
            
            # Schedule on all machines
            for m in required_machines:
                self.machine_schedules[m].append({
                    'job': job_key,
                    'start': start_time,
                    'end': end_time
                })
            
            self.job_assignments[job_key] = {
                'machines': required_machines,
                'start': start_time,
                'end': end_time
            }
            
            self.multi_machine_jobs_scheduled += 1
            base_reward += self.reward_config.get('multi_machine_bonus', 15.0)
            
            return base_reward
        
        # Regular single-machine scheduling
        return base_reward
    
    def step(self, action):
        """Override step to handle complex scheduling."""
        # Check if this is a complex job before standard step
        job_idx, machine_idx = action
        
        if job_idx < len(self.family_ids) and machine_idx < len(self.machine_ids):
            family_id = self.family_ids[job_idx]
            machine_id = self.machine_ids[machine_idx]
            
            # Validate complex constraints
            is_valid, reason = self._validate_complex_action(family_id, machine_id)
            
            if not is_valid:
                # Return invalid action result
                obs = self._get_observation()
                reward = self.reward_config['invalid_action_penalty']
                info = {
                    'action_valid': False,
                    'action_type': 'invalid',
                    'reason': reason
                }
                self.steps += 1
                done = self.steps >= self.max_steps
                return obs, reward, done, False, info
        
        # Proceed with standard step
        obs, reward, done, truncated, info = super().step(action)
        
        # Add complex-specific rewards
        if info.get('action_valid') and info.get('scheduled_job'):
            job_family = '_'.join(info['scheduled_job'].split('_')[:-1])
            
            # Dependency bonus
            if job_family in self.families_with_dependencies:
                reward += self.reward_config.get('dependency_bonus', 10.0)
                info['dependency_respected'] = True
            
            # Check if completed a complex family
            if job_family in self.completed_jobs:
                if job_family in self.families_with_dependencies:
                    self.completed_families_with_deps += 1
        
        return obs, reward, done, truncated, info
    
    def _get_info(self) -> Dict:
        """Get environment info with complex metrics."""
        info = super()._get_info() if hasattr(super(), '_get_info') else {}
        
        # Complex completion rate
        complex_families = len(self.families_with_dependencies)
        complex_completion = (
            self.completed_families_with_deps / complex_families 
            if complex_families > 0 else 1.0
        )
        
        info.update({
            'scenario': 'small_complex',
            'scheduled_jobs': len(self.scheduled_jobs),
            'total_tasks': self.total_tasks,
            'completion_rate': len(self.scheduled_jobs) / self.total_tasks if self.total_tasks > 0 else 0,
            'multi_machine_jobs': len(self.multi_machine_tasks),
            'multi_machine_scheduled': self.multi_machine_jobs_scheduled,
            'complex_families': complex_families,
            'complex_completed': self.completed_families_with_deps,
            'complex_completion_rate': complex_completion,
            'avg_sequence_length': sum(f.get('total_sequences', 1) for f in self.families.values()) / len(self.families)
        })
        
        return info