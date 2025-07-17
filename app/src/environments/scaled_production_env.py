"""
Scaled production environment with 40 machines and 50 families.
Week 7-8: Scaling up from 10 to 40 machines while keeping all 50 families.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import gymnasium as gym
from gymnasium import spaces
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path

from .base_env import BaseSchedulingEnv
from .break_time_constraints import BreakTimeConstraints

logger = logging.getLogger(__name__)


class ScaledProductionEnv(BaseSchedulingEnv):
    """
    Scaled production environment for Week 7-8.
    
    Key features:
    - 40 machines (subset of 152 production machines)
    - 50 families from SAMPLE_50.md
    - Machine type constraints
    - Boolean importance system
    - Focus on efficient utilization of more resources
    """
    
    def __init__(self,
                 n_machines: int = 40,
                 data_file: str = None,
                 snapshot_file: str = None,
                 max_episode_steps: int = 1000,
                 max_valid_actions: int = 100,
                 use_break_constraints: bool = True,
                 seed: Optional[int] = None):
        """
        Initialize scaled production environment.
        
        Args:
            n_machines: Number of machines (default 40)
            data_file: Path to parsed production data
            snapshot_file: Path to production snapshot for machine info
            max_episode_steps: Maximum steps per episode
            max_valid_actions: Maximum valid actions to present
            use_break_constraints: Whether to apply break time constraints (default True)
            seed: Random seed
        """
        # Set default paths
        if data_file is None:
            data_file = 'data/parsed_production_data_boolean.json'
        if snapshot_file is None:
            snapshot_file = 'data/production_snapshot_latest.json'
            
        # Load production data (families/jobs)
        with open(data_file, 'r') as f:
            self.families_data = json.load(f)
        
        # Load machine data from snapshot
        with open(snapshot_file, 'r') as f:
            snapshot = json.load(f)
            
        # Select first 40 machines with diverse types
        self.machines = self._select_diverse_machines(snapshot['machines'], n_machines)
        self.machine_types = self._extract_machine_types()
        
        # Initialize base class
        n_jobs = sum(len(f['tasks']) for f in self.families_data.values())
        super().__init__(n_machines, n_jobs, seed)
        
        self.n_families = len(self.families_data)
        self.max_episode_steps = max_episode_steps
        self.max_valid_actions = max_valid_actions
        
        # Define observation space
        # [machine_loads(40), family_progress(50), family_importance(50), 
        #  family_urgency(50), next_job_ready(50), machine_utilization(40), time(1)]
        obs_dim = n_machines + 4 * self.n_families + n_machines + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action space
        self.action_space = spaces.Discrete(max_valid_actions)
        
        # Environment state
        self.machine_loads = None
        self.machine_utilization = None
        self.family_progress = None
        self.completed_tasks = None
        self.current_step = None
        self.current_time = None
        self.valid_actions = None
        self.episode_makespan = None
        
        # Machine capabilities based on type
        self.machine_capabilities = self._setup_machine_capabilities()
        
        # Setup times between different product types
        self.setup_times = self._setup_times()
        
        # Tracking metrics
        self.total_setup_time = 0
        self.total_processing_time = 0
        self.machine_idle_time = None
        
        # Parse families
        self.family_ids = list(self.families_data.keys())
        self.family_idx_map = {fid: idx for idx, fid in enumerate(self.family_ids)}
        
        # Initialize break time constraints
        self.use_break_constraints = use_break_constraints
        if self.use_break_constraints:
            self.break_constraints = BreakTimeConstraints()
        else:
            self.break_constraints = None
            
        # Base date for time conversion (start of scheduling horizon)
        self.base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
    def _select_diverse_machines(self, all_machines: List[Dict], n_machines: int) -> List[Dict]:
        """Select diverse set of machines to cover different types."""
        # Group by type
        by_type = {}
        for machine in all_machines[:100]:  # Consider first 100 machines
            type_id = machine['machine_type_id']
            if type_id not in by_type:
                by_type[type_id] = []
            by_type[type_id].append(machine)
        
        # Select machines to ensure type diversity
        selected = []
        type_ids = list(by_type.keys())
        
        # First, take one machine from each type (if possible)
        for type_id in type_ids[:n_machines]:
            if by_type[type_id]:
                selected.append(by_type[type_id][0])
                by_type[type_id].pop(0)
        
        # Fill remaining slots
        idx = 0
        while len(selected) < n_machines:
            type_id = type_ids[idx % len(type_ids)]
            if by_type[type_id]:
                selected.append(by_type[type_id][0])
                by_type[type_id].pop(0)
            idx += 1
            
            # Safety check
            if idx > 200:
                # Just fill with remaining machines
                for machine in all_machines:
                    if len(selected) >= n_machines:
                        break
                    if machine not in selected:
                        selected.append(machine)
        
        return selected[:n_machines]
    
    def _extract_machine_types(self) -> Set[int]:
        """Extract unique machine types."""
        return set(m['machine_type_id'] for m in self.machines)
    
    def _setup_machine_capabilities(self) -> Dict[int, Set[str]]:
        """
        Setup which machines can process which product types.
        In real production, this comes from database.
        For now, use heuristic based on machine types.
        """
        capabilities = {}
        
        for idx, machine in enumerate(self.machines):
            machine_type = machine['machine_type_id']
            
            # Handle None machine type
            if machine_type is None:
                # Universal machine if type is unknown
                capabilities[idx] = set(['CF', 'CP', 'CD', 'CH', 'CM'])
                continue
            
            # Heuristic: Different machine types can process different products
            # Type 1-10: Can process CF products
            # Type 11-20: Can process CP products  
            # Type 21-30: Can process CD/CH/CM products
            # Type 31+: Universal machines
            
            capable_products = set()
            
            if machine_type <= 10:
                capable_products.update(['CF'])
            elif machine_type <= 20:
                capable_products.update(['CP'])
            elif machine_type <= 30:
                capable_products.update(['CD', 'CH', 'CM'])
            else:
                # Universal machines
                capable_products.update(['CF', 'CP', 'CD', 'CH', 'CM'])
            
            # Some overlap for flexibility
            if machine_type % 5 == 0:  # Every 5th type is more flexible
                capable_products.update(['CF', 'CP'])
                
            capabilities[idx] = capable_products
            
        return capabilities
    
    def _setup_times(self) -> Dict[Tuple[str, str], float]:
        """Setup times when switching between different product types."""
        product_prefixes = ['CF', 'CP', 'CD', 'CH', 'CM']
        setup_times = {}
        
        for p1 in product_prefixes:
            for p2 in product_prefixes:
                if p1 == p2:
                    setup_times[(p1, p2)] = 0.1  # Minor setup for same type
                else:
                    # Different setup times for different transitions
                    setup_times[(p1, p2)] = 0.3 + (abs(ord(p1[0]) - ord(p2[0])) % 3) * 0.2
                    
        return setup_times
    
    def _get_product_prefix(self, product_code: str) -> str:
        """Extract product prefix (CF, CP, etc.)."""
        for prefix in ['CF', 'CP', 'CD', 'CH', 'CM']:
            if product_code.startswith(prefix):
                return prefix
        return 'XX'  # Unknown
    
    def _can_machine_process(self, machine_idx: int, product_code: str) -> bool:
        """Check if machine can process this product type."""
        prefix = self._get_product_prefix(product_code)
        return prefix in self.machine_capabilities.get(machine_idx, set())
    
    def _reset_impl(self, options: Optional[Dict] = None) -> np.ndarray:
        """Reset environment to initial state."""
        # Reset machine loads and utilization
        self.machine_loads = np.zeros(self.n_machines, dtype=np.float32)
        self.machine_utilization = np.zeros(self.n_machines, dtype=np.float32)
        self.machine_idle_time = np.zeros(self.n_machines, dtype=np.float32)
        
        # Reset family progress
        self.family_progress = np.zeros(self.n_families, dtype=np.float32)
        
        # Reset completed tasks
        self.completed_tasks = {
            fid: set() for fid in self.family_ids
        }
        
        # Track when each family's last task completed (for dependencies)
        self.family_last_completion_time = {
            fid: 0.0 for fid in self.family_ids
        }
        
        # Reset tracking
        self.current_step = 0
        self.current_time = 0.0
        self.episode_makespan = 0.0
        self.total_setup_time = 0.0
        self.total_processing_time = 0.0
        
        # Track last product on each machine for setup times
        self.last_product_on_machine = [None] * self.n_machines
        
        # Get initial valid actions
        self.valid_actions = self._get_valid_actions()
        
        return self._get_observation()
    
    def _get_valid_actions(self) -> List[Tuple[str, int, Dict]]:
        """Get valid actions respecting dependencies and machine capabilities."""
        valid = []
        
        for family_id, family_data in self.families_data.items():
            # Find next uncompleted task
            for task_idx, task in enumerate(family_data['tasks']):
                task_seq = task['sequence']
                
                # Skip if completed
                if task_seq in self.completed_tasks[family_id]:
                    continue
                
                # Check dependencies
                dependencies_met = True
                for other_task in family_data['tasks']:
                    if other_task['sequence'] < task_seq:
                        if other_task['sequence'] not in self.completed_tasks[family_id]:
                            dependencies_met = False
                            break
                
                if dependencies_met:
                    # Check if any machine can process this
                    product_code = family_data['product']
                    capable_machines = [
                        m for m in range(self.n_machines)
                        if self._can_machine_process(m, product_code)
                    ]
                    
                    if capable_machines:
                        # Add machine list to task info
                        task_with_machines = task.copy()
                        task_with_machines['capable_machines'] = capable_machines
                        valid.append((family_id, task_idx, task_with_machines))
                    break
        
        # Sort by importance and urgency combined
        # Lower days_remaining = higher priority
        valid.sort(key=lambda x: (
            0 if self.families_data[x[0]]['is_important'] else 1,  # Important first (0 < 1)
            self._calculate_urgency(x[0]),  # Then by days remaining (ascending - urgent first)
            x[2]['sequence']  # Then by sequence order
        ))
        
        return valid[:self.max_valid_actions]
    
    def _calculate_urgency(self, family_id: str) -> float:
        """Calculate urgency based on LCD date - days remaining until deadline."""
        family = self.families_data[family_id]
        
        # Check if we have the new format with lcd_days_remaining
        if 'lcd_days_remaining' in family:
            return family['lcd_days_remaining']
        
        # Fallback to old format if needed
        if 'lcd_date' in family:
            lcd_date = datetime.fromisoformat(family['lcd_date'])
            base_date = datetime(2024, 1, 15)
            days_to_lcd = (lcd_date - base_date).days
            return days_to_lcd
            
        # Default if no LCD info
        return 30.0  # Assume 30 days if not specified
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation including machine utilization."""
        # Normalize machine loads
        max_load = max(self.machine_loads) if np.any(self.machine_loads > 0) else 1.0
        normalized_loads = self.machine_loads / max_load if max_load > 0 else self.machine_loads
        
        # Family progress
        family_progress = np.zeros(self.n_families)
        for idx, family_id in enumerate(self.family_ids):
            family = self.families_data[family_id]
            total_tasks = len(family['tasks'])
            completed = len(self.completed_tasks[family_id])
            family_progress[idx] = completed / total_tasks if total_tasks > 0 else 1.0
        
        # Family importance
        family_importance = np.zeros(self.n_families)
        for idx, family_id in enumerate(self.family_ids):
            family_importance[idx] = 1.0 if self.families_data[family_id]['is_important'] else 0.0
        
        # Family urgency (normalized: 0 = not urgent, 1 = very urgent)
        family_urgency = np.zeros(self.n_families)
        for idx, family_id in enumerate(self.family_ids):
            days_remaining = self._calculate_urgency(family_id)
            # Convert to urgency score: fewer days = higher urgency
            # 0 days = 1.0 (max urgency), 30+ days = 0.0 (no urgency)
            urgency_score = max(0, 1.0 - days_remaining / 30.0)
            family_urgency[idx] = urgency_score
        
        # Next job ready
        next_job_ready = np.zeros(self.n_families)
        for valid_action in self.valid_actions:
            family_idx = self.family_idx_map[valid_action[0]]
            next_job_ready[family_idx] = 1.0
        
        # Machine utilization (how busy each machine has been)
        if self.current_time > 0:
            self.machine_utilization = self.machine_loads / self.current_time
        
        # Time progress
        time_progress = min(self.current_step / self.max_episode_steps, 1.0)
        
        # Concatenate all features
        obs = np.concatenate([
            normalized_loads,
            family_progress,
            family_importance,
            family_urgency,
            next_job_ready,
            self.machine_utilization,
            [time_progress]
        ])
        
        return obs.astype(np.float32)
    
    def _step_impl(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action in environment."""
        reward = 0.0
        info = {}
        
        # Validate action
        if action >= len(self.valid_actions):
            reward = -20.0
            info['invalid_action'] = True
            self.current_step += 1
            obs = self._get_observation()
            terminated = False
            truncated = self.current_step >= self.max_episode_steps
            return obs, reward, terminated, truncated, info
        
        # Get selected job
        family_id, task_idx, task = self.valid_actions[action]
        family = self.families_data[family_id]
        product_code = family['product']
        
        # Choose best machine from capable ones
        capable_machines = task['capable_machines']
        
        # Consider both load and setup time
        best_machine = None
        best_start_time = float('inf')
        
        for machine_idx in capable_machines:
            # Calculate setup time
            last_product = self.last_product_on_machine[machine_idx]
            if last_product is None:
                setup_time = 0.2  # Initial setup
            else:
                last_prefix = self._get_product_prefix(last_product)
                curr_prefix = self._get_product_prefix(product_code)
                setup_time = self.setup_times.get((last_prefix, curr_prefix), 0.5)
            
            # Start time must respect both machine availability AND family dependencies
            machine_available_time = self.machine_loads[machine_idx] + setup_time
            family_available_time = self.family_last_completion_time[family_id]
            
            # Job can only start when BOTH machine and previous family task are done
            start_time = max(machine_available_time, family_available_time)
            
            if start_time < best_start_time:
                best_start_time = start_time
                best_machine = machine_idx
        
        # Schedule on best machine
        # Calculate ACTUAL setup time (not including dependency wait)
        if self.last_product_on_machine[best_machine] is None:
            actual_setup_time = 0.2  # Initial setup
        else:
            last_prefix = self._get_product_prefix(self.last_product_on_machine[best_machine])
            curr_prefix = self._get_product_prefix(product_code)
            actual_setup_time = self.setup_times.get((last_prefix, curr_prefix), 0.5)
        
        processing_time = task['processing_time']
        
        # Check break constraints only if enabled
        if self.use_break_constraints and self.break_constraints:
            # Convert hours to datetime for break checking
            proposed_start_dt = self.break_constraints.hours_to_datetime(best_start_time, self.base_date)
            
            # Always find next available work window - simplify the logic
            # Jobs will naturally pause during breaks in the real simulation
            valid_start_dt = self.break_constraints._find_next_work_window(proposed_start_dt)
            best_start_time = self.break_constraints.datetime_to_hours(valid_start_dt, self.base_date)
        
        end_time = best_start_time + processing_time
        
        # For tracking, still use total wait time
        total_wait_time = best_start_time - self.machine_loads[best_machine]
        
        # Update state
        self.machine_loads[best_machine] = end_time
        self.last_product_on_machine[best_machine] = product_code
        self.completed_tasks[family_id].add(task['sequence'])
        self.family_last_completion_time[family_id] = end_time  # Update family completion time
        self.current_time = max(self.current_time, end_time)
        self.episode_makespan = max(self.episode_makespan, end_time)
        self.current_step += 1
        
        # Track metrics
        self.total_setup_time += actual_setup_time
        self.total_processing_time += processing_time
        
        # Calculate rewards
        
        # Base completion reward
        reward += 10.0
        
        # Importance bonus
        if family['is_important']:
            reward += 20.0
        
        # Efficiency rewards
        
        # 1. Machine utilization reward (encourage using all machines)
        active_machines = np.sum(self.machine_loads > 0)
        utilization_ratio = active_machines / self.n_machines
        reward += utilization_ratio * 5.0
        
        # 2. Load balancing reward
        if active_machines > 0:
            load_variance = np.var(self.machine_loads[self.machine_loads > 0])
            mean_load = np.mean(self.machine_loads[self.machine_loads > 0])
            if mean_load > 0:
                cv = np.sqrt(load_variance) / mean_load  # Coefficient of variation
                balance_reward = max(0, 1 - cv) * 10.0
                reward += balance_reward
        
        # 3. Setup time penalty (encourage batching similar products)
        if actual_setup_time > 0.3:
            setup_penalty = actual_setup_time * 2.0
            reward -= setup_penalty
        
        # Urgency bonus - the "game" element
        days_remaining = self._calculate_urgency(family_id)
        
        # Urgent job bonus: more reward for jobs closer to deadline
        if days_remaining < 7:
            # Critical urgency: 1-7 days
            urgency_bonus = (7 - days_remaining) * 3.0  # Up to 18 points for 1-day deadline
            reward += urgency_bonus
            if family['is_important']:
                reward += 5.0  # Extra bonus for important+urgent
        elif days_remaining < 14:
            # Medium urgency: 7-14 days
            reward += 5.0
        
        # Early completion bonus: reward finishing well before deadline
        # Calculate how many days from NOW until job completes
        job_completion_days = end_time / (8 * 3600)  # Total days from start (end_time is in seconds)
        if job_completion_days < days_remaining - 3:
            reward += 2.0  # Bonus for finishing with >3 days buffer
        
        # Debug huge negative rewards
        if reward < -100:
            print(f"DEBUG: Huge negative reward {reward:.1f}")
            print(f"  end_time={end_time}, days={job_completion_days:.1f}, days_remaining={days_remaining}")
            print(f"  setup_time={setup_time}")
        
        # Time penalty
        reward -= 0.1
        
        # Store info
        info['scheduled_job'] = f"{family_id}-{task['sequence']}"
        info['on_machine'] = f"{best_machine} ({self.machines[best_machine]['machine_name']})"
        info['machine_type'] = self.machines[best_machine]['machine_type_id']
        info['setup_time'] = actual_setup_time
        info['processing_time'] = processing_time
        info['is_important'] = family['is_important']
        info['lcd_days_remaining'] = days_remaining
        info['active_machines'] = active_machines
        info['utilization'] = utilization_ratio
        
        # Add break constraint info
        info['start_time'] = best_start_time
        info['end_time'] = end_time
        if total_wait_time > actual_setup_time + 0.1:  # If significant delay beyond setup
            info['break_delay'] = total_wait_time - actual_setup_time
        
        # Check completion
        total_completed = sum(len(completed) for completed in self.completed_tasks.values())
        all_done = total_completed >= self.n_jobs
        
        if all_done:
            # Episode completion bonus
            reward += 50.0
            
            # Efficiency bonus based on theoretical minimum
            total_work = sum(
                task['processing_time'] 
                for family in self.families_data.values()
                for task in family['tasks']
            )
            theoretical_min = total_work / self.n_machines
            efficiency = theoretical_min / self.episode_makespan
            reward += efficiency * 50.0
            
            # Machine utilization bonus
            avg_utilization = np.mean(self.machine_utilization)
            reward += avg_utilization * 30.0
            
            info['makespan'] = self.episode_makespan
            info['efficiency'] = efficiency
            info['avg_utilization'] = avg_utilization
            info['total_setup_time'] = self.total_setup_time
            info['setup_ratio'] = self.total_setup_time / (self.total_setup_time + self.total_processing_time)
            info['all_tasks_completed'] = True
        
        # Update valid actions
        self.valid_actions = self._get_valid_actions()
        
        # Get new observation
        obs = self._get_observation()
        
        terminated = all_done
        truncated = self.current_step >= self.max_episode_steps
        
        return obs, reward, terminated, truncated, info
    
    def get_action_mask(self) -> np.ndarray:
        """Get mask of valid actions."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[:len(self.valid_actions)] = True
        return mask
    
    def render(self, mode: str = 'human'):
        """Render current environment state."""
        if mode == 'human':
            print(f"\n=== Step {self.current_step}/{self.max_episode_steps} ===")
            print(f"Current time: {self.current_time:.1f}h")
            print(f"Active machines: {np.sum(self.machine_loads > 0)}/{self.n_machines}")
            
            # Show top 5 busiest machines
            busy_indices = np.argsort(self.machine_loads)[-5:][::-1]
            print("\nBusiest machines:")
            for idx in busy_indices:
                if self.machine_loads[idx] > 0:
                    machine = self.machines[idx]
                    print(f"  {machine['machine_name']} (Type {machine['machine_type_id']}): "
                          f"{self.machine_loads[idx]:.1f}h load")
            
            # Family progress
            print("\nFamily progress:")
            important_families = [
                fid for fid in self.family_ids 
                if self.families_data[fid]['is_important']
            ]
            for fid in important_families[:3]:
                family = self.families_data[fid]
                progress = len(self.completed_tasks[fid]) / len(family['tasks'])
                print(f"  {fid} (Important): {progress:.0%}")
            
            # Valid actions
            print(f"\nValid actions: {len(self.valid_actions)}")
            if self.valid_actions:
                print("Next 3 options:")
                for i, (fid, _, task) in enumerate(self.valid_actions[:3]):
                    family = self.families_data[fid]
                    days_remaining = self._calculate_urgency(fid)
                    imp = "❗" if family['is_important'] else "  "
                    urgency_mark = "🔴" if days_remaining < 7 else "🟡" if days_remaining < 14 else "🟢"
                    print(f"{imp}{i+1}. {fid}-{task['sequence']} "
                          f"{urgency_mark} ({days_remaining:.0f} days left, "
                          f"{len(task['capable_machines'])} machines)")
    
    def _calculate_reward(self, action: int, valid_action: bool) -> float:
        """Calculate reward (implemented in _step_impl)."""
        return 0.0
    
    def _is_done(self) -> bool:
        """Check if episode is complete."""
        total_completed = sum(len(completed) for completed in self.completed_tasks.values())
        return total_completed >= self.n_jobs or self.current_step >= self.max_episode_steps