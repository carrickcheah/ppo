"""
Full production environment with 152 machines and 500+ jobs.
Phase 4: Scaling to full production capacity for final validation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import gymnasium as gym
from gymnasium import spaces
import json
from datetime import datetime, timedelta, date
import logging
from pathlib import Path

from .scaled_production_env import ScaledProductionEnv
from .break_time_constraints import BreakTimeConstraints

logger = logging.getLogger(__name__)


class FullProductionEnv(ScaledProductionEnv):
    """
    Full production environment for Phase 4 scale testing.
    
    Key features:
    - 152 machines (full production capacity)
    - 500+ jobs across 100+ families
    - All production constraints (breaks, holidays, machine types)
    - Optimized for large-scale scheduling
    """
    
    def __init__(self,
                 n_machines: int = 152,
                 n_jobs: int = 500,
                 data_file: str = None,
                 snapshot_file: str = None,
                 max_episode_steps: int = 2000,
                 max_valid_actions: int = 200,
                 use_break_constraints: bool = True,
                 use_holiday_constraints: bool = True,
                 seed: Optional[int] = None,
                 state_compression: str = "hierarchical"):
        """
        Initialize full production environment.
        
        Args:
            n_machines: Number of machines (default 152 for full production)
            n_jobs: Target number of jobs to generate (default 500)
            data_file: Path to parsed production data
            snapshot_file: Path to production snapshot for machine info
            max_episode_steps: Maximum steps per episode
            max_valid_actions: Maximum valid actions to present
            use_break_constraints: Whether to apply break time constraints
            use_holiday_constraints: Whether to apply holiday constraints
            seed: Random seed for reproducibility
            state_compression: State representation method ("full", "hierarchical", "compressed")
        """
        self.n_jobs = n_jobs
        self.state_compression = state_compression
        self.use_holiday_constraints = use_holiday_constraints
        self.data_file = data_file
        self._seed = seed  # Store seed value, don't override seed method
        
        # Initialize base environment with full machine count
        super().__init__(
            n_machines=n_machines,
            data_file=data_file,
            snapshot_file=snapshot_file,
            max_episode_steps=max_episode_steps,
            max_valid_actions=max_valid_actions,
            use_break_constraints=use_break_constraints,
            seed=seed
        )
        
        # Ensure we have jobs loaded
        if not hasattr(self, 'jobs') or not self.jobs:
            logger.warning("No jobs loaded from parent class, generating production data")
            self._load_data()
            # Initialize job structures if parent didn't
            if not hasattr(self, 'jobs'):
                self.jobs = []
                self.families = []
                if hasattr(self, 'data') and self.data:
                    self._process_loaded_data()
        
        # Update observation space based on compression method
        self._update_observation_space()
        
        logger.info(f"Initialized FullProductionEnv with {n_machines} machines, {len(self.jobs) if hasattr(self, 'jobs') else 0} jobs loaded")
        
    def _select_diverse_machines(self, all_machines: List[Dict], n_machines: int) -> List[Dict]:
        """Override to handle full 152 machine set."""
        # For full production, we want ALL machines (up to n_machines)
        selected = []
        
        for machine in all_machines[:n_machines]:
            # Ensure machine_type_id is valid
            if 'machine_type_id' in machine and machine['machine_type_id'] is not None:
                selected.append(machine)
            else:
                logger.warning(f"Skipping machine {machine.get('machine_id', 'unknown')} with invalid type")
                
        # If we don't have enough, fill with synthetic machines
        while len(selected) < n_machines:
            machine_id = len(selected) + 1
            selected.append({
                'machine_id': machine_id,
                'machine_name': f'M{machine_id:03d}',
                'machine_type_id': (machine_id % 10) + 1  # Distribute across types 1-10
            })
            
        return selected[:n_machines]
        
    def _load_data(self):
        """Override to load larger dataset for full production scale."""
        if self.data_file is None:
            self.data_file = Path(__file__).parent.parent.parent / "data" / "full_production_data.json"
            
        # Check if full production data exists, otherwise generate it
        if not self.data_file.exists():
            logger.info(f"Full production data not found, generating {self.n_jobs} jobs...")
            self._generate_full_production_data()
            
        # Load the data
        with open(self.data_file, 'r') as f:
            self.data = json.load(f)
            
        # Ensure we have enough jobs
        total_jobs = sum(len(family['jobs']) for family in self.data['families'])
        if total_jobs < self.n_jobs:
            logger.warning(f"Only {total_jobs} jobs in data, less than target {self.n_jobs}")
            
        logger.info(f"Loaded {len(self.data['families'])} families with {total_jobs} total jobs")
        
    def _generate_full_production_data(self):
        """Generate realistic production data at scale."""
        import random
        from datetime import datetime, timedelta
        
        # Set seed for reproducibility
        if self._seed is not None:
            random.seed(self._seed)
            
        families = []
        job_counter = 0
        family_id = 1
        
        # Product type distribution (based on production patterns)
        product_types = [
            ("CF", 0.15, 1),  # 15% high priority
            ("CH", 0.35, 2),  # 35% medium-high priority  
            ("CM", 0.35, 3),  # 35% medium priority
            ("CP", 0.15, 4),  # 15% low priority
        ]
        
        # Generate families until we have enough jobs
        while job_counter < self.n_jobs:
            # Select product type based on distribution
            rand_val = random.random()
            cumulative = 0
            for prefix, prob, priority in product_types:
                cumulative += prob
                if rand_val <= cumulative:
                    selected_prefix = prefix
                    selected_priority = priority
                    break
                    
            # Generate family
            family_name = f"{selected_prefix}{random.randint(10,99)}-{random.randint(1,999):03d}"
            
            # Number of jobs in family (1-8, weighted towards 3-4)
            n_jobs_in_family = random.choices([1,2,3,4,5,6,7,8], 
                                            weights=[5,10,25,25,15,10,5,5])[0]
            
            # Don't exceed target
            if job_counter + n_jobs_in_family > self.n_jobs * 1.1:  # Allow 10% overflow
                n_jobs_in_family = min(n_jobs_in_family, self.n_jobs - job_counter)
                
            jobs = []
            for seq in range(1, n_jobs_in_family + 1):
                # Processing time (hours) - realistic distribution
                processing_time = random.choices(
                    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0],
                    weights=[5, 15, 20, 20, 15, 10, 8, 4, 2, 1]
                )[0]
                
                # Setup time - correlated with processing time
                setup_time = round(processing_time * random.uniform(0.1, 0.3), 1)
                
                # LCD date (1-30 days from now, important jobs sooner)
                if selected_priority == 1:
                    days_until_due = random.randint(1, 7)
                elif selected_priority == 2:
                    days_until_due = random.randint(3, 14)
                else:
                    days_until_due = random.randint(7, 30)
                    
                lcd_date = (datetime.now() + timedelta(days=days_until_due)).strftime("%Y-%m-%d")
                
                # Machine type (1-10, some jobs restricted)
                if random.random() < 0.3:  # 30% have restrictions
                    allowed_types = random.sample(range(1, 11), k=random.randint(3, 7))
                else:
                    allowed_types = list(range(1, 11))
                    
                job = {
                    "WorkorderLotId_v": f"JOAW{random.randint(20000,30000):05d}",
                    "FamilyId_v": f"{family_name}",
                    "Spec_v": f"{family_name}-{seq}/{n_jobs_in_family}",
                    "Qty_i": random.randint(100, 10000),
                    "LCDdate_d": lcd_date,
                    "MachinetypeId_i": allowed_types,
                    "StandardTime_f": processing_time,
                    "SetupTime_f": setup_time,
                    "Seq_i": seq,
                    "Priority_i": selected_priority,
                    "IsImportant_b": selected_priority <= 2  # CF and CH are important
                }
                jobs.append(job)
                job_counter += 1
                
            family = {
                "family_id": f"FAM_{family_id:04d}",
                "family_name": family_name,
                "priority": selected_priority,
                "total_jobs": len(jobs),
                "jobs": jobs
            }
            families.append(family)
            family_id += 1
            
        # Save the generated data
        data = {
            "metadata": {
                "generated_date": datetime.now().isoformat(),
                "total_families": len(families),
                "total_jobs": job_counter,
                "target_jobs": self.n_jobs,
                "seed": self._seed
            },
            "families": families
        }
        
        # Create data directory if needed
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Generated {len(families)} families with {job_counter} jobs")
        
    def _process_loaded_data(self):
        """Process loaded data into jobs and families structures."""
        if not hasattr(self, 'data') or not self.data:
            return
            
        self.jobs = []
        self.families = []
        
        job_id = 0
        for family_data in self.data.get('families', []):
            family = {
                'family_id': family_data['family_id'],
                'family_name': family_data['family_name'],
                'priority': family_data.get('priority', 3),
                'total_jobs': family_data['total_jobs'],
                'jobs': []
            }
            
            for job_data in family_data.get('jobs', []):
                job = {
                    'job_id': job_id,
                    'family_id': family_data['family_id'],
                    'workorder': job_data.get('WorkorderLotId_v', f"WO{job_id}"),
                    'spec': job_data.get('Spec_v', ''),
                    'sequence': job_data.get('Seq_i', 1),
                    'processing_time': job_data.get('StandardTime_f', 1.0),
                    'setup_time': job_data.get('SetupTime_f', 0.5),
                    'priority': job_data.get('Priority_i', 3),
                    'is_important': job_data.get('IsImportant_b', False),
                    'lcd_date': datetime.strptime(job_data.get('LCDdate_d', '2025-12-31'), '%Y-%m-%d').date(),
                    'allowed_machine_types': job_data.get('MachinetypeId_i', list(range(1, 11)))
                }
                self.jobs.append(job)
                family['jobs'].append(job_id)
                job_id += 1
                
            self.families.append(family)
            
        logger.info(f"Processed {len(self.families)} families with {len(self.jobs)} jobs")
        
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation with efficient state representation for scale.
        """
        if self.state_compression == "hierarchical":
            return self._get_hierarchical_observation()
        elif self.state_compression == "compressed":
            return self._get_compressed_observation()
        else:
            return super()._get_observation()
            
    def _get_hierarchical_observation(self) -> np.ndarray:
        """
        Hierarchical state representation for large-scale scheduling.
        Groups machines and families for more efficient representation.
        """
        # Ensure machine_loads is initialized
        if not hasattr(self, 'machine_loads') or self.machine_loads is None:
            self.machine_loads = np.zeros(self.n_machines, dtype=np.float32)
        if not hasattr(self, 'current_time') or self.current_time is None:
            self.current_time = 0.0
        if not hasattr(self, 'current_step') or self.current_step is None:
            self.current_step = 0
            
        # Machine type aggregation (cap at 10 types for fixed observation size)
        # Types > 10 will be grouped into type 10
        machine_type_loads = np.zeros(10)
        machine_type_available = np.zeros(10)
        
        for i, machine in enumerate(self.machines):
            type_idx = min(machine['machine_type_id'] - 1, 9)  # Cap at index 9 (type 10)
            machine_type_loads[type_idx] += self.machine_loads[i]
            if self.machine_loads[i] <= self.current_time:  # Machine is available
                machine_type_available[type_idx] += 1
                
        # Normalize
        max_load = max(self.machine_loads) if np.any(self.machine_loads > 0) else 1.0
        machine_type_loads = machine_type_loads / (max_load * 10 + 1e-6)  # Divide by max_load * 10 types
        machine_type_available = machine_type_available / (self.n_machines + 1e-6)
        
        # Family progress aggregation by priority
        priority_progress = np.zeros(5)  # Priorities 1-5
        priority_remaining = np.zeros(5)
        priority_urgency = np.zeros(5)
        
        # Get current date (approximation based on time)
        from datetime import datetime, timedelta
        base_date = datetime.now().date()
        current_date = base_date + timedelta(hours=self.current_time)
        
        for family in self.families:
            priority_idx = family['priority'] - 1
            
            # Calculate progress based on completed jobs
            family_id = family['family_id']
            if hasattr(self, 'completed_tasks') and family_id in self.completed_tasks:
                completed_count = len(self.completed_tasks[family_id])
            else:
                completed_count = 0
            
            progress = completed_count / family['total_jobs'] if family['total_jobs'] > 0 else 0
            priority_progress[priority_idx] += progress
            priority_remaining[priority_idx] += (family['total_jobs'] - completed_count)
            
            # Average urgency of remaining jobs
            remaining_urgency = []
            for job_id in family['jobs']:
                if hasattr(self, 'completed_tasks') and family_id in self.completed_tasks:
                    job = self.jobs[job_id]
                    # Check if job is not completed
                    if job['sequence'] not in self.completed_tasks[family_id]:
                        days_until_due = (job['lcd_date'] - current_date).days
                        urgency = 1.0 / (days_until_due + 1)
                        remaining_urgency.append(urgency)
            if remaining_urgency:
                priority_urgency[priority_idx] += np.mean(remaining_urgency)
                
        # Normalize
        n_families_by_priority = [sum(1 for f in self.families if f['priority'] == p) for p in range(1, 6)]
        for i in range(5):
            if n_families_by_priority[i] > 0:
                priority_progress[i] /= n_families_by_priority[i]
                priority_urgency[i] /= n_families_by_priority[i]
        priority_remaining = priority_remaining / (sum(priority_remaining) + 1e-6)
        
        # Global statistics
        total_completed = sum(len(completed) for completed in self.completed_tasks.values()) if hasattr(self, 'completed_tasks') else 0
        total_remaining = len(self.jobs) - total_completed if self.jobs else 0
        completion_rate = (total_completed / len(self.jobs)) if self.jobs else 0
        max_load = max(self.machine_loads) if np.any(self.machine_loads > 0) else 1.0
        avg_machine_load = np.mean(self.machine_loads) / max_load
        time_progress = self.current_step / self.max_episode_steps
        
        # Next jobs preview (top 10 by priority/urgency)
        next_jobs_features = np.zeros(20)  # 10 jobs x 2 features (processing time, urgency)
        
        # Get available jobs from valid_actions if available
        if hasattr(self, 'valid_actions') and self.valid_actions:
            # Extract job info from valid actions
            job_previews = []
            for action in self.valid_actions[:20]:  # Look at more to get variety
                family_id, machine_idx, task = action
                # Find the job
                for job_id, job in enumerate(self.jobs):
                    if job['family_id'] == family_id and job['sequence'] == task['sequence']:
                        days_until_due = (job['lcd_date'] - current_date).days
                        job_previews.append((job, days_until_due))
                        break
            
            # Sort by priority then urgency
            job_previews.sort(key=lambda x: (x[0]['priority'], -1.0 / (x[1] + 1)))
            
            # Fill features
            for i, (job, days_until_due) in enumerate(job_previews[:10]):
                next_jobs_features[i*2] = job['processing_time'] / 10.0  # Normalize
                next_jobs_features[i*2 + 1] = 1.0 / (days_until_due + 1)
                
        # Combine all features
        obs = np.concatenate([
            machine_type_loads,          # 10
            machine_type_available,      # 10
            priority_progress,           # 5
            priority_remaining,          # 5
            priority_urgency,           # 5
            next_jobs_features,         # 20
            [total_remaining / 500.0,   # 1 (normalized by expected job count)
             completion_rate,           # 1
             avg_machine_load,         # 1
             time_progress,            # 1
             self.current_step / self.max_episode_steps]  # 1
        ])
        
        return obs.astype(np.float32)
        
    def _get_compressed_observation(self) -> np.ndarray:
        """
        Compressed state representation using summary statistics.
        More aggressive compression for very large scale.
        """
        # Ensure machine_loads is initialized
        if not hasattr(self, 'machine_loads') or self.machine_loads is None:
            self.machine_loads = np.zeros(self.n_machines, dtype=np.float32)
        if not hasattr(self, 'current_time') or self.current_time is None:
            self.current_time = 0.0
        if not hasattr(self, 'current_step') or self.current_step is None:
            self.current_step = 0
            
        # Machine statistics (5 features)
        machine_loads = self.machine_loads.tolist()
        max_load = max(machine_loads) if max(machine_loads) > 0 else 1.0
        machine_stats = [
            np.mean(machine_loads) / max_load,
            np.std(machine_loads) / max_load,
            np.min(machine_loads) / max_load,
            np.max(machine_loads) / max_load,
            sum(1 for i in range(self.n_machines) if self.machine_loads[i] <= self.current_time) / self.n_machines
        ]
        
        # Job statistics by priority (4 priorities x 3 features = 12)
        job_stats = []
        
        # Get current date
        from datetime import datetime, timedelta
        base_date = datetime.now().date()
        current_date = base_date + timedelta(hours=self.current_time)
        
        for priority in range(1, 5):
            # Find remaining jobs of this priority
            priority_jobs = []
            for job_id, job in enumerate(self.jobs):
                if job['priority'] == priority:
                    # Check if job is not completed
                    family_id = job['family_id']
                    if hasattr(self, 'completed_tasks') and family_id in self.completed_tasks:
                        if job['sequence'] not in self.completed_tasks[family_id]:
                            priority_jobs.append(job)
                    else:
                        priority_jobs.append(job)
            
            if priority_jobs:
                processing_times = [j['processing_time'] for j in priority_jobs]
                urgencies = [1.0 / ((j['lcd_date'] - current_date).days + 1) 
                           for j in priority_jobs]
                job_stats.extend([
                    len(priority_jobs) / len(self.jobs),
                    np.mean(processing_times) / 10.0,
                    np.mean(urgencies)
                ])
            else:
                job_stats.extend([0, 0, 0])
                
        # Global progress (3 features)
        total_completed = sum(len(completed) for completed in self.completed_tasks.values()) if hasattr(self, 'completed_tasks') else 0
        global_stats = [
            total_completed / len(self.jobs) if self.jobs else 0,
            self.current_step / self.max_episode_steps,
            self.current_time / (max_load + 1e-6)  # Normalized current time
        ]
        
        obs = np.array(machine_stats + job_stats + global_stats, dtype=np.float32)
        return obs
        
    def _update_observation_space(self):
        """Update observation space based on compression method."""
        if self.state_compression == "hierarchical":
            # 10 + 10 + 5 + 5 + 5 + 20 + 5 = 60 features
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(60,), dtype=np.float32
            )
        elif self.state_compression == "compressed":
            # 5 + 12 + 3 = 20 features
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(20,), dtype=np.float32
            )
        else:
            super()._update_observation_space()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for large-scale performance analysis."""
        # Check if parent has get_stats method
        if hasattr(super(), 'get_stats'):
            stats = super().get_stats()
        else:
            # Basic stats if parent doesn't have the method
            stats = {
                'episode_reward': getattr(self, 'current_episode_reward', 0),
                'episode_length': getattr(self, 'current_episode_length', 0),
                'current_time': getattr(self, 'current_time', 0),
                'makespan': getattr(self, 'episode_makespan', 0)
            }
        
        # Add scale-specific metrics
        stats['scale_metrics'] = {
            'total_machines': self.n_machines,
            'total_jobs': len(self.jobs),
            'total_families': len(self.families),
            'jobs_per_machine': len(self.jobs) / self.n_machines,
            'avg_family_size': np.mean([f['total_jobs'] for f in self.families]),
            'state_compression': self.state_compression,
            'observation_dim': self.observation_space.shape[0]
        }
        
        # Machine type utilization
        type_utilization = {}
        for i, machine in enumerate(self.machines):
            machine_type = machine['machine_type_id']
            if machine_type not in type_utilization:
                type_utilization[machine_type] = []
            type_utilization[machine_type].append(self.machine_loads[i] / self.current_time 
                                                 if self.current_time > 0 else 0)
                                                 
        stats['machine_type_utilization'] = {
            f"type_{t}": np.mean(utils) for t, utils in type_utilization.items()
        }
        
        return stats