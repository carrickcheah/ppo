"""
Generate Edge Case Scenarios for Robust Training

Creates synthetic scenarios that test the model's ability to handle
extreme or unusual scheduling situations.
"""

import os
import sys
import json
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from copy import deepcopy

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EdgeCaseGenerator:
    """Generates edge case scenarios for training robustness."""
    
    def __init__(self):
        """Initialize edge case generator."""
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'snapshots'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_same_machine_scenario(self, n_jobs: int = 50) -> str:
        """Generate scenario where all jobs need the same machine."""
        logger.info("Generating same-machine edge case...")
        
        # Create a bottleneck machine
        bottleneck_machine_id = 1
        machines = [
            {
                "machine_id": bottleneck_machine_id,
                "machine_name": "BOTTLENECK-01",
                "machine_type_id": 1
            }
        ]
        
        # Add a few other machines that won't be used much
        for i in range(2, 6):
            machines.append({
                "machine_id": i,
                "machine_name": f"OTHER-{i:02d}",
                "machine_type_id": 2
            })
        
        # Create jobs that mostly need the bottleneck machine
        families = {}
        task_id = 0
        
        for family_idx in range(n_jobs // 3):  # Assume 3 tasks per family
            family_id = f"EDGE_FAM_{family_idx:03d}"
            
            tasks = []
            for seq in range(1, 4):
                task_id += 1
                
                # 80% need bottleneck machine, 20% can use others
                if random.random() < 0.8:
                    capable_machines = [bottleneck_machine_id]
                else:
                    capable_machines = [random.choice([2, 3, 4, 5])]
                
                task = {
                    "sequence": seq,
                    "process_name": f"EDGE_PROCESS_{task_id}",
                    "processing_time": random.uniform(1.0, 5.0),
                    "capable_machines": capable_machines,
                    "status": "pending",
                    "balance_quantity": 100.0
                }
                tasks.append(task)
            
            families[family_id] = {
                "job_reference": family_id,
                "product": "EDGE_PRODUCT",
                "is_important": random.random() < 0.3,
                "lcd_days_remaining": random.randint(3, 10),
                "total_sequences": 3,
                "tasks": tasks
            }
        
        # Create data structure
        edge_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "description": "Edge case: All jobs need same machine",
                "snapshot_type": "edge_same_machine",
                "bottleneck_machine_id": bottleneck_machine_id,
                "total_families": len(families),
                "total_tasks": task_id
            },
            "families": families,
            "machines": machines
        }
        
        # Save
        output_path = os.path.join(self.output_dir, 'edge_case_same_machine.json')
        with open(output_path, 'w') as f:
            json.dump(edge_data, f, indent=2)
            
        logger.info(f"Created same-machine edge case: {output_path}")
        return output_path
        
    def generate_cascading_deadlines(self, n_chains: int = 10) -> str:
        """Generate jobs with cascading deadline dependencies."""
        logger.info("Generating cascading deadlines edge case...")
        
        # Create sufficient machines
        machines = []
        for i in range(1, 21):
            machines.append({
                "machine_id": i,
                "machine_name": f"MACHINE-{i:02d}",
                "machine_type_id": (i - 1) // 5 + 1
            })
        
        # Create chains of dependent jobs
        families = {}
        task_id = 0
        
        for chain_idx in range(n_chains):
            # Each chain has 5-8 jobs
            chain_length = random.randint(5, 8)
            
            # Start deadline for the chain
            base_deadline = random.randint(5, 15)
            
            for job_in_chain in range(chain_length):
                family_id = f"CHAIN_{chain_idx:02d}_JOB_{job_in_chain:02d}"
                
                # Calculate cascading deadline
                # Each job needs to finish before the next can start
                job_deadline = base_deadline - (chain_length - job_in_chain - 1)
                
                # Make earlier jobs in chain more important
                is_important = job_in_chain < 2
                
                # Create tasks with dependencies
                tasks = []
                for seq in range(1, 3):  # 2 tasks per job
                    task_id += 1
                    
                    task = {
                        "sequence": seq,
                        "process_name": f"CASCADE_{task_id}",
                        "processing_time": random.uniform(0.5, 2.0),
                        "capable_machines": random.sample(
                            range(1, 21), 
                            random.randint(1, 3)
                        ),
                        "status": "pending",
                        "chain_position": job_in_chain,
                        "chain_id": chain_idx
                    }
                    tasks.append(task)
                
                families[family_id] = {
                    "job_reference": family_id,
                    "product": f"CHAIN_PRODUCT_{chain_idx}",
                    "is_important": is_important,
                    "lcd_days_remaining": max(1, job_deadline),
                    "total_sequences": 2,
                    "tasks": tasks,
                    "chain_info": {
                        "chain_id": chain_idx,
                        "position": job_in_chain,
                        "total_in_chain": chain_length
                    }
                }
        
        # Create data structure
        edge_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "description": "Edge case: Cascading deadline dependencies",
                "snapshot_type": "edge_cascading",
                "total_chains": n_chains,
                "total_families": len(families),
                "total_tasks": task_id
            },
            "families": families,
            "machines": machines
        }
        
        # Save
        output_path = os.path.join(self.output_dir, 'edge_case_cascading.json')
        with open(output_path, 'w') as f:
            json.dump(edge_data, f, indent=2)
            
        logger.info(f"Created cascading deadlines edge case: {output_path}")
        return output_path
        
    def generate_conflict_scenario(self, n_jobs: int = 40) -> str:
        """Generate scenario with conflicting priorities and impossible deadlines."""
        logger.info("Generating priority conflict edge case...")
        
        # Limited machines
        machines = []
        for i in range(1, 11):  # Only 10 machines
            machines.append({
                "machine_id": i,
                "machine_name": f"LIMITED-{i:02d}",
                "machine_type_id": 1
            })
        
        # Calculate total available hours
        planning_horizon_hours = 24 * 7  # 1 week
        total_machine_hours = len(machines) * planning_horizon_hours
        
        # Create jobs that require more hours than available
        families = {}
        task_id = 0
        total_hours_needed = 0
        
        for job_idx in range(n_jobs):
            family_id = f"CONFLICT_{job_idx:03d}"
            
            tasks = []
            for seq in range(1, 3):  # 2 tasks per job
                task_id += 1
                
                # Long processing times
                processing_time = random.uniform(8.0, 24.0)
                total_hours_needed += processing_time
                
                task = {
                    "sequence": seq,
                    "process_name": f"CONFLICT_TASK_{task_id}",
                    "processing_time": processing_time,
                    "capable_machines": random.sample(
                        range(1, 11), 
                        random.randint(1, 3)
                    ),
                    "status": "pending"
                }
                tasks.append(task)
            
            # All jobs are important and urgent
            families[family_id] = {
                "job_reference": family_id,
                "product": "URGENT_PRODUCT",
                "is_important": True,  # All important
                "lcd_days_remaining": random.randint(1, 3),  # All urgent
                "total_sequences": 2,
                "tasks": tasks
            }
        
        # Create data structure
        edge_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "description": "Edge case: Conflicting priorities with impossible deadlines",
                "snapshot_type": "edge_conflicts",
                "total_machine_hours": total_machine_hours,
                "total_hours_needed": total_hours_needed,
                "overload_factor": total_hours_needed / total_machine_hours,
                "total_families": len(families),
                "total_tasks": task_id
            },
            "families": families,
            "machines": machines
        }
        
        # Save
        output_path = os.path.join(self.output_dir, 'edge_case_conflicts.json')
        with open(output_path, 'w') as f:
            json.dump(edge_data, f, indent=2)
            
        logger.info(f"Created priority conflicts edge case: {output_path}")
        logger.info(f"Overload factor: {total_hours_needed / total_machine_hours:.2f}x")
        return output_path
        
    def generate_multi_machine_complex(self, n_jobs: int = 30) -> str:
        """Generate complex multi-machine scenarios."""
        logger.info("Generating complex multi-machine edge case...")
        
        # Create machine groups
        machines = []
        machine_groups = {
            "assembly": list(range(1, 6)),      # Machines 1-5
            "processing": list(range(6, 16)),   # Machines 6-15
            "finishing": list(range(16, 21)),   # Machines 16-20
            "testing": list(range(21, 26))      # Machines 21-25
        }
        
        for i in range(1, 26):
            machines.append({
                "machine_id": i,
                "machine_name": f"COMPLEX-{i:02d}",
                "machine_type_id": (i - 1) // 5 + 1
            })
        
        # Create jobs that need machines from multiple groups
        families = {}
        task_id = 0
        
        for job_idx in range(n_jobs):
            family_id = f"MULTI_COMPLEX_{job_idx:03d}"
            
            tasks = []
            for seq in range(1, 4):  # 3 tasks per job
                task_id += 1
                
                # Different patterns of multi-machine requirements
                pattern = random.choice([
                    # Need one from each group
                    "cross_group",
                    # Need multiple from same group
                    "same_group",
                    # Need specific combination
                    "specific_combo"
                ])
                
                if pattern == "cross_group":
                    # Pick one machine from 2-3 different groups
                    groups_needed = random.sample(list(machine_groups.keys()), random.randint(2, 3))
                    capable_machines = []
                    for group in groups_needed:
                        capable_machines.append(random.choice(machine_groups[group]))
                        
                elif pattern == "same_group":
                    # Pick 3-5 machines from same group
                    group = random.choice(list(machine_groups.keys()))
                    num_machines = min(random.randint(3, 5), len(machine_groups[group]))
                    capable_machines = random.sample(machine_groups[group], num_machines)
                    
                else:  # specific_combo
                    # Specific problematic combinations
                    combos = [
                        [1, 6, 16, 21],  # One from each group
                        [1, 2, 3, 4, 5],  # All assembly
                        [10, 11, 12, 20, 21],  # Mixed groups
                    ]
                    capable_machines = random.choice(combos)
                
                task = {
                    "sequence": seq,
                    "process_name": f"MULTI_COMPLEX_{task_id}",
                    "processing_time": random.uniform(2.0, 8.0),
                    "capable_machines": capable_machines,
                    "status": "pending",
                    "multi_machine": True,
                    "pattern": pattern
                }
                tasks.append(task)
            
            families[family_id] = {
                "job_reference": family_id,
                "product": "COMPLEX_PRODUCT",
                "is_important": random.random() < 0.5,
                "lcd_days_remaining": random.randint(3, 14),
                "total_sequences": 3,
                "tasks": tasks
            }
        
        # Create data structure
        edge_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "description": "Edge case: Complex multi-machine requirements",
                "snapshot_type": "edge_multi_complex",
                "machine_groups": machine_groups,
                "total_families": len(families),
                "total_tasks": task_id
            },
            "families": families,
            "machines": machines
        }
        
        # Save
        output_path = os.path.join(self.output_dir, 'edge_case_multi_complex.json')
        with open(output_path, 'w') as f:
            json.dump(edge_data, f, indent=2)
            
        logger.info(f"Created complex multi-machine edge case: {output_path}")
        return output_path
        
    def generate_all_edge_cases(self) -> List[str]:
        """Generate all edge case scenarios."""
        edge_cases = []
        
        generators = [
            ('same_machine', self.generate_same_machine_scenario),
            ('cascading', self.generate_cascading_deadlines),
            ('conflicts', self.generate_conflict_scenario),
            ('multi_complex', self.generate_multi_machine_complex)
        ]
        
        for name, generator in generators:
            try:
                logger.info(f"\nGenerating {name} edge case...")
                path = generator()
                edge_cases.append(path)
                logger.info(f"✓ Generated: {path}")
            except Exception as e:
                logger.error(f"✗ Failed to generate {name} edge case: {e}")
                
        return edge_cases


def main():
    """Main function to generate edge cases."""
    generator = EdgeCaseGenerator()
    
    logger.info("Generating all edge case scenarios...")
    edge_cases = generator.generate_all_edge_cases()
    
    logger.info(f"\nGenerated {len(edge_cases)} edge case scenarios!")
    logger.info("\nEdge cases created:")
    for path in edge_cases:
        logger.info(f"  - {os.path.basename(path)}")


if __name__ == '__main__':
    main()