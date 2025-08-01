"""
Generate small-scale strategy datasets from real production data
Each dataset tests different scheduling challenges
"""

import os
import json
import random
from typing import Dict, List, Any
from datetime import datetime, timedelta


class StrategyDataGenerator:
    """Generate strategy-specific datasets from production data."""
    
    def __init__(self, source_data_dir: str = "/home/azureuser/ppo/app_2/data"):
        self.source_dir = source_data_dir
        self.output_dir = "/home/azureuser/ppo/app_2/phase4/data"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load all available production data
        self.all_families = {}
        self.all_machines = []
        self._load_production_data()
    
    def _load_production_data(self):
        """Load and merge production data from all toy stages."""
        machine_set = set()
        
        for stage in ['toy_normal', 'toy_hard', 'toy_multi']:
            file_path = os.path.join(self.source_dir, f"stage_{stage}_real_data.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Merge families
                    for fid, family in data['families'].items():
                        if fid not in self.all_families:
                            self.all_families[fid] = family
                    
                    # Collect unique machines
                    for machine in data['machines']:
                        machine_set.add(machine['machine_id'])
        
        # Convert machine set to list with proper format
        self.all_machines = [{'machine_id': mid} for mid in sorted(machine_set)]
        
        print(f"Loaded {len(self.all_families)} job families")
        print(f"Loaded {len(self.all_machines)} unique machines")
    
    def generate_small_balanced(self) -> Dict[str, Any]:
        """
        Small Balanced: 30 jobs, 15 machines
        - Mix of deadlines (urgent and relaxed)
        - Balanced workload
        """
        # Select 30 random families
        selected_families = dict(random.sample(
            list(self.all_families.items()), 
            min(30, len(self.all_families))
        ))
        
        # Select 15 machines
        selected_machines = random.sample(
            self.all_machines, 
            min(15, len(self.all_machines))
        )
        
        # Update machine IDs in tasks
        machine_ids = [m['machine_id'] for m in selected_machines]
        selected_families = self._update_machine_references(selected_families, machine_ids)
        
        data = {
            'scenario': 'small_balanced',
            'description': 'Balanced workload with mix of deadlines',
            'families': selected_families,
            'machines': selected_machines,
            'metrics': {
                'total_jobs': len(selected_families),
                'total_machines': len(selected_machines),
                'job_machine_ratio': len(selected_families) / len(selected_machines)
            }
        }
        
        self._save_data(data, 'small_balanced_data.json')
        return data
    
    def generate_small_rush(self) -> Dict[str, Any]:
        """
        Small Rush: 50 jobs, 20 machines
        - Many urgent jobs (tight deadlines)
        - Tests prioritization under pressure
        """
        # Sort families by deadline urgency
        urgent_families = sorted(
            self.all_families.items(),
            key=lambda x: x[1].get('lcd_days_remaining', 999)
        )
        
        # Select 50 most urgent
        selected_families = dict(urgent_families[:min(50, len(urgent_families))])
        
        # Make deadlines even tighter (reduce by 20%)
        for fid, family in selected_families.items():
            if 'lcd_days_remaining' in family:
                family['lcd_days_remaining'] = max(1, int(family['lcd_days_remaining'] * 0.8))
        
        # Select 20 machines
        selected_machines = random.sample(
            self.all_machines,
            min(20, len(self.all_machines))
        )
        
        machine_ids = [m['machine_id'] for m in selected_machines]
        selected_families = self._update_machine_references(selected_families, machine_ids)
        
        data = {
            'scenario': 'small_rush',
            'description': 'High pressure scenario with urgent deadlines',
            'families': selected_families,
            'machines': selected_machines,
            'metrics': {
                'total_jobs': len(selected_families),
                'total_machines': len(selected_machines),
                'avg_deadline_days': sum(f.get('lcd_days_remaining', 0) for f in selected_families.values()) / len(selected_families)
            }
        }
        
        self._save_data(data, 'small_rush_data.json')
        return data
    
    def generate_small_bottleneck(self) -> Dict[str, Any]:
        """
        Small Bottleneck: 40 jobs, 10 machines
        - High job-to-machine ratio (4:1)
        - Tests resource allocation
        """
        # Select 40 random families
        selected_families = dict(random.sample(
            list(self.all_families.items()),
            min(40, len(self.all_families))
        ))
        
        # Select only 10 machines (bottleneck)
        selected_machines = random.sample(
            self.all_machines,
            min(10, len(self.all_machines))
        )
        
        machine_ids = [m['machine_id'] for m in selected_machines]
        selected_families = self._update_machine_references(selected_families, machine_ids)
        
        data = {
            'scenario': 'small_bottleneck',
            'description': 'Resource constrained with high job-to-machine ratio',
            'families': selected_families,
            'machines': selected_machines,
            'metrics': {
                'total_jobs': len(selected_families),
                'total_machines': len(selected_machines),
                'job_machine_ratio': len(selected_families) / len(selected_machines),
                'bottleneck_severity': 'high'
            }
        }
        
        self._save_data(data, 'small_bottleneck_data.json')
        return data
    
    def generate_small_complex(self) -> Dict[str, Any]:
        """
        Small Complex: 50 jobs, 25 machines
        - Complex dependencies
        - Multi-machine jobs
        - Long processing times
        """
        # Select families with most sequences (complex jobs)
        complex_families = sorted(
            self.all_families.items(),
            key=lambda x: x[1].get('total_sequences', 1),
            reverse=True
        )
        
        # Select 50 most complex
        selected_families = dict(complex_families[:min(50, len(complex_families))])
        
        # Select 25 machines
        selected_machines = random.sample(
            self.all_machines,
            min(25, len(self.all_machines))
        )
        
        machine_ids = [m['machine_id'] for m in selected_machines]
        
        # Add some multi-machine requirements
        for fid, family in selected_families.items():
            for task in family['tasks']:
                # 20% chance of needing multiple machines
                if random.random() < 0.2:
                    # Need 2-3 machines simultaneously
                    n_machines = random.randint(2, 3)
                    task['required_machines'] = random.sample(machine_ids, min(n_machines, len(machine_ids)))
                    task['multi_machine'] = True
        
        selected_families = self._update_machine_references(selected_families, machine_ids)
        
        data = {
            'scenario': 'small_complex',
            'description': 'Complex jobs with dependencies and multi-machine requirements',
            'families': selected_families,
            'machines': selected_machines,
            'metrics': {
                'total_jobs': len(selected_families),
                'total_machines': len(selected_machines),
                'avg_sequences': sum(f.get('total_sequences', 1) for f in selected_families.values()) / len(selected_families),
                'multi_machine_jobs': sum(
                    1 for f in selected_families.values() 
                    for t in f['tasks'] 
                    if t.get('multi_machine', False)
                )
            }
        }
        
        self._save_data(data, 'small_complex_data.json')
        return data
    
    def _update_machine_references(self, families: Dict, available_machines: List[int]) -> Dict:
        """Update machine references in tasks to only use available machines."""
        for family in families.values():
            for task in family['tasks']:
                # If task has multi-machine requirement, keep those
                if 'required_machines' in task:
                    task['capable_machines'] = task['required_machines']
                else:
                    # Map to available machines
                    n_capable = min(3, len(available_machines))
                    task['capable_machines'] = random.sample(available_machines, n_capable)
        
        return families
    
    def _save_data(self, data: Dict, filename: str):
        """Save data to JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {data['scenario']} to {filepath}")
    
    def generate_all_strategies(self):
        """Generate all strategy datasets."""
        print("\nGenerating Strategy Datasets from Real Production Data")
        print("=" * 60)
        
        print("\n1. Small Balanced (30 jobs, 15 machines)")
        balanced = self.generate_small_balanced()
        print(f"   - Created with {balanced['metrics']['total_jobs']} jobs")
        
        print("\n2. Small Rush (50 jobs, 20 machines)")
        rush = self.generate_small_rush()
        print(f"   - Average deadline: {rush['metrics']['avg_deadline_days']:.1f} days")
        
        print("\n3. Small Bottleneck (40 jobs, 10 machines)")
        bottleneck = self.generate_small_bottleneck()
        print(f"   - Job/Machine ratio: {bottleneck['metrics']['job_machine_ratio']:.1f}")
        
        print("\n4. Small Complex (50 jobs, 25 machines)")
        complex_data = self.generate_small_complex()
        print(f"   - Average sequences: {complex_data['metrics']['avg_sequences']:.1f}")
        print(f"   - Multi-machine jobs: {complex_data['metrics']['multi_machine_jobs']}")
        
        print("\nAll strategy datasets generated successfully!")


if __name__ == "__main__":
    generator = StrategyDataGenerator()
    generator.generate_all_strategies()