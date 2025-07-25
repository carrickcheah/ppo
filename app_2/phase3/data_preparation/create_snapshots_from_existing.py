"""
Create Training Snapshots from Existing Production Data

This script creates variations of the existing production snapshot
without needing database access.
"""

import os
import sys
import json
import random
import logging
from datetime import datetime
from typing import Dict, List, Any
from copy import deepcopy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SnapshotVariationCreator:
    """Creates snapshot variations from existing data."""
    
    def __init__(self, base_snapshot_path: str):
        """Initialize with base snapshot."""
        self.base_snapshot_path = base_snapshot_path
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'snapshots'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load base data
        with open(base_snapshot_path, 'r') as f:
            self.base_data = json.load(f)
            
        logger.info(f"Loaded base snapshot with {self.base_data['metadata']['total_tasks']} tasks")
        
    def create_normal_snapshot(self) -> str:
        """Copy base snapshot as normal."""
        logger.info("Creating normal load snapshot...")
        
        # Deep copy base data
        normal_data = deepcopy(self.base_data)
        
        # Update metadata
        normal_data['metadata']['snapshot_type'] = 'normal'
        normal_data['metadata']['description'] = 'Normal production load'
        normal_data['metadata']['created_at'] = datetime.now().isoformat()
        
        # Save
        output_path = os.path.join(self.output_dir, 'snapshot_normal.json')
        with open(output_path, 'w') as f:
            json.dump(normal_data, f, indent=2)
            
        logger.info(f"Created normal snapshot: {output_path}")
        return output_path
        
    def create_rush_snapshot(self) -> str:
        """Create rush order snapshot with urgent deadlines."""
        logger.info("Creating rush orders snapshot...")
        
        # Deep copy base data
        rush_data = deepcopy(self.base_data)
        
        families = rush_data.get('families', {})
        total_families = len(families)
        rush_count = 0
        
        # Make 80% of jobs urgent
        family_list = list(families.items())
        random.shuffle(family_list)
        
        for i, (family_id, family_data) in enumerate(family_list):
            if i < int(total_families * 0.8):
                # Make urgent
                family_data['lcd_days_remaining'] = random.randint(1, 7)
                family_data['is_important'] = random.random() < 0.67
                rush_count += 1
                
        # Update metadata
        rush_data['metadata']['snapshot_type'] = 'rush_orders'
        rush_data['metadata']['description'] = 'Rush orders (80% urgent)'
        rush_data['metadata']['created_at'] = datetime.now().isoformat()
        rush_data['metadata']['rush_percentage'] = (rush_count / total_families * 100)
        
        # Save
        output_path = os.path.join(self.output_dir, 'snapshot_rush.json')
        with open(output_path, 'w') as f:
            json.dump(rush_data, f, indent=2)
            
        logger.info(f"Created rush snapshot: {output_path} ({rush_count}/{total_families} rush)")
        return output_path
        
    def create_heavy_snapshot(self) -> str:
        """Create heavy load by duplicating some jobs."""
        logger.info("Creating heavy load snapshot...")
        
        # Deep copy base data
        heavy_data = deepcopy(self.base_data)
        
        families = heavy_data.get('families', {})
        original_count = len(families)
        
        # Duplicate 70% of jobs with variations
        families_to_duplicate = random.sample(
            list(families.items()), 
            int(original_count * 0.7)
        )
        
        new_id_counter = 9000  # Start with high number to avoid conflicts
        
        for family_id, family_data in families_to_duplicate:
            # Create duplicate with variations
            new_family_id = f"DUP_{new_id_counter:05d}"
            new_family = deepcopy(family_data)
            
            # Vary the deadline
            new_family['lcd_days_remaining'] = max(
                1,
                family_data.get('lcd_days_remaining', 10) + random.randint(-5, 10)
            )
            
            # Vary importance
            new_family['is_important'] = random.random() < 0.3
            
            # Update IDs
            new_family['job_reference'] = new_family_id
            
            # Slightly vary processing times
            for task in new_family.get('tasks', []):
                task['processing_time'] *= random.uniform(0.8, 1.2)
                
            families[new_family_id] = new_family
            new_id_counter += 1
            
        # Recalculate totals
        total_tasks = sum(
            len(family.get('tasks', [])) 
            for family in families.values()
        )
        
        # Update metadata
        heavy_data['metadata']['snapshot_type'] = 'heavy_load'
        heavy_data['metadata']['description'] = f'Heavy load ({len(families)} families)'
        heavy_data['metadata']['created_at'] = datetime.now().isoformat()
        heavy_data['metadata']['total_families'] = len(families)
        heavy_data['metadata']['total_tasks'] = total_tasks
        heavy_data['metadata']['duplication_factor'] = len(families) / original_count
        
        # Save
        output_path = os.path.join(self.output_dir, 'snapshot_heavy.json')
        with open(output_path, 'w') as f:
            json.dump(heavy_data, f, indent=2)
            
        logger.info(f"Created heavy snapshot: {output_path} ({total_tasks} tasks)")
        return output_path
        
    def create_bottleneck_snapshot(self) -> str:
        """Create bottleneck by limiting available machines."""
        logger.info("Creating machine bottleneck snapshot...")
        
        # Deep copy base data
        bottleneck_data = deepcopy(self.base_data)
        
        # Calculate machine usage from families
        machine_usage = {}
        families = bottleneck_data.get('families', {})
        
        for family_id, family_data in families.items():
            for task in family_data.get('tasks', []):
                for m_id in task.get('capable_machines', []):
                    machine_usage[m_id] = machine_usage.get(m_id, 0) + 1
                    
        # Keep only top 50 most used machines
        sorted_usage = sorted(machine_usage.items(), key=lambda x: x[1], reverse=True)
        top_machine_ids = {m[0] for m in sorted_usage[:50]}
        
        # Filter machines
        original_machines = bottleneck_data.get('machines', [])
        filtered_machines = [
            m for m in original_machines 
            if m.get('machine_id') in top_machine_ids
        ]
        
        # Update jobs to only use available machines
        removed_tasks = 0
        
        for family_id, family_data in families.items():
            valid_tasks = []
            for task in family_data.get('tasks', []):
                # Check if any capable machine is available
                capable_machines = set(task.get('capable_machines', []))
                available_machines = list(capable_machines.intersection(top_machine_ids))
                
                if available_machines:
                    task['capable_machines'] = available_machines
                    valid_tasks.append(task)
                else:
                    removed_tasks += 1
                    
            family_data['tasks'] = valid_tasks
            
        # Remove empty families
        families_to_remove = [
            fid for fid, fdata in families.items() 
            if len(fdata.get('tasks', [])) == 0
        ]
        for fid in families_to_remove:
            del families[fid]
            
        # Recalculate totals
        total_tasks = sum(
            len(family.get('tasks', [])) 
            for family in families.values()
        )
        
        # Update data
        bottleneck_data['machines'] = filtered_machines
        bottleneck_data['metadata']['snapshot_type'] = 'machine_bottleneck'
        bottleneck_data['metadata']['description'] = f'Machine bottleneck ({len(filtered_machines)} machines)'
        bottleneck_data['metadata']['created_at'] = datetime.now().isoformat()
        bottleneck_data['metadata']['total_machines'] = len(filtered_machines)
        bottleneck_data['metadata']['total_families'] = len(families)
        bottleneck_data['metadata']['total_tasks'] = total_tasks
        bottleneck_data['metadata']['removed_tasks'] = removed_tasks
        
        # Save
        output_path = os.path.join(self.output_dir, 'snapshot_bottleneck.json')
        with open(output_path, 'w') as f:
            json.dump(bottleneck_data, f, indent=2)
            
        logger.info(f"Created bottleneck snapshot: {output_path} ({len(filtered_machines)} machines)")
        return output_path
        
    def create_multi_heavy_snapshot(self) -> str:
        """Create snapshot with many multi-machine jobs."""
        logger.info("Creating multi-machine heavy snapshot...")
        
        # Deep copy base data
        multi_data = deepcopy(self.base_data)
        
        families = multi_data.get('families', {})
        machines = multi_data.get('machines', [])
        machine_ids = [m['machine_id'] for m in machines]
        
        total_tasks = 0
        multi_machine_tasks = 0
        converted_tasks = 0
        
        for family_id, family_data in families.items():
            for task in family_data.get('tasks', []):
                total_tasks += 1
                current_machines = task.get('capable_machines', [])
                
                if len(current_machines) > 1:
                    multi_machine_tasks += 1
                elif len(current_machines) == 1:
                    # Try to convert to multi-machine
                    if random.random() < 0.35:  # Convert 35% to reach ~30% total
                        primary_machine = current_machines[0]
                        
                        # Find compatible machines (similar type or nearby)
                        compatible = []
                        for m in machines:
                            m_id = m['machine_id']
                            if m_id != primary_machine:
                                # Same type or nearby ID
                                if abs(m_id - primary_machine) < 15:
                                    compatible.append(m_id)
                                    
                        if len(compatible) >= 2:
                            # Add 2-4 additional machines
                            num_additional = min(random.randint(2, 4), len(compatible))
                            additional_machines = random.sample(compatible, num_additional)
                            
                            # Update task
                            task['capable_machines'] = [primary_machine] + additional_machines
                            task['multi_machine'] = True
                            multi_machine_tasks += 1
                            converted_tasks += 1
                            
        # Update metadata
        multi_percentage = (multi_machine_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        multi_data['metadata']['snapshot_type'] = 'multi_machine_heavy'
        multi_data['metadata']['description'] = f'Multi-machine heavy ({multi_percentage:.1f}% multi)'
        multi_data['metadata']['created_at'] = datetime.now().isoformat()
        multi_data['metadata']['multi_machine_tasks'] = multi_machine_tasks
        multi_data['metadata']['multi_machine_percentage'] = multi_percentage
        multi_data['metadata']['converted_tasks'] = converted_tasks
        
        # Save
        output_path = os.path.join(self.output_dir, 'snapshot_multi_heavy.json')
        with open(output_path, 'w') as f:
            json.dump(multi_data, f, indent=2)
            
        logger.info(f"Created multi-machine snapshot: {output_path} ({multi_percentage:.1f}% multi)")
        return output_path
        
    def create_all_variations(self) -> List[str]:
        """Create all snapshot variations."""
        snapshots = []
        
        creators = [
            ('normal', self.create_normal_snapshot),
            ('rush', self.create_rush_snapshot),
            ('heavy', self.create_heavy_snapshot),
            ('bottleneck', self.create_bottleneck_snapshot),
            ('multi_heavy', self.create_multi_heavy_snapshot)
        ]
        
        for name, creator in creators:
            try:
                path = creator()
                snapshots.append(path)
            except Exception as e:
                logger.error(f"Failed to create {name} snapshot: {e}")
                
        return snapshots


def main():
    """Main function."""
    # Path to existing production snapshot
    base_snapshot = '/Users/carrickcheah/Project/ppo/app_2/data/real_production_snapshot.json'
    
    if not os.path.exists(base_snapshot):
        logger.error(f"Base snapshot not found: {base_snapshot}")
        return
        
    creator = SnapshotVariationCreator(base_snapshot)
    
    logger.info("Creating all snapshot variations...")
    snapshots = creator.create_all_variations()
    
    logger.info(f"\nCreated {len(snapshots)} snapshot variations!")
    

if __name__ == '__main__':
    main()