"""
Create Multiple Training Snapshots for Diverse Training Scenarios

This script generates various data snapshots to ensure the model
experiences different production scenarios during training.
"""

import os
import sys
import json
import random
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from copy import deepcopy

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_ingestion.ingest_data import ProductionDataIngester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingSnapshotCreator:
    """Creates various training snapshots with different characteristics."""
    
    def __init__(self, base_ingester: ProductionDataIngester = None):
        """Initialize with base data ingester."""
        self.ingester = base_ingester or ProductionDataIngester()
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'snapshots'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
    def create_normal_snapshot(self) -> str:
        """Create baseline snapshot with current production data."""
        logger.info("Creating normal load snapshot...")
        
        # Fetch current production data
        production_data = self.ingester.fetch_production_data(
            planning_horizon_days=30,
            limit=1000
        )
        
        # Save snapshot
        output_path = os.path.join(self.output_dir, 'snapshot_normal.json')
        self._save_snapshot(production_data, output_path, "Normal production load")
        
        return output_path
        
    def create_rush_snapshot(self) -> str:
        """Create snapshot with 80% urgent orders."""
        logger.info("Creating rush orders snapshot...")
        
        # Fetch production data
        production_data = self.ingester.fetch_production_data(
            planning_horizon_days=30,
            limit=1000
        )
        
        # Modify to make 80% urgent
        families = production_data.get('families', {})
        total_tasks = 0
        urgent_tasks = 0
        
        for family_id, family_data in families.items():
            # Make most jobs urgent (LCD < 7 days)
            if random.random() < 0.8:
                # Urgent: 1-7 days remaining
                new_lcd_days = random.randint(1, 7)
                family_data['lcd_days_remaining'] = new_lcd_days
                
                # Also increase importance
                if random.random() < 0.67:
                    family_data['is_important'] = True
                    
                # Update all tasks in family
                for task in family_data.get('tasks', []):
                    urgent_tasks += 1
            
            total_tasks += len(family_data.get('tasks', []))
        
        # Update metadata
        production_data['metadata']['snapshot_type'] = 'rush_orders'
        production_data['metadata']['urgent_percentage'] = (urgent_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Save snapshot
        output_path = os.path.join(self.output_dir, 'snapshot_rush.json')
        self._save_snapshot(production_data, output_path, "Rush orders (80% urgent)")
        
        return output_path
        
    def create_heavy_snapshot(self) -> str:
        """Create snapshot with 500+ jobs by extending planning horizon."""
        logger.info("Creating heavy load snapshot...")
        
        # Fetch with extended horizon
        production_data = self.ingester.fetch_production_data(
            planning_horizon_days=60,  # Double the horizon
            limit=2000  # Increase limit
        )
        
        # Verify we got more jobs
        total_tasks = sum(
            len(family.get('tasks', [])) 
            for family in production_data.get('families', {}).values()
        )
        
        logger.info(f"Heavy snapshot contains {total_tasks} tasks")
        
        # Update metadata
        production_data['metadata']['snapshot_type'] = 'heavy_load'
        production_data['metadata']['planning_horizon_days'] = 60
        
        # Save snapshot
        output_path = os.path.join(self.output_dir, 'snapshot_heavy.json')
        self._save_snapshot(production_data, output_path, f"Heavy load ({total_tasks} jobs)")
        
        return output_path
        
    def create_bottleneck_snapshot(self) -> str:
        """Create snapshot with limited machines (top 50 most used)."""
        logger.info("Creating machine bottleneck snapshot...")
        
        # Fetch production data
        production_data = self.ingester.fetch_production_data(
            planning_horizon_days=30,
            limit=1000
        )
        
        # Calculate machine usage
        machine_usage = self._calculate_machine_usage(production_data)
        
        # Keep only top 50 machines
        top_machines = sorted(
            machine_usage.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:50]
        top_machine_ids = {m[0] for m in top_machines}
        
        # Filter machines
        original_machines = production_data.get('machines', [])
        filtered_machines = [
            m for m in original_machines 
            if m.get('machine_id') in top_machine_ids
        ]
        
        # Update jobs to only use available machines
        families = production_data.get('families', {})
        removed_jobs = 0
        
        for family_id, family_data in families.items():
            valid_tasks = []
            for task in family_data.get('tasks', []):
                # Check if task can run on available machines
                capable_machines = set(task.get('capable_machines', []))
                if capable_machines.intersection(top_machine_ids):
                    # Update task to only use available machines
                    task['capable_machines'] = list(capable_machines.intersection(top_machine_ids))
                    valid_tasks.append(task)
                else:
                    removed_jobs += 1
                    
            family_data['tasks'] = valid_tasks
        
        # Update data
        production_data['machines'] = filtered_machines
        production_data['metadata']['snapshot_type'] = 'machine_bottleneck'
        production_data['metadata']['total_machines'] = len(filtered_machines)
        production_data['metadata']['removed_incompatible_jobs'] = removed_jobs
        
        # Save snapshot
        output_path = os.path.join(self.output_dir, 'snapshot_bottleneck.json')
        self._save_snapshot(
            production_data, 
            output_path, 
            f"Machine bottleneck ({len(filtered_machines)} machines)"
        )
        
        return output_path
        
    def create_multi_heavy_snapshot(self) -> str:
        """Create snapshot with 30% multi-machine jobs."""
        logger.info("Creating multi-machine heavy snapshot...")
        
        # Fetch production data
        production_data = self.ingester.fetch_production_data(
            planning_horizon_days=30,
            limit=1000
        )
        
        # Convert single-machine jobs to multi-machine
        families = production_data.get('families', {})
        machines = production_data.get('machines', [])
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
                elif len(current_machines) == 1 and random.random() < 0.28:
                    # Convert to multi-machine (aim for 30% total)
                    primary_machine = current_machines[0]
                    
                    # Find compatible machines (same type or nearby IDs)
                    compatible = [
                        m_id for m_id in machine_ids
                        if m_id != primary_machine and abs(m_id - primary_machine) < 20
                    ]
                    
                    if compatible:
                        # Add 2-4 additional machines
                        num_additional = min(random.randint(2, 4), len(compatible))
                        additional_machines = random.sample(compatible, num_additional)
                        
                        # Update task to require ALL these machines
                        task['capable_machines'] = [primary_machine] + additional_machines
                        task['multi_machine'] = True
                        task['simultaneous_occupation'] = True
                        multi_machine_tasks += 1
                        converted_tasks += 1
        
        # Update metadata
        multi_percentage = (multi_machine_tasks / total_tasks * 100) if total_tasks > 0 else 0
        production_data['metadata']['snapshot_type'] = 'multi_machine_heavy'
        production_data['metadata']['multi_machine_percentage'] = multi_percentage
        production_data['metadata']['converted_to_multi'] = converted_tasks
        
        logger.info(f"Multi-machine percentage: {multi_percentage:.1f}%")
        
        # Save snapshot
        output_path = os.path.join(self.output_dir, 'snapshot_multi_heavy.json')
        self._save_snapshot(
            production_data, 
            output_path, 
            f"Multi-machine heavy ({multi_percentage:.1f}% multi)"
        )
        
        return output_path
        
    def create_all_snapshots(self) -> List[str]:
        """Create all snapshot types."""
        snapshots = []
        
        snapshot_creators = [
            ('normal', self.create_normal_snapshot),
            ('rush', self.create_rush_snapshot),
            ('heavy', self.create_heavy_snapshot),
            ('bottleneck', self.create_bottleneck_snapshot),
            ('multi_heavy', self.create_multi_heavy_snapshot)
        ]
        
        for name, creator in snapshot_creators:
            try:
                logger.info(f"\nCreating {name} snapshot...")
                path = creator()
                snapshots.append(path)
                logger.info(f"✓ Created: {path}")
            except Exception as e:
                logger.error(f"✗ Failed to create {name} snapshot: {e}")
                
        return snapshots
        
    def _calculate_machine_usage(self, production_data: Dict) -> Dict[int, int]:
        """Calculate how many jobs use each machine."""
        machine_usage = {}
        
        families = production_data.get('families', {})
        for family_id, family_data in families.items():
            for task in family_data.get('tasks', []):
                for machine_id in task.get('capable_machines', []):
                    machine_usage[machine_id] = machine_usage.get(machine_id, 0) + 1
                    
        return machine_usage
        
    def _save_snapshot(self, data: Dict, path: str, description: str):
        """Save snapshot with metadata."""
        # Add creation info
        data['metadata']['created_at'] = datetime.now().isoformat()
        data['metadata']['description'] = description
        
        # Calculate statistics
        total_tasks = sum(
            len(family.get('tasks', [])) 
            for family in data.get('families', {}).values()
        )
        data['metadata']['total_tasks'] = total_tasks
        
        # Save
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved snapshot: {path} ({total_tasks} tasks)")


def main():
    """Main function to create training snapshots."""
    parser = argparse.ArgumentParser(description='Create training snapshots')
    parser.add_argument(
        '--type',
        choices=['normal', 'rush', 'heavy', 'bottleneck', 'multi_heavy', 'all'],
        default='all',
        help='Type of snapshot to create'
    )
    parser.add_argument(
        '--host',
        default=os.getenv('MARIADB_HOST', 'localhost'),
        help='Database host'
    )
    parser.add_argument(
        '--user',
        default=os.getenv('MARIADB_USERNAME', 'myuser'),
        help='Database user'
    )
    parser.add_argument(
        '--password',
        default=os.getenv('MARIADB_PASSWORD', 'mypassword'),
        help='Database password'
    )
    parser.add_argument(
        '--database',
        default=os.getenv('MARIADB_DATABASE', 'nex_valiant'),
        help='Database name'
    )
    
    args = parser.parse_args()
    
    # Create ingester
    ingester = ProductionDataIngester(
        host=args.host,
        user=args.user,
        password=args.password,
        database=args.database
    )
    
    # Create snapshot creator
    creator = TrainingSnapshotCreator(ingester)
    
    # Create requested snapshots
    if args.type == 'all':
        snapshots = creator.create_all_snapshots()
        logger.info(f"\nCreated {len(snapshots)} snapshots successfully!")
    else:
        # Create single snapshot
        create_method = getattr(creator, f'create_{args.type}_snapshot')
        path = create_method()
        logger.info(f"\nCreated {args.type} snapshot: {path}")


if __name__ == '__main__':
    main()