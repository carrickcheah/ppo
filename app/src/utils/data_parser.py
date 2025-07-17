"""
Data parser for SAMPLE_50.md production data.
Parses real production job sequences for medium environment testing.
"""

import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random
import json


class ProductionDataParser:
    """Parse production data from SAMPLE_50 format."""
    
    def __init__(self, seed: Optional[int] = 42):
        """Initialize parser with random seed for reproducibility."""
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
        # Product patterns for priority assignment
        self.priority_patterns = {
            'CF': 1,    # Critical family - highest priority
            'CH': 2,    # High priority
            'CM': 3,    # Medium priority  
            'CP': 4,    # Standard priority
        }
        
    def parse_job_line(self, line: str) -> Optional[Dict]:
        """
        Parse a single job line.
        Format: 7993CF03-007-1/4
        Returns: {'transaction': 7993, 'product': 'CF03-007', 'sequence': 1, 'total': 4}
        """
        line = line.strip()
        if not line:
            return None
            
        # Pattern: TransactionID + ProductCode + Sequence/Total
        match = re.match(r'(\d+)([A-Z]+\d+-\d+[A-Z]?)-(\d+)/(\d+)', line)
        if not match:
            return None
            
        return {
            'transaction': int(match.group(1)),
            'product': match.group(2),
            'sequence': int(match.group(3)),
            'total': int(match.group(4))
        }
    
    def assign_priority(self, product_code: str) -> int:
        """Assign priority based on product code prefix."""
        prefix = product_code[:2]
        return self.priority_patterns.get(prefix, 4)  # Default to standard priority
    
    def generate_lcd_date(self, transaction_id: int, priority: int, 
                         base_date: datetime = None) -> datetime:
        """
        Generate realistic LCD date based on priority.
        Higher priority = nearer deadline.
        """
        if base_date is None:
            base_date = datetime(2024, 1, 15)  # Default start date
            
        # Priority-based deadline ranges (in days)
        deadline_ranges = {
            1: (1, 7),    # 1-7 days (urgent)
            2: (7, 14),   # 1-2 weeks
            3: (14, 30),  # 2-4 weeks
            4: (30, 60),  # 1-2 months
            5: (60, 90),  # 2-3 months
        }
        
        min_days, max_days = deadline_ranges.get(priority, (30, 60))
        
        # Add some randomness but keep consistent for same transaction
        random.seed(self.seed + transaction_id if self.seed else None)
        days_ahead = random.randint(min_days, max_days)
        random.seed(self.seed)  # Reset seed
        
        return base_date + timedelta(days=days_ahead)
    
    def generate_processing_time(self, sequence: int, product: str) -> float:
        """Generate realistic processing time (hours) based on sequence position."""
        # Base time depends on product complexity
        if 'A' in product or 'B' in product:
            base_time = 2.0
        else:
            base_time = 1.5
            
        # Early sequences often take longer (setup, preparation)
        if sequence <= 2:
            time_multiplier = 1.2
        elif sequence >= 5:
            time_multiplier = 0.8  # Later steps often quicker
        else:
            time_multiplier = 1.0
            
        # Add some randomness
        variation = random.uniform(0.8, 1.2)
        
        return round(base_time * time_multiplier * variation, 1)
    
    def assign_capable_machines(self, product: str, sequence: int, 
                              n_machines: int = 10) -> List[int]:
        """Assign which machines can process this job."""
        # Simple rule: Different products use different machine groups
        product_hash = sum(ord(c) for c in product)
        
        # First sequences need specialized machines
        if sequence == 1:
            machine_group = product_hash % 3  # 3 groups of specialized machines
            if machine_group == 0:
                return [0, 1, 2]
            elif machine_group == 1:
                return [3, 4, 5]
            else:
                return [6, 7, 8]
        
        # Middle sequences can use more machines
        elif sequence <= 3:
            return list(range(n_machines - 2))  # Can't use last 2 machines
        
        # Final sequences use finishing machines
        else:
            return list(range(2, n_machines))  # Can't use first 2 machines
    
    def parse_file(self, filepath: str) -> Dict[str, Dict]:
        """
        Parse entire SAMPLE_50.md file.
        Returns nested dictionary: {transaction_id: {family_data}}
        """
        families = {}
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        current_date = datetime(2024, 1, 15)
        
        for line in lines:
            parsed = self.parse_job_line(line)
            if not parsed:
                continue
                
            transaction_id = str(parsed['transaction'])
            
            # Initialize family if not exists
            if transaction_id not in families:
                priority = self.assign_priority(parsed['product'])
                lcd_date = self.generate_lcd_date(
                    parsed['transaction'], 
                    priority, 
                    current_date
                )
                
                families[transaction_id] = {
                    'transaction_id': parsed['transaction'],
                    'product': parsed['product'],
                    'priority': priority,
                    'lcd_date': lcd_date.isoformat(),
                    'total_sequences': parsed['total'],
                    'tasks': []
                }
            
            # Add task to family
            task = {
                'sequence': parsed['sequence'],
                'processing_time': self.generate_processing_time(
                    parsed['sequence'], 
                    parsed['product']
                ),
                'capable_machines': self.assign_capable_machines(
                    parsed['product'],
                    parsed['sequence']
                ),
                'status': 'pending'
            }
            
            families[transaction_id]['tasks'].append(task)
        
        # Sort tasks within each family by sequence
        for family in families.values():
            family['tasks'].sort(key=lambda x: x['sequence'])
            
        return families
    
    def get_statistics(self, families: Dict[str, Dict]) -> Dict:
        """Calculate statistics about the parsed data."""
        total_tasks = sum(len(f['tasks']) for f in families.values())
        priorities = [f['priority'] for f in families.values()]
        sequences_per_family = [len(f['tasks']) for f in families.values()]
        
        # Check for gaps in sequences
        gaps_found = 0
        for family in families.values():
            sequences = [t['sequence'] for t in family['tasks']]
            expected = list(range(1, max(sequences) + 1))
            if sequences != expected:
                gaps_found += 1
        
        return {
            'n_families': len(families),
            'n_tasks': total_tasks,
            'avg_tasks_per_family': round(total_tasks / len(families), 2),
            'priority_distribution': {
                p: priorities.count(p) for p in range(1, 6)
            },
            'min_sequence_length': min(sequences_per_family),
            'max_sequence_length': max(sequences_per_family),
            'families_with_gaps': gaps_found,
            'gap_percentage': round(gaps_found / len(families) * 100, 1)
        }
    
    def save_to_json(self, families: Dict[str, Dict], filepath: str):
        """Save parsed data to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(families, f, indent=2, default=str)
    
    def load_from_json(self, filepath: str) -> Dict[str, Dict]:
        """Load parsed data from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)


def main():
    """Example usage of the parser."""
    parser = ProductionDataParser(seed=42)
    
    # Parse the sample data
    sample_file = '/Users/carrickcheah/Project/ppo/docs/sample_data/SAMPLE_50.md'
    families = parser.parse_file(sample_file)
    
    # Print statistics
    stats = parser.get_statistics(families)
    print("=== Production Data Statistics ===")
    print(f"Families: {stats['n_families']}")
    print(f"Total tasks: {stats['n_tasks']}")
    print(f"Average tasks per family: {stats['avg_tasks_per_family']}")
    print(f"Priority distribution: {stats['priority_distribution']}")
    print(f"Families with sequence gaps: {stats['families_with_gaps']} ({stats['gap_percentage']}%)")
    
    # Save to JSON for use in environment
    output_file = '/Users/carrickcheah/Project/ppo/app/data/parsed_production_data.json'
    parser.save_to_json(families, output_file)
    print(f"\nData saved to: {output_file}")
    
    # Example: Print first family
    first_family_id = list(families.keys())[0]
    first_family = families[first_family_id]
    print(f"\nExample family {first_family_id}:")
    print(f"  Product: {first_family['product']}")
    print(f"  Priority: {first_family['priority']}")
    print(f"  LCD Date: {first_family['lcd_date']}")
    print(f"  Tasks: {len(first_family['tasks'])}")
    for task in first_family['tasks'][:3]:  # First 3 tasks
        print(f"    Seq {task['sequence']}: {task['processing_time']}h on machines {task['capable_machines']}")


if __name__ == "__main__":
    main()