"""
Data parser for SAMPLE_50.md production data with boolean importance.
For learning phase - in production, is_important comes from database.
"""

import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random
import json


class ProductionDataParserBoolean:
    """Parse production data with boolean is_important flag instead of 1-5 priority."""
    
    def __init__(self, seed: Optional[int] = 42, important_ratio: float = 0.1):
        """
        Initialize parser with random seed for reproducibility.
        
        Args:
            seed: Random seed
            important_ratio: Fraction of jobs to mark as important (default 10%)
        """
        self.seed = seed
        self.important_ratio = important_ratio
        if seed is not None:
            random.seed(seed)
    
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
    
    def assign_importance(self, transaction_id: int, product_code: str) -> bool:
        """
        Assign boolean importance for learning phase.
        In production, this comes from database.
        
        Strategy for learning:
        - 10% of jobs marked as important (realistic ratio)
        - OR: Critical Family products (CF prefix) are always important
        """
        # Option 1: Product-based (CF = Critical Family)
        if product_code.startswith('CF'):
            return True
            
        # Option 2: Random assignment based on ratio
        # Use transaction_id for consistent assignment
        random.seed(self.seed + transaction_id if self.seed else None)
        is_important = random.random() < self.important_ratio
        random.seed(self.seed)  # Reset seed
        
        return is_important
    
    def generate_lcd_date(self, transaction_id: int, is_important: bool, 
                         base_date: datetime = None) -> datetime:
        """
        Generate realistic LCD date based on importance.
        Important jobs tend to have nearer deadlines.
        """
        if base_date is None:
            base_date = datetime(2024, 1, 15)
            
        # Deadline ranges based on importance
        if is_important:
            # Important: 1-14 days (urgent)
            min_days, max_days = 1, 14
        else:
            # Not important: 14-60 days (normal)
            min_days, max_days = 14, 60
        
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
                is_important = self.assign_importance(
                    parsed['transaction'], 
                    parsed['product']
                )
                lcd_date = self.generate_lcd_date(
                    parsed['transaction'], 
                    is_important,
                    current_date
                )
                
                families[transaction_id] = {
                    'transaction_id': parsed['transaction'],
                    'product': parsed['product'],
                    'is_important': is_important,  # BOOLEAN instead of priority
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
        important_families = sum(1 for f in families.values() if f['is_important'])
        sequences_per_family = [len(f['tasks']) for f in families.values()]
        
        # Check for gaps in sequences
        gaps_found = 0
        for family in families.values():
            sequences = [t['sequence'] for t in family['tasks']]
            expected = list(range(1, max(sequences) + 1))
            if sequences != expected:
                gaps_found += 1
        
        # LCD distribution for important vs not important
        important_lcds = []
        normal_lcds = []
        base_date = datetime(2024, 1, 15)
        
        for family in families.values():
            lcd = datetime.fromisoformat(family['lcd_date'])
            days_to_lcd = (lcd - base_date).days
            
            if family['is_important']:
                important_lcds.append(days_to_lcd)
            else:
                normal_lcds.append(days_to_lcd)
        
        return {
            'n_families': len(families),
            'n_tasks': total_tasks,
            'avg_tasks_per_family': round(total_tasks / len(families), 2),
            'important_families': important_families,
            'important_percentage': round(important_families / len(families) * 100, 1),
            'normal_families': len(families) - important_families,
            'min_sequence_length': min(sequences_per_family),
            'max_sequence_length': max(sequences_per_family),
            'families_with_gaps': gaps_found,
            'gap_percentage': round(gaps_found / len(families) * 100, 1),
            'avg_lcd_important': round(sum(important_lcds) / len(important_lcds), 1) if important_lcds else 0,
            'avg_lcd_normal': round(sum(normal_lcds) / len(normal_lcds), 1) if normal_lcds else 0
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
    """Example usage of the boolean parser."""
    parser = ProductionDataParserBoolean(seed=42, important_ratio=0.1)
    
    # Parse the sample data
    sample_file = '/Users/carrickcheah/Project/ppo/docs/sample_data/SAMPLE_50.md'
    families = parser.parse_file(sample_file)
    
    # Print statistics
    stats = parser.get_statistics(families)
    print("=== Production Data Statistics (Boolean) ===")
    print(f"Families: {stats['n_families']}")
    print(f"Total tasks: {stats['n_tasks']}")
    print(f"Average tasks per family: {stats['avg_tasks_per_family']}")
    print(f"\nImportance distribution:")
    print(f"  Important families: {stats['important_families']} ({stats['important_percentage']}%)")
    print(f"  Normal families: {stats['normal_families']} ({100-stats['important_percentage']}%)")
    print(f"\nAverage LCD (days from start):")
    print(f"  Important jobs: {stats['avg_lcd_important']} days")
    print(f"  Normal jobs: {stats['avg_lcd_normal']} days")
    print(f"\nFamilies with sequence gaps: {stats['families_with_gaps']} ({stats['gap_percentage']}%)")
    
    # Save to JSON for use in environment
    output_file = '/Users/carrickcheah/Project/ppo/app/data/parsed_production_data_boolean.json'
    parser.save_to_json(families, output_file)
    print(f"\nData saved to: {output_file}")
    
    # Example: Print first few families
    print("\nExample families:")
    for i, (fid, family) in enumerate(list(families.items())[:5]):
        print(f"\nFamily {fid}:")
        print(f"  Product: {family['product']}")
        print(f"  Important: {family['is_important']}")
        print(f"  LCD Date: {family['lcd_date']}")
        print(f"  Tasks: {len(family['tasks'])}")


if __name__ == "__main__":
    main()