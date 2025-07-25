"""
Validate Training Snapshots

Checks data integrity and generates statistics for all training snapshots.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SnapshotValidator:
    """Validates training snapshots and generates reports."""
    
    def __init__(self):
        """Initialize validator."""
        self.snapshot_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'snapshots'
        )
        self.report_path = os.path.join(self.snapshot_dir, 'validation_report.md')
        
    def validate_snapshot(self, snapshot_path: str) -> Dict[str, Any]:
        """Validate a single snapshot and return statistics."""
        logger.info(f"Validating: {os.path.basename(snapshot_path)}")
        
        try:
            with open(snapshot_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load snapshot: {e}")
            return {"error": str(e)}
        
        # Basic structure validation
        if not all(key in data for key in ['metadata', 'families', 'machines']):
            return {"error": "Missing required keys: metadata, families, machines"}
        
        # Extract statistics
        stats = {
            'file_name': os.path.basename(snapshot_path),
            'metadata': data.get('metadata', {}),
            'validation_errors': [],
            'warnings': []
        }
        
        # Analyze families and tasks
        families = data.get('families', {})
        machines = data.get('machines', [])
        
        # Count statistics
        total_tasks = 0
        multi_machine_tasks = 0
        processing_times = []
        lcd_days = []
        important_count = 0
        sequence_lengths = []
        machine_usage = defaultdict(int)
        
        for family_id, family_data in families.items():
            # Validate family structure
            if 'tasks' not in family_data:
                stats['validation_errors'].append(f"Family {family_id} missing tasks")
                continue
                
            tasks = family_data.get('tasks', [])
            sequence_lengths.append(len(tasks))
            
            if family_data.get('is_important', False):
                important_count += 1
                
            lcd_days.append(family_data.get('lcd_days_remaining', 0))
            
            for task in tasks:
                total_tasks += 1
                
                # Check task structure
                if 'capable_machines' not in task:
                    stats['validation_errors'].append(f"Task in {family_id} missing capable_machines")
                    continue
                    
                capable_machines = task.get('capable_machines', [])
                if len(capable_machines) > 1:
                    multi_machine_tasks += 1
                    
                # Track machine usage
                for m_id in capable_machines:
                    machine_usage[m_id] += 1
                    
                # Processing time
                proc_time = task.get('processing_time', 0)
                if proc_time <= 0:
                    stats['warnings'].append(f"Task in {family_id} has invalid processing time: {proc_time}")
                processing_times.append(proc_time)
        
        # Calculate statistics
        stats['statistics'] = {
            'total_families': len(families),
            'total_tasks': total_tasks,
            'total_machines': len(machines),
            'multi_machine_tasks': multi_machine_tasks,
            'multi_machine_percentage': (multi_machine_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            'important_families': important_count,
            'important_percentage': (important_count / len(families) * 100) if families else 0
        }
        
        # Processing time analysis
        if processing_times:
            stats['processing_times'] = {
                'min': min(processing_times),
                'max': max(processing_times),
                'mean': np.mean(processing_times),
                'median': np.median(processing_times),
                'total_hours': sum(processing_times)
            }
        
        # Deadline analysis
        if lcd_days:
            stats['deadlines'] = {
                'min_days': min(lcd_days),
                'max_days': max(lcd_days),
                'mean_days': np.mean(lcd_days),
                'urgent_jobs': sum(1 for d in lcd_days if d <= 7),
                'urgent_percentage': (sum(1 for d in lcd_days if d <= 7) / len(lcd_days) * 100)
            }
        
        # Machine utilization preview
        if machine_usage:
            sorted_usage = sorted(machine_usage.items(), key=lambda x: x[1], reverse=True)
            stats['machine_utilization'] = {
                'most_used': sorted_usage[:5],
                'least_used': sorted_usage[-5:],
                'unused_machines': len(machines) - len(machine_usage)
            }
        
        # Sequence length distribution
        if sequence_lengths:
            stats['sequence_lengths'] = {
                'min': min(sequence_lengths),
                'max': max(sequence_lengths),
                'mean': np.mean(sequence_lengths),
                'distribution': dict(zip(*np.unique(sequence_lengths, return_counts=True)))
            }
        
        # Check for data issues
        if not machines:
            stats['validation_errors'].append("No machines found")
        if not families:
            stats['validation_errors'].append("No families found")
        if total_tasks == 0:
            stats['validation_errors'].append("No tasks found")
            
        # Machine ID validation
        machine_ids = {m.get('machine_id') for m in machines}
        for family_id, family_data in families.items():
            for task in family_data.get('tasks', []):
                for m_id in task.get('capable_machines', []):
                    if m_id not in machine_ids:
                        stats['warnings'].append(f"Task references non-existent machine {m_id}")
                        break
        
        return stats
        
    def validate_all_snapshots(self) -> List[Dict[str, Any]]:
        """Validate all snapshots in the directory."""
        results = []
        
        # Find all JSON files
        snapshot_files = [
            f for f in os.listdir(self.snapshot_dir)
            if f.endswith('.json') and f != 'validation_report.json'
        ]
        
        logger.info(f"Found {len(snapshot_files)} snapshots to validate")
        
        for filename in sorted(snapshot_files):
            path = os.path.join(self.snapshot_dir, filename)
            result = self.validate_snapshot(path)
            results.append(result)
            
        return results
        
    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate markdown report from validation results."""
        report = ["# Training Snapshot Validation Report"]
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nTotal Snapshots: {len(results)}\n")
        
        # Summary table
        report.append("## Summary Table\n")
        report.append("| Snapshot | Tasks | Machines | Multi-Machine % | Urgent % | Status |")
        report.append("|----------|-------|----------|-----------------|----------|--------|")
        
        for result in results:
            if 'error' in result:
                report.append(f"| {result.get('file_name', 'Unknown')} | - | - | - | - | ❌ ERROR |")
                continue
                
            stats = result['statistics']
            deadlines = result.get('deadlines', {})
            status = "✅ Valid" if not result['validation_errors'] else "⚠️ Issues"
            
            report.append(
                f"| {result['file_name']} | "
                f"{stats['total_tasks']} | "
                f"{stats['total_machines']} | "
                f"{stats['multi_machine_percentage']:.1f}% | "
                f"{deadlines.get('urgent_percentage', 0):.1f}% | "
                f"{status} |"
            )
        
        # Detailed results
        report.append("\n## Detailed Results\n")
        
        for result in results:
            report.append(f"### {result.get('file_name', 'Unknown')}\n")
            
            if 'error' in result:
                report.append(f"**Error**: {result['error']}\n")
                continue
            
            # Metadata
            metadata = result.get('metadata', {})
            report.append(f"**Description**: {metadata.get('description', 'N/A')}")
            report.append(f"**Type**: {metadata.get('snapshot_type', 'N/A')}\n")
            
            # Statistics
            stats = result['statistics']
            report.append("**Basic Statistics:**")
            report.append(f"- Families: {stats['total_families']}")
            report.append(f"- Tasks: {stats['total_tasks']}")
            report.append(f"- Machines: {stats['total_machines']}")
            report.append(f"- Multi-machine tasks: {stats['multi_machine_tasks']} ({stats['multi_machine_percentage']:.1f}%)")
            report.append(f"- Important families: {stats['important_families']} ({stats['important_percentage']:.1f}%)\n")
            
            # Processing times
            if 'processing_times' in result:
                pt = result['processing_times']
                report.append("**Processing Times (hours):**")
                report.append(f"- Range: {pt['min']:.1f} - {pt['max']:.1f}")
                report.append(f"- Mean: {pt['mean']:.1f}")
                report.append(f"- Total: {pt['total_hours']:.1f}\n")
            
            # Deadlines
            if 'deadlines' in result:
                dl = result['deadlines']
                report.append("**Deadline Distribution:**")
                report.append(f"- Range: {dl['min_days']} - {dl['max_days']} days")
                report.append(f"- Urgent jobs (≤7 days): {dl['urgent_jobs']} ({dl['urgent_percentage']:.1f}%)\n")
            
            # Machine utilization
            if 'machine_utilization' in result:
                mu = result['machine_utilization']
                report.append("**Machine Utilization Preview:**")
                report.append("- Most used machines:")
                for m_id, count in mu['most_used']:
                    report.append(f"  - Machine {m_id}: {count} tasks")
                if mu['unused_machines'] > 0:
                    report.append(f"- Unused machines: {mu['unused_machines']}\n")
            
            # Issues
            if result['validation_errors']:
                report.append("**⚠️ Validation Errors:**")
                for error in result['validation_errors']:
                    report.append(f"- {error}")
                report.append("")
                
            if result['warnings']:
                report.append("**⚠️ Warnings:**")
                for warning in result['warnings'][:5]:  # Show first 5
                    report.append(f"- {warning}")
                if len(result['warnings']) > 5:
                    report.append(f"- ... and {len(result['warnings']) - 5} more")
                report.append("")
            
            report.append("---\n")
        
        # Summary statistics across all snapshots
        report.append("## Overall Statistics\n")
        
        total_tasks = sum(r['statistics']['total_tasks'] for r in results if 'statistics' in r)
        total_multi = sum(r['statistics']['multi_machine_tasks'] for r in results if 'statistics' in r)
        
        report.append(f"- Total tasks across all snapshots: {total_tasks}")
        report.append(f"- Total multi-machine tasks: {total_multi}")
        report.append(f"- Overall multi-machine percentage: {(total_multi/total_tasks*100) if total_tasks > 0 else 0:.1f}%")
        
        return "\n".join(report)
        
    def save_report(self, report: str, results: List[Dict[str, Any]]):
        """Save validation report and raw results."""
        # Save markdown report
        with open(self.report_path, 'w') as f:
            f.write(report)
        logger.info(f"Saved validation report: {self.report_path}")
        
        # Save JSON results - convert numpy types to regular Python types
        json_path = self.report_path.replace('.md', '.json')
        
        # Convert numpy types in results
        def convert_types(obj):
            if isinstance(obj, dict):
                return {str(k): convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
                
        converted_results = convert_types(results)
        
        with open(json_path, 'w') as f:
            json.dump(converted_results, f, indent=2)
        logger.info(f"Saved validation results: {json_path}")


def main():
    """Main function to validate all snapshots."""
    validator = SnapshotValidator()
    
    logger.info("Starting snapshot validation...")
    results = validator.validate_all_snapshots()
    
    logger.info("Generating validation report...")
    report = validator.generate_report(results)
    
    validator.save_report(report, results)
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    valid_count = sum(1 for r in results if 'error' not in r and not r.get('validation_errors', []))
    warning_count = sum(1 for r in results if r.get('warnings', []))
    error_count = sum(1 for r in results if 'error' in r or r.get('validation_errors', []))
    
    print(f"Total snapshots: {len(results)}")
    print(f"Valid snapshots: {valid_count}")
    print(f"Snapshots with warnings: {warning_count}")
    print(f"Snapshots with errors: {error_count}")
    print(f"\nFull report saved to: {validator.report_path}")
    

if __name__ == '__main__':
    main()