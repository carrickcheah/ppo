"""
Generate Statistics and Visualizations for Training Snapshots

Creates charts and summary statistics for all training data.
"""

import os
import json
import logging
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SnapshotStatistics:
    """Generate statistics and visualizations for snapshots."""
    
    def __init__(self):
        """Initialize statistics generator."""
        self.snapshot_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'snapshots'
        )
        self.output_dir = os.path.join(self.snapshot_dir, 'statistics')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_snapshot(self, filename: str) -> Dict[str, Any]:
        """Load a snapshot file."""
        path = os.path.join(self.snapshot_dir, filename)
        with open(path, 'r') as f:
            return json.load(f)
            
    def analyze_all_snapshots(self) -> Dict[str, Dict]:
        """Analyze all snapshots and return statistics."""
        snapshot_files = [
            f for f in os.listdir(self.snapshot_dir)
            if f.endswith('.json') and f.startswith(('snapshot_', 'edge_case_'))
        ]
        
        statistics = {}
        
        for filename in snapshot_files:
            try:
                data = self.load_snapshot(filename)
                stats = self.analyze_snapshot(data, filename)
                statistics[filename] = stats
            except Exception as e:
                logger.error(f"Failed to analyze {filename}: {e}")
                
        return statistics
        
    def analyze_snapshot(self, data: Dict, filename: str) -> Dict:
        """Analyze a single snapshot."""
        families = data.get('families', {})
        machines = data.get('machines', [])
        
        # Initialize counters
        stats = {
            'filename': filename,
            'total_families': len(families),
            'total_machines': len(machines),
            'total_tasks': 0,
            'multi_machine_tasks': 0,
            'processing_times': [],
            'lcd_distribution': defaultdict(int),
            'sequence_lengths': [],
            'machine_usage': defaultdict(int),
            'important_count': 0
        }
        
        # Analyze families
        for family_id, family_data in families.items():
            tasks = family_data.get('tasks', [])
            stats['sequence_lengths'].append(len(tasks))
            
            if family_data.get('is_important', False):
                stats['important_count'] += 1
                
            # Categorize by deadline
            lcd_days = family_data.get('lcd_days_remaining', 0)
            if lcd_days <= 3:
                stats['lcd_distribution']['Critical (≤3 days)'] += 1
            elif lcd_days <= 7:
                stats['lcd_distribution']['Urgent (4-7 days)'] += 1
            elif lcd_days <= 14:
                stats['lcd_distribution']['Normal (8-14 days)'] += 1
            else:
                stats['lcd_distribution']['Relaxed (>14 days)'] += 1
                
            # Analyze tasks
            for task in tasks:
                stats['total_tasks'] += 1
                
                # Processing time
                proc_time = task.get('processing_time', 0)
                stats['processing_times'].append(proc_time)
                
                # Multi-machine
                capable_machines = task.get('capable_machines', [])
                if len(capable_machines) > 1:
                    stats['multi_machine_tasks'] += 1
                    
                # Machine usage
                for m_id in capable_machines:
                    stats['machine_usage'][m_id] += 1
                    
        return stats
        
    def create_visualizations(self, statistics: Dict[str, Dict]):
        """Create visualization charts."""
        # 1. Task count comparison
        self.plot_task_counts(statistics)
        
        # 2. Multi-machine percentage comparison
        self.plot_multi_machine_percentages(statistics)
        
        # 3. Deadline distribution
        self.plot_deadline_distributions(statistics)
        
        # 4. Processing time distributions
        self.plot_processing_times(statistics)
        
        # 5. Machine utilization heatmap
        self.plot_machine_utilization(statistics)
        
    def plot_task_counts(self, statistics: Dict[str, Dict]):
        """Plot task counts across snapshots."""
        plt.figure(figsize=(12, 6))
        
        snapshots = []
        task_counts = []
        colors = []
        
        for filename, stats in sorted(statistics.items()):
            name = filename.replace('snapshot_', '').replace('edge_case_', '').replace('.json', '')
            snapshots.append(name)
            task_counts.append(stats['total_tasks'])
            
            # Color code by type
            if 'edge' in filename:
                colors.append('orange')
            elif 'heavy' in name:
                colors.append('red')
            elif 'rush' in name:
                colors.append('darkred')
            elif 'bottleneck' in name:
                colors.append('purple')
            else:
                colors.append('blue')
                
        bars = plt.bar(snapshots, task_counts, color=colors)
        plt.xlabel('Snapshot Type')
        plt.ylabel('Total Tasks')
        plt.title('Task Count by Snapshot Type')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, task_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha='center', va='bottom')
                    
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'task_counts.png'))
        plt.close()
        
    def plot_multi_machine_percentages(self, statistics: Dict[str, Dict]):
        """Plot multi-machine task percentages."""
        plt.figure(figsize=(12, 6))
        
        snapshots = []
        percentages = []
        
        for filename, stats in sorted(statistics.items()):
            name = filename.replace('snapshot_', '').replace('edge_case_', '').replace('.json', '')
            snapshots.append(name)
            
            if stats['total_tasks'] > 0:
                pct = (stats['multi_machine_tasks'] / stats['total_tasks']) * 100
            else:
                pct = 0
            percentages.append(pct)
            
        bars = plt.bar(snapshots, percentages, color='green')
        plt.xlabel('Snapshot Type')
        plt.ylabel('Multi-Machine Task %')
        plt.title('Multi-Machine Task Percentage by Snapshot')
        plt.xticks(rotation=45, ha='right')
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom')
                    
        # Add target line at 30%
        plt.axhline(y=30, color='red', linestyle='--', label='Target: 30%')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'multi_machine_percentages.png'))
        plt.close()
        
    def plot_deadline_distributions(self, statistics: Dict[str, Dict]):
        """Plot deadline distributions for each snapshot."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (filename, stats) in enumerate(sorted(statistics.items())):
            if idx >= 6:
                break
                
            ax = axes[idx]
            name = filename.replace('snapshot_', '').replace('edge_case_', '').replace('.json', '')
            
            # Get deadline distribution
            lcd_dist = stats['lcd_distribution']
            if lcd_dist:
                categories = list(lcd_dist.keys())
                counts = list(lcd_dist.values())
                
                colors_map = {
                    'Critical (≤3 days)': 'darkred',
                    'Urgent (4-7 days)': 'orange',
                    'Normal (8-14 days)': 'yellow',
                    'Relaxed (>14 days)': 'green'
                }
                colors = [colors_map.get(cat, 'blue') for cat in categories]
                
                ax.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%')
                ax.set_title(f'{name}')
                
        plt.suptitle('Deadline Distribution by Snapshot', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'deadline_distributions.png'))
        plt.close()
        
    def plot_processing_times(self, statistics: Dict[str, Dict]):
        """Plot processing time distributions."""
        plt.figure(figsize=(14, 8))
        
        all_data = []
        labels = []
        
        for filename, stats in sorted(statistics.items()):
            if stats['processing_times']:
                name = filename.replace('snapshot_', '').replace('edge_case_', '').replace('.json', '')
                all_data.append(stats['processing_times'])
                labels.append(name)
                
        # Create box plot
        bp = plt.boxplot(all_data, labels=labels, patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            
        plt.xlabel('Snapshot Type')
        plt.ylabel('Processing Time (hours)')
        plt.title('Processing Time Distribution by Snapshot')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'processing_times.png'))
        plt.close()
        
    def plot_machine_utilization(self, statistics: Dict[str, Dict]):
        """Plot machine utilization patterns."""
        # Select a few key snapshots
        key_snapshots = ['snapshot_normal.json', 'snapshot_bottleneck.json', 'snapshot_multi_heavy.json']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, snapshot_name in enumerate(key_snapshots):
            if snapshot_name not in statistics:
                continue
                
            ax = axes[idx]
            stats = statistics[snapshot_name]
            machine_usage = stats['machine_usage']
            
            if machine_usage:
                # Get top 20 most used machines
                sorted_usage = sorted(machine_usage.items(), key=lambda x: x[1], reverse=True)[:20]
                machines = [f"M{m[0]}" for m in sorted_usage]
                usage = [m[1] for m in sorted_usage]
                
                bars = ax.bar(machines, usage)
                ax.set_xlabel('Machine ID')
                ax.set_ylabel('Number of Tasks')
                ax.set_title(snapshot_name.replace('.json', '').replace('snapshot_', ''))
                ax.tick_params(axis='x', rotation=45)
                
                # Color gradient
                colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(bars)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                    
        plt.suptitle('Top 20 Machine Utilization Patterns', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'machine_utilization.png'))
        plt.close()
        
    def generate_summary_report(self, statistics: Dict[str, Dict]):
        """Generate a summary report of all statistics."""
        report = ["# Training Data Statistics Summary\n"]
        
        report.append("## Overview\n")
        report.append(f"Total snapshots analyzed: {len(statistics)}\n")
        
        # Summary table
        report.append("| Snapshot | Tasks | Machines | Multi-Machine % | Important % |")
        report.append("|----------|-------|----------|-----------------|-------------|")
        
        for filename, stats in sorted(statistics.items()):
            name = filename.replace('.json', '')
            multi_pct = (stats['multi_machine_tasks'] / stats['total_tasks'] * 100) if stats['total_tasks'] > 0 else 0
            important_pct = (stats['important_count'] / stats['total_families'] * 100) if stats['total_families'] > 0 else 0
            
            report.append(
                f"| {name} | {stats['total_tasks']} | {stats['total_machines']} | "
                f"{multi_pct:.1f}% | {important_pct:.1f}% |"
            )
            
        report.append("\n## Key Insights\n")
        
        # Find interesting patterns
        max_tasks = max(stats['total_tasks'] for stats in statistics.values())
        max_multi = max(
            (stats['multi_machine_tasks'] / stats['total_tasks'] * 100) if stats['total_tasks'] > 0 else 0
            for stats in statistics.values()
        )
        
        report.append(f"- Maximum tasks in a snapshot: {max_tasks}")
        report.append(f"- Maximum multi-machine percentage: {max_multi:.1f}%")
        report.append(f"- Visualization files saved to: {self.output_dir}")
        
        # Save report
        report_path = os.path.join(self.output_dir, 'statistics_summary.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
            
        logger.info(f"Summary report saved to: {report_path}")


def main():
    """Main function to generate statistics."""
    stats_gen = SnapshotStatistics()
    
    logger.info("Analyzing all snapshots...")
    statistics = stats_gen.analyze_all_snapshots()
    
    logger.info("Creating visualizations...")
    stats_gen.create_visualizations(statistics)
    
    logger.info("Generating summary report...")
    stats_gen.generate_summary_report(statistics)
    
    logger.info(f"All statistics saved to: {stats_gen.output_dir}")


if __name__ == '__main__':
    main()