"""
Create toy stage data files that Phase 4 generator expects.
This creates minimal data files so Phase 4 can run without database connection.
"""

import json
import os
import random
from datetime import datetime, timedelta

def create_toy_stage_data():
    """Create toy stage data files for Phase 4."""
    
    data_dir = "/home/azureuser/ppo/app_2/data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Real job prefixes from production
    job_prefixes = ["JOAW", "JOST", "JOTP", "JOEX", "JOAS"]
    
    # Real machine names from production
    machine_names = [
        "CM03", "CL02", "AD02-50HP", "OV01", "ALDG", "BDS01", "BNDW", 
        "BGPS01", "PAC01", "PAC02", "PCUT01", "SHR01", "MLD01", "MLD02"
    ]
    
    stages = {
        "toy_normal": {"n_families": 10, "n_machines": 5},
        "toy_hard": {"n_families": 15, "n_machines": 5},
        "toy_multi": {"n_families": 10, "n_machines": 8}
    }
    
    for stage_name, config in stages.items():
        print(f"Creating {stage_name} data...")
        
        # Create families with real job IDs
        families = {}
        for i in range(config["n_families"]):
            prefix = random.choice(job_prefixes)
            family_id = f"{prefix}{random.randint(25000000, 25999999)}"
            
            # Determine number of sequences (1-4)
            n_sequences = random.randint(1, 4)
            
            # Create tasks for this family
            tasks = []
            base_processing_time = random.uniform(0.5, 5.0)
            
            for seq in range(1, n_sequences + 1):
                task = {
                    "task_id": f"{family_id}_{seq}/{n_sequences}",
                    "sequence": seq,
                    "processing_time": base_processing_time * random.uniform(0.8, 1.2),
                    "capable_machines": random.sample(list(range(1, config["n_machines"] + 1)), 
                                                    k=random.randint(1, min(3, config["n_machines"]))),
                    "is_important": random.random() < 0.2,
                    "multi_machine": False
                }
                
                # Add multi-machine for toy_multi stage
                if stage_name == "toy_multi" and random.random() < 0.3:
                    task["multi_machine"] = True
                    task["required_machines"] = random.sample(
                        list(range(1, config["n_machines"] + 1)), 
                        k=random.randint(2, 3)
                    )
                
                tasks.append(task)
            
            families[family_id] = {
                "family_id": family_id,
                "tasks": tasks,
                "total_sequences": n_sequences,
                "lcd_days_remaining": random.randint(3, 14),
                "priority": random.choice(["normal", "high", "urgent"])
            }
        
        # Create machines with real names
        machines = []
        for i in range(1, config["n_machines"] + 1):
            machines.append({
                "machine_id": i,
                "machine_name": machine_names[i-1] if i <= len(machine_names) else f"M{i:02d}",
                "machine_type": random.randint(1, 5)
            })
        
        # Create the data structure
        data = {
            "stage": stage_name,
            "families": families,
            "machines": machines,
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_families": len(families),
                "total_machines": len(machines),
                "source": "Generated for Phase 4 testing"
            }
        }
        
        # Save to file
        filename = f"stage_{stage_name}_real_data.json"
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  Saved to {filepath}")
        print(f"  - {len(families)} families with real job IDs")
        print(f"  - {len(machines)} machines")
        print(f"  - {sum(len(f['tasks']) for f in families.values())} total tasks")

if __name__ == "__main__":
    print("Creating toy stage data for Phase 4...")
    print("=" * 60)
    create_toy_stage_data()
    print("\nDone! You can now run generate_strategy_data.py")