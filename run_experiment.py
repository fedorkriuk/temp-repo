#!/usr/bin/env python3
"""
Experiment Runner for Shepherd-Grid
Helps run both basic and advanced experiments with proper AI systems
"""

import sys
import subprocess
import os
import shutil
from typing import Tuple, Dict

# Experiment configurations
BASIC_SCENARIOS = {
    '1': ('scenario1.txt', 'Basic scenario 1'),
    '2': ('scenario2.txt', 'Basic scenario 2'),
    '3': ('scenario3.txt', 'Basic scenario 3'),
    '4': ('scenario4.txt', 'Basic scenario 4'),
    '5': ('scenario_3d_one_target.txt', '3D single target scenario'),
}

ADVANCED_EXPERIMENTS = {
    '1': {
        'module': 'rules_multi_target',
        'class': 'AI_System_MultiTarget',
        'scenario': 'experiment1_multi_target.txt',
        'name': 'Multi-Target Dynamic Priority',
        'description': 'Multiple targets with priority assignment, pack splitting/merging'
    },
    '2': {
        'module': 'rules_evasion',
        'class': 'AI_System_Evasion',
        'scenario': 'experiment2_evasion_patterns.txt',
        'name': 'Adversarial Evasion Patterns',
        'description': 'Targets with zigzag, spiral, random walk, barrel roll evasions'
    },
    '3': {
        'module': 'rules_comm_degradation',
        'class': 'AI_System_CommDegradation',
        'scenario': 'experiment3_comm_degradation.txt',
        'name': 'Communication Degradation',
        'description': 'Jamming zones, signal loss, autonomous behavior'
    },
    '4': {
        'module': 'rules_heterogeneous',
        'class': 'AI_System_Heterogeneous',
        'scenario': 'experiment4_heterogeneous_packs.txt',
        'name': 'Heterogeneous Pack Composition',
        'description': 'Mixed drone types (scout, striker, heavy, sensor) with specialized roles'
    },
    '5': {
        'module': 'rules_energy',
        'class': 'AI_System_Energy',
        'scenario': 'experiment5_energy_constraints.txt',
        'name': 'Energy/Resource Constraints',
        'description': 'Fuel management, efficient routing, return-to-base decisions'
    },
    '6': {
        'module': 'rules_swarm_combat',
        'class': 'AI_System_SwarmCombat',
        'scenario': 'experiment6_swarm_combat.txt',
        'name': 'Swarm vs Swarm Combat',
        'description': 'Large-scale team battles with tactical formations'
    },
}


def print_header():
    """Print welcome header"""
    print("\n" + "="*60)
    print("🚁 SHEPHERD-GRID EXPERIMENT RUNNER 🚁")
    print("="*60)


def print_menu():
    """Print main menu"""
    print("\nSelect experiment type:")
    print("1. Basic Experiments (original scenarios)")
    print("2. Advanced Experiments (ICDTDE2025 publication)")
    print("Q. Quit")
    print("-"*40)


def print_basic_menu():
    """Print basic experiments menu"""
    print("\n📋 BASIC EXPERIMENTS:")
    print("-"*40)
    for key, (scenario, desc) in BASIC_SCENARIOS.items():
        print(f"{key}. {desc} ({scenario})")
    print("B. Back to main menu")
    print("-"*40)


def print_advanced_menu():
    """Print advanced experiments menu"""
    print("\n🚀 ADVANCED EXPERIMENTS:")
    print("-"*60)
    for key, exp in ADVANCED_EXPERIMENTS.items():
        print(f"{key}. {exp['name']}")
        print(f"   {exp['description']}")
    print("B. Back to main menu")
    print("-"*60)


def run_basic_experiment(scenario_file: str):
    """Run a basic experiment with original AI system"""
    print(f"\n▶️  Running basic experiment: {scenario_file}")
    print("   Using standard AI_System from rules.py")
    print("-"*40)
    
    # Try python3 first, then python
    python_cmd = 'python3' if shutil.which('python3') else 'python'
    
    try:
        subprocess.run([python_cmd, '3d_app.py', '--scenario', scenario_file])
    except KeyboardInterrupt:
        print("\n⏹️  Experiment stopped by user")
    except Exception as e:
        print(f"❌ Error running experiment: {e}")


def create_temp_app(module: str, classname: str) -> str:
    """Create temporary 3d_app.py with modified import"""
    # Read original
    with open('3d_app.py', 'r') as f:
        content = f.read()
    
    # Replace import
    old_import = "from rules import AI_System"
    new_import = f"from {module} import {classname} as AI_System"
    new_content = content.replace(old_import, new_import)
    
    # Write temporary file
    temp_filename = '3d_app_temp.py'
    with open(temp_filename, 'w') as f:
        f.write(new_content)
    
    return temp_filename


def run_advanced_experiment(exp_config: Dict):
    """Run an advanced experiment with specialized AI system"""
    print(f"\n▶️  Running advanced experiment: {exp_config['name']}")
    print(f"   Scenario: {exp_config['scenario']}")
    print(f"   AI System: {exp_config['class']} from {exp_config['module']}")
    print("-"*60)
    
    # Try python3 first, then python
    python_cmd = 'python3' if shutil.which('python3') else 'python'
    
    # Create temporary app file
    temp_file = None
    try:
        temp_file = create_temp_app(exp_config['module'], exp_config['class'])
        
        # Run the experiment
        subprocess.run([python_cmd, temp_file, '--scenario', exp_config['scenario']])
        
    except KeyboardInterrupt:
        print("\n⏹️  Experiment stopped by user")
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
    finally:
        # Cleanup
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


def print_experiment_tips(exp_type: str, exp_num: str):
    """Print tips for capturing good visuals"""
    print("\n💡 TIPS FOR CAPTURING VISUALS:")
    
    if exp_type == 'advanced':
        tips = {
            '1': [
                "Watch for pack splitting when targets diverge",
                "Notice priority indicators in console output",
                "Capture moment when packs merge after elimination"
            ],
            '2': [
                "Best visuals during spiral and barrel roll patterns",
                "Watch prediction lines adapt to evasion",
                "Capture failed intercept attempts on random walk"
            ],
            '3': [
                "Look for drones becoming isolated (console shows ❌)",
                "Watch autonomous behavior in jamming zones",
                "Capture emergency RTB movements"
            ],
            '4': [
                "Notice different speeds between drone types",
                "Watch role-based formation positions",
                "Capture scout drones leading the pack"
            ],
            '5': [
                "Monitor fuel percentages in console",
                "Watch for RTB decisions (⚠️ BINGO FUEL)",
                "Capture efficient vs aggressive pursuit paths"
            ],
            '6': [
                "Best during large formation movements",
                "Watch for coordinated strikes (💥)",
                "Capture tactical formations (wedge, sphere)"
            ]
        }
        
        if exp_num in tips:
            for tip in tips[exp_num]:
                print(f"  • {tip}")
    else:
        print("  • Press SPACE to pause/unpause simulation")
        print("  • Use mouse to rotate camera view")
        print("  • Watch for successful intercepts (💥 in console)")


def main():
    """Main program loop"""
    print_header()
    
    while True:
        print_menu()
        choice = input("Enter choice: ").strip().upper()
        
        if choice == 'Q':
            print("\n👋 Goodbye!")
            break
        elif choice == '1':
            # Basic experiments
            while True:
                print_basic_menu()
                basic_choice = input("Select basic experiment: ").strip().upper()
                
                if basic_choice == 'B':
                    break
                elif basic_choice in BASIC_SCENARIOS:
                    scenario, _ = BASIC_SCENARIOS[basic_choice]
                    print_experiment_tips('basic', basic_choice)
                    input("\nPress Enter to start experiment...")
                    run_basic_experiment(scenario)
                    input("\nPress Enter to continue...")
                else:
                    print("❌ Invalid choice!")
                    
        elif choice == '2':
            # Advanced experiments
            while True:
                print_advanced_menu()
                adv_choice = input("Select advanced experiment: ").strip().upper()
                
                if adv_choice == 'B':
                    break
                elif adv_choice in ADVANCED_EXPERIMENTS:
                    exp_config = ADVANCED_EXPERIMENTS[adv_choice]
                    print_experiment_tips('advanced', adv_choice)
                    input("\nPress Enter to start experiment...")
                    run_advanced_experiment(exp_config)
                    input("\nPress Enter to continue...")
                else:
                    print("❌ Invalid choice!")
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists('3d_app.py'):
        print("❌ Error: 3d_app.py not found!")
        print("   Please run this script from the shepherd-grid directory")
        sys.exit(1)
    
    # Check for required files
    missing_files = []
    
    # Check basic scenarios
    for scenario, _ in BASIC_SCENARIOS.values():
        if not os.path.exists(scenario):
            missing_files.append(scenario)
    
    # Check advanced experiments
    for exp in ADVANCED_EXPERIMENTS.values():
        if not os.path.exists(exp['scenario']):
            missing_files.append(exp['scenario'])
        module_file = exp['module'] + '.py'
        if not os.path.exists(module_file):
            missing_files.append(module_file)
    
    if missing_files:
        print("⚠️  Warning: Some files are missing:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nSome experiments may not work properly.")
        input("Press Enter to continue anyway...")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)