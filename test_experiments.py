#!/usr/bin/env python3
"""
Test script to verify all experiments can be loaded without errors
"""

import sys
import importlib

def test_imports():
    """Test that all AI modules can be imported"""
    modules = [
        ('rules', 'AI_System'),
        ('rules_multi_target', 'AI_System_MultiTarget'),
        ('rules_evasion', 'AI_System_Evasion'),
        ('rules_comm_degradation', 'AI_System_CommDegradation'),
        ('rules_heterogeneous', 'AI_System_Heterogeneous'),
        ('rules_energy', 'AI_System_Energy'),
        ('rules_swarm_combat', 'AI_System_SwarmCombat'),
    ]
    
    print("Testing module imports...")
    for module_name, class_name in modules:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name} - OK")
        except Exception as e:
            print(f"❌ {module_name}.{class_name} - ERROR: {e}")
            return False
    
    return True

def test_scenario_parsing():
    """Test that scenario files can be parsed"""
    # Import the parsing functions
    try:
        # Read and extract the functions we need
        with open('3d_app.py', 'r') as f:
            code = f.read()
        
        # Create a namespace and exec the code
        namespace = {}
        exec(code, namespace)
        
        # Get the functions we need
        parse_scenario_config = namespace['parse_scenario_config']
        load_scenario_file = namespace['load_scenario_file']
    except Exception as e:
        print(f"Error loading 3d_app.py functions: {e}")
        return False
    
    scenarios = [
        'scenario1.txt',
        'scenario2.txt', 
        'scenario3.txt',
        'scenario4.txt',
        'scenario_3d_one_target.txt',
        'experiment1_multi_target.txt',
        'experiment2_evasion_patterns.txt',
        'experiment3_comm_degradation.txt',
        'experiment4_heterogeneous_packs.txt',
        'experiment5_energy_constraints.txt',
        'experiment6_swarm_combat.txt',
    ]
    
    print("\nTesting scenario file parsing...")
    for scenario in scenarios:
        try:
            content = load_scenario_file(scenario)
            config = parse_scenario_config(content)
            
            # Basic validation
            assert 'targets' in config
            assert 'interceptors' in config
            assert len(config['targets']) > 0
            assert len(config['interceptors']) > 0
            
            print(f"✅ {scenario} - {len(config['targets'])} targets, {len(config['interceptors'])} interceptors")
        except FileNotFoundError:
            print(f"⚠️  {scenario} - File not found (might be OK if basic scenario)")
        except Exception as e:
            print(f"❌ {scenario} - ERROR: {e}")
            return False
    
    return True

def test_ai_initialization():
    """Test that AI systems can be initialized with scenario configs"""
    import importlib
    
    test_config = {
        'physics': {
            'green_max_acceleration': 70.0,
            'green_max_velocity': 220.0,
        },
        'ai_parameters': {
            'pack_formation_distance': 400.0,
        },
        'mission_parameters': {},
        'environment': {}
    }
    
    ai_systems = [
        ('rules', 'AI_System'),
        ('rules_multi_target', 'AI_System_MultiTarget'),
        ('rules_evasion', 'AI_System_Evasion'),
        ('rules_comm_degradation', 'AI_System_CommDegradation'),
        ('rules_heterogeneous', 'AI_System_Heterogeneous'),
        ('rules_energy', 'AI_System_Energy'),
        ('rules_swarm_combat', 'AI_System_SwarmCombat'),
    ]
    
    print("\nTesting AI system initialization...")
    for module_name, class_name in ai_systems:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            ai = cls(test_config)
            print(f"✅ {class_name} initialized successfully")
        except Exception as e:
            print(f"❌ {class_name} - ERROR: {e}")
            return False
    
    return True

def main():
    print("🔍 SHEPHERD-GRID EXPERIMENT VALIDATION")
    print("="*50)
    
    all_good = True
    
    # Test 1: Module imports
    if not test_imports():
        all_good = False
    
    # Test 2: Scenario parsing
    if not test_scenario_parsing():
        all_good = False
    
    # Test 3: AI initialization
    if not test_ai_initialization():
        all_good = False
    
    print("\n" + "="*50)
    if all_good:
        print("✅ All tests passed! Experiments should run correctly.")
        print("\nRequired packages:")
        print("  - Flask==2.3.3")
        print("  - numpy==1.24.3")
        print("\nInstall with: pip3 install -r requirements.txt")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())