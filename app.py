from flask import Flask, render_template, jsonify, request
from rules import AI_System
import argparse
import os
import sys

app = Flask(__name__)
ai_system = None
current_scenario = "scenario1.txt"  # Default scenario

def parse_scenario_config(content: str) -> dict:
    """Parse scenario file into structured config"""
    config = {
        'physics': {},
        'ai_parameters': {},
        'mission_parameters': {},
        'environment': {}
    }
    
    lines = content.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Section headers
        if line.startsWith('[') and line.endswith(']'):
            section_name = line[1:-1].lower()
            current_section = section_name
            continue
        
        # Key-value pairs
        if '=' in line and current_section:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Map sections to config
            if current_section == 'physics':
                config['physics'][key] = value
            elif current_section == 'ai_parameters':
                config['ai_parameters'][key] = value
            elif current_section == 'mission_parameters':
                config['mission_parameters'][key] = value
            elif current_section == 'environment':
                config['environment'][key] = value
    
    return config

def load_scenario_file(scenario_name: str) -> str:
    """Load scenario file with error handling"""
    try:
        scenario_path = scenario_name
        if not os.path.exists(scenario_path):
            # Try with .txt extension
            scenario_path = f"{scenario_name}.txt"
            if not os.path.exists(scenario_path):
                # Try in scenarios directory
                scenario_path = f"scenarios/{scenario_name}"
                if not os.path.exists(scenario_path):
                    scenario_path = f"scenarios/{scenario_name}.txt"
        
        if not os.path.exists(scenario_path):
            raise FileNotFoundError(f"Scenario file not found: {scenario_name}")
        
        with open(scenario_path, 'r') as f:
            content = f.read()
        
        print(f"🎯 SCENARIO LOADED: {scenario_path}")
        return content
        
    except Exception as e:
        print(f"❌ ERROR loading scenario {scenario_name}: {e}")
        # Fallback to default
        with open('scenario1.txt', 'r') as f:
            return f.read()

@app.route('/')
def index():
    return render_template('radar.html')

@app.route('/api/scenario')
def get_scenario():
    try:
        content = load_scenario_file(current_scenario)
        return jsonify({
            'scenario': content,
            'scenario_name': current_scenario
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai_update', methods=['POST'])
def ai_update():
    global ai_system
    
    try:
        data = request.json
        
        # Initialize AI system with scenario config if first time
        if ai_system is None:
            scenario_config = data.get('scenario_config', {})
            ai_system = AI_System(scenario_config)
        
        # AI processes everything
        decisions = ai_system.update(
            data['targets'], 
            data['interceptors'], 
            data['dt'],
            data.get('wind', {'x': 2.0, 'y': -1.0}),
            data.get('scenario_config', {})
        )
        
        # Return updated objects and decisions
        return jsonify({
            'decisions': decisions,
            'updated_targets': data['targets'],
            'updated_interceptors': data['interceptors']
        })
        
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Agile Killer Pack Defense System')
    parser.add_argument('--scenario', '-s', 
                       default='scenario1.txt',
                       help='Scenario file to load (default: scenario1.txt)')
    parser.add_argument('--port', '-p', 
                       type=int, default=8903,
                       help='Port to run server on (default: 8903)')
    parser.add_argument('--debug', '-d', 
                       action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Set global scenario
    current_scenario = args.scenario
    
    # Validate scenario file exists
    if not os.path.exists(current_scenario):
        if not os.path.exists(f"{current_scenario}.txt"):
            print(f"❌ ERROR: Scenario file '{current_scenario}' not found!")
            print("Available scenarios:")
            for file in os.listdir('.'):
                if file.startswith('scenario') and file.endswith('.txt'):
                    print(f"  - {file}")
            sys.exit(1)
        else:
            current_scenario = f"{current_scenario}.txt"
    
    print(f"🚀 STARTING AGILE KILLER SYSTEM")
    print(f"🎯 Scenario: {current_scenario}")
    print(f"🌐 Server: http://localhost:{args.port}")
    print(f"🐺 Ready for pack defense!")
    
    app.run(debug=args.debug, port=args.port, host='0.0.0.0')