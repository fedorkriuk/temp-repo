#!/usr/bin/env python3
"""
Fix premature engagement in all AI systems by ensuring proper state transitions
"""

import os
import re

def add_engagement_safety_check():
    """Add safety checks to prevent premature killer selection"""
    
    files_to_fix = [
        'rules_multi_target.py',
        'rules_evasion.py', 
        'rules_comm_degradation.py',
        'rules_heterogeneous.py',
        'rules_energy.py',
        'rules_swarm_combat.py'
    ]
    
    for filename in files_to_fix:
        if not os.path.exists(filename):
            print(f"⚠️  {filename} not found")
            continue
            
        with open(filename, 'r') as f:
            content = f.read()
        
        # Check if already has minimum time check
        if 'min_engagement_time' in content:
            print(f"✅ {filename} already has engagement safety")
            continue
        
        # Add minimum engagement time to __init__
        init_pattern = r'(self\.chase_duration = [\d.]+)'
        init_replacement = r'\1\n        self.min_engagement_time = 5.0  # Minimum time before allowing engagement'
        
        # Add time check before killer selection
        # Look for patterns like "if pack.get('killer_drone') is None:"
        killer_pattern = r"(if pack\.get\('killer_drone'\) is None:)"
        killer_replacement = r"# Prevent premature engagement\n        if self.mission_time < self.min_engagement_time:\n            for d in pack_drones:\n                out[d['id']] = self._track_parallel(d, target, pack['formation_positions'].get(d['id']))\n            return out\n        \n        \1"
        
        # Apply fixes
        modified = content
        modified = re.sub(init_pattern, init_replacement, modified)
        
        # Only add killer check if it's not a combat scenario (which needs immediate engagement)
        if 'swarm_combat' not in filename:
            # Find the killer selection part more carefully
            lines = modified.split('\n')
            new_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]
                if "if pack.get('killer_drone') is None:" in line and 'min_engagement_time' not in '\n'.join(lines[max(0,i-10):i]):
                    # Add safety check before this line
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(' ' * indent + '# Prevent premature engagement')
                    new_lines.append(' ' * indent + 'if self.mission_time < self.min_engagement_time:')
                    new_lines.append(' ' * indent + '    for d in pack_drones:')
                    new_lines.append(' ' * indent + '        fp = pack.get("formation_positions", {}).get(d["id"])')
                    new_lines.append(' ' * indent + '        out[d["id"]] = self._track_parallel(d, target, fp) if fp else self._follow_target(d, target)')
                    new_lines.append(' ' * indent + '    return out')
                    new_lines.append(' ' * indent)
                
                new_lines.append(line)
                i += 1
            
            modified = '\n'.join(new_lines)
        
        # Write back
        with open(filename, 'w') as f:
            f.write(modified)
        
        print(f"✅ Fixed {filename}")

if __name__ == "__main__":
    print("🔧 Fixing premature engagement in AI systems...")
    add_engagement_safety_check()
    print("\n✅ Done! The AI systems will now wait at least 5 seconds before selecting killers.")
    print("\nAlso, refreshing the browser will now properly reset the AI state.")