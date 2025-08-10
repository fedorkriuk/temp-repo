import math
from typing import List, Dict, Tuple, Optional

class AI_System:
    """SCENARIO-DRIVEN PACK SYSTEM: All parameters from scenario file"""
    
    def __init__(self, scenario_config: Dict = None):
        # DEFAULT VALUES (overridden by scenario)
        self.pack_formation_distance = 70.0
        self.pack_activation_distance = 125.0
        self.strike_distance = 8.0
        
        self.green_max_velocity = 60.0
        self.green_max_acceleration = 30.0
        
        self.killer_max_velocity = 120.0
        self.killer_max_acceleration = 80.0
        self.killer_deceleration_multiplier = 2.0
        
        self.strike_acceleration_multiplier = 3.0
        
        # PACK DATA
        self.packs = {}
        self.drone_pack_map = {}
        self.packs_initialized = False
        
        self.frame_count = 0
        self.mission_time = 0.0
        self.dt = 1/30
        
        # LOAD ALL PARAMETERS FROM SCENARIO
        if scenario_config:
            self._load_all_parameters(scenario_config)
        
        print(f"🐺 SCENARIO-DRIVEN PACK SYSTEM: All parameters from scenario file")
    
    def _load_all_parameters(self, scenario_config: Dict):
        """Load ALL parameters from scenario file"""
        try:
            # PHYSICS PARAMETERS
            physics = scenario_config.get('physics', {})
            self.green_max_acceleration = float(physics.get('green_max_acceleration', 30.0))
            self.green_max_velocity = float(physics.get('green_max_velocity', 60.0))
            self.killer_max_acceleration = float(physics.get('killer_max_acceleration', 80.0))
            self.killer_max_velocity = float(physics.get('killer_max_velocity', 120.0))
            self.killer_deceleration_multiplier = float(physics.get('killer_deceleration_multiplier', 2.0))
            self.strike_acceleration_multiplier = float(physics.get('strike_acceleration_multiplier', 3.0))
            
            # AI PARAMETERS
            ai_params = scenario_config.get('ai_parameters', {})
            self.pack_formation_distance = float(ai_params.get('pack_formation_distance', 70.0))
            self.pack_activation_distance = float(ai_params.get('pack_activation_distance', 125.0))
            self.strike_distance = float(ai_params.get('strike_distance', 8.0))
            
            # MISSION PARAMETERS
            mission_params = scenario_config.get('mission_parameters', {})
            self.max_engagement_range = float(mission_params.get('max_engagement_range', 200.0))
            self.role_switch_distance = float(mission_params.get('role_switch_distance', 150.0))
            
            print(f"🎯 SCENARIO CONFIG LOADED:")
            print(f"   Green: {self.green_max_velocity}m/s, {self.green_max_acceleration}acc")
            print(f"   Killer: {self.killer_max_velocity}m/s, {self.killer_max_acceleration}acc")
            print(f"   Formation: {self.pack_formation_distance}m, Activation: {self.pack_activation_distance}m")
            print(f"   Strike: {self.strike_distance}m, Multiplier: {self.strike_acceleration_multiplier}x")
                
        except Exception as e:
            print(f"⚠️ Parameter loading error: {e} - using defaults")
    
    def update(self, targets: List[Dict], interceptors: List[Dict], dt: float, wind: Dict, scenario_config: Dict = None) -> Dict[str, Dict]:
        """Main update with scenario-driven parameters"""
        
        # Reload parameters if scenario changes
        if scenario_config:
            self._load_all_parameters(scenario_config)
        
        self.frame_count += 1
        self.mission_time += dt
        self.dt = dt
        
        # Step 1: Physics update
        self._update_agile_physics(targets, interceptors, dt, wind)
        
        # Step 2: Initialize packs (once)
        if not self.packs_initialized:
            self._create_independent_packs(targets, interceptors)
            self.packs_initialized = True
        
        # Step 3: Update each pack
        decisions = {}
        for pack_id, pack_data in self.packs.items():
            pack_decisions = self._update_independent_pack(pack_data, targets, interceptors)
            decisions.update(pack_decisions)
        
        return decisions
    
    def _update_agile_physics(self, targets: List[Dict], interceptors: List[Dict], dt: float, wind: Dict):
        """Physics with scenario-defined agility"""
        
        # Update targets
        for target in targets:
            if not target.get('active', True):
                continue
                
            wind_x = wind.get('x', 0)
            wind_y = wind.get('y', 0)
            
            target['x'] += (target.get('vx', 0) + wind_x * 0.1) * dt
            target['y'] += (target.get('vy', 0) + wind_y * 0.1) * dt
            target['z'] = target.get('z', 0) + target.get('vz', 0) * dt
        
        # Update interceptors with scenario physics
        for interceptor in interceptors:
            if not interceptor.get('active', True):
                continue
            
            # Determine drone type from scenario
            pack_id = self.drone_pack_map.get(interceptor['id'])
            is_killer = False
            if pack_id and pack_id in self.packs:
                pack_data = self.packs[pack_id]
                is_killer = (pack_data['killer_drone'] == interceptor['id'])
            
            # Apply scenario-defined physics
            if is_killer:
                max_accel = self.killer_max_acceleration
                max_vel = self.killer_max_velocity
                
                # Enhanced braking from scenario
                current_vx = interceptor.get('vx', 0)
                current_vy = interceptor.get('vy', 0)
                current_speed = math.sqrt(current_vx**2 + current_vy**2)
                
                ax = interceptor.get('ax', 0)
                ay = interceptor.get('ay', 0)
                
                if current_speed > 1.0:
                    vel_ax_dot = (current_vx * ax + current_vy * ay) / current_speed
                    if vel_ax_dot < 0:  # Braking
                        max_accel *= self.killer_deceleration_multiplier
                
            else:
                max_accel = self.green_max_acceleration
                max_vel = self.green_max_velocity
            
            ax = interceptor.get('ax', 0)
            ay = interceptor.get('ay', 0)
            az = interceptor.get('az', 0)
            
            # Apply scenario limits
            accel_mag = math.sqrt(ax**2 + ay**2 + az**2)
            if accel_mag > max_accel:
                scale = max_accel / accel_mag
                ax *= scale
                ay *= scale
                az *= scale
            
            # Update velocity
            interceptor['vx'] = interceptor.get('vx', 0) + ax * dt
            interceptor['vy'] = interceptor.get('vy', 0) + ay * dt
            interceptor['vz'] = interceptor.get('vz', 0) + az * dt
            
            # Apply scenario velocity limits
            speed = math.sqrt(interceptor['vx']**2 + interceptor['vy']**2 + interceptor['vz']**2)
            if speed > max_vel:
                scale = max_vel / speed
                interceptor['vx'] *= scale
                interceptor['vy'] *= scale
                interceptor['vz'] *= scale
            
            # Update position
            interceptor['x'] += interceptor['vx'] * dt
            interceptor['y'] += interceptor['vy'] * dt
            interceptor['z'] = interceptor.get('z', 0) + interceptor['vz'] * dt
    
    def _create_independent_packs(self, targets: List[Dict], interceptors: List[Dict]):
        """Create packs with scenario parameters"""
        
        active_targets = [t for t in targets if t.get('active', True)]
        active_interceptors = [i for i in interceptors if i.get('active', True)]
        
        print(f"🐺 CREATING SCENARIO PACKS: {len(active_targets)} targets, {len(active_interceptors)} drones")
        
        drone_index = 0
        
        for target_idx, target in enumerate(active_targets):
            target_id = target['id']
            pack_id = f"pack_{target_id}"
            
            pack_drones = []
            for slot in range(4):
                if drone_index < len(active_interceptors):
                    drone = active_interceptors[drone_index]
                    pack_drones.append(drone['id'])
                    self.drone_pack_map[drone['id']] = pack_id
                    drone_index += 1
            
            if len(pack_drones) > 0:
                self.packs[pack_id] = {
                    'target_id': target_id,
                    'drone_ids': pack_drones,
                    'killer_drone': pack_drones[0] if pack_drones else None,
                    'green_drones': pack_drones[1:] if len(pack_drones) > 1 else [],
                    'pack_state': 'forming',
                    'formation_positions': {},
                    'failed_strikes': []
                }
                
                print(f"🎯 SCENARIO PACK [{pack_id}]: Killer {pack_drones[0]}, Green {pack_drones[1:]}")
        
        print(f"🐺 SCENARIO PACK CREATION COMPLETE")
    
    def _update_independent_pack(self, pack_data: Dict, targets: List[Dict], interceptors: List[Dict]) -> Dict[str, Dict]:
        """Update pack with scenario parameters"""
        
        pack_decisions = {}
        target_id = pack_data['target_id']
        
        # Find target
        target = next((t for t in targets if t['id'] == target_id and t.get('active', True)), None)
        if not target:
            for drone_id in pack_data['drone_ids']:
                drone = next((i for i in interceptors if i['id'] == drone_id), None)
                if drone:
                    pack_decisions[drone_id] = self._create_decision(0, 0, 0, target_id, 'idle', 'target_destroyed')
            return pack_decisions
        
        # Get active drones
        pack_drones = []
        for drone_id in pack_data['drone_ids']:
            drone = next((i for i in interceptors if i['id'] == drone_id and i.get('active', True)), None)
            if drone:
                pack_drones.append(drone)
        
        if not pack_drones:
            return pack_decisions
        
        # Calculate formation
        self._calculate_scenario_formation(pack_data, target, pack_drones)
        
        # Check readiness
        pack_ready = self._check_scenario_readiness(pack_data, target, pack_drones)
        
        # Manage roles
        self._manage_scenario_roles(pack_data, target, pack_drones)
        
        # Generate decisions
        for drone in pack_drones:
            decision = self._generate_scenario_decision(pack_data, target, drone, pack_ready)
            pack_decisions[drone['id']] = decision
        
        return pack_decisions
    
    def _calculate_scenario_formation(self, pack_data: Dict, target: Dict, pack_drones: List[Dict]):
        """Calculate formation with scenario distance"""
        
        target_vx = target.get('vx', 0)
        target_vy = target.get('vy', 0)
        target_speed = math.sqrt(target_vx**2 + target_vy**2)
        
        if target_speed > 1.0:
            target_heading = math.atan2(target_vy, target_vx)
        else:
            target_heading = 0
        
        grid_angles = [0, math.pi/2, math.pi, 3*math.pi/2]
        formation_positions = {}
        
        for i, drone_id in enumerate(pack_data['drone_ids']):
            if i < len(grid_angles):
                angle = target_heading + grid_angles[i]
                
                # Use scenario formation distance
                pos_x = target['x'] + self.pack_formation_distance * math.cos(angle)
                pos_y = target['y'] + self.pack_formation_distance * math.sin(angle)
                pos_z = target.get('z', 0)
                
                formation_positions[drone_id] = {
                    'x': pos_x, 'y': pos_y, 'z': pos_z,
                    'angle': math.degrees(angle) % 360,
                    'grid_position': i
                }
        
        pack_data['formation_positions'] = formation_positions
    
    def _check_scenario_readiness(self, pack_data: Dict, target: Dict, pack_drones: List[Dict]) -> bool:
        """Check readiness with scenario activation distance"""
        
        distances = []
        for drone in pack_drones:
            distance = self._distance_3d(drone, target)
            distances.append(distance)
        
        # Use scenario activation distance
        all_ready = all(d <= self.pack_activation_distance for d in distances)
        
        if all_ready and pack_data['pack_state'] != 'ready':
            pack_data['pack_state'] = 'ready'
            print(f"🎯 PACK READY [{pack_data['target_id']}]: All within {self.pack_activation_distance}m")
        
        return all_ready
    
    def _manage_scenario_roles(self, pack_data: Dict, target: Dict, pack_drones: List[Dict]):
        """Manage roles with scenario switch distance"""
        
        current_killer = pack_data['killer_drone']
        killer_drone = next((d for d in pack_drones if d['id'] == current_killer), None)
        
        if not killer_drone:
            self._promote_new_killer(pack_data, target, pack_drones)
            return
        
        killer_distance = self._distance_3d(killer_drone, target)
        
        # Use scenario role switch distance
        if killer_distance > self.role_switch_distance and len(pack_data['green_drones']) > 0:
            closest_green = None
            closest_distance = float('inf')
            
            for drone in pack_drones:
                if drone['id'] in pack_data['green_drones']:
                    distance = self._distance_3d(drone, target)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_green = drone
            
            if closest_green and closest_distance < killer_distance - 30.0:
                self._switch_killer(pack_data, current_killer, closest_green['id'])
    
    def _promote_new_killer(self, pack_data: Dict, target: Dict, pack_drones: List[Dict]):
        """Promote new killer"""
        if not pack_data['green_drones']:
            return
        
        closest_green = None
        closest_distance = float('inf')
        
        for drone in pack_drones:
            if drone['id'] in pack_data['green_drones']:
                distance = self._distance_3d(drone, target)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_green = drone
        
        if closest_green:
            old_killer = pack_data['killer_drone']
            new_killer = closest_green['id']
            
            pack_data['killer_drone'] = new_killer
            pack_data['green_drones'].remove(new_killer)
            
            if old_killer and old_killer not in pack_data['failed_strikes']:
                pack_data['green_drones'].append(old_killer)
            
            print(f"🔄 SCENARIO KILLER PROMOTED: {new_killer}")
    
    def _switch_killer(self, pack_data: Dict, old_killer: str, new_killer: str):
        """Switch killer roles"""
        pack_data['killer_drone'] = new_killer
        pack_data['green_drones'].remove(new_killer)
        pack_data['green_drones'].append(old_killer)
        
        print(f"🔄 SCENARIO KILLER SWITCH: {new_killer} → killer, {old_killer} → green")
    
    def _generate_scenario_decision(self, pack_data: Dict, target: Dict, drone: Dict, pack_ready: bool) -> Dict:
        """Generate decision with scenario parameters"""
        
        drone_id = drone['id']
        target_id = target['id']
        distance = self._distance_3d(drone, target)
        
        # Check strike success with scenario distance
        if distance <= self.strike_distance:
            target['active'] = False
            drone['active'] = False
            
            print(f"💥💥💥 SCENARIO STRIKE SUCCESS: {drone_id} destroyed {target_id}!")
            return self._create_decision(0, 0, 0, target_id, 'killer', 'STRIKE_SUCCESS')
        
        formation_pos = pack_data['formation_positions'].get(drone_id)
        if not formation_pos:
            return self._create_decision(0, 0, 0, target_id, 'unknown', 'no_formation')
        
        is_killer = (pack_data['killer_drone'] == drone_id)
        
        if is_killer and pack_ready:
            return self._generate_scenario_killer_behavior(drone, target, distance)
        else:
            return self._maintain_scenario_formation(drone, target, formation_pos)
    
    def _generate_scenario_killer_behavior(self, drone: Dict, target: Dict, distance: float) -> Dict:
        """Killer behavior with scenario acceleration"""
        
        target_id = target['id']
        
        dx = target['x'] - drone['x']
        dy = target['y'] - drone['y']
        dz = target.get('z', 0) - drone.get('z', 0)
        strike_distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if strike_distance < 1.0:
            return self._create_decision(0, 0, 0, target_id, 'killer', 'on_target')
        
        current_vx = drone.get('vx', 0)
        current_vy = drone.get('vy', 0)
        current_speed = math.sqrt(current_vx**2 + current_vy**2)
        
        # Use scenario killer velocity
        desired_speed = min(self.killer_max_velocity, distance * 2)
        desired_vx = desired_speed * (dx / strike_distance)
        desired_vy = desired_speed * (dy / strike_distance)
        
        # Use scenario acceleration multiplier
        agile_accel = self.killer_max_acceleration * self.strike_acceleration_multiplier
        ax = agile_accel * (desired_vx - current_vx) / max(1.0, current_speed)
        ay = agile_accel * (desired_vy - current_vy) / max(1.0, current_speed)
        az = agile_accel * (dz / strike_distance)
        
        return self._create_decision(ax, ay, az, target_id, 'killer', 'SCENARIO_STRIKE')
    
    def _maintain_scenario_formation(self, drone: Dict, target: Dict, formation_pos: Dict) -> Dict:
        """Formation maintenance with scenario parameters"""
        
        target_id = target['id']
        
        dx = formation_pos['x'] - drone['x']
        dy = formation_pos['y'] - drone['y']
        dz = formation_pos['z'] - drone.get('z', 0)
        formation_distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if formation_distance < 10.0:
            target_vx = target.get('vx', 0)
            target_vy = target.get('vy', 0)
            target_vz = target.get('vz', 0)
            
            current_vx = drone.get('vx', 0)
            current_vy = drone.get('vy', 0)
            current_vz = drone.get('vz', 0)
            
            ax = (target_vx - current_vx) * 3.0
            ay = (target_vy - current_vy) * 3.0
            az = (target_vz - current_vz) * 3.0
            
            return self._create_decision(ax, ay, az, target_id, 'green', 'in_formation')
        else:
            # Use scenario green acceleration
            ax = self.green_max_acceleration * 0.8 * (dx / formation_distance)
            ay = self.green_max_acceleration * 0.8 * (dy / formation_distance)
            az = self.green_max_acceleration * 0.8 * (dz / formation_distance)
            
            return self._create_decision(ax, ay, az, target_id, 'green', 'moving_to_formation')
    
    def _create_decision(self, ax: float, ay: float, az: float, target_id: str, role: str, status: str, **kwargs) -> Dict:
        return {'ax': ax, 'ay': ay, 'az': az, 'target_id': target_id, 'role': role, 'status': status, **kwargs}
    
    def _distance_3d(self, obj1: Dict, obj2: Dict) -> float:
        dx = obj1['x'] - obj2['x']
        dy = obj1['y'] - obj2['y']
        dz = obj1.get('z', 0) - obj2.get('z', 0)
        return math.sqrt(dx*dx + dy*dy + dz*dz)