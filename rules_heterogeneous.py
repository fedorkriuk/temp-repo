import math
from typing import List, Dict, Optional, Tuple
import numpy as np

class AI_System_Heterogeneous:
    """Pack logic with heterogeneous drone capabilities and role optimization"""

    def __init__(self, scenario_config: Dict = None):
        # Base pack parameters
        self.pack_formation_distance = 400.0
        self.pack_activation_distance = 1000.0
        self.strike_distance = 35.0
        
        # Heterogeneous parameters
        self.role_assignment_strategy = 'capability_based'
        self.scout_detection_bonus = 2.0
        self.sensor_prediction_bonus = 1.5
        self.heavy_strike_bonus = 1.8
        
        self.heterogeneous_formation_spacing = 1.2
        self.speed_matching_tolerance = 50.0
        self.capability_weight_matrix = True
        
        # Role-specific behaviors
        self.scout_advance_distance = 200.0
        self.sensor_optimal_range = 3000.0
        self.heavy_engagement_range = 100.0
        
        # Drone type specifications
        self.drone_specs = {
            'scout_drone': {
                'max_acceleration': 120.0,
                'max_velocity': 350.0,
                'payload_capacity': 0.5,
                'detection_range': 2000.0,
                'role_preference': ['scout', 'killer'],
                'color': 'yellow'
            },
            'fast_interceptor': {
                'max_acceleration': 100.0,
                'max_velocity': 300.0,
                'payload_capacity': 1.0,
                'detection_range': 1500.0,
                'role_preference': ['killer', 'pursuit'],
                'color': 'blue'
            },
            'heavy_interceptor': {
                'max_acceleration': 60.0,
                'max_velocity': 200.0,
                'payload_capacity': 2.0,
                'detection_range': 1000.0,
                'role_preference': ['killer', 'formation'],
                'color': 'red'
            },
            'sensor_drone': {
                'max_acceleration': 80.0,
                'max_velocity': 250.0,
                'payload_capacity': 0.8,
                'detection_range': 5000.0,
                'tracking_accuracy': 0.95,
                'role_preference': ['sensor', 'coordinator'],
                'color': 'green'
            }
        }
        
        # Default specs for standard drones
        self.default_specs = {
            'max_acceleration': 75.0,
            'max_velocity': 240.0,
            'payload_capacity': 1.0,
            'detection_range': 1500.0,
            'role_preference': ['green', 'killer'],
            'color': 'gray'
        }
        
        # Killer role multipliers
        self.killer_acceleration_multiplier = 1.5
        self.killer_velocity_multiplier = 1.2
        self.killer_deceleration_multiplier = 3.5
        
        self.strike_acceleration_multiplier = 5.0
        
        # Mission parameters
        self.max_engagement_range = 12000.0
        self.cooperation_radius = 3500.0
        self.communication_range = 20000.0
        self.prediction_horizon = 3.5
        
        # Green orbit parameters
        self.ring_orbit_speed = 30.0
        self.ring_orbit_direction = 1
        self.green_speed_margin = 20.0
        
        # State tracking
        self.chase_duration = 8.0
        self.follow_distance = 600.0
        self.ready_epsilon = 150.0
        
        # Runtime state
        self.packs: Dict[str, Dict] = {}
        self.drone_pack_map: Dict[str, str] = {}
        self.drone_capabilities: Dict[str, Dict] = {}
        self.packs_initialized = False
        
        # Role assignment tracking
        self.drone_roles: Dict[str, str] = {}
        self.role_performance: Dict[str, Dict] = {}
        
        self.frame_count = 0
        self.mission_time = 0.0
        self.dt = 1.0 / 30.0
        
        if scenario_config:
            self._load_all_parameters(scenario_config)
            
        print("=" * 60)
        print("🚁 HETEROGENEOUS PACK SYSTEM INITIALIZED")
        print(f"   Drone types: {len(self.drone_specs)}")
        print(f"   Role assignment: {self.role_assignment_strategy}")
        print(f"   Formation spacing factor: {self.heterogeneous_formation_spacing}")
        print("=" * 60)

    def _load_all_parameters(self, scenario_config: Dict):
        try:
            physics = scenario_config.get('physics', {})
            
            # Load type-specific physics
            for drone_type in ['scout', 'fast', 'heavy', 'sensor']:
                if f'{drone_type}_max_acceleration' in physics:
                    type_key = f'{drone_type}_drone' if drone_type == 'scout' or drone_type == 'sensor' else f'{drone_type}_interceptor'
                    if type_key not in self.drone_specs:
                        self.drone_specs[type_key] = self.default_specs.copy()
                    
                    self.drone_specs[type_key]['max_acceleration'] = float(physics.get(f'{drone_type}_max_acceleration'))
                    self.drone_specs[type_key]['max_velocity'] = float(physics.get(f'{drone_type}_max_velocity'))
                    
                    if f'{drone_type}_payload_capacity' in physics:
                        self.drone_specs[type_key]['payload_capacity'] = float(physics.get(f'{drone_type}_payload_capacity'))
                    if f'{drone_type}_detection_range' in physics:
                        self.drone_specs[type_key]['detection_range'] = float(physics.get(f'{drone_type}_detection_range'))
            
            self.killer_acceleration_multiplier = float(physics.get('killer_acceleration_multiplier', self.killer_acceleration_multiplier))
            self.killer_velocity_multiplier = float(physics.get('killer_velocity_multiplier', self.killer_velocity_multiplier))
            self.killer_deceleration_multiplier = float(physics.get('killer_deceleration_multiplier', self.killer_deceleration_multiplier))
            self.strike_acceleration_multiplier = float(physics.get('strike_acceleration_multiplier', self.strike_acceleration_multiplier))
            
            ai = scenario_config.get('ai_parameters', {})
            self.role_assignment_strategy = ai.get('role_assignment_strategy', self.role_assignment_strategy)
            self.scout_detection_bonus = float(ai.get('scout_detection_bonus', self.scout_detection_bonus))
            self.sensor_prediction_bonus = float(ai.get('sensor_prediction_bonus', self.sensor_prediction_bonus))
            self.heavy_strike_bonus = float(ai.get('heavy_strike_bonus', self.heavy_strike_bonus))
            
            self.heterogeneous_formation_spacing = float(ai.get('heterogeneous_formation_spacing', self.heterogeneous_formation_spacing))
            self.speed_matching_tolerance = float(ai.get('speed_matching_tolerance', self.speed_matching_tolerance))
            
            self.scout_advance_distance = float(ai.get('scout_advance_distance', self.scout_advance_distance))
            self.sensor_optimal_range = float(ai.get('sensor_optimal_range', self.sensor_optimal_range))
            self.heavy_engagement_range = float(ai.get('heavy_engagement_range', self.heavy_engagement_range))
            
            self.pack_formation_distance = float(ai.get('pack_formation_distance', self.pack_formation_distance))
            self.pack_activation_distance = float(ai.get('pack_activation_distance', self.pack_activation_distance))
            
        except Exception as e:
            print(f"⚠️ Parameter loading error: {e} - using defaults")

    def _initialize_drone_capabilities(self, interceptors: List[Dict]):
        """Initialize capability profiles for all drones"""
        for drone in interceptors:
            if drone['id'] not in self.drone_capabilities:
                drone_type = drone.get('type', 'standard_drone')
                
                # Get specs based on drone type
                specs = self.drone_specs.get(drone_type, self.default_specs).copy()
                
                # Calculate capability scores
                speed_score = specs['max_velocity'] / 350.0  # Normalized to max possible
                accel_score = specs['max_acceleration'] / 120.0
                payload_score = specs['payload_capacity'] / 2.0
                detection_score = specs.get('detection_range', 1500) / 5000.0
                
                self.drone_capabilities[drone['id']] = {
                    'type': drone_type,
                    'specs': specs,
                    'speed_score': speed_score,
                    'accel_score': accel_score,
                    'payload_score': payload_score,
                    'detection_score': detection_score,
                    'overall_score': (speed_score + accel_score + payload_score + detection_score) / 4,
                    'preferred_roles': specs.get('role_preference', ['green']),
                    'current_role': None,
                    'performance_history': []
                }
                
                print(f"🚁 {drone['id']} ({drone_type}): speed={speed_score:.2f}, accel={accel_score:.2f}, " +
                      f"payload={payload_score:.2f}, detect={detection_score:.2f}")

    def _assign_pack_roles(self, pack: Dict, interceptors: List[Dict]):
        """Assign roles within pack based on drone capabilities"""
        pack_drones = [d for d in interceptors if d['id'] in pack['drone_ids'] and d.get('active', True)]
        
        if self.role_assignment_strategy == 'capability_based':
            # Analyze pack composition
            drone_caps = [(d, self.drone_capabilities[d['id']]) for d in pack_drones]
            
            # Sort by different capabilities for role assignment
            speed_sorted = sorted(drone_caps, key=lambda x: x[1]['speed_score'], reverse=True)
            payload_sorted = sorted(drone_caps, key=lambda x: x[1]['payload_score'], reverse=True)
            detection_sorted = sorted(drone_caps, key=lambda x: x[1]['detection_score'], reverse=True)
            
            # Assign scout role to fastest with good detection
            scout_candidates = [d for d, cap in drone_caps 
                              if 'scout' in cap['preferred_roles'] or cap['speed_score'] > 0.8]
            if scout_candidates:
                scout = scout_candidates[0]
                self.drone_roles[scout['id']] = 'scout'
                pack['scout_drone'] = scout['id']
            
            # Assign sensor role to best detection
            sensor_candidates = [d for d, cap in detection_sorted 
                               if 'sensor' in cap['preferred_roles'] or cap['detection_score'] > 0.7]
            if sensor_candidates and sensor_candidates[0][0]['id'] != pack.get('scout_drone'):
                sensor = sensor_candidates[0][0]
                self.drone_roles[sensor['id']] = 'sensor'
                pack['sensor_drone'] = sensor['id']
            
            # Remaining drones are candidates for killer/green roles
            assigned = {pack.get('scout_drone'), pack.get('sensor_drone')}
            remaining = [d for d in pack_drones if d['id'] not in assigned]
            
            if remaining:
                # Best overall capability becomes primary killer candidate
                killer_candidates = sorted(remaining, 
                                         key=lambda d: self.drone_capabilities[d['id']]['overall_score'], 
                                         reverse=True)
                pack['killer_candidates'] = [d['id'] for d in killer_candidates]
                
                # Others are green drones
                for d in remaining:
                    if d['id'] not in pack['killer_candidates'][:1]:  # Keep top candidate
                        self.drone_roles[d['id']] = 'green'

    def _get_drone_physics_params(self, drone: Dict, is_killer: bool = False) -> Tuple[float, float]:
        """Get physics parameters based on drone type and role"""
        caps = self.drone_capabilities.get(drone['id'])
        if not caps:
            return 75.0, 240.0  # Default values
        
        specs = caps['specs']
        max_a = specs['max_acceleration']
        max_v = specs['max_velocity']
        
        # Apply killer role modifiers
        if is_killer:
            max_a *= self.killer_acceleration_multiplier
            max_v *= self.killer_velocity_multiplier
            
            # Check for deceleration boost
            vx, vy, vz = drone.get('vx', 0), drone.get('vy', 0), drone.get('vz', 0)
            ax, ay, az = drone.get('ax', 0), drone.get('ay', 0), drone.get('az', 0)
            sp = math.sqrt(vx*vx + vy*vy + vz*vz)
            if sp > 1.0 and (vx*ax + vy*ay + vz*az)/sp < 0:
                max_a *= self.killer_deceleration_multiplier
                
        return max_a, max_v

    def _compute_heterogeneous_formation(self, pack: Dict, target: Dict, interceptors: List[Dict]):
        """Compute formation positions considering different drone capabilities"""
        pack_drones = [d for d in interceptors if d['id'] in pack['drone_ids'] and d.get('active', True)]
        
        # Get average speed capability of pack
        avg_speed = sum(self.drone_capabilities[d['id']]['specs']['max_velocity'] 
                       for d in pack_drones) / len(pack_drones)
        
        # Adjust formation radius based on pack composition
        base_radius = self.pack_formation_distance
        radius_adjustment = 1.0
        
        # If pack has high speed variance, increase spacing
        speed_variance = np.var([self.drone_capabilities[d['id']]['specs']['max_velocity'] 
                                for d in pack_drones])
        if speed_variance > 2500:  # High variance threshold
            radius_adjustment = self.heterogeneous_formation_spacing
        
        effective_radius = base_radius * radius_adjustment
        
        # Target velocity for formation alignment
        tvx, tvy = target.get('vx', 0.0), target.get('vy', 0.0)
        sp = math.sqrt(tvx*tvx + tvy*tvy)
        base_heading = math.atan2(tvy, tvx) if sp > 0.5 else 0.0
        
        # Assign positions based on roles and capabilities
        positions = {}
        
        # Scout goes ahead
        if pack.get('scout_drone'):
            scout_angle = base_heading
            scout_radius = effective_radius + self.scout_advance_distance
            positions[pack['scout_drone']] = {
                'x': target['x'] + scout_radius * math.cos(scout_angle),
                'y': target['y'] + scout_radius * math.sin(scout_angle),
                'z': target.get('z', 0.0),
                'angle': scout_angle,
                'role': 'scout',
                'radius': scout_radius
            }
        
        # Sensor maintains optimal range
        if pack.get('sensor_drone'):
            sensor_angle = base_heading + math.pi  # Behind target
            sensor_radius = min(effective_radius, self.sensor_optimal_range)
            positions[pack['sensor_drone']] = {
                'x': target['x'] + sensor_radius * math.cos(sensor_angle),
                'y': target['y'] + sensor_radius * math.sin(sensor_angle),
                'z': target.get('z', 0.0),
                'angle': sensor_angle,
                'role': 'sensor',
                'radius': sensor_radius
            }
        
        # Distribute remaining drones
        remaining_drones = [d for d in pack_drones 
                           if d['id'] not in [pack.get('scout_drone'), pack.get('sensor_drone')]]
        
        if remaining_drones:
            # Place heavier/slower drones at sides for better intercept angles
            sorted_by_speed = sorted(remaining_drones, 
                                   key=lambda d: self.drone_capabilities[d['id']]['specs']['max_velocity'])
            
            side_angles = [base_heading + math.pi/2, base_heading - math.pi/2]
            for i, drone in enumerate(sorted_by_speed):
                if i < len(side_angles):
                    angle = side_angles[i]
                else:
                    # Additional drones fill in gaps
                    angle = base_heading + (2 * math.pi * i / len(remaining_drones))
                
                positions[drone['id']] = {
                    'x': target['x'] + effective_radius * math.cos(angle),
                    'y': target['y'] + effective_radius * math.sin(angle),
                    'z': target.get('z', 0.0),
                    'angle': angle,
                    'role': self.drone_roles.get(drone['id'], 'green'),
                    'radius': effective_radius
                }
        
        pack['formation_positions'] = positions

    def _select_optimal_killer(self, pack: Dict, target: Dict, interceptors: List[Dict]) -> Optional[str]:
        """Select the best drone for killer role based on situation"""
        candidates = pack.get('killer_candidates', [])
        if not candidates:
            return None
        
        target_type = target.get('type', '')
        best_score = -1
        best_drone = None
        
        for drone_id in candidates:
            drone = next((d for d in interceptors if d['id'] == drone_id and d.get('active', True)), None)
            if not drone:
                continue
                
            caps = self.drone_capabilities[drone_id]
            score = 0
            
            # Distance factor
            dist = self._dist(drone, target)
            distance_score = max(0, 1.0 - dist / 5000)
            score += distance_score * 2
            
            # Speed matching for different target types
            target_speed = math.sqrt(target.get('vx', 0)**2 + target.get('vy', 0)**2)
            drone_max_speed = caps['specs']['max_velocity']
            
            if 'bomber' in target_type or 'cargo' in target_type:
                # Heavy payload preferred for large targets
                score += caps['payload_score'] * self.heavy_strike_bonus
            elif 'fighter' in target_type or 'agile' in target_type:
                # Speed critical for agile targets
                if drone_max_speed > target_speed * 1.2:
                    score += caps['speed_score'] * 2
            elif 'stealth' in target_type:
                # Detection important for stealth
                score += caps['detection_score'] * 1.5
            
            # Overall capability
            score += caps['overall_score']
            
            if score > best_score:
                best_score = score
                best_drone = drone_id
                
        return best_drone

    def update(self, targets: List[Dict], interceptors: List[Dict], dt: float, wind: Dict, scenario_config: Dict = None) -> Dict[str, Dict]:
        if scenario_config:
            self._load_all_parameters(scenario_config)
            
        eff_dt = dt if (dt is not None and dt > 0.0) else (self.dt if self.frame_count == 0 else 0.0)
        
        self.frame_count += 1
        self.mission_time += eff_dt
        self.dt = eff_dt
        
        # Initialize drone capabilities
        if not self.drone_capabilities:
            self._initialize_drone_capabilities(interceptors)
        
        # Physics step with heterogeneous parameters
        self._update_physics_heterogeneous(targets, interceptors, eff_dt, wind)
        
        # Initialize packs once
        if not self.packs_initialized:
            self._create_heterogeneous_packs(targets, interceptors)
            self.packs_initialized = True
        
        # Debug output
        if self.frame_count % 30 == 0:
            self._print_heterogeneous_debug(targets, interceptors)
            
        # Generate decisions for all packs
        decisions: Dict[str, Dict] = {}
        for pack_id, pack in self.packs.items():
            pack_decisions = self._update_heterogeneous_pack(pack, targets, interceptors)
            decisions.update(pack_decisions)
            
        return decisions

    def _update_physics_heterogeneous(self, targets: List[Dict], interceptors: List[Dict], dt: float, wind: Dict):
        """Update physics with drone-specific parameters"""
        if dt <= 0.0:
            return

        # Update targets
        for t in targets:
            if not t.get('active', True):
                continue
            
            if 'vx' not in t: t['vx'] = 0.0
            if 'vy' not in t: t['vy'] = 0.0
            if 'vz' not in t: t['vz'] = 0.0
            
            wx = wind.get('x', 0.0); wy = wind.get('y', 0.0); wz = wind.get('z', 0.0)
            t['x'] += (t['vx'] + 0.1*wx) * dt
            t['y'] += (t['vy'] + 0.1*wy) * dt
            t['z'] = t.get('z', 0.0) + (t['vz'] + 0.1*wz) * dt

        # Update interceptors with type-specific physics
        for d in interceptors:
            if not d.get('active', True):
                continue
            
            if 'vx' not in d: d['vx'] = 0.0
            if 'vy' not in d: d['vy'] = 0.0
            if 'vz' not in d: d['vz'] = 0.0
            if 'ax' not in d: d['ax'] = 0.0
            if 'ay' not in d: d['ay'] = 0.0
            if 'az' not in d: d['az'] = 0.0
            
            # Get drone-specific parameters
            pack_id = self.drone_pack_map.get(d['id'])
            is_killer = False
            if pack_id and pack_id in self.packs:
                is_killer = (self.packs[pack_id].get('killer_drone') == d['id'])
            
            max_a, max_v = self._get_drone_physics_params(d, is_killer)
            
            # Apply acceleration limits
            ax = float(d['ax']); ay = float(d['ay']); az = float(d['az'])
            amag = math.sqrt(ax*ax + ay*ay + az*az)
            if amag > max_a and amag > 0:
                s = max_a/amag; ax*=s; ay*=s; az*=s

            # Update velocity
            d['vx'] += ax*dt
            d['vy'] += ay*dt
            d['vz'] += az*dt

            # Apply velocity limits
            sp = math.sqrt(d['vx']**2 + d['vy']**2 + d['vz']**2)
            if sp > max_v and sp > 0:
                s = max_v/sp; d['vx']*=s; d['vy']*=s; d['vz']*=s

            # Update position
            d['x'] += d['vx']*dt
            d['y'] += d['vy']*dt
            d['z'] = d.get('z', 0.0) + d['vz']*dt

    def _create_heterogeneous_packs(self, targets: List[Dict], interceptors: List[Dict]):
        """Create packs with optimal heterogeneous composition"""
        act_targets = [t for t in targets if t.get('active', True)]
        act_interceptors = [i for i in interceptors if i.get('active', True)]
        
        print("\n" + "🚁" * 20)
        print(f"🚁 CREATING HETEROGENEOUS PACKS: {len(act_targets)} targets, {len(act_interceptors)} drones")
        
        # Analyze interceptor composition
        type_counts = {}
        for drone in act_interceptors:
            drone_type = self.drone_capabilities[drone['id']]['type']
            type_counts[drone_type] = type_counts.get(drone_type, 0) + 1
        
        print(f"📊 FLEET COMPOSITION: {type_counts}")
        
        # Sort interceptors by capability diversity for optimal pack mixing
        interceptor_pool = act_interceptors.copy()
        
        idx = 0
        for tgt in act_targets:
            pack_id = f"pack_{tgt['id']}"
            drones = []
            
            # Try to create diverse packs
            desired_types = ['sensor_drone', 'scout_drone', 'fast_interceptor', 'heavy_interceptor']
            
            # First pass: try to get one of each type
            for desired_type in desired_types:
                for i, drone in enumerate(interceptor_pool):
                    if self.drone_capabilities[drone['id']]['type'] == desired_type:
                        drones.append(drone['id'])
                        self.drone_pack_map[drone['id']] = pack_id
                        interceptor_pool.pop(i)
                        break
                        
                if len(drones) >= 4:
                    break
            
            # Second pass: fill remaining slots
            while len(drones) < 4 and interceptor_pool:
                drone = interceptor_pool.pop(0)
                drones.append(drone['id'])
                self.drone_pack_map[drone['id']] = pack_id
            
            if drones:
                self.packs[pack_id] = {
                    'target_id': tgt['id'],
                    'drone_ids': drones,
                    'killer_drone': None,
                    'killer_candidates': [],
                    'scout_drone': None,
                    'sensor_drone': None,
                    'green_drones': drones.copy(),
                    'pack_state': 'chasing',
                    'formation_positions': {},
                    'chase_start_time': self.mission_time,
                    'first_ready_time': None,
                    'engage_unlocked': False,
                }
                
                # Assign roles within pack
                self._assign_pack_roles(self.packs[pack_id], act_interceptors)
                
                # Print pack composition
                comp = []
                for did in drones:
                    dtype = self.drone_capabilities[did]['type'].replace('_', ' ')
                    comp.append(f"{did}({dtype})")
                print(f"🎯 PACK [{pack_id}] - Composition: {', '.join(comp)}")
                
        print("🚁" * 20 + "\n")

    def _update_heterogeneous_pack(self, pack: Dict, targets: List[Dict], interceptors: List[Dict]) -> Dict[str, Dict]:
        """Update pack with heterogeneous drone coordination"""
        out: Dict[str, Dict] = {}

        tgt_id = pack['target_id']
        target = next((t for t in targets if t['id'] == tgt_id and t.get('active', True)), None)
        if not target:
            for did in pack['drone_ids']:
                d = next((x for x in interceptors if x['id'] == did and x.get('active', True)), None)
                if d:
                    out[did] = self._decision(0,0,0, tgt_id, 'idle', 'target_destroyed')
            return out

        pack_drones = [x for x in interceptors if x['id'] in pack['drone_ids'] and x.get('active', True)]
        if not pack_drones:
            return out

        # Time since chase started
        chase_time = self.mission_time - pack.get('chase_start_time', 0.0)
        current_state = pack.get('pack_state', 'chasing')
        
        # STATE MACHINE with heterogeneous adaptations
        if current_state == 'chasing':
            if chase_time >= self.chase_duration:
                pack['pack_state'] = 'following'
            
            # Different chase behaviors based on drone type
            for d in pack_drones:
                role = self.drone_roles.get(d['id'], 'green')
                if role == 'scout' and pack.get('scout_drone') == d['id']:
                    out[d['id']] = self._scout_advance(d, target)
                else:
                    out[d['id']] = self._chase_target(d, target)

        elif current_state == 'following':
            # Calculate average distance considering different speeds
            weighted_dist = 0
            total_weight = 0
            for d in pack_drones:
                dist = self._dist(d, target)
                speed_cap = self.drone_capabilities[d['id']]['speed_score']
                weight = 1.0 / (speed_cap + 0.1)  # Slower drones weighted more
                weighted_dist += dist * weight
                total_weight += weight
            
            avg_dist = weighted_dist / total_weight if total_weight > 0 else 999999
            
            if avg_dist <= self.follow_distance:
                pack['pack_state'] = 'forming'
            
            for d in pack_drones:
                role = self.drone_roles.get(d['id'], 'green')
                if role == 'sensor' and pack.get('sensor_drone') == d['id']:
                    out[d['id']] = self._sensor_tracking(d, target)
                else:
                    out[d['id']] = self._follow_target(d, target)

        elif current_state == 'forming':
            # Compute heterogeneous formation
            self._compute_heterogeneous_formation(pack, target, interceptors)
            
            # Check formation readiness with capability weighting
            formation_score = 0
            for d in pack_drones:
                fp = pack['formation_positions'].get(d['id'])
                if fp:
                    slot_dist = math.sqrt((fp['x'] - d['x'])**2 + (fp['y'] - d['y'])**2)
                    # Adjust tolerance based on drone speed
                    drone_tolerance = self.ready_epsilon * (1.0 + (1.0 - self.drone_capabilities[d['id']]['speed_score']))
                    if slot_dist <= drone_tolerance:
                        formation_score += 1
            
            formation_ready = (formation_score >= min(3, len(pack_drones) - 1))
            
            if formation_ready:
                pack['pack_state'] = 'engaging'
                pack['engage_unlocked'] = True
            
            for d in pack_drones:
                fp = pack['formation_positions'].get(d['id'])
                out[d['id']] = self._move_to_formation_heterogeneous(d, target, fp)

        elif current_state == 'engaging':
            # Update formation
            self._compute_heterogeneous_formation(pack, target, interceptors)
            
            # Select optimal killer based on situation
            if pack.get('killer_drone') is None:
                best_killer = self._select_optimal_killer(pack, target, interceptors)
                if best_killer:
                    pack['killer_drone'] = best_killer
                    pack['green_drones'] = [d['id'] for d in pack_drones if d['id'] != best_killer]
                    print(f"🎯 KILLER SELECTED: {best_killer} ({self.drone_capabilities[best_killer]['type']}) " +
                          f"for {target.get('type', 'unknown')} target")
            
            # Generate role-specific decisions
            for d in pack_drones:
                if d['id'] == pack.get('killer_drone'):
                    out[d['id']] = self._killer_pursuit_heterogeneous(pack, d, target)
                else:
                    role = self.drone_roles.get(d['id'], 'green')
                    fp = pack['formation_positions'].get(d['id'])
                    
                    if role == 'scout':
                        out[d['id']] = self._scout_support(d, target, pack)
                    elif role == 'sensor':
                        out[d['id']] = self._sensor_support(d, target, pack)
                    else:
                        out[d['id']] = self._track_parallel(d, target, fp)

        return out

    def _scout_advance(self, drone: Dict, target: Dict) -> Dict:
        """Scout drone advances ahead of pack"""
        # Lead the chase
        dx = target['x'] - drone['x']
        dy = target['y'] - drone['y']
        dz = target.get('z', 0.0) - drone.get('z', 0.0)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
        
        # Predict further ahead
        tvx, tvy = target.get('vx', 0), target.get('vy', 0)
        lead_time = 2.0
        pred_x = target['x'] + tvx * lead_time
        pred_y = target['y'] + tvy * lead_time
        
        pdx = pred_x - drone['x']
        pdy = pred_y - drone['y']
        pdist = math.sqrt(pdx*pdx + pdy*pdy) or 1.0
        
        ux, uy, uz = pdx/pdist, pdy/pdist, dz/dist
        
        # Use scout's superior speed
        caps = self.drone_capabilities[drone['id']]
        gain = caps['specs']['max_acceleration'] * 0.9
        ax = gain * ux
        ay = gain * uy
        az = gain * uz
        
        return self._decision(ax, ay, az, target['id'], 'scout', 'advance_pursuit')

    def _sensor_tracking(self, drone: Dict, target: Dict) -> Dict:
        """Sensor drone maintains optimal tracking distance"""
        dx = target['x'] - drone['x']
        dy = target['y'] - drone['y']
        dz = target.get('z', 0.0) - drone.get('z', 0.0)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
        
        # Try to maintain optimal sensor range
        error = dist - self.sensor_optimal_range
        
        if abs(error) < 200.0:
            # At good range - match target velocity for tracking
            tvx, tvy, tvz = target.get('vx', 0.0), target.get('vy', 0.0), target.get('vz', 0.0)
            dvx, dvy, dvz = drone.get('vx', 0.0), drone.get('vy', 0.0), drone.get('vz', 0.0)
            
            gain = 2.0
            ax = gain * (tvx - dvx)
            ay = gain * (tvy - dvy)
            az = gain * (tvz - dvz)
        else:
            # Adjust distance
            ux, uy, uz = dx/dist, dy/dist, dz/dist
            caps = self.drone_capabilities[drone['id']]
            gain = caps['specs']['max_acceleration'] * 0.5 * (1.0 if error > 0 else -0.8)
            ax = gain * ux
            ay = gain * uy
            az = gain * uz
        
        return self._decision(ax, ay, az, target['id'], 'sensor', 'optimal_tracking')

    def _move_to_formation_heterogeneous(self, drone: Dict, target: Dict, fp: Optional[Dict]) -> Dict:
        """Move to formation with speed-appropriate acceleration"""
        if not fp:
            return self._decision(0,0,0, target['id'], 'forming', 'no_formation_slot')

        dx = fp['x'] - drone['x']
        dy = fp['y'] - drone['y']
        dz = fp.get('z', target.get('z', 0.0)) - drone.get('z', 0.0)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0

        ux, uy, uz = dx/dist, dy/dist, dz/dist
        
        # Use drone-specific acceleration
        caps = self.drone_capabilities[drone['id']]
        gain = caps['specs']['max_acceleration'] * 0.8
        ax = gain * ux
        ay = gain * uy
        az = gain * uz
        
        return self._decision(ax, ay, az, target['id'], 'forming', f'moving_to_{fp.get("role", "position")}')

    def _scout_support(self, drone: Dict, target: Dict, pack: Dict) -> Dict:
        """Scout provides forward reconnaissance during engagement"""
        # Position ahead to cut off escape routes
        tvx, tvy = target.get('vx', 0), target.get('vy', 0)
        tspeed = math.sqrt(tvx*tvx + tvy*tvy)
        
        if tspeed > 1:
            # Get ahead of target
            lead_dist = self.scout_advance_distance * 1.5
            pred_x = target['x'] + (tvx/tspeed) * lead_dist
            pred_y = target['y'] + (tvy/tspeed) * lead_dist
            
            dx = pred_x - drone['x']
            dy = pred_y - drone['y']
            dist = math.sqrt(dx*dx + dy*dy) or 1.0
            
            if dist > 50:
                ux, uy = dx/dist, dy/dist
                caps = self.drone_capabilities[drone['id']]
                ax = caps['specs']['max_acceleration'] * 0.7 * ux
                ay = caps['specs']['max_acceleration'] * 0.7 * uy
                az = 0
            else:
                # Block escape route
                ax = -tvx * 2
                ay = -tvy * 2
                az = 0
        else:
            # Circle at high speed
            return self._track_parallel(drone, target, pack['formation_positions'].get(drone['id']))
        
        return self._decision(ax, ay, az, target['id'], 'scout', 'blocking_escape')

    def _sensor_support(self, drone: Dict, target: Dict, pack: Dict) -> Dict:
        """Sensor provides enhanced tracking data during engagement"""
        # Maintain optimal tracking position
        fp = pack['formation_positions'].get(drone['id'])
        if fp:
            # Adjust position for best sensor coverage
            dx = fp['x'] - drone['x']
            dy = fp['y'] - drone['y']
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist > 30:
                # Move to optimal position
                caps = self.drone_capabilities[drone['id']]
                ux, uy = dx/dist, dy/dist
                ax = caps['specs']['max_acceleration'] * 0.5 * ux
                ay = caps['specs']['max_acceleration'] * 0.5 * uy
                az = 0
            else:
                # Maintain position and track
                tvx, tvy = target.get('vx', 0), target.get('vy', 0)
                dvx, dvy = drone.get('vx', 0), drone.get('vy', 0)
                ax = 2.0 * (tvx - dvx)
                ay = 2.0 * (tvy - dvy)
                az = 0
                
            return self._decision(ax, ay, az, target['id'], 'sensor', 'tracking_support')
        else:
            return self._track_parallel(drone, target, None)

    def _killer_pursuit_heterogeneous(self, pack: Dict, drone: Dict, target: Dict) -> Dict:
        """Killer pursuit with capability-based adjustments"""
        d = self._dist(drone, target)
        caps = self.drone_capabilities[drone['id']]
        
        # Apply strike bonus for heavy interceptors
        effective_strike_distance = self.strike_distance
        if 'heavy' in caps['type']:
            effective_strike_distance *= 1.2  # Larger strike envelope
        
        # STRIKE CHECK
        if d <= effective_strike_distance:
            target['active'] = False
            drone['active'] = False
            print(f"💥💥💥 HETEROGENEOUS STRIKE: {drone['id']} ({caps['type']}) destroyed {target['id']} at {d:.1f}m 💥💥💥")
            return self._decision(0,0,0, target['id'], 'killer', 'STRIKE_SUCCESS')

        # Pursuit strategy based on drone type
        if 'scout' in caps['type'] or 'fast' in caps['type']:
            # Fast intercept
            predict_time = min(self.prediction_horizon * 0.7, d / caps['specs']['max_velocity'])
        else:
            # Standard prediction
            predict_time = min(self.prediction_horizon, d / max(caps['specs']['max_velocity'], 50.0))
        
        tvx, tvy, tvz = target.get('vx', 0.0), target.get('vy', 0.0), target.get('vz', 0.0)
        pred_x = target['x'] + tvx * predict_time
        pred_y = target['y'] + tvy * predict_time
        pred_z = target.get('z', 0.0) + tvz * predict_time

        pdx = pred_x - drone['x']
        pdy = pred_y - drone['y']
        pdz = pred_z - drone.get('z', 0.0)
        pdist = math.sqrt(pdx*pdx + pdy*pdy + pdz*pdz) or 1.0

        # Type-specific speed planning
        if 'heavy' in caps['type'] and d < self.heavy_engagement_range:
            # Heavy interceptors use maximum force at close range
            desired_speed = caps['specs']['max_velocity'] * self.killer_velocity_multiplier
        else:
            desired_speed = min(caps['specs']['max_velocity'] * self.killer_velocity_multiplier, 
                              max(100.0, d * 1.8))
        
        dirx, diry, dirz = pdx/pdist, pdy/pdist, pdz/pdist
        
        dvx_des = desired_speed * dirx
        dvy_des = desired_speed * diry
        dvz_des = desired_speed * dirz

        cvx, cvy, cvz = drone.get('vx',0.0), drone.get('vy',0.0), drone.get('vz',0.0)

        # Type-specific agility
        base_agility = caps['specs']['max_acceleration'] * self.killer_acceleration_multiplier
        if d < 150 and 'heavy' in caps['type']:
            # Heavy drones get extra boost for final strike
            agility = base_agility * self.strike_acceleration_multiplier * 1.2
        else:
            agility = base_agility * self.strike_acceleration_multiplier * 0.8
            
        ax = agility * (dvx_des - cvx) / max(1.0, desired_speed)
        ay = agility * (dvy_des - cvy) / max(1.0, desired_speed)
        az = agility * (dvz_des - cvz) / max(1.0, desired_speed)

        return self._decision(ax, ay, az, target['id'], 'killer', f'{caps["type"]}_pursuit')

    def _print_heterogeneous_debug(self, targets: List[Dict], interceptors: List[Dict]):
        """Debug output for heterogeneous pack operations"""
        print("\n" + "="*80)
        print(f"🚁 HETEROGENEOUS PACK STATUS: t={self.mission_time:.2f}s")
        
        # Fleet composition summary
        active_types = {}
        for d in interceptors:
            if d.get('active', True):
                dtype = self.drone_capabilities[d['id']]['type']
                active_types[dtype] = active_types.get(dtype, 0) + 1
        
        print(f"📊 ACTIVE FLEET: {active_types}")
        
        # Pack effectiveness by composition
        for pack_id, pack in self.packs.items():
            target = next((t for t in targets if t['id'] == pack['target_id']), None)
            if not target or not target.get('active', True):
                continue
                
            pack_types = []
            avg_capability = 0
            for did in pack['drone_ids']:
                drone = next((d for d in interceptors if d['id'] == did), None)
                if drone and drone.get('active', True):
                    caps = self.drone_capabilities[did]
                    pack_types.append(caps['type'].split('_')[0])
                    avg_capability += caps['overall_score']
                    
            if pack_types:
                avg_capability /= len(pack_types)
                print(f"   {pack_id}: [{'-'.join(pack_types)}] " +
                      f"avg_cap={avg_capability:.2f} state={pack.get('pack_state', 'unknown')} " +
                      f"killer={pack.get('killer_drone', 'NONE')}")
        
        print("="*80)

    # Include base methods
    def _chase_target(self, drone: Dict, target: Dict) -> Dict:
        dx = target['x'] - drone['x']
        dy = target['y'] - drone['y']
        dz = target.get('z', 0.0) - drone.get('z', 0.0)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
        
        ux, uy, uz = dx/dist, dy/dist, dz/dist
        
        # Use drone-specific acceleration
        caps = self.drone_capabilities.get(drone['id'])
        if caps:
            gain = caps['specs']['max_acceleration'] * 0.8
        else:
            gain = 75.0 * 0.8
            
        ax = gain * ux
        ay = gain * uy
        az = gain * uz
        
        return self._decision(ax, ay, az, target['id'], 'chasing', 'direct_chase')

    def _follow_target(self, drone: Dict, target: Dict) -> Dict:
        dx = target['x'] - drone['x']
        dy = target['y'] - drone['y']
        dz = target.get('z', 0.0) - drone.get('z', 0.0)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
        
        error = dist - self.follow_distance
        
        if abs(error) < 100.0:
            tvx, tvy, tvz = target.get('vx', 0.0), target.get('vy', 0.0), target.get('vz', 0.0)
            dvx, dvy, dvz = drone.get('vx', 0.0), drone.get('vy', 0.0), drone.get('vz', 0.0)
            
            gain = 2.0
            ax = gain * (tvx - dvx)
            ay = gain * (tvy - dvy)
            az = gain * (tvz - dvz)
        else:
            ux, uy, uz = dx/dist, dy/dist, dz/dist
            caps = self.drone_capabilities.get(drone['id'])
            if caps:
                base_gain = caps['specs']['max_acceleration'] * 0.6
            else:
                base_gain = 75.0 * 0.6
            gain = base_gain * (1.0 if error > 0 else -0.5)
            ax = gain * ux
            ay = gain * uy
            az = gain * uz
        
        return self._decision(ax, ay, az, target['id'], 'following', 'distance_control')

    def _green_desired_velocity(self, target: Dict, fp: Dict) -> Tuple[float, float, float]:
        tvx, tvy, tvz = target.get('vx',0.0), target.get('vy',0.0), target.get('vz',0.0)

        rx = fp['x'] - target['x']; ry = fp['y'] - target['y']
        rn = math.sqrt(rx*rx + ry*ry) or 1.0
        tx, ty = (-ry/rn, rx/rn)
        tx *= self.ring_orbit_direction; ty *= self.ring_orbit_direction

        vdx = tvx + self.ring_orbit_speed * tx
        vdy = tvy + self.ring_orbit_speed * ty
        vdz = tvz

        tmag = math.sqrt(tvx*tvx + tvy*tvy + tvz*tvz)
        dmag = math.sqrt(vdx*vdx + vdy*vdy + vdz*vdz)
        min_speed = tmag + self.green_speed_margin
        
        if dmag < min_speed and dmag > 1e-3:
            s = min_speed / dmag
            vdx *= s; vdy *= s; vdz *= s

        return vdx, vdy, vdz

    def _track_parallel(self, drone: Dict, target: Dict, fp: Optional[Dict]) -> Dict:
        if not fp:
            return self._decision(0,0,0, target['id'], 'green', 'no_formation_slot')

        dx = fp['x'] - drone['x']
        dy = fp['y'] - drone['y']
        dz = fp.get('z', target.get('z', 0.0)) - drone.get('z', 0.0)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        caps = self.drone_capabilities.get(drone['id'])
        if caps:
            base_accel = caps['specs']['max_acceleration']
        else:
            base_accel = 75.0

        if dist > 10.0:
            ux, uy, uz = dx/dist, dy/dist, dz/dist
            pos_ax = base_accel * 0.5 * ux
            pos_ay = base_accel * 0.5 * uy
            pos_az = base_accel * 0.5 * uz
        else:
            pos_ax = pos_ay = pos_az = 0.0

        vdx, vdy, vdz = self._green_desired_velocity(target, fp)

        dvx, dvy, dvz = drone.get('vx',0.0), drone.get('vy',0.0), drone.get('vz',0.0)
        vel_gain = 2.0
        vel_ax = vel_gain * (vdx - dvx)
        vel_ay = vel_gain * (vdy - dvy)
        vel_az = vel_gain * (vdz - dvz)

        ax = pos_ax + vel_ax
        ay = pos_ay + vel_ay
        az = pos_az + vel_az
        return self._decision(ax, ay, az, target['id'], 'green', 'tracking_orbit')

    def _decision(self, ax: float, ay: float, az: float, target_id: str, role: str, status: str, **kwargs) -> Dict:
        return {'ax': ax, 'ay': ay, 'az': az, 'target_id': target_id, 'role': role, 'status': status, **kwargs}

    def _dist(self, a: Optional[Dict], b: Optional[Dict]) -> float:
        if not a or not b:
            return float('inf')
        dx = a.get('x',0.0) - b.get('x',0.0)
        dy = a.get('y',0.0) - b.get('y',0.0)
        dz = a.get('z',0.0) - b.get('z',0.0)
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _closest(self, drones: List[Dict], target: Dict) -> Tuple[Optional[Dict], float]:
        best = None; bestd = float('inf')
        for d in drones:
            dist = self._dist(d, target)
            if dist < bestd:
                bestd = dist; best = d
        return best, bestd