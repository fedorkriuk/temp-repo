import math
from typing import List, Dict, Optional, Tuple
import time

class AI_System_MultiTarget:
    """Enhanced pack logic with multi-target dynamic priority assignment"""

    def __init__(self, scenario_config: Dict = None):
        # Base pack parameters
        self.pack_formation_distance = 350.0
        self.pack_activation_distance = 900.0
        self.strike_distance = 30.0
        
        # Pack splitting/merging thresholds
        self.pack_split_threshold = 2500.0
        self.pack_merge_threshold = 1500.0
        
        # Priority assignment weights
        self.threat_priority_distance_weight = 0.4
        self.threat_priority_speed_weight = 0.3
        self.threat_priority_altitude_weight = 0.2
        self.threat_priority_type_weight = 0.1
        
        # Dynamic priority parameters
        self.priority_reassignment_interval = 5.0
        self.critical_distance_threshold = 3000.0
        self.high_priority_speed_threshold = 100.0
        
        self.green_max_velocity = 250.0
        self.green_max_acceleration = 80.0
        
        self.killer_max_velocity = 280.0
        self.killer_max_acceleration = 160.0
        self.killer_deceleration_multiplier = 3.5
        
        self.strike_acceleration_multiplier = 5.0
        
        # Mission parameters
        self.max_engagement_range = 15000.0
        self.prediction_horizon = 3.5
        self.cooperation_radius = 3000.0
        self.communication_range = 20000.0
        
        # Green orbit parameters
        self.ring_orbit_speed = 35.0
        self.ring_orbit_direction = 1
        self.green_speed_margin = 25.0
        
        # State tracking
        self.chase_duration = 8.0
        self.follow_distance = 600.0
        self.ready_epsilon = 150.0
        
        # Runtime state
        self.packs: Dict[str, Dict] = {}
        self.drone_pack_map: Dict[str, str] = {}
        self.packs_initialized = False
        self.target_priorities: Dict[str, float] = {}
        self.last_priority_update = 0.0
        
        self.frame_count = 0
        self.mission_time = 0.0
        self.dt = 1.0 / 30.0
        
        if scenario_config:
            self._load_all_parameters(scenario_config)
            
        print("=" * 60)
        print("🎯 MULTI-TARGET DYNAMIC PRIORITY SYSTEM INITIALIZED")
        print(f"   Pack split threshold: {self.pack_split_threshold}m")
        print(f"   Pack merge threshold: {self.pack_merge_threshold}m")
        print(f"   Priority update interval: {self.priority_reassignment_interval}s")
        print("=" * 60)

    def _load_all_parameters(self, scenario_config: Dict):
        try:
            physics = scenario_config.get('physics', {})
            self.green_max_acceleration = float(physics.get('green_max_acceleration', self.green_max_acceleration))
            self.green_max_velocity = float(physics.get('green_max_velocity', self.green_max_velocity))
            self.killer_max_acceleration = float(physics.get('killer_max_acceleration', self.killer_max_acceleration))
            self.killer_max_velocity = float(physics.get('killer_max_velocity', self.killer_max_velocity))
            self.killer_deceleration_multiplier = float(physics.get('killer_deceleration_multiplier', self.killer_deceleration_multiplier))
            self.strike_acceleration_multiplier = float(physics.get('strike_acceleration_multiplier', self.strike_acceleration_multiplier))
            
            ai = scenario_config.get('ai_parameters', {})
            self.pack_formation_distance = float(ai.get('pack_formation_distance', self.pack_formation_distance))
            self.pack_activation_distance = float(ai.get('pack_activation_distance', self.pack_activation_distance))
            self.pack_split_threshold = float(ai.get('pack_split_threshold', self.pack_split_threshold))
            self.pack_merge_threshold = float(ai.get('pack_merge_threshold', self.pack_merge_threshold))
            
            # Priority weights
            self.threat_priority_distance_weight = float(ai.get('threat_priority_distance_weight', self.threat_priority_distance_weight))
            self.threat_priority_speed_weight = float(ai.get('threat_priority_speed_weight', self.threat_priority_speed_weight))
            self.threat_priority_altitude_weight = float(ai.get('threat_priority_altitude_weight', self.threat_priority_altitude_weight))
            self.threat_priority_type_weight = float(ai.get('threat_priority_type_weight', self.threat_priority_type_weight))
            
            mission = scenario_config.get('mission_parameters', {})
            self.priority_reassignment_interval = float(mission.get('priority_reassignment_interval', self.priority_reassignment_interval))
            self.critical_distance_threshold = float(mission.get('critical_distance_threshold', self.critical_distance_threshold))
            self.high_priority_speed_threshold = float(mission.get('high_priority_speed_threshold', self.high_priority_speed_threshold))
            
        except Exception as e:
            print(f"⚠️ Parameter loading error: {e} - using defaults")

    def update(self, targets: List[Dict], interceptors: List[Dict], dt: float, wind: Dict, scenario_config: Dict = None) -> Dict[str, Dict]:
        if scenario_config:
            self._load_all_parameters(scenario_config)
            
        eff_dt = dt if (dt is not None and dt > 0.0) else (self.dt if self.frame_count == 0 else 0.0)
        
        self.frame_count += 1
        self.mission_time += eff_dt
        self.dt = eff_dt
        
        # Physics step
        self._update_physics(targets, interceptors, eff_dt, wind)
        
        # Initialize packs once
        if not self.packs_initialized:
            self._create_initial_packs(targets, interceptors)
            self.packs_initialized = True
            
        # Update target priorities periodically
        if self.mission_time - self.last_priority_update > self.priority_reassignment_interval:
            self._update_target_priorities(targets, interceptors)
            self.last_priority_update = self.mission_time
            
        # Check for pack splitting/merging opportunities
        self._evaluate_pack_reorganization(targets, interceptors)
        
        # Debug output
        if self.frame_count % 30 == 0:  # Every second
            self._print_multi_target_debug(targets, interceptors)
            
        # Generate decisions for all packs
        decisions: Dict[str, Dict] = {}
        for pack_id, pack in self.packs.items():
            pack_decisions = self._update_pack(pack, targets, interceptors)
            decisions.update(pack_decisions)
            
        return decisions

    def _calculate_threat_priority(self, target: Dict, interceptors: List[Dict]) -> float:
        """Calculate priority score for a target based on multiple factors"""
        if not target.get('active', True):
            return 0.0
            
        # Find nearest interceptor base (assuming origin)
        base_x, base_y = 0.0, 0.0
        dist_to_base = math.sqrt((target['x'] - base_x)**2 + (target['y'] - base_y)**2)
        
        # Calculate target speed
        speed = math.sqrt(target.get('vx', 0)**2 + target.get('vy', 0)**2 + target.get('vz', 0)**2)
        
        # Altitude factor (lower altitude = higher threat)
        altitude_factor = max(0.1, 1.0 - target.get('z', 0) / 5000.0)
        
        # Type factor (from target description)
        type_factor = 1.0
        target_type = target.get('type', '')
        if 'priority' in target_type:
            type_factor = 2.0
        elif 'high_speed' in target_type:
            type_factor = 1.5
        elif 'bomber' in target_type:
            type_factor = 1.2
            
        # Calculate priority score
        distance_score = max(0, 1.0 - dist_to_base / self.max_engagement_range)
        speed_score = speed / self.high_priority_speed_threshold
        
        priority = (self.threat_priority_distance_weight * distance_score +
                   self.threat_priority_speed_weight * speed_score +
                   self.threat_priority_altitude_weight * altitude_factor +
                   self.threat_priority_type_weight * type_factor)
                   
        # Critical distance override
        if dist_to_base < self.critical_distance_threshold:
            priority *= 2.0
            
        return priority

    def _update_target_priorities(self, targets: List[Dict], interceptors: List[Dict]):
        """Recalculate priorities for all active targets"""
        self.target_priorities.clear()
        
        for target in targets:
            if target.get('active', True):
                priority = self._calculate_threat_priority(target, interceptors)
                self.target_priorities[target['id']] = priority
                
        # Print priority updates
        print(f"\n🎯 TARGET PRIORITY UPDATE (t={self.mission_time:.1f}s):")
        sorted_targets = sorted(self.target_priorities.items(), key=lambda x: x[1], reverse=True)
        for tid, priority in sorted_targets:
            print(f"   {tid}: priority={priority:.2f}")

    def _create_initial_packs(self, targets: List[Dict], interceptors: List[Dict]):
        """Create initial pack assignments based on target priorities"""
        act_targets = [t for t in targets if t.get('active', True)]
        act_interceptors = [i for i in interceptors if i.get('active', True)]
        
        # Calculate initial priorities
        self._update_target_priorities(act_targets, act_interceptors)
        
        # Sort targets by priority
        sorted_targets = sorted(act_targets, 
                               key=lambda t: self.target_priorities.get(t['id'], 0), 
                               reverse=True)
        
        print("\n" + "🐺" * 20)
        print(f"🐺 CREATING MULTI-TARGET PACKS: {len(act_targets)} targets, {len(act_interceptors)} drones")
        
        # Assign drones to targets based on priority
        drones_per_target = max(1, len(act_interceptors) // len(act_targets))
        drone_idx = 0
        
        for target in sorted_targets:
            pack_id = f"pack_{target['id']}"
            drones = []
            
            # Assign drones based on priority (higher priority gets more drones)
            priority = self.target_priorities.get(target['id'], 0)
            if priority > 1.5:
                num_drones = min(6, drones_per_target + 2)  # High priority gets extra drones
            elif priority < 0.5:
                num_drones = max(2, drones_per_target - 1)  # Low priority gets fewer
            else:
                num_drones = drones_per_target
                
            for _ in range(num_drones):
                if drone_idx < len(act_interceptors):
                    d = act_interceptors[drone_idx]
                    drones.append(d['id'])
                    self.drone_pack_map[d['id']] = pack_id
                    drone_idx += 1
                    
            if drones:
                self.packs[pack_id] = {
                    'target_id': target['id'],
                    'drone_ids': drones,
                    'killer_drone': None,
                    'green_drones': drones.copy(),
                    'pack_state': 'chasing',
                    'formation_positions': {},
                    'chase_start_time': self.mission_time,
                    'first_ready_time': None,
                    'engage_unlocked': False,
                    'priority': priority
                }
                print(f"🎯 PACK [{pack_id}] - Priority: {priority:.2f} - Drones: {len(drones)} - {drones}")
                
        print("🐺" * 20 + "\n")

    def _evaluate_pack_reorganization(self, targets: List[Dict], interceptors: List[Dict]):
        """Check if packs should split or merge based on target distances"""
        # Skip if too early in mission
        if self.mission_time < 10.0:
            return
            
        # Find packs that might benefit from reorganization
        pack_distances = {}
        for pack_id, pack in self.packs.items():
            target = next((t for t in targets if t['id'] == pack['target_id'] and t.get('active', True)), None)
            if target:
                # Calculate distance to other targets
                min_other_dist = float('inf')
                nearest_other = None
                for other in targets:
                    if other['id'] != target['id'] and other.get('active', True):
                        dist = self._dist(target, other)
                        if dist < min_other_dist:
                            min_other_dist = dist
                            nearest_other = other['id']
                            
                pack_distances[pack_id] = {
                    'target': target['id'],
                    'nearest_other': nearest_other,
                    'distance': min_other_dist
                }
                
        # Check for split opportunities
        for pack_id, info in pack_distances.items():
            if info['distance'] < self.pack_split_threshold and len(self.packs[pack_id]['drone_ids']) >= 4:
                # Consider splitting this pack
                other_pack_id = f"pack_{info['nearest_other']}"
                if other_pack_id not in self.packs:
                    self._split_pack(pack_id, info['nearest_other'])

    def _split_pack(self, pack_id: str, new_target_id: str):
        """Split a pack to engage a nearby target"""
        pack = self.packs.get(pack_id)
        if not pack or len(pack['drone_ids']) < 4:
            return
            
        print(f"\n💥 PACK SPLIT: {pack_id} splitting to engage {new_target_id}")
        
        # Split drones in half
        split_point = len(pack['drone_ids']) // 2
        new_drones = pack['drone_ids'][split_point:]
        pack['drone_ids'] = pack['drone_ids'][:split_point]
        
        # Update drone mappings
        new_pack_id = f"pack_{new_target_id}"
        for drone_id in new_drones:
            self.drone_pack_map[drone_id] = new_pack_id
            
        # Create new pack
        self.packs[new_pack_id] = {
            'target_id': new_target_id,
            'drone_ids': new_drones,
            'killer_drone': None,
            'green_drones': new_drones.copy(),
            'pack_state': 'chasing',
            'formation_positions': {},
            'chase_start_time': self.mission_time,
            'first_ready_time': None,
            'engage_unlocked': False,
            'priority': self.target_priorities.get(new_target_id, 1.0)
        }
        
        # Reset killer drone if it was split away
        if pack['killer_drone'] in new_drones:
            pack['killer_drone'] = None
            pack['green_drones'] = pack['drone_ids'].copy()

    def _print_multi_target_debug(self, targets: List[Dict], interceptors: List[Dict]):
        """Enhanced debug output for multi-target scenarios"""
        print("\n" + "="*80)
        print(f"🕒 MULTI-TARGET STATUS: t={self.mission_time:.2f}s (Frame {self.frame_count})")
        
        # Active targets summary
        active_targets = [t for t in targets if t.get('active', True)]
        print(f"📊 ACTIVE TARGETS: {len(active_targets)}")
        
        for t in active_targets:
            priority = self.target_priorities.get(t['id'], 0)
            speed = math.sqrt(t.get('vx',0)**2 + t.get('vy',0)**2)
            dist_to_base = math.sqrt(t['x']**2 + t['y']**2)
            print(f"   🎯 {t['id']}: pos=({t['x']:.0f},{t['y']:.0f},{t.get('z',0):.0f}) " +
                  f"speed={speed:.1f} dist={dist_to_base:.0f} priority={priority:.2f}")
        
        # Pack status
        print(f"\n📦 PACK STATUS:")
        for pack_id, pack in self.packs.items():
            state = pack.get('pack_state', 'unknown')
            num_drones = len([d for d in interceptors if d['id'] in pack['drone_ids'] and d.get('active', True)])
            killer = pack.get('killer_drone', 'NONE')
            priority = pack.get('priority', 0)
            print(f"   {pack_id}: TARGET={pack['target_id']} STATE={state} " +
                  f"DRONES={num_drones} KILLER={killer} PRIORITY={priority:.2f}")
                  
        print("="*80)

    # Include all the other methods from the original AI_System
    def _update_physics(self, targets: List[Dict], interceptors: List[Dict], dt: float, wind: Dict):
        if dt <= 0.0:
            return

        # Targets with movement
        for t in targets:
            if not t.get('active', True):
                continue
            
            # Ensure target has velocity values
            if 'vx' not in t: t['vx'] = 0.0
            if 'vy' not in t: t['vy'] = 0.0
            if 'vz' not in t: t['vz'] = 0.0
            
            wx = wind.get('x', 0.0); wy = wind.get('y', 0.0); wz = wind.get('z', 0.0)
            t['x'] += (t['vx'] + 0.1*wx) * dt
            t['y'] += (t['vy'] + 0.1*wy) * dt
            t['z'] = t.get('z', 0.0) + (t['vz'] + 0.1*wz) * dt

        # Interceptors with role-based limits
        for d in interceptors:
            if not d.get('active', True):
                continue
            
            # Ensure drone has velocity values
            if 'vx' not in d: d['vx'] = 0.0
            if 'vy' not in d: d['vy'] = 0.0
            if 'vz' not in d: d['vz'] = 0.0
            if 'ax' not in d: d['ax'] = 0.0
            if 'ay' not in d: d['ay'] = 0.0
            if 'az' not in d: d['az'] = 0.0
            
            pack_id = self.drone_pack_map.get(d['id'])
            is_killer = False
            if pack_id and pack_id in self.packs:
                is_killer = (self.packs[pack_id].get('killer_drone') == d['id'])

            if is_killer:
                max_a = self.killer_max_acceleration
                max_v = self.killer_max_velocity
                vx, vy, vz = d['vx'], d['vy'], d['vz']
                ax, ay, az = d['ax'], d['ay'], d['az']
                sp = math.sqrt(vx*vx + vy*vy + vz*vz)
                if sp > 1.0 and (vx*ax + vy*ay + vz*az)/sp < 0:
                    max_a *= self.killer_deceleration_multiplier
            else:
                max_a = self.green_max_acceleration
                max_v = self.green_max_velocity

            ax = float(d['ax']); ay = float(d['ay']); az = float(d['az'])
            amag = math.sqrt(ax*ax + ay*ay + az*az)
            if amag > max_a and amag > 0:
                s = max_a/amag; ax*=s; ay*=s; az*=s

            d['vx'] += ax*dt
            d['vy'] += ay*dt
            d['vz'] += az*dt

            sp = math.sqrt(d['vx']**2 + d['vy']**2 + d['vz']**2)
            if sp > max_v and sp > 0:
                s = max_v/sp; d['vx']*=s; d['vy']*=s; d['vz']*=s

            d['x'] += d['vx']*dt
            d['y'] += d['vy']*dt
            d['z'] = d.get('z', 0.0) + d['vz']*dt

    def _update_pack(self, pack: Dict, targets: List[Dict], interceptors: List[Dict]) -> Dict[str, Dict]:
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

        # STATE MACHINE: chasing -> following -> forming -> engaging
        current_state = pack.get('pack_state', 'chasing')
        
        # STATE TRANSITION LOGIC
        if current_state == 'chasing':
            if chase_time >= self.chase_duration:
                pack['pack_state'] = 'following'
            
            # All drones chase target directly
            for d in pack_drones:
                out[d['id']] = self._chase_target(d, target)

        elif current_state == 'following':
            avg_dist = sum(self._dist(d, target) for d in pack_drones) / len(pack_drones)
            
            if avg_dist <= self.follow_distance:
                pack['pack_state'] = 'forming'
            
            # All drones follow at distance
            for d in pack_drones:
                out[d['id']] = self._follow_target(d, target)

        elif current_state == 'forming':
            ring_r = self.pack_formation_distance
            self._compute_ring(pack, target, ring_r)
            
            # CHECK: At least 3 out of 4 drones must be close to their slots
            close_drones = 0
            for d in pack_drones:
                fp = pack['formation_positions'].get(d['id'])
                if fp:
                    slot_dist = math.sqrt((fp['x'] - d['x'])**2 + (fp['y'] - d['y'])**2)
                    if slot_dist <= self.ready_epsilon:
                        close_drones += 1
            
            formation_ready = (close_drones >= min(3, len(pack_drones) - 1))
            
            if formation_ready:
                pack['pack_state'] = 'engaging'
                pack['engage_unlocked'] = True
            
            # All drones move to formation positions
            for d in pack_drones:
                fp = pack['formation_positions'].get(d['id'])
                out[d['id']] = self._move_to_formation(d, target, fp)

        elif current_state == 'engaging':
            ring_r = self.pack_formation_distance
            self._compute_ring(pack, target, ring_r)
            
            # Select killer if none
            if pack.get('killer_drone') is None:
                closest, closest_dist = self._closest(pack_drones, target)
                if closest:
                    pack['killer_drone'] = closest['id']
                    pack['green_drones'] = [d['id'] for d in pack_drones if d['id'] != pack['killer_drone']]
            
            # Generate decisions
            for d in pack_drones:
                if d['id'] == pack.get('killer_drone'):
                    out[d['id']] = self._killer_pursuit(pack, d, target)
                else:
                    fp = pack['formation_positions'].get(d['id'])
                    out[d['id']] = self._track_parallel(d, target, fp)

        return out

    # Include all other helper methods from original AI_System...
    def _chase_target(self, drone: Dict, target: Dict) -> Dict:
        dx = target['x'] - drone['x']
        dy = target['y'] - drone['y']
        dz = target.get('z', 0.0) - drone.get('z', 0.0)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
        
        ux, uy, uz = dx/dist, dy/dist, dz/dist
        gain = self.green_max_acceleration * 0.8
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
            gain = self.green_max_acceleration * 0.6 * (1.0 if error > 0 else -0.5)
            ax = gain * ux
            ay = gain * uy
            az = gain * uz
        
        return self._decision(ax, ay, az, target['id'], 'following', 'distance_control')

    def _compute_ring(self, pack: Dict, target: Dict, radius: float):
        tvx, tvy, tvz = target.get('vx', 0.0), target.get('vy', 0.0), target.get('vz', 0.0)
        sp = math.sqrt(tvx*tvx + tvy*tvy)
        base_heading = math.atan2(tvy, tvx) if sp > 0.5 else 0.0

        num_drones = len(pack['drone_ids'])
        angle_step = 2 * math.pi / max(num_drones, 4)
        
        pos: Dict[str, Dict] = {}
        for i, did in enumerate(pack['drone_ids']):
            ang = base_heading + i * angle_step
            pos[did] = {
                'x': target['x'] + radius * math.cos(ang),
                'y': target['y'] + radius * math.sin(ang),
                'z': target.get('z', 0.0),
                'angle': ang,
                'slot': i,
                'radius': radius,
            }
        pack['formation_positions'] = pos

    def _move_to_formation(self, drone: Dict, target: Dict, fp: Optional[Dict]) -> Dict:
        if not fp:
            return self._decision(0,0,0, target['id'], 'forming', 'no_formation_slot')

        dx = fp['x'] - drone['x']
        dy = fp['y'] - drone['y']
        dz = fp.get('z', target.get('z', 0.0)) - drone.get('z', 0.0)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0

        ux, uy, uz = dx/dist, dy/dist, dz/dist
        gain = self.green_max_acceleration * 0.9
        ax = gain * ux
        ay = gain * uy
        az = gain * uz
        return self._decision(ax, ay, az, target['id'], 'forming', 'moving_to_formation')

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

        if dist > 10.0:
            ux, uy, uz = dx/dist, dy/dist, dz/dist
            pos_ax = self.green_max_acceleration * 0.5 * ux
            pos_ay = self.green_max_acceleration * 0.5 * uy
            pos_az = self.green_max_acceleration * 0.5 * uz
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

    def _killer_pursuit(self, pack: Dict, drone: Dict, target: Dict) -> Dict:
        d = self._dist(drone, target)
        
        # STRIKE CHECK
        if d <= self.strike_distance:
            target['active'] = False
            drone['active'] = False
            print(f"💥💥💥 STRIKE SUCCESS: {drone['id']} destroyed {target['id']} at {d:.1f}m 💥💥💥")
            return self._decision(0,0,0, target['id'], 'killer', 'STRIKE_SUCCESS')

        # AGGRESSIVE PURSUIT
        if d <= 100.0:
            dx = target['x'] - drone['x']
            dy = target['y'] - drone['y'] 
            dz = target.get('z', 0.0) - drone.get('z', 0.0)
            dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
            
            ux, uy, uz = dx/dist, dy/dist, dz/dist
            max_accel = self.killer_max_acceleration * self.strike_acceleration_multiplier
            ax = max_accel * ux
            ay = max_accel * uy
            az = max_accel * uz
            
            return self._decision(ax, ay, az, target['id'], 'killer', 'final_approach')
        
        # LONG RANGE PURSUIT
        dx = target['x'] - drone['x']
        dy = target['y'] - drone['y']
        dz = target.get('z', 0.0) - drone.get('z', 0.0)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0

        tvx, tvy, tvz = target.get('vx', 0.0), target.get('vy', 0.0), target.get('vz', 0.0)
        predict_time = min(self.prediction_horizon, dist / max(self.killer_max_velocity, 50.0))
        
        pred_x = target['x'] + tvx * predict_time
        pred_y = target['y'] + tvy * predict_time
        pred_z = target.get('z', 0.0) + tvz * predict_time

        pdx = pred_x - drone['x']
        pdy = pred_y - drone['y']
        pdz = pred_z - drone.get('z', 0.0)
        pdist = math.sqrt(pdx*pdx + pdy*pdy + pdz*pdz) or 1.0

        desired_speed = min(self.killer_max_velocity, max(80.0, dist * 1.5))
        dirx, diry, dirz = pdx/pdist, pdy/pdist, pdz/pdist
        
        dvx_des = desired_speed * dirx
        dvy_des = desired_speed * diry
        dvz_des = desired_speed * dirz

        cvx, cvy, cvz = drone.get('vx',0.0), drone.get('vy',0.0), drone.get('vz',0.0)

        agile = self.killer_max_acceleration * self.strike_acceleration_multiplier * 0.8
        ax = agile * (dvx_des - cvx) / max(1.0, desired_speed)
        ay = agile * (dvy_des - cvy) / max(1.0, desired_speed)
        az = agile * (dvz_des - cvz) / max(1.0, desired_speed)

        return self._decision(ax, ay, az, target['id'], 'killer', 'pursuit')

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