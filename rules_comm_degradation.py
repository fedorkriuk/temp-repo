import math
import random
from typing import List, Dict, Optional, Tuple, Set
import numpy as np

class AI_System_CommDegradation:
    """Pack logic with communication degradation and failure scenarios"""

    def __init__(self, scenario_config: Dict = None):
        # Base pack parameters
        self.pack_formation_distance = 350.0
        self.pack_activation_distance = 900.0
        self.strike_distance = 30.0
        
        # Communication parameters
        self.communication_range = 15000.0
        self.communication_base_latency = 0.05
        self.communication_packet_loss_base = 0.01
        self.jamming_effectiveness = 0.8
        self.terrain_mask_threshold = 500.0
        
        # Degraded communication behaviors
        self.degraded_formation_distance = 500.0
        self.degraded_coordination_factor = 0.6
        self.local_decision_radius = 800.0
        self.emergency_protocol_distance = 1000.0
        
        # Autonomous behavior parameters
        self.autonomous_engagement_threshold = 500.0
        self.mesh_network_hop_limit = 3
        self.hardened_comm_resistance = 0.7
        
        # Communication failure handling
        self.comm_failure_timeout = 2.0
        self.reconnection_attempt_interval = 0.5
        self.fallback_behavior = 'local_autonomy'
        self.emergency_rtb_threshold = 10.0
        
        self.green_max_velocity = 240.0
        self.green_max_acceleration = 75.0
        
        self.killer_max_velocity = 270.0
        self.killer_max_acceleration = 150.0
        self.killer_deceleration_multiplier = 3.5
        
        self.strike_acceleration_multiplier = 4.8
        
        # Mission parameters
        self.max_engagement_range = 10000.0
        self.cooperation_radius = 2500.0
        self.prediction_horizon = 3.0
        
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
        
        # Communication state tracking
        self.comm_status: Dict[str, Dict] = {}
        self.comm_history: Dict[str, List] = {}
        self.jamming_zones: List[Dict] = []
        self.last_comm_update: Dict[str, float] = {}
        self.isolated_drones: Set[str] = set()
        self.mesh_networks: Dict[str, Set[str]] = {}
        
        self.frame_count = 0
        self.mission_time = 0.0
        self.dt = 1.0 / 30.0
        
        if scenario_config:
            self._load_all_parameters(scenario_config)
            
        print("=" * 60)
        print("📡 COMMUNICATION DEGRADATION SYSTEM INITIALIZED")
        print(f"   Communication range: {self.communication_range}m")
        print(f"   Jamming effectiveness: {self.jamming_effectiveness * 100}%")
        print(f"   Fallback behavior: {self.fallback_behavior}")
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
            self.communication_range = float(ai.get('communication_range', self.communication_range))
            self.communication_base_latency = float(ai.get('communication_base_latency', self.communication_base_latency))
            self.communication_packet_loss_base = float(ai.get('communication_packet_loss_base', self.communication_packet_loss_base))
            self.jamming_effectiveness = float(ai.get('jamming_effectiveness', self.jamming_effectiveness))
            self.terrain_mask_threshold = float(ai.get('terrain_mask_threshold', self.terrain_mask_threshold))
            
            self.degraded_formation_distance = float(ai.get('degraded_formation_distance', self.degraded_formation_distance))
            self.degraded_coordination_factor = float(ai.get('degraded_coordination_factor', self.degraded_coordination_factor))
            self.local_decision_radius = float(ai.get('local_decision_radius', self.local_decision_radius))
            self.emergency_protocol_distance = float(ai.get('emergency_protocol_distance', self.emergency_protocol_distance))
            
            self.autonomous_engagement_threshold = float(ai.get('autonomous_engagement_threshold', self.autonomous_engagement_threshold))
            self.mesh_network_hop_limit = int(ai.get('mesh_network_hop_limit', self.mesh_network_hop_limit))
            self.hardened_comm_resistance = float(ai.get('hardened_comm_resistance', self.hardened_comm_resistance))
            
            mission = scenario_config.get('mission_parameters', {})
            self.comm_failure_timeout = float(mission.get('comm_failure_timeout', self.comm_failure_timeout))
            self.reconnection_attempt_interval = float(mission.get('reconnection_attempt_interval', self.reconnection_attempt_interval))
            self.fallback_behavior = mission.get('fallback_behavior', self.fallback_behavior)
            self.emergency_rtb_threshold = float(mission.get('emergency_rtb_threshold', self.emergency_rtb_threshold))
            
            # Load jamming zones
            env = scenario_config.get('environment', {})
            num_zones = int(env.get('jamming_zones', 0))
            self.jamming_zones = []
            for i in range(1, num_zones + 1):
                zone_data = env.get(f'jamming_zone_{i}', '0,0,0').split(',')
                if len(zone_data) >= 3:
                    self.jamming_zones.append({
                        'x': float(zone_data[0]),
                        'y': float(zone_data[1]),
                        'radius': float(zone_data[2])
                    })
                    
        except Exception as e:
            print(f"⚠️ Parameter loading error: {e} - using defaults")

    def _update_communication_status(self, interceptors: List[Dict], targets: List[Dict]):
        """Update communication status for all drones"""
        # Initialize comm status for new drones
        for drone in interceptors:
            if drone['id'] not in self.comm_status:
                self.comm_status[drone['id']] = {
                    'signal_strength': 1.0,
                    'latency': self.communication_base_latency,
                    'packet_loss': self.communication_packet_loss_base,
                    'last_contact': self.mission_time,
                    'connected': True,
                    'comm_type': drone.get('type', 'standard_comms'),
                    'mesh_neighbors': set()
                }
                self.comm_history[drone['id']] = []
                self.last_comm_update[drone['id']] = self.mission_time
        
        # Update communication effects
        for drone in interceptors:
            if not drone.get('active', True):
                continue
                
            status = self.comm_status[drone['id']]
            
            # 1. Distance-based signal degradation
            base_distance = math.sqrt(drone['x']**2 + drone['y']**2)
            distance_factor = max(0.1, 1.0 - base_distance / self.communication_range)
            
            # 2. Altitude effects (lower altitude = worse comms)
            altitude = drone.get('z', 0)
            altitude_factor = max(0.3, min(1.0, altitude / 1000.0))
            
            # 3. Jamming zone effects
            jamming_factor = 1.0
            for zone in self.jamming_zones:
                dist_to_zone = math.sqrt((drone['x'] - zone['x'])**2 + 
                                       (drone['y'] - zone['y'])**2)
                if dist_to_zone < zone['radius']:
                    zone_effect = 1.0 - (dist_to_zone / zone['radius'])
                    jamming_factor *= (1.0 - self.jamming_effectiveness * zone_effect)
            
            # 4. Target jamming effects
            for target in targets:
                if target.get('active', True) and 'jammer' in target.get('type', ''):
                    dist_to_jammer = self._dist(drone, target)
                    if dist_to_jammer < 2000:
                        jammer_effect = 1.0 - (dist_to_jammer / 2000)
                        jamming_factor *= (1.0 - 0.5 * jammer_effect)
            
            # 5. Communication type modifiers
            comm_type = status['comm_type']
            type_modifier = 1.0
            if 'hardened' in comm_type:
                type_modifier = 1.0 + self.hardened_comm_resistance
            elif 'autonomous' in comm_type:
                type_modifier = 1.2
            
            # Calculate final signal strength
            signal_strength = distance_factor * altitude_factor * jamming_factor * type_modifier
            signal_strength = max(0.0, min(1.0, signal_strength))
            
            # Update communication parameters
            status['signal_strength'] = signal_strength
            status['latency'] = self.communication_base_latency / max(0.1, signal_strength)
            status['packet_loss'] = min(0.95, self.communication_packet_loss_base + (1.0 - signal_strength) * 0.5)
            
            # Determine if connection is lost
            if signal_strength < 0.1 or random.random() < status['packet_loss']:
                if status['connected']:
                    status['connected'] = False
                    status['last_contact'] = self.mission_time
                    self.isolated_drones.add(drone['id'])
                    print(f"📡❌ COMM LOST: {drone['id']} (signal={signal_strength:.2f})")
            else:
                if not status['connected'] and self.mission_time - status['last_contact'] > self.reconnection_attempt_interval:
                    status['connected'] = True
                    self.isolated_drones.discard(drone['id'])
                    print(f"📡✅ COMM RESTORED: {drone['id']}")
            
            # Update mesh network connections
            if 'mesh' in comm_type:
                self._update_mesh_network(drone, interceptors)

    def _update_mesh_network(self, drone: Dict, interceptors: List[Dict]):
        """Update mesh network connections for mesh-capable drones"""
        drone_id = drone['id']
        status = self.comm_status[drone_id]
        
        # Find nearby mesh-capable drones
        mesh_neighbors = set()
        for other in interceptors:
            if other['id'] == drone_id or not other.get('active', True):
                continue
                
            other_status = self.comm_status.get(other['id'])
            if other_status and 'mesh' in other_status.get('comm_type', ''):
                dist = self._dist(drone, other)
                if dist < self.local_decision_radius:
                    mesh_neighbors.add(other['id'])
        
        status['mesh_neighbors'] = mesh_neighbors
        
        # Update mesh network groups
        if drone_id not in self.mesh_networks:
            self.mesh_networks[drone_id] = {drone_id}
        
        # Merge mesh networks
        for neighbor in mesh_neighbors:
            if neighbor in self.mesh_networks:
                self.mesh_networks[drone_id].update(self.mesh_networks[neighbor])
                self.mesh_networks[neighbor] = self.mesh_networks[drone_id]

    def _get_effective_comm_range(self, drone: Dict) -> float:
        """Get effective communication range based on drone status"""
        status = self.comm_status.get(drone['id'], {})
        signal = status.get('signal_strength', 1.0)
        
        if not status.get('connected', True):
            # Isolated drone - only local awareness
            return self.local_decision_radius
        elif 'mesh' in status.get('comm_type', ''):
            # Mesh network - extended range through relays
            mesh_size = len(self.mesh_networks.get(drone['id'], {drone['id']}))
            return self.local_decision_radius * min(mesh_size, self.mesh_network_hop_limit)
        else:
            # Standard communication
            return self.cooperation_radius * signal

    def _make_local_decision(self, drone: Dict, targets: List[Dict], interceptors: List[Dict]) -> Dict:
        """Make autonomous decisions when communication is lost"""
        # Find nearest target within local awareness
        effective_range = self._get_effective_comm_range(drone)
        nearest_target = None
        min_dist = float('inf')
        
        for target in targets:
            if target.get('active', True):
                dist = self._dist(drone, target)
                if dist < min_dist and dist < effective_range:
                    min_dist = dist
                    nearest_target = target
        
        if not nearest_target:
            # No target in range - return to base behavior
            return self._return_to_base(drone)
        
        # Check if we can engage autonomously
        if min_dist < self.autonomous_engagement_threshold:
            # Autonomous engagement
            return self._autonomous_pursuit(drone, nearest_target)
        else:
            # Maintain defensive posture
            return self._defensive_patrol(drone, nearest_target)

    def _autonomous_pursuit(self, drone: Dict, target: Dict) -> Dict:
        """Pursue target autonomously without pack coordination"""
        d = self._dist(drone, target)
        
        # Strike check
        if d <= self.strike_distance:
            target['active'] = False
            drone['active'] = False
            print(f"💥💥💥 AUTONOMOUS STRIKE: {drone['id']} destroyed {target['id']} at {d:.1f}m 💥💥💥")
            return self._decision(0,0,0, target['id'], 'autonomous', 'STRIKE_SUCCESS')
        
        # Direct pursuit
        dx = target['x'] - drone['x']
        dy = target['y'] - drone['y']
        dz = target.get('z', 0.0) - drone.get('z', 0.0)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
        
        # Use conservative acceleration when isolated
        status = self.comm_status.get(drone['id'], {})
        coord_factor = 1.0 if status.get('connected', True) else self.degraded_coordination_factor
        
        ux, uy, uz = dx/dist, dy/dist, dz/dist
        max_accel = self.killer_max_acceleration * coord_factor
        ax = max_accel * ux
        ay = max_accel * uy
        az = max_accel * uz
        
        return self._decision(ax, ay, az, target['id'], 'autonomous', 'isolated_pursuit')

    def _defensive_patrol(self, drone: Dict, target: Dict) -> Dict:
        """Maintain defensive position when isolated"""
        # Circle at safe distance
        dx = target['x'] - drone['x']
        dy = target['y'] - drone['y']
        dist = math.sqrt(dx*dx + dy*dy) or 1.0
        
        desired_dist = self.emergency_protocol_distance
        error = dist - desired_dist
        
        if abs(error) > 50:
            # Adjust distance
            ux, uy = dx/dist, dy/dist
            gain = self.green_max_acceleration * 0.3 * (1.0 if error > 0 else -1.0)
            ax = gain * ux
            ay = gain * uy
            az = 0.0
        else:
            # Orbit
            tx, ty = -dy/dist, dx/dist
            orbit_speed = 20.0
            dvx, dvy = drone.get('vx', 0), drone.get('vy', 0)
            ax = 2.0 * (orbit_speed * tx - dvx)
            ay = 2.0 * (orbit_speed * ty - dvy)
            az = 0.0
        
        return self._decision(ax, ay, az, target['id'], 'defensive', 'isolated_patrol')

    def _return_to_base(self, drone: Dict) -> Dict:
        """Return to base when no targets and comms lost"""
        # Head back to origin
        dx = -drone['x']
        dy = -drone['y']
        dz = -drone.get('z', 0)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
        
        ux, uy, uz = dx/dist, dy/dist, dz/dist
        gain = self.green_max_acceleration * 0.5
        ax = gain * ux
        ay = gain * uy
        az = gain * uz
        
        return self._decision(ax, ay, az, 'base', 'rtb', 'comm_failure_rtb')

    def update(self, targets: List[Dict], interceptors: List[Dict], dt: float, wind: Dict, scenario_config: Dict = None) -> Dict[str, Dict]:
        if scenario_config:
            self._load_all_parameters(scenario_config)
            
        eff_dt = dt if (dt is not None and dt > 0.0) else (self.dt if self.frame_count == 0 else 0.0)
        
        self.frame_count += 1
        self.mission_time += eff_dt
        self.dt = eff_dt
        
        # Update communication status
        self._update_communication_status(interceptors, targets)
        
        # Physics step
        self._update_physics(targets, interceptors, eff_dt, wind)
        
        # Initialize packs once
        if not self.packs_initialized:
            self._create_packs(targets, interceptors)
            self.packs_initialized = True
        
        # Debug output
        if self.frame_count % 30 == 0:
            self._print_comm_debug(targets, interceptors)
            
        # Generate decisions
        decisions: Dict[str, Dict] = {}
        
        for drone in interceptors:
            if not drone.get('active', True):
                continue
                
            drone_id = drone['id']
            status = self.comm_status.get(drone_id, {})
            
            # Check if drone is isolated
            if drone_id in self.isolated_drones or not status.get('connected', True):
                # Make local autonomous decision
                decisions[drone_id] = self._make_local_decision(drone, targets, interceptors)
            else:
                # Normal pack-based decisions
                pack_id = self.drone_pack_map.get(drone_id)
                if pack_id and pack_id in self.packs:
                    pack = self.packs[pack_id]
                    # Only process if we haven't already for this pack
                    if f"pack_{pack['target_id']}_processed" not in decisions:
                        pack_decisions = self._update_pack_with_comm(pack, targets, interceptors)
                        decisions.update(pack_decisions)
                        decisions[f"pack_{pack['target_id']}_processed"] = True
                        
        return {k: v for k, v in decisions.items() if not k.endswith('_processed')}

    def _update_pack_with_comm(self, pack: Dict, targets: List[Dict], interceptors: List[Dict]) -> Dict[str, Dict]:
        """Update pack with communication awareness"""
        out: Dict[str, Dict] = {}

        tgt_id = pack['target_id']
        target = next((t for t in targets if t['id'] == tgt_id and t.get('active', True)), None)
        if not target:
            for did in pack['drone_ids']:
                d = next((x for x in interceptors if x['id'] == did and x.get('active', True)), None)
                if d:
                    out[did] = self._decision(0,0,0, tgt_id, 'idle', 'target_destroyed')
            return out

        # Get connected pack members
        connected_drones = []
        degraded_drones = []
        
        for d in interceptors:
            if d['id'] not in pack['drone_ids'] or not d.get('active', True):
                continue
                
            status = self.comm_status.get(d['id'], {})
            if status.get('connected', True):
                if status.get('signal_strength', 1.0) > 0.5:
                    connected_drones.append(d)
                else:
                    degraded_drones.append(d)
        
        if not connected_drones and not degraded_drones:
            return out
        
        # Adjust formation based on communication quality
        avg_signal = sum(self.comm_status[d['id']]['signal_strength'] 
                        for d in connected_drones + degraded_drones) / max(1, len(connected_drones + degraded_drones))
        
        if avg_signal < 0.7:
            # Use degraded formation distance
            formation_distance = self.degraded_formation_distance
        else:
            formation_distance = self.pack_formation_distance
        
        # Time since chase started
        chase_time = self.mission_time - pack.get('chase_start_time', 0.0)
        current_state = pack.get('pack_state', 'chasing')
        
        # Process connected drones normally
        if current_state == 'chasing':
            if chase_time >= self.chase_duration:
                pack['pack_state'] = 'following'
            
            for d in connected_drones + degraded_drones:
                out[d['id']] = self._chase_target(d, target)

        elif current_state == 'following':
            all_drones = connected_drones + degraded_drones
            avg_dist = sum(self._dist(d, target) for d in all_drones) / len(all_drones)
            
            if avg_dist <= self.follow_distance:
                pack['pack_state'] = 'forming'
            
            for d in all_drones:
                out[d['id']] = self._follow_target(d, target)

        elif current_state == 'forming':
            self._compute_ring(pack, target, formation_distance)
            
            close_drones = 0
            all_drones = connected_drones + degraded_drones
            for d in all_drones:
                fp = pack['formation_positions'].get(d['id'])
                if fp:
                    slot_dist = math.sqrt((fp['x'] - d['x'])**2 + (fp['y'] - d['y'])**2)
                    if slot_dist <= self.ready_epsilon * (2.0 if avg_signal < 0.7 else 1.0):
                        close_drones += 1
            
            # More lenient formation requirements with degraded comms
            required_drones = max(2, len(all_drones) - 1) if avg_signal < 0.5 else 3
            formation_ready = (close_drones >= required_drones)
            
            if formation_ready:
                pack['pack_state'] = 'engaging'
                pack['engage_unlocked'] = True
            
            for d in all_drones:
                fp = pack['formation_positions'].get(d['id'])
                out[d['id']] = self._move_to_formation(d, target, fp)

        elif current_state == 'engaging':
            self._compute_ring(pack, target, formation_distance)
            
            # Select killer from connected drones only
            if pack.get('killer_drone') is None and connected_drones:
                closest, closest_dist = self._closest(connected_drones, target)
                if closest:
                    pack['killer_drone'] = closest['id']
                    pack['green_drones'] = [d['id'] for d in connected_drones + degraded_drones 
                                          if d['id'] != pack['killer_drone']]
            
            for d in connected_drones + degraded_drones:
                if d['id'] == pack.get('killer_drone'):
                    out[d['id']] = self._killer_pursuit(pack, d, target)
                else:
                    fp = pack['formation_positions'].get(d['id'])
                    out[d['id']] = self._track_parallel(d, target, fp)

        return out

    def _print_comm_debug(self, targets: List[Dict], interceptors: List[Dict]):
        """Debug output for communication status"""
        print("\n" + "="*80)
        print(f"📡 COMMUNICATION STATUS: t={self.mission_time:.2f}s")
        
        # Jamming zones
        if self.jamming_zones:
            print(f"⚡ JAMMING ZONES: {len(self.jamming_zones)} active")
        
        # Drone communication status
        total_drones = len([d for d in interceptors if d.get('active', True)])
        isolated_count = len(self.isolated_drones)
        
        print(f"📊 NETWORK STATUS: {total_drones - isolated_count}/{total_drones} connected")
        
        # Pack communication health
        for pack_id, pack in self.packs.items():
            connected = 0
            degraded = 0
            lost = 0
            
            for did in pack['drone_ids']:
                drone = next((d for d in interceptors if d['id'] == did), None)
                if drone and drone.get('active', True):
                    status = self.comm_status.get(did, {})
                    if not status.get('connected', True):
                        lost += 1
                    elif status.get('signal_strength', 1.0) < 0.5:
                        degraded += 1
                    else:
                        connected += 1
            
            print(f"   {pack_id}: ✅{connected} 🟡{degraded} ❌{lost}")
        
        # Isolated drones
        if self.isolated_drones:
            print(f"🚨 ISOLATED DRONES: {list(self.isolated_drones)}")
        
        # Mesh networks
        unique_meshes = set()
        for mesh_id, members in self.mesh_networks.items():
            if len(members) > 1:
                unique_meshes.add(frozenset(members))
        
        if unique_meshes:
            print(f"🔗 MESH NETWORKS: {len(unique_meshes)} active")
            for i, mesh in enumerate(unique_meshes):
                print(f"   Mesh {i+1}: {len(mesh)} nodes")
        
        print("="*80)

    def _create_packs(self, targets: List[Dict], interceptors: List[Dict]):
        """Create packs with communication awareness"""
        act_targets = [t for t in targets if t.get('active', True)]
        act_interceptors = [i for i in interceptors if i.get('active', True)]
        
        print("\n" + "🐺" * 20)
        print(f"🐺 CREATING COMM-AWARE PACKS: {len(act_targets)} targets, {len(act_interceptors)} drones")
        
        # Group drones by communication type
        comm_groups = {
            'standard': [],
            'hardened': [],
            'mesh': [],
            'autonomous': []
        }
        
        for drone in act_interceptors:
            comm_type = drone.get('type', 'standard_comms')
            if 'hardened' in comm_type:
                comm_groups['hardened'].append(drone)
            elif 'mesh' in comm_type:
                comm_groups['mesh'].append(drone)
            elif 'autonomous' in comm_type:
                comm_groups['autonomous'].append(drone)
            else:
                comm_groups['standard'].append(drone)
        
        # Assign mixed packs for resilience
        idx = 0
        for tgt in act_targets:
            pack_id = f"pack_{tgt['id']}"
            drones = []
            
            # Try to mix communication types
            for _ in range(4):
                if idx < len(act_interceptors):
                    # Rotate through comm types
                    for comm_type in ['autonomous', 'hardened', 'mesh', 'standard']:
                        if comm_groups[comm_type]:
                            d = comm_groups[comm_type].pop(0)
                            drones.append(d['id'])
                            self.drone_pack_map[d['id']] = pack_id
                            idx += 1
                            break
                            
            if drones:
                self.packs[pack_id] = {
                    'target_id': tgt['id'],
                    'drone_ids': drones,
                    'killer_drone': None,
                    'green_drones': drones.copy(),
                    'pack_state': 'chasing',
                    'formation_positions': {},
                    'chase_start_time': self.mission_time,
                    'first_ready_time': None,
                    'engage_unlocked': False,
                }
                print(f"🎯 PACK [{pack_id}] - Mixed comm types - Drones: {drones}")
                
        print("🐺" * 20 + "\n")

    # Include all base methods from original AI_System
    def _update_physics(self, targets: List[Dict], interceptors: List[Dict], dt: float, wind: Dict):
        if dt <= 0.0:
            return

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

        for d in interceptors:
            if not d.get('active', True):
                continue
            
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

        grid_angles = [0.0, math.pi/2, math.pi, 3*math.pi/2]
        pos: Dict[str, Dict] = {}
        for i, did in enumerate(pack['drone_ids']):
            ang = base_heading + grid_angles[i % 4]
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
        
        if d <= self.strike_distance:
            target['active'] = False
            drone['active'] = False
            print(f"💥💥💥 STRIKE SUCCESS: {drone['id']} destroyed {target['id']} at {d:.1f}m 💥💥💥")
            return self._decision(0,0,0, target['id'], 'killer', 'STRIKE_SUCCESS')

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