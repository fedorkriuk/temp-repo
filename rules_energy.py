import math
from typing import List, Dict, Optional, Tuple, Set
import numpy as np

class AI_System_Energy:
    """Pack logic with energy/fuel constraints and efficient path planning"""

    def __init__(self, scenario_config: Dict = None):
        # Base pack parameters
        self.pack_formation_distance = 380.0
        self.pack_activation_distance = 1000.0
        self.strike_distance = 30.0
        
        # Energy management parameters
        self.fuel_critical_threshold = 20.0
        self.fuel_bingo_threshold = 30.0
        self.fuel_conservative_threshold = 50.0
        
        # Fuel consumption parameters
        self.base_fuel_consumption_rate = 0.1
        self.acceleration_fuel_multiplier = 2.5
        self.velocity_fuel_multiplier = 1.5
        self.altitude_fuel_modifier = 0.8
        self.headwind_fuel_penalty = 1.8
        
        # Type-specific consumption rates
        self.consumption_rates = {
            'standard_fuel': 1.0,
            'extended_fuel': 0.9,
            'efficient_engine': 0.6,
            'sprint_capable': 1.5
        }
        
        # Efficiency optimization
        self.optimal_cruise_speed = 180.0
        self.formation_draft_benefit = 0.85
        self.altitude_optimization = True
        self.route_optimization = True
        
        # Energy-aware behaviors
        self.energy_prediction_horizon = 10.0
        self.rtb_safety_margin = 1.2
        self.pursuit_efficiency_weight = 0.7
        self.intercept_probability_threshold = 0.6
        
        # Gliding parameters
        self.glide_ratio = 15.0
        self.min_glide_speed = 50.0
        self.glide_fuel_consumption = 0.1
        
        # Physics parameters
        self.green_max_velocity = 230.0
        self.green_max_acceleration = 70.0
        
        self.killer_max_velocity = 260.0
        self.killer_max_acceleration = 140.0
        self.killer_deceleration_multiplier = 3.0
        
        self.strike_acceleration_multiplier = 4.5
        
        # Mission parameters
        self.max_engagement_range = 10000.0
        self.cooperation_radius = 3000.0
        self.communication_range = 18000.0
        self.prediction_horizon = 4.0
        
        # Pursuit authorization based on fuel
        self.min_fuel_for_pursuit = 40.0
        self.min_fuel_for_strike = 25.0
        self.abort_pursuit_fuel = 15.0
        
        # Green orbit parameters
        self.ring_orbit_speed = 25.0
        self.ring_orbit_direction = 1
        self.green_speed_margin = 15.0
        
        # State tracking
        self.chase_duration = 8.0
        self.follow_distance = 600.0
        self.ready_epsilon = 150.0
        
        # Runtime state
        self.packs: Dict[str, Dict] = {}
        self.drone_pack_map: Dict[str, str] = {}
        self.drone_fuel: Dict[str, Dict] = {}
        self.fuel_history: Dict[str, List] = {}
        self.headwind_zones: List[Dict] = []
        self.packs_initialized = False
        
        # Mission tracking
        self.mission_start_fuel: Dict[str, float] = {}
        self.rtb_drones: Set[str] = set()
        self.fuel_emergency_drones: Set[str] = set()
        
        self.frame_count = 0
        self.mission_time = 0.0
        self.dt = 1.0 / 30.0
        
        if scenario_config:
            self._load_all_parameters(scenario_config)
            
        print("=" * 60)
        print("⚡ ENERGY CONSTRAINT SYSTEM INITIALIZED")
        print(f"   Fuel thresholds: Critical={self.fuel_critical_threshold}%, " +
              f"Bingo={self.fuel_bingo_threshold}%, Conservative={self.fuel_conservative_threshold}%")
        print(f"   Optimal cruise speed: {self.optimal_cruise_speed} m/s")
        print(f"   RTB safety margin: {self.rtb_safety_margin}x")
        print("=" * 60)

    def _load_all_parameters(self, scenario_config: Dict):
        try:
            physics = scenario_config.get('physics', {})
            self.base_fuel_consumption_rate = float(physics.get('base_fuel_consumption_rate', self.base_fuel_consumption_rate))
            self.acceleration_fuel_multiplier = float(physics.get('acceleration_fuel_multiplier', self.acceleration_fuel_multiplier))
            self.velocity_fuel_multiplier = float(physics.get('velocity_fuel_multiplier', self.velocity_fuel_multiplier))
            self.altitude_fuel_modifier = float(physics.get('altitude_fuel_modifier', self.altitude_fuel_modifier))
            self.headwind_fuel_penalty = float(physics.get('headwind_fuel_penalty', self.headwind_fuel_penalty))
            
            # Load type-specific rates
            for fuel_type in ['standard', 'extended', 'efficient', 'sprint']:
                key = f'{fuel_type}_consumption_rate'
                if key in physics:
                    self.consumption_rates[f'{fuel_type}_fuel'] = float(physics[key])
                    
            self.glide_ratio = float(physics.get('glide_ratio', self.glide_ratio))
            self.min_glide_speed = float(physics.get('min_glide_speed', self.min_glide_speed))
            self.glide_fuel_consumption = float(physics.get('glide_fuel_consumption', self.glide_fuel_consumption))
            
            self.green_max_acceleration = float(physics.get('green_max_acceleration', self.green_max_acceleration))
            self.green_max_velocity = float(physics.get('green_max_velocity', self.green_max_velocity))
            self.killer_max_acceleration = float(physics.get('killer_max_acceleration', self.killer_max_acceleration))
            self.killer_max_velocity = float(physics.get('killer_max_velocity', self.killer_max_velocity))
            
            ai = scenario_config.get('ai_parameters', {})
            self.fuel_critical_threshold = float(ai.get('fuel_critical_threshold', self.fuel_critical_threshold))
            self.fuel_bingo_threshold = float(ai.get('fuel_bingo_threshold', self.fuel_bingo_threshold))
            self.fuel_conservative_threshold = float(ai.get('fuel_conservative_threshold', self.fuel_conservative_threshold))
            
            self.optimal_cruise_speed = float(ai.get('optimal_cruise_speed', self.optimal_cruise_speed))
            self.formation_draft_benefit = float(ai.get('formation_draft_benefit', self.formation_draft_benefit))
            self.energy_prediction_horizon = float(ai.get('energy_prediction_horizon', self.energy_prediction_horizon))
            self.rtb_safety_margin = float(ai.get('rtb_safety_margin', self.rtb_safety_margin))
            
            mission = scenario_config.get('mission_parameters', {})
            self.min_fuel_for_pursuit = float(mission.get('min_fuel_for_pursuit', self.min_fuel_for_pursuit))
            self.min_fuel_for_strike = float(mission.get('min_fuel_for_strike', self.min_fuel_for_strike))
            self.abort_pursuit_fuel = float(mission.get('abort_pursuit_fuel', self.abort_pursuit_fuel))
            
            # Load headwind zones
            env = scenario_config.get('environment', {})
            num_zones = int(env.get('headwind_zones', 0))
            self.headwind_zones = []
            for i in range(1, num_zones + 1):
                zone_data = env.get(f'headwind_zone_{i}', '0,0,0,0').split(',')
                if len(zone_data) >= 4:
                    self.headwind_zones.append({
                        'x': float(zone_data[0]),
                        'y': float(zone_data[1]),
                        'radius': float(zone_data[2]),
                        'strength': float(zone_data[3])
                    })
                    
        except Exception as e:
            print(f"⚠️ Parameter loading error: {e} - using defaults")

    def _initialize_drone_fuel(self, interceptors: List[Dict]):
        """Initialize fuel tracking for all drones"""
        for drone in interceptors:
            if drone['id'] not in self.drone_fuel:
                # Extract initial fuel from drone data (last parameter)
                initial_fuel = 100.0  # Default
                if len(drone) > 11 and isinstance(drone.get('fuel', None), (int, float)):
                    initial_fuel = float(drone['fuel'])
                
                drone_type = drone.get('type', 'standard_fuel')
                
                self.drone_fuel[drone['id']] = {
                    'current': initial_fuel,
                    'initial': initial_fuel,
                    'type': drone_type,
                    'consumption_rate': self.consumption_rates.get(drone_type, 1.0),
                    'last_consumption': 0.0,
                    'total_consumed': 0.0,
                    'efficiency_score': 0.0,
                    'in_draft': False,
                    'gliding': False
                }
                
                self.mission_start_fuel[drone['id']] = initial_fuel
                self.fuel_history[drone['id']] = []
                
                print(f"⚡ {drone['id']} initialized: fuel={initial_fuel}%, type={drone_type}")

    def _calculate_fuel_consumption(self, drone: Dict, dt: float) -> float:
        """Calculate fuel consumption based on drone state"""
        fuel_data = self.drone_fuel.get(drone['id'])
        if not fuel_data:
            return 0.0
        
        # Base consumption
        consumption = self.base_fuel_consumption_rate * fuel_data['consumption_rate'] * dt
        
        # Acceleration penalty
        ax, ay, az = drone.get('ax', 0), drone.get('ay', 0), drone.get('az', 0)
        accel_mag = math.sqrt(ax*ax + ay*ay + az*az)
        if accel_mag > 0:
            consumption += accel_mag / 100.0 * self.acceleration_fuel_multiplier * dt
        
        # Velocity penalty (non-linear, encourages optimal cruise speed)
        vx, vy, vz = drone.get('vx', 0), drone.get('vy', 0), drone.get('vz', 0)
        speed = math.sqrt(vx*vx + vy*vy + vz*vz)
        
        # Efficiency curve - minimum consumption at optimal cruise speed
        speed_efficiency = 1.0
        if speed > 0:
            deviation = abs(speed - self.optimal_cruise_speed) / self.optimal_cruise_speed
            speed_efficiency = 1.0 + deviation * 0.5
        
        consumption += (speed / 100.0) * self.velocity_fuel_multiplier * speed_efficiency * dt
        
        # Altitude modifier (higher = more efficient)
        altitude = drone.get('z', 0)
        if altitude > 1000:
            altitude_factor = self.altitude_fuel_modifier ** (altitude / 1000)
            consumption *= altitude_factor
        
        # Headwind penalty
        for zone in self.headwind_zones:
            dist_to_zone = math.sqrt((drone['x'] - zone['x'])**2 + (drone['y'] - zone['y'])**2)
            if dist_to_zone < zone['radius']:
                wind_effect = 1.0 - (dist_to_zone / zone['radius'])
                headwind_penalty = 1.0 + (self.headwind_fuel_penalty - 1.0) * wind_effect * abs(zone['strength']) / 20
                consumption *= headwind_penalty
        
        # Formation draft benefit
        if fuel_data.get('in_draft', False):
            consumption *= self.formation_draft_benefit
        
        # Gliding mode
        if fuel_data.get('gliding', False) and speed > self.min_glide_speed:
            consumption = self.glide_fuel_consumption * dt
        
        return consumption

    def _update_fuel_status(self, interceptors: List[Dict], dt: float):
        """Update fuel levels and check for emergency conditions"""
        for drone in interceptors:
            if not drone.get('active', True):
                continue
                
            fuel_data = self.drone_fuel.get(drone['id'])
            if not fuel_data:
                continue
            
            # Calculate and apply consumption
            consumption = self._calculate_fuel_consumption(drone, dt)
            fuel_data['current'] = max(0, fuel_data['current'] - consumption)
            fuel_data['last_consumption'] = consumption
            fuel_data['total_consumed'] += consumption
            
            # Update efficiency score
            if fuel_data['total_consumed'] > 0:
                distance_traveled = math.sqrt(drone['x']**2 + drone['y']**2)
                fuel_data['efficiency_score'] = distance_traveled / fuel_data['total_consumed']
            
            # Check fuel states
            fuel_percent = fuel_data['current']
            
            if fuel_percent <= 0:
                # Out of fuel - drone crashes
                drone['active'] = False
                print(f"💥 FUEL EXHAUSTED: {drone['id']} crashed!")
                
            elif fuel_percent < self.fuel_critical_threshold and drone['id'] not in self.fuel_emergency_drones:
                # Critical fuel - emergency RTB
                self.fuel_emergency_drones.add(drone['id'])
                self.rtb_drones.add(drone['id'])
                print(f"🚨 FUEL CRITICAL: {drone['id']} at {fuel_percent:.1f}% - EMERGENCY RTB!")
                
            elif fuel_percent < self.fuel_bingo_threshold and drone['id'] not in self.rtb_drones:
                # Bingo fuel - must RTB
                rtb_distance = math.sqrt(drone['x']**2 + drone['y']**2)
                rtb_fuel_needed = self._estimate_rtb_fuel(drone, rtb_distance)
                
                if fuel_percent < rtb_fuel_needed * self.rtb_safety_margin:
                    self.rtb_drones.add(drone['id'])
                    print(f"⚠️ BINGO FUEL: {drone['id']} at {fuel_percent:.1f}% - RTB initiated")
            
            # Store history
            self.fuel_history[drone['id']].append({
                'time': self.mission_time,
                'fuel': fuel_data['current'],
                'consumption': consumption
            })

    def _estimate_rtb_fuel(self, drone: Dict, distance: float) -> float:
        """Estimate fuel needed to return to base"""
        # Assume optimal cruise speed for RTB
        time_to_base = distance / self.optimal_cruise_speed
        
        # Base consumption at cruise
        fuel_data = self.drone_fuel.get(drone['id'])
        if not fuel_data:
            return 50.0  # Conservative default
        
        base_rate = self.base_fuel_consumption_rate * fuel_data['consumption_rate']
        cruise_rate = (self.optimal_cruise_speed / 100.0) * self.velocity_fuel_multiplier
        
        estimated_consumption = (base_rate + cruise_rate) * time_to_base
        
        # Add margin for descent and landing
        return estimated_consumption * 1.2

    def _check_formation_draft(self, drone: Dict, pack: Dict, interceptors: List[Dict]):
        """Check if drone is in formation draft position"""
        fuel_data = self.drone_fuel.get(drone['id'])
        if not fuel_data:
            return
        
        # Reset draft status
        fuel_data['in_draft'] = False
        
        # Check if in formation and behind another drone
        if pack.get('pack_state') in ['forming', 'engaging']:
            pack_drones = [d for d in interceptors if d['id'] in pack['drone_ids'] and d.get('active', True)]
            
            for other in pack_drones:
                if other['id'] == drone['id']:
                    continue
                    
                # Check if behind and close
                dx = other['x'] - drone['x']
                dy = other['y'] - drone['y']
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < 100:  # Within draft distance
                    # Check if behind relative to velocity
                    vx, vy = drone.get('vx', 0), drone.get('vy', 0)
                    if vx*dx + vy*dy > 0:  # Behind
                        fuel_data['in_draft'] = True
                        break

    def _energy_efficient_pursuit(self, drone: Dict, target: Dict) -> Dict:
        """Pursue target with energy efficiency considerations"""
        fuel_data = self.drone_fuel.get(drone['id'])
        if not fuel_data:
            return self._decision(0,0,0, target['id'], 'no_fuel_data', 'error')
        
        fuel_percent = fuel_data['current']
        
        # Check if we have enough fuel for pursuit
        if fuel_percent < self.min_fuel_for_pursuit:
            return self._conservative_orbit(drone, target)
        
        d = self._dist(drone, target)
        
        # Estimate intercept feasibility
        intercept_time = d / self.killer_max_velocity
        intercept_fuel = self._estimate_pursuit_fuel(drone, d, intercept_time)
        rtb_distance = math.sqrt(drone['x']**2 + drone['y']**2)
        rtb_fuel = self._estimate_rtb_fuel(drone, rtb_distance)
        
        total_required = intercept_fuel + rtb_fuel * self.rtb_safety_margin
        
        if fuel_percent < total_required:
            # Not enough fuel for intercept and RTB
            return self._conservative_orbit(drone, target)
        
        # Energy-efficient pursuit strategy
        if d > 1000 and fuel_percent < self.fuel_conservative_threshold:
            # Long range - use optimal cruise speed
            return self._efficient_intercept(drone, target)
        else:
            # Standard pursuit
            return self._standard_killer_pursuit(drone, target)

    def _efficient_intercept(self, drone: Dict, target: Dict) -> Dict:
        """Calculate energy-efficient intercept trajectory"""
        # Predict intercept point
        tvx, tvy, tvz = target.get('vx', 0), target.get('vy', 0), target.get('vz', 0)
        
        # Use optimal cruise speed for efficiency
        intercept_speed = self.optimal_cruise_speed
        
        # Iterative intercept calculation
        best_time = 0
        best_point = (target['x'], target['y'], target.get('z', 0))
        
        for t in np.linspace(1, 20, 10):
            pred_x = target['x'] + tvx * t
            pred_y = target['y'] + tvy * t
            pred_z = target.get('z', 0) + tvz * t
            
            dist_to_pred = math.sqrt((pred_x - drone['x'])**2 + 
                                   (pred_y - drone['y'])**2 + 
                                   (pred_z - drone.get('z', 0))**2)
            
            time_to_reach = dist_to_pred / intercept_speed
            
            if abs(time_to_reach - t) < 0.5:
                best_time = t
                best_point = (pred_x, pred_y, pred_z)
                break
        
        # Navigate to intercept point
        dx = best_point[0] - drone['x']
        dy = best_point[1] - drone['y']
        dz = best_point[2] - drone.get('z', 0)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
        
        # Gentle acceleration for efficiency
        ux, uy, uz = dx/dist, dy/dist, dz/dist
        
        # Adjust acceleration based on fuel state
        fuel_data = self.drone_fuel.get(drone['id'])
        fuel_factor = min(1.0, fuel_data['current'] / self.fuel_conservative_threshold)
        
        gain = self.killer_max_acceleration * 0.5 * fuel_factor
        ax = gain * ux
        ay = gain * uy
        az = gain * uz
        
        return self._decision(ax, ay, az, target['id'], 'killer', 'efficient_intercept')

    def _conservative_orbit(self, drone: Dict, target: Dict) -> Dict:
        """Maintain conservative orbit when low on fuel"""
        # Orbit at safe distance
        dx = target['x'] - drone['x']
        dy = target['y'] - drone['y']
        dist = math.sqrt(dx*dx + dy*dy) or 1.0
        
        desired_dist = 1500  # Safe distance
        
        if abs(dist - desired_dist) > 100:
            # Adjust distance
            ux, uy = dx/dist, dy/dist
            gain = self.green_max_acceleration * 0.3 * (1 if dist > desired_dist else -1)
            ax = gain * ux
            ay = gain * uy
        else:
            # Gentle orbit
            tx, ty = -dy/dist, dx/dist
            orbit_speed = 50.0  # Slow orbit
            dvx, dvy = drone.get('vx', 0), drone.get('vy', 0)
            ax = 1.5 * (orbit_speed * tx - dvx)
            ay = 1.5 * (orbit_speed * ty - dvy)
        
        az = 0
        
        return self._decision(ax, ay, az, target['id'], 'conservative', 'fuel_saving_orbit')

    def _return_to_base_efficient(self, drone: Dict) -> Dict:
        """Efficient return to base trajectory"""
        # Calculate vector to base
        base_x, base_y, base_z = 0, 0, 0
        dx = base_x - drone['x']
        dy = base_y - drone['y']
        dz = base_z - drone.get('z', 0)
        
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
        
        fuel_data = self.drone_fuel.get(drone['id'])
        if not fuel_data:
            # Emergency - just go straight
            ux, uy, uz = dx/dist, dy/dist, dz/dist
            ax = self.green_max_acceleration * ux
            ay = self.green_max_acceleration * uy
            az = self.green_max_acceleration * uz
            return self._decision(ax, ay, az, 'base', 'rtb', 'emergency_rtb')
        
        # Check if we can glide
        altitude = drone.get('z', 0)
        glide_distance = altitude * self.glide_ratio
        
        if glide_distance > dist * 0.8 and altitude > 500:
            # Initiate glide
            fuel_data['gliding'] = True
            
            # Gentle descent toward base
            glide_angle = math.atan2(altitude, dist)
            ux, uy = dx/dist, dy/dist
            
            # Maintain glide speed
            target_speed = max(self.min_glide_speed, self.optimal_cruise_speed * 0.7)
            current_speed = math.sqrt(drone.get('vx', 0)**2 + drone.get('vy', 0)**2)
            
            if current_speed < target_speed:
                # Gentle acceleration to glide speed
                ax = self.green_max_acceleration * 0.3 * ux
                ay = self.green_max_acceleration * 0.3 * uy
                az = -self.green_max_acceleration * 0.1  # Gentle descent
            else:
                # Maintain glide
                ax = 0
                ay = 0
                az = -self.green_max_acceleration * 0.05
                
            return self._decision(ax, ay, az, 'base', 'rtb', 'gliding_home')
        else:
            fuel_data['gliding'] = False
            
            # Direct flight at optimal cruise speed
            ux, uy, uz = dx/dist, dy/dist, dz/dist
            
            # Speed control for efficiency
            current_speed = math.sqrt(drone.get('vx', 0)**2 + drone.get('vy', 0)**2)
            speed_error = self.optimal_cruise_speed - current_speed
            
            # Proportional control for speed
            accel_factor = min(1.0, abs(speed_error) / 50.0)
            
            if drone['id'] in self.fuel_emergency_drones:
                # Emergency - more aggressive
                gain = self.green_max_acceleration * 0.8
            else:
                # Normal RTB - efficient
                gain = self.green_max_acceleration * 0.5 * accel_factor
            
            ax = gain * ux
            ay = gain * uy
            az = gain * uz
            
            return self._decision(ax, ay, az, 'base', 'rtb', 'efficient_rtb')

    def _estimate_pursuit_fuel(self, drone: Dict, distance: float, time: float) -> float:
        """Estimate fuel consumption for pursuit"""
        fuel_data = self.drone_fuel.get(drone['id'])
        if not fuel_data:
            return 50.0
        
        # Aggressive pursuit consumption
        base_rate = self.base_fuel_consumption_rate * fuel_data['consumption_rate']
        accel_rate = self.killer_max_acceleration / 100.0 * self.acceleration_fuel_multiplier
        speed_rate = self.killer_max_velocity / 100.0 * self.velocity_fuel_multiplier * 1.2
        
        total_rate = base_rate + accel_rate + speed_rate
        return total_rate * time

    def update(self, targets: List[Dict], interceptors: List[Dict], dt: float, wind: Dict, scenario_config: Dict = None) -> Dict[str, Dict]:
        if scenario_config:
            self._load_all_parameters(scenario_config)
            
        eff_dt = dt if (dt is not None and dt > 0.0) else (self.dt if self.frame_count == 0 else 0.0)
        
        self.frame_count += 1
        self.mission_time += eff_dt
        self.dt = eff_dt
        
        # Initialize fuel tracking
        if not self.drone_fuel:
            self._initialize_drone_fuel(interceptors)
        
        # Update fuel status
        self._update_fuel_status(interceptors, eff_dt)
        
        # Physics step
        self._update_physics(targets, interceptors, eff_dt, wind)
        
        # Initialize packs once
        if not self.packs_initialized:
            self._create_packs(targets, interceptors)
            self.packs_initialized = True
        
        # Debug output
        if self.frame_count % 30 == 0:
            self._print_energy_debug(targets, interceptors)
            
        # Generate decisions
        decisions: Dict[str, Dict] = {}
        
        # Handle RTB drones first
        for drone_id in self.rtb_drones:
            drone = next((d for d in interceptors if d['id'] == drone_id and d.get('active', True)), None)
            if drone:
                decisions[drone_id] = self._return_to_base_efficient(drone)
        
        # Normal pack operations for non-RTB drones
        for pack_id, pack in self.packs.items():
            pack_decisions = self._update_energy_aware_pack(pack, targets, interceptors)
            decisions.update(pack_decisions)
            
        return decisions

    def _update_energy_aware_pack(self, pack: Dict, targets: List[Dict], interceptors: List[Dict]) -> Dict[str, Dict]:
        """Update pack with energy awareness"""
        out: Dict[str, Dict] = {}

        tgt_id = pack['target_id']
        target = next((t for t in targets if t['id'] == tgt_id and t.get('active', True)), None)
        if not target:
            for did in pack['drone_ids']:
                d = next((x for x in interceptors if x['id'] == did and x.get('active', True)), None)
                if d and d['id'] not in self.rtb_drones:
                    out[did] = self._decision(0,0,0, tgt_id, 'idle', 'target_destroyed')
            return out

        # Get active pack members not RTB
        pack_drones = [x for x in interceptors 
                      if x['id'] in pack['drone_ids'] 
                      and x.get('active', True) 
                      and x['id'] not in self.rtb_drones]
        
        if not pack_drones:
            return out

        # Check formation draft opportunities
        for drone in pack_drones:
            self._check_formation_draft(drone, pack, pack_drones)

        # Time since chase started
        chase_time = self.mission_time - pack.get('chase_start_time', 0.0)
        current_state = pack.get('pack_state', 'chasing')
        
        # STATE MACHINE with energy considerations
        if current_state == 'chasing':
            if chase_time >= self.chase_duration:
                pack['pack_state'] = 'following'
            
            for d in pack_drones:
                # Energy-efficient chase
                fuel_data = self.drone_fuel.get(d['id'])
                if fuel_data and fuel_data['current'] < self.fuel_conservative_threshold:
                    out[d['id']] = self._efficient_chase(d, target)
                else:
                    out[d['id']] = self._chase_target(d, target)

        elif current_state == 'following':
            avg_dist = sum(self._dist(d, target) for d in pack_drones) / len(pack_drones)
            
            if avg_dist <= self.follow_distance:
                pack['pack_state'] = 'forming'
            
            for d in pack_drones:
                out[d['id']] = self._follow_target_efficient(d, target)

        elif current_state == 'forming':
            ring_r = self.pack_formation_distance
            self._compute_ring(pack, target, ring_r)
            
            close_drones = 0
            for d in pack_drones:
                fp = pack['formation_positions'].get(d['id'])
                if fp:
                    slot_dist = math.sqrt((fp['x'] - d['x'])**2 + (fp['y'] - d['y'])**2)
                    if slot_dist <= self.ready_epsilon:
                        close_drones += 1
            
            formation_ready = (close_drones >= min(3, len(pack_drones)))
            
            if formation_ready:
                # Check if pack has enough fuel to engage
                avg_fuel = sum(self.drone_fuel[d['id']]['current'] for d in pack_drones) / len(pack_drones)
                if avg_fuel > self.min_fuel_for_strike:
                    pack['pack_state'] = 'engaging'
                    pack['engage_unlocked'] = True
                else:
                    # Not enough fuel - maintain formation
                    pack['pack_state'] = 'conserving'
            
            for d in pack_drones:
                fp = pack['formation_positions'].get(d['id'])
                out[d['id']] = self._move_to_formation(d, target, fp)

        elif current_state == 'engaging':
            ring_r = self.pack_formation_distance
            self._compute_ring(pack, target, ring_r)
            
            # Select killer based on fuel availability
            if pack.get('killer_drone') is None:
                best_candidate = None
                best_score = -1
                
                for d in pack_drones:
                    fuel_data = self.drone_fuel.get(d['id'])
                    if fuel_data and fuel_data['current'] > self.min_fuel_for_strike:
                        # Score based on fuel and distance
                        dist = self._dist(d, target)
                        fuel_score = fuel_data['current'] / 100.0
                        dist_score = max(0, 1.0 - dist / 1000.0)
                        total_score = fuel_score * 0.4 + dist_score * 0.6
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_candidate = d
                
                if best_candidate:
                    pack['killer_drone'] = best_candidate['id']
                    pack['green_drones'] = [d['id'] for d in pack_drones if d['id'] != pack['killer_drone']]
                    print(f"⚡ KILLER SELECTED: {pack['killer_drone']} with fuel={self.drone_fuel[pack['killer_drone']]['current']:.1f}%")
            
            for d in pack_drones:
                if d['id'] == pack.get('killer_drone'):
                    # Check abort conditions
                    fuel_data = self.drone_fuel.get(d['id'])
                    if fuel_data and fuel_data['current'] < self.abort_pursuit_fuel:
                        # Abort pursuit - low fuel
                        pack['killer_drone'] = None
                        out[d['id']] = self._conservative_orbit(d, target)
                    else:
                        out[d['id']] = self._energy_efficient_pursuit(d, target)
                else:
                    fp = pack['formation_positions'].get(d['id'])
                    out[d['id']] = self._track_parallel_efficient(d, target, fp)

        elif current_state == 'conserving':
            # Low fuel state - maintain defensive formation
            ring_r = self.pack_formation_distance * 1.5
            self._compute_ring(pack, target, ring_r)
            
            for d in pack_drones:
                fp = pack['formation_positions'].get(d['id'])
                out[d['id']] = self._conservative_track(d, target, fp)

        return out

    def _efficient_chase(self, drone: Dict, target: Dict) -> Dict:
        """Energy-efficient chase behavior"""
        dx = target['x'] - drone['x']
        dy = target['y'] - drone['y']
        dz = target.get('z', 0.0) - drone.get('z', 0.0)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
        
        # Predict with shorter horizon for efficiency
        tvx, tvy = target.get('vx', 0), target.get('vy', 0)
        pred_time = min(2.0, dist / self.optimal_cruise_speed)
        
        pred_x = target['x'] + tvx * pred_time
        pred_y = target['y'] + tvy * pred_time
        
        pdx = pred_x - drone['x']
        pdy = pred_y - drone['y']
        pdist = math.sqrt(pdx*pdx + pdy*pdy) or 1.0
        
        ux, uy, uz = pdx/pdist, pdy/pdist, dz/dist
        
        # Moderate acceleration for efficiency
        gain = self.green_max_acceleration * 0.6
        ax = gain * ux
        ay = gain * uy
        az = gain * uz
        
        return self._decision(ax, ay, az, target['id'], 'chasing', 'efficient_chase')

    def _follow_target_efficient(self, drone: Dict, target: Dict) -> Dict:
        """Efficient following behavior"""
        dx = target['x'] - drone['x']
        dy = target['y'] - drone['y']
        dz = target.get('z', 0.0) - drone.get('z', 0.0)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
        
        error = dist - self.follow_distance
        
        if abs(error) < 100.0:
            # Match target velocity efficiently
            tvx, tvy, tvz = target.get('vx', 0.0), target.get('vy', 0.0), target.get('vz', 0.0)
            dvx, dvy, dvz = drone.get('vx', 0.0), drone.get('vy', 0.0), drone.get('vz', 0.0)
            
            # Gentle velocity matching
            gain = 1.5
            ax = gain * (tvx - dvx)
            ay = gain * (tvy - dvy)
            az = gain * (tvz - dvz)
        else:
            # Adjust distance with minimal acceleration
            ux, uy, uz = dx/dist, dy/dist, dz/dist
            gain = self.green_max_acceleration * 0.4 * (1.0 if error > 0 else -0.3)
            ax = gain * ux
            ay = gain * uy
            az = gain * uz
        
        return self._decision(ax, ay, az, target['id'], 'following', 'efficient_follow')

    def _track_parallel_efficient(self, drone: Dict, target: Dict, fp: Optional[Dict]) -> Dict:
        """Energy-efficient parallel tracking"""
        if not fp:
            return self._decision(0,0,0, target['id'], 'green', 'no_formation_slot')

        # Maintain formation with minimal energy
        dx = fp['x'] - drone['x']
        dy = fp['y'] - drone['y']
        dz = fp.get('z', target.get('z', 0.0)) - drone.get('z', 0.0)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        if dist > 20.0:
            # Gentle correction
            ux, uy, uz = dx/dist, dy/dist, dz/dist
            pos_ax = self.green_max_acceleration * 0.3 * ux
            pos_ay = self.green_max_acceleration * 0.3 * uy
            pos_az = self.green_max_acceleration * 0.3 * uz
        else:
            pos_ax = pos_ay = pos_az = 0.0

        # Efficient velocity matching
        vdx, vdy, vdz = self._green_desired_velocity(target, fp)
        
        # Limit desired speed for efficiency
        desired_speed = math.sqrt(vdx*vdx + vdy*vdy + vdz*vdz)
        if desired_speed > self.optimal_cruise_speed:
            scale = self.optimal_cruise_speed / desired_speed
            vdx *= scale
            vdy *= scale
            vdz *= scale

        dvx, dvy, dvz = drone.get('vx',0.0), drone.get('vy',0.0), drone.get('vz',0.0)
        vel_gain = 1.5
        vel_ax = vel_gain * (vdx - dvx)
        vel_ay = vel_gain * (vdy - dvy)
        vel_az = vel_gain * (vdz - dvz)

        ax = pos_ax + vel_ax
        ay = pos_ay + vel_ay
        az = pos_az + vel_az
        
        return self._decision(ax, ay, az, target['id'], 'green', 'efficient_tracking')

    def _conservative_track(self, drone: Dict, target: Dict, fp: Optional[Dict]) -> Dict:
        """Ultra-conservative tracking for low fuel"""
        if not fp:
            # Just match target velocity
            tvx, tvy = target.get('vx', 0), target.get('vy', 0)
            dvx, dvy = drone.get('vx', 0), drone.get('vy', 0)
            ax = 1.0 * (tvx - dvx)
            ay = 1.0 * (tvy - dvy)
            az = 0
            return self._decision(ax, ay, az, target['id'], 'conserving', 'velocity_match')
        
        # Very gentle formation keeping
        dx = fp['x'] - drone['x']
        dy = fp['y'] - drone['y']
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist > 50.0:
            ux, uy = dx/dist, dy/dist
            ax = self.green_max_acceleration * 0.2 * ux
            ay = self.green_max_acceleration * 0.2 * uy
            az = 0
        else:
            ax = ay = az = 0
            
        return self._decision(ax, ay, az, target['id'], 'conserving', 'minimal_energy')

    def _standard_killer_pursuit(self, drone: Dict, target: Dict) -> Dict:
        """Standard killer pursuit when fuel is adequate"""
        d = self._dist(drone, target)
        
        if d <= self.strike_distance:
            target['active'] = False
            drone['active'] = False
            fuel_data = self.drone_fuel.get(drone['id'])
            fuel_used = fuel_data['initial'] - fuel_data['current'] if fuel_data else 0
            print(f"💥💥💥 ENERGY STRIKE: {drone['id']} destroyed {target['id']} at {d:.1f}m " +
                  f"(fuel used: {fuel_used:.1f}%) 💥💥💥")
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
        
        # Standard long-range pursuit
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

    def _print_energy_debug(self, targets: List[Dict], interceptors: List[Dict]):
        """Debug output for energy status"""
        print("\n" + "="*80)
        print(f"⚡ ENERGY STATUS: t={self.mission_time:.2f}s")
        
        # Fleet fuel summary
        total_fuel = 0
        critical_count = 0
        bingo_count = 0
        rtb_count = len(self.rtb_drones)
        
        for drone in interceptors:
            if drone.get('active', True):
                fuel_data = self.drone_fuel.get(drone['id'])
                if fuel_data:
                    fuel_percent = fuel_data['current']
                    total_fuel += fuel_percent
                    
                    if fuel_percent < self.fuel_critical_threshold:
                        critical_count += 1
                    elif fuel_percent < self.fuel_bingo_threshold:
                        bingo_count += 1
        
        active_count = len([d for d in interceptors if d.get('active', True)])
        avg_fuel = total_fuel / active_count if active_count > 0 else 0
        
        print(f"📊 FLEET FUEL: avg={avg_fuel:.1f}% critical={critical_count} bingo={bingo_count} rtb={rtb_count}")
        
        # Pack fuel status
        for pack_id, pack in self.packs.items():
            pack_fuel = []
            for did in pack['drone_ids']:
                drone = next((d for d in interceptors if d['id'] == did), None)
                if drone and drone.get('active', True) and did not in self.rtb_drones:
                    fuel_data = self.drone_fuel.get(did)
                    if fuel_data:
                        pack_fuel.append(f"{did}:{fuel_data['current']:.0f}%")
            
            if pack_fuel:
                print(f"   {pack_id}: [{', '.join(pack_fuel)}] state={pack.get('pack_state', 'unknown')}")
        
        # Efficiency leaders
        efficiency_scores = []
        for did, fuel_data in self.drone_fuel.items():
            if fuel_data['efficiency_score'] > 0:
                efficiency_scores.append((did, fuel_data['efficiency_score']))
        
        if efficiency_scores:
            efficiency_scores.sort(key=lambda x: x[1], reverse=True)
            print(f"🏆 TOP EFFICIENCY: {efficiency_scores[0][0]} ({efficiency_scores[0][1]:.1f} m/fuel)")
        
        # Headwind zones
        if self.headwind_zones:
            print(f"💨 HEADWIND ZONES: {len(self.headwind_zones)} active")
        
        print("="*80)

    def _create_packs(self, targets: List[Dict], interceptors: List[Dict]):
        """Create packs with fuel considerations"""
        act_targets = [t for t in targets if t.get('active', True)]
        act_interceptors = [i for i in interceptors if i.get('active', True)]
        
        print("\n" + "⚡" * 20)
        print(f"⚡ CREATING ENERGY-AWARE PACKS: {len(act_targets)} targets, {len(act_interceptors)} drones")
        
        # Sort interceptors by fuel capacity
        interceptor_fuel = []
        for drone in act_interceptors:
            fuel_data = self.drone_fuel.get(drone['id'], {'initial': 100})
            interceptor_fuel.append((drone, fuel_data['initial']))
        
        interceptor_fuel.sort(key=lambda x: x[1], reverse=True)
        
        # Assign balanced packs
        idx = 0
        for tgt in act_targets:
            pack_id = f"pack_{tgt['id']}"
            drones = []
            
            # Try to mix fuel capacities
            for _ in range(4):
                if idx < len(interceptor_fuel):
                    drone, fuel = interceptor_fuel[idx]
                    drones.append(drone['id'])
                    self.drone_pack_map[drone['id']] = pack_id
                    idx += 1
            
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
                
                # Print pack fuel summary
                pack_fuel = []
                for did in drones:
                    fuel_data = self.drone_fuel.get(did)
                    if fuel_data:
                        pack_fuel.append(f"{fuel_data['initial']:.0f}%")
                
                print(f"🎯 PACK [{pack_id}] - Fuel capacities: [{', '.join(pack_fuel)}]")
                
        print("⚡" * 20 + "\n")

    # Include base methods
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