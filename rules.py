import math
from typing import List, Dict, Optional, Tuple

class AI_System:
    """Pack logic with chase-follow-surround-strike behavior"""

    def __init__(self, scenario_config: Dict = None):
        # MUCH stricter formation requirements
        self.pack_formation_distance = 400.0
        self.pack_activation_distance = 800.0
        self.strike_distance = 25.0

        self.green_max_velocity = 220.0
        self.green_max_acceleration = 70.0

        self.killer_max_velocity = 240.0
        self.killer_max_acceleration = 140.0
        self.killer_deceleration_multiplier = 3.0

        self.strike_acceleration_multiplier = 4.5

        # Optional mission/AI
        self.max_engagement_range = 1000.0
        self.prediction_horizon = 3.0
        self.cooperation_radius = 2000.0
        self.communication_range = 15000.0

        # Green orbit parameters
        self.ring_orbit_speed = 30.0
        self.ring_orbit_direction = 1
        self.green_speed_margin = 20.0

        # Force longer phases and MUCH MORE LENIENT requirements
        self.chase_duration = 8.0  # 8 seconds of chasing
        self.follow_distance = 600.0  # Must get within 600m to start forming
        self.ready_epsilon = 150.0  # Allow 150m tolerance for formation positions (was 50m)

        # Runtime state
        self.packs: Dict[str, Dict] = {}
        self.drone_pack_map: Dict[str, str] = {}
        self.packs_initialized = False

        self.frame_count = 0
        self.mission_time = 0.0
        self.dt = 1.0 / 30.0

        if scenario_config:
            self._load_all_parameters(scenario_config)

        print("=" * 60)
        print("🐺 CHASE-FOLLOW-SURROUND PACK SYSTEM INITIALIZED")
        print(f"   Chase duration: {self.chase_duration}s")
        print(f"   Follow distance: {self.follow_distance}m")
        print(f"   Formation distance: {self.pack_formation_distance}m")
        print(f"   Formation tolerance: {self.ready_epsilon}m")
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
            self.strike_distance = float(ai.get('strike_distance', self.strike_distance))

            # Ring orbit parameters
            self.ring_orbit_speed = float(ai.get('ring_orbit_speed', self.ring_orbit_speed))
            self.ring_orbit_direction = int(ai.get('ring_orbit_direction', self.ring_orbit_direction)) or 1
            self.green_speed_margin = float(ai.get('green_speed_margin', self.green_speed_margin))

            mission = scenario_config.get('mission_parameters', {})
            self.max_engagement_range = float(mission.get('max_engagement_range', self.max_engagement_range))
            if 'intercept_radius' in mission:
                self.strike_distance = float(mission.get('intercept_radius'))

            print("🎯 LOADED CONFIG:")
            print(f"   Formation distance={self.pack_formation_distance} m, activation={self.pack_activation_distance} m")
            print(f"   Chase duration={self.chase_duration}s, follow distance={self.follow_distance} m")
            print(f"   Formation tolerance={self.ready_epsilon} m, Strike distance={self.strike_distance} m")
            print(f"   Green max_vel={self.green_max_velocity} m/s, max_acc={self.green_max_acceleration} m/s²")
        except Exception as e:
            print(f"⚠️ Parameter loading error: {e} - using defaults")

    def update(self, targets: List[Dict], interceptors: List[Dict], dt: float, wind: Dict, scenario_config: Dict = None) -> Dict[str, Dict]:
        if scenario_config:
            self._load_all_parameters(scenario_config)

        eff_dt = dt if (dt is not None and dt > 0.0) else (self.dt if self.frame_count == 0 else 0.0)

        self.frame_count += 1
        self.mission_time += eff_dt
        self.dt = eff_dt

        # Physics step first with HEAVY DEBUG
        self._update_physics(targets, interceptors, eff_dt, wind)

        # Initialize packs once
        if not self.packs_initialized:
            self._create_packs(targets, interceptors)
            self.packs_initialized = True

        # HEAVY DEBUG every 10 frames (3 times per second)
        if self.frame_count % 10 == 0 and targets and interceptors:
            print("\n" + "="*80)
            print(f"🕒 MISSION TIME: {self.mission_time:.2f}s (Frame {self.frame_count})")
            
            t = targets[0]
            print(f"🎯 TARGET: pos=({t['x']:.0f},{t['y']:.0f},{t.get('z',0):.0f}) vel=({t.get('vx',0):.1f},{t.get('vy',0):.1f}) speed={math.sqrt(t.get('vx',0)**2 + t.get('vy',0)**2):.1f}")
            
            for pack_id, pack in self.packs.items():
                state = pack.get('pack_state', 'unknown')
                chase_time = self.mission_time - pack.get('chase_start_time', 0.0)
                killer = pack.get('killer_drone', 'NONE')
                
                print(f"📦 PACK {pack_id}: STATE={state} | CHASE_TIME={chase_time:.1f}s | KILLER={killer}")
                
                pack_drones = [x for x in interceptors if x['id'] in pack['drone_ids'] and x.get('active', True)]
                for i, drone in enumerate(pack_drones):
                    dist_to_target = self._dist(drone, t)
                    role = 'KILLER' if drone['id'] == pack.get('killer_drone') else 'GREEN'
                    speed = math.sqrt(drone.get('vx',0)**2 + drone.get('vy',0)**2)
                    print(f"   🚁 {drone['id']}: pos=({drone['x']:.0f},{drone['y']:.0f}) dist={dist_to_target:.0f}m speed={speed:.1f} role={role}")
            print("="*80)

        # Decisions pack by pack
        decisions: Dict[str, Dict] = {}
        for _, pack in self.packs.items():
            pack_decisions = self._update_pack(pack, targets, interceptors)
            decisions.update(pack_decisions)

        return decisions

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

    def _create_packs(self, targets: List[Dict], interceptors: List[Dict]):
        act_targets = [t for t in targets if t.get('active', True)]
        act_interceptors = [i for i in interceptors if i.get('active', True)]
        
        print("\n" + "🐺" * 20)
        print(f"🐺 CREATING PACKS: {len(act_targets)} targets, {len(act_interceptors)} drones")

        idx = 0
        for tgt in act_targets:
            pack_id = f"pack_{tgt['id']}"
            drones = []
            for _ in range(4):
                if idx < len(act_interceptors):
                    d = act_interceptors[idx]
                    drones.append(d['id'])
                    self.drone_pack_map[d['id']] = pack_id
                    idx += 1
            if drones:
                self.packs[pack_id] = {
                    'target_id': tgt['id'],
                    'drone_ids': drones,
                    'killer_drone': None,
                    'green_drones': drones.copy(),
                    'pack_state': 'chasing',  # Start with chasing phase
                    'formation_positions': {},
                    'chase_start_time': self.mission_time,  # Track chase phase
                    'first_ready_time': None,
                    'engage_unlocked': False,
                }
                print(f"🎯 PACK [{pack_id}] CREATED - STATE: CHASING - DRONES: {drones}")
        print("🐺" * 20 + "\n")

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
        
        # STATE TRANSITION LOGIC with HEAVY DEBUG
        if current_state == 'chasing':
            print(f"🏃 PACK {tgt_id} CHASING: chase_time={chase_time:.1f}s (need {self.chase_duration}s)")
            
            if chase_time >= self.chase_duration:
                pack['pack_state'] = 'following'
                print(f"🔄 PACK {tgt_id} TRANSITION: CHASING -> FOLLOWING (chase complete)")
            
            # All drones chase target directly
            for d in pack_drones:
                out[d['id']] = self._chase_target(d, target)

        elif current_state == 'following':
            avg_dist = sum(self._dist(d, target) for d in pack_drones) / len(pack_drones)
            print(f"🏃 PACK {tgt_id} FOLLOWING: avg_dist={avg_dist:.0f}m (need <{self.follow_distance}m)")
            
            if avg_dist <= self.follow_distance:
                pack['pack_state'] = 'forming'
                print(f"🔄 PACK {tgt_id} TRANSITION: FOLLOWING -> FORMING (close enough)")
            
            # All drones follow at distance
            for d in pack_drones:
                out[d['id']] = self._follow_target(d, target)

        elif current_state == 'forming':
            ring_r = self.pack_formation_distance
            self._compute_ring(pack, target, ring_r)
            
            # CHECK: At least 3 out of 4 drones must be close to their slots (MORE LENIENT)
            close_drones = 0
            slot_distances = []
            for d in pack_drones:
                fp = pack['formation_positions'].get(d['id'])
                if fp:
                    slot_dist = math.sqrt((fp['x'] - d['x'])**2 + (fp['y'] - d['y'])**2)
                    slot_distances.append(slot_dist)
                    if slot_dist <= self.ready_epsilon:  # 150m tolerance now
                        close_drones += 1
                else:
                    slot_distances.append(999)
            
            formation_ready = (close_drones >= 3)  # At least 3 out of 4 drones
            
            print(f"🔄 PACK {tgt_id} FORMING: close_drones={close_drones}/4, tolerance={self.ready_epsilon}m")
            print(f"   Slot distances: {[f'{d:.0f}' for d in slot_distances]}")
            
            if formation_ready:
                pack['pack_state'] = 'engaging'
                pack['engage_unlocked'] = True
                print(f"🔄 PACK {tgt_id} TRANSITION: FORMING -> ENGAGING ({close_drones} drones in position)")
            
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
                    print(f"🔫 PACK {tgt_id} KILLER ACTIVATED: {pack['killer_drone']} (dist={closest_dist:.0f}m)")
            
            print(f"⚔️ PACK {tgt_id} ENGAGING: killer={pack.get('killer_drone', 'NONE')}")
            
            # Generate decisions
            for d in pack_drones:
                if d['id'] == pack.get('killer_drone'):
                    out[d['id']] = self._killer_pursuit(pack, d, target)
                else:
                    fp = pack['formation_positions'].get(d['id'])
                    out[d['id']] = self._track_parallel(d, target, fp)

        return out

    def _chase_target(self, drone: Dict, target: Dict) -> Dict:
        """Phase 1: Simple chase - all drones go directly toward target"""
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
        """Phase 2: Follow at distance - maintain follow_distance"""
        dx = target['x'] - drone['x']
        dy = target['y'] - drone['y']
        dz = target.get('z', 0.0) - drone.get('z', 0.0)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
        
        # Try to maintain follow_distance
        error = dist - self.follow_distance
        
        if abs(error) < 100.0:
            # Close to desired distance - match target velocity
            tvx, tvy, tvz = target.get('vx', 0.0), target.get('vy', 0.0), target.get('vz', 0.0)
            dvx, dvy, dvz = drone.get('vx', 0.0), drone.get('vy', 0.0), drone.get('vz', 0.0)
            
            gain = 2.0
            ax = gain * (tvx - dvx)
            ay = gain * (tvy - dvy)
            az = gain * (tvz - dvz)
        else:
            # Too far or too close - adjust distance
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

        # Move towards formation position with aggressive acceleration
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
        
        # STRIKE CHECK - if close enough, strike!
        if d <= self.strike_distance:
            target['active'] = False
            drone['active'] = False
            print(f"💥💥💥 STRIKE SUCCESS: {drone['id']} destroyed {target['id']} at {d:.1f}m 💥💥💥")
            return self._decision(0,0,0, target['id'], 'killer', 'STRIKE_SUCCESS')

        # AGGRESSIVE PURSUIT - go directly for the target when close
        if d <= 100.0:  # Within 100m - go straight for current position
            dx = target['x'] - drone['x']
            dy = target['y'] - drone['y'] 
            dz = target.get('z', 0.0) - drone.get('z', 0.0)
            dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
            
            # Maximum aggression - ram the target
            ux, uy, uz = dx/dist, dy/dist, dz/dist
            max_accel = self.killer_max_acceleration * self.strike_acceleration_multiplier
            ax = max_accel * ux
            ay = max_accel * uy
            az = max_accel * uz
            
            print(f"🚀 KILLER {drone['id']} FINAL APPROACH: {d:.1f}m -> TARGET!")
            return self._decision(ax, ay, az, target['id'], 'killer', 'final_approach')
        
        # LONG RANGE PURSUIT - use prediction for distant targets
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

        desired_speed = min(self.killer_max_velocity, max(80.0, dist * 1.5))  # Increased minimum speed
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