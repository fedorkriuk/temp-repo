import math
import random
from typing import List, Dict, Optional, Tuple
import numpy as np

class AI_System_Evasion:
    """Enhanced pack logic with evasion pattern recognition and adaptive pursuit"""

    def __init__(self, scenario_config: Dict = None):
        # Base pack parameters
        self.pack_formation_distance = 300.0
        self.pack_activation_distance = 850.0
        self.strike_distance = 35.0
        
        # Evasion tracking parameters
        self.prediction_horizon = 4.0
        self.evasion_prediction_samples = 5
        self.evasion_pattern_recognition_time = 3.0
        self.pattern_lock_time = 2.5
        self.prediction_confidence_threshold = 0.7
        self.adaptive_pursuit_enable = True
        
        # Evasion pattern parameters
        self.zigzag_amplitude = 200.0
        self.zigzag_period = 2.0
        self.spiral_radius_growth = 50.0
        self.spiral_angular_velocity = 1.5
        self.random_walk_variance = 100.0
        self.barrel_roll_radius = 150.0
        self.barrel_roll_frequency = 0.5
        
        self.green_max_velocity = 260.0
        self.green_max_acceleration = 90.0
        
        self.killer_max_velocity = 300.0
        self.killer_max_acceleration = 180.0
        self.killer_deceleration_multiplier = 4.0
        
        self.strike_acceleration_multiplier = 5.5
        
        # Mission parameters
        self.max_engagement_range = 12000.0
        self.cooperation_radius = 2500.0
        self.communication_range = 18000.0
        
        # Green orbit parameters
        self.ring_orbit_speed = 40.0
        self.ring_orbit_direction = 1
        self.green_speed_margin = 30.0
        
        # State tracking
        self.chase_duration = 8.0
        self.follow_distance = 600.0
        self.ready_epsilon = 150.0
        
        # Runtime state
        self.packs: Dict[str, Dict] = {}
        self.drone_pack_map: Dict[str, str] = {}
        self.packs_initialized = False
        
        # Evasion pattern tracking
        self.target_patterns: Dict[str, Dict] = {}
        self.pattern_history: Dict[str, List] = {}
        
        self.frame_count = 0
        self.mission_time = 0.0
        self.dt = 1.0 / 30.0
        
        if scenario_config:
            self._load_all_parameters(scenario_config)
            
        print("=" * 60)
        print("🎯 ADVERSARIAL EVASION PATTERN SYSTEM INITIALIZED")
        print(f"   Pattern recognition time: {self.evasion_pattern_recognition_time}s")
        print(f"   Prediction samples: {self.evasion_prediction_samples}")
        print(f"   Adaptive pursuit: {'ENABLED' if self.adaptive_pursuit_enable else 'DISABLED'}")
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
            self.prediction_horizon = float(ai.get('prediction_horizon', self.prediction_horizon))
            self.evasion_prediction_samples = int(ai.get('evasion_prediction_samples', self.evasion_prediction_samples))
            self.evasion_pattern_recognition_time = float(ai.get('evasion_pattern_recognition_time', self.evasion_pattern_recognition_time))
            
            # Load evasion pattern parameters
            self.zigzag_amplitude = float(ai.get('zigzag_amplitude', self.zigzag_amplitude))
            self.zigzag_period = float(ai.get('zigzag_period', self.zigzag_period))
            self.spiral_radius_growth = float(ai.get('spiral_radius_growth', self.spiral_radius_growth))
            self.spiral_angular_velocity = float(ai.get('spiral_angular_velocity', self.spiral_angular_velocity))
            self.random_walk_variance = float(ai.get('random_walk_variance', self.random_walk_variance))
            self.barrel_roll_radius = float(ai.get('barrel_roll_radius', self.barrel_roll_radius))
            self.barrel_roll_frequency = float(ai.get('barrel_roll_frequency', self.barrel_roll_frequency))
            
            mission = scenario_config.get('mission_parameters', {})
            self.pattern_lock_time = float(mission.get('pattern_lock_time', self.pattern_lock_time))
            self.prediction_confidence_threshold = float(mission.get('prediction_confidence_threshold', self.prediction_confidence_threshold))
            self.adaptive_pursuit_enable = bool(mission.get('adaptive_pursuit_enable', self.adaptive_pursuit_enable))
            
        except Exception as e:
            print(f"⚠️ Parameter loading error: {e} - using defaults")

    def _apply_evasion_pattern(self, target: Dict, dt: float):
        """Apply evasion patterns to targets based on their type"""
        target_type = target.get('type', '')
        tid = target['id']
        
        # Initialize pattern tracking
        if tid not in self.target_patterns:
            self.target_patterns[tid] = {
                'type': 'unknown',
                'start_time': self.mission_time,
                'phase': 0.0,
                'confidence': 0.0
            }
            self.pattern_history[tid] = []
        
        # Store current position for pattern analysis
        self.pattern_history[tid].append({
            'time': self.mission_time,
            'x': target['x'],
            'y': target['y'],
            'z': target.get('z', 0),
            'vx': target.get('vx', 0),
            'vy': target.get('vy', 0),
            'vz': target.get('vz', 0)
        })
        
        # Keep only recent history
        if len(self.pattern_history[tid]) > 90:  # 3 seconds at 30fps
            self.pattern_history[tid] = self.pattern_history[tid][-90:]
        
        # Apply evasion based on target type
        if 'zigzag' in target_type:
            self._apply_zigzag_evasion(target, dt)
        elif 'spiral' in target_type:
            self._apply_spiral_evasion(target, dt)
        elif 'random' in target_type:
            self._apply_random_walk_evasion(target, dt)
        elif 'barrel' in target_type:
            self._apply_barrel_roll_evasion(target, dt)

    def _apply_zigzag_evasion(self, target: Dict, dt: float):
        """Zigzag evasion pattern"""
        tid = target['id']
        pattern = self.target_patterns[tid]
        pattern['phase'] += dt
        
        # Calculate perpendicular direction
        vx, vy = target.get('vx', 0), target.get('vy', 0)
        speed = math.sqrt(vx*vx + vy*vy)
        if speed > 0:
            # Perpendicular vector
            perp_x, perp_y = -vy/speed, vx/speed
            
            # Zigzag offset
            offset = self.zigzag_amplitude * math.sin(2 * math.pi * pattern['phase'] / self.zigzag_period)
            
            # Apply lateral acceleration
            lateral_acc = 50.0 * math.cos(2 * math.pi * pattern['phase'] / self.zigzag_period)
            target['ax'] = target.get('ax', 0) + lateral_acc * perp_x
            target['ay'] = target.get('ay', 0) + lateral_acc * perp_y

    def _apply_spiral_evasion(self, target: Dict, dt: float):
        """Spiral evasion pattern"""
        tid = target['id']
        pattern = self.target_patterns[tid]
        pattern['phase'] += dt
        
        # Calculate spiral motion
        radius = self.spiral_radius_growth * pattern['phase']
        angle = self.spiral_angular_velocity * pattern['phase']
        
        # Apply spiral acceleration
        spiral_ax = -radius * self.spiral_angular_velocity**2 * math.cos(angle)
        spiral_ay = -radius * self.spiral_angular_velocity**2 * math.sin(angle)
        
        target['ax'] = target.get('ax', 0) + spiral_ax * 0.3
        target['ay'] = target.get('ay', 0) + spiral_ay * 0.3

    def _apply_random_walk_evasion(self, target: Dict, dt: float):
        """Random walk evasion pattern"""
        tid = target['id']
        pattern = self.target_patterns[tid]
        
        # Apply random acceleration changes every 0.5 seconds
        if int(pattern['phase'] * 2) != int((pattern['phase'] + dt) * 2):
            # New random direction
            rand_angle = random.uniform(0, 2 * math.pi)
            rand_mag = random.uniform(0, self.random_walk_variance)
            
            target['ax'] = rand_mag * math.cos(rand_angle)
            target['ay'] = rand_mag * math.sin(rand_angle)
            target['az'] = random.uniform(-20, 20)
        
        pattern['phase'] += dt

    def _apply_barrel_roll_evasion(self, target: Dict, dt: float):
        """Barrel roll evasion pattern"""
        tid = target['id']
        pattern = self.target_patterns[tid]
        pattern['phase'] += dt
        
        # Calculate barrel roll motion
        angle = 2 * math.pi * self.barrel_roll_frequency * pattern['phase']
        
        # Apply circular motion in perpendicular plane
        vx, vy = target.get('vx', 0), target.get('vy', 0)
        speed = math.sqrt(vx*vx + vy*vy)
        if speed > 0:
            # Up vector (perpendicular to velocity)
            up_z = 1.0
            
            # Apply barrel roll acceleration
            target['az'] = self.barrel_roll_radius * math.sin(angle) * 10
            
            # Lateral component
            perp_x, perp_y = -vy/speed, vx/speed
            lateral_force = self.barrel_roll_radius * math.cos(angle) * 10
            target['ax'] = target.get('ax', 0) + lateral_force * perp_x
            target['ay'] = target.get('ay', 0) + lateral_force * perp_y

    def _predict_evasive_target(self, target: Dict, prediction_time: float) -> Dict:
        """Predict future position of evasive target"""
        tid = target['id']
        
        # If we don't have enough pattern history, use simple prediction
        if tid not in self.pattern_history or len(self.pattern_history[tid]) < 30:
            return {
                'x': target['x'] + target.get('vx', 0) * prediction_time,
                'y': target['y'] + target.get('vy', 0) * prediction_time,
                'z': target.get('z', 0) + target.get('vz', 0) * prediction_time,
                'confidence': 0.3
            }
        
        # Analyze pattern for better prediction
        history = self.pattern_history[tid]
        pattern_type = self._detect_evasion_pattern(history)
        
        predictions = []
        for sample in range(self.evasion_prediction_samples):
            # Monte Carlo sampling for prediction
            sample_time = prediction_time * (0.5 + sample / self.evasion_prediction_samples)
            
            if pattern_type == 'zigzag':
                pred = self._predict_zigzag(target, history, sample_time)
            elif pattern_type == 'spiral':
                pred = self._predict_spiral(target, history, sample_time)
            elif pattern_type == 'random':
                pred = self._predict_random(target, history, sample_time)
            elif pattern_type == 'barrel':
                pred = self._predict_barrel_roll(target, history, sample_time)
            else:
                pred = self._predict_linear(target, sample_time)
            
            predictions.append(pred)
        
        # Average predictions
        avg_x = sum(p['x'] for p in predictions) / len(predictions)
        avg_y = sum(p['y'] for p in predictions) / len(predictions)
        avg_z = sum(p['z'] for p in predictions) / len(predictions)
        
        # Calculate confidence based on prediction variance
        var_x = sum((p['x'] - avg_x)**2 for p in predictions) / len(predictions)
        var_y = sum((p['y'] - avg_y)**2 for p in predictions) / len(predictions)
        total_var = math.sqrt(var_x + var_y)
        confidence = max(0.3, min(0.9, 1.0 - total_var / 1000))
        
        return {
            'x': avg_x,
            'y': avg_y,
            'z': avg_z,
            'confidence': confidence,
            'pattern': pattern_type
        }

    def _detect_evasion_pattern(self, history: List[Dict]) -> str:
        """Detect the type of evasion pattern from movement history"""
        if len(history) < 30:
            return 'unknown'
        
        # Extract position arrays
        times = [h['time'] for h in history]
        xs = [h['x'] for h in history]
        ys = [h['y'] for h in history]
        
        # Calculate lateral accelerations
        lateral_accs = []
        for i in range(2, len(history)):
            dt = times[i] - times[i-1]
            if dt > 0:
                ax = (xs[i] - 2*xs[i-1] + xs[i-2]) / (dt*dt)
                ay = (ys[i] - 2*ys[i-1] + ys[i-2]) / (dt*dt)
                lateral_accs.append(math.sqrt(ax*ax + ay*ay))
        
        # Pattern detection heuristics
        if len(lateral_accs) > 10:
            # Check for periodic patterns (zigzag/barrel)
            acc_fft = np.fft.fft(lateral_accs)
            freqs = np.fft.fftfreq(len(lateral_accs))
            dominant_freq_idx = np.argmax(np.abs(acc_fft[1:len(acc_fft)//2])) + 1
            
            if np.abs(acc_fft[dominant_freq_idx]) > len(lateral_accs) * 10:
                # Strong periodic component
                period = 1.0 / freqs[dominant_freq_idx] if freqs[dominant_freq_idx] != 0 else 0
                if 1.5 < period < 3.0:
                    return 'zigzag'
                elif period > 3.0:
                    return 'barrel'
            
            # Check for spiral (increasing radius)
            radii = [math.sqrt((xs[i] - xs[0])**2 + (ys[i] - ys[0])**2) for i in range(len(xs))]
            if all(radii[i] <= radii[i+1] * 1.1 for i in range(len(radii)-1)):
                return 'spiral'
            
            # Check for random walk (high variance, no pattern)
            acc_variance = np.var(lateral_accs)
            if acc_variance > 1000:
                return 'random'
        
        return 'unknown'

    def _predict_zigzag(self, target: Dict, history: List[Dict], pred_time: float) -> Dict:
        """Predict zigzag pattern position"""
        # Estimate zigzag phase and amplitude from history
        current_phase = self.target_patterns[target['id']]['phase']
        future_phase = current_phase + pred_time
        
        # Base trajectory
        base_x = target['x'] + target.get('vx', 0) * pred_time
        base_y = target['y'] + target.get('vy', 0) * pred_time
        
        # Add zigzag offset
        vx, vy = target.get('vx', 0), target.get('vy', 0)
        speed = math.sqrt(vx*vx + vy*vy)
        if speed > 0:
            perp_x, perp_y = -vy/speed, vx/speed
            offset = self.zigzag_amplitude * math.sin(2 * math.pi * future_phase / self.zigzag_period)
            base_x += offset * perp_x
            base_y += offset * perp_y
        
        return {
            'x': base_x,
            'y': base_y,
            'z': target.get('z', 0) + target.get('vz', 0) * pred_time
        }

    def _predict_spiral(self, target: Dict, history: List[Dict], pred_time: float) -> Dict:
        """Predict spiral pattern position"""
        current_phase = self.target_patterns[target['id']]['phase']
        future_phase = current_phase + pred_time
        
        # Calculate spiral position
        radius = self.spiral_radius_growth * future_phase
        angle = self.spiral_angular_velocity * future_phase
        
        # Get spiral center (approximate from history)
        center_x = sum(h['x'] for h in history[:10]) / 10
        center_y = sum(h['y'] for h in history[:10]) / 10
        
        return {
            'x': center_x + radius * math.cos(angle),
            'y': center_y + radius * math.sin(angle),
            'z': target.get('z', 0) + target.get('vz', 0) * pred_time
        }

    def _predict_random(self, target: Dict, history: List[Dict], pred_time: float) -> Dict:
        """Predict random walk pattern - use Monte Carlo"""
        # For random walk, add uncertainty
        base_x = target['x'] + target.get('vx', 0) * pred_time
        base_y = target['y'] + target.get('vy', 0) * pred_time
        
        # Add random offset based on pattern variance
        rand_offset = self.random_walk_variance * math.sqrt(pred_time)
        rand_angle = random.uniform(0, 2 * math.pi)
        
        return {
            'x': base_x + rand_offset * math.cos(rand_angle),
            'y': base_y + rand_offset * math.sin(rand_angle),
            'z': target.get('z', 0) + target.get('vz', 0) * pred_time
        }

    def _predict_barrel_roll(self, target: Dict, history: List[Dict], pred_time: float) -> Dict:
        """Predict barrel roll pattern position"""
        current_phase = self.target_patterns[target['id']]['phase']
        future_phase = current_phase + pred_time
        
        # Base trajectory
        base_x = target['x'] + target.get('vx', 0) * pred_time
        base_y = target['y'] + target.get('vy', 0) * pred_time
        base_z = target.get('z', 0) + target.get('vz', 0) * pred_time
        
        # Add barrel roll offset
        angle = 2 * math.pi * self.barrel_roll_frequency * future_phase
        z_offset = self.barrel_roll_radius * math.sin(angle)
        
        return {
            'x': base_x,
            'y': base_y,
            'z': base_z + z_offset
        }

    def _predict_linear(self, target: Dict, pred_time: float) -> Dict:
        """Simple linear prediction"""
        return {
            'x': target['x'] + target.get('vx', 0) * pred_time,
            'y': target['y'] + target.get('vy', 0) * pred_time,
            'z': target.get('z', 0) + target.get('vz', 0) * pred_time
        }

    def update(self, targets: List[Dict], interceptors: List[Dict], dt: float, wind: Dict, scenario_config: Dict = None) -> Dict[str, Dict]:
        if scenario_config:
            self._load_all_parameters(scenario_config)
            
        eff_dt = dt if (dt is not None and dt > 0.0) else (self.dt if self.frame_count == 0 else 0.0)
        
        self.frame_count += 1
        self.mission_time += eff_dt
        self.dt = eff_dt
        
        # Apply evasion patterns to targets
        for target in targets:
            if target.get('active', True):
                self._apply_evasion_pattern(target, eff_dt)
        
        # Physics step
        self._update_physics(targets, interceptors, eff_dt, wind)
        
        # Initialize packs once
        if not self.packs_initialized:
            self._create_packs(targets, interceptors)
            self.packs_initialized = True
        
        # Debug output
        if self.frame_count % 30 == 0:
            self._print_evasion_debug(targets, interceptors)
            
        # Generate decisions for all packs
        decisions: Dict[str, Dict] = {}
        for pack_id, pack in self.packs.items():
            pack_decisions = self._update_pack(pack, targets, interceptors)
            decisions.update(pack_decisions)
            
        return decisions

    def _print_evasion_debug(self, targets: List[Dict], interceptors: List[Dict]):
        """Debug output for evasion scenarios"""
        print("\n" + "="*80)
        print(f"🕒 EVASION SCENARIO STATUS: t={self.mission_time:.2f}s")
        
        # Target pattern analysis
        for t in targets:
            if t.get('active', True):
                tid = t['id']
                pattern = 'unknown'
                confidence = 0.0
                
                if tid in self.pattern_history and len(self.pattern_history[tid]) > 20:
                    detected = self._detect_evasion_pattern(self.pattern_history[tid])
                    pattern = detected
                    if tid in self.target_patterns:
                        confidence = self.target_patterns[tid].get('confidence', 0.0)
                
                speed = math.sqrt(t.get('vx',0)**2 + t.get('vy',0)**2)
                print(f"🎯 {tid}: pattern={pattern} confidence={confidence:.2f} " +
                      f"pos=({t['x']:.0f},{t['y']:.0f},{t.get('z',0):.0f}) speed={speed:.1f}")
        
        # Pack effectiveness
        print("\n📊 PACK EFFECTIVENESS:")
        for pack_id, pack in self.packs.items():
            target = next((t for t in targets if t['id'] == pack['target_id']), None)
            if target and target.get('active', True):
                pack_drones = [d for d in interceptors if d['id'] in pack['drone_ids'] and d.get('active', True)]
                if pack_drones:
                    avg_dist = sum(self._dist(d, target) for d in pack_drones) / len(pack_drones)
                    print(f"   {pack_id}: avg_dist={avg_dist:.0f}m state={pack.get('pack_state', 'unknown')}")
        
        print("="*80)

    def _create_packs(self, targets: List[Dict], interceptors: List[Dict]):
        """Create initial pack assignments"""
        act_targets = [t for t in targets if t.get('active', True)]
        act_interceptors = [i for i in interceptors if i.get('active', True)]
        
        print("\n" + "🐺" * 20)
        print(f"🐺 CREATING EVASION PACKS: {len(act_targets)} targets, {len(act_interceptors)} drones")
        
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
                    'pack_state': 'chasing',
                    'formation_positions': {},
                    'chase_start_time': self.mission_time,
                    'first_ready_time': None,
                    'engage_unlocked': False,
                }
                print(f"🎯 PACK [{pack_id}] CREATED - TARGET TYPE: {tgt.get('type', 'unknown')}")
        print("🐺" * 20 + "\n")

    def _killer_pursuit_evasive(self, pack: Dict, drone: Dict, target: Dict) -> Dict:
        """Enhanced killer pursuit for evasive targets"""
        d = self._dist(drone, target)
        
        # STRIKE CHECK
        if d <= self.strike_distance:
            target['active'] = False
            drone['active'] = False
            print(f"💥💥💥 EVASION DEFEATED: {drone['id']} intercepted {target['id']} at {d:.1f}m 💥💥💥")
            return self._decision(0,0,0, target['id'], 'killer', 'STRIKE_SUCCESS')

        # Use advanced prediction for evasive targets
        if self.adaptive_pursuit_enable:
            # Predict future position with pattern recognition
            predict_time = min(self.prediction_horizon, d / max(self.killer_max_velocity, 50.0))
            prediction = self._predict_evasive_target(target, predict_time)
            
            # Adjust pursuit based on confidence
            if prediction['confidence'] > self.prediction_confidence_threshold:
                # High confidence - pursue predicted position
                pred_x = prediction['x']
                pred_y = prediction['y']
                pred_z = prediction['z']
            else:
                # Low confidence - use multiple prediction points
                predictions = []
                for t in [0.5, 1.0, 1.5, 2.0]:
                    predictions.append(self._predict_evasive_target(target, t))
                
                # Weighted average based on confidence
                total_conf = sum(p['confidence'] for p in predictions)
                if total_conf > 0:
                    pred_x = sum(p['x'] * p['confidence'] for p in predictions) / total_conf
                    pred_y = sum(p['y'] * p['confidence'] for p in predictions) / total_conf
                    pred_z = sum(p['z'] * p['confidence'] for p in predictions) / total_conf
                else:
                    # Fallback to current position
                    pred_x, pred_y, pred_z = target['x'], target['y'], target.get('z', 0)
        else:
            # Standard prediction
            tvx, tvy, tvz = target.get('vx', 0.0), target.get('vy', 0.0), target.get('vz', 0.0)
            predict_time = min(self.prediction_horizon, d / max(self.killer_max_velocity, 50.0))
            pred_x = target['x'] + tvx * predict_time
            pred_y = target['y'] + tvy * predict_time
            pred_z = target.get('z', 0.0) + tvz * predict_time

        # Calculate pursuit vector
        pdx = pred_x - drone['x']
        pdy = pred_y - drone['y']
        pdz = pred_z - drone.get('z', 0.0)
        pdist = math.sqrt(pdx*pdx + pdy*pdy + pdz*pdz) or 1.0

        # Aggressive pursuit with pattern compensation
        desired_speed = min(self.killer_max_velocity, max(100.0, d * 2.0))
        dirx, diry, dirz = pdx/pdist, pdy/pdist, pdz/pdist
        
        dvx_des = desired_speed * dirx
        dvy_des = desired_speed * diry
        dvz_des = desired_speed * dirz

        cvx, cvy, cvz = drone.get('vx',0.0), drone.get('vy',0.0), drone.get('vz',0.0)

        # Extra aggressive acceleration for evasive targets
        agile = self.killer_max_acceleration * self.strike_acceleration_multiplier
        ax = agile * (dvx_des - cvx) / max(1.0, desired_speed)
        ay = agile * (dvy_des - cvy) / max(1.0, desired_speed)
        az = agile * (dvz_des - cvz) / max(1.0, desired_speed)

        return self._decision(ax, ay, az, target['id'], 'killer', 'adaptive_pursuit')

    def _update_pack(self, pack: Dict, targets: List[Dict], interceptors: List[Dict]) -> Dict[str, Dict]:
        """Update pack with evasion-aware logic"""
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

        # STATE MACHINE with evasion adaptations
        current_state = pack.get('pack_state', 'chasing')
        
        if current_state == 'chasing':
            if chase_time >= self.chase_duration:
                pack['pack_state'] = 'following'
            
            for d in pack_drones:
                out[d['id']] = self._chase_target(d, target)

        elif current_state == 'following':
            avg_dist = sum(self._dist(d, target) for d in pack_drones) / len(pack_drones)
            
            if avg_dist <= self.follow_distance:
                pack['pack_state'] = 'forming'
            
            for d in pack_drones:
                out[d['id']] = self._follow_target(d, target)

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
            
            formation_ready = (close_drones >= 3)
            
            if formation_ready:
                pack['pack_state'] = 'engaging'
                pack['engage_unlocked'] = True
            
            for d in pack_drones:
                fp = pack['formation_positions'].get(d['id'])
                out[d['id']] = self._move_to_formation(d, target, fp)

        elif current_state == 'engaging':
            ring_r = self.pack_formation_distance
            self._compute_ring(pack, target, ring_r)
            
            if pack.get('killer_drone') is None:
                closest, closest_dist = self._closest(pack_drones, target)
                if closest:
                    pack['killer_drone'] = closest['id']
                    pack['green_drones'] = [d['id'] for d in pack_drones if d['id'] != pack['killer_drone']]
            
            for d in pack_drones:
                if d['id'] == pack.get('killer_drone'):
                    # Use enhanced evasive pursuit
                    out[d['id']] = self._killer_pursuit_evasive(pack, d, target)
                else:
                    fp = pack['formation_positions'].get(d['id'])
                    out[d['id']] = self._track_parallel(d, target, fp)

        return out

    # Include all other base methods from original AI_System
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