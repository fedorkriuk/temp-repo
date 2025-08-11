import math
import random
from typing import List, Dict, Optional, Tuple, Set
import numpy as np

class AI_System_SwarmCombat:
    """Advanced pack logic for swarm vs swarm combat scenarios"""

    def __init__(self, scenario_config: Dict = None):
        # Base pack parameters
        self.pack_formation_distance = 400.0
        self.pack_activation_distance = 1200.0
        self.strike_distance = 40.0
        
        # Swarm combat parameters
        self.swarm_coordination_radius = 5000.0
        self.engagement_range = 3000.0
        self.disengage_threshold = 500.0
        self.pursuit_commitment = 0.7
        self.flanking_enabled = True
        
        # Team tactics
        self.blue_team_strategy = 'coordinated_assault'
        self.red_team_behavior = 'aggressive_defense'
        
        # Tactical maneuvers
        self.pincer_movement_enabled = True
        self.feint_attack_probability = 0.3
        self.coordinated_strike_bonus = 1.5
        
        # Team-specific physics (Blue team)
        self.team_specs = {
            'blue_assault': {
                'max_acceleration': 85.0,
                'max_velocity': 270.0,
                'role': 'assault',
                'color': 'blue'
            },
            'blue_defender': {
                'max_acceleration': 70.0,
                'max_velocity': 240.0,
                'role': 'defender',
                'color': 'blue'
            },
            'blue_striker': {
                'max_acceleration': 100.0,
                'max_velocity': 320.0,
                'role': 'striker',
                'color': 'blue'
            },
            'blue_support': {
                'max_acceleration': 75.0,
                'max_velocity': 250.0,
                'role': 'support',
                'color': 'blue'
            }
        }
        
        # Default physics
        self.green_max_velocity = 260.0
        self.green_max_acceleration = 80.0
        
        self.killer_max_velocity = 300.0
        self.killer_max_acceleration = 160.0
        self.killer_deceleration_multiplier = 4.0
        
        self.strike_acceleration_multiplier = 5.5
        
        # Mission parameters
        self.max_engagement_range = 15000.0
        self.cooperation_radius = 4000.0
        self.communication_range = 25000.0
        self.prediction_horizon = 4.5
        
        # Combat rules
        self.friendly_fire_enabled = False
        self.simultaneous_engagement_limit = 3
        self.retreat_allowed = True
        
        # Green orbit parameters
        self.ring_orbit_speed = 35.0
        self.ring_orbit_direction = 1
        self.green_speed_margin = 25.0
        
        # State tracking
        self.chase_duration = 6.0  # Faster engagement for combat
        self.follow_distance = 500.0
        self.ready_epsilon = 120.0
        
        # Runtime state
        self.packs: Dict[str, Dict] = {}
        self.drone_pack_map: Dict[str, str] = {}
        self.drone_teams: Dict[str, str] = {}
        self.packs_initialized = False
        
        # Swarm combat tracking
        self.swarm_state: Dict[str, Dict] = {
            'blue': {
                'strategy': 'coordinated_assault',
                'formation': 'distributed',
                'morale': 100.0,
                'kills': 0,
                'losses': 0,
                'center': {'x': 0, 'y': 0, 'z': 0},
                'target_priorities': {}
            },
            'red': {
                'strategy': 'aggressive_defense',
                'formation': 'tight',
                'morale': 100.0,
                'kills': 0,
                'losses': 0,
                'center': {'x': 0, 'y': 0, 'z': 0},
                'target_priorities': {}
            }
        }
        
        self.engagement_matrix: Dict[str, Set[str]] = {}
        self.combat_history: List[Dict] = []
        self.tactical_objectives: Dict[str, Dict] = {}
        
        self.frame_count = 0
        self.mission_time = 0.0
        self.dt = 1.0 / 30.0
        
        if scenario_config:
            self._load_all_parameters(scenario_config)
            
        print("=" * 60)
        print("⚔️ SWARM COMBAT SYSTEM INITIALIZED")
        print(f"   Blue strategy: {self.blue_team_strategy}")
        print(f"   Red behavior: {self.red_team_behavior}")
        print(f"   Engagement range: {self.engagement_range}m")
        print(f"   Tactical maneuvers: {'ENABLED' if self.pincer_movement_enabled else 'DISABLED'}")
        print("=" * 60)

    def _load_all_parameters(self, scenario_config: Dict):
        try:
            physics = scenario_config.get('physics', {})
            
            # Load team-specific physics
            for team_type in ['blue_assault', 'blue_defender', 'blue_striker', 'blue_support']:
                if f'{team_type}_acceleration' in physics:
                    if team_type not in self.team_specs:
                        self.team_specs[team_type] = {}
                    self.team_specs[team_type]['max_acceleration'] = float(physics[f'{team_type}_acceleration'])
                    self.team_specs[team_type]['max_velocity'] = float(physics[f'{team_type}_velocity'])
            
            self.green_max_acceleration = float(physics.get('green_max_acceleration', self.green_max_acceleration))
            self.green_max_velocity = float(physics.get('green_max_velocity', self.green_max_velocity))
            self.killer_max_acceleration = float(physics.get('killer_max_acceleration', self.killer_max_acceleration))
            self.killer_max_velocity = float(physics.get('killer_max_velocity', self.killer_max_velocity))
            
            ai = scenario_config.get('ai_parameters', {})
            self.swarm_coordination_radius = float(ai.get('swarm_coordination_radius', self.swarm_coordination_radius))
            self.engagement_range = float(ai.get('engagement_range', self.engagement_range))
            self.disengage_threshold = float(ai.get('disengage_threshold', self.disengage_threshold))
            self.pursuit_commitment = float(ai.get('pursuit_commitment', self.pursuit_commitment))
            
            self.blue_team_strategy = ai.get('blue_team_strategy', self.blue_team_strategy)
            self.red_team_behavior = ai.get('red_team_behavior', self.red_team_behavior)
            
            self.pincer_movement_enabled = bool(ai.get('pincer_movement_enabled', self.pincer_movement_enabled))
            self.feint_attack_probability = float(ai.get('feint_attack_probability', self.feint_attack_probability))
            self.coordinated_strike_bonus = float(ai.get('coordinated_strike_bonus', self.coordinated_strike_bonus))
            
            mission = scenario_config.get('mission_parameters', {})
            self.friendly_fire_enabled = bool(mission.get('friendly_fire_enabled', self.friendly_fire_enabled))
            self.simultaneous_engagement_limit = int(mission.get('simultaneous_engagement_limit', self.simultaneous_engagement_limit))
            self.retreat_allowed = bool(mission.get('retreat_allowed', self.retreat_allowed))
            
        except Exception as e:
            print(f"⚠️ Parameter loading error: {e} - using defaults")

    def _identify_drone_team(self, drone: Dict) -> str:
        """Identify which team a drone belongs to"""
        drone_type = drone.get('type', '')
        
        # Check if already classified
        if drone['id'] in self.drone_teams:
            return self.drone_teams[drone['id']]
        
        # Classify based on type
        if 'blue' in drone_type or drone_type.startswith('pack_'):
            team = 'blue'
        elif 'red' in drone_type or drone_type.startswith('enemy_'):
            team = 'red'
        else:
            # Default based on ID prefix
            team = 'blue' if not drone['id'].startswith('enemy_') else 'red'
        
        self.drone_teams[drone['id']] = team
        return team

    def _update_swarm_state(self, targets: List[Dict], interceptors: List[Dict]):
        """Update overall swarm state and tactics"""
        # Separate teams
        blue_drones = []
        red_drones = []
        
        # Classify all active drones
        all_drones = targets + interceptors
        for drone in all_drones:
            if not drone.get('active', True):
                continue
                
            team = self._identify_drone_team(drone)
            if team == 'blue':
                blue_drones.append(drone)
            else:
                red_drones.append(drone)
        
        # Update team centers
        for team, drones in [('blue', blue_drones), ('red', red_drones)]:
            if drones:
                center_x = sum(d['x'] for d in drones) / len(drones)
                center_y = sum(d['y'] for d in drones) / len(drones)
                center_z = sum(d.get('z', 0) for d in drones) / len(drones)
                
                self.swarm_state[team]['center'] = {
                    'x': center_x,
                    'y': center_y,
                    'z': center_z
                }
                
                # Update morale based on K/D ratio
                kills = self.swarm_state[team]['kills']
                losses = self.swarm_state[team]['losses']
                if losses > 0:
                    kd_ratio = kills / losses
                    self.swarm_state[team]['morale'] = min(100, 50 + kd_ratio * 25)
                elif kills > 0:
                    self.swarm_state[team]['morale'] = min(100, 75 + kills * 5)
        
        # Update tactical objectives
        self._update_tactical_objectives(blue_drones, red_drones)

    def _update_tactical_objectives(self, blue_drones: List[Dict], red_drones: List[Dict]):
        """Determine high-level tactical objectives"""
        if not blue_drones or not red_drones:
            return
        
        # Calculate force ratios
        blue_strength = len(blue_drones)
        red_strength = len(red_drones)
        force_ratio = blue_strength / max(red_strength, 1)
        
        # Blue team objectives
        if self.blue_team_strategy == 'coordinated_assault':
            if force_ratio > 1.2:
                # Numerical advantage - aggressive push
                self.tactical_objectives['blue'] = {
                    'type': 'all_out_assault',
                    'target_zone': self.swarm_state['red']['center'],
                    'formation': 'wedge'
                }
            elif force_ratio < 0.8:
                # Outnumbered - defensive retreat
                self.tactical_objectives['blue'] = {
                    'type': 'fighting_retreat',
                    'target_zone': {'x': 0, 'y': 0, 'z': 1000},
                    'formation': 'sphere'
                }
            else:
                # Even match - tactical engagement
                self.tactical_objectives['blue'] = {
                    'type': 'tactical_engagement',
                    'target_zone': self._calculate_engagement_zone(blue_drones, red_drones),
                    'formation': 'line'
                }
        
        # Implement flanking maneuvers
        if self.pincer_movement_enabled and force_ratio > 0.9:
            self._plan_pincer_movement(blue_drones, red_drones)

    def _calculate_engagement_zone(self, blue_drones: List[Dict], red_drones: List[Dict]) -> Dict:
        """Calculate optimal engagement zone between swarms"""
        blue_center = self.swarm_state['blue']['center']
        red_center = self.swarm_state['red']['center']
        
        # Midpoint with bias toward enemy
        engagement_x = blue_center['x'] * 0.3 + red_center['x'] * 0.7
        engagement_y = blue_center['y'] * 0.3 + red_center['y'] * 0.7
        engagement_z = (blue_center['z'] + red_center['z']) / 2
        
        return {'x': engagement_x, 'y': engagement_y, 'z': engagement_z}

    def _plan_pincer_movement(self, blue_drones: List[Dict], red_drones: List[Dict]):
        """Plan coordinated pincer attack"""
        # Find packs suitable for flanking
        flanking_packs = []
        
        for pack_id, pack in self.packs.items():
            # Only blue team packs
            pack_drones = [d for d in blue_drones if d['id'] in pack['drone_ids']]
            if len(pack_drones) >= 3:
                # Check if pack has striker or assault role
                pack_type = pack_drones[0].get('type', '')
                if 'striker' in pack_type or 'assault' in pack_type:
                    flanking_packs.append(pack_id)
        
        if len(flanking_packs) >= 2:
            # Assign flanking roles
            red_center = self.swarm_state['red']['center']
            
            # Left flank
            self.tactical_objectives[flanking_packs[0]] = {
                'type': 'flank_left',
                'target': {
                    'x': red_center['x'] - 1000,
                    'y': red_center['y'] + 1000,
                    'z': red_center['z']
                }
            }
            
            # Right flank
            self.tactical_objectives[flanking_packs[1]] = {
                'type': 'flank_right',
                'target': {
                    'x': red_center['x'] + 1000,
                    'y': red_center['y'] + 1000,
                    'z': red_center['z']
                }
            }
            
            print(f"⚔️ PINCER MOVEMENT: {flanking_packs[0]} LEFT, {flanking_packs[1]} RIGHT")

    def _select_swarm_target(self, drone: Dict, enemies: List[Dict], pack: Dict) -> Optional[Dict]:
        """Select target with swarm tactics consideration"""
        team = self._identify_drone_team(drone)
        best_target = None
        best_score = -float('inf')
        
        # Get current engagements
        drone_engagements = self.engagement_matrix.get(drone['id'], set())
        
        for enemy in enemies:
            if not enemy.get('active', True):
                continue
            
            # Skip if too many drones engaging this target
            enemy_engagements = sum(1 for d_id, engaged in self.engagement_matrix.items() 
                                  if enemy['id'] in engaged)
            if enemy_engagements >= self.simultaneous_engagement_limit:
                continue
            
            # Calculate tactical score
            score = 0
            
            # Distance factor
            dist = self._dist(drone, enemy)
            if dist > self.engagement_range:
                continue
            distance_score = 1.0 - (dist / self.engagement_range)
            score += distance_score * 3
            
            # Threat assessment
            enemy_type = enemy.get('type', '')
            if 'elite' in enemy_type:
                score += 2.0  # High priority
            elif 'bomber' in enemy_type:
                score += 1.5  # Medium-high priority
            
            # Focus fire bonus
            if enemy['id'] in self.swarm_state[team]['target_priorities']:
                score += self.swarm_state[team]['target_priorities'][enemy['id']]
            
            # Wounded target bonus
            # (In real implementation, would track drone health)
            
            # Tactical objective alignment
            if pack['id'] in self.tactical_objectives:
                objective = self.tactical_objectives[pack['id']]
                if objective['type'] in ['flank_left', 'flank_right']:
                    # Prefer targets on the flank side
                    flank_alignment = self._calculate_flank_alignment(drone, enemy, objective)
                    score += flank_alignment * 2
            
            if score > best_score:
                best_score = score
                best_target = enemy
        
        # Update engagement matrix
        if best_target:
            if drone['id'] not in self.engagement_matrix:
                self.engagement_matrix[drone['id']] = set()
            self.engagement_matrix[drone['id']].add(best_target['id'])
        
        return best_target

    def _calculate_flank_alignment(self, drone: Dict, target: Dict, objective: Dict) -> float:
        """Calculate how well a target aligns with flanking objective"""
        # Vector from drone to objective
        obj_dx = objective['target']['x'] - drone['x']
        obj_dy = objective['target']['y'] - drone['y']
        obj_dist = math.sqrt(obj_dx*obj_dx + obj_dy*obj_dy) or 1.0
        
        # Vector from drone to target
        tgt_dx = target['x'] - drone['x']
        tgt_dy = target['y'] - drone['y']
        tgt_dist = math.sqrt(tgt_dx*tgt_dx + tgt_dy*tgt_dy) or 1.0
        
        # Dot product for alignment
        alignment = (obj_dx*tgt_dx + obj_dy*tgt_dy) / (obj_dist * tgt_dist)
        
        return max(0, alignment)

    def _swarm_combat_formation(self, pack: Dict, enemies: List[Dict], friendlies: List[Dict]):
        """Compute combat formation based on tactical situation"""
        if not enemies:
            return
        
        # Find nearest enemy cluster
        enemy_center_x = sum(e['x'] for e in enemies[:5]) / min(5, len(enemies))
        enemy_center_y = sum(e['y'] for e in enemies[:5]) / min(5, len(enemies))
        
        # Get pack center
        pack_drones = [d for d in friendlies if d['id'] in pack['drone_ids']]
        if not pack_drones:
            return
        
        pack_center_x = sum(d['x'] for d in pack_drones) / len(pack_drones)
        pack_center_y = sum(d['y'] for d in pack_drones) / len(pack_drones)
        
        # Vector toward enemy
        dx = enemy_center_x - pack_center_x
        dy = enemy_center_y - pack_center_y
        dist = math.sqrt(dx*dx + dy*dy) or 1.0
        
        # Formation based on pack role and tactical objective
        pack_type = pack_drones[0].get('type', '')
        
        if 'assault' in pack_type or pack.get('tactical_role') == 'flank_left':
            # Wedge formation for assault
            self._compute_wedge_formation(pack, dx/dist, dy/dist, pack_center_x, pack_center_y)
        elif 'defender' in pack_type:
            # Sphere formation for defense
            self._compute_sphere_formation(pack, pack_center_x, pack_center_y, 1500)
        elif 'striker' in pack_type:
            # Line formation for strike
            self._compute_line_formation(pack, dx/dist, dy/dist, pack_center_x, pack_center_y)
        else:
            # Default distributed formation
            self._compute_combat_ring(pack, enemy_center_x, enemy_center_y)

    def _compute_wedge_formation(self, pack: Dict, dir_x: float, dir_y: float, center_x: float, center_y: float):
        """Wedge formation pointing toward enemy"""
        positions = {}
        spacing = 150
        
        # Perpendicular to direction
        perp_x, perp_y = -dir_y, dir_x
        
        for i, drone_id in enumerate(pack['drone_ids']):
            if i == 0:
                # Leader at point
                offset_x = dir_x * spacing
                offset_y = dir_y * spacing
            else:
                # Wings
                row = (i - 1) // 2 + 1
                side = 1 if (i - 1) % 2 == 0 else -1
                offset_x = -dir_x * spacing * row * 0.5
                offset_y = -dir_y * spacing * row * 0.5
                offset_x += perp_x * spacing * row * side
                offset_y += perp_y * spacing * row * side
            
            positions[drone_id] = {
                'x': center_x + offset_x,
                'y': center_y + offset_y,
                'z': 1000,
                'formation': 'wedge',
                'role': 'leader' if i == 0 else 'wing'
            }
        
        pack['formation_positions'] = positions

    def _compute_sphere_formation(self, pack: Dict, center_x: float, center_y: float, radius: float):
        """Defensive sphere formation"""
        positions = {}
        num_drones = len(pack['drone_ids'])
        
        # Distribute drones on sphere surface
        phi = math.pi * (3 - math.sqrt(5))  # Golden angle
        
        for i, drone_id in enumerate(pack['drone_ids']):
            y = 1 - (i / float(num_drones - 1)) * 2  # -1 to 1
            radius_at_y = math.sqrt(1 - y * y)
            
            theta = phi * i
            
            x = math.cos(theta) * radius_at_y
            z = math.sin(theta) * radius_at_y
            
            positions[drone_id] = {
                'x': center_x + x * radius,
                'y': center_y + y * radius,
                'z': 1000 + z * radius * 0.3,
                'formation': 'sphere',
                'role': 'defender'
            }
        
        pack['formation_positions'] = positions

    def _compute_line_formation(self, pack: Dict, dir_x: float, dir_y: float, center_x: float, center_y: float):
        """Line formation perpendicular to attack direction"""
        positions = {}
        spacing = 200
        
        # Perpendicular to direction
        perp_x, perp_y = -dir_y, dir_x
        
        num_drones = len(pack['drone_ids'])
        offset = -(num_drones - 1) / 2.0
        
        for i, drone_id in enumerate(pack['drone_ids']):
            positions[drone_id] = {
                'x': center_x + perp_x * spacing * (offset + i),
                'y': center_y + perp_y * spacing * (offset + i),
                'z': 1000,
                'formation': 'line',
                'role': 'striker'
            }
        
        pack['formation_positions'] = positions

    def _compute_combat_ring(self, pack: Dict, target_x: float, target_y: float):
        """Standard ring formation for combat"""
        radius = self.pack_formation_distance
        positions = {}
        
        num_drones = len(pack['drone_ids'])
        angle_step = 2 * math.pi / num_drones
        
        for i, drone_id in enumerate(pack['drone_ids']):
            angle = i * angle_step
            positions[drone_id] = {
                'x': target_x + radius * math.cos(angle),
                'y': target_y + radius * math.sin(angle),
                'z': 1000,
                'formation': 'ring',
                'role': 'combat'
            }
        
        pack['formation_positions'] = positions

    def _swarm_killer_pursuit(self, pack: Dict, drone: Dict, target: Dict, friendlies: List[Dict]) -> Dict:
        """Coordinated killer pursuit with swarm tactics"""
        d = self._dist(drone, target)
        
        # Strike check
        if d <= self.strike_distance:
            # Record combat event
            self._record_combat_event(drone, target, 'kill')
            
            target['active'] = False
            drone['active'] = False
            
            # Update swarm state
            drone_team = self._identify_drone_team(drone)
            target_team = self._identify_drone_team(target)
            self.swarm_state[drone_team]['kills'] += 1
            self.swarm_state[target_team]['losses'] += 1
            
            print(f"💥💥💥 SWARM COMBAT KILL: {drone['id']} ({drone_team}) destroyed {target['id']} ({target_team}) 💥💥💥")
            return self._decision(0,0,0, target['id'], 'killer', 'COMBAT_KILL')
        
        # Check for coordinated strike opportunity
        supporting_drones = []
        for friendly in friendlies:
            if friendly['id'] != drone['id'] and friendly.get('active', True):
                if self._dist(friendly, target) < self.engagement_range:
                    supporting_drones.append(friendly)
        
        # Enhanced pursuit with support
        if len(supporting_drones) >= 2:
            # Coordinated attack pattern
            return self._coordinated_attack(drone, target, supporting_drones)
        else:
            # Standard aggressive pursuit
            return self._aggressive_combat_pursuit(drone, target)

    def _coordinated_attack(self, drone: Dict, target: Dict, supporters: List[Dict]) -> Dict:
        """Execute coordinated attack with supporting drones"""
        # Predict target escape route
        tvx, tvy = target.get('vx', 0), target.get('vy', 0)
        tspeed = math.sqrt(tvx*tvx + tvy*tvy)
        
        if tspeed > 10:
            # Attack from predicted position
            escape_dir_x, escape_dir_y = tvx/tspeed, tvy/tspeed
            
            # Lead the target
            lead_time = min(2.0, self._dist(drone, target) / self.killer_max_velocity)
            pred_x = target['x'] + tvx * lead_time
            pred_y = target['y'] + tvy * lead_time
            pred_z = target.get('z', 0) + target.get('vz', 0) * lead_time
            
            # Approach from side to cut off escape
            approach_angle = math.pi / 4  # 45 degrees
            side = 1 if drone['y'] > target['y'] else -1
            
            attack_x = pred_x + escape_dir_y * 200 * side
            attack_y = pred_y - escape_dir_x * 200 * side
            
            dx = attack_x - drone['x']
            dy = attack_y - drone['y']
            dz = pred_z - drone.get('z', 0)
        else:
            # Direct attack if target is slow
            dx = target['x'] - drone['x']
            dy = target['y'] - drone['y']
            dz = target.get('z', 0) - drone.get('z', 0)
        
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
        ux, uy, uz = dx/dist, dy/dist, dz/dist
        
        # Maximum aggression with coordination bonus
        max_accel = self.killer_max_acceleration * self.strike_acceleration_multiplier * self.coordinated_strike_bonus
        ax = max_accel * ux
        ay = max_accel * uy
        az = max_accel * uz
        
        return self._decision(ax, ay, az, target['id'], 'killer', 'coordinated_strike')

    def _aggressive_combat_pursuit(self, drone: Dict, target: Dict) -> Dict:
        """Aggressive pursuit for combat scenarios"""
        d = self._dist(drone, target)
        
        # Very aggressive close range
        if d <= 150.0:
            dx = target['x'] - drone['x']
            dy = target['y'] - drone['y'] 
            dz = target.get('z', 0.0) - drone.get('z', 0.0)
            dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
            
            ux, uy, uz = dx/dist, dy/dist, dz/dist
            max_accel = self.killer_max_acceleration * self.strike_acceleration_multiplier * 1.2
            ax = max_accel * ux
            ay = max_accel * uy
            az = max_accel * uz
            
            return self._decision(ax, ay, az, target['id'], 'killer', 'close_combat')
        
        # Predictive interception
        tvx, tvy, tvz = target.get('vx', 0.0), target.get('vy', 0.0), target.get('vz', 0.0)
        
        # Shorter prediction for combat
        predict_time = min(2.0, d / max(self.killer_max_velocity, 50.0))
        
        pred_x = target['x'] + tvx * predict_time
        pred_y = target['y'] + tvy * predict_time
        pred_z = target.get('z', 0.0) + tvz * predict_time

        pdx = pred_x - drone['x']
        pdy = pred_y - drone['y']
        pdz = pred_z - drone.get('z', 0.0)
        pdist = math.sqrt(pdx*pdx + pdy*pdy + pdz*pdz) or 1.0

        # High speed intercept
        desired_speed = self.killer_max_velocity
        dirx, diry, dirz = pdx/pdist, pdy/pdist, pdz/pdist
        
        dvx_des = desired_speed * dirx
        dvy_des = desired_speed * diry
        dvz_des = desired_speed * dirz

        cvx, cvy, cvz = drone.get('vx',0.0), drone.get('vy',0.0), drone.get('vz',0.0)

        # Maximum agility for combat
        agile = self.killer_max_acceleration * self.strike_acceleration_multiplier
        ax = agile * (dvx_des - cvx) / max(1.0, desired_speed)
        ay = agile * (dvy_des - cvy) / max(1.0, desired_speed)
        az = agile * (dvz_des - cvz) / max(1.0, desired_speed)

        return self._decision(ax, ay, az, target['id'], 'killer', 'combat_pursuit')

    def _record_combat_event(self, attacker: Dict, victim: Dict, event_type: str):
        """Record combat events for analysis"""
        self.combat_history.append({
            'time': self.mission_time,
            'type': event_type,
            'attacker': attacker['id'],
            'attacker_team': self._identify_drone_team(attacker),
            'victim': victim['id'],
            'victim_team': self._identify_drone_team(victim),
            'location': {
                'x': victim['x'],
                'y': victim['y'],
                'z': victim.get('z', 0)
            }
        })

    def update(self, targets: List[Dict], interceptors: List[Dict], dt: float, wind: Dict, scenario_config: Dict = None) -> Dict[str, Dict]:
        if scenario_config:
            self._load_all_parameters(scenario_config)
            
        eff_dt = dt if (dt is not None and dt > 0.0) else (self.dt if self.frame_count == 0 else 0.0)
        
        self.frame_count += 1
        self.mission_time += eff_dt
        self.dt = eff_dt
        
        # Update swarm state
        self._update_swarm_state(targets, interceptors)
        
        # Physics step
        self._update_physics_combat(targets, interceptors, eff_dt, wind)
        
        # Initialize packs once
        if not self.packs_initialized:
            self._create_combat_packs(targets, interceptors)
            self.packs_initialized = True
        
        # Clear engagement matrix for fresh target selection
        self.engagement_matrix.clear()
        
        # Debug output
        if self.frame_count % 30 == 0:
            self._print_combat_debug(targets, interceptors)
            
        # Generate decisions for all packs
        decisions: Dict[str, Dict] = {}
        
        # Get all active drones
        all_active = [d for d in targets + interceptors if d.get('active', True)]
        
        # Process blue team packs
        for pack_id, pack in self.packs.items():
            # Determine enemies and friendlies
            pack_team = 'blue'  # Assuming player controls blue team
            enemies = [d for d in all_active if self._identify_drone_team(d) != pack_team]
            friendlies = [d for d in all_active if self._identify_drone_team(d) == pack_team]
            
            pack_decisions = self._update_combat_pack(pack, enemies, friendlies, interceptors)
            decisions.update(pack_decisions)
            
        return decisions

    def _update_physics_combat(self, targets: List[Dict], interceptors: List[Dict], dt: float, wind: Dict):
        """Update physics with combat-specific parameters"""
        if dt <= 0.0:
            return

        # All drones use combat physics
        all_drones = targets + interceptors
        
        for d in all_drones:
            if not d.get('active', True):
                continue
            
            if 'vx' not in d: d['vx'] = 0.0
            if 'vy' not in d: d['vy'] = 0.0
            if 'vz' not in d: d['vz'] = 0.0
            if 'ax' not in d: d['ax'] = 0.0
            if 'ay' not in d: d['ay'] = 0.0
            if 'az' not in d: d['az'] = 0.0
            
            # Get type-specific parameters
            drone_type = d.get('type', '')
            
            # Check if killer
            pack_id = self.drone_pack_map.get(d['id'])
            is_killer = False
            if pack_id and pack_id in self.packs:
                is_killer = (self.packs[pack_id].get('killer_drone') == d['id'])
            
            # Get max values based on type
            if drone_type in self.team_specs:
                spec = self.team_specs[drone_type]
                max_a = spec['max_acceleration']
                max_v = spec['max_velocity']
            else:
                # Default values
                max_a = self.green_max_acceleration
                max_v = self.green_max_velocity
            
            # Apply killer modifiers
            if is_killer:
                max_a *= 1.2
                max_v *= 1.1
                
                # Deceleration boost
                vx, vy, vz = d['vx'], d['vy'], d['vz']
                ax, ay, az = d['ax'], d['ay'], d['az']
                sp = math.sqrt(vx*vx + vy*vy + vz*vz)
                if sp > 1.0 and (vx*ax + vy*ay + vz*az)/sp < 0:
                    max_a *= self.killer_deceleration_multiplier
            
            # Apply physics
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

            wx = wind.get('x', 0.0); wy = wind.get('y', 0.0); wz = wind.get('z', 0.0)
            d['x'] += (d['vx'] + 0.1*wx) * dt
            d['y'] += (d['vy'] + 0.1*wy) * dt
            d['z'] = d.get('z', 0.0) + (d['vz'] + 0.1*wz) * dt

    def _create_combat_packs(self, targets: List[Dict], interceptors: List[Dict]):
        """Create packs for swarm combat"""
        # Separate teams
        all_drones = targets + interceptors
        blue_drones = []
        red_drones = []
        
        for drone in all_drones:
            if drone.get('active', True):
                team = self._identify_drone_team(drone)
                if team == 'blue':
                    blue_drones.append(drone)
                else:
                    red_drones.append(drone)
        
        print("\n" + "⚔️" * 20)
        print(f"⚔️ INITIALIZING SWARM COMBAT: Blue={len(blue_drones)} vs Red={len(red_drones)}")
        
        # Create blue team packs (player controlled)
        self._organize_team_packs(blue_drones, 'blue')
        
        # Note: Red team would have its own AI system in full implementation
        
        print("⚔️" * 20 + "\n")

    def _organize_team_packs(self, team_drones: List[Dict], team: str):
        """Organize drones into tactical packs"""
        # Group by type/role
        role_groups = {}
        for drone in team_drones:
            drone_type = drone.get('type', 'unknown')
            role = 'general'
            
            if 'assault' in drone_type:
                role = 'assault'
            elif 'defender' in drone_type:
                role = 'defender'
            elif 'striker' in drone_type:
                role = 'striker'
            elif 'support' in drone_type:
                role = 'support'
            
            if role not in role_groups:
                role_groups[role] = []
            role_groups[role].append(drone)
        
        # Create packs from role groups
        pack_num = 0
        for role, drones in role_groups.items():
            # Split into packs of 4
            for i in range(0, len(drones), 4):
                pack_drones = drones[i:i+4]
                if pack_drones:
                    pack_id = f"{team}_pack_{pack_num}"
                    pack_num += 1
                    
                    drone_ids = [d['id'] for d in pack_drones]
                    
                    self.packs[pack_id] = {
                        'team': team,
                        'role': role,
                        'target_id': None,
                        'drone_ids': drone_ids,
                        'killer_drone': None,
                        'green_drones': drone_ids.copy(),
                        'pack_state': 'searching',
                        'formation_positions': {},
                        'chase_start_time': self.mission_time,
                        'first_ready_time': None,
                        'engage_unlocked': False,
                    }
                    
                    # Update drone-pack mapping
                    for did in drone_ids:
                        self.drone_pack_map[did] = pack_id
                    
                    print(f"⚔️ {team.upper()} {role.upper()} PACK: {pack_id} with {len(drone_ids)} drones")

    def _update_combat_pack(self, pack: Dict, enemies: List[Dict], friendlies: List[Dict], all_interceptors: List[Dict]) -> Dict[str, Dict]:
        """Update pack with combat tactics"""
        out: Dict[str, Dict] = {}
        
        # Get active pack members
        pack_drones = [d for d in friendlies if d['id'] in pack['drone_ids'] and d.get('active', True)]
        if not pack_drones:
            return out
        
        # Find target
        current_target = None
        if pack.get('target_id'):
            current_target = next((e for e in enemies if e['id'] == pack['target_id'] and e.get('active', True)), None)
        
        # Select new target if needed
        if not current_target:
            current_target = self._select_swarm_target(pack_drones[0], enemies, pack)
            if current_target:
                pack['target_id'] = current_target['id']
            else:
                # No enemies in range - patrol or regroup
                for d in pack_drones:
                    out[d['id']] = self._patrol_behavior(d, pack)
                return out
        
        # Update formation for combat
        self._swarm_combat_formation(pack, enemies, friendlies)
        
        # Combat state machine
        current_state = pack.get('pack_state', 'searching')
        
        if current_state == 'searching':
            # Move toward engagement zone
            pack['pack_state'] = 'engaging'
            pack['chase_start_time'] = self.mission_time
            
        elif current_state == 'engaging':
            # Direct combat engagement
            
            # Select killer if needed
            if pack.get('killer_drone') is None:
                # Choose based on position and capability
                best_killer = None
                best_score = -1
                
                for d in pack_drones:
                    dist = self._dist(d, current_target)
                    drone_type = d.get('type', '')
                    
                    score = 1000 - dist  # Closer is better
                    
                    # Type bonuses
                    if 'striker' in drone_type:
                        score += 200
                    elif 'assault' in drone_type:
                        score += 100
                    
                    if score > best_score:
                        best_score = score
                        best_killer = d
                
                if best_killer:
                    pack['killer_drone'] = best_killer['id']
                    pack['green_drones'] = [d['id'] for d in pack_drones if d['id'] != best_killer['id']]
            
            # Generate combat decisions
            for d in pack_drones:
                if d['id'] == pack.get('killer_drone'):
                    out[d['id']] = self._swarm_killer_pursuit(pack, d, current_target, friendlies)
                else:
                    # Support roles
                    fp = pack['formation_positions'].get(d['id'])
                    out[d['id']] = self._combat_support(d, current_target, fp, pack)
        
        elif current_state == 'regrouping':
            # Regroup after engagement
            for d in pack_drones:
                fp = pack['formation_positions'].get(d['id'])
                out[d['id']] = self._move_to_formation(d, None, fp)
            
            # Check if regrouped
            if self.mission_time - pack.get('regroup_start_time', 0) > 3.0:
                pack['pack_state'] = 'searching'
        
        return out

    def _patrol_behavior(self, drone: Dict, pack: Dict) -> Dict:
        """Patrol behavior when no enemies in range"""
        # Move toward tactical objective or patrol area
        team = self._identify_drone_team(drone)
        
        if team in self.tactical_objectives:
            objective = self.tactical_objectives[team]
            target_pos = objective.get('target_zone', {'x': 0, 'y': 0, 'z': 1000})
        else:
            # Default patrol around team center
            target_pos = self.swarm_state[team]['center']
        
        dx = target_pos['x'] - drone['x']
        dy = target_pos['y'] - drone['y']
        dz = target_pos['z'] - drone.get('z', 0)
        
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
        
        if dist > 100:
            # Move toward patrol point
            ux, uy, uz = dx/dist, dy/dist, dz/dist
            gain = self.green_max_acceleration * 0.5
            ax = gain * ux
            ay = gain * uy
            az = gain * uz
        else:
            # Orbit patrol point
            orbit_radius = 500
            angle = math.atan2(drone['y'] - target_pos['y'], drone['x'] - target_pos['x'])
            angle += 0.02  # Slow rotation
            
            desired_x = target_pos['x'] + orbit_radius * math.cos(angle)
            desired_y = target_pos['y'] + orbit_radius * math.sin(angle)
            
            dx = desired_x - drone['x']
            dy = desired_y - drone['y']
            
            ax = dx * 0.5
            ay = dy * 0.5
            az = 0
        
        return self._decision(ax, ay, az, 'patrol', 'patrol', 'searching')

    def _combat_support(self, drone: Dict, target: Dict, fp: Optional[Dict], pack: Dict) -> Dict:
        """Support behavior for non-killer drones in combat"""
        # Maintain formation while supporting killer
        if fp:
            # Move to formation position
            dx = fp['x'] - drone['x']
            dy = fp['y'] - drone['y']
            dz = fp.get('z', 1000) - drone.get('z', 0)
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            if dist > 30:
                ux, uy, uz = dx/dist, dy/dist, dz/dist
                gain = self.green_max_acceleration * 0.7
                pos_ax = gain * ux
                pos_ay = gain * uy
                pos_az = gain * uz
            else:
                pos_ax = pos_ay = pos_az = 0
            
            # Also track target
            tvx, tvy = target.get('vx', 0), target.get('vy', 0)
            dvx, dvy = drone.get('vx', 0), drone.get('vy', 0)
            
            vel_ax = 2.0 * (tvx - dvx)
            vel_ay = 2.0 * (tvy - dvy)
            vel_az = 0
            
            # Combine position and velocity control
            ax = pos_ax * 0.6 + vel_ax * 0.4
            ay = pos_ay * 0.6 + vel_ay * 0.4
            az = pos_az
            
            return self._decision(ax, ay, az, target['id'], 'support', 'combat_support')
        else:
            # No formation position - circle target
            return self._defensive_orbit(drone, target, 300)

    def _defensive_orbit(self, drone: Dict, target: Dict, radius: float) -> Dict:
        """Defensive orbit around target"""
        dx = target['x'] - drone['x']
        dy = target['y'] - drone['y']
        dist = math.sqrt(dx*dx + dy*dy) or 1.0
        
        error = dist - radius
        
        if abs(error) > 50:
            # Adjust distance
            ux, uy = dx/dist, dy/dist
            radial_ax = self.green_max_acceleration * 0.4 * (1 if error > 0 else -1) * ux
            radial_ay = self.green_max_acceleration * 0.4 * (1 if error > 0 else -1) * uy
        else:
            radial_ax = radial_ay = 0
        
        # Orbital velocity
        tx, ty = -dy/dist, dx/dist
        orbit_speed = 60.0
        dvx, dvy = drone.get('vx', 0), drone.get('vy', 0)
        
        tangent_ax = 2.0 * (orbit_speed * tx - dvx)
        tangent_ay = 2.0 * (orbit_speed * ty - dvy)
        
        ax = radial_ax + tangent_ax
        ay = radial_ay + tangent_ay
        az = 0
        
        return self._decision(ax, ay, az, target['id'], 'support', 'defensive_orbit')

    def _print_combat_debug(self, targets: List[Dict], interceptors: List[Dict]):
        """Debug output for swarm combat"""
        print("\n" + "="*80)
        print(f"⚔️ SWARM COMBAT STATUS: t={self.mission_time:.2f}s")
        
        # Team status
        all_drones = targets + interceptors
        blue_active = len([d for d in all_drones if d.get('active', True) and self._identify_drone_team(d) == 'blue'])
        red_active = len([d for d in all_drones if d.get('active', True) and self._identify_drone_team(d) == 'red'])
        
        print(f"🔵 BLUE: {blue_active} active, {self.swarm_state['blue']['kills']} kills, " +
              f"{self.swarm_state['blue']['losses']} losses, morale={self.swarm_state['blue']['morale']:.0f}")
        print(f"🔴 RED: {red_active} active, {self.swarm_state['red']['kills']} kills, " +
              f"{self.swarm_state['red']['losses']} losses, morale={self.swarm_state['red']['morale']:.0f}")
        
        # Tactical status
        if self.tactical_objectives:
            print("\n📋 TACTICAL OBJECTIVES:")
            for obj_id, objective in self.tactical_objectives.items():
                if isinstance(obj_id, str) and obj_id in ['blue', 'red']:
                    print(f"   {obj_id.upper()}: {objective.get('type', 'unknown')}")
        
        # Pack status
        print("\n📦 PACK STATUS:")
        for pack_id, pack in self.packs.items():
            active_count = len([d for d in all_drones 
                              if d['id'] in pack['drone_ids'] and d.get('active', True)])
            if active_count > 0:
                print(f"   {pack_id}: {active_count} active, role={pack.get('role', 'unknown')}, " +
                      f"state={pack.get('pack_state', 'unknown')}, target={pack.get('target_id', 'NONE')}")
        
        # Recent combat events
        recent_events = [e for e in self.combat_history if self.mission_time - e['time'] < 5.0]
        if recent_events:
            print("\n💥 RECENT COMBAT:")
            for event in recent_events[-3:]:
                print(f"   t={event['time']:.1f}: {event['attacker']} killed {event['victim']}")
        
        print("="*80)

    # Include base methods
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

    def _move_to_formation(self, drone: Dict, target: Optional[Dict], fp: Optional[Dict]) -> Dict:
        if not fp:
            return self._decision(0,0,0, 'none', 'forming', 'no_formation_slot')

        dx = fp['x'] - drone['x']
        dy = fp['y'] - drone['y']
        dz = fp.get('z', 1000) - drone.get('z', 0.0)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0

        ux, uy, uz = dx/dist, dy/dist, dz/dist
        gain = self.green_max_acceleration * 0.9
        ax = gain * ux
        ay = gain * uy
        az = gain * uz
        return self._decision(ax, ay, az, 'formation', 'forming', 'moving_to_formation')

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