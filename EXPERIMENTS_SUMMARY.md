# Enhanced Experiments for ICDTDE2025 Publication

## Overview

The shepherd-grid project has been enhanced with six sophisticated experiments that demonstrate advanced multi-agent coordination, adaptive behavior, and strategic depth. These experiments provide rich visual elements and comprehensive test scenarios suitable for publication at the International Conference on Digital Technology Driven Engineering 2025.

## Experiment Suite

### 1. Multi-Target Scenarios with Dynamic Priority Assignment
**File:** `experiment1_multi_target.txt`, `rules_multi_target.py`

**Key Features:**
- Dynamic threat prioritization based on distance, speed, altitude, and target type
- Pack splitting and merging strategies
- Real-time priority reassignment
- 16 interceptors vs 4 diverse targets

**Visual Elements:**
- Dynamic pack formations
- Target priority indicators (color/size coding)
- Split/merge animations
- Threat assessment overlays

**Research Value:**
- Demonstrates scalability and adaptability
- Shows intelligent resource allocation
- Validates distributed decision-making

### 2. Adversarial Evasion Patterns
**File:** `experiment2_evasion_patterns.txt`, `rules_evasion.py`

**Key Features:**
- Four evasion patterns: zigzag, spiral, random walk, barrel roll
- Pattern recognition and prediction
- Adaptive pursuit strategies
- Monte Carlo prediction sampling

**Visual Elements:**
- Evasion pattern trails (different colors/styles)
- Prediction confidence indicators
- Pattern detection overlays
- Intercept trajectory visualization

**Research Value:**
- Tests robustness against intelligent adversaries
- Showcases advanced prediction algorithms
- Demonstrates learning and adaptation

### 3. Communication Degradation/Failure Scenarios
**File:** `experiment3_comm_degradation.txt`, `rules_comm_degradation.py`

**Key Features:**
- Jamming zones and signal degradation
- Mesh networking capabilities
- Autonomous local decision-making
- Graceful degradation strategies

**Visual Elements:**
- Communication link visualization
- Signal strength heat maps
- Isolated drone indicators
- Mesh network topology display

**Research Value:**
- Critical for real-world deployment
- Shows emergent behavior
- Validates resilience mechanisms

### 4. Heterogeneous Pack Composition
**File:** `experiment4_heterogeneous_packs.txt`, `rules_heterogeneous.py`

**Key Features:**
- Four drone types: scout, fast interceptor, heavy interceptor, sensor
- Capability-based role assignment
- Optimized mixed formations
- Type-specific tactics

**Visual Elements:**
- Different drone models/colors by type
- Role indicators (icons/labels)
- Capability comparison charts
- Formation adaptation animations

**Research Value:**
- Demonstrates resource optimization
- Shows advanced coordination algorithms
- Validates heterogeneous swarm control

### 5. Energy/Resource Constraint Experiments
**File:** `experiment5_energy_constraints.txt`, `rules_energy.py`

**Key Features:**
- Fuel tracking and consumption modeling
- Efficient path planning
- Return-to-base decisions
- Formation draft benefits

**Visual Elements:**
- Fuel gauge overlays
- Efficiency heat trails
- RTB path indicators
- Energy optimization zones

**Research Value:**
- Addresses practical constraints
- Shows trade-off optimization
- Demonstrates sustainable operations

### 6. Swarm vs Swarm Combat
**File:** `experiment6_swarm_combat.txt`, `rules_swarm_combat.py`

**Key Features:**
- 16 vs 12 drone combat
- Tactical formations (wedge, sphere, line)
- Coordinated assault strategies
- Morale and combat tracking

**Visual Elements:**
- Team colors (blue vs red)
- Combat engagement zones
- Tactical formation displays
- Kill indicators and score tracking

**Research Value:**
- Shows strategic depth
- Demonstrates complex multi-agent scenarios
- Validates combat coordination algorithms

## Running the Experiments

### For 3D Visualization:
```bash
python 3d_app.py --scenario experiment1_multi_target.txt
python 3d_app.py --scenario experiment2_evasion_patterns.txt
# ... etc
```

### To Use Enhanced AI Systems:
Modify `3d_app.py` to import the appropriate AI system:
```python
# For multi-target scenarios
from rules_multi_target import AI_System_MultiTarget as AI_System

# For evasion patterns
from rules_evasion import AI_System_Evasion as AI_System

# etc...
```

## Visual Elements for Publication

Each experiment provides:
1. **Real-time 3D visualization** with distinct visual cues
2. **Formation patterns** that are visually striking
3. **State transitions** with clear visual feedback
4. **Performance metrics** suitable for graphs/charts
5. **Emergent behaviors** that photograph well

## Recommended Figures for Paper

1. **Figure 1:** Multi-target engagement showing pack split
2. **Figure 2:** Evasion pattern trails comparison
3. **Figure 3:** Communication degradation heat map
4. **Figure 4:** Heterogeneous formation adaptation
5. **Figure 5:** Fuel efficiency comparison chart
6. **Figure 6:** Swarm combat tactical formations

## Key Metrics to Track

- Success rate per experiment
- Time to intercept
- Fuel efficiency
- Communication resilience
- Formation coherence
- Tactical effectiveness

## Conclusion

These six experiments significantly enhance the research value of the shepherd-grid project, providing comprehensive test scenarios that demonstrate:
- Advanced multi-agent coordination
- Adaptive behavior under constraints
- Robust performance in adverse conditions
- Strategic depth and tactical flexibility

The visual elements and quantifiable metrics make these experiments ideal for publication at ICDTDE2025.