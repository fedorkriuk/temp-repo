from flask import Flask, jsonify, request, render_template
from rules import AI_System
import argparse
import os
import sys
from typing import Dict, List, Tuple

app = Flask(__name__)

# Global simulation state (for 3D control endpoints)
ai_system: AI_System = None
current_scenario = "scenario_3d_one_target.txt"
sim_targets: List[Dict] = []
sim_interceptors: List[Dict] = []
sim_dt: float = 1.0 / 30.0
sim_wind: Dict = {"x": 0.0, "y": 0.0, "z": 0.0}
sim_config: Dict = {}
sim_started: bool = False


def load_scenario_file(scenario_name: str) -> str:
    """Load scenario file with flexible fallbacks."""
    try:
        candidates = [
            scenario_name,
            f"{scenario_name}.txt",
            os.path.join("scenarios", scenario_name),
            os.path.join("scenarios", f"{scenario_name}.txt"),
        ]
        scenario_path = None
        for c in candidates:
            if os.path.exists(c):
                scenario_path = c
                break
        if scenario_path is None:
            raise FileNotFoundError(f"Scenario file not found: {scenario_name}")

        with open(scenario_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"🎯 SCENARIO LOADED: {scenario_path}")
        return content
    except Exception as e:
        print(f"❌ ERROR loading scenario {scenario_name}: {e}")
        # Fallback to default if available in project root
        with open("scenario_3d_one_target.txt", "r", encoding="utf-8") as f:
            return f.read()


def _smart_cast(val: str):
    """Attempt to cast to int/float, fallback to stripped string."""
    try:
        if "." in val:
            return float(val)
        return int(val)
    except:
        try:
            return float(val)
        except:
            return val.strip()


def parse_scenario_config(content: str) -> dict:
    """
    Parse scenario file into structured config.
    Supports sections:
      [PHYSICS], [AI_PARAMETERS], [MISSION_PARAMETERS], [ENVIRONMENT],
      [TARGETS], [INTERCEPTORS], [RADAR].
    """
    config = {
        "physics": {},
        "ai_parameters": {},
        "mission_parameters": {},
        "environment": {},
        "targets": [],
        "interceptors": [],
        "radar": {},
    }

    lines = content.splitlines()
    current_section = None

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # Section header
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip().lower()
            continue

        # Key=Value pairs
        if "=" in line and current_section in {
            "physics",
            "ai_parameters",
            "mission_parameters",
            "environment",
            "radar",
        }:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            config[current_section][key] = _smart_cast(value)
            continue

        # CSV rows in [TARGETS] or [INTERCEPTORS]
        if current_section in {"targets", "interceptors"}:
            # Strip inline comments
            row = line.split("#", 1)[0].strip()
            if not row:
                continue
            parts = [p.strip() for p in row.split(",") if p.strip()]
            if len(parts) < 4:
                continue  # need at least id, x, y, z
            # Expected leading fields: id, x, y, z, vx, vy, vz (others ignored)
            obj_id = parts[0]

            def getf(idx: int, default: float = 0.0) -> float:
                if idx < len(parts):
                    try:
                        return float(parts[idx])
                    except:
                        return default
                return default

            x = getf(1, 0.0)
            y = getf(2, 0.0)
            z = getf(3, 0.0)
            vx = getf(4, 0.0)
            vy = getf(5, 0.0)
            vz = getf(6, 0.0)

            obj = {
                "id": obj_id,
                "x": x,
                "y": y,
                "z": z,
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "ax": 0.0,
                "ay": 0.0,
                "az": 0.0,
                "active": True,
            }
            if current_section == "targets":
                config["targets"].append(obj)
            else:
                config["interceptors"].append(obj)

    return config


def create_objects_from_config(cfg: Dict) -> Tuple[List[Dict], List[Dict], Dict, float]:
    """Build targets, interceptors, wind, dt from parsed config."""
    targets = [dict(t) for t in cfg.get("targets", [])]
    interceptors = [dict(i) for i in cfg.get("interceptors", [])]

    env = cfg.get("environment", {})
    wind = {
        "x": float(env.get("wind_x", 0.0)),
        "y": float(env.get("wind_y", 0.0)),
        "z": float(env.get("wind_z", 0.0)),
    }
    dt = 1.0 / 30.0  # default
    return targets, interceptors, wind, dt


def _apply_decisions_to_interceptors(interceptors: List[Dict], decisions: Dict[str, Dict]):
    """
    Write ax, ay, az, role from decisions back into interceptor objects.
    This ensures physics in the next step uses the latest accelerations,
    and the viewer gets correct roles/colors.
    """
    if not decisions:
        return
    by_id = {d["id"]: d for d in interceptors}
    for did, dec in decisions.items():
        d = by_id.get(did)
        if not d:
            continue
        d["ax"] = float(dec.get("ax", 0.0))
        d["ay"] = float(dec.get("ay", 0.0))
        d["az"] = float(dec.get("az", 0.0))
        if "role" in dec:
            d["role"] = dec["role"]


# ----------------------------
# UI compatibility endpoints (for your existing radar.html)
# ----------------------------

@app.route("/")
def index():
    # Requires templates/radar.html to exist
    return render_template("radar.html")


@app.route("/api/scenario")
def api_scenario():
    """Return scenario text for the UI to parse."""
    try:
        content = load_scenario_file(current_scenario)
        return jsonify({"scenario": content, "scenario_name": current_scenario})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ai_update", methods=["POST"])
def api_ai_update():
    """
    UI loop-compatible AI update.
    Body JSON should include:
      targets, interceptors, dt, wind, scenario_config
    """
    global ai_system
    try:
        data = request.json or {}

        # Initialize AI system once with scenario config
        if ai_system is None:
            scenario_config = data.get("scenario_config", {})
            ai_system = AI_System(scenario_config)

        decisions = ai_system.update(
            data["targets"],
            data["interceptors"],
            data.get("dt", 1.0 / 30.0),
            data.get("wind", {"x": 0.0, "y": 0.0}),
            data.get("scenario_config", {}),
        )

        return jsonify(
            {
                "decisions": decisions,
                "updated_targets": data["targets"],
                "updated_interceptors": data["interceptors"],
            }
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)}), 500


# ----------------------------
# 3D control endpoints
# ----------------------------

def center_camera_on_objects(targets: List[Dict], interceptors: List[Dict]) -> Dict:
    """Calculate optimal camera position to see all objects"""
    all_objects = targets + interceptors
    if not all_objects:
        return {"x": 0, "y": 0, "z": 600}
    
    # Find bounding box
    min_x = min(obj['x'] for obj in all_objects)
    max_x = max(obj['x'] for obj in all_objects)
    min_y = min(obj['y'] for obj in all_objects)
    max_y = max(obj['y'] for obj in all_objects)
    min_z = min(obj.get('z', 0) for obj in all_objects)
    max_z = max(obj.get('z', 0) for obj in all_objects)
    
    # Center point
    center = {
        "x": (min_x + max_x) / 2,
        "y": (min_y + max_y) / 2,
        "z": (min_z + max_z) / 2
    }
    
    print(f"📹 CAMERA CENTER: {center} (objects span: x={min_x:.0f}→{max_x:.0f}, y={min_y:.0f}→{max_y:.0f}, z={min_z:.0f}→{max_z:.0f})")
    return center


@app.route("/api/start_3d", methods=["POST"])
def api_start_3d():
    """
    Initialize a 3D simulation from a scenario file.
    Body JSON (optional):
      { "scenario": "scenario_3d_one_target.txt" }
    """
    global ai_system, sim_targets, sim_interceptors, sim_dt, sim_wind, sim_config, sim_started, current_scenario

    try:
        payload = request.json or {}
        scenario_name = payload.get("scenario", current_scenario)
        content = load_scenario_file(scenario_name)
        cfg = parse_scenario_config(content)

        # Create objects and wind
        targets, interceptors, wind, dt = create_objects_from_config(cfg)

        # Calculate optimal camera position
        camera_center = center_camera_on_objects(targets, interceptors)

        # Initialize AI with scenario config (physics/ai/mission only)
        ai_system = AI_System(
            {
                "physics": cfg.get("physics", {}),
                "ai_parameters": cfg.get("ai_parameters", {}),
                "mission_parameters": cfg.get("mission_parameters", {}),
            }
        )

        sim_targets = targets
        sim_interceptors = interceptors
        sim_wind = wind
        sim_dt = dt
        sim_config = cfg
        sim_started = True
        current_scenario = scenario_name

        # Seed one zero-dt update to assign packs/roles and initial accelerations
        seed_decisions = ai_system.update(
            targets=sim_targets,
            interceptors=sim_interceptors,
            dt=0.0,
            wind={"x": sim_wind["x"], "y": sim_wind["y"]},
            scenario_config={
                "physics": sim_config.get("physics", {}),
                "ai_parameters": sim_config.get("ai_parameters", {}),
                "mission_parameters": sim_config.get("mission_parameters", {}),
            },
        )
        _apply_decisions_to_interceptors(sim_interceptors, seed_decisions)

        return jsonify(
            {
                "status": "started",
                "scenario": scenario_name,
                "targets": sim_targets,
                "interceptors": sim_interceptors,
                "wind": sim_wind,
                "dt": sim_dt,
                "camera_center": camera_center,  # Send camera position to frontend
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/step_3d", methods=["POST"])
def api_step_3d():
    """
    Advance the 3D simulation by n steps (default 1).
    Body JSON (optional):
      { "n": 1, "dt": 0.0333 }
    """
    global ai_system, sim_targets, sim_interceptors, sim_dt, sim_wind, sim_config, sim_started

    if not sim_started or ai_system is None:
        return jsonify({"error": "Simulation not started. Call /api/start_3d first."}), 400

    try:
        payload = request.json or {}
        n = int(payload.get("n", 1))
        dt = float(payload.get("dt", sim_dt))

        last_decisions = {}
        for _ in range(max(1, n)):
            last_decisions = ai_system.update(
                targets=sim_targets,
                interceptors=sim_interceptors,
                dt=dt,
                wind={"x": sim_wind["x"], "y": sim_wind["y"]},  # rules.py uses x,y
                scenario_config={
                    "physics": sim_config.get("physics", {}),
                    "ai_parameters": sim_config.get("ai_parameters", {}),
                    "mission_parameters": sim_config.get("mission_parameters", {}),
                },
            )
            # Apply decisions so the next physics step uses new accelerations, and roles are visible
            _apply_decisions_to_interceptors(sim_interceptors, last_decisions)

        return jsonify(
            {
                "decisions": last_decisions,
                "targets": sim_targets,
                "interceptors": sim_interceptors,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/state_3d", methods=["GET"])
def api_state_3d():
    """Get current simulation state."""
    if not sim_started or ai_system is None:
        return jsonify({"error": "Simulation not started. Call /api/start_3d first."}), 400

    return jsonify(
        {
            "scenario": current_scenario,
            "targets": sim_targets,
            "interceptors": sim_interceptors,
            "wind": sim_wind,
            "dt": sim_dt,
        }
    )


@app.route("/3d")
def intercept_3d():
    """3D viewer page."""
    return render_template("intercept_3d.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Agile Killer Pack Defense System")
    parser.add_argument(
        "--scenario",
        "-s",
        default="scenario_3d_one_target.txt",
        help="Scenario file to load (default: scenario_3d_one_target.txt)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8910,
        help="Port to run server on (default: 8910)",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug mode",
    )
    args = parser.parse_args()

    # Set default scenario (loaded by /api/start_3d or /api/scenario)
    current_scenario = args.scenario

    print("🚀 STARTING 3D AGILE KILLER SYSTEM")
    print(f"🎯 Default Scenario: {current_scenario}")
    print(f"🌐 Server: http://localhost:{args.port}")
    print("🐺 Ready for 3D pack defense!")

    app.run(debug=args.debug, port=args.port, host="0.0.0.0")