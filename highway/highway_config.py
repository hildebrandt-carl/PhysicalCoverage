class HighwayConfig:
    def __init__(self, environment_vehicles=20):
        policy_freq = 5 #[hz]
        duration = 80 #[s]
        self.env_configuration = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "color", "lane"],
                "absolute": True,
                "normalize": False,
                "see_behind": True,
                "ego_frame": True
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": environment_vehicles,
            "duration": duration * policy_freq,  # [s]
            "initial_spacing": 1,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": policy_freq,  # [Hz]
            "other_vehicles_type": "highway_env_v2.vehicle.behavior.IDMVehicle",
            "screen_width": 1200,  # [px]
            "screen_height": 520,  # [px]
            "centering_position": [0.3, 0.5],
            "scaling": 7,
            "controlled_vehicles": 1,
            "ego_spacing": 1,
            "initial_lane_id": None,
            "show_trajectories": False,
            "offroad_terminal": False,
            "vehicles_density": 2,
            "render_agent": True,
            "offscreen_rendering": False,
            "manual_control": False,
            "real_time_rendering": True
        }
