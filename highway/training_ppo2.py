import gym
import argparse
import highway_env_v2

import numpy as np

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from misc.highway_config import HighwayEnvironmentConfig

parser = argparse.ArgumentParser()
parser.add_argument('--environment_vehicles', type=int, default=15, help="total_number of vehicles in the environment")
args = parser.parse_args()

# Suppress exponential notation
np.set_printoptions(suppress=True)

# Setup the configuration
hw_config = HighwayEnvironmentConfig(environment_vehicles=args.environment_vehicles)

# Create the environment
env = gym.make("highway-v0")
env.config = hw_config.env_configuration
obs = env.reset()
obs = np.round(obs, 4)

# Create the model
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

# Run the algorithm
for i in range(1000):
    # Predict
    action, _states = model.predict(obs)

    # Get reward
    obs, reward, done, info = env.step(action)
    print(reward)
    obs = np.round(obs, 4)

    # Render
    env.render()

env.close()
