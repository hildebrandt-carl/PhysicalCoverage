import gym
import argparse
import highway_env_v2

import numpy as np

from stable_baselines import DQN
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
model = DQN('MlpPolicy', env, verbose=1, exploration_final_eps=0.1)
# model.learn(total_timesteps=int(1e6))
# model.save("dqn_test")
model.load("dqn_test")
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
