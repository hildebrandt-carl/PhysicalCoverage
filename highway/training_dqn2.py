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
model = DQN('MlpPolicy', env,
            gamma=0.8,
            learning_rate=5e-4,
            buffer_size=40*1000,
            learning_starts=200,
            exploration_fraction=0.6,
            batch_size=128,
            verbose=1,
            tensorboard_log="logs/")

# model.learn(total_timesteps=int(2e5))
# model.save("dqn_test")
model.load("dqn_test")

# Run the algorithm
done = False
while not done:
    # Predict
    action, _states = model.predict(obs)

    # Get reward
    obs, reward, done, info = env.step(action)
    print(reward)

    # Render
    env.render()

env.close()
