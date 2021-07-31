import os
import gym
import argparse
import highway_env

import numpy as np
from stable_baselines import DQN

parser = argparse.ArgumentParser()
parser.add_argument('--model_name',             type=str,   default="output",       help="The save name of the run")
args = parser.parse_args()

# Create the environment
env = gym.make("highway-v0")
obs = env.reset()

# Create the model
model = DQN('MlpPolicy', env,
            gamma=0.8,
            learning_rate=5e-4,
            buffer_size=40*1000,
            learning_starts=200,
            exploration_fraction=0.6,
            batch_size=128,
            verbose=1,
            tensorboard_log="output/dqn_models/logs/" + str(args.model_name))

# Train the model
model.load('output/dqn_models/models/' + str(args.model_name))

# Init a crash counter
crash_counter = []
total_runs = 2

for _ in range(total_runs):

    # Run the algorithm
    done = False
    obs = env.reset()
    while not done:
        # Predict
        action, _states = model.predict(obs)
        # Get reward
        obs, reward, done, info = env.step(action)
        # Render
        env.render()

    if info["crashed"] == True:
        crash_counter.append(1)
    else:
        crash_counter.append(0)

env.close()
print("The results:")
print("Crash array: {}".format(crash_counter))
print("Crash percentage: {}".format(np.sum(crash_counter) / len(crash_counter)))