import os
import gym
import argparse
import highway_env

import numpy as np
from stable_baselines import DQN

parser = argparse.ArgumentParser()
parser.add_argument('--model_name',             type=str,   default="output",       help="The save name of the run")
args = parser.parse_args()

# Make the output directory
if not os.path.exists('output/dqn_models/models/'):
    os.makedirs('output/dqn_models/models/')

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
model.learn(total_timesteps=int(5e4))
model.save('output/dqn_models/models/' + str(args.model_name))
model.load('output/dqn_models/models/' + str(args.model_name))

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

env.close()