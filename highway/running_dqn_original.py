import gym
import argparse
import highway_env

import numpy as np
from tqdm import tqdm
from stable_baselines import DQN

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="output", help="The save name of the run")
parser.add_argument('--episodes', type=int, default=1, help="The number of episodes you want to run")
parser.add_argument('--render', action='store_true')
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
total_runs = args.episodes

for e_count in tqdm(range(total_runs), desc="Episode"): 

    # Run the algorithm
    done = False
    obs = env.reset()
    while not done:
        # Predict
        action, _states = model.predict(obs)
        # Get reward
        obs, reward, done, info = env.step(action)
        if args.render:
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