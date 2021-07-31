import os
import gym
import argparse
import highway_env_v2

import numpy as np

from stable_baselines import DQN
from misc.highway_config import HighwayEnvironmentConfig

parser = argparse.ArgumentParser()
parser.add_argument('--environment_vehicles',   type=int,   default=15,             help="total_number of vehicles in the environment")
parser.add_argument('--model_name',             type=str,   default="output",       help="The save name of the run")
parser.add_argument('--policy',                 type=int,   default=1,              help="Can 1 - (MlpPolicy), 2 - (LnMlpPolicy), 3 - (CnnPolicy), 4 - (LnCnnPolicy)")
parser.add_argument('--gamma',                  type=float, default=0.99,           help="The DQN discount factor")
parser.add_argument('--learning_rate',          type=float, default=5e-4,           help="The DQN learning rate for adam optimizer")
parser.add_argument('--buffer_size',            type=int,   default=5e4,            help="The DQN size of the replay buffer")
parser.add_argument('--learning_starts',        type=int,   default=1e3,            help="How many steps of the model to collect transitions for before learning starts in the DQN")
parser.add_argument('--exploration_fraction',   type=float, default=0.1,            help="The fraction of entire training period over which the exploration rate is annealed in the DQN")
parser.add_argument('--batch_size',             type=int,   default=32,             help="The size of a batched sampled from replay buffer for training in the DQN")
args = parser.parse_args()

# Make an output folder
if not os.path.exists('output/dqn_models/models/'):
    os.makedirs('output/dqn_models/models/')

# Get the policy
policy = ""
if args.policy == 1:
    policy = "MlpPolicy"
elif args.policy == 2:
    policy = "LnMlpPolicy"
elif args.policy == 3:
    policy = "CnnPolicy"
elif args.policy == 4:
    policy = "LnCnnPolicy"
else:
    exit()
    
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
model = DQN(policy=policy,
            env=env,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            exploration_fraction=args.exploration_fraction,
            batch_size=args.batch_size,
            verbose=1,
            tensorboard_log="output/dqn_models/logs/" + str(args.model_name))


model.learn(total_timesteps=int(5e6))
model.save('output/dqn_models/' + str(args.model_name))
model.load('output/dqn_models/' + str(args.model_name))

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
