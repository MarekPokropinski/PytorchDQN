import gym
import random
from dqn import DQN
import numpy as np
import os
import time
import argparse

parser = argparse.ArgumentParser(description='Run deep q learning')

parser.add_argument('--env', dest='env', default="CartPole-v1",
                    help='gym environment (default: CartPole-v1)')

parser.add_argument('--path', dest='path', default="model.pt",
                    help='file path to saved model (default: model.pt)')

args = parser.parse_args()

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

warmup_steps = 1000
epsilon = 1.0
# epsilon_decay = 1-1e-5
epsilon_decay = 1-5e-5


env = gym.make(args.env)


# dqn = DQN(env.observation_space.shape[0], env.action_space.n, device='cuda:0')
dqn = DQN(env.observation_space.shape[0], env.action_space.n, device='cpu')
dqn.load(args.path, eval=False)

for episode in range(100):
    done = False
    state = env.reset()
    env.render()
    score = 0

    while not done:
        if random.random()<0.00:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)
        state, reward, done, _ = env.step(action)
        env.render()
        score += reward

    print(f'episode {episode:02d}, score: {score:.4f}')
