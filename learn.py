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
                    help='file path for saved model (default: model.pt)')

args = parser.parse_args()

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def eval(dqn, eval_env, episodes):
    scores = []
    for e in range(episodes):
        done = False
        score = 0.0
        state = eval_env.reset()
        while not done:
            action = dqn.act(state)
            state, reward, done, _ = eval_env.step(action)
            score += reward

        scores.append(score)

    return np.mean(scores)


BATCH_SIZE = 64
warmup_steps = 256
total_steps = 1000000
epsilon_min = 0.05
epsilon = 1.0
# # epsilon_decay = 1-1e-5
# epsilon_decay = 1-1e-5
epsilon_fn = lambda x:epsilon_min if x>total_steps/2 else 1.0 - (1-epsilon_min)/(total_steps/2)*x


env = gym.make(args.env)
eval_env = gym.make(args.env)
state = env.reset()

# dqn = DQN(env.observation_space.shape[0], env.action_space.n, device='cuda:0')
dqn = DQN(env.observation_space.shape[0], env.action_space.n, device='cpu')
# dqn.load(args.path, eval=False)

scores = []
losses = []
score = 0.0
loss = float('nan')

best_score = 0

start_time = time.time()

for step in range(1000000):
    if step<warmup_steps or random.random()<epsilon:
        action = env.action_space.sample()
    else:
        action = dqn.act(state)

    next_state, reward, done, _ = env.step(action)
    score += reward

    dqn.store_transition(state, action, reward, next_state, done)

    state = next_state
    if done:
        state = env.reset()
        scores.append(score)
        score = 0.0

    
    if step>=warmup_steps:
        loss = dqn.train_step(mb_size=BATCH_SIZE)
        losses.append(loss)
    

    if step%10==0:
        dqn.update_target()

    if step%5000==0:
        avg_score = np.mean(scores[-5000:])
        evaluation = eval(dqn, eval_env, 10)
        print(f'step: {step}, avg. score: {avg_score:.4f}, eps: {epsilon:.5f}, time: {time.time()-start_time:.2f}, loss: {np.mean(losses[-5000:]):.4f}, eval: {evaluation:.2f}')
        start_time = time.time()
        if evaluation>=best_score:
            dqn.save(args.path)
            best_score = evaluation

    epsilon = epsilon_fn(step)
    if epsilon<0.1:
        epsilon = 0.1

# avg_score = np.mean(scores[-1000:])
# if avg_score>=best_score:
#     dqn.save('model.ph')
#     best_score = avg_score
