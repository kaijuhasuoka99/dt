import numpy as np
import torch
from dt.dt import DecisionTransformer, DecisionTransformerConfig
from environments.atari_env import AtariEnv
from collections import deque
import argparse

from utils import softmax

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', '-e', type=str, default='BreakoutNoFrameskip-v4')
parser.add_argument('--load_weights_folder', type=str, default='./weights/15/')
args = parser.parse_args()

env_name = args.env_name
load_weights_folder = args.load_weights_folder

env = AtariEnv(env_name, frame_skip=4, frame_stack=4, render=True)

config = DecisionTransformerConfig()
config.n_action = env.action_space
seq_len = config.seq_len
device = config.device
dt = DecisionTransformer(config).to(device)
dt.load_state_dict(torch.load(load_weights_folder + 'dt.pth', weights_only=True))
dt.eval()

max_episode_steps = 80000
eval_episodes = 20

episode_returns = []
for e in range(eval_episodes):
    next_state = env.reset()
    reward = 0.0
    done = False
    states = deque(maxlen=seq_len)
    actions = deque(maxlen=seq_len-1)
    episode_return = 0.0
    for s in range(max_episode_steps):
        state = next_state
        states.append(state)
        logits = dt.infer(np.array(states), np.array(actions))
        p = softmax(logits)
        
        action = np.random.choice(len(p), p=p)

        next_state, reward, done, info = env.step(action)
        episode_return += reward

        actions.append(action)
        if done:
            break
    episode_returns.append(episode_return)

print(f'eval return: {sum(episode_returns) / len(episode_returns):.2f}')