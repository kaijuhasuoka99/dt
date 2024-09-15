import numpy as np
import pickle
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from dt.dt import DecisionTransformer, DecisionTransformerConfig
from environments.atari_env import AtariEnv
from tqdm import tqdm
from collections import deque
import os
import argparse

from utils import softmax

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', '-e', type=str, default='BreakoutNoFrameskip-v4')
parser.add_argument('--save_weights_folder', type=str, default='./weights/')
parser.add_argument('--file_path', type=str, default='./datas/breakoutnoframeskip_v4.pkl')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--lr', type=float, default=6.0e-4)
parser.add_argument('--grad_norm_clip', type=float, default=1.0)
args = parser.parse_args()

env_name = args.env_name
file_path = args.file_path
save_weights_folder = args.save_weights_folder
epochs = args.epochs
batch = args.batch
lr = args.lr
grad_norm_clip = args.grad_norm_clip

class Dataset(Dataset):
    def __init__(self, states, actions, indices, seq_len):
        self.states = states
        self.actions = actions
        self.indices = indices
        self.seq_len = seq_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        states = self.states[idx:idx+self.seq_len]
        actions = self.actions[idx:idx+self.seq_len]
        states = torch.tensor(states, dtype=torch.float32) / 255.0
        actions = torch.tensor(actions, dtype=torch.long)

        return states, actions

env = AtariEnv(env_name, frame_skip=4, frame_stack=4)

with open(file_path, 'rb') as f:
    episodes = pickle.load(f)

config = DecisionTransformerConfig()
config.n_action = env.action_space
seq_len = config.seq_len
device = config.device
    
states = []
actions = []
rewards = []
is_terminals = []
for (state, action, reward, done, info) in episodes:
    states.append(state)
    actions.append(action)
    rewards.append(reward)
    is_terminals.append(done)

states = np.array(states)
actions = np.array(actions)
rewards = np.array(rewards)
is_terminals = np.array(is_terminals)

# Not include the series is not satisfied the seq_len.
done_indices = np.where(is_terminals == True)[0]
indices = np.arange(len(is_terminals))

mask = np.ones(indices.shape, dtype=bool)
for done_idx in done_indices:
    mask[max(0, done_idx+1-seq_len+1):done_idx+1] = False

indices = indices[mask]

dataset = Dataset(states, actions, indices, seq_len=seq_len)
dt = DecisionTransformer(config).to(device)
optimizer = torch.optim.Adam(dt.parameters(), lr=lr)

def run_epoch():
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch)
    with tqdm(total=len(dataloader)) as pbar:
        for x, a in dataloader:
            x = x.to(device)
            a = a.to(device)

            optimizer.zero_grad()
            a_hat = dt(x, a)
            loss = F.cross_entropy(a_hat.view(-1, a_hat.size(-1)), a.view(-1), ignore_index=-1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(dt.parameters(), grad_norm_clip)
            optimizer.step()
            pbar.set_description(f'epoch: {epoch:3d} loss: {loss:.4f}')
            pbar.update(1)

def evaluate(eval_episodes=20, max_episode_steps=80000):
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

    return sum(episode_returns) / len(episode_returns)
    
patience = 0
for epoch in range(1, epochs+1):
    dt.train()
    run_epoch()
    dt.eval()
    eval_return = evaluate()
    tqdm.write(f'eval return: {eval_return:.2f}')

    if epoch > 1:
        if eval_return < prev_eval_return:
            patience += 1
            tqdm.write(f'patience: {patience:3d}')
            if patience >= 5:
                break
        else:
            if not os.path.exists(save_weights_folder + str(epoch)):
                os.makedirs(save_weights_folder + str(epoch))
            torch.save(dt.state_dict(), save_weights_folder + str(epoch) + '/dt.pth')
            prev_eval_return = eval_return
    else:
        prev_eval_return = eval_return
        if not os.path.exists(save_weights_folder + str(epoch)):
            os.makedirs(save_weights_folder + str(epoch))
        torch.save(dt.state_dict(), save_weights_folder + str(epoch) + '/dt.pth')