"""
This script defines andt rains a DQN model on the Snake Game through the
gymnasium_snake_game package by lychanl
"""

import gymnasium
import gymnasium_snake_game  # this is a package for snake by lychanl
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from visualize import TrainingStats, plot_training
import matplotlib.pyplot as plt
import numpy as np

# globals for training options
EPISODES = 6000
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
MEMORY_SIZE = 10_000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9995
TARGET_UPDATE = 100

# make compatible with cuda or metal, taken from pytorch docs
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")

env = gymnasium.make(
    "Snake-v1",
    render_mode=None,
    width=10,
    height=10,
    max_step=500,
    dist_reward=0.0,
    food_reward=10.0,  # bigger food reward
    death_penalty=-1.0,
    living_bonus=0.1,  # small reward each step for surviving
)
obs, info = env.reset()
obs_shape = obs.shape
n_actions = env.action_space.n
stats = TrainingStats(window=50)  # stat tracker and plotting


class DQN(nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * (obs_shape[0] // 2) * (obs_shape[1] // 2), 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def preprocess(obs):
    """Convert (H, W, C) uint8 numpy array to (1, C, H, W) float tensor."""
    x = torch.tensor(obs, dtype=torch.float32) / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    return x.to(device)


in_channels = obs_shape[-1]
policy_net = DQN(in_channels, n_actions).to(device)
target_net = DQN(in_channels, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=MEMORY_SIZE)
epsilon = EPSILON_START


def select_action(obs):
    if random.random() < epsilon:
        return env.action_space.sample()
    with torch.no_grad():
        return policy_net(preprocess(obs)).argmax().item()


def store(obs, action, reward, next_obs, done):
    memory.append((obs, action, reward, next_obs, done))


def train_step():
    if len(memory) < BATCH_SIZE:
        return None, None

    batch = random.sample(memory, BATCH_SIZE)
    obs_b, act_b, rew_b, nobs_b, done_b = zip(*batch)

    obs_t = torch.cat([preprocess(o) for o in obs_b])
    nobs_t = torch.cat([preprocess(o) for o in nobs_b])
    act_t = torch.tensor(act_b, dtype=torch.long, device=device)
    rew_t = torch.tensor(rew_b, dtype=torch.float32, device=device)
    done_t = torch.tensor(done_b, dtype=torch.float32, device=device)

    q_vals = policy_net(obs_t).gather(1, act_t.unsqueeze(1)).squeeze()
    with torch.no_grad():
        next_q = target_net(nobs_t).max(1).values
        targets = rew_t + GAMMA * next_q * (1 - done_t)

    loss = nn.functional.mse_loss(q_vals, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), q_vals.max().item()


for episode in range(EPISODES):
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done:
        action = select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        store(obs, action, reward, next_obs, done)
        loss, q_val = train_step()
        stats.log_step(loss=loss, q_val=q_val)

        obs = next_obs
        total_reward += reward
        steps += 1

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    stats.log_episode(total_reward, steps, epsilon)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if episode % 50 == 0:
        print(
            f"Episode {episode:4d} | Reward: {total_reward:6.1f} | Epsilon: {epsilon:.3f}"
        )

env.close()
torch.save(policy_net.state_dict(), "./models/snake_dqn.pth")
plot_training(stats, save_path="training_stats.png")
print("Done")
