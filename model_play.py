"""
This script will take a pretrained model and show it playing snake

NOTE: This won't run on mac because of how view thread perms work
on mac
"""

import gymnasium
import gymnasium_snake_game
import pygame
import torch
import torch.nn as nn
import numpy as np
from train import DQN, preprocess

model_path = "./models/snake_dqn.pth"
episodes = 5
board_w = 10
board_h = 10

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

pygame.init()
env = gymnasium.make(
    "Snake-v1",
    render_mode="human",
    width=board_w,
    height=board_h,
    max_step=500,
    dist_reward=0.0,
    food_reward=2.0,
    death_penalty=-1.0,
    living_bonus=0.1,
)
obs, _ = env.reset()
obs_shape = obs.shape
n_actions = env.action_space.n


model = DQN(obs_shape[-1], n_actions).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"loaded model from {model_path}")

for episode in range(episodes):
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done:
        with torch.no_grad():
            action = model(preprocess(obs)).argmax().item()

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    print(f"episode {episode + 1} | steps: {steps:4d} | reward: {total_reward:6.1f}")

env.close()
