"""
This is a helper script used to define the data collection and visualisations
for the model training (train.py)
"""

import matplotlib.pyplot as plt
import numpy as np


class TrainingStats:
    def __init__(self, window=50):
        self.rewards = []
        self.lengths = []
        self.losses = []
        self.epsilons = []
        self.q_values = []
        self.window = window
        self._loss_buf = []

    def log_step(self, loss=None, q_val=None):
        if loss is not None:
            self._loss_buf.append(loss)
        if q_val is not None:
            self.q_values.append(q_val)

    def log_episode(self, reward, length, epsilon):
        self.rewards.append(reward)
        self.lengths.append(length)
        self.epsilons.append(epsilon)
        self.losses.append(np.mean(self._loss_buf) if self._loss_buf else 0.0)
        self._loss_buf = []

    def rolling(self, data):
        return [
            np.mean(data[max(0, i - self.window) : i + 1]) for i in range(len(data))
        ]


def plot_training(stats, save_path="training_stats.png"):
    episodes = np.arange(1, len(stats.rewards) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("DQN Training Stats")

    # reward
    ax = axes[0, 0]
    ax.plot(episodes, stats.rewards, alpha=0.4, label="raw")
    ax.plot(
        episodes, stats.rolling(stats.rewards), label=f"rolling avg ({stats.window}ep)"
    )
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")
    ax.legend()

    # epsilon
    ax = axes[0, 1]
    ax.plot(episodes, stats.epsilons)
    ax.set_title("Epsilon")
    ax.set_xlabel("Episode")
    ax.set_ylim(0, 1.05)

    # loss
    ax = axes[1, 0]
    ax.plot(episodes, stats.losses, alpha=0.4, label="raw")
    ax.plot(episodes, stats.rolling(stats.losses), label="rolling avg")
    ax.set_title("Mean Episode Loss")
    ax.set_xlabel("Episode")
    ax.legend()

    # episode length
    ax = axes[1, 1]
    ax.plot(episodes, stats.lengths, alpha=0.4, label="raw")
    ax.plot(episodes, stats.rolling(stats.lengths), label="rolling avg")
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    N, eps, stats = 600, 1.0, TrainingStats()

    for i in range(N):
        p = i / N
        stats.log_step(
            loss=max(0.01, 1.5 * (1 - p) + np.random.rand() * 0.3),
            q_val=-1.0 + 4.0 * p + np.random.randn() * 0.5,
        )
        eps = max(0.05, eps * 0.995)
        stats.log_episode(
            -1.0 + 3.5 * p + np.random.randn() * 1.2,
            int(20 + 180 * p + np.random.randn() * 20),
            eps,
        )

    plot_training(stats)
