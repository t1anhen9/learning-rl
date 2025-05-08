import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 策略网络


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

# 收集轨迹


def collect_trajectory(env, policy, max_steps=1000):
    state, _ = env.reset()
    log_probs = []
    rewards = []
    total_reward = 0

    for _ in range(max_steps):
        state_tensor = torch.FloatTensor(state).to(device)
        probs = policy(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        total_reward += reward

        if terminated or truncated:
            break
        state = next_state

    return log_probs, rewards, total_reward

# 折扣回报


def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns).float().to(device)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def compute_returns_with_baseline(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns).float().to(device)

    # 使用 baseline（这里是平均值）来降低方差
    baseline = returns.mean()
    advantage = returns - baseline  # G_t - baseline

    return advantage


# reward 曲线绘图
def plot_rewards(all_rewards, window=10):
    plt.figure(figsize=(10, 5))
    smoothed = [np.mean(all_rewards[max(0, i - window):i + 1])
                for i in range(len(all_rewards))]
    plt.plot(all_rewards, label="Episode Reward")
    plt.plot(smoothed, label=f"Smoothed ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("REINFORCE on LunarLander-v2 (Gymnasium)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("reinforce_rewards.png")
    plt.close()

# 主训练函数


def train_reinforce(env_name="LunarLander-v2", episodes=1000, gamma=0.99, lr=1e-3):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    all_rewards = []

    for episode in range(episodes):
        log_probs, rewards, total_reward = collect_trajectory(env, policy)
        # returns = compute_returns(rewards, gamma)
        returns = compute_returns(rewards, gamma)

        loss = -torch.sum(torch.stack(log_probs) * returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}, Return: {total_reward:.2f}")
            plot_rewards(all_rewards)

    env.close()


if __name__ == "__main__":
    train_reinforce()