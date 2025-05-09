import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Q网络定义
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden=64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size)
        )

    def forward(self, x):
        return self.net(x)


# 经验回放
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(states, dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.int64, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(next_states, dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)


# 训练 DQN
def train_dqn(env, q_network, target_network, episodes=1000, batch_size=64, gamma=0.99,
              epsilon=1.0, epsilon_min=0.001, epsilon_decay=0.995, lr=1e-3,
              target_update_freq=10):

    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    buffer = ReplayBuffer()
    rewards_per_episode = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                    action = q_network(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                states, actions, rewards_, next_states, dones = buffer.sample(batch_size)

                q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    next_q_values = target_network(next_states).max(1)[0]
                    targets = rewards_ + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

        print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    return rewards_per_episode


# 绘图
def plot_rewards(rewards, save_path="reward_plot.png"):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN on LunarLander-v2")
    plt.savefig(save_path)
    # plt.show()


# 评估 + 录视频
def evaluate_dqn(model, video_dir="videos", episodes=5):
    os.makedirs(video_dir, exist_ok=True)

    eval_env = gym.wrappers.RecordVideo(
        gym.make("LunarLander-v2", render_mode="rgb_array"),
        video_folder=video_dir,
        episode_trigger=lambda episode_id: True
    )

    for ep in range(episodes):
        state, _ = eval_env.reset()
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                action = model(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        print(f"[Eval Episode {ep}] Total Reward: {total_reward:.2f}")

    eval_env.close()
    print(f"视频已保存到 {video_dir}/")


# 主函数
def main():
    env = gym.make("LunarLander-v2")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    q_net = QNetwork(state_size, action_size).to(device)
    target_net = QNetwork(state_size, action_size).to(device)
    target_net.load_state_dict(q_net.state_dict())

    rewards = train_dqn(env, q_net, target_net)
    plot_rewards(rewards)
    evaluate_dqn(q_net)


if __name__ == "__main__":
    main()
