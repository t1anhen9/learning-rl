import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Q网络
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.out(x)

# 经验回放
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        return (
            torch.tensor(np.array(state), dtype=torch.float32, device=device),
            torch.tensor(action, dtype=torch.int64, device=device),
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.tensor(np.array(next_state), dtype=torch.float32, device=device),
            torch.tensor(done, dtype=torch.float32, device=device)
        )

    def __len__(self):
        return len(self.buffer)

# 初始化环境与模型
env = gym.make("LunarLander-v2")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

q_network = QNetwork(state_size, action_size).to(device)
target_network = QNetwork(state_size, action_size).to(device)
target_network.load_state_dict(q_network.state_dict())

optimizer = optim.Adam(q_network.parameters(), lr=1e-3)
buffer = ReplayBuffer()
batch_size = 64
gamma = 0.99
epsilon = 1
epsilon_min = 0.001
epsilon_decay = 0.995
update_target_every = 10

rewards = []

# 训练循环
for episode in range(1000):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                q_values = q_network(state_tensor)
                action = torch.argmax(q_values).item()

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

    if episode % update_target_every == 0:
        target_network.load_state_dict(q_network.state_dict())

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)
    print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

# 绘制学习曲线
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Lunar Lander - DQN with GPU")
plt.savefig("reward_plot.png")
plt.show()
