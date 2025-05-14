import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm

# ===== 设备选择 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== 策略网络 =====
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)  # 直接输出 logits

# ===== 采样轨迹 =====
def sample_trajectory(env, policy, max_steps=1000, min_reward=None):
    obs, _ = env.reset()
    obs_list, act_list, logp_list = [], [], []
    total_reward = 0

    for _ in range(max_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        logits = policy(obs_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())

        obs_list.append(obs_tensor)
        act_list.append(action)
        logp_list.append(logp)
        total_reward += reward
        obs = next_obs

        if terminated or truncated:
            break

    # 跳过过短或极差的轨迹
    if min_reward is not None and total_reward < min_reward:
        return None

    return {
        "observations": torch.stack(obs_list),
        "actions": torch.stack(act_list),
        "log_probs": torch.stack(logp_list),
        "reward": total_reward
    }

# ===== DPO 损失函数 =====
def dpo_loss(traj_a, traj_b, beta=1.0):
    logp_a = traj_a["log_probs"].sum()
    logp_b = traj_b["log_probs"].sum()
    logits = beta * torch.stack([logp_a, logp_b])
    pref_logp = torch.log_softmax(logits, dim=0)[0]  # a 更优
    return -pref_logp

# ===== 环境与策略初始化 =====
env = gym.make("LunarLander-v2")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy = PolicyNet(state_dim, action_dim).to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-4)

# ===== 训练参数 =====
episodes = 3000
batch_size = 8
beta = 1.0
min_reward = -200  # 最低 reward 筛选，防止非常失败的轨迹进入 DPO
use_pretrain = True
pretrain_steps = 5000

# ===== 可选 REINFORCE 预训练 =====
if use_pretrain:
    print("🔧 Running REINFORCE warm-up...")
    for step in tqdm(range(pretrain_steps)):
        traj = sample_trajectory(env, policy)
        if traj is None:
            continue
        loss = -traj["log_probs"].sum() * traj["reward"] / 100.0  # 简单加权
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ===== 主训练循环（DPO）=====
print("🚀 Starting DPO training...")
for epoch in tqdm(range(episodes)):
    pairs = []
    while len(pairs) < batch_size:
        traj1 = sample_trajectory(env, policy, min_reward=min_reward)
        traj2 = sample_trajectory(env, policy, min_reward=min_reward)
        if traj1 is None or traj2 is None:
            continue

        if traj1["reward"] >= traj2["reward"]:
            traj_a, traj_b = traj1, traj2
        else:
            traj_a, traj_b = traj2, traj1
        pairs.append((traj_a, traj_b))

    losses = []
    for traj_a, traj_b in pairs:
        loss = dpo_loss(traj_a, traj_b, beta=beta)
        losses.append(loss)

    total_loss = torch.stack(losses).mean()

    if torch.isnan(total_loss):
        print("⚠️ Loss became NaN. Stopping training.")
        break

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, DPO loss: {total_loss.item():.3f}")

env.close()

# ===== 视频保存（仅测试一次）=====
from gymnasium.wrappers import RecordVideo

# 创建环境并包装视频记录器
env = gym.make("LunarLander-v2", render_mode="rgb_array")
video_dir = "./videos"  # 替换为你希望保存视频的路径
env = RecordVideo(
    env, 
    video_dir, 
    episode_trigger=lambda episode_id: True,  # 每一局都录
    name_prefix="dpo_lander_final"
)

num_episodes = 5  # 你想玩的次数

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        logits = policy(obs_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample().item()

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

env.close()
print(f"🎬 所有视频已保存到：{video_dir}")