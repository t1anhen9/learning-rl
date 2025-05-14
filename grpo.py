import gymnasium as gym
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from datetime import datetime

# 使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        shared = self.shared(x)
        logits = self.actor(shared)
        value = self.critic(shared)
        return logits, value


def collect_trajectories(env, model, steps, gamma=0.99):
    obs = env.reset()[0]
    obs_list, actions, rewards, dones, values, log_probs = [], [], [], [], [], []

    for _ in range(steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        logits, value = model(obs_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()

        next_obs, reward, done, truncated, info = env.step(action.item())
        obs_list.append(obs)
        actions.append(action.item())
        rewards.append(reward)
        dones.append(done)
        values.append(value.item())
        log_probs.append(dist.log_prob(action).item())

        obs = next_obs
        if done or truncated:
            obs = env.reset()[0]

    return {
        "obs": np.array(obs_list),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "dones": np.array(dones),
        "values": np.array(values),
        "log_probs": np.array(log_probs)
    }


def compute_advantages(data, gamma=0.99, lam=0.95):
    rewards, values, dones = data["rewards"], data["values"], data["dones"]
    advantages = np.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(rewards) else 0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
    returns = advantages + values
    return advantages, returns


def grpo_update(model, optimizer, data, advantages, returns, old_log_probs, kl_beta=1.0):
    obs = torch.tensor(data["obs"], dtype=torch.float32).to(device)
    actions = torch.tensor(data["actions"], dtype=torch.int64).to(device)
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(device)

    logits, values = model(obs)
    dist = Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)
    ratio = torch.exp(new_log_probs - old_log_probs)

    kl_div = torch.mean(old_log_probs - new_log_probs)
    surrogate = ratio * advantages
    loss_actor = -torch.mean(surrogate - kl_beta * kl_div)
    loss_critic = F.mse_loss(values.squeeze(), returns)

    loss = loss_actor + 0.5 * loss_critic

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def evaluate_and_record(model, video_dir, episode=1, render=False):
    eval_env = gym.make("LunarLander-v2", render_mode="rgb_array")
    eval_env = gym.wrappers.RecordVideo(
        eval_env,
        video_folder=video_dir,
        episode_trigger=lambda x: True,
        name_prefix=f"eval"
    )

    model.eval()
    for ep in range(episode):
        obs = eval_env.reset()[0]
        done, total_reward = False, 0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(obs_tensor)
                action = torch.argmax(logits, dim=-1).item()
            obs, reward, done, truncated, _ = eval_env.step(action)
            total_reward += reward

        print(f"[Evaluation] Episode {ep + 1}: Reward = {total_reward:.2f}")
    eval_env.close()
    model.train()


def train():
    env = gym.make("LunarLander-v2")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    video_output = f"videos/lunarlander_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(video_output, exist_ok=True)

    for epoch in range(1, 501):
        data = collect_trajectories(env, model, steps=2048)
        advantages, returns = compute_advantages(data)
        grpo_update(model, optimizer, data, advantages, returns, data["log_probs"])

        avg_reward = np.sum(data["rewards"])
        print(f"Epoch {epoch}, Total Reward: {avg_reward:.2f}")

        if epoch % 50 == 0:
            print("Running evaluation...")
            evaluate_and_record(model, video_output)

    env.close()
    torch.save(model.state_dict(), "grpo_lunarlander_model.pth")
    print("✅ Training complete. Model saved.")


if __name__ == "__main__":
    train()
