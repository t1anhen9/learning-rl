import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_rewards(all_rewards, window=10):
    plt.figure(figsize=(10, 5))
    smoothed = [np.mean(all_rewards[max(0, i - window):i + 1])
                for i in range(len(all_rewards))]
    plt.plot(all_rewards, label="Episode Reward")
    plt.plot(smoothed, label=f"Smoothed ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("A2C on LunarLander-v2 (Gymnasium)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("a2c_rewards.png")
    plt.close()


class A2CPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        probs = self.policy_net(state)
        value = self.value_net(state)
        return probs, value.squeeze(-1)


def collect_trajectory(env, policy, t_max=2048):
    state, _ = env.reset()
    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

    for _ in range(t_max):
        state_tensor = torch.FloatTensor(state).to(device)
        probs, value = policy(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()

        next_state, reward, terminated, truncated, _ = env.step(action.item())

        states.append(state_tensor)
        actions.append(action)
        log_probs.append(dist.log_prob(action).detach())
        rewards.append(reward)
        dones.append(float(terminated or truncated))
        values.append(value.detach())

        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    return states, actions, log_probs, rewards, dones, values


def compute_gae(rewards, dones, values, gamma=0.99, lam=0.95):
    returns = []
    advantages = []
    gae = 0
    values = values + [torch.tensor(0.0).to(device)]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])

    adv = torch.tensor(advantages).float().to(device)
    ret = torch.tensor(returns).float().to(device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret


def a2c_update(policy, optimizer, states, actions, log_probs, advantages, returns,
               epochs=1, batch_size=64):
    states = torch.stack(states).to(device)
    actions = torch.stack(actions).to(device)

    for _ in range(epochs):
        indices = np.arange(len(states))
        np.random.shuffle(indices)
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            s_batch = states[batch_idx]
            a_batch = actions[batch_idx]
            adv_batch = advantages[batch_idx]
            ret_batch = returns[batch_idx]

            probs, values = policy(s_batch)
            dist = Categorical(probs)
            entropy = dist.entropy().mean()
            logp = dist.log_prob(a_batch)

            actor_loss = -(logp * adv_batch).mean()
            critic_loss = nn.MSELoss()(values, ret_batch)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_a2c(env_name="LunarLander-v2", episodes=1000, t_max=2048, gamma=0.99, lr=3e-4):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = A2CPolicy(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    all_rewards = []
    ep_rewards = []

    for episode in range(episodes):
        states, actions, log_probs, rewards, dones, values = collect_trajectory(
            env, policy, t_max)
        advs, returns = compute_gae(rewards, dones, values, gamma)

        a2c_update(policy, optimizer, states,
                   actions, log_probs, advs, returns)

        total_reward = sum(rewards)
        ep_rewards.append(total_reward)
        all_rewards.append(np.mean(ep_rewards[-10:]))

        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode+1}, Average Reward: {all_rewards[-1]:.2f}")
            plot_rewards(ep_rewards)

    env.close()


if __name__ == "__main__":
    train_a2c()
