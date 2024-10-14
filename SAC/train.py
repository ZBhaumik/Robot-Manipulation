import gym
import torch
from sac import SoftActorCritic
from replay_buffer import ReplayBuffer
import numpy as np

# Initialize environment
env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Initialize SAC agent and replay buffer
sac_agent = SoftActorCritic(state_dim, action_dim)
replay_buffer = ReplayBuffer(100000)

# Training parameters
episodes = 1000
epochs = 50
batch_size = 128
start_timesteps = 1000
episode_reward_history = []
expl_noise = 0.1

for episode in range(episodes):
    # We do this because state also returns debug info that we do not use in SAC.
    state = env.reset()[0]
    episode_reward = 0
    done = False

    while not done:
        action = sac_agent.sample_action(state)
        #action = action + np.random.normal(0, expl_noise, size=action_dim)
        action = action.clip(env.action_space.low, env.action_space.high)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        if(len(replay_buffer) > start_timesteps):
            sac_agent.train(replay_buffer, batch_size)
    
    episode_reward_history.append(episode_reward)
    print(f"Episode: {episode}, Reward: {episode_reward}")

    if (episode + 1) % 10 == 0:
        torch.save(sac_agent.policy.state_dict(), "sac_actor.pth")
        torch.save(sac_agent.q1.state_dict(), "sac_critic1.pth")
        torch.save(sac_agent.q2.state_dict(), "sac_critic2.pth")

# Close the environment
env.close()