import torch
import torch.optim as optim
from DQN.replay_buffer import ReplayBuffer
from DQN.dqn import DQN
import numpy as np
import robot_env  # Ensure robot_env.py is in the same directory or in the Python path

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
GAMMA = 0.99
TAU = 1e-3
LR = 1e-3
BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_EPISODES = 1000
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
UPDATE_EVERY = 4

# Initialize environment
env = robot_env.RobotEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
print(action_size)

# Initialize replay buffer and DQN agent
replay_buffer = ReplayBuffer(BUFFER_SIZE)
agent = DQN(state_size, action_size, GAMMA, TAU, LR)

# Training loop
epsilon = EPSILON_START
for episode in range(1, NUM_EPISODES + 1):
    state = env.reset()
    total_reward = 0

    while True:
        # Agent selects action using epsilon-greedy policy
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            action = agent.act(state, epsilon=0.0)  # Use the policy to select action

        # Take action and observe next state, reward, and done flag
        next_state, reward, done, _ = env.step(action)
        
        # Store transition in replay buffer
        replay_buffer.add(state, action, reward, next_state, done)
        
        # Update agent every UPDATE_EVERY steps if buffer is sufficiently filled
        if len(replay_buffer) > BATCH_SIZE and episode % UPDATE_EVERY == 0:
            batch = replay_buffer.sample(BATCH_SIZE)
            loss = agent.update(batch)
        
        # Accumulate total reward
        total_reward += reward
        state = next_state
        
        # Break if episode is done
        if done:
            break
    
    # Decay epsilon after each episode
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    
    # Print episode statistics
    print(f"Episode {episode}/{NUM_EPISODES} - Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

# Save trained model
torch.save(agent.model.state_dict(), 'dqn_model.pth')

# Close environment
env.close()