import time
from robot_env import RobotEnv

# Create the environment
env = RobotEnv()

# Reset the environment to get the initial state
state = env.reset()

# Render the environment for a few seconds to visualize it
for _ in range(1000):
    # Just to keep the simulation running
    time.sleep(1.0 / 240.0)  # PyBullet runs at 240 Hz

# Close the environment
env.close()