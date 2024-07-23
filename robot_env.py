import pybullet as p
import pybullet_data
import numpy as np
import gym
from gym import spaces

class RobotEnv(gym.Env): # Class inherits from gym.Env, so we can use with OpenAI gym.
    def __init__(self):
        super(RobotEnv, self).__init__ # Calls the constructor of OpenAI gym, to ensure proper intialization.
        self.action_space = spaces.Box(-1, 1, (3,))
        self.observation_space = spaces.Box(-np.inf, np.inf, (14,))
        self.physics = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # Access PyBullet URDF files.
        p.setGravity(0, 0, -9.81)
        self.planeId = p.loadURDF("plane.urdf")
        self.robotId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
    
    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.planeId = p.loadURDF("plane.urdf")
        self.robotId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        self.goal_position = np.random.uniform(low=-0.5, high=0.5, size=(3,))
        return self._get_observation()
    
    def step(self, action):
        # Apply action to the robot arm
        for i in range(3):
            p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, action[i])
        p.stepSimulation()
        
        # Calculate reward and observation
        observation = self._get_observation()
        reward = self._calculate_reward(observation)
        done = self._is_done(observation)
        
        return observation, reward, done, {}
    
    def _get_observation(self):
        # Implement the observation logic here
        joint_states = p.getJointStates(self.robotId, range(7))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        return np.array(joint_positions + joint_velocities)
    
    def _calculate_reward(self, observation):
        # Implement the reward calculation logic here
        end_effector_position = p.getLinkState(self.robotId, 6)[0]
        distance_to_goal = np.linalg.norm(np.array(end_effector_position) - self.goal_position)
        reward = -distance_to_goal
        return reward
    
    def _is_done(self, observation):
        # Implement the done condition logic here
        end_effector_position = p.getLinkState(self.robotId, 6)[0]
        distance_to_goal = np.linalg.norm(np.array(end_effector_position) - self.goal_position)
        print(distance_to_goal)
        return (distance_to_goal < 0.05) or (distance_to_goal > 0.5)
    
    def close(self):
        p.disconnect()