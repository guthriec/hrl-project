import gymnasium as gym
import gymnasium_robotics
import time
import numpy as np

class CarWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # --- Car Physics Constants ---
        self.dt = 0.1       # Simulation timestep (approx)
        self.L = 0.2         # Wheelbase length
        self.max_steer = 0.5 # Max steering angle (radians)
        self.max_speed = 2.0 

        # --- Internal State ---
        # We must track theta (heading) and velocity manually
        self.theta = 0.0
        self.velocity = 0.0

        # --- Action Space: [Steering, Acceleration] ---
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # --- Observation Space ---
        # We need to add cos(theta) and sin(theta) to the obs so the agent knows direction
        obs_space = env.observation_space["observation"]
        ach_space = env.observation_space["achieved_goal"]
        
        # Expanded size: Original Obs + 2 (for heading) + Goal Size
        # Note: We add 2 to dimensions for heading
        new_low  = np.concatenate([ach_space.low, obs_space.low, [-1, -1]])
        new_high = np.concatenate([ach_space.high, obs_space.high, [1, 1]])

        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=new_low, high=new_high, dtype=obs_space.dtype),
            "achieved_goal": ach_space,
            "desired_goal": env.observation_space["desired_goal"]
        })
        self.state_dim = int(np.prod(self.observation_space["observation"].shape))
        self.action_dim = int(np.prod(self.action_space.shape))

    def step(self, action):
        # 1. Parse Action (Steer, Throttle)
        steer = np.clip(action[0], -1, 1) * self.max_steer
        accel = np.clip(action[1], -1, 1)

        # 2. Update Car Physics (Kinematic Bicycle Model)
        # theta_new = theta + (v / L) * tan(delta) * dt
        self.theta += (self.velocity / self.L) * np.tan(steer) * self.dt
        self.theta %= (2 * np.pi) # Normalize angle

        # v_new = v + a * dt
        self.velocity += accel * self.dt * 10
        self.velocity = np.clip(self.velocity, -self.max_speed, self.max_speed)

        # 3. Apply to MuJoCo (Method 2)
        # Calculate global velocity vector
        vx = self.velocity * np.cos(self.theta)
        vy = self.velocity * np.sin(self.theta)

        # OVERRIDE the underlying physics state directly
        # qvel[0] is x-velocity, qvel[1] is y-velocity
        self.unwrapped.data.qvel[:2] = [vx, vy]

        # 4. Step environment with ZERO force
        # We send [0, 0] because we already forced the velocity above.
        # This lets MuJoCo handle collisions normally.
        obs, reward, terminated, truncated, info = self.env.step(np.zeros(2))

        # 5. Custom Reward & Obs Construction
        obs = self._get_obs(obs)
        
        distance = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        reward = -distance

        return obs, reward, terminated or truncated, info

    def reset(self, **kwargs):
        # Reset internal physics state
        self.theta = 0.0
        self.velocity = 0.0
        
        obs, info = self.env.reset(**kwargs)
        return self._get_obs(obs)

    def _get_obs(self, obs):
        # Create heading vector
        heading = np.array([np.cos(self.theta), np.sin(self.theta)], dtype=np.float32)
        
        # Structure: [Achieved Goal, Original Obs, Heading]
        obs["observation"] = np.concatenate([
            obs["achieved_goal"], 
            obs["observation"], 
            heading
        ])
        return obs
    

training_map =  [[1, 1, 1, 1, 1],
                    [1, 'g', 'r', 0, 1],
                    [1, 1, 1, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1]]

eval_map =  [[1, 1, 1, 1, 1],
                    [1, 'g', 'r', 'r', 1],
                    [1, 1, 1, 'r', 1],
                    [1, 'r', 'r', 'r', 1],
                    [1, 1, 1, 1, 1]]

# In the original paper, I believe they only used a single starting point but
# that makes learning a lot slower. Let's only eval using the single starting point but
# train with multiple goals.
def simple_car_maze_training():
    env = gym.make('PointMaze_UMaze-v3', 
                   continuing_task=False, 
                   #render_mode="human", 
                   maze_map=training_map)
    env = CarWrapper(env)
    return env

# Only 1 starting point
def simple_car_maze_eval():
    env = gym.make('PointMaze_UMaze-v3', 
                   continuing_task=False, 
                   #render_mode="human", 
                   maze_map=training_map)
    env = CarWrapper(env)
    return env