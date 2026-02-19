import pandas as pd
import numpy as np
import gymnasium as gym
from gym_env import SalesNegotiationEnv
from tqdm import tqdm

class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # We redefine the observation space as a Box (continuous vector)
        # The shape remains the same (6 + max_round), but we use float32
        low = np.zeros(env.observation_space.shape, dtype=np.float32)
        high = np.array([max(s.nvec) for s in [env.observation_space]], dtype=np.float32).flatten()
        
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=100, # A safe upper bound for your round/count/action indices
            shape=env.observation_space.shape, 
            dtype=np.float32
        )

    def observation(self, observation):
        # Convert the MultiDiscrete array to a float32 array
        return np.array(observation, dtype=np.float32)

def evaluate_strategy(model, env, n_episodes=100, verbose = False):
    test_record = []
    round_count_record = []
    for _ in tqdm(range(n_episodes), desc="Evaluating Models", disable=not verbose):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        current_round = 0
        while not done:
            current_round += 1
            action = model.predict(np.expand_dims(obs, axis=0))[0]
            if obs[int(5 + obs[0])] == 4:
                if model.predict_value(np.expand_dims(obs, axis=0), 0) > model.predict_value(np.expand_dims(obs, axis=0), 2):
                    action = 0
                else:
                    action = 2
            elif bool(obs[4]):
                if model.predict_value(np.expand_dims(obs, axis=0), 0) > model.predict_value(np.expand_dims(obs, axis=0), 2):
                    action = 0
                else:
                    action = 2
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        test_record.append(episode_reward)
        round_count_record.append(current_round)

    return test_record, round_count_record