import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SalesNegotiationEnv(gym.Env):
    def __init__(
        self, 
        labor_cost=0.2, 
        incentive_A_cost=2, 
        incentive_B_cost=5, 
        incentive_C_cost=7, 
        profit = 10, 
        exit_penalty=-2,
        max_round=30,
        exit_start_round=5,
        p_factor=2.7):
        super(SalesNegotiationEnv, self).__init__()
        
        self.L = labor_cost
        self.incentive_cost = {'A': incentive_A_cost, 'B': incentive_B_cost, 'C': incentive_C_cost, 'D': 0}
        self.V = profit
        self.C_exit = exit_penalty
        self.max_round = max_round
        self.p_factor = p_factor
        self.previous_objection_resolved = True

        self.multiplier ={"A": profit/(profit-incentive_A_cost), "B": profit/(profit-incentive_B_cost), "C": profit/(profit-incentive_C_cost), "D": 0}

        # 0: Persuade, 1: Incentive, 2: Closing
        self.action_space = spaces.Discrete(3)
        
        # Observation Space:
        # [current round, A/B/C resolved, Incentive Used, Previous Objection Resolved] + [Action History for all max_rounds] + [Topic History]
        obs_shape = [self.max_round + 1, 2, 2 , 2, 2, 2] + [4] * (self.max_round + 1) + [5] * (self.max_round + 1)
        self.observation_space = spaces.MultiDiscrete(obs_shape)

        self.exit_start_round = exit_start_round


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Latent Variables Determination and Patience
        self.D = np.random.uniform(0.8, 1)
        self.P = np.random.uniform(0.6,0.9)
        
        self.round = 0
        self.resolved = {"A": False, "B": False, "C": False}
        
        self.full_action_history = np.zeros(self.max_round + 1, dtype=np.int32)
        self.topic_history = np.zeros(self.max_round + 1, dtype=np.int32)
        self.current_obj = self._pick_new_objection()
        self.topic_history[0] = self._topic_to_index(self.current_obj)
        self.previous_objection_resolved = True
        
        return self._get_obs(), {
            "latent_D": self.D, 
            "latent_P": self.P
        }

    def step(self, action):
        current_step_index = self.round
        self.round += 1
        reward = -self.L
        terminated = False
        truncated = False
        
        if current_step_index < len(self.full_action_history):
            self.full_action_history[current_step_index] = action + 1
        if self.round > self.exit_start_round:
            if np.random.random() > self.P:
                return self._get_obs(), self.C_exit + reward, True, False, {}

        # Closing
        if action == 2: 
            return self._get_obs(), reward, True, False, {}
        
        # Persuade
        elif action == 0: 
            if np.random.random() < ((1 - self.D)/self.p_factor):
                self.previous_objection_resolved = True
                return self._get_obs(), self.V + reward, True, False, {}
            elif np.random.random() < 0.5:
                # Current Objection resolved
                self._mark_resolved(self.current_obj)
                self.current_obj = self._pick_new_objection()
                return self._get_obs(), reward, False, False, {}
            else:
                # Current Objection not resolved
                self.previous_objection_resolved = False
                return self._get_obs(), reward, False, False, {}

        # Incentive
        elif action == 1: 
            if self.incentive_used:
                # End the episode immediately with a exit penalty
                return self._get_obs(), self.C_exit, True, False, {"reason": "double_incentive_violation"}
            self.incentive_used = True
            if self.current_obj == "D":
                # End the episode immediately with a exit penalty (D has no incentive, violation of rule)
                return self._get_obs(), self.C_exit, True, False, {"reason": "D_incentive_violation"}
            elif np.random.random() < (1 - self.D) * self.multiplier[self.current_obj]:
                # Success = Instant Sale
                self.previous_objection_resolved = True
                return self._get_obs(), self.V + reward - self.incentive_cost[self.current_obj], True, False, {}
            else:
                # If incentive failed, the objection is resolved, pick a new objection
                self._mark_resolved(self.current_obj)
                self.current_obj = self._pick_new_objection()
                return self._get_obs(), reward, False, False, {}

        if self.round >= self.max_round:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        base_obs = [
            min(self.round, self.max_round),
            int(self.resolved["A"]), int(self.resolved["B"]), int(self.resolved["C"]),
            int(self.incentive_used),
            int(self.previous_objection_resolved)]
        if self.round <= self.max_round:
            self.topic_history[self.round] = self._topic_to_index(self.current_obj)
        return np.concatenate([base_obs, self.full_action_history, self.topic_history]).astype(np.int32)

    def _pick_new_objection(self):
        # A, B, C can only be picked if not resolved. D is infinite.
        pool = [k for k, v in self.resolved.items() if not v] + ["D"]
        # Force the choice to be a standard Python string
        self.current_obj = str(np.random.choice(pool))
        self.previous_objection_resolved = True
        self.incentive_used = False
        return self.current_obj

    def _mark_resolved(self, obj):
        # Ensure obj is a string to avoid NumPy KeyErrors
        obj_str = str(obj)
        if obj_str in self.resolved:
            self.resolved[obj_str] = True
        
    def _topic_to_index(self, topic):
        return {
            "None": 0,
            "A": 1,
            "B": 2,
            "C": 3,
            "D": 4,
        }.get(topic, 0)
    
    def _index_to_topic(self, index):
        return {
            0: "None",
            1: "A",
            2: "B",
            3: "C",
            4: "D",
        }.get(index, "None")

    def unpack_obs(self, obs):
        state = {}
        state["current_round"] = obs[0]
        state["resolved"] = {
            "A": bool(obs[1]),
            "B": bool(obs[2]),
            "C": bool(obs[3]),
        }
        state["incentive_used"] = bool(obs[4])
        state["previous_objection_resolved"] = bool(obs[5])
        state["action_history"] = obs[6: 6+self.max_round + 1]
        state["topic_history"] = obs[6+self.max_round + 1: 6+self.max_round + 1+self.max_round + 1]
        return state

