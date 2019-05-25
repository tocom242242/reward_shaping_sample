import numpy as np
import copy
import ipdb
import random

class QLearningAgent():
    """
       qlearning エージェント。
       Reward shaping_の設定は引数で行う。
       is_reward_shapingでReward Shapingを行うかの決定、
       potential_funでポテンシャル関数の設定を行う。
    """
    def __init__(self, alpha=0.2, 
                 policy=None, 
                 gamma=0.99, 
                 actions=None, 
                 observation=None, 
                 alpha_decay_rate=None, 
                 epsilon_decay_rate=None, 
                 is_reward_shaping=False, 
                 potential_fun=None):

        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.reward_history = []
        self.name = "qlearning"
        self.actions = actions
        self.gamma = gamma
        self.alpha_decay_rate = alpha_decay_rate
        self.epsilon_decay_rate = epsilon_decay_rate
        self.state = str(observation)
        self.ini_state = str(observation)
        self.previous_state = str(observation)
        self.previous_action_id = None
        self.q_values = self._init_q_values()
        self.is_reward_shaping = is_reward_shaping
        self.potential_fun = potential_fun

        if is_reward_shaping:
            if self.potential_fun == "random":
                self.phi = {}
                self.phi[self.state] = random.random()*10

    def _init_q_values(self):
        """
           Q テーブルの初期化
        """
        q_values = {}
        q_values[self.state] = np.repeat(0.0, len(self.actions))
        return q_values

    def init_state(self):
        """
            状態の初期化 
        """
        self.previous_state = copy.deepcopy(self.ini_state)
        self.state = copy.deepcopy(self.ini_state)
        return self.state

    def init_policy(self, policy):
        self.policy = policy

    def act(self, q_values=None, step=0):
        action_id = self.policy.select_action(self.q_values[self.state])
        self.previous_action_id = action_id
        action = self.actions[action_id]
        return action

    def observe(self, next_state, reward, is_learn=True):
        """
            次の状態と報酬の観測 
        """
        self.observe_state(next_state)
        if is_learn:
            self.learn(reward)

    def observe_state(self, next_state):
        next_state = str(next_state)
        if next_state not in self.q_values: # 始めて訪れる状態であれば
            self.q_values[next_state] = np.repeat(0.0, len(self.actions))

        if self.is_reward_shaping and self.potential_fun == "random":
            self.phi[next_state] = random.random()*10

        self.previous_state = copy.deepcopy(self.state)
        self.state = next_state

    def learn(self, reward, is_finish=True, step=0):
        """
            報酬の獲得とQ値の更新 
        """
        self.reward_history.append(reward)
        self.q_values[self.previous_state][self.previous_action_id] = self.compute_q_value(reward)

    def compute_q_value(self, reward):
        """
            Q値の更新 
        """
        q = self.q_values[self.previous_state][self.previous_action_id] # Q(s, a)
        max_q = max(self.q_values[self.state]) # max Q(s')
        if self.is_reward_shaping:
            if self.potential_fun == "V":
                # phi(s) = sum_a Q(s, a)
                phi1 = np.sum([v for v in self.q_values[self.previous_state]]) 
                phi2 = np.sum([v for v in self.q_values[self.state]]) 
            elif self.potential_fun == "random":
                phi1 = self.phi[self.previous_state]
                phi2 = self.phi[self.state] 

            reward = reward + self.gamma*phi2 - phi1
        # Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        updated_q = q + (self.alpha * (reward + (self.gamma*max_q) - q))
        return updated_q
