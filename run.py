import os,sys
import random
import numpy as np
import matplotlib.pyplot as plt
from agents.qlearning_agent import QLearningAgent
from agents.policy import EpsGreedyQPolicy
from envs.grid_world import GridWorld
import ipdb
import copy

if __name__ == '__main__':
    # 三種類のエージェントの用意
    agent_names = [
            "no shaping", 
            "shaping_random",
            "shaping_V",
            ]

    # エージェント毎に実験を行う
    for agent_name in agent_names:
        iter_num = 3  # 試行回数
        results = []
        nb_episode = 500   #エピソード数
        for it in range(iter_num):
            grid_env = GridWorld() # grid worldの環境の初期化
            ini_state = grid_env.start_pos  # 初期状態（エージェントのスタート地点の位置）
            policy = EpsGreedyQPolicy(epsilon=0.1) # 方策の初期化。ここではε-greedy

            # Q学習エージェントの初期化
            if agent_name == "no shaping":
                agent = QLearningAgent(actions=np.arange(4), observation=ini_state, policy=policy)
            elif agent_name == "shaping_random":
                agent = QLearningAgent(actions=np.arange(4), observation=ini_state, policy=policy, is_reward_shaping=True, potential_fun="random")
            elif agent_name == "shaping_V":
                agent = QLearningAgent(actions=np.arange(4), observation=ini_state, policy=policy, is_reward_shaping=True, potential_fun="V")


            reward_history = []    # 評価用報酬の保存
            is_goal = False # エージェントがゴールしてるかどうか？
            for episode in range(nb_episode):
                episode_reward = [] # 1エピソードの累積報酬
                while(is_goal == False):    # ゴールするまで続ける
                    action = agent.act()    # 行動選択
                    state, reward, is_goal = grid_env.step(action)
                    agent.observe(state, reward)   # 状態と報酬の観測
                    episode_reward.append(reward)
                reward_history.append(np.sum(episode_reward)) # このエピソードの平均報酬を与える
                state = grid_env.reset()    #  初期化
                agent.observe_state(state)    # エージェントを初期位置に
                is_goal = False
            results.append(copy.deepcopy(reward_history))

        results = np.array(results)
        results = results.mean(axis=0)  # 試行毎の平均の保存

        # 結果のプロット
        plt.plot(np.arange(len(results)), results, label=agent_name)
        print("================================================")
        print(agent_name)
        # 最適な行動戦略の確認
        while(is_goal == False):    # ゴールするまで続ける
            agent.policy.eps = 0
            action = agent.act()    # 行動選択
            print("state:{}, action:{}".format(state, action))
            state, reward, is_goal = grid_env.step(action)
            agent.observe(state, reward, is_learn=False)   # 状態と報酬の観測
            episode_reward.append(reward)

    plt.xlabel("episode")
    plt.ylabel("accumulated reward")
    plt.ylim(-100, 100)
    plt.legend()
    plt.savefig("result.jpg")
    plt.show()
