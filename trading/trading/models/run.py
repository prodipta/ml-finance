# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 23:54:28 2021

@author: prodi
"""
import numpy as np
import pandas as pd
import tqdm

from trading.envs import TradingEnv
from trading.models.dqn import Agent

env = TradingEnv()
agent = Agent(env.nstates, env.nactions)
nepisodes = 100
            
# evaluate the agent, we use training data, so around (1-eps) random
rewards = agent.fit(env, nepisodes)
rewards.cumsum().plot()

# evaluate a random strategy
env2 = TradingEnv()
rewards_rand = np.zeros(nepisodes)
idx = np.zeros(nepisodes, dtype=np.int64)
for i in tqdm.tqdm(range(nepisodes)):
    env2.reset()
    done = False
    while not done:
        action = np.random.randint(0,2)
        state, reward, done, extra = env2.step(action)
        if done:
            rewards_rand[i] = reward
            idx[i] = extra['date'].value

idx = pd.to_datetime(idx)
rewards_rand = pd.Series(rewards_rand, index=idx)
rewards_rand.cumsum().plot()

# evaluate the ddqn strategy
env3 = TradingEnv()
rewards_ddqn = np.zeros(nepisodes)
idx = np.zeros(nepisodes, dtype=np.int64)
for i in tqdm.tqdm(range(nepisodes)):
    env3.reset()
    done = False
    state = env3.state
    while not done:
        action = agent.act(state)
        state, reward, done, extra = env3.step(action)
        if done:
            rewards_ddqn[i] = reward
            idx[i] = extra['date'].value

idx = pd.to_datetime(idx)
rewards_ddqn = pd.Series(rewards_ddqn, index=idx)
rewards_ddqn.cumsum().plot()