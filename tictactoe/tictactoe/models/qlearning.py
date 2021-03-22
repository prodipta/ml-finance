# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 16:45:18 2021

@author: prodi
"""
import numpy as np
import pandas as pd
import tqdm
import warnings

def reset_Q(env):
    actions = env.action_space.moves
    Q = pd.DataFrame(np.zeros((0,len(actions))),
                 columns=[str(action) for action in actions], 
                 index=[])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        setattr(Q, 'actions', actions)
    return Q

def act(env, state, Q, random=False):
    if random or state not in Q.index:
        return env.action_space.sample()
    
    idx = Q.loc[state,:].values.argmax()
    if Q.actions[idx] in env.action_space:
        return Q.actions[idx]
    
    action = env.action_space.sample()
    best = get_q(Q, state, action)
    for move in env.action_space.moves:
        if get_q(Q, state, move) > best:
            best = get_q(Q, state, move)
            action = move
    
    return action

def get_q(Q, state, action):
    if str(action) not in Q.columns or state not in Q.index:
        return 0
    return Q.loc[state, str(action)]

def get_max_q(Q, state):
    if state not in Q.index:
        return 0
    return Q.loc[state,:].max()

def update(Q, state, action, value):
    if str(action) not in Q.columns:
        return
    
    if state not in Q.index:
        idx = Q.index.tolist()
        idx.append(state)
        Q.loc[len(Q.index)]=np.zeros(len(Q.columns))
        Q.index = idx
    Q.loc[state, str(action)] = value
    
def fit(env, episodes=10000, gamma=0.2, eps=0.5, nu=0.6, 
          decay=0.005, verbose=False):
    # decay parameter is ignored.
    env.reset()
    Q = reset_Q(env)
    for i in tqdm.tqdm(range(episodes)):
        done = False
        env.reset()
        state = env.state
        while not done:
            if np.random.uniform() < eps:
                action = act(env, state, Q, random=True)
            else:
                action = act(env, state, Q)
            
            next_state, reward, done, _ = env.step(action)
            current_q = get_q(Q, state, action)
            max_q = get_max_q(Q, next_state)
            value = (1-nu)*current_q + nu*(reward + gamma*max_q)
            update(Q, state, action, value)
            state = next_state
            
            if done:
                if verbose:
                    print(f'batch:{i}, winner is {env.winner}.')
                    env.render()
                break
            
    return Q
            
def test(env, Q, episodes=100, verbose=True, random=False):
    wins = losses = draws = 0
    
    for i in tqdm.tqdm(range(episodes)):
        done = False
        state = env.state
        env.reset()
        while not done:
            if random:
                action = env.action_space.sample()
            else:
                action = act(env, state, Q)
            
            next_state, reward, done, _ = env.step(action)
            state = next_state
        
        if env.winner == 0:
            draws += 1
        elif env.winner == 1:
            wins += 1
        elif env.winner == 2:
            losses += 1
        else:
            raise ValueError('illegal game, something went wrong.')
    
    wins = round(100*(wins/episodes), 2)
    losses = round(100*(losses/episodes), 2)
    draws = round(100*(draws/episodes), 2)
    print(f'played {episodes}, wins%:{wins}, losses%:{losses}, draws%:{draws}')