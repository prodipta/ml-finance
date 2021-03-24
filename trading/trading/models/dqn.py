# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 16:45:18 2021

@author: prodi
"""
import numpy as np
import pandas as pd
import random
import tqdm
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

class Agent(object):
    
    def __init__(self, nstates, nactions, memsize=100, nbatch=5,
                 gamma=0.8, eps=0.5, nu=0.01):
        self.nstates = nstates
        self.nactions = nactions
        self.nbatch = nbatch
        
        self.gamma = gamma
        self.eps = eps
        self.nu = nu
        
        self.memory = deque(maxlen=memsize)
        self.qmodel = self._create_model()
        self.rmodel = self._create_model()
        self.update_model_epoch = 50
        
        self.niter = 0
    

    def _create_model(self):
        model = Sequential()
        model.add(Dense(8, input_dim=self.nstates, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(self.nactions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.nu))
#        print(model.summary())
#        plot_model(model, 
#                   to_file='model_plot.png', 
#                   show_shapes=True, 
#                   show_layer_names=True)
        return model
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))
        
    def replay(self):
        if len(self.memory) < self.nbatch:
            return
        
        X_train = np.zeros((self.nbatch, self.nstates))
        Y_train = np.zeros((self.nbatch, self.nactions))
        batch = random.sample(self.memory, self.nbatch)
        
        for i, sample in enumerate(batch):
            state, action, reward, new_state, done = sample
            if done:
                target=reward
            else:
#                target = reward + self.gamma*np.max(
#                        self.qmodel.predict(new_state)[0])
                target = reward + self.gamma*self.rmodel.predict(
                        new_state)[0][np.argmax(self.qmodel.predict(
                                new_state)[0])]
            X_train[i] = state
            Y_train[i] = self.qmodel.predict(state)
            #Y_train[i] = self.rmodel.predict(state)
            Y_train[i][action] = target
        
        if self.niter % self.update_model_epoch==0:
            self.rmodel.set_weights(self.qmodel.get_weights())
        
        self.qmodel.train_on_batch(X_train, Y_train)
        #self.qmodel.fit(X_train, Y_train, epochs=1, verbose=False)
        
    def act(self, state, explore=False):
        if explore:
            return random.randrange(self.nactions)
        
        q = self.qmodel.predict(state)
        return np.argmax(q[0])
        
    def fit(self, env, episodes=500, verbose=False):
        rewards = np.zeros(episodes)
        idx = np.zeros(episodes, dtype=np.int64)
        env.reset()
        for i in tqdm.tqdm(range(episodes)):
            done = False
            env.reset()
            state = env.state
            while not done:
                self.niter += 1
                
                if np.random.uniform() < self.eps:
                    action = self.act(state, explore=True)
                else:
                    action = self.act(state)
                
                next_state, reward, done, extra = env.step(action)
                self.remember(state, action, reward, next_state, done)
                self.replay()
                state = next_state
                
                if done:
                    rewards[i] = reward
                    idx[i] = extra['date'].value
                    if verbose:
                        print(f'batch:{i}, reward is {reward}.')
                    break
        idx = pd.to_datetime(idx)
        return pd.Series(rewards, index=idx)

