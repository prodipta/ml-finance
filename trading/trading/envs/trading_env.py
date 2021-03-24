# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 14:02:26 2021

@author: prodi
"""

import gym
from gym import spaces

import numpy as np
import pandas as pd
import talib as ta

class TradingEnv(gym.Env):
    """
        Stock market environment.
        
        Args:
            `path (int)`: None.
            
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, lookback=100, max_period=20, test=120):
        df = pd.read_csv('../data/data.csv')
        df = df.set_index('Date')
        df.index = pd.to_datetime(df.index)
        self._input = df
        
        self._data = df.iloc[:-test]
        self._test = df.iloc[-test:]
        
        self.action_space = spaces.Discrete(2)
        
        self.nsteps = 0
        self.maxsteps = max_period
        self.lookback = lookback
        self.current_idx = lookback
        self.max_idx = len(self._data)-1
        
        self.position = False
        self.entry_price = None
        self.entry_date = None
        
        self._states = None
        self.reset()
    
    @property
    def state(self):
        if self._states:
            return pd.Series(self._states).values.reshape(
                    1,self.nstates)
        
    @property
    def nstates(self):
        if self._states:
            return len(self._states)
        
        return 0
    
    @property
    def nactions(self):
        return 2
    
    def _compute_states(self):
        start = self.current_idx - self.lookback
        end = self.current_idx
        df = self._data.iloc[start:end,:]
        states = {}
        
        states['roc1'] = ta.ROC(df.close, timeperiod=5).iloc[-1]
        states['roc2'] = ta.ROC(df.close, timeperiod=20).iloc[-1]
        states['rsi1'] = ta.RSI(df.close, timeperiod=14).iloc[-1]
        states['rsi2'] = ta.RSI(df.close, timeperiod=28).iloc[-1]
        states['cci1'] = ta.CCI(df.high, df.low, df.close, timeperiod=14).iloc[-1]
        states['cci2'] = ta.CCI(df.high, df.low, df.close, timeperiod=28).iloc[-1]
        states['adx1'] = ta.ADX(df.high, df.low, df.close, timeperiod=14).iloc[-1]
        states['adx2'] = ta.ADX(df.high, df.low, df.close, timeperiod=28).iloc[-1]
        states['atr'] = ta.ATR(df.high, df.low, df.close, timeperiod=14).iloc[-1]
        states['aroon1'] = ta.AROONOSC(df.high, df.low, timeperiod=14).iloc[-1]
        states['aroon2'] = ta.AROONOSC(df.high, df.low, timeperiod=28).iloc[-1]
        states['bop'] = ta.BOP(df.open, df.high, df.low, df.close).iloc[-1]
        states['cmo'] = ta.CMO(df.close, timeperiod=14).iloc[-1]
        states['close'] = df.close.iloc[-1]
        states['date'] = df.index[-1]
        states['position'] = self.position
        states['entry_price'] = self.entry_price
        states['entry_date'] = self.entry_date
        
        return states

    def step(self, action):
        self.current_idx += 1
        self.nsteps += 1
        reward = 0
        done = False
        
        states = self._compute_states()
        px = states['close']
        
        if self.position:
            reward = (px-self.entry_price)*(self.nsteps/self.maxsteps)
            
        if self.nsteps >= self.maxsteps:
            done = True
            self.nsteps = 0
            
        if self.current_idx >= self.max_idx:
            self.current_idx = np.random.randint(
                    self.lookback, len(self._data)-self.lookback)
            done = True
        
        if action == 1 and not self.position:
            self.entry_price = px
            self.entry_date = states['date']
            self.position = True
        elif action == 0 and self.position:
            self.position = False
            self.entry_price = None
            self.entry_date = None
            done = True
        elif done:
            self.position = False
            self.entry_price = None
            self.entry_date = None
        
        extra = {'last_position':states.pop('position')}
        extra['current_position'] = self.position
        extra['date'] = states.pop('date')
        extra['close'] = states.pop('close')
        extra['entry_price'] = states.pop('entry_price')
        extra['entry_date'] = states.pop('entry_date')
        
        self._states = states
            
        return self.state, round(reward,2), done, extra

    def reset(self):
        self.nsteps = 0
        self.position = False
        self.entry_price = None
        self.entry_date = None
        self.reward = 0
        self._states = None
        self.step(0)

    def render(self, mode='human'):
        print(f'position:{self.position}')

    def close(self):
        pass