# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 23:54:28 2021

@author: prodi
"""

import gym
from tictactoe.models.qlearning import fit, test

env = gym.make('tictactoe-v0')
Q = fit(env)
test(env, Q)

