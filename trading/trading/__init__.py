# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 13:59:06 2021

@author: prodi
"""

from gym.envs.registration import register

register(
    id='tradingenv-v0',
    entry_point='trading.envs:TradingEnv',
)

