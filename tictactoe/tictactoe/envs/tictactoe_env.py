# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 14:02:26 2021

@author: prodi
"""

import gym
from gym import spaces

import numpy as np

class Moves(spaces.Space):
    """
        Action space for tic-tac-toe game. The action space (available 
        moves) are the cells in the board that are still left unmarked.
        
        Args:
            ``moves (list)``: list of (row, column) - moves available.
            
    """
    def __init__(self, moves):
        self._moves = moves
        self.np_random = None
        self.shape = None
        self.dtype = None
        self.seed()
        
    @property
    def moves(self):
        return self._moves
        
    def sample(self):
        if len(self._moves) < 1:
            raise ValueError('no valid moves left.')
        
        idx = np.random.choice(range(len(self._moves)))
        return self._moves[idx]
    
    def contains(self, x):
        return x in self._moves
    
    def __len__(self):
        return len(self._moves)

class TictactoeEnv(gym.Env):
    """
        Tick-tac-toe game environment.
        
        Args:
            `player (int)`: Player - the first mover if equals 1.
            
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, player=1):
        if player not in [1,2]:
            raise ValueError('player must be 1 (first-mover) or 2.')
            
        self._player1 = player
        self._player2 = 2 if player==1 else 1
        self._blank = 0
        
        self.reset()
        
    @property
    def action_space(self):
        rows, cols = np.where(self._board==self._blank)
        moves = [e for e in zip(rows, cols)]
        self._action_space = Moves(moves)
        return self._action_space
    
    @property
    def state(self):
        state = self._board.flatten()
        state = '|'.join([str(i) for i in state])
        return state
    
    @property
    def winner(self):
        return self._winner
        
    def _lines(self):
        for row in self._board:yield row  
        for colum in self._board.T:yield colum
        yield self._board.diagonal()
        yield np.fliplr(self._board).diagonal()

    def _check_winner(self):
        for line in self._lines():
            if np.all(line==self._player1):
                self._winner = self._player1
                return
            elif np.all(line==self._player2):
                self._winner = self._player2
                return
            elif np.count_nonzero(line==self._blank) > 1:
                return
            elif len(self.action_space) > 1:
                return
        # we have a tie
        self._winner = 0
    
    def _process_move(self, player, move):
        if self.winner is not None:
            return
        
        self._board[move[0],move[1]] = player
        self._check_winner()
        self._history.append((player, move, self.winner))
        self._next_player = self._player1 \
                if self._player2==player else self._player2

    def _make_move(self, player=None, action=None):
        if player:
            if player != self._next_player :
                raise ValueError(
                        f'player {player} is playing out of turn')
        else:
            player = self._next_player
        
        if not action:
            action = self.action_space.sample()
        
        if action not in self.action_space:
            raise ValueError('illegal move, cell already marked.')
        
        self._process_move(player, action)

    def step(self, action):
        if self._player1 != self._next_player:
            raise ValueError('out-of-turn play.')
            
        reward = 0
        done = True
        
        self._make_move(self._player1, action)
        if self.winner is None:
            self._make_move()
        
        if self.winner == self._player1:
            reward = 100
        elif self.winner == 0:
            reward = 0
        elif self.winner == self._player2:
            reward = -100
        else:
            done = False
            
        state = self._board.flatten()
        state = '|'.join([str(i) for i in state])
            
        return state, reward, done, {}

    def reset(self):
        self._board = np.full((3,3),self._blank)
        self._next_player = self._player1 \
            if self._player1==1 else self._player2
        self._winner = None
        self._history = []
        
        if self._player2 == self._next_player:
            self._make_move()

    def render(self, mode='human'):
        value = ''
        for idx, line in enumerate(self._board):
            value = value + '{:^5}|{:^5}|{:^5}'.format(str(line[0]),str(line[1]),str(line[2])) + '\n'
            if idx != 2:value = value + '-----+-----+-----\n'
        print(value)

    def close(self):
        pass