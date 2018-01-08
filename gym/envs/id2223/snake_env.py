import numpy as np
import gym
from gym import spaces, utils
from gym.utils import seeding

from gym.envs.id2223 import snake

NUM_ACTIONS = 4

class SnakeEnv(gym.Env,utils.EzPickle):
    def __init__(self, shape=(42,42), seed=None, num_apples=3):
        self.shape = shape + (1,)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low=0,high=3,shape=self.shape)
        self.num_apples = num_apples

        self.start_pos = (self.shape[0]//2, self.shape[1]//2)

        self._seed(seed)

    @property
    def board(self):
        return self.game.board.reshape(self.shape)

    def _step(self, action):
        assert self.game, "Cannot call env.step() before calling reset()"
        assert self.action_space.contains(action), "Invalid action given"

        score = self.game.score
        info = {}
        try:
            self.game.next(snake.dirs[action])
        except snake.GameOver as e:
            info['gameover!'] = str(*e.args)

        new_score = self.game.score

        reward = float(-10 if self.game.is_over else new_score - score)

        # (observation, reward, terminal, info) in accordance with gym api
        return (self.board, reward, self.game.is_over, info)

    def _reset(self):
        self.game = snake.game(np.zeros(self.shape[:2]),
                               self.start_pos,
                               apples=self.num_apples)
        return self.board

    #def _render(self, mode='human', close=False):
    #    raise NotImplementedError

    def _close(self):
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
