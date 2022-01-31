import numpy as np
import utils

import copy

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def maxQ(self, s_prime):
        maxQ_val = 0
        for action_take in [utils.RIGHT, utils.LEFT, utils.DOWN, utils.UP]:
            if self.Q[s_prime][action_take] > maxQ_val:
                maxQ_val = self.Q[s_prime][action_take]
        return maxQ_val

    def find_action(self, s_prime):
        final_action = 0
        action_base = -1e10
        
        for action_take in [utils.RIGHT, utils.LEFT, utils.DOWN, utils.UP]:
            action_val = 0
            if (self.N[s_prime][action_take] < self.Ne and self._train):
                action_val = 1
            else:
                action_val = self.Q[s_prime][action_take]
            
            if (action_val > action_base):
                final_action = action_take
                action_base = action_val
        return final_action
        
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)
        
        if (self.s is not None and self.a is not None and self.train):
            self.N[self.s][self.a] += 1

        reward_val = -0.1
        if (points > self.points):
            reward_val = 1
        if (dead):
            reward_val = -1

        alpha_val = self.C / (self.C + self.N[self.s][self.a])
        if (self.s is not None and self.a is not None and self.train):
            maxQ_val = max(self.Q[s_prime])
            self.Q[self.s][self.a] += alpha_val * (reward_val + self.gamma * maxQ_val - self.Q[self.s][self.a])
            if (dead):
                self.reset()
                return 0

        action = self.find_action(s_prime)

        self.s = s_prime
        self.a = action
        self.points = points

        return action

    def generate_state(self, environment):
        headx, heady, mainbody, pelletx, pellety = environment

        if (headx - pelletx > 0):
            pellet_directionx = 1
        elif (headx - pelletx == 0):
            pellet_directionx = 0
        else:
            pellet_directionx = 2

        if (heady - pellety > 0):
            pellet_directiony = 1
        elif (heady - pellety == 0):
            pellet_directiony = 0
        else:
            pellet_directiony = 2

        adj_wallx = 0
        adj_wally = 0

        if (headx == utils.GRID_SIZE):
            adj_wallx = 1
        if (headx == (utils.DISPLAY_SIZE/utils.GRID_SIZE - 2) * utils.GRID_SIZE):
            adj_wallx = 2
        if (heady == utils.GRID_SIZE):
            adj_wally = 1
        if (heady == (utils.DISPLAY_SIZE/utils.GRID_SIZE - 2) * utils.GRID_SIZE):
            adj_wally = 2

        adj_bodyT = 0
        adj_bodyB = 0
        adj_bodyL = 0
        adj_body_R = 0

        for (cell_x, cell_y) in mainbody:
            if (cell_y == heady and cell_x == headx + utils.GRID_SIZE):
                adj_body_R = 1
            if (cell_y == heady and cell_x == headx - utils.GRID_SIZE):
                adj_bodyL = 1
            if (cell_x == headx and cell_y == heady - utils.GRID_SIZE):
                adj_bodyB = 1
            if (cell_x == headx and cell_y == heady + utils.GRID_SIZE):
                adj_bodyT = 1
        env_discrete = (pellet_directionx, pellet_directiony, adj_wallx, adj_wally,
                         adj_bodyT, adj_bodyB, adj_bodyL, adj_body_R)
        return env_discrete
