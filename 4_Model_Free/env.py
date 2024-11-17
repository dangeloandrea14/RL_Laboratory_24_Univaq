import utils
import numpy as np
import pandas as pd

class env:

    ## Here we define the constants of the environment
    LAST_STATE = 6
    FIRST_STATE = 0
    DISCOUNT_FACTOR = 0.5


    def __init__(self, starting_grid = np.full((LAST_STATE-FIRST_STATE+1), 1/(LAST_STATE-FIRST_STATE+1))):
        
        ## STATES AND ACTIONS
        self.states = [env.build_state(i) for i in range(env.FIRST_STATE,env.LAST_STATE+1)]

        ## STARTING STATE
        self.starting_grid = starting_grid
        self.starting_state = env.build_state(utils.select_first_state(self.starting_grid))
        self.current_state = self.starting_state

        ## We build the actions
        self.actions = []
        for action in env.actions:
            self.actions.append(env.build_action(env.actions[action], action))

        
        ## PROBABILITY MATRIX AND REWARDS

        self.probability_matrix, self.stateaction_dict = utils.build_probability_matrix(self.states, self.actions)
        self.probability_matrix_df = utils.build_df(self.probability_matrix, self.stateaction_dict)

        self.rewards, self.action_dict = utils.build_rewards(self.states, self.actions)
        self.rewards_df = utils.build_df(self.rewards, self.action_dict)
        
    
    rewards = {0:1000000, 6:10}


    def ask_state(state,action):
        if action == 'right':
            probs =  {env.build_state(state.number + 1) : 0.5, state:0.5} if state.number < env.LAST_STATE else {state:1}
            return np.random.choice(list(probs.keys()), p=list(probs.values()))
        if action == 'left': 
            probs = {env.build_state(state.number - 1) : 0.5, state:0.5} if state.number > env.FIRST_STATE else {state:1}
            return np.random.choice(list(probs.keys()), p=list(probs.values()))
        else:
            return env.build_state(state.number)

    ## ACTIONS
    def action_right(state):
        return {env.build_state(state.number + 1) : 0.5, state:0.5} if state.number < env.LAST_STATE else {state:1}

    def action_left(state):
        return {env.build_state(state.number - 1) : 0.5, state:0.5} if state.number > env.FIRST_STATE else {state:1}

    actions = {'left': action_left, 'right': action_right}

    def build_action(function, name):
        return utils.action(function, name)

    def build_state(state_number):
        action_list = []

        for action in env.actions:
            action_list.append( env.build_action( env.actions[action] ,action ) )

        rew = env.rewards[state_number] if state_number in env.rewards else 0

        return utils.state(state_number, action_list, rew)
    
    

