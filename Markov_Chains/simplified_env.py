import utils
import numpy as np


class env:
    LAST_STATE = 6
    FIRST_STATE = 0
    DISCOUNT_FACTOR = 0.5


    def __init__(self, starting_grid = np.full((LAST_STATE-FIRST_STATE+1), 1/(LAST_STATE-FIRST_STATE+1))):
        self.states = [i for i in range(env.FIRST_STATE,env.LAST_STATE+1)]
        self.starting_grid = starting_grid
        self.starting_state = utils.select_first_state(self.starting_grid)
        self.probability_matrix = utils.build_probability_matrix_states(self.states, env.probabilities)
        print("Environment Ready.")        

    '''    probabilities = {0: {1: 0.4, 0 : 0.6},
                     1: {0: 0.4, 2: 0.4, 1 : 0.2},
                    2: {1: 0.4, 3: 0.4, 2 : 0.2},
                    3: {2: 0.4, 4: 0.4, 3 : 0.2},
                    4: {3: 0.4, 5: 0.4, 4 : 0.2},
                    5: {4: 0.4, 6: 0.4, 5 : 0.2},
                    6: {5: 0.4, 6 : 0.6}}

    '''
    
    
    probabilities = {0: {1: 0.4, 0 : 0.6},
                     1: {0: 0.4, 2: 0.4, 1 : 0.2},
                    2: {1: 0.4, 3: 0.4, 2 : 0.2},
                    3: {2: 0.9, 4: 0.1, 3 : 0.0},
                    4: {3: 0.4, 5: 0.4, 4 : 0.2},
                    5: {4: 0.4, 6: 0.4, 5 : 0.2},
                    6: {5: 0.4, 6 : 0.6}}
    

    def sample_episode(self, length, random_first_state = False):

        state = utils.select_first_state(self.starting_grid) if random_first_state else self.starting_state
        
        episode = [state]

        while len(episode) < length:
            next_state = utils.select_next_state(state, env.probabilities)
            state = next_state
            episode.append(state)

        return episode

