import utils
import numpy as np
import pandas as pd
from env import env
import networkx as nx

class agent:

    EPSILON = 0.0001

    def __init__(self,starting_policy,environment):

        self.policy = starting_policy
        self.policy_evaluation = agent.evaluate_policy(self,starting_policy,environment)
        self.easy_policy = agent.get_easy_policy(self)
        self.environment = environment
        self.actions = []
        for action in agent.actions:
            self.actions.append(agent.build_action(agent.actions[action], action))

        

    ## ACTIONS
    def action_right(state):
        return {env.build_state(state.number + 1) : 0.5, state:0.5} if state.number < env.LAST_STATE else {state:1}

    def action_left(state):
        return {env.build_state(state.number - 1) : 0.5, state:0.5} if state.number > env.FIRST_STATE else {state:1}

    actions = {'left': action_left, 'right': action_right}

    ## FUNCTIONS
    def build_action(function, name):
        return utils.action(function, name)
    

    #The following code just shrinks the policy for a more compact view:
    def get_easy_policy(self):
        pol = {}
        for k,v in self.policy.items():
            if v != 0:
                pol[k[0]] = k[1]
        return pol
    
    #STEPS
    def step(self):
        action_probabilities = [self.policy[(self.environment.current_state.number, action.name)] for action in self.actions]
        action = np.random.choice(list(self.actions), p=action_probabilities)

        return action
    
    def sample_episode(self, length):
        self.environment = env()
        next_state = self.environment.starting_state 
        episode = [next_state.number]

        while len(episode) < length:
            next_action = agent.step(self)
            next_state = next_action.function(next_state)

            next_state = np.random.choice(list(next_state.keys()), p=list(next_state.values()))
            self.environment.current_state = next_state
            episode.append(next_state.number)

        return episode

    def change_policy(self, new_policy):
        self.policy = new_policy
        self.policy_evaluation = agent.evaluate_policy(self,new_policy,self.environment)
        self.easy_policy = agent.get_easy_policy(self)


    def evaluate_policy(self, policy, environment):
        r_pi = [0 for i in range(len(environment.states))]

        for state in environment.states:
            for action in state.actions:
                r_pi[state.number] += policy[(state.number, action.name)] * environment.rewards_df.loc[state.number, action.name]

        

        p_pi = np.zeros( ( len(environment.states), len(environment.states) ) )

        for state in environment.states:
            for action in state.actions:
                for state_key,state_value in action.function(state).items():
                    p_pi[state_key.number, state.number] += policy[(state.number, action.name)] * state_value

        

        v_pi = {}
        v_pi[0] = r_pi
        gamma = environment.DISCOUNT_FACTOR

        pol = {}
        for k,v in policy.items():
            if v != 0:
                pol[k[0]] = k[1]

        v_pi[1] = []
        for state in environment.states:
            v_pi[1].append(0)
            summ = sum([  environment.probability_matrix_df.loc[s_prime.number, (state.number, pol[state.number])] * v_pi[0][s_prime.number] for s_prime in environment.states  ])
            v_pi[1][state.number] = environment.rewards_df.loc[state.number, pol[state.number]] + gamma * summ


        for i in range(2,100):
            v_pi[i] = []
            for state in environment.states:
                v_pi[i].append(0)
                summ = sum([  environment.probability_matrix_df.loc[s_prime.number, (state.number, pol[state.number])] * v_pi[i-1][s_prime.number] for s_prime in environment.states  ])
                v_pi[i][state.number] = environment.rewards_df.loc[state.number, pol[state.number]] + gamma * summ

        return v_pi[i]


    def policy_graph(self, environment):
        policy = self.policy
        
        G = nx.DiGraph()
        G.add_nodes_from([i for i in range(0, len(environment.states))])



        for state in environment.states:
            for action in state.actions:
                for state_key,state_value in action.function(state).items():
                    G.add_edge(state.number, state_key.number, weight=policy[(state.number, action.name)]*state_value)
                    
        return G

    
