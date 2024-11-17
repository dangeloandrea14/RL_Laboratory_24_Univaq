import utils
import numpy as np
import pandas as pd
import networkx as nx
from env import env
from tqdm import tqdm

class agent:

    EPSILON = 0.0001
    class env_representation:
        DISCOUNT_FACTOR = 0.5
        def __init__(self, states, actions, rewards):
            self.states = states if type(states) == list else [states]
            self.actions = actions
            self.rewards, self.action_dict = utils.build_rewards(self.states, self.actions)
            self.rewards_df = utils.build_df(self.rewards, self.action_dict)
            self.current_state = self.states[0]


    def __init__(self, policy_function, starting_state = 3):

        self.starting_state = env.build_state(starting_state)
        self.policy_function = policy_function
        self.policy = {}
        self.actions = []
        for action in agent.actions:
            self.actions.append(agent.build_action(agent.actions[action], action))
            self.policy[(self.starting_state.number, action)] = 1 if action == policy_function(self.starting_state) else 0


        self.easy_policy = agent.get_easy_policy(self)

        self.my_env = agent.env_representation(self.starting_state, self.actions, {self.starting_state.number : self.starting_state.reward})

        
    
    ## ACTIONS
    def action_right(state):
        return env.ask_state(state, 'right')

    def action_left(state):
        return env.ask_state(state, 'left')

    actions = {'left': action_left, 'right': action_right}

    ## FUNCTIONS
    def build_action(function, name):
        return utils.action(function, name)
    
    def reset_state(self):
        self.my_env.current_state = self.starting_state

    #The following code just shrinks the policy for a more compact view:
    def get_easy_policy(self):
        pol = {}
        for k,v in self.policy.items():
            if v != 0:
                pol[k[0]] = k[1]
        return pol
    
    def q(self,state, action, agent):
        if action in state.actions:
            p = self.environment.probability_matrix_df[(state.number,action.name)].to_list()

            return self.environment.rewards_df.loc[state.number, action.name] + self.environment.DISCOUNT_FACTOR * sum([  p[next_state.number] * agent.policy_evaluation[next_state.number] for next_state in self.environment.states])
        
    #STEPS
    def step(self):
        action_probabilities = [self.policy[(self.my_env.current_state.number, action.name)] for action in self.actions]
        action = np.random.choice(list(self.actions), p=action_probabilities)

        return action
    

    def make_step(self, action):
        self.my_env.current_state = action.function(self.my_env.current_state)

        if self.my_env.current_state.number not in [state.number for state in self.my_env.states]:
            self.my_env.states.append(self.my_env.current_state)

        newpolicy = self.policy.copy()
        for action in self.actions:
            newpolicy[(self.my_env.current_state.number, action.name)] = 1 if action.name == self.policy_function(self.my_env.current_state) else 0


        self.change_policy(newpolicy)

        return self.my_env.current_state
    

    def sample_episode(self, length, return_actions=False, first_state=None, first_action=None, random_first_state = False):
        
        next_state = self.starting_state if first_state == None else first_state 
        if random_first_state:
            next_state = np.random.choice(self.my_env.states)
        actions = []
        episode = [next_state.number]

        if first_action != None:
            actions.append(first_action)
            next_state = first_action.function(next_state)
            episode.append(next_state.number)
    
            #self.current_state = next_state

        
        while len(episode) < length:
            if next_state.number not in [s.number for s in self.my_env.states]:
                self.my_env.states.append(next_state)
                for action in self.actions:
                    self.policy[(next_state.number, action.name)] = 0
                self.policy[(next_state.number, self.policy_function(next_state))] = 1

            action_probabilities = [self.policy[(next_state.number, action.name)] for action in self.actions]
            action = np.random.choice(list(self.actions), p=action_probabilities)
            actions.append(action)
            next_state = action.function(next_state)
            episode.append(next_state.number)

        actions.append(agent.step(self))

        self.reset_state()

        return episode if not return_actions else (episode, actions)
    

    def change_policy(self, new_policy):
        self.policy = new_policy
        self.easy_policy = agent.get_easy_policy(self)


    def compute_returns(self, episode, index):
        if index == None:
            return 0
        episode = episode[index:]
        sum_returns = 0
        for i in range(0,len(episode)):
            sum_returns += env.build_state(episode[i]).reward * (self.my_env.DISCOUNT_FACTOR ** i)

        return sum_returns

    def evaluate_policy_first_visit_montecarlo(self, max_episodes):
        self.reset_state()
        n = {state.number:0 for state in self.my_env.states}
        g = {state.number:0 for state in self.my_env.states}
        v = {state.number:0 for state in self.my_env.states}

        for i in range(max_episodes):
            episode = agent.sample_episode(self, 10)
            visited = []
            self.reset_state()
            for state in episode:
                if state not in visited:
                    visited.append(state)
                    n[state] += 1
                    g[state] += agent.compute_returns(self, episode, episode.index(state))
                    v[state] = g[state]/n[state]

        return v

    def evaluate_policy_every_visit_montecarlo(self, max_episodes):
        self.reset_state()
        n = {state.number:0 for state in self.my_env.states}
        g = {state.number:0 for state in self.my_env.states}
        v = {state.number:0 for state in self.my_env.states}

        for i in range(max_episodes):
            episode = agent.sample_episode(self, 10)
            self.reset_state()
            for j in range(len(episode)):
                n[episode[j]] += 1
                g[episode[j]] += agent.compute_returns(self, episode, j)
                v[episode[j]] = g[episode[j]]/n[episode[j]]

        return v


    def evaluate_policy_incremental(self, max_episodes):
        self.reset_state()
        v = {state.number:0 for state in self.my_env.states}
        n = {state.number:0 for state in self.my_env.states}

        for i in range(max_episodes):
            episode = agent.sample_episode(self, 10)
            self.reset_state()
            for j in range(len(episode)):
                n[episode[j]] += 1
                alpha = 1/n[episode[j]]
                temp = v[episode[j]]
                v[episode[j]] = temp + alpha * (agent.compute_returns(self, episode, j) - temp)

        return v



    def improve_policy(self):
        n = {(state.number, action.name): 0 for state in self.my_env.states for action in self.actions}
        g = {(state.number, action.name): 0 for state in self.my_env.states for action in self.actions}
        q = {(state.number, action.name): 0 for state in self.my_env.states for action in self.actions}
        known_states = [state.number for state in self.my_env.states]

        for i in range(100):
            for state in self.my_env.states:
                for action in self.actions:
                    episode, actions = agent.sample_episode(self, 10, return_actions=True, first_state = state, first_action = action)
                    self.reset_state()

                    #check if all states are known
                    for state_no in episode:
                        if state_no not in known_states:
                            known_states.append(state_no)
                            for action in self.actions:
                                n[(state_no, action.name)] = 0
                                g[(state_no, action.name)] = 0
                                q[(state_no, action.name)] = 0

                    ## COMPUTATION OF Q-VALUES
                    for j in range(len(episode)):
                        n[(episode[j], actions[j].name)] += 1
                        g[(episode[j], actions[j].name)] += agent.compute_returns(self, episode, j)
                        q[(episode[j], actions[j].name)] = g[(episode[j], actions[j].name)]/n[(episode[j], actions[j].name)]
                        

        print(q)
        ## POLICY IMPROVEMENT
        
        for state_no,ac in q.keys():

            if state_no not in [s.number for s in self.my_env.states]:
                self.my_env.states.append(env.build_state(state_no))

            max_action = None
            max_value = -np.inf
            for action in self.actions:
                if q[(state_no, action.name)] > max_value:
                    max_value = q[(state_no, action.name)]
                    max_action = action.name


            newpolicy = self.policy.copy()
            for action in self.actions:
                newpolicy[(state_no, action.name)] = 1 if action.name == max_action else 0

            self.change_policy(newpolicy)

        return self.policy


    ##TODO: INCREMENTAL POLICY EVALUATION

    def policy_graph(self, environment):

        policy = self.policy
        
        G = nx.DiGraph()
        G.add_nodes_from([i for i in [s.number for s in environment.states]])

        for state in environment.states:
            for action in state.actions:
                for state_key,state_value in action.function(state).items():
                    if state_key.number in [s.number for s in environment.states]:
                        G.add_edge(state.number, state_key.number, weight=policy[(state.number, action.name)]*state_value)
                    
        return G

    
