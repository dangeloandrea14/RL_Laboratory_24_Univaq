import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from netgraph import Graph,InteractiveGraph,ArcDiagram


class state:
    def __init__(self, number, actions, reward):
        self.number = number
        self.actions = actions
        self.reward = reward

class action:
    def __init__(self, function, name):
        self.function = function
        self.name = name

def build_probability_matrix(states, actions):

    matrix = np.zeros((len(states), len(states) * len(actions)))
    sa_dict = {}
    i = 0

    for state in states:
        for act in actions:
            sa_dict[(state.number, act.name)] = i
            i += 1

    for state in states:
        for action in state.actions:
            for next_state, ns_prob in action.function(state).items():   
                matrix[next_state.number, sa_dict[(state.number,action.name)]] = ns_prob

    return matrix, sa_dict


def build_rewards(states,actions):
    rewards = np.zeros( (len(states),len(actions)) )

    act_dict = {}
    i = 0

    for action in actions:
        act_dict[action.name] = i
        i += 1

    for state in states:
        for action in state.actions:
            rewards[state.number, act_dict[action.name]] += state.reward


    return rewards, act_dict

def build_df(matrix, stateaction_dict):
    return pd.DataFrame(matrix, columns = stateaction_dict.keys())

def select_next_move(state):
    action_list = state.actions
    action_probabilities = [action_list[action] for action in action_list]
    action = np.random.choice(list(action_list.keys()), p=action_probabilities)
    return action


def select_first_state(grid):
    pos = list(range(len(grid)))
    state = np.random.choice(pos, p=grid)
    return state

def plot_heatmap(result):
    
    cmap = plt.get_cmap('viridis')
    normed_values = (result - np.min(result)) / (np.max(result) - np.min(result))
    fig, ax = plt.subplots(figsize=(7, 1))
    im = plt.imshow(normed_values.reshape(1, -1), cmap=cmap, aspect='auto', extent=[0, len(result), 0, 1])

    for i, value in enumerate(result):
        plt.text(i + 0.5, 0.5, f'{value:.3f}', color='red', ha='center', va='center')

    plt.axis('off')

    return im

def return_function(episode, rewards, discount_factor):
    total = 0
    for i in range(len(episode)):
        total += rewards[episode[i]] * (discount_factor ** i)
    return total


def visualize_environment(environment):

    adj_matrix = environment.probability_matrix

    sources, targets = np.where(adj_matrix)
    
    weights = adj_matrix[sources, targets]
    edges = list(zip(sources, targets))

    node_positions = {}
    for i in range(0, len(environment.states)):
        node_positions[i] = (0.5, 0.5 + i * 0.1)

    edge_labels = dict(zip(edges, weights))

    fig, ax = plt.subplots()
    plt.ion()
    fig.set_size_inches(15, 15)
    ArcDiagram(edges, node_labels=True,  node_label_fontdict ={'size': 15},  edge_labels=edge_labels, edge_alpha=0.5, edge_width=0.3, edge_label_position=0.66, arrows=True, ax=ax)
    ax.invert_xaxis()
    plt.show()


def visualize_policy(environment, agent):

    G = agent.policy_graph(environment)
    toremove = [edge for edge in G.edges if G.edges[edge]['weight'] == 0]
    for edge in toremove:
        G.remove_edge(edge[0],edge[1])

    edge_labels = nx.get_edge_attributes(G,'weight')

    for key in edge_labels:
        edge_labels[key] = round(edge_labels[key],2)

    fig, ax = plt.subplots()
    ArcDiagram(G, node_labels=True, node_order=G.nodes, node_label_fontdict ={'size': 15}, edge_labels=edge_labels,edge_alpha=0.7, edge_width=0.8, arrows=True, ax=ax)
    plt.show()


def p_df(p):
    return pd.DataFrame(p)