import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import networkx as nx
import pulp
from collections import Counter, defaultdict

from scipy.stats import poisson # for clique problem
from networkx.algorithms import bipartite # for clique problem
import random # for the demo
import gc # for memory management associated with the demo
import time # performance evaluation

def generate_transport_problem(
    n_nodes = 50,
    n_item_types = 5,
    max_degree = 5, # This should be fun
    ba_edges = 2, #barabasi-albert tuning parameter
    perc_sparsity = .9,
    supply_demand_balance = .5,
    inv_scalar = 100
    ):
    """
    Generate a transport problem on a small-world graph
    """
    attr_d = defaultdict(Counter)

    for i in range(n_item_types):
        for j in range(n_nodes):
            if random.random() < perc_sparsity: # Does this node have non-zero inventory for this item?
                continue
            s = 1
            if random.random() < supply_demand_balance: # What is the sign on the item's inventory?
                s = -1
            attr_d['item'+str(i)][str(j)] = s*int(inv_scalar * random.random())

    g = nx.barabasi_albert_graph(n_nodes, ba_edges)
    g = nx.relabel_nodes(g, {n:str(n) for n in range(n_nodes)})

    for _,d in g.nodes(data=True):
        d['max_degree'] = max_degree

    # Set all edges to default weight of '1'
    for u,v,d in g.edges(data=True):
        d['cost'] = 1 # I should play with this

    for k,v in attr_d.items():
        d = dict(v)
        d.update({a:0 for a in g.nodes() if a not in d.keys()}) # Explicitly add node attributes for zero inventory nodes
        nx.set_node_attributes(g, k, d)

    return g, attr_d
    

def generate_clique_mcf_problem(
    n_nodes = 50,
    n_comm = 10,
    n_item_types = 5,
    max_degree = 5, 
    prob_non_zero_inv = .1,
    supply_demand_balance = .5,
    max_item_inv = 100,
    max_comm = 5,
    avg_comm_per_node = 2, # poisson lambda
    edge_capacity = 300   # we'll see if I end up using this. Don't want the system to be over complicated
    ):
    """
    Generate an MCF problem on a graph of overlapping cliques.
    """
    
    node_communities = {} #defaultdict(list)
    community_nodes  = defaultdict(list)
    attr_d = defaultdict(Counter)
    
    g_bipart = nx.Graph()
    for n in range(n_nodes):
        # determine community assignments
        k =0 
        while k==0 or k>max_comm:
            k = poisson.rvs(avg_comm_per_node, size=1)
        comm = np.random.randint(n_comm, size=k)
        node_communities[str(n)] = comm
        for c in comm:
            community_nodes[c].append(n)
            g_bipart.add_edge(str(n),'comm'+str(c))
        
        # Add inventory
        for i in range(n_item_types):       
            if random.random() > prob_non_zero_inv: # Does this node have non-zero inventory for this item?
                continue
            s = 1
            if random.random() < supply_demand_balance: # What is the sign on the item's inventory?
                s = -1
            attr_d['item'+str(i)][str(n)] = s*random.randint(1,max_item_inv)

    
    g        = bipartite.projected_graph(g_bipart, [str(n) for n in range(n_nodes)])
    g_agents = bipartite.weighted_projected_graph(g_bipart, [str(n) for n in range(n_nodes)])
    g_comm   = bipartite.weighted_projected_graph(g_bipart, ['comm'+str(c) for c in range(n_comm)])

    
    ## Scavenged from generate_transport_problem()
    
    ##############
    
    for _,d in g.nodes(data=True):
        d['max_degree'] = max_degree

    # Set all edges to default weight of '1'
    for u,v,d in g.edges(data=True):
        d['cost'] = 1 # I should play with this

    for k,v in attr_d.items():
        d = dict(v)
        d.update({a:0 for a in g.nodes() if a not in d.keys()}) # Explicitly add node attributes for zero inventory nodes
        nx.set_node_attributes(g, k, d)
    
    ##############
    
    #nx.set_node_attributes(g, 'communities', node_communities)
    
    return {'community_skeleton':g_comm, 'agent_skeleton':g_agents, 'graph':g,
            'node_communities':node_communities, 'community_nodes':community_nodes,
            'attr_d':attr_d}