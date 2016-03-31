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


### Graph helper functions ###

def build_solution_graph(solution):
    """Given a solved pulp.LpProblem transport problem, returns a graph representation of the flows recommended by the solver"""
    g = nx.DiGraph()
    for v in solution.variables():
        if int (v.varValue) > 0: # This shouldn't be necessary, but I was getting some O(1e-9) values in my solution.
            tokens = v.name.split('_')
            if tokens[0] == 'path':
                _, a, p, q = tokens
                g.add_edge(p, q, {'flow_{}'.format(a):int(v.varValue)})
    return g

def add_dummy_supply_sink(g, attr_name, dummy_name='DummySinkNode'):
    """
    Given a problem where demand != supply, we need to balance supply and demand by adding a node connected to all
    supply or demand nodes to balance the difference. An edge directed towards this node essentially represents a recommendation 
    for storage at the supply node or unsatisified demand at the demand node. For this problem, storage cost will be zero
    but it does not necessarily need to be.
    
    Whether or not a dummy node is added, the input graph is always returned as a directed graph.
    """
    g0 = g.to_directed()
    g = g.to_directed()
    g.add_node(dummy_name)
    balanced = True
    for a in attr_name:
        supply = 0
        demand = 0
        for n,v in nx.get_node_attributes(g, a).items():
            if v>0:
                g.add_edge(n, dummy_name, {'cost':0})
                supply += v
            elif v<0:
                g.add_edge(dummy_name, n, {'cost':0})
                demand += v
        delta = supply + demand # demand is negative valued
        if delta != 0:
            balanced=False
        nx.set_node_attributes(g, a, {dummy_name:-delta})
    if balanced:
        return g0
    return g

def plot_solution(gs, attr_d):
    layout = nx.spring_layout(gs)

    print("Node degree")
    nx.draw(gs, pos=layout)
    nx.draw_networkx_labels(gs, pos=layout, labels=gs.to_undirected().degree())
    plt.show()

    for a in attr_d.keys():
        print(a)
        g2 = nx.DiGraph()
        e_attr = nx.get_edge_attributes(gs, 'flow_'+a)
        g2.add_edges_from(e_attr.keys())
        d2 = {k:v for k,v in attr_d[a].items() if k in g2.nodes()} # starting inventory

        layout = nx.spring_layout(g2)
        nx.draw(g2, pos=layout)
        nx.draw_networkx_labels(g2, pos=layout, labels=d2)
        nx.draw_networkx_edge_labels(g2, pos=layout, edge_labels=nx.get_edge_attributes(gs, 'flow_'+a))
        plt.show()

    return layout
    
# Let's try a version where we add "promises". A "prommise" happens when two people agree to exchange gear, or when one of the
# two involved in an exchange asserts that the exchange will not happen. There are a few ways I can do this:
# 1. Set an inequality constraint on the relevant flow to denote "at least this much gear needs to pass between these people."
# 2. Set an equality constrain to denote "Exactly this gear needs to pass between these two people."
# 3. Set an equality constraint on the indicator variable to denote "these two people will meet."
# 4. Increment inventory on the node attributes as though the exchange has already happened to indicate that the gear has 
#    essentially transferred.
#
# I'd *like* to set a promise using the indicator node and let people figure out how much gear should be transferred when they
# meet by referencing the app, but in reality it would be better to let users accept "fixed" assignments and not need to refer
# back to the application afterwards.
#
# The binary variable is going to need to be involved to allow users to at least deny that an exchange is going to happen, so 
# let's start there and maybe add in the constraint on flow as well later on.

# Could really let the users decide.
# 1. Promise to meet:          for given p,q set indicator to 1
# 2. Reject meeting:           for given p,q set indicator to 0
# 3. Promise to exchange gear: for given p,q,a set flow >= promise value 
# 4. Reject gear exchange:     for given p,q,a set flow = 0

# In production, we could then calculate a new solution every half hour or whatever to grab account for new promises.

# Thought for the future: GAMIFICATION. Is there anything we can do to encourage user participation in this project?
# -- Points/badges for amount of gear pumped into the system
# -- points/badges for engaging in an exchange of any kind
# -- points/badges for the number of exchanges in a given period

# Could also include an "honor system" for people to enter in transfers that weren't associated with a specific 
# solution. This way people can still get points for exchanges that happen without checking in to a solution. For instance,
# the results of this project might end up constructing regular supply lines, in which case people wouldn't need to check
# in all the time because they'll have found what they need.

# Similarly, point tracking of this kind would allow us to quantitatively identify users who are generally flush with
# gear, users who are generally in need of gear, and users who are critical for transmitting gear. 

# I think the biggest thing will be if we have a big op or anomaly where we want to centralize a particular kind of rare gear, like 
# mods or viruses, it would be very useful to have something like this in place.

### Linear programming ###

status_dict = {1:'Optimal solution found.', 
       -1:'Optimal solution infeasible. Suboptimal solution returned. Expect some constraints unsatisfied.', 
        0:'Unable to solve.',
       -2:'Problem unbounded.',
       -3:'Solution undefined.'
       }

# NB: looks like the LP solver doesn't tolerate underscores in node names
def lp_supply_chain_from_nx(g, attr_name, promises=[], dummy_name='DummySinkNode', maxflow=8000, alpha=1, beta=1): 
    """
    Workhorse function for LP solution of transport problems represented in a graph. Currently only supports a single
    node attribute, but will expand function in future to operate on a list of node attributes.
    
    maxflow: Upper bound on flow through any given edge
    alpha: tuning parameter for regularizaiton term minimizing count of active edges
    beta: tuning paramter for regularization term minimizing transfers in excess of a node's given max degree threshold
    """
    g = g.copy() # Just to make sure we don't modify the original object in-place
    for p,q,v in promises:
        if v == 0:
            g.remove_edge(p,q)
    g = add_dummy_supply_sink(g, attr_name, dummy_name)
    agents = g.nodes()
    in_paths = defaultdict(list)
    out_paths = defaultdict(list)
    for p,q in g.edges_iter():
        out_paths[p].append((p,q))
        in_paths[q].append((p,q))
    
    prob = pulp.LpProblem("Supply Chain", pulp.LpMinimize)

    # Flow along each respective path for each item type
    x = pulp.LpVariable.dicts("path", (attr_name, agents, agents), 
                            lowBound = 0 ,
                            cat = pulp.LpInteger)
    
    # Undirected edge indicator variables (will be used for regularization)
    x2 = pulp.LpVariable.dicts("pathInd", (agents, agents), 
                            lowBound = 0 ,
                            upBound = 1,
                            cat = pulp.LpInteger)
    
    maxdeg = nx.get_node_attributes(g, 'max_degree')
    
    ### Objective function ###
    # Primary objective: minimize amount of gear transferred (edge weight)
    # Regularization: minimize number of transfers (edge count)
    # Regularization: minimize number of transfers in excess of agent's preferred max threshold (node degree)
    #    * How can we change this to ensure that we aren't slamming our hubs?
    # To do: 
    #   * minimize number of agents with unsatisfied inventory (i.e. ignoring contribution of dummy node)
    #   --> is it preferable to have 2 agents who are both unsatisified, or one satisifed and one not? 
    #   --> I feel like this should be tied into the regularization term that reduces the number of transfers.
    #   --> Maybe we could somehow weight "inconvenience" differently for hubs? 
    #      --> MAGIC! We can use the indicator variables to count the number of dummy demand edges!
    #      --> This *should* work, but in practice, the objective score doesn't change and the solver reports the solution
    #          as undefined. No idea why.
    # Should the regularization term really be the full objective function and the gear transfers should be just a constraint?
    prob += sum([x[a][p][q] for a in attr_name for (p,q) in g.edges_iter() if dummy_name not in p and dummy_name not in q]) + \
                 alpha * sum(x2[p][q] * g[p][q]['cost'] for (p,q) in g.edges()) + \
                 beta  * sum(maxdeg[p] - sum(x2[p][q] for q in g.neighbors(p) if dummy_name not in p and dummy_name not in q) \
                            for p in agents if p in maxdeg), \
                 "Objective"
                 # gamma * sum(x2[dummy_name][q] for q in g.neighbors(dummy_name))
                
    # Flow constraints    
    for agent in agents:
        for a in attr_name:
            inventory = nx.get_node_attributes(g, a)[agent]
            prob += sum([x[a][p][q] for (p,q) in in_paths[agent]])  - \
                         sum([x[a][p][q] for (p,q) in out_paths[agent]]) + \
                         inventory == 0, \
                         "netFlowConstr_{}_{}".format(a,agent)

    # Associate undirected indicator variable with relevant edges
    for p,q in g.to_undirected().edges():
        if p>q:
            p,q = q,p
        prob += sum([x[a][p][q] + x[a][q][p] for a in attr_name]) <= 1e9*x2[p][q]
            
    ## Ensure the dummy node only acts as a sink
    for p,q in g.edges_iter():
        if dummy_name in p or dummy_name in q:
            for a in attr_name:
                invp = nx.get_node_attributes(g, a)[p]
                invq = nx.get_node_attributes(g, a)[q]
                if invp==0 or invq==0 or np.sign(invp)==np.sign(invq):
                    #print("BLOCKING[{}]:{}({}) -> {}({}) ".format(a, p, invp, q, invq ))
                    prob+=x[a][p][q] == 0    
        
    # Force "promised" exchanges. 
    # Rejected exchanges handled earlier by removing edges from graph, only need to force agreed upon exchanges here (v=1).
    for p,q,v in promises:
        if v>0:
            if p>q:
                p,q = q,p
            prob+=x2[p][q] == v
        
    prob.solve()
    
    gs = build_solution_graph(prob)
    for n in gs.nodes():
        if dummy_name in n:
            gs.remove_node(n)
        
    cost = pulp.value(prob.objective)
    
    return prob, gs, cost

