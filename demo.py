from solver import lp_supply_chain_from_nx, plot_solution
from problem_generators import generate_transport_problem, generate_clique_mcf_problem

if __name__ == "__main__":
    random.seed(1)
    g, attr_d = generate_transport_problem(50)
    layout = nx.spring_layout(g)
    nx.draw(g, pos=layout)

    solution, gs, cost = lp_supply_chain_from_nx(g, attr_d.keys())
    print (cost)
    print (status_dict[solution.status])
    _ = plot_solution(gs, attr_d)
    gc.collect()

    #path_item3_17_37 63.0
    #path_item3_2_17 63.0
    solution, gs, cost = lp_supply_chain_from_nx(g, attr_d.keys(), 
                                                 promises=[('2','28',0), 
                                                           ('2','32',0),
                                                           ('2','38',0),
                                                           ('2','42',0)
                                                          ])
    print (cost)
    print (status_dict[solution.status])
    _ = plot_solution(gs, attr_d)
    gc.collect()

    for v in solution.variables():
        if 'path' == v.name.split('_')[0] and v.varValue >0:
            print (v.name, v.varValue)
            
    # Try out networkx's pure-python network simplex algorithm. 
    test = nx.min_cost_flow(add_dummy_supply_sink(g, attr_d.keys()), demand="item1", weight="cost")

    test
    gtest = nx.DiGraph()
    for p,d in test.items():
        for q,v in d.items():
            if v>0:
                print (p,q,v)
                gtest.add_edge(p,q,{'flow_item1':v})
    plot_solution(gtest, attr_d)
    # I guess this is technically a solution, but I got a WAY more compact result from the LP approach. This definitely is
    # not the "minimum cost" flow solution, even for just one commodity type. 
    #
    # NB: Flow is in the wrong direction. nx interprets negative demand as supply (which makes sense I guess).

    #################################################
    # ### Breadcrumbs
    # * Minimum cost flow problem
    # * Multi-commodity flow problem
    # * Network simplex algorithm
    # * Online algorithm: http://arxiv.org/abs/1201.5030

    # ### Multi-scale graph decomposition for scalable LP
    # 1. Decompose the graph into communities. In the ingress network, this is straightforward by setting each local group (which is a clique) to its own community
    # 2. Create a super-graph with communities as nodes and inter community connections as edges, with edge cost equal to the lowest cost of any edge between those communities. Set supply/demand for each comunity node as the net demand of the community.
    # 3. Calculate a gear transfer solution on the super graph to determine how items should be exchanged between communities.
    # 4. For each community, calculate the optimal gear transfer solution within the community, adding in hyper-nodes to represent external communities and/or agents from external communities who we have already determined should flow gear with respect to the chosen community.
        # - If a given community is too large, decompose it into smaller communities before calculating a solution. 
            # - If the community is a clique, construct a pseudo-arbitrary decomposition via sampling subsets of nodes in a fashion that attempts to maintain supply-demand balance in the sub-communities.
            
    # ### To do:
    # * New problem generator that builds a graph as a collection of labeled cliques
    # * Wrapper to workhorse function that builds the supergraph from the true graph, solves the supergraph problem, then iterates through communities adding nodes to represent inter-community transfers from the supergraph solution.
    # * Add function to construct pseudo-arbitrary decomposition for large cliques.

    #################################################
                
    test = generate_clique_mcf_problem(n_nodes=50, n_comm=10, avg_comm_per_node=1)

    for k in test.keys():
        if type(test[k])==nx.Graph and k!='graph':
            print(k)
            nx.draw(test[k]) 
            plt.show()

                
    g      = test['graph']
    attr_d = test['attr_d']

    start = time.time()
    solution, gs, cost = lp_supply_chain_from_nx(g, attr_d.keys())
    end = time.time()

    print("{} nodes, {} edges".format(g.number_of_nodes(), g.number_of_edges()))
    print("Elapsed: {:.2f} seconds".format(end-start)) # 6 seconds for 50 nodes
    print ("Cost:", cost)
    print (status_dict[solution.status],'\n')
    _ = plot_solution(gs, attr_d)
    gc.collect()

    #Next step:
    # 1. Calculate "community inventory" as the net inventory for all agents who are only members of a given community.
    # 2. setup and solve the transport problem to determine how to allocatve inventory for all agents connected to multiple 
    #     communities
    # 3. Infer the graph for the community skeleton
    # 4. solve the MCF problem over the skeleton graph to determine inter-community exchanges.
    # 5. Solve the MCF problem within communities (which i guess is now just a multi-commodity transport problem?) to pair up
    #    people who have/need gear and assign out-of-community ecxhanges to specific agents.
    gc.collect()