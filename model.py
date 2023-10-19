import networkx as nx
import numpy as np

def generate_random_regular_graph(simul):
    graph = nx.random_regular_graph(simul.k, simul.N)
    # Guarantee initial graph is connected
    i = 0
    while(not nx.is_connected(graph)):
        if i == 5:
            return None
            print("Couldn't generate connected graph. Consider increasing the average degree.")
        else:   
            graph = nx.gnp_random_graph(simul.N, simul.k/simul.N)
            i += 1
    return graph

def get_neighbor_opinion_distribution(G, node, exclude = None):
    """
    Return bin count of the opinions of the neighbors
    """ 
    count = {}
    for neigh in nx.neighbors(G, node):
        opinion = G.nodes[neigh]['opinion']
        if opinion != exclude:
            count[opinion] = count.get(opinion, 0) + 1
    return count

def get_majority_opinion(G, node):
    """
    Calculate the majority opinion of the nodes, i.e., the opinion held by the largest 
    number of individuals among the neighbors of the node. Possibly returns the node's current 
    opinion if that is the majority opinion among its neighbors
    """
    count = get_neighbor_opinion_distribution(G, node)
    # we give preference to the current node opinion. This way if there is a tie between
    # the current node opinion and some other majority opinion we return the current node opinion.
    maj_opinion = G.nodes[node]['opinion']
    # Can be zero if we have a different opinion from all our neighbors.
    max_count = count.get(maj_opinion, 0)
    for o, c in count.items():
        if c > max_count:
            maj_opinion = o
            max_count = c  
    return maj_opinion 

class Simulation:
    def __init__(self, N=15, avg_degree=5, p=0.5, phi=0.5, num_opinions=3, initial_graph=generate_random_regular_graph):
        self.N = N
        self.k = avg_degree
        self.p = p
        self.phi = phi
        self.G = num_opinions
        self.time = 0
        self.num_surviving_opinions = 0
        self.stall = 0
        self.stall_bin = {}
        # tuple indicating information about the previous performed operation:
        # 1: whether it was successful (0 or 1)
        # 2: whether it was MP (-1) or MA with all (0) or MA with neighbor of neighbors (1)
        self.prev = (None, None)
        self.status = 0            
        self.initialize_graph_method = initial_graph
        self.graph = initial_graph(self)
        if self.graph == None:
            self.status = -1
            return
        self.init_opinions()
    def init_opinions(self):
        """ 
        Set the opinion of all individuals in the graph. 
        All opinions are equally likely (uniform distribution).
        """
        opinions = {i: 1+np.random.choice(self.G) for i in range(self.N)}
        nx.set_node_attributes(self.graph, opinions, 'opinion')
    def reset(self):
        self.time = 0
        self.stall = 0
        self.stall_bin = {}
        self.status = 0
    def rewire(self, node, rule):
        """ 
        Disconnect node from a neighbor according to MA rule and connect him with probability phi
        to a random neighbor of its neighbors who holds the same
        opinion, or otherwise to a random one selected from the
        whole population except its nearest neighbors with probability 1-phi.
        Upon success return 0 else it returns:
        1: we have the same opinion as all our neighbors
        2: couldn't find a candidadate removal neighbor
        3: couldn't find a neighbor of my neighbors with the same opinion as me
        4: couldn't find another node in the network, that isn't my neighbor, having
        the same opinion as me
        5: Removing and adding the selected edges made the graph unconnected
        """
        node_opinion = self.graph.nodes[node]['opinion']
        count = get_neighbor_opinion_distribution(self.graph, node, exclude = node_opinion)
        if not count:
            return 1
        min_opinion_count = min(count.values()) 
        candidate_removal_nodes = [ neigh for neigh in nx.neighbors(self.graph, node) if 
                                    self.graph.nodes[neigh]['opinion'] != node_opinion and
                                    count[self.graph.nodes[neigh]['opinion']] == min_opinion_count and
                                    self.graph.degree[neigh] > 1
                                ]
        if not candidate_removal_nodes:
            return 2
        neigh_remove = np.random.choice(candidate_removal_nodes)
        neigh_add = None
        # rewire to neighbor of neighbors with same opinion
        if rule == 1:
            neighbor_of_neighbors = { neigh_of_neigh 
                                    for neigh in nx.neighbors(self.graph, node) 
                                    for neigh_of_neigh in nx.neighbors(self.graph, neigh) 
                                    if self.graph.nodes[neigh_of_neigh]['opinion'] == node_opinion
                                    }
            neighbor_of_neighbors.remove(node)
            neighbor_of_neighbors -= set(nx.neighbors(self.graph, node))
            if not neighbor_of_neighbors:
                return 3
            neigh_add = np.random.choice(np.fromiter(neighbor_of_neighbors, int, len(neighbor_of_neighbors))) 
        # rewire to any non-neighboring node in the network with same opinion
        elif rule == 0:
            candidates = { other_node 
                        for other_node in nx.nodes(self.graph)
                        if self.graph.nodes[other_node]['opinion'] == node_opinion
                        }   
            candidates.remove(node)
            candidates -= set(nx.neighbors(self.graph, node))
            if not candidates:
                return 4
            neigh_add = np.random.choice(np.fromiter(candidates, int, len(candidates)))
        self.graph.remove_edge(node, neigh_remove)    
        self.graph.add_edge(node, neigh_add)
        if not nx.is_connected(self.graph):
            # undo removal and addition of neighbors if it results in a disconnected graph
            self.graph.remove_edge(node, neigh_add)    
            self.graph.add_edge(node, neigh_remove)
            return 5
        return 0
    def step(self):
        """ 
        One step of the simulation. Application of one of the rules (MA or MP) to a single node. 
        """
        # choose between MA and MP according to the parameter p
        rule = np.random.binomial(1, self.p)
        # choose a random node to apply the rule to
        node = np.random.choice(nx.nodes(self.graph))
        stall_code = 0

        # rule==1 -> Apply majority preference rule to node
        if rule == 1:
            # if prev was (failed, MP), we have converged so no need to do anything
            if self.prev == (0, -1):
                return
            first_node = node 
            change = False
            node_prev_opinion = self.graph.nodes[node]['opinion']
            maj_opinion = get_majority_opinion(self.graph, node)
            while not change:
                if node_prev_opinion != maj_opinion:
                    self.graph.nodes[node]['opinion'] = maj_opinion
                    self.time += 1
                    # set prev to (successful, MP)
                    self.prev = (1, -1)
                    change = True
                else:
                    node += 1
                    if node == self.N:
                        node = 0
                    # this means we have converged because all nodes hold the same opinion
                    # the majority of their neighbors hold
                    if node == first_node:
                        # set prev to (failed, MP)
                        self.prev = (0, -1)
                        return
                    node_prev_opinion = self.graph.nodes[node]['opinion']
                    maj_opinion = get_majority_opinion(self.graph, node)

        # rule==0 -> Apply minority avoidance rule to node
        else: 
            ma_rule = np.random.binomial(1, self.phi)
            # if the previous operation was unsuccessful (note it has to have been an MA)
            if self.prev[0] == 0:
                # if [we are applying MA with neighbor of neighbors (1)] or 
                # [are applying MA with all (0) and the previous was MA with all
                # or MP, and they failed] then nothing to be done, the network didn't
                # change and we are trying to do the same thing again, or in the case
                # of MP failing, it means we have converged already
                if ma_rule == 1 or (ma_rule == 0 and self.prev[1] != 1):
                    self.stall += 1
                    return
            first_node = node 
            stall_code = self.rewire(node, ma_rule) 
            while stall_code:
                # keep looping through the nodes until you find one that you can apply rewire to
                # or you get back to the initial node. Notice it's still random because the first node
                # was selected randomly
                node += 1
                if node == self.N:
                    node = 0
                if node == first_node:
                    self.stall += 1
                    self.prev = (0, ma_rule)
                    return
                stall_code = self.rewire(node, ma_rule) 
            self.time += 1
            self.prev = (1, ma_rule)
    def convergence_condition(self):
        """
        When we reach the consensus state we can stop the simulation, i.e., when each individuals
        opinion agrees with the majority of its neighbors.
        """
        def agrees_with_majority(node):
            count = get_neighbor_opinion_distribution(self.graph, node)
            max_opinion_count = max(count.values())
            s = list(count.values())
            s.sort(reverse=True)
            if s[0] == s[1]: return False
            return count.get(self.graph.nodes[node]['opinion'], 0) == max_opinion_count
        return all(
                agrees_with_majority(node)
                for node in nx.nodes(self.graph)
             )
    def halt_condition(self):
        """
        Condition to stop the simulation run due to possible loop.
        Here we expect more stalls with more nodes, less connectedness, smaller p
        and larger phi (for simplicity's sake we ignore phi).
        If there are too many stalls for each advance in the simulation we
        consider the simulation is advancing too slowly. 
        """
        return self.stall >= 50 + 10 * (1-self.p) * self.time * np.log2(self.N) / self.k
    def run(self):
        if self.status:
            return self.status
        while(not (self.convergence_condition() )):

            self.step()
            self.step()
            self.step()
            self.step()
            self.step()
            self.step()
            self.step()
            self.step()
            self.step()
            self.step()
        if self.convergence_condition():
            self.status = 1    
        self.num_surviving_opinions = len(set(nx.get_node_attributes(self.graph, "opinion").values()))
        return self.status
    def run_retry(self, limit=5):
        self.run()
        i=1
        while self.status == -1 and i < limit:
            self.reset()
            self.graph = self.initialize_graph_method(self)
            if(self.graph == None):
                continue
            self.init_opinions()
            self.run()
            i+=1
        return self.status