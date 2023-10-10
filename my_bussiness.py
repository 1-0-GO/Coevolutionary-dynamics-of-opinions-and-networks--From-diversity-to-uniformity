import time

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use
use('TkAgg')
from collections import Counter
log = False
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
    maj_opinion = G.nodes[node]['opinion']
    max_count = count.get(maj_opinion, 0)
    for o, c in count.items():
        if c > max_count:
            maj_opinion = o
            max_count = c
    return maj_opinion
def rewire(G, node, phi):
    """
    Disconnect node from a neighbor according to MA rule and connect him with probability phi
    to a random neighbor of its neighbors who holds the same
    opinion, or otherwise to a random one selected from the
    whole population except its nearest neighbors with probability 1-phi.
    """
    node_opinion = G.nodes[node]['opinion']
    count = get_neighbor_opinion_distribution(G, node, exclude = node_opinion)
    if not count:
        return 1
    min_opinion_count = min(count.values())
    candidate_removal_nodes = [ neigh for neigh in nx.neighbors(G, node) if
                                G.nodes[neigh]['opinion'] != node_opinion and
                                count[G.nodes[neigh]['opinion']] == min_opinion_count and
                                G.degree[neigh] > 1
                                ]
    if not candidate_removal_nodes:
        return 2
    neigh_remove = np.random.choice(candidate_removal_nodes)
    neigh_add = None
    rule = np.random.binomial(1, phi)
    # rewire to neighbor of neighbors with same opinion
    if rule == 1:
        neighbor_of_neighbors = { neigh_of_neigh
                                  for neigh in nx.neighbors(G, node)
                                  for neigh_of_neigh in nx.neighbors(G, neigh)
                                  if G.nodes[neigh_of_neigh]['opinion'] == node_opinion
                                  }
        neighbor_of_neighbors.remove(node)
        neighbor_of_neighbors -= set(nx.neighbors(G, node))
        if not neighbor_of_neighbors:
            return 3
        neigh_add = np.random.choice(np.fromiter(neighbor_of_neighbors, int, len(neighbor_of_neighbors)))
        # rewire to any non-neighboring node in the network with same opinion
    elif rule == 0:
        candidates = { other_node
                       for other_node in nx.nodes(G)
                       if G.nodes[other_node]['opinion'] == node_opinion
                       }
        candidates.remove(node)
        candidates -= set(nx.neighbors(G, node))
        if not candidates:
            return 4
        neigh_add = np.random.choice(np.fromiter(candidates, int, len(candidates)))
    G.remove_edge(node, neigh_remove)
    G.add_edge(node, neigh_add)
    if not nx.is_connected(G):
        # undo removal and addition of neighbors if it results in a disconnected graph
        G.remove_edge(node, neigh_add)
        G.add_edge(node, neigh_remove)
        return 5
    if log:
        print('MA for node {} -> rule:{}, neigh_removed:{}, neigh_added:{}'
              .format(node, rule, neigh_remove, neigh_add))
    return 0

class Simulation:
    def __init__(self, N=15, avg_degree=5, p=0.5, phi=0.5, num_opinions=3):
        self.N = N
        self.k = avg_degree
        self.p = p
        self.phi = phi
        self.G = num_opinions
        self.time = 0
        self.num_surviving_opinions = 0
        self.stall = 0
        self.stall_bin = {}
        self.status = 0
        self.init_graph()
    def init_graph(self):
        self.graph = nx.gnp_random_graph(self.N, self.k/self.N)
        # Guarantee initial graph is connected
        i = 0
        while(not nx.is_connected(self.graph)):
            if i == 5:
                self.status = -1
                print("Couldn't generate connected graph. Consider increasing the average degree.")
            else:
                self.graph = nx.gnp_random_graph(self.N, self.k/self.N)
                i += 1
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
            maj_opinion = get_majority_opinion(self.graph, node)
            if log:
                prev_opinion = self.graph.nodes[node]['opinion']
                print('MP for node {} -> prev_opinion:{}, curr_opinion:{}'.format(node, prev_opinion, maj_opinion))
            self.graph.nodes[node]['opinion'] = maj_opinion
        # rule==0 -> Apply minority avoidance rule to node
        else:
            stall_code = rewire(self.graph, node, self.phi)
        if not stall_code:
            self.time += 1
        # error_code 1 means the node doesn't have neighbors with an opinion different than his
        # so it's no problem since it just means the node converged already
        elif stall_code != 1:
            self.stall += 1
            if log:
                self.stall_bin[stall_code] = self.stall_bin.get(stall_code, 0) + 1
    def stop_condition(self):
        """
        When we reach the consensus state we can stop the simulation, i.e., when each individuals
        opinion agrees with the majority of its neighbors.
        """
        def agrees_with_majority(node):
            count = get_neighbor_opinion_distribution(self.graph, node)
            max_opinion_count = max(count.values())
            return count.get(self.graph.nodes[node]['opinion'], 0) == max_opinion_count
        return all(
            agrees_with_majority(node)
            for node in nx.nodes(self.graph)
        )
    def loop_condition(self):
        """
        Condition to stop the simulation run due to possible loop.
        Here we expect more stalls with more nodes, less connectedness and
        larger phi (for simplicity's sake we ignore phi).
        If there are too many stalls for each advance in the simulation we
        consider the simulation is probably looping.
        """
        return self.stall >= 20 + self.time * np.log2(self.N) / self.k
    def run(self):
        if self.status:
            return self.status
        while(not self.stop_condition()):
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
            if self.loop_condition():
                self.status = -1
                return -1
        self.status = 1
        self.num_surviving_opinions = len(set(nx.get_node_attributes(self.graph, "opinion").values()))
        return 1
    def run_retry(self, limit=5):
        self.run()
        i=1
        while self.status == -1 and i < limit:
            self.reset()
            self.init_graph()
            self.run()
            i+=1
        return self.status



####################### PLOTS #################################
def draw(G):
    nx.draw(G, with_labels= True, node_color=list(nx.get_node_attributes(G, "opinion").values()))

def contour_plot_number_of_surviving_opinions(N=100, runs=5, avg_degree=10, num_opinions=4, p_count=20, phi_count=20, phi_init=0.2):
    def mean(func, *args):
        sum = 0
        count = 0
        for i in range(runs):
            res = func(*args)
            if res >= 0:
                sum += res
                count += 1
        return sum/count if count > 0 else -1
    def run_with_params(p, phi):
        simul = Simulation(N=N, avg_degree=avg_degree, p=p, phi=phi, num_opinions=num_opinions)
        simul.run_retry()
        if simul.status == 1:
            return simul.num_surviving_opinions
    p_range = np.linspace(0, 1, p_count)
    phi_range = np.linspace(phi_init, 1, phi_count)
    P, PHI = np.meshgrid(p_range, phi_range)
    params = np.dstack((P, PHI)).reshape(-1,2)
    # split params in 8 parts and run each part in a different process
    import os
    n = 0
    pid = -1
    pids = []
    for n in range(0,8):
        if pid == 0:
            break
        pid = os.fork()
        print("iteration n",n )
        pids.insert(0, pid)
        if pid == 0:
            n = n


    print("pid: ", pid, "n: ", n)

    params = params[n*len(params)//8:(n+1)*len(params)//8]
    res = [
        mean(run_with_params, p,
                phi #phi
                )
   #     if (id * len(params) // 8) < n < ((id + 1) * len(params) // 8) else 0
        for p, phi in params
    ]
    surviving_opinions = np.array(res)
    np.save('surviving_opinions_{}'.format(n), surviving_opinions)

    # wait for all children to exit

    if pid == 0:
        exit(0)
    os.wait()

    # read each file and merge the arrays
    surviving_opinions = np.concatenate([np.load('surviving_opinions_{}.npy'.format(n)) for n in range(8)])
    Z = surviving_opinions.reshape(PHI.shape)

    # return Z to the parent process
    contour = plt.contourf(P, PHI, Z, levels=list(range(1, num_opinions+1)), cmap='jet')
    plt.colorbar(contour).ax.invert_yaxis()
    plt.ylabel('Φ')
    plt.xlabel('p')
    plt.show()
    print("HERE")


