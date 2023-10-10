# just a normal py file..

import networkx as nx
from my_bussiness import *

def jupyter_flow():

    simul = Simulation(N = 1000, avg_degree = 10, p = 1)
    G = simul.graph
    if log:
        print(simul.graph.edges)
        print(nx.get_node_attributes(G, "opinion"))
        draw(G)
    for _ in range(5) :
        simul.step()
    if log:
        print(nx.get_node_attributes(G, "opinion"))
        draw(G)
    import time
    tstart = time.time()
    simul.run()
    print(time.time() - tstart)
    if log:
        draw(G)
    print(simul.status, simul.time, simul.stall)
    print(simul.stall_bin)


    """
        import os
        id = 0
        pid  = 0
        children = []
        for n in range(8):
            id = n
            #pid = os.fork()
            children.push(pid)
            if pid !=0:
                break

        # split params in 8 parts and run each part in a different process
        params = params[id*len(params)//8:(id+1)*len(params)//8]
        res = [
            mean(run_with_params, params[n][0],
                    params[n][1] #phi
                    )
            if (id * len(params) // 8) < n < ((id + 1) * len(params) // 8) else 0
            for n in range(params)
        ]
        """

    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    # Call your function
    import time
    tstart = time.time()
    contour_plot_number_of_surviving_opinions(N=300, p_count=10, phi_count=5, num_opinions=8)
    print(time.time() - tstart )

    profiler.disable()
    profiler.dump_stats('profile_data.prof')#jupyter_flow()

#import cProfile
#profiler = cProfile.Profile()
#profiler.enable()
#import time
tstart = time.time()
contour_plot_number_of_surviving_opinions(N=800, p_count=50, phi_count=40, phi_init=0.2, num_opinions=30, runs=50)
#contour_plot_number_of_surviving_opinions(N=55, p_count=10, phi_count=5, num_opinions=8)
print(time.time() - tstart )

#profiler.disable()
#profiler.dump_stats('profile_data.prof')#jupyter_flow()
#from multiprocessing import freeze_support, Process
#freeze_support()
#if __name__ == '__main__':
   # freeze_support()
   # Process(target=mean).start()
