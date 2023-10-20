# just a normal py file..

from model import Simulation, run_sim
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def contour_plot_number_of_surviving_opinions(N=100, runs=5, avg_degree=10, num_opinions=4, p_count=20, phi_count=20, phi_init=0.2):
    def mean(func, *args):
        sum = 0
        count = 0
        for i in range(runs):
            res = func(*args,run=i)
            sum += res
            count += 1
        return sum/count if count > 0 else -1
    def run_with_params(p, phi,run=0):
        simul = run_sim(N=N, avg_degree=avg_degree, p=p, phi=phi, num_opinions=num_opinions, run=run)
        simul.run_retry()
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
    surviving_opinions = np.concatenate([np.load('surviving_opinions_{}.npy'.format(n)) for n in range(1,8)])
    Z = surviving_opinions.reshape(PHI.shape)

    # return Z to the parent process
    contour = plt.contourf(P, PHI, Z, levels=list(range(1, num_opinions+1)), cmap='jet')
    plt.colorbar(contour).ax.invert_yaxis()
    plt.ylabel('Φ')
    plt.xlabel('p')
    plt.show()
    print("HERE")


def run():
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
print("HERE")

import time
print("hi")
tstart = time.time()
contour_plot_number_of_surviving_opinions(N=1000, p_count=15, phi_count=15, phi_init=0.2, num_opinions=50, runs=1, avg_degree=10)
#contour_plot_number_of_surviving_opinions(N=55, p_count=10, phi_count=5, num_opinions=8)
print(time.time() - tstart )

#profiler.disable()
#profiler.dump_stats('profile_data.prof')#jupyter_flow()
#from multiprocessing import freeze_support, Process
#freeze_support()
#if __name__ == '__main__':
   # freeze_support()
   # Process(target=mean).start()
