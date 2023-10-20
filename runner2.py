from model import run_sim
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

def parallel_array(function, generator):
    Parallel(n_jobs=7)(delayed(mean)(runs, lambda p, phi, run: run_with_params(p, phi, run), phi,p) for phi in phi_range)


def mean(runs, func, *args):
    sum = 0
    count = 0
    for i in range(runs):
        res = func(*args,i)
        if res >= 0:
            sum += res
            count += 1
    return sum/count if count > 0 else -1
def plot_phi_vs_final_number_of_opinions(N=100, p=0.2, phi_count=5, runs=10, avg_degree=10, num_opinions=10):
    def run_with_params(p, phi,run):
        simul = run_sim(N=N, avg_degree=avg_degree, p=p, phi=phi, num_opinions=num_opinions, run=run)
        simul.run_retry()
        return simul.num_surviving_opinions
    phi_range = np.linspace(0, 1, phi_count)
    results = Parallel(n_jobs=7)(delayed(mean)(runs, lambda p, phi, run: run_with_params(p, phi, run), phi,p) for phi in phi_range)
    surviving_opinions = np.array(results)
    plt.plot(phi_range, surviving_opinions, 'o')
    plt.xlabel('Φ')
    plt.ylabel('Number of opinions')
    plt.show()


def contour_plot_number_of_surviving_opinions(N=100, runs=50, avg_degree=10, num_opinions=4, p_count=20, phi_count=20):
    def run_with_params(p, phi,run):
        simul = run_sim(N=N, avg_degree=avg_degree, p=p, phi=phi, num_opinions=num_opinions, run=run)
        simul.run_retry()
        return simul.num_surviving_opinions

    p_range = np.linspace(0, 1, p_count)
    phi_range = np.linspace(0, 1, phi_count)
    P, PHI = np.meshgrid(p_range, phi_range)
    params = np.dstack((P, PHI)).reshape(-1,2)

    results = Parallel(n_jobs=4)(delayed(mean)(runs, run_with_params, p, phi) for p,phi in params)
    surviving_opinions = np.array(results)
    Z = surviving_opinions.reshape(PHI.shape)
    contour = plt.contourf(P, PHI, Z, levels=list(np.arange(1, num_opinions + 0.1, 0.1)), cmap='jet')
    plt.colorbar(contour).ax.invert_yaxis()
    plt.ylabel('Φ')
    plt.xlabel('p')
    plt.show()


contour_plot_number_of_surviving_opinions(N=1000, avg_degree=10, p_count=160, phi_count=160, num_opinions=50, runs=20)
#plot_phi_vs_final_number_of_opinions(N=1000, p=0.6, phi_count=50, runs=10, avg_degree=10, num_opinions=50)