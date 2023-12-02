import simpy
import numpy as np
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import csv
import multiprocessing
from numba import jit

def interarrival(n, rho, mu):
    return np.random.exponential(1/(n*rho*mu))

def service_time(mu):
    return np.random.exponential(1/mu)

def customer(env, queue, mu, t_waiting_time):
    t_arrive = env.now
    t_service_time = service_time(mu)
    with queue.request() as req:
        yield req
        t_wait = env.now
        yield env.timeout(t_service_time)
        t_waiting_time.append(t_wait - t_arrive)

def source(env, queue, n, rho, mu, t_waiting_time):
    while True:
        yield env.timeout(interarrival(n, rho, mu))
        c = customer(env, queue, mu, t_waiting_time)
        env.process(c)

@jit
def run_simulation(n, rho, mu, run_time=60000, run_id=0):
    env = simpy.Environment()
    queue = simpy.Resource(env, capacity=n)
    wait_times = []
    env.process(source(env, queue, n, rho, mu, wait_times))
    env.run(until=run_time)
    return wait_times

def worker(n, rho, mu, run_id):
    return run_simulation(n, rho, mu, 60000, run_id)

def analyze(ns, rhos, mu, runs, output_file="simulation_results-3.csv"):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['n', 'rho', 'mu', 'mean_waiting_time', 'ci_lower', 'ci_upper', 'variance', 'runs'])

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

        for rho in rhos:
            for n in ns:
                for run_count in runs:
                    all_wait_times = []
                    results = []

                    for i in range(int(run_count)):
                        result = pool.apply_async(worker, args=(n, rho, mu, i))
                        results.append(result)

                    for result in results:
                        try:
                            wait_times = result.get()
                            all_wait_times.extend(wait_times)
                        except Exception as e:
                            print(f"Error in run: {e}")

                    mean = np.mean(all_wait_times)
                    ci = stats.t.interval(0.95, len(all_wait_times)-1, loc=mean, scale=stats.sem(all_wait_times))
                    variance = np.var(all_wait_times)

                    print(f'n={n}, rho={rho}, Runs: {run_count}, Mean Waiting Time: {mean}, CI: {ci}, Variance: {variance}')
                    writer.writerow([n, rho, mu, mean, ci[0], ci[1], variance, run_count])

        pool.close()
        pool.join()

    print(f"Results written to {output_file}")

# Simulation parameters
mu = 1
ns = [1, 2, 4]
rhos = [0.5, 0.6, 0.7, 0.8, 0.9]
runs = [10, 25, 50, 75, 100, 125, 150]

analyze(ns, rhos, mu, runs)
