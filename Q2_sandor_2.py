import simpy
import numpy as np
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import csv
from numba import jit
import time

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
def run_simulation(n, rho, mu, run_time):
    env = simpy.Environment()
    queue = simpy.Resource(env, capacity=n)
    wait_times = []
    env.process(source(env, queue, n, rho, mu, wait_times))
    env.run(until=run_time)
    return wait_times

def analyze(ns, rhos, mu, run_times, output_file="simulation_results_runtimes_2.csv"):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['n', 'rho', 'mu', 'mean_waiting_time', 'ci_lower', 'ci_upper', 'variance', 'run_time', 'duration'])

        for rho in rhos:
            for n in ns:
                for run_time in run_times:
                    start_time = time.time()
                    wait_times = run_simulation(n, rho, mu, run_time)
                    end_time = time.time()
                    duration = end_time - start_time
                    mean = np.mean(wait_times)
                    ci = stats.t.interval(0.95, len(wait_times)-1, loc=mean, scale=stats.sem(wait_times))
                    variance = np.var(wait_times)

                    print(f'n={n}, rho={rho}, Run Time: {run_time}, Mean Waiting Time: {mean}, CI: {ci}, Variance: {variance}', f'Duration: {duration:.2f} seconds')
                    writer.writerow([n, rho, mu, mean, ci[0], ci[1], variance, run_time])

    print(f"Results written to {output_file}")

# Simulation parameters
mu = 1
ns = [1, 2, 4]
rhos = [0.5, 0.6, 0.7, 0.8, 0.9]
run_times = [100, 500, 1000, 5000, 10000, 50000, 100000]  

analyze(ns, rhos, mu, run_times)