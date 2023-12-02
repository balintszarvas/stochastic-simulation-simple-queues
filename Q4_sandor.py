import simpy
import numpy as np
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import csv
from numba import jit
from enum import Enum
import time

class ServiceRateDistribution(Enum):
    M_M_1 = 1
    M_D_1 = 2
    HYPEREXPONENTIAL = 3

def interarrival(n, rho, mu):
    return np.random.exponential(1/(n*rho*mu))

def service_time(mu, distribution):
    if distribution == ServiceRateDistribution.M_M_1:
        return np.random.exponential(1/mu)
    elif distribution == ServiceRateDistribution.M_D_1:
        return 1/mu 
    elif distribution == ServiceRateDistribution.HYPEREXPONENTIAL:
        if np.random.rand() < 0.75:
            return np.random.exponential(1)
        else:
            return np.random.exponential(5)

def customer(env, queue, mu, t_waiting_time, distribution):
    t_arrive = env.now
    t_service_time = service_time(mu, distribution)
    with queue.request() as req:
        yield req
        t_wait = env.now
        yield env.timeout(t_service_time)
        t_waiting_time.append(t_wait - t_arrive)

def source(env, queue, n, rho, mu, t_waiting_time, distribution):
    while True:
        yield env.timeout(interarrival(n, rho, mu))
        c = customer(env, queue, mu, t_waiting_time, distribution)
        env.process(c)

@jit
def run_simulation(n, rho, mu, distribution, run_time=60000):
    env = simpy.Environment()
    queue = simpy.Resource(env, capacity=n)
    wait_times = []
    env.process(source(env, queue, n, rho, mu, wait_times, distribution))
    env.run(until=run_time)
    return wait_times

def analyze(ns, rhos, mu, distribution, runs, output_file="simulation_results.csv"):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['n', 'rho', 'mu', 'mean_waiting_time', 'ci_lower', 'ci_upper', 'variance', 'runs', 'duration'])

        for rho in rhos:
            for n in ns:
                for run_count in runs:
                    all_wait_times = []
                    start_time = time.time() 
                    
                    for _ in range(int(run_count)):
                        wait_times = run_simulation(n, rho, mu, distribution)
                        all_wait_times.extend(wait_times)
                    
                    end_time = time.time() 
                    duration = end_time - start_time

                    mean = np.mean(all_wait_times)
                    ci = stats.t.interval(0.95, len(all_wait_times)-1, loc=mean, scale=stats.sem(all_wait_times))
                    variance = np.var(all_wait_times)

                    print(f'n={n}, rho={rho}, Distribution={distribution.name}, Runs: {run_count}, Mean Waiting Time: {mean}, CI: {ci}, Variance: {variance}, Duration: {duration:.2f} seconds')
                    writer.writerow([n, rho, mu, distribution.name, mean, ci[0], ci[1], variance, run_count])

    print(f"Results written to {output_file}")

# Simulation parameters
mu = 1
ns = [1, 2, 4]
rhos = [0.5, 0.6, 0.7, 0.8, 0.9]
runs = [10, 25, 50, 75, 100, 125, 150]

# M/D/1 queue
analyze(ns, rhos, mu, ServiceRateDistribution.M_D_1, runs, output_file="md1_simulation_results.csv")

# Hyperexponential queue
analyze(ns, rhos, mu, ServiceRateDistribution.HYPEREXPONENTIAL, runs, output_file="hyperexponential_simulation_results.csv")
