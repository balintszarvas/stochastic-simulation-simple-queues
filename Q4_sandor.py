import simpy
import numpy as np
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import csv
from enum import Enum

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
            return np.random.exponential(1)  # 75% chance for average service time of 1.0
        else:
            return np.random.exponential(5)  # 25% chance for average service time of 5.0

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

def run_simulation(n, rho, mu, distribution, run_time=60000):
    env = simpy.Environment()
    queue = simpy.Resource(env, capacity=n)
    wait_times = []
    env.process(source(env, queue, n, rho, mu, wait_times, distribution))
    env.run(until=run_time)
    return wait_times

def analyze_runs(ns, rhos, mu, distribution, start_runs=100, max_runs=300, run_increment=100, output_file="simulation_results.csv"):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['n', 'rho', 'mu', 'distribution', 'mean_waiting_time', 'ci_lower', 'ci_upper', 'variance', 'runs'])

        for rho in rhos:
            for n in ns:
                for run_count in range(start_runs, max_runs + 1, run_increment):
                    all_wait_times = []
                    for _ in range(run_count):
                        wait_times = run_simulation(n, rho, mu, distribution)
                        all_wait_times.extend(wait_times)

                    mean = np.mean(all_wait_times)
                    ci = stats.t.interval(0.95, len(all_wait_times)-1, loc=mean, scale=stats.sem(all_wait_times))
                    variance = np.var(all_wait_times)

                    print(f'n={n}, rho={rho}, Distribution={distribution.name}, Runs: {run_count}, Mean Waiting Time: {mean}, CI: {ci}, Variance: {variance}')
                    writer.writerow([n, rho, mu, distribution.name, mean, ci[0], ci[1], variance, run_count])

    print(f"Results written to {output_file}")

# Simulation parameters
mu = 1
ns = [1, 2, 4]
rhos = np.linspace(0.6, 1.0, 5)

# M/D/1 queue
analyze_runs(ns, rhos, mu, ServiceRateDistribution.M_D_1, output_file="md1_simulation_results.csv")

# Hyperexponential queue
analyze_runs(ns, rhos, mu, ServiceRateDistribution.HYPEREXPONENTIAL, output_file="hyperexponential_simulation_results.csv")
