import simpy
import numpy as np
import scipy.stats as stats
import pandas as pd
from enum import Enum
import time

class ServiceRateDistribution(Enum):
    M_M_N = 1
    M_D_N = 2
    HYPEREXPONENTIAL = 3
    SHORTEST_JOB_FIRST = 4
    M_G_N = 5

def interarrival(n, rho, mu):
    return np.random.exponential(1/(n*rho*mu))

def service_time(mu, distribution, general_dist_params=None):
    if distribution in [ServiceRateDistribution.M_M_N, ServiceRateDistribution.SHORTEST_JOB_FIRST]:
        return np.random.exponential(1/mu)
    elif distribution == ServiceRateDistribution.M_D_N:
        return 1/mu
    elif distribution == ServiceRateDistribution.HYPEREXPONENTIAL:
        return np.random.exponential(1) if np.random.rand() < 0.75 else np.random.exponential(5)
    elif distribution == ServiceRateDistribution.M_G_N:
        return np.random.normal(*general_dist_params)

def customer(env, queue, mu, waiting_times, distribution):
    arrival_time = env.now
    with queue.request() as req:
        yield req
        wait_time = env.now - arrival_time
        yield env.timeout(service_time(mu, distribution))
        waiting_times.append(wait_time)

def customer_prio(env, queue, mu, waiting_times, distribution):
    arrival_time = env.now
    service_duration = service_time(mu, distribution)
    priority = service_duration
    with queue.request(priority=priority) as req:
        yield req
        wait_time = env.now - arrival_time
        yield env.timeout(service_duration)
        waiting_times.append(wait_time)

def source(env, queue, n, rho, mu, waiting_times, distribution):
    while True:
        yield env.timeout(interarrival(n, rho, mu))
        if distribution == ServiceRateDistribution.SHORTEST_JOB_FIRST:
            env.process(customer_prio(env, queue, mu, waiting_times, distribution))
        else:
            env.process(customer(env, queue, mu, waiting_times, distribution))

def run_simulation(n, rho, mu, distribution, general_dist_parameters, run_time):
    env = simpy.Environment()
    if distribution == ServiceRateDistribution.SHORTEST_JOB_FIRST:
        queue = simpy.PriorityResource(env, capacity=n)
    else:
        queue = simpy.Resource(env, capacity=n)
    
    waiting_times = []
    env.process(source(env, queue, n, rho, mu, waiting_times, distribution))
    env.run(until=run_time)
    return waiting_times

def analyze(ns, rhos, mu, distribution, general_dist_params, output_file):
    results = []

    for n in ns:
        for rho in rhos:
            all_waiting_times = []
            for _ in range(30):
                waiting_times = run_simulation(n, rho, mu, distribution, general_dist_params, 60000)
                all_waiting_times.extend(waiting_times)  # Collect waiting times across all runs

            mean_list = np.concatenate((np.arange(100, 1000, 500), np.arange(1000, 20000, 3000), np.arange(10000, 40000, 2000)))

            for num_customers in mean_list:
                sampled_times = all_waiting_times[:num_customers]
                mean_wait = np.mean(sampled_times)
                variance = np.var(sampled_times, ddof=1)
                std_dev = np.std(sampled_times, ddof=1)
                ci = stats.t.interval(0.95, len(sampled_times)-1, loc=mean_wait, scale=stats.sem(sampled_times))
                results.append({
                    'n': n, 
                    'rho': rho, 
                    'mu': mu, 
                    'distribution': distribution.name, 
                    'mean_waiting_time': mean_wait, 
                    'std_dev': std_dev,
                    'ci_lower': ci[0], 
                    'ci_upper': ci[1], 
                    'variance': variance, 
                    'number_of_customers': num_customers
                })

            print(f"Completed analysis for n={n}, rho={rho}, distribution={distribution.name}")

    df = pd.DataFrame(results)
    df.to_csv(output_file)
    print(f"Results written to {output_file}")

# Simulation parameters
mu = 1
ns = [1, 2, 4]
rhos = [0.7, 0.8, 0.9, 0.95]


mean = 1
std_dev = 0.5
general_dist_params = (mean, std_dev)


# Run simulations for each distribution
for distribution in ServiceRateDistribution:
    output_file = f"{distribution.name.lower()}_simulation_results.csv"
    analyze(ns, rhos, mu, distribution, general_dist_params, output_file)
