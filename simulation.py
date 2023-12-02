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
    M_M_1_K = 5

def interarrival(n, rho, mu):
    return np.random.exponential(1/(n*rho*mu))

def service_time(mu, distribution, general_dist_params=None):
    if distribution == ServiceRateDistribution.M_M_N or distribution == ServiceRateDistribution.SHORTEST_JOB_FIRST:
        return np.random.exponential(1/mu)
    elif distribution == ServiceRateDistribution.M_D_N:
        return 1/mu
    elif distribution == ServiceRateDistribution.HYPEREXPONENTIAL:
        return np.random.exponential(1) if np.random.rand() < 0.75 else np.random.exponential(0.2)
    elif distribution == ServiceRateDistribution.M_M_1_K:
        return np.random.exponential(1/mu)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")

def customer(env, queue, mu, waiting_times, distribution, buffer=None):
    if buffer is not None:
        yield buffer.put(1)
    arrival_time = env.now
    with queue.request() as req:
        yield req
        wait_time = env.now - arrival_time
        yield env.timeout(service_time(mu, distribution))
        waiting_times.append(wait_time)
    if buffer is not None:
        yield buffer.get(1)

def customer_prio(env, queue, mu, waiting_times, distribution):
    arrival_time = env.now
    service_duration = service_time(mu, distribution)
    priority = service_duration
    with queue.request(priority=priority) as req:
        yield req
        wait_time = env.now - arrival_time
        yield env.timeout(service_duration)
        waiting_times.append(wait_time)

def source(env, queue, n, rho, mu, waiting_times, distribution, buffer=None):
    while True:
        yield env.timeout(interarrival(n, rho, mu))
        if buffer is None or buffer.level < buffer.capacity:
            env.process(customer(env, queue, mu, waiting_times, distribution, buffer))

def run_simulation(n, rho, mu, distribution, run_time, K=None):
    env = simpy.Environment()
    waiting_times = []

    if distribution == ServiceRateDistribution.M_M_1_K and K is not None:
        queue = simpy.Resource(env, capacity=1)  # One service channel
        buffer = simpy.Container(env, capacity=K, init=0)  # Finite capacity buffer
        env.process(source(env, queue, n, rho, mu, waiting_times, distribution, buffer))
    elif distribution == ServiceRateDistribution.SHORTEST_JOB_FIRST:
        queue = simpy.PriorityResource(env, capacity=n)
        env.process(source(env, queue, n, rho, mu, waiting_times, distribution))
    else:
        queue = simpy.Resource(env, capacity=n)
        env.process(source(env, queue, n, rho, mu, waiting_times, distribution))

    env.run(until=run_time)
    return waiting_times



def analytical_mean_waiting_time_mmn(n, rho, mu):
    if rho >= 1:
        return float('inf')  # System is unstable for rho >= 1
    if n == 1:
        # M/M/1 Queue
        return rho / (mu * (1 - rho))
    else:
        # M/M/n Queue using Erlang C formula
        rho_total = n * rho
        erlang_c_numerator = (rho_total ** n / np.math.factorial(n)) * (n / (n - rho_total))
        erlang_c_denominator = sum([rho_total ** k / np.math.factorial(k) for k in range(n)]) + erlang_c_numerator
        erlang_c = erlang_c_numerator / erlang_c_denominator
        return (erlang_c / (n * mu - n * rho * mu))

def analytical_mean_waiting_time_mm1k(rho, mu, K):
    # Calculate the probability P0 that the system is empty
    P0 = (1 - rho) / (1 - rho**(K+1))

    # Calculate the mean number of customers in the system, L
    if rho != 1:
        L = (rho * (1 - (K+1) * rho**K + K * rho**(K+1))) / ((1 - rho) * (1 - rho**(K+1)))
    else: # Handle the special case when rho is 1
        L = K / 2

    # Calculate the mean waiting time in the system, W
    # Arrival rate, lambda = mu * rho
    W = L / (mu * rho)

    return W
    

    

def analyze(ns, rhos, mu, distribution, output_file, K=None):
    results = []
    t_test_results = []

    for n in ns:
        for rho in rhos:
            all_waiting_times = []

            for _ in range(30):
                if distribution == ServiceRateDistribution.M_M_1_K:
                    # For M/M/1/K queue, pass K to the simulation
                    waiting_times = run_simulation(n, rho, mu, distribution, 60000, K)
                else:
                    # For other queue types
                    waiting_times = run_simulation(n, rho, mu, distribution, 60000)

                all_waiting_times.extend(waiting_times)
                mean_list = np.concatenate((np.arange(100, 1000, 500), np.arange(1000, 20000, 3000), np.arange(10000, 40000, 2000)))

                for num_customers in mean_list:
                    sampled_times = waiting_times[:num_customers]
                    mean_wait = np.mean(sampled_times)
                    variance = np.var(sampled_times, ddof=1)
                    ci = stats.t.interval(0.95, len(sampled_times)-1, loc=mean_wait, scale=stats.sem(sampled_times))
                    results.append({
                        'n': n, 
                        'rho': rho, 
                        'mu': mu, 
                        'distribution': distribution.name, 
                        'mean_waiting_time': mean_wait, 
                        'ci_lower': ci[0], 
                        'ci_upper': ci[1], 
                        'variance': variance, 
                        'number_of_customers': num_customers
                    })

            if distribution == ServiceRateDistribution.M_M_1_K:
                analytical_mean = analytical_mean_waiting_time_mm1k(rho, mu, K)
            else:
                analytical_mean = analytical_mean_waiting_time_mmn(n, rho, mu)

            t_stat, p_value = stats.ttest_1samp(all_waiting_times, analytical_mean)
            t_test_results.append({
                'n': n,
                'rho': rho,
                't_stat': t_stat,
                'p_value': p_value
            })
            print(f"Completed T-test for n={n}, rho={rho}: t_stat={t_stat}, p_value={p_value}")

    df_results = pd.DataFrame(results)
    df_t_test = pd.DataFrame(t_test_results)
    combined_df = pd.merge(df_results, df_t_test, on=['n', 'rho'], how='outer')
    combined_df.to_csv(output_file)
    print(f"Results written to {output_file}")



# Simulation parameters
mu = 1
ns = [1, 2, 4]
rhos = [0.7, 0.8, 0.9, 0.95]



distribution = ServiceRateDistribution.M_M_N
K=10
output_file = f"{distribution.name.lower()}_simulation_results.csv"
analyze(ns, rhos, mu, distribution, output_file, K)
