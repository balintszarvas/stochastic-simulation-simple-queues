import simpy
import numpy as np
import scipy.stats as stats
import pandas as pd
from enum import Enum
import time


""" 
This script contains the simulation code for the different queue types.
The ServiceRateDistribution enum is used to specify the distribution of the service rate.

"""


class ServiceRateDistribution(Enum):
    M_M_N = 1
    M_D_N = 2
    HYPEREXPONENTIAL = 3
    LONG_TAIL_HYPEREXPONENTIAL = 4
    SHORTEST_JOB_FIRST = 5
    M_M_N_K = 6



def interarrival(n, rho, mu):
    """
    Calculates the arrival time for a customer in a queue.
    
    Parameters:
    - n: int, number of servers.
    - rho: float, traffic intensity of the system.
    - mu: float, service rate of the system.
    """
    return np.random.exponential(1/(n*rho*mu))


def service_time(mu, distribution, general_dist_params=None):
    """
    Calculates the service time for a customer based on the given distribution.
    
    """
    if distribution in [ServiceRateDistribution.M_M_N, ServiceRateDistribution.M_M_N_K, ServiceRateDistribution.SHORTEST_JOB_FIRST]:
        return np.random.exponential(1/mu)
    elif distribution == ServiceRateDistribution.M_D_N:
        return 1/mu
    elif distribution == ServiceRateDistribution.HYPEREXPONENTIAL:
        return np.random.exponential(1) if np.random.rand() < 0.75 else np.random.exponential(0.2)
    elif distribution == ServiceRateDistribution.LONG_TAIL_HYPEREXPONENTIAL: 
        return np.random.exponential(1) if np.random.rand() < 0.75 else np.random.exponential(5)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")

def customer(env, queue, mu, waiting_times, system_times, customer_count, distribution, buffer=None):
    """
    Simulates a customer in the queue system.
    
    """
    arrival_time = env.now
    customer_count[0] += 1 

    with queue.request() as req:
        yield req
        wait_time = env.now - arrival_time
        service_start_time = env.now
        yield env.timeout(service_time(mu, distribution))
        service_end_time = env.now

        waiting_times.append(wait_time)
        system_times.append(service_end_time - arrival_time)

    customer_count[0] -= 1 


def customer_prio(env, queue, mu, waiting_times, distribution):
    """
    Simulate a priority customer in the queue system.
    
    """
    arrival_time = env.now
    service_duration = service_time(mu, distribution)
    priority = service_duration
    with queue.request(priority=priority) as req:
        yield req
        wait_time = env.now - arrival_time
        yield env.timeout(service_duration)
        waiting_times.append(wait_time)


def source(env, queue, n, rho, mu, waiting_times, system_times, customer_count, distribution, buffer=None):
    """
    Generates customers and add them to the queue.

    """
    while True:
        yield env.timeout(interarrival(n, rho, mu))
        if buffer is None or buffer.level < buffer.capacity:
            env.process(customer(env, queue, mu, waiting_times, system_times, customer_count, distribution, buffer))


def run_simulation(n, rho, mu, distribution, run_time, K=None):
    """
    Runs the queue simulation for a given setup and time.
    
    """
    env = simpy.Environment()
    waiting_times = []
    system_times = []
    customer_count = [0]
    customer_times = []

    if distribution == ServiceRateDistribution.M_M_N_K and K is not None:
        queue = simpy.PriorityResource(env, capacity=n)  
        buffer = simpy.Container(env, capacity=K, init=0)
        env.process(source(env, queue, n, rho, mu, waiting_times, system_times, customer_count, distribution, buffer))
    elif distribution == ServiceRateDistribution.SHORTEST_JOB_FIRST:
        queue = simpy.PriorityResource(env, capacity=n)
        env.process(source(env, queue, n, rho, mu, waiting_times, system_times, customer_count, distribution))
    else:
        queue = simpy.Resource(env, capacity=n)
        env.process(source(env, queue, n, rho, mu, waiting_times, system_times, customer_count, distribution))

    def monitor_customer_count(env, customer_count, customer_times):
        while True:
            customer_times.append((env.now, customer_count[0]))
            yield env.timeout(1) 

    env.process(monitor_customer_count(env, customer_count, customer_times))
    env.run(until=run_time)

    total_customers = sum(count * (customer_times[i+1][0] - time) for i, (time, count) in enumerate(customer_times[:-1]))
    mean_customers = total_customers / run_time

    return waiting_times, system_times, customer_count, mean_customers


def analytical_mean_waiting_time_mmn(n, rho, mu):
    """
    Calculate the analytical mean waiting time for M/M/n queue.
    
    """
    if rho >= 1:
        return float('inf')
    if n == 1:
        return rho / (mu * (1 - rho))
    else:
        rho_total = n * rho
        erlang_c_numerator = (rho_total ** n / np.math.factorial(n)) * (n / (n - rho_total))
        erlang_c_denominator = sum([rho_total ** k / np.math.factorial(k) for k in range(n)]) + erlang_c_numerator
        erlang_c = erlang_c_numerator / erlang_c_denominator
        return (erlang_c / (n * mu - n * rho * mu))

def analyze(ns, rhos, mu, distribution, output_file, K=None):
    """
    Analyzes the simulation results and writes them to a CSV file.
    
    Parameters:
    - ns: list of ints, numbers of servers to simulate.
    - rhos: list of floats, traffic intensities to simulate.
    - mu: float, service rate of the system.
    - distribution: ServiceRateDistribution, type of distribution for service time.
    - output_file: str, filename to save results.
    - K: int, optional; capacity of the buffer for M/M/N/K simulation.
    """
    results = []
    t_test_results = []

    for n in ns:
        for rho in rhos:
            all_waiting_times = []
            all_system_times = []
            all_mean_customers = []

            for _ in range(iterations):
                waiting_times, system_times, _, mean_customers = run_simulation(n, rho, mu, distribution, 60000, K)

                all_waiting_times.extend(waiting_times)
                all_system_times.extend(system_times)
                all_mean_customers.append(mean_customers)
                mean_list = np.concatenate((np.arange(100, 1000, 500), np.arange(1000, 20000, 3000), np.arange(10000, 40000, 2000)))

                for num_customers in mean_list:
                    sampled_times = waiting_times[:num_customers]
                    sampled_system_times = system_times[:num_customers]
                    mean_wait = np.mean(sampled_times)
                    mean_system_time = np.mean(sampled_system_times)
                    variance = np.var(sampled_times, ddof=1)
                    ci = stats.t.interval(0.95, len(sampled_times)-1, loc=mean_wait, scale=stats.sem(sampled_times))
                    results.append({
                        'n': n, 
                        'rho': rho, 
                        'mu': mu, 
                        'distribution': distribution.name, 
                        'mean_waiting_time': mean_wait, 
                        'mean_system_time': mean_system_time,
                        'mean_customers': np.mean(all_mean_customers),
                        'ci_lower': ci[0], 
                        'ci_upper': ci[1], 
                        'variance': variance, 
                        'number_of_customers': num_customers
                    })

            if distribution == ServiceRateDistribution.M_M_N:
                analytical_mean = analytical_mean_waiting_time_mmn(n, rho, mu)
                t_stat, p_value = stats.ttest_1samp(all_waiting_times, analytical_mean)
                t_test_results.append({
                    'n': n,
                    'rho': rho,
                    't_stat': t_stat,
                    'p_value': p_value
                })
                print(f"Completed T-test for n={n}, rho={rho}: t_stat={t_stat}, p_value={p_value}")
            else:
                print(f"Completed simulation for n={n}, rho={rho}, distribution={distribution.name}")

    df_results = pd.DataFrame(results)
    if t_test_results:
        df_t_test = pd.DataFrame(t_test_results)
        combined_df = pd.merge(df_results, df_t_test, on=['n', 'rho'], how='outer')
    else:
        combined_df = df_results
    combined_df.to_csv(output_file)
    print(f"Results written to {output_file}")



"""
Simulation parameters and execution loop.

Add more iterations if needed for the statistical analysis. 
The set of distributions can be changed as well according to the enum list at the top.
The output files are saved in the same directory as this script and named after the distribution.
The plotting script uses these exact names.

"""
mu = 0.5
ns = [1, 2, 4]
rhos = [0.7, 0.8, 0.9, 0.95]
iterations = 30

    
distributions = [ServiceRateDistribution.M_M_N,
                 ServiceRateDistribution.M_D_N,
                 ServiceRateDistribution.SHORTEST_JOB_FIRST]

output_files = []
for distribution in distributions:
    output_file = f"{distribution.name.lower()}_simulation_results.csv"
    analyze(ns, rhos, mu, distribution, output_file, K=10)
    output_files.append(output_file)

