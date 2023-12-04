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

