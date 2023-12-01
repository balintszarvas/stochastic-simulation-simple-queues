#Average waiting times are shorter for an M/M/n queue and a system load ρ and processor capacity μ 
# than for a single M/M/1 queue with the same load characteristics

# DES program to verify this for n=1, n=2 and n=4


import simpy
import numpy as np


def interarrival(n,rho,mu):
    """Interarrival generates the arrival interval between customers
    following an exponential distribution"""
    return np.random.exponential(1/n*rho*mu)

def service_time(mu):
    """service_time generates the service time per customer
    following an exponential distribution"""
    return np.random.exponential(1/mu)


def customer(env, customer, queue, mu, t_waiting_time):
    t_arrive = env.now
    t_service_time = service_time(mu)
    with queue.request() as req:
        
        yield req 
        t_wait = env.now
        yield env.timeout(t_service_time)
        t_waiting_time.append(t_wait - t_arrive)
        
        
def source(env, queue,n, rho, mu, t_waiting_time):
    """Source generates customers"""
    i = 0
    while True:
        i = i+1
        yield env.timeout(interarrival(n,rho,mu))
        c = customer(env, i, queue, mu, t_waiting_time)
        env.process(c)



def main(n,rho, mu):
    env = simpy.Environment()
    queue = simpy.Resource(env, capacity=n)  
    wait_times = []
    env.process(source(env, queue, n, rho, mu, wait_times))
    env.run(until = 60000)
    average_waiting_time = np.mean(wait_times)
    print(f"Average Waiting Time: {average_waiting_time}")
    

# Simulation parameters
rho = 0.5               #system load 
mu = 1                #service rate   
n = 1

main(n,rho, mu)
