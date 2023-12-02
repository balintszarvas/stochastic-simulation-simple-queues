#Average waiting times are shorter for an M/M/n queue and a system load ρ and processor capacity μ 
# than for a single M/M/1 queue with the same load characteristics

# DES program to verify this for n=1, n=2 and n=4


import simpy
import numpy as np

def interarrival(rho,mu):
    return np.random.exponential(1/rho*mu)

def service_time(mu):
    return np.random.exponential(1/ mu)


def customer(env, customer, queue, mu):
    t_arrive = env.now
    t_service_time = service_time(mu)

    with queue.request() as req:
        
        yield req 
        t_wait = env.now
        yield env.timeout(t_service_time)
        t_waiting_time = t_wait - t_arrive

def customer_prio(env, queue, mu, t_waiting_time):
    t_arrive = env.now
    t_service_time = service_time(mu)
    prio = t_service_time
    with queue.request(priority=prio) as req:
        yield req
        t_wait = env.now
        yield env.timeout(t_service_time)
        t_waiting_time.append(t_wait - t_arrive)

def source_prio(env, queue, n, rho, mu, t_waiting_time):
    while True:
        yield env.timeout(interarrival(n, rho, mu))
        c = customer_prio(env, queue, mu, t_waiting_time)
        env.process(c)

@jit
def run_simulation_prior(n, rho, mu, run_time=60000):
    env = simpy.Environment()
    queue = simpy.PriorityResource(env, capacity=n)
    wait_times = []
    env.process(source_prio(env, queue, n, rho, mu, wait_times))
    env.run(until=run_time)
    return wait_times
        


# M/M/1 queue simulation
def mm1_queue(env, system_load, queue, service_rate, num_customers, wait_times):

    for i in range(num_customers):
        arrival_time = env.now
        with queue.request() as request:
            yield request

            service_start_time = env.now
            yield env.timeout(service_time(service_rate))

            service_end_time = env.now
            wait_times.append(service_start_time - arrival_time)


def main(rho, mu, num_customers):
    env = simpy.Environment()
    queue = simpy.Resource(env, capacity=1)  
    wait_times = []
    env.process(mm1_queue(env, queue, rho, mu, num_customers, wait_times))
    env.run()
    average_waiting_time = np.mean(wait_times)
    print(f"Average Waiting Time: {average_waiting_time}")
    

# Simulation parameters
rho = 0.5               #system load 
mu = 1                #service rate 
num_customers = 100  

main(rho, mu, num_customers)