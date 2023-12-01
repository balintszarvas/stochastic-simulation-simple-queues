import simpy
import numpy as np
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import csv
from numba import jit
from Q2_sandor import run_simulation

def plot_results(output_file):
    with open(output_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        data = [row for row in reader if float(row[1]) != 0.99]  # Exclude rows where rho is a certain value

    variance_column_index = 6 

    for n in set(row[0] for row in data):
        plt.figure()
        for rho in set(row[1] for row in data if row[0] == n):
            filtered_data = [row for row in data if row[0] == n and row[1] == rho]
            run_counts = [int(row[7]) for row in filtered_data]
            variances = [float(row[variance_column_index]) for row in filtered_data]
            plt.plot(run_counts, variances, label=f'ρ={rho}')

        plt.xlabel('Number of Runs')
        plt.ylabel('Variance')
        plt.title(f'Variance vs Number of Runs for n={n}')
        plt.legend()
        plt.show()


def plot_waiting_times(wait_times):
    plt.hist(wait_times, bins='auto', color='blue', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Waiting Time')
    plt.ylabel('Frequency')
    plt.title('Distribution of Waiting Times')
    plt.show()

    
def plot_additional_results(output_file):
    with open(output_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        data = [row for row in reader]

    for plot_type in ["mean_waiting_time", "variance", "ci_width"]:
        plt.figure()
        for n in set(row[0] for row in data):
            filtered_data = [row for row in data if row[0] == n]
            rhos = [float(row[1]) for row in filtered_data]
            
            if plot_type == "mean_waiting_time":
                values = [float(row[3]) for row in filtered_data]
                plt.ylabel('Mean Waiting Time')
            elif plot_type == "variance":
                values = [float(row[6]) for row in filtered_data]
                plt.ylabel('Variance of Waiting Time')
            elif plot_type == "ci_width":
                values = [float(row[5]) - float(row[4]) for row in filtered_data]  # ci_upper - ci_lower
                plt.ylabel('Width of Confidence Interval')

            plt.plot(rhos, values, label=f'n={n}')
        
        plt.xlabel('System Load (ρ)')
        plt.title(f'{plot_type.replace("_", " ").title()} vs System Load')
        plt.legend()
        plt.show()

# plot_results("simulation_results-2.csv")

wait_times = run_simulation(n, rho, mu)
plot_waiting_times(wait_times)