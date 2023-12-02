import simpy
import numpy as np
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import csv
from numba import jit
# from Q2_sandor import run_simulation

def plot_results(output_file):
    with open(output_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        data = [row for row in reader if float(row[1]) != 0.99]  # Exclude rows where rho is 0.99

    variance_column_index = 6 

    for n in set(row[0] for row in data):
        plt.figure()
        for rho in set(row[1] for row in data if row[0] == n):
            filtered_data = [row for row in data if row[0] == n and row[1] == rho]
            run_times = [int(row[7]) for row in filtered_data]
            variances = [float(row[variance_column_index]) for row in filtered_data]
            plt.plot(run_times, variances, label=f'ρ={rho}')

        plt.xlabel('Simulation Run Time')
        plt.ylabel('Variance')
        plt.title(f'Variance vs Simulation Run Time for n={n}')
        plt.legend()
        plt.show()

def plot_additional_results(output_file):
    with open(output_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
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

# Call the plotting functions with the appropriate output file name
plot_results("hyperexponential_simulation_results.csv")
# plot_additional_results("simulation_results_runtimes.csv")

# Uncomment below lines if you have the run_simulation function and want to visualize waiting times
# wait_times = run_simulation(n, rho, mu)
# plot_waiting_times(wait_times)
