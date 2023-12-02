import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import norm, kstest
import scipy.stats as stats

# Load the CSV results into a DataFrame
df = pd.read_csv('m_m_n_simulation_results.csv')

# Define the number of bins for the histogram
bins = 30

# Create the plot grid
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

# Flatten the axes array for easy iteration
axes = axes.flatten()

n_values = [1, 2, 4]
rho_values = [0.7, 0.8, 0.9, 0.95]
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 20))

# Iterate over each subplot and plot the distribution
for i, n in enumerate(n_values):
    for j, rho in enumerate(rho_values):
        ax = axes[j, i]
        subset = df[(df['n'] == n) & (df['rho'] == rho)]
        mean_wait = subset['mean_waiting_time'].mean()
        std_wait = subset['mean_waiting_time'].std()
        sns.histplot(subset['mean_waiting_time'], bins=30, kde=False, ax=ax, stat="probability")
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean_wait, std_wait)
        ax.plot(x, p, 'k', linewidth=2)
        D, p_value = kstest(subset['mean_waiting_time'], 'norm', args=(mean_wait, std_wait))
        ax.set_title(f'n = {n}, ρ = {rho}, p-value = {p_value:.4f}')
        ax.set_xlabel('Mean waiting time')
        ax.set_ylabel('Probability')

# Adjust the layout to prevent overlapping
plt.tight_layout()
plt.show()
