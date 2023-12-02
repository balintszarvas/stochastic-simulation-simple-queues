import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats

sns.set(style="darkgrid")

def load_data(distribution, n=None):
    df = pd.read_csv(f"{distribution.lower()}_simulation_results.csv")
    if n is not None:
        df = df[df['n'] == n]
    return df

# Mean Waiting Time vs Number of Customers for MM1 and MD1
def subplot_mean_waiting_time_mm1_md1(ns):
    fig, axes = plt.subplots(len(ns), 1, figsize=(10, 6 * len(ns)))

    for i, n in enumerate(ns):
        df_mm1 = load_data("M_M_N", n)
        df_md1 = load_data("M_D_N", n)

        sns.lineplot(ax=axes[i], x='number_of_customers', y='mean_waiting_time', data=df_mm1, label='M/M/1')
        sns.lineplot(ax=axes[i], x='number_of_customers', y='mean_waiting_time', data=df_md1, label='M/D/1')
        
        axes[i].set_xlabel('Number of Customers')
        axes[i].set_ylabel('Mean Waiting Time')
        axes[i].set_title(f'M/M/1 and M/D/1 (n={n})')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

# Mean Waiting Time for M/M/N with confidence intervals
def subplot_mean_waiting_time_mmn_with_ci(ns):
    fig, axes = plt.subplots(len(ns), 1, figsize=(10, 6 * len(ns)))

    for i, n in enumerate(ns):
        df = load_data("M_M_N", n)

        for rho in df['rho'].unique():
            df_rho = df[df['rho'] == rho]
            sns.lineplot(ax=axes[i], x='number_of_customers', y='mean_waiting_time', data=df_rho, label=f'rho={rho}')
            axes[i].fill_between(df_rho['number_of_customers'], df_rho['ci_lower'], df_rho['ci_upper'], alpha=0.3)

        axes[i].set_xlabel('Number of Customers')
        axes[i].set_ylabel('Mean Waiting Time')
        axes[i].set_title(f'M/M/N (n={n})')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

# Standard Deviation for M/M/N
def subplot_std_deviation_mmn(ns):
    fig, axes = plt.subplots(len(ns), 1, figsize=(10, 6 * len(ns)))

    for i, n in enumerate(ns):
        df = load_data("M_M_N", n)

        for rho in df['rho'].unique():
            df_rho = df[df['rho'] == rho]
            std_dev = np.sqrt(df_rho['variance'])
            std_dev_df = pd.DataFrame({
                'number_of_customers': df_rho['number_of_customers'],
                'std_deviation': std_dev
            })
            sns.lineplot(ax=axes[i], x='number_of_customers', y='std_deviation', data=std_dev_df, label=f'rho={rho}')

        axes[i].set_xlabel('Number of Customers')
        axes[i].set_ylabel('Standard Deviation of Waiting Time')
        axes[i].set_title(f'Standard Deviation for M/M/N (n={n})')
        axes[i].legend()

    plt.tight_layout()
    plt.show()


# Average Waiting Time for FIFO vs Shortest Job Priority
def plot_fifo_vs_sjf(ns):
    num_subplots = len(ns)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 6 * num_subplots))

    if num_subplots == 1:
        axes = [axes]

    for idx, n in enumerate(ns):
        df_fifo = load_data("M_M_N", n)
        df_sjf = load_data("SHORTEST_JOB_FIRST", n)

        sns.lineplot(x='number_of_customers', y='mean_waiting_time', data=df_fifo, label=f'FIFO (n={n})', ax=axes[idx])
        sns.lineplot(x='number_of_customers', y='mean_waiting_time', data=df_sjf, label=f'Shortest Job First (n={n})', ax=axes[idx])

        axes[idx].set_xlabel('Number of Customers')
        axes[idx].set_ylabel('Average Waiting Time')
        axes[idx].set_title(f'FIFO vs Shortest Job First (n={n})')
        axes[idx].legend()

    plt.tight_layout()
    plt.show()

def subplot_mean_waiting_time_mmn_with_ci(ns):
    # Create a 3-column subplot structure for each 'n'.
    fig, axes = plt.subplots(len(ns), 3, figsize=(21, 6 * len(ns)), sharex=True)

    for i, n in enumerate(ns):
        df_mmn = load_data("M_M_N", n)
        for rho in df_mmn['rho'].unique():
            df_rho = df_mmn[df_mmn['rho'] == rho]

            # Calculate the standard deviation.
            df_rho['std_dev'] = np.sqrt(df_rho['variance'])

            # Calculate the absolute size of the confidence interval.
            df_rho['ci_size'] = df_rho['ci_upper'] - df_rho['ci_lower']
            
            # grouped = df_rho.groupby('number_of_customers').agg({'mean_waiting_time': 'mean', 
            #                                                  'ci_lower': 'mean', 
            #                                                  'ci_upper': 'mean'})

            # # Plot mean waiting time.
            # axes[i, 0].plot(grouped.index, grouped['mean_waiting_time'], label=f'rho={rho}')

            # # Fill between ci_lower and ci_upper.
            # axes[i, 0].fill_between(grouped.index, grouped['ci_lower'], grouped['ci_upper'], alpha=0.3)
            
            # Plot mean waiting time with standard deviation as the confidence interval.
            sns.lineplot(ax=axes[i, 0], x='number_of_customers', y='mean_waiting_time', data=df_rho,
                         label=f'rho={rho}', estimator='mean', ci='sd')
            
            # Plot standard deviation without confidence intervals.
            sns.lineplot(ax=axes[i, 1], x='number_of_customers', y='std_dev', data=df_rho,
                         label=f'rho={rho}', ci=None)
            
            # Plot the size of the confidence interval for mean waiting time without confidence intervals.
            sns.lineplot(ax=axes[i, 2], x='number_of_customers', y='ci_size', data=df_rho,
                         label=f'rho={rho}', ci=None)

        # Set labels and titles for the first column.
        axes[i, 0].set_xlabel('Number of Measurements (customers)')
        axes[i, 0].set_ylabel('Mean Waiting Time')
        axes[i, 0].set_title(f'M/M/N (n={n}) Mean Waiting Time')
        axes[i, 0].legend()

        # Set labels and titles for the second column.
        axes[i, 1].set_xlabel('Number of Measurements (customers)')
        axes[i, 1].set_ylabel('Standard Deviation of Waiting Time')
        axes[i, 1].set_title(f'M/M/N (n={n}) Standard Deviation')
        axes[i, 1].legend()

        # Set labels and titles for the third column.
        axes[i, 2].set_xlabel('Number of Measurements (customers)')
        axes[i, 2].set_ylabel('CI Size of Mean Waiting Time')
        axes[i, 2].set_title(f'M/M/N (n={n}) CI Size')
        axes[i, 2].legend()

    plt.tight_layout()
    plt.show()
    

ns = [1, 2, 4]

# Run the plotting functions
subplot_mean_waiting_time_mmn_with_ci(ns)
# subplot_mean_waiting_time_mm1_md1(ns)
# subplot_mean_waiting_time_mmn_with_ci(ns)
# subplot_std_deviation_mmn(ns)
# plot_fifo_vs_sjf(ns)
