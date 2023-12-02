import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from scipy.stats import kstest, expon, anderson, shapiro

fontsize = 18
colormap = "viridis"

plt.rcParams.update({
    'font.size': fontsize,
    "text.usetex": True,
    "figure.dpi": 300
})

sns.set(style="darkgrid", font='Palatino', font_scale=1.25)
sns.set_palette(sns.color_palette(colormap))


os.makedirs('final_plots', exist_ok=True)

def load_data(queue_model, n=None):
    file_map = {
        "MMN": "m_m_n_simulation_results.csv",
        "MDN": "m_d_n_simulation_results.csv",
        "Hyperexponential": "hyperexponential_simulation_results.csv",
        "ShortestJobFirst": "shortest_job_first_simulation_results.csv"
    }
    filename = file_map.get(queue_model, "m_m_n_simulation_results.csv")
    df = pd.read_csv(filename)
    if n is not None:
        df = df[df['n'] == n]
    return df

queue_model_format = {
    "MMN": {"title": "M/M/N", "filename": "MMN"},
    "MDN": {"title": "M/D/N", "filename": "MDN"},
    "Hyperexponential": {"title": "Hyperexponential", "filename": "Hyperexponential"},
    "ShortestJobFirst": {"title": "Shortest Job First", "filename": "ShortestJobFirst"}
}

def plot_mean_for_n(queue_models, ns):
    for queue_model in queue_models:
        for n in ns:
            df_mmn = load_data(queue_model, n)
            plt.figure(figsize=(7, 5))

            for rho in df_mmn['rho'].unique():
                df_rho = df_mmn[df_mmn['rho'] == rho]
                df_rho['std_dev'] = np.sqrt(df_rho['variance'])
                df_rho['ci_size'] = df_rho['ci_upper'] - df_rho['ci_lower']
                sns.lineplot(x='number_of_customers', y='mean_waiting_time', data=df_rho,
                             label=fr'$\rho$={rho}', estimator='mean', ci='sd')

            plt.xlabel('Number of Customers')
            plt.ylabel('Mean Waiting Time')
            title = f'Mean Waiting Time for the {queue_model_format[queue_model]["title"]} Queue (N={n})'
            plt.title(title)
            plt.legend(loc='upper right')
            plt.savefig(f'final_plots/{queue_model_format[queue_model]["filename"]}_plot_n_{n}.png')
            plt.close()

def plot_mean_for_rho(queue_models, ns):
    for queue_model in queue_models:
        df_all = pd.concat([load_data(queue_model, n) for n in ns])

        for rho in df_all['rho'].unique():
            plt.figure(figsize=(7, 5))

            for n in ns:
                df_mmn = load_data(queue_model, n)
                df_rho = df_mmn[df_mmn['rho'] == rho]
                if not df_rho.empty:
                    df_rho['std_dev'] = np.sqrt(df_rho['variance'])
                    df_rho['ci_size'] = df_rho['ci_upper'] - df_rho['ci_lower']
                    sns.lineplot(x='number_of_customers', y='mean_waiting_time', data=df_rho,
                                 label=f'N={n}', estimator='mean', ci='sd')

            plt.xlabel('Number of Customers')
            plt.ylabel('Mean Waiting Time')
            title = fr'Mean Waiting Time for the {queue_model_format[queue_model]["title"]} Queue ($\rho$={rho})'
            plt.title(title)
            plt.legend(loc='upper right')
            plt.savefig(f'final_plots/{queue_model_format[queue_model]["filename"]}_plot_rho_{rho}.png')
            plt.close()
        
def compare_queue_models(queue_models, n, rho):
    plt.figure(figsize=(10, 7))

    model_titles = []
    model_filenames = []

    for queue_model in queue_models:
        df = load_data(queue_model, n)

        if not df.empty:
            df_rho = df[df['rho'] == rho]
            if not df_rho.empty:
                sns.lineplot(x='number_of_customers', y='mean_waiting_time', data=df_rho,
                             label=f'{queue_model_format[queue_model]["title"]}', estimator='mean', ci='sd')
                model_titles.append(queue_model_format[queue_model]["title"])
                model_filenames.append(queue_model_format[queue_model]["filename"])

    models_title = " vs. ".join(model_titles)
    models_filename = "_vs_".join(model_filenames)

    plt.xlabel('Number of Customers', fontsize=fontsize)
    plt.ylabel('Mean Waiting Time', fontsize=fontsize)
    title = fr'Comparison of {models_title} ($\rho={rho}, N={n}$)'
    plt.title(title, fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(f'final_plots/comparison_{models_filename}_rho_{rho}_N_{n}.png')
    plt.close()

def plot_distribution_and_analyze(queue_model, n, rho):
    df = load_data(queue_model, n)
    df_rho = df[df['rho'] == rho]
    
    data = df_rho['mean_waiting_time']

    plt.figure(figsize=(10, 7))
    sns.histplot(data, bins=30, kde=True)
    plt.xlabel('Mean Waiting Time')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Mean Waiting Times for {queue_model_format[queue_model]["title"]} (N={n}, $\\rho$={rho})')

    lambda_est = 1 / np.mean(data)
    
    ks_stat, ks_p_value = kstest(data, expon(scale=lambda_est).cdf)
    ad_stat, _, _ = anderson(data, dist='expon')
    shapiro_stat, shapiro_p_value = shapiro(data)

    text_props = dict(horizontalalignment='right', verticalalignment='top', 
                      transform=plt.gca().transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.5))

    plt.text(0.95, 0.95, f'KS Test Statistic: {ks_stat:.10f}\nP-value: {ks_p_value:.10f}', **text_props)

    plt.text(0.95, 0.80, f'Anderson-Darling Test Statistic: {ad_stat:.10f}', **text_props)

    plt.text(0.95, 0.70, f'Shapiro-Wilk Test Statistic: {shapiro_stat:.10f}\nP-value: {shapiro_p_value:.10f}', **text_props)

    plt.savefig(f'final_plots/distribution_{queue_model_format[queue_model]["filename"]}_N_{n}_rho_{rho}.png')
    plt.close()


ns = [1, 2, 4]

# queue_models = ["MMN", "MDN", "Hyperexponential", "ShortestJobFirst"]
# plot_mean_for_n(queue_models, ns)
# plot_mean_for_rho(queue_models, ns)

# compare_queue_models(["MMN", "MDN"], 1, 0.9)

plot_distribution_and_analyze("MMN", 1, 0.9)