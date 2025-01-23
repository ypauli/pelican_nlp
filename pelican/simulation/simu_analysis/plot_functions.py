import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, gaussian_kde
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import KernelDensity
        
def plot_sensitivities(files_dir, files, parameters, metrics):
    """
    Plots the sensitivities of metrics to each parameter using boxplots for Pearson and Spearman correlations.

    Args:
    - files_dir: Directory containing the data files.
    - files: List of file names to read for each parameter.
    - parameters: List of parameter names.
    - metrics: List of metric names.
    """
    # Initialize data to store correlations
    sensitivity_data = {parameter: {'pearson': [], 'spearman': []} for parameter in parameters}
    
    for parameter in parameters:
        for metric in metrics:
            pearson_corrs = []
            spearman_corrs = []
            
            # Iterate over files and calculate correlations
            for file in files:
                filepath = f'{files_dir}{parameter}-{file}'
                df = pd.read_csv(filepath)
                df = df[df[parameter] >= 0]  # Filter parameter values
                
                if parameter == 'sampling':
                    df = df[df[parameter] <= 1]
                
                if not df.empty:
                    # Calculate Pearson and Spearman correlations
                    pearson_corr_value, _ = pearsonr(df[parameter], df[metric])
                    spearman_corr_value, _ = spearmanr(df[parameter], df[metric])
                    
                    pearson_corrs.append(pearson_corr_value)
                    spearman_corrs.append(spearman_corr_value)
            
            # Store correlations for the parameter-metric pair
            sensitivity_data[parameter]['pearson'].append(pearson_corrs)
            sensitivity_data[parameter]['spearman'].append(spearman_corrs)
    
    # Plot boxplots for each parameter
    fig, axes = plt.subplots(2, len(parameters), figsize=(15, 10))
    
    for i, parameter in enumerate(parameters):
        # Prepare data for boxplots
        pearson_data = sensitivity_data[parameter]['pearson']
        spearman_data = sensitivity_data[parameter]['spearman']
        
        # Boxplot for Pearson correlations
        axes[0, i].boxplot(pearson_data, labels=metrics)
        axes[0, i].set_title(f'Pearson Correlations for {parameter}')
        axes[0, i].set_ylabel('Pearson Correlation')
        axes[0, i].set_xlabel('Metrics')
        
        # Boxplot for Spearman correlations
        axes[1, i].boxplot(spearman_data, labels=metrics)
        axes[1, i].set_title(f'Spearman Correlations for {parameter}')
        axes[1, i].set_ylabel('Spearman Correlation')
        axes[1, i].set_xlabel('Metrics')
    
    # plt.tight_layout()
    # plt.show() 
    return sensitivity_data  
    
    
def plot_correlation_heatmap(sensitivity_data, parameters, metrics):
    """
    Plots heatmaps for Pearson and Spearman correlations for each parameter-metric pair.

    Args:
    - sensitivity_data: Dictionary containing correlations for each parameter-metric pair.
    - parameters: List of parameters.
    - metrics: List of metrics.
    """

    fig, axes = plt.subplots(1, len(parameters), figsize=(15, 5))

    for i, parameter in enumerate(parameters):
        # Calculate average correlations for heatmap
        pearson_avg = [sum(corrs) / len(corrs) for corrs in sensitivity_data[parameter]['pearson']]
        spearman_avg = [sum(corrs) / len(corrs) for corrs in sensitivity_data[parameter]['spearman']]

        # Create a DataFrame for the heatmap
        heatmap_data = pd.DataFrame({
            'Metric': metrics,
            'Pearson': pearson_avg,
            'Spearman': spearman_avg
        }).set_index('Metric')

        # Plot heatmap
        sns.heatmap(heatmap_data.T, annot=True, cmap='coolwarm', ax=axes[i], cbar=True)
        axes[i].set_title(f'Correlation Heatmap for {parameter}')
        axes[i].set_xlabel('Metrics')

    plt.tight_layout()
    plt.show()
    
def plot_scatter_correlations(files_dir, files, parameter, metrics):
    """
    Plots scatter plots for correlations between a parameter and its metrics.

    Args:
    - files_dir: Directory containing the data files.
    - files: List of file names to read for the parameter.
    - parameter: The parameter to analyze.
    - metrics: List of metrics.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        all_x, all_y = [], []
        for file in files:
            filepath = f'{files_dir}{parameter}-{file}'
            df = pd.read_csv(filepath)
            df = df[df[parameter] >= 0]  # Filter invalid data

            if parameter == 'sampling':
                df = df[df[parameter] <= 1]

            if not df.empty:
                axes[i].scatter(df[parameter], df[metric], alpha=0.6, label=file)
                all_x.extend(df[parameter])
                all_y.extend(df[metric])

        # Trendline
        if len(all_x) > 1:
            z = np.polyfit(all_x, all_y, 1)
            p = np.poly1d(z)
            axes[i].plot(sorted(all_x), p(sorted(all_x)), color='red', label='Trendline')

        axes[i].set_title(f'{parameter.capitalize()} vs {metric.capitalize()}')
        axes[i].set_xlabel(parameter.capitalize())
        axes[i].set_ylabel(metric.capitalize())
        axes[i].legend()

    plt.tight_layout()
    plt.show()
    
def plot_average_correlations(sensitivity_data, parameters, metrics):
    """
    Plots average Pearson and Spearman correlations for each parameter-metric pair.

    Args:
    - sensitivity_data: Dictionary containing correlations for each parameter-metric pair.
    - parameters: List of parameters.
    - metrics: List of metrics.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for parameter in parameters:
        pearson_avg = [sum(corrs) / len(corrs) for corrs in sensitivity_data[parameter]['pearson']]
        spearman_avg = [sum(corrs) / len(corrs) for corrs in sensitivity_data[parameter]['spearman']]

        # Line plot for Pearson and Spearman
        ax.plot(metrics, pearson_avg, marker='o', label=f'{parameter} - Pearson')
        ax.plot(metrics, spearman_avg, marker='s', label=f'{parameter} - Spearman')

    ax.set_title('Average Correlations Across Metrics')
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Metrics')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
def plot_effect_sizes_by_group(sensitivity_data, parameters, metrics):
    """
    Plots the effect sizes (absolute correlation values) for each metric within each parameter as grouped bar plots.

    Args:
    - sensitivity_data: Dictionary containing correlations for each parameter-metric pair.
    - parameters: List of parameters.
    - metrics: List of metrics.
    """
    # Calculate average effect sizes
    effect_sizes = []
    for parameter in parameters:
        for metric_idx, metric in enumerate(metrics):
            # Calculate average absolute Pearson and Spearman correlations
            avg_pearson = sum(abs(corr) for corr in sensitivity_data[parameter]['pearson'][metric_idx]) / len(
                sensitivity_data[parameter]['pearson'][metric_idx]
            )
            avg_spearman = sum(abs(corr) for corr in sensitivity_data[parameter]['spearman'][metric_idx]) / len(
                sensitivity_data[parameter]['spearman'][metric_idx]
            )
            effect_sizes.append({'Parameter': parameter, 'Metric': metric, 'Type': 'Pearson', 'Effect Size': avg_pearson})
            effect_sizes.append({'Parameter': parameter, 'Metric': metric, 'Type': 'Spearman', 'Effect Size': avg_spearman})

    # Create a DataFrame for plotting
    effect_size_df = pd.DataFrame(effect_sizes)

    # Pivot the DataFrame to create a grouped bar plot structure
    plot_df = effect_size_df.pivot_table(
        values='Effect Size',
        index=['Parameter', 'Metric'],
        columns='Type'
    ).reset_index()

    # Plot the grouped bar plot
    import seaborn as sns
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=effect_size_df,
        x='Parameter',
        y='Effect Size',
        hue='Metric',
        ci=None,
        palette='muted'
    )
    plt.title('Effect Sizes by Parameter and Metric')
    plt.xlabel('Parameters')
    plt.ylabel('Effect Size (Absolute Correlation)')
    plt.xticks(rotation=45)
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
