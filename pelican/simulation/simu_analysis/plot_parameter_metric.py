import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, gaussian_kde
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import KernelDensity

# def run(axes, files_dir, files, parameter, metrics):
#     for i in range(4):
#         ax = axes[i]
#         filepaths = [f'{files_dir}target_length-{files[i]}' for file in files_dir]
#         plot_parameter_metric(ax, filepaths, parameter, metrics[i])
    
def plot_parameter_metric(files_dir, files, parameters, metrics):
    
    # Ensure `parameters` is a tuple (even if it contains only one element)
    if isinstance(parameters, str):
        parameters = (parameters,)
    
    # Dynamically calculate figure size, prepare plot layout
    rows = len(parameters)
    cols = len(metrics)
    fig_width = cols * 4  # Scale width per metric
    fig_height = rows * 3  # Scale height per parameter
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = axes.flatten()
    
    # Iterate over all combinations of parameters and metrics
    for i, (parameter, metric) in enumerate([(p, m) for p in parameters for m in metrics]):
        ax = axes[i]  # Select the corresponding subplot
        filepaths = [f'{files_dir}{parameter}-{file}' for file in files]  # Generate filepaths
        
        # Iterate over the filepaths and process each file
        for file in filepaths:
            # Read the data and filter invalid rows
            df = pd.read_csv(file)
            df = df[df[parameter] >= 0]
            
            if parameter == 'sampling':
                df = df[df[parameter] <= 1]
            
            # Bin data into 10 bins and calculate bin means and midpoints
            binned = pd.cut(df[parameter], bins=100)
            bin_means = df.groupby(binned, observed=False).mean()
            bin_midpoints = [(interval.left + interval.right) / 2 for interval in binned.cat.categories]
            
            # total_points += len(bin_midpoints)
            
            # Plot in the provided axes
            # ax.plot(bin_midpoints, bin_means[metric], label=f"Prompt {file[-5]}")
            ax.plot(bin_midpoints, bin_means[metric]) # without label
        
        # Add titles and labels for the subplot
        ax.set_title(f'{metric.capitalize()} vs {parameter.capitalize()}')
        ax.set_xlabel(parameter.capitalize())
        ax.set_ylabel(metric.replace('_', ' ').capitalize())
        # ax.legend() 
    
    plt.show()
    # print(f"Total number of data points plotted: {total_points}")


# def plot_parameter_metric(ax, filepaths, parameter, metric):
#     # global n_datapoints, pearson_corr, spearman_corr
    
#     # Read each file and plot the data
#     for file in filepaths:
#         # Read files and filter out rows where the parameter is less than 0
#         df = pd.read_csv(file)
#         df = df[df[parameter] >= 0]
        
#         if parameter == 'sampling':
#             df = df[df[parameter] <= 1]
        
#         # Increment the total number of datapoints
#         n_datapoints += df.shape[0]
        
#         # Calculate Pearson and Spearman correlations
#         # pearson_corr_value, _ = pearsonr(df[parameter], df[metric])
#         # spearman_corr_value, _ = spearmanr(df[parameter], df[metric])
#         # pearson_corr.append(pearson_corr_value)
#         # spearman_corr.append(spearman_corr_value)
        
#         # 5. Plot with seaborn (lowess regression with confidence intervals)
#         # sns.regplot(x=parameter, y=metric, data=df, ax=ax, lowess=True, scatter_kws={'alpha': 0.5})
        
#         # 4. Trim edges
#         lower_bound, upper_bound = df[parameter].quantile([0.05, 0.95])
#         df = df[(df[parameter] >= lower_bound) & (df[parameter] <= upper_bound)]
        
#         # n_datapoints += df.shape[0]
#         # pearson_corr_value, _ = pearsonr(df[parameter], df[metric])
#         # spearman_corr_value, _ = spearmanr(df[parameter], df[metric])
#         # pearson_corr.append(pearson_corr_value)
#         # spearman_corr.append(spearman_corr_value)
        
#         binned = pd.cut(df[parameter], bins=15)
#         bin_means = df.groupby(binned).mean()
#         bin_midpoints = [(interval.left + interval.right) / 2 for interval in binned.cat.categories]
#         ax.plot(bin_midpoints, bin_means[metric], label=f"Prompt {file[-5]}")
        
#         # 3. Apply kernel regression smoothing
#         # df = df.sort_values(parameter)
#         # x = df[parameter].values.reshape(-1, 1)  # Reshape x for sklearn compatibility
#         # y = df[metric].values
#         # kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(x)
#         # density = np.exp(kde.score_samples(x))
#         # smoothed_y = np.cumsum(y * density) / np.cumsum(density)
#         # ax.plot(x.flatten(), smoothed_y, label=f"Prompt {file[-5]}")
        
#         # 2. Sort data and apply Gaussian smoothing
#         # df = df.sort_values(parameter)
#         # x = df[parameter].values
#         # y = df[metric].values
#         # smoothed_y = gaussian_filter1d(y, sigma=2)
#         # ax.plot(x, smoothed_y, label=f"Prompt {file[-5]}")

#         # #1.  Estimate density of x values
#         # density = gaussian_kde(df[parameter])(df[parameter])
#         # # Bin data and calculate weighted means
#         # binned = pd.cut(df[parameter], bins=10)
#         # weighted_means = df.groupby(binned).apply(
#         #     lambda group: np.average(group[metric], weights=density[group.index])
#         # )
#         # bin_midpoints = [(interval.left + interval.right) / 2 for interval in binned.cat.categories]
        
#         # Automatically calculate bin edges using quantiles
#         # quantiles = df[parameter].quantile([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
#         # binned = pd.cut(df[parameter], bins=quantiles, include_lowest=True)
#         # bin_means = df.groupby(binned, observed=False).mean()
#         # bin_midpoints = [(interval.left + interval.right) / 2 for interval in binned.cat.categories]
        
#         # Bin data into variable-width bins with equal frequency
#         # binned = pd.qcut(df[parameter], q=20, duplicates='drop')  # Adjust 'q' for the number of bins
#         # bin_means = df.groupby(binned, observed=False).mean()
#         # bin_midpoints = [(interval.left + interval.right) / 2 for interval in binned.cat.categories]
        
#         # # Bin data into 10 bins and calculate bin means and midpoints
#         # binned = pd.cut(df[parameter], bins=10)
#         # bin_means = df.groupby(binned, observed=False).mean()
#         # bin_midpoints = [(interval.left + interval.right) / 2 for interval in binned.cat.categories]
        
#         # Plot in the provided axes
#         # ax.plot(bin_midpoints, weighted_means, label=f"Prompt {file[-5]}")
#         ax.set_title(f'{metric.capitalize()} vs {parameter.capitalize()}')
#         ax.set_xlabel(parameter.capitalize())
#         ax.set_ylabel(metric.replace('_', ' ').capitalize())
#         ax.legend()