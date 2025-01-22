import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

parameter = 'target_length'
metric = 'perplexity'

# List of CSV filenames
files_dir = '/home/ubuntu/emilia/PELICAN/pelican/simulation/simu_analysis/data/'
files = ['0.csv', '1.csv', '2.csv', '3.csv']
filepaths = [f'{files_dir}{parameter}-{file}' for file in files]

# Create the plot
plt.figure(figsize=(10, 6))

# Read each file and plot the data
for file in filepaths:
    # Read files and filter out rows where the paramter is less than 0
    df = pd.read_csv(file)
    df = df[df[parameter] >= 0]
    
    if parameter == 'sampling':
        df = df[df[parameter] <= 1]
    
    df = df.sort_values(by=parameter)
    
    # Bin data into 20 bins and calculate bin means and midpoints
    binned = pd.cut(df[parameter], bins=20)
    bin_means = df.groupby(binned).mean()
    bin_midpoints = [(interval.left + interval.right) / 2 for interval in binned.cat.categories]
    plt.plot(bin_midpoints, bin_means[metric], label=f"Prompt {file[-5]}")
    
    # Apply Savitzky-Golay filter
    # y_smooth = savgol_filter(df[metric], window_length=11, polyorder=2)  # Adjust parameters
    # plt.plot(df[parameter], y_smooth, label=f"Prompt {file[-5]}")
    
    # Apply rolling mean
    # df['smoothed_metric'] = df[metric].rolling(window=10, center=True).mean()  # Adjust window size
    # plt.plot(df[parameter], df['smoothed_metric'], label=f"Prompt {file[-5]}")
    
    # Apply Gaussian Filter
    # smoothed_data = gaussian_filter1d(df[metric], sigma=10)
    # plt.plot(df[parameter], smoothed_data, label=f"Prompt {file[-5]}")
    # plt.scatter(df[parameter], df[metric], s=10, alpha=0.6)
    
    # Unsmoothed plot
    # plt.plot(df[parameter], df[metric], label=f"Prompt {file[-5]}")
    
    # Scatter plot
    # plt.scatter(df[parameter], df[metric], label=f"Prompt {file[-5]}")


# Customize the plot
plt.xlabel(parameter)
plt.ylabel(metric)
plt.title(f'{metric} vs {parameter}')
plt.legend()

# Show the plot
plt.show()