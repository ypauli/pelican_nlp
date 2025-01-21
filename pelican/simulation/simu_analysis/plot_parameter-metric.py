import pandas as pd
import matplotlib.pyplot as plt

parameter = 'temperature'
metric = 'entropy'

# List of CSV filenames
files_dir = '/home/ubuntu/emilia/PELICAN/pelican/simulation/simu_analysis/data/'
files = ['0.csv', '1.csv', '2.csv', '3.csv']
filepaths = [f'{files_dir}{parameter}-{file}' for file in files]

# Create the plot
plt.figure(figsize=(10, 6))

# Read each file and plot the data
for file in filepaths:
    df = pd.read_csv(file)
    df = df.sort_values(by=parameter)
    plt.plot(df[parameter], df[metric], label=f"Prompt {file[-5]}")

# Customize the plot
plt.xlabel(parameter)
plt.ylabel(metric)
plt.title(f'{metric} vs {parameter}')
plt.legend()

# Show the plot
plt.show()

# TODO check cases where tempearture is not valid/negative