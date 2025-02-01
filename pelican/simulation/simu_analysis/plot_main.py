import plot_parameter_metric
import plot_functions
import matplotlib.pyplot as plt

# Initialize variables
files_dir = '/home/ubuntu/emilia/data_normal/'
files = ['0.csv', '1.csv', '2.csv', '3.csv']

parameters = ("temperature", "sampling", "context_span")
metrics = ("semantic_distance", "perplexity", "entropy")
    

plot_parameter_metric.plot_parameter_metric(files_dir, files, parameters, metrics)

# files_dir = '/home/ubuntu/data/data_full_prompt/'

# plot_parameter_metric.plot_parameter_metric(files_dir, files, "target_length", metrics)

# sensitivity_data = plot_functions.plot_sensitivities(files_dir, files, parameters, metrics)
# plot_functions.plot_correlation_heatmap(sensitivity_data, parameters, metrics)
# # plot_functions.plot_average_correlations(sensitivity_data, parameters, metrics)
# plot_functions.plot_effect_sizes_by_group(sensitivity_data, parameters, metrics)