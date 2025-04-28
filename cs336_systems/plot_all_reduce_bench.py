import matplotlib.pyplot as plt
import numpy as np

# NCCL timing data
data_sizes = [1, 10, 100, 1024]  # Data size in MB
process_counts = [2, 4, 6]       # Number of processes

# Time in ms for each configuration (data_size, process_count)
nccl_times = {
    2: [0.04, 0.07, 0.45, 3.88],
    4: [0.05, 0.10, 0.56, 5.20],
    6: [0.05, 0.13, 0.53, 4.89]
}

gloo_times = {
    2: [0.48, 4.12, 38.86, 429.73],
    4: [0.88, 6.47, 61.60, 914.61],
    6: [1.13, 6.28, 74.08, 1002.73]
}

# Create a new figure with a specific size
plt.figure(figsize=(10, 6))

# Set up colors and markers for different process counts
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', '^']

# Plot a line for each process count
for i, procs in enumerate(process_counts):
    plt.plot(data_sizes, gloo_times[procs],
             label=f'{procs} Processes',
             color=colors[i],
             marker=markers[i],
             linewidth=2,
             markersize=8)

# Set x-axis to log scale since data sizes vary by orders of magnitude
plt.xscale('log')
# Set y-axis to log scale to better visualize performance across orders of magnitude
plt.yscale('log')

# Add labels and title
plt.xlabel('Data Size (MB)', fontsize=12)
plt.ylabel('Average Time (ms)', fontsize=12)
plt.title('Gloo All-Reduce Performance', fontsize=14)

# Add grid for better readability
plt.grid(True, which="both", ls="-", alpha=0.2)

# Add custom x-tick labels
plt.xticks(data_sizes, [str(size) for size in data_sizes])

# Customize y-axis range for better visualization
plt.ylim(0.4, 1200)

# Add a legend
plt.legend(loc='best', fontsize=10)

# Annotate some data points for clarity
plt.annotate(f"{gloo_times[2][-1]} ms",
             xy=(data_sizes[-1], gloo_times[2][-1]),
             xytext=(data_sizes[-1]*0.8, gloo_times[2][-1]*0.7),
             arrowprops=dict(arrowstyle='->'))

plt.annotate(f"{gloo_times[6][-1]} ms",
             xy=(data_sizes[-1], gloo_times[6][-1]),
             xytext=(data_sizes[-1]*0.8, gloo_times[6][-1]*1.3),
             arrowprops=dict(arrowstyle='->'))

# Ensure the layout is tight
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Save the figure
plt.savefig('gloo_performance.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()