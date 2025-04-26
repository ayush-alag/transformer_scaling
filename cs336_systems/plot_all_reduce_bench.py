import matplotlib.pyplot as plt
import numpy as np

# NCCL timing data
data_sizes = [1, 10, 100, 1024]  # Data size in MB
process_counts = [2, 4, 6]       # Number of processes

# Time in ms for each configuration (data_size, process_count)
times = {
    2: [0.04, 0.07, 0.45, 3.88],
    4: [0.05, 0.10, 0.56, 5.20],
    6: [0.05, 0.13, 0.53, 4.89]
}

# Create a new figure with a specific size
plt.figure(figsize=(10, 6))

# Set up colors and markers for different process counts
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', '^']

# Plot a line for each process count
for i, procs in enumerate(process_counts):
    plt.plot(data_sizes, times[procs],
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
plt.title('NCCL All-Reduce Performance by Data Size and Process Count', fontsize=14)

# Add grid for better readability
plt.grid(True, which="both", ls="-", alpha=0.2)

# Add custom x-tick labels
plt.xticks(data_sizes, [str(size) for size in data_sizes])

# Customize y-axis range for better visualization
plt.ylim(0.03, 10)

# Add a legend
plt.legend(loc='best', fontsize=10)

# Annotate some data points for clarity
plt.annotate(f"{times[2][-1]} ms",
             xy=(data_sizes[-1], times[2][-1]),
             xytext=(data_sizes[-1]*0.8, times[2][-1]*0.7),
             arrowprops=dict(arrowstyle='->'))

plt.annotate(f"{times[6][-1]} ms",
             xy=(data_sizes[-1], times[6][-1]),
             xytext=(data_sizes[-1]*0.8, times[6][-1]*1.3),
             arrowprops=dict(arrowstyle='->'))

# Add text annotation explaining key insights
plt.figtext(0.5, 0.01,
            "Note: NCCL performance scales efficiently with both data size and process count",
            ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

# Ensure the layout is tight
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Save the figure
plt.savefig('nccl_performance.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()