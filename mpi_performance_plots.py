import matplotlib.pyplot as plt
import numpy as np

# Data
processes = [1, 2, 4, 8]
execution_time_sec = [5.720622, 2.383671, 1.140496, 0.846883]
speedup = [1.00, 2.40, 5.02, 6.75]

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Graph 1: Processes vs Execution Time
ax1.plot(processes, execution_time_sec, marker='o', linewidth=2.5, markersize=10, color='#06A77D', label='Execution Time')
ax1.set_xlabel('Number of Processes', fontsize=12, fontweight='bold')
ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('MPI Processes vs Execution Time', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(processes)

# Add value labels on the points
for i, (p, time) in enumerate(zip(processes, execution_time_sec)):
    ax1.annotate(f'{time:.2f}s', 
                 xy=(p, time), 
                 xytext=(0, 10), 
                 textcoords='offset points',
                 ha='center', 
                 fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

# Highlight improvements
ax1.annotate('Fast drop', 
             xy=(2, 2.383671), 
             xytext=(2.5, 3.5),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2),
             fontsize=10, 
             color='blue', 
             fontweight='bold')
ax1.annotate('Gentle drop', 
             xy=(8, 0.846883), 
             xytext=(6.5, 1.8),
             arrowprops=dict(arrowstyle='->', color='purple', lw=2),
             fontsize=10, 
             color='purple', 
             fontweight='bold')

# Graph 2: Processes vs Speedup
ax2.plot(processes, speedup, marker='D', linewidth=2.5, markersize=10, color='#D62828', label='Actual Speedup')

# Add ideal linear speedup line for comparison
ideal_speedup = processes
ax2.plot(processes, ideal_speedup, linestyle='--', linewidth=2, color='#F77F00', alpha=0.7, label='Ideal Linear Speedup')

ax2.set_xlabel('Number of Processes', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup', fontsize=12, fontweight='bold')
ax2.set_title('MPI Processes vs Speedup', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xticks(processes)
ax2.legend(loc='upper left', fontsize=10)

# Add value labels on the points
for i, (p, sp) in enumerate(zip(processes, speedup)):
    if p == 8:
        ax2.annotate(f'{sp:.2f}× (best)', 
                     xy=(p, sp), 
                     xytext=(0, -18), 
                     textcoords='offset points',
                     ha='center', 
                     fontsize=9,
                     fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    else:
        ax2.annotate(f'{sp:.2f}×', 
                     xy=(p, sp), 
                     xytext=(0, -15), 
                     textcoords='offset points',
                     ha='center', 
                     fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

# Annotate key observations
ax2.annotate('Nearly linear\nuntil 4 processes', 
             xy=(4, 5.02), 
             xytext=(2.5, 7),
             arrowprops=dict(arrowstyle='->', color='green', lw=2),
             fontsize=10, 
             color='green', 
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

ax2.annotate('Communication +\nSync overhead', 
             xy=(8, 6.75), 
             xytext=(5.5, 3),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=10, 
             color='red', 
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

# Overall title
fig.suptitle('MPI Bitonic Sort Performance Analysis', fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()

# Save the figure
plt.savefig('mpi_performance_analysis.png', dpi=300, bbox_inches='tight')
print("Graph saved as 'mpi_performance_analysis.png'")

# Print summary statistics
print("\n=== Performance Summary ===")
print(f"Best execution time: {min(execution_time_sec):.3f}s at {processes[execution_time_sec.index(min(execution_time_sec))]} processes")
print(f"Maximum speedup: {max(speedup):.2f}× at {processes[speedup.index(max(speedup))]} processes")
print(f"Parallel efficiency at 8 processes: {(speedup[-1]/processes[-1])*100:.1f}%")
print(f"Speedup improvement: {execution_time_sec[0]/execution_time_sec[-1]:.2f}× faster with 8 processes")

# Display the plot
plt.show()
