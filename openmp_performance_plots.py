import matplotlib.pyplot as plt
import numpy as np

# Data
threads = [1, 2, 4, 8]
execution_time_sec = [80.286723, 41.455774, 35.552498, 24.108243]
speedup = [1.00, 1.94, 2.26, 3.33]

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Graph 1: Threads vs Execution Time
ax1.plot(threads, execution_time_sec, marker='o', linewidth=2.5, markersize=10, color='#E63946', label='Execution Time')
ax1.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('OpenMP Threads vs Execution Time', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(threads)

# Add value labels on the points
for i, (t, time) in enumerate(zip(threads, execution_time_sec)):
    ax1.annotate(f'{time:.2f}s', 
                 xy=(t, time), 
                 xytext=(0, 10), 
                 textcoords='offset points',
                 ha='center', 
                 fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))

# Highlight the improvement
ax1.annotate('Steep drop', 
             xy=(1.5, 60), 
             fontsize=10, 
             color='green', 
             fontweight='bold',
             ha='center')
ax1.annotate('Good gain', 
             xy=(8, 24.108243), 
             xytext=(7, 30),
             arrowprops=dict(arrowstyle='->', color='green', lw=2),
             fontsize=10, 
             color='green', 
             fontweight='bold')

# Graph 2: Threads vs Speedup
ax2.plot(threads, speedup, marker='s', linewidth=2.5, markersize=10, color='#457B9D', label='Actual Speedup')

# Add ideal linear speedup line for comparison
ideal_speedup = threads
ax2.plot(threads, ideal_speedup, linestyle='--', linewidth=2, color='#F1A208', alpha=0.7, label='Ideal Linear Speedup')

ax2.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup', fontsize=12, fontweight='bold')
ax2.set_title('OpenMP Threads vs Speedup', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xticks(threads)
ax2.legend(loc='upper left', fontsize=10)

# Add value labels on the points
for i, (t, sp) in enumerate(zip(threads, speedup)):
    ax2.annotate(f'{sp:.2f}×', 
                 xy=(t, sp), 
                 xytext=(0, -15), 
                 textcoords='offset points',
                 ha='center', 
                 fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

# Annotate sub-linear nature
ax2.annotate('Sub-linear speedup\n(overhead + sync)', 
             xy=(8, 3.33), 
             xytext=(5.5, 6),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=10, 
             color='red', 
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

# Overall title
fig.suptitle('OpenMP Bitonic Sort Performance Analysis', fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()

# Save the figure
plt.savefig('openmp_performance_analysis.png', dpi=300, bbox_inches='tight')
print("Graph saved as 'openmp_performance_analysis.png'")

# Print summary statistics
print("\n=== Performance Summary ===")
print(f"Best execution time: {min(execution_time_sec):.2f}s at {threads[execution_time_sec.index(min(execution_time_sec))]} threads")
print(f"Maximum speedup: {max(speedup):.2f}× at {threads[speedup.index(max(speedup))]} threads")
print(f"Parallel efficiency at 8 threads: {(speedup[-1]/threads[-1])*100:.1f}%")

# Display the plot
plt.show()
