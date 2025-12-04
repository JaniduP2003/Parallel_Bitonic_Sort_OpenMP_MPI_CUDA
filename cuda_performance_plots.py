import matplotlib.pyplot as plt
import numpy as np

# Data
threads_per_block = [32, 64, 128, 256, 512, 1024]
execution_time_ms = [235.135, 121.118, 115.121, 103.406, 112.963, 129.649]
speedup = [1.00, 1.94, 2.04, 2.27, 2.08, 1.81]

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Graph 1: Threads/block vs Execution Time
ax1.plot(threads_per_block, execution_time_ms, marker='o', linewidth=2, markersize=8, color='#2E86AB')
ax1.set_xlabel('Threads per Block', fontsize=12, fontweight='bold')
ax1.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
ax1.set_title('Threads per Block vs Execution Time', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xscale('log', base=2)
ax1.set_xticks(threads_per_block)
ax1.set_xticklabels(threads_per_block)

# Annotate minimum point
min_idx = execution_time_ms.index(min(execution_time_ms))
ax1.annotate(f'Min: {execution_time_ms[min_idx]:.2f} ms\n@ {threads_per_block[min_idx]} threads',
             xy=(threads_per_block[min_idx], execution_time_ms[min_idx]),
             xytext=(threads_per_block[min_idx]*0.5, execution_time_ms[min_idx]+30),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=10, color='red', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# Graph 2: Threads/block vs Speedup
ax2.plot(threads_per_block, speedup, marker='s', linewidth=2, markersize=8, color='#A23B72')
ax2.set_xlabel('Threads per Block', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup', fontsize=12, fontweight='bold')
ax2.set_title('Threads per Block vs Speedup', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xscale('log', base=2)
ax2.set_xticks(threads_per_block)
ax2.set_xticklabels(threads_per_block)
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')

# Annotate maximum speedup point
max_idx = speedup.index(max(speedup))
ax2.annotate(f'Peak: {speedup[max_idx]:.2f}x\n@ {threads_per_block[max_idx]} threads',
             xy=(threads_per_block[max_idx], speedup[max_idx]),
             xytext=(threads_per_block[max_idx]*1.5, speedup[max_idx]-0.3),
             arrowprops=dict(arrowstyle='->', color='green', lw=2),
             fontsize=10, color='green', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

# Overall title
fig.suptitle('CUDA Bitonic Sort Performance Analysis', fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()

# Save the figure
plt.savefig('cuda_performance_analysis.png', dpi=300, bbox_inches='tight')
print("Graph saved as 'cuda_performance_analysis.png'")

# Display the plot
plt.show()
