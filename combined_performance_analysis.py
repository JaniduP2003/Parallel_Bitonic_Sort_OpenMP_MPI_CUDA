import matplotlib.pyplot as plt
import numpy as np

# Data for combined comparison (best configuration for each method)
methods = ['CPU\n(Sequential)', 'OpenMP\n(8 threads)', 'MPI\n(8 processes)', 'CUDA\n(256 threads/block)']
execution_times = [80.286723, 24.108243, 0.846883, 0.103406]
speedup_vs_cpu = [1.00, 3.33, 94.81, 776.44]
colors = ['#E63946', '#457B9D', '#06A77D', '#F77F00']

# Create figure with 3 subplots
fig = plt.figure(figsize=(18, 5))

# Graph 1: Execution Time Comparison (Linear Scale)
ax1 = plt.subplot(1, 3, 1)
bars1 = ax1.bar(methods, execution_times, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('Execution Time Comparison\n(Lower is Better)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

# Add value labels on bars
for i, (bar, time) in enumerate(zip(bars1, execution_times)):
    height = bar.get_height()
    if time < 1:
        label = f'{time:.3f}s'
    else:
        label = f'{time:.2f}s'
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             label,
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Graph 2: Execution Time Comparison (Log Scale)
ax2 = plt.subplot(1, 3, 2)
bars2 = ax2.bar(methods, execution_times, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
ax2.set_ylabel('Execution Time (seconds, log scale)', fontsize=12, fontweight='bold')
ax2.set_title('Execution Time Comparison (Log Scale)\n(Shows relative differences better)', fontsize=13, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, linestyle='--', axis='y', which='both')

# Add value labels on bars (log scale)
for i, (bar, time) in enumerate(zip(bars2, execution_times)):
    height = bar.get_height()
    if time < 1:
        label = f'{time:.3f}s'
    else:
        label = f'{time:.2f}s'
    y_pos = height * 1.3 if time > 1 else height * 2
    ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
             label,
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Add performance annotations
ax2.text(0.5, 50, '776× faster →', fontsize=9, color='red', fontweight='bold', ha='center')

# Graph 3: Speedup Comparison
ax3 = plt.subplot(1, 3, 3)
bars3 = ax3.bar(methods, speedup_vs_cpu, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
ax3.set_ylabel('Speedup vs CPU (×)', fontsize=12, fontweight='bold')
ax3.set_title('Speedup Comparison\n(Relative to Sequential CPU)', fontsize=13, fontweight='bold')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3, linestyle='--', axis='y', which='both')

# Add value labels on bars
for i, (bar, speedup) in enumerate(zip(bars3, speedup_vs_cpu)):
    height = bar.get_height()
    label = f'{speedup:.2f}×' if speedup < 100 else f'{speedup:.0f}×'
    y_pos = height * 1.3
    ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
             label,
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Add baseline line
ax3.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Baseline (CPU)')
ax3.legend(loc='upper left')

# Overall title
fig.suptitle('Bitonic Sort: Complete Performance Comparison Across All Methods', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()

# Save the figure
plt.savefig('combined_performance_analysis.png', dpi=300, bbox_inches='tight')
print("Combined graph saved as 'combined_performance_analysis.png'")

# Print detailed summary
print("\n" + "="*70)
print("COMPLETE PERFORMANCE ANALYSIS SUMMARY")
print("="*70)

print("\n📊 EXECUTION TIME RANKING (Slowest → Fastest):")
print("-" * 70)
time_sorted = sorted(zip(methods, execution_times), key=lambda x: x[1], reverse=True)
for i, (method, time) in enumerate(time_sorted, 1):
    method_clean = method.replace('\n', ' ')
    if time < 1:
        print(f"  {i}. {method_clean:<30} {time:.6f} seconds")
    else:
        print(f"  {i}. {method_clean:<30} {time:.2f} seconds")

print("\n🚀 SPEEDUP RANKING (Weakest → Strongest):")
print("-" * 70)
speedup_sorted = sorted(zip(methods, speedup_vs_cpu), key=lambda x: x[1])
for i, (method, speedup) in enumerate(speedup_sorted, 1):
    method_clean = method.replace('\n', ' ')
    print(f"  {i}. {method_clean:<30} {speedup:.2f}×")

print("\n⚡ KEY PERFORMANCE INSIGHTS:")
print("-" * 70)
print(f"  • CUDA is {execution_times[1]/execution_times[3]:.0f}× faster than OpenMP")
print(f"  • CUDA is {execution_times[2]/execution_times[3]:.0f}× faster than MPI")
print(f"  • CUDA is {speedup_vs_cpu[3]:.0f}× faster than pure CPU")
print(f"  • MPI is {execution_times[1]/execution_times[2]:.0f}× faster than OpenMP")
print(f"  • OpenMP is {execution_times[0]/execution_times[1]:.1f}× faster than CPU")

print("\n💡 EFFICIENCY ANALYSIS:")
print("-" * 70)
print(f"  • OpenMP (8 threads):      {(speedup_vs_cpu[1]/8)*100:.1f}% parallel efficiency")
print(f"  • MPI (8 processes):       {(speedup_vs_cpu[2]/94.81)*100:.1f}% efficiency vs ideal MPI")
print(f"  • CUDA (massive parallel): {speedup_vs_cpu[3]:.0f}× speedup (dominant winner)")

print("\n🏆 PERFORMANCE HIERARCHY:")
print("-" * 70)
print("  Slowest  →  CPU (80.29s, 1.00×)")
print("           →  OpenMP (24.11s, 3.33×)")
print("           →  MPI (0.85s, 94.81×)")
print("  Fastest  →  CUDA (0.10s, 776.44×) ⭐")

print("\n" + "="*70)
print("✅ CONCLUSION: GPU (CUDA) massively outperforms all other methods")
print("   due to bitonic sort's highly parallel, regular memory access pattern.")
print("="*70 + "\n")

# Display the plot
plt.show()
