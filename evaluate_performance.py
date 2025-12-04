#!/usr/bin/env python3
"""
Bitonic Sort Performance Evaluation Script
Automatically runs serial, OpenMP, and MPI versions and generates comparison graphs.
"""

import subprocess
import re
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Configuration
SERIAL_EXECUTABLE = "./Serial bitonic sort/Serial_Botonic_Sort"
OPENMP_EXECUTABLE = "./OpenMP_Bitonic Sort/openMp_bitonic_sort"
MPI_EXECUTABLE = "./MPI parallel bitonic sort/MPI_bitonic_Sort"

OPENMP_THREADS = [1, 2, 4, 8]
MPI_PROCESSES = [1, 2, 4, 8]

OUTPUT_DIR = "./evaluation_results"


class ExecutionError(Exception):
    """Custom exception for execution errors"""
    pass


def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"✓ Created output directory: {OUTPUT_DIR}")


def extract_execution_time(output: str) -> Optional[float]:
    """
    Extract execution time from program output.
    Assumes the time is printed as the last number in the output.
    Looks for patterns like "Time: X.XXX seconds" or just "X.XXX"
    """
    # Try to find time patterns
    patterns = [
        r'Time:\s*([\d.]+)\s*seconds',
        r'Execution time:\s*([\d.]+)',
        r'Time taken:\s*([\d.]+)',
        r'([\d.]+)\s*seconds',
        r'([\d.]+)\s*ms',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            time_val = float(match.group(1))
            # Convert ms to seconds if needed
            if 'ms' in pattern.lower():
                time_val = time_val / 1000.0
            return time_val
    
    # Fallback: try to get the last floating point number
    numbers = re.findall(r'[\d.]+', output)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None


def run_serial() -> float:
    """Run the serial version and return execution time"""
    print("\n" + "="*60)
    print("Running Serial Version")
    print("="*60)
    
    if not os.path.exists(SERIAL_EXECUTABLE):
        raise ExecutionError(f"Serial executable not found: {SERIAL_EXECUTABLE}")
    
    try:
        result = subprocess.run(
            [SERIAL_EXECUTABLE],
            capture_output=True,
            text=True,
            timeout=300,
            check=True
        )
        
        output = result.stdout + result.stderr
        print(output)
        
        exec_time = extract_execution_time(output)
        if exec_time is None:
            raise ExecutionError("Could not extract execution time from serial output")
        
        print(f"✓ Serial execution time: {exec_time:.6f} seconds")
        return exec_time
        
    except subprocess.TimeoutExpired:
        raise ExecutionError("Serial execution timed out")
    except subprocess.CalledProcessError as e:
        raise ExecutionError(f"Serial execution failed: {e}")


def run_openmp(threads: int) -> float:
    """Run OpenMP version with specified number of threads"""
    print(f"\n→ Running OpenMP with {threads} threads...")
    
    if not os.path.exists(OPENMP_EXECUTABLE):
        raise ExecutionError(f"OpenMP executable not found: {OPENMP_EXECUTABLE}")
    
    try:
        # Set OMP_NUM_THREADS environment variable
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(threads)
        
        result = subprocess.run(
            [OPENMP_EXECUTABLE, str(threads)],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
            env=env
        )
        
        output = result.stdout + result.stderr
        exec_time = extract_execution_time(output)
        
        if exec_time is None:
            raise ExecutionError(f"Could not extract execution time for OpenMP with {threads} threads")
        
        print(f"  Time: {exec_time:.6f} seconds")
        return exec_time
        
    except subprocess.TimeoutExpired:
        raise ExecutionError(f"OpenMP execution timed out with {threads} threads")
    except subprocess.CalledProcessError as e:
        raise ExecutionError(f"OpenMP execution failed with {threads} threads: {e}")


def run_mpi(processes: int, use_oversubscribe: bool = False) -> float:
    """Run MPI version with specified number of processes"""
    print(f"\n→ Running MPI with {processes} processes...")
    
    if not os.path.exists(MPI_EXECUTABLE):
        raise ExecutionError(f"MPI executable not found: {MPI_EXECUTABLE}")
    
    try:
        # Build command with optional oversubscribe flag
        cmd = ['mpirun', '-np', str(processes)]
        if use_oversubscribe:
            cmd.append('--oversubscribe')
        cmd.append(MPI_EXECUTABLE)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            check=True
        )
        
        output = result.stdout + result.stderr
        exec_time = extract_execution_time(output)
        
        if exec_time is None:
            raise ExecutionError(f"Could not extract execution time for MPI with {processes} processes")
        
        print(f"  Time: {exec_time:.6f} seconds")
        return exec_time
        
    except subprocess.TimeoutExpired:
        raise ExecutionError(f"MPI execution timed out with {processes} processes")
    except subprocess.CalledProcessError as e:
        raise ExecutionError(f"MPI execution failed with {processes} processes: {e}")
    except FileNotFoundError:
        raise ExecutionError("mpirun not found. Make sure MPI is installed.")


def collect_data() -> Tuple[float, Dict[int, float], Dict[int, float]]:
    """
    Run all experiments and collect timing data.
    Returns: (serial_time, openmp_results, mpi_results)
    """
    # Run serial version
    serial_time = run_serial()
    
    # Run OpenMP versions
    print("\n" + "="*60)
    print("Running OpenMP Versions")
    print("="*60)
    openmp_results = {}
    for threads in OPENMP_THREADS:
        try:
            openmp_results[threads] = run_openmp(threads)
        except ExecutionError as e:
            print(f"✗ Error with {threads} threads: {e}")
            openmp_results[threads] = None
    
    # Run MPI versions
    print("\n" + "="*60)
    print("Running MPI Versions")
    print("="*60)
    mpi_results = {}
    for processes in MPI_PROCESSES:
        try:
            mpi_results[processes] = run_mpi(processes)
        except ExecutionError as e:
            print(f"✗ Error with {processes} processes: {e}")
            mpi_results[processes] = None
    
    return serial_time, openmp_results, mpi_results


def calculate_speedup(serial_time: float, parallel_times: Dict[int, float]) -> Dict[int, float]:
    """Calculate speedup for each parallel configuration"""
    speedup = {}
    for config, time in parallel_times.items():
        if time is not None and time > 0:
            speedup[config] = serial_time / time
        else:
            speedup[config] = None
    return speedup


def plot_openmp_results(openmp_results: Dict[int, float], openmp_speedup: Dict[int, float]):
    """Generate OpenMP performance graphs"""
    threads = [t for t in OPENMP_THREADS if openmp_results.get(t) is not None]
    times = [openmp_results[t] for t in threads]
    speedups = [openmp_speedup[t] for t in threads]
    
    if not threads:
        print("⚠ No valid OpenMP results to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Execution time plot
    ax1.plot(threads, times, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Number of Threads', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_title('OpenMP: Threads vs Execution Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(threads)
    
    # Speedup plot
    ax2.plot(threads, speedups, 'o-', linewidth=2, markersize=8, color='#A23B72', label='Actual Speedup')
    ax2.plot(threads, threads, '--', linewidth=2, color='#F18F01', alpha=0.6, label='Ideal Speedup')
    ax2.set_xlabel('Number of Threads', fontsize=12)
    ax2.set_ylabel('Speedup', fontsize=12)
    ax2.set_title('OpenMP: Threads vs Speedup', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(threads)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/openmp_performance.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/openmp_performance.png")
    plt.close()


def plot_mpi_results(mpi_results: Dict[int, float], mpi_speedup: Dict[int, float]):
    """Generate MPI performance graphs"""
    processes = [p for p in MPI_PROCESSES if mpi_results.get(p) is not None]
    times = [mpi_results[p] for p in processes]
    speedups = [mpi_speedup[p] for p in processes]
    
    if not processes:
        print("⚠ No valid MPI results to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Execution time plot
    ax1.plot(processes, times, 'o-', linewidth=2, markersize=8, color='#06A77D')
    ax1.set_xlabel('Number of Processes', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_title('MPI: Processes vs Execution Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(processes)
    
    # Speedup plot
    ax2.plot(processes, speedups, 'o-', linewidth=2, markersize=8, color='#D62246', label='Actual Speedup')
    ax2.plot(processes, processes, '--', linewidth=2, color='#F18F01', alpha=0.6, label='Ideal Speedup')
    ax2.set_xlabel('Number of Processes', fontsize=12)
    ax2.set_ylabel('Speedup', fontsize=12)
    ax2.set_title('MPI: Processes vs Speedup', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(processes)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/mpi_performance.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/mpi_performance.png")
    plt.close()


def plot_comparative_results(serial_time: float, 
                            openmp_results: Dict[int, float], 
                            mpi_results: Dict[int, float],
                            openmp_speedup: Dict[int, float],
                            mpi_speedup: Dict[int, float]):
    """Generate comparative graphs for all approaches"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Execution time comparison
    openmp_threads = [t for t in OPENMP_THREADS if openmp_results.get(t) is not None]
    openmp_times = [openmp_results[t] for t in openmp_threads]
    
    mpi_processes = [p for p in MPI_PROCESSES if mpi_results.get(p) is not None]
    mpi_times = [mpi_results[p] for p in mpi_processes]
    
    if openmp_threads:
        ax1.plot(openmp_threads, openmp_times, 'o-', linewidth=2, markersize=8, 
                label='OpenMP', color='#2E86AB')
    if mpi_processes:
        ax1.plot(mpi_processes, mpi_times, 's-', linewidth=2, markersize=8, 
                label='MPI', color='#06A77D')
    
    ax1.axhline(y=serial_time, color='#E63946', linestyle='--', linewidth=2, 
               label=f'Serial ({serial_time:.4f}s)')
    ax1.set_xlabel('Number of Threads/Processes', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Speedup comparison
    openmp_speedup_values = [openmp_speedup[t] for t in openmp_threads]
    mpi_speedup_values = [mpi_speedup[p] for p in mpi_processes]
    
    if openmp_threads:
        ax2.plot(openmp_threads, openmp_speedup_values, 'o-', linewidth=2, markersize=8, 
                label='OpenMP', color='#2E86AB')
    if mpi_processes:
        ax2.plot(mpi_processes, mpi_speedup_values, 's-', linewidth=2, markersize=8, 
                label='MPI', color='#06A77D')
    
    # Ideal speedup line
    max_config = max(max(openmp_threads) if openmp_threads else 0,
                     max(mpi_processes) if mpi_processes else 0)
    ideal_x = list(range(1, max_config + 1))
    ax2.plot(ideal_x, ideal_x, '--', linewidth=2, color='#F18F01', 
            alpha=0.6, label='Ideal Speedup')
    
    ax2.set_xlabel('Number of Threads/Processes', fontsize=12)
    ax2.set_ylabel('Speedup', fontsize=12)
    ax2.set_title('Speedup Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparative_performance.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/comparative_performance.png")
    plt.close()


def print_summary_table(serial_time: float,
                       openmp_results: Dict[int, float],
                       mpi_results: Dict[int, float],
                       openmp_speedup: Dict[int, float],
                       mpi_speedup: Dict[int, float]):
    """Print summary table of all results"""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"\nSerial Execution Time: {serial_time:.6f} seconds")
    
    # OpenMP results
    print("\n" + "-"*80)
    print("OpenMP Results:")
    print("-"*80)
    print(f"{'Threads':<10} {'Time (s)':<15} {'Speedup':<15} {'Efficiency (%)':<15}")
    print("-"*80)
    for threads in OPENMP_THREADS:
        time = openmp_results.get(threads)
        speedup = openmp_speedup.get(threads)
        if time is not None and speedup is not None:
            efficiency = (speedup / threads) * 100
            print(f"{threads:<10} {time:<15.6f} {speedup:<15.2f} {efficiency:<15.2f}")
        else:
            print(f"{threads:<10} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    # MPI results
    print("\n" + "-"*80)
    print("MPI Results:")
    print("-"*80)
    print(f"{'Processes':<10} {'Time (s)':<15} {'Speedup':<15} {'Efficiency (%)':<15}")
    print("-"*80)
    for processes in MPI_PROCESSES:
        time = mpi_results.get(processes)
        speedup = mpi_speedup.get(processes)
        if time is not None and speedup is not None:
            efficiency = (speedup / processes) * 100
            print(f"{processes:<10} {time:<15.6f} {speedup:<15.2f} {efficiency:<15.2f}")
        else:
            print(f"{processes:<10} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    print("="*80)


def save_results_to_csv(serial_time: float,
                        openmp_results: Dict[int, float],
                        mpi_results: Dict[int, float],
                        openmp_speedup: Dict[int, float],
                        mpi_speedup: Dict[int, float]):
    """Save all results to CSV files"""
    
    # OpenMP results
    openmp_data = []
    for threads in OPENMP_THREADS:
        time = openmp_results.get(threads)
        speedup = openmp_speedup.get(threads)
        if time is not None and speedup is not None:
            efficiency = (speedup / threads) * 100
            openmp_data.append({
                'Threads': threads,
                'Time (s)': time,
                'Speedup': speedup,
                'Efficiency (%)': efficiency
            })
    
    if openmp_data:
        df_openmp = pd.DataFrame(openmp_data)
        df_openmp.to_csv(f"{OUTPUT_DIR}/openmp_results.csv", index=False)
        print(f"✓ Saved: {OUTPUT_DIR}/openmp_results.csv")
    
    # MPI results
    mpi_data = []
    for processes in MPI_PROCESSES:
        time = mpi_results.get(processes)
        speedup = mpi_speedup.get(processes)
        if time is not None and speedup is not None:
            efficiency = (speedup / processes) * 100
            mpi_data.append({
                'Processes': processes,
                'Time (s)': time,
                'Speedup': speedup,
                'Efficiency (%)': efficiency
            })
    
    if mpi_data:
        df_mpi = pd.DataFrame(mpi_data)
        df_mpi.to_csv(f"{OUTPUT_DIR}/mpi_results.csv", index=False)
        print(f"✓ Saved: {OUTPUT_DIR}/mpi_results.csv")
    
    # Summary
    summary_data = {
        'Implementation': ['Serial'],
        'Configuration': ['N/A'],
        'Time (s)': [serial_time],
        'Speedup': [1.0],
        'Efficiency (%)': [100.0]
    }
    
    for threads in OPENMP_THREADS:
        time = openmp_results.get(threads)
        speedup = openmp_speedup.get(threads)
        if time is not None and speedup is not None:
            summary_data['Implementation'].append('OpenMP')
            summary_data['Configuration'].append(f'{threads} threads')
            summary_data['Time (s)'].append(time)
            summary_data['Speedup'].append(speedup)
            summary_data['Efficiency (%)'].append((speedup / threads) * 100)
    
    for processes in MPI_PROCESSES:
        time = mpi_results.get(processes)
        speedup = mpi_speedup.get(processes)
        if time is not None and speedup is not None:
            summary_data['Implementation'].append('MPI')
            summary_data['Configuration'].append(f'{processes} processes')
            summary_data['Time (s)'].append(time)
            summary_data['Speedup'].append(speedup)
            summary_data['Efficiency (%)'].append((speedup / processes) * 100)
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(f"{OUTPUT_DIR}/summary_results.csv", index=False)
    print(f"✓ Saved: {OUTPUT_DIR}/summary_results.csv")


def main():
    """Main execution function"""
    print("="*80)
    print("BITONIC SORT PERFORMANCE EVALUATION")
    print("="*80)
    
    # Ensure output directory exists
    ensure_output_dir()
    
    try:
        # Collect all performance data
        serial_time, openmp_results, mpi_results = collect_data()
        
        # Calculate speedups
        openmp_speedup = calculate_speedup(serial_time, openmp_results)
        mpi_speedup = calculate_speedup(serial_time, mpi_results)
        
        # Generate graphs
        print("\n" + "="*60)
        print("Generating Graphs")
        print("="*60)
        
        plot_openmp_results(openmp_results, openmp_speedup)
        plot_mpi_results(mpi_results, mpi_speedup)
        plot_comparative_results(serial_time, openmp_results, mpi_results,
                                openmp_speedup, mpi_speedup)
        
        # Save results to CSV
        print("\n" + "="*60)
        print("Saving Results to CSV")
        print("="*60)
        save_results_to_csv(serial_time, openmp_results, mpi_results,
                           openmp_speedup, mpi_speedup)
        
        # Print summary
        print_summary_table(serial_time, openmp_results, mpi_results,
                           openmp_speedup, mpi_speedup)
        
        print("\n✓ Evaluation completed successfully!")
        print(f"✓ All results saved to: {OUTPUT_DIR}/")
        
    except ExecutionError as e:
        print(f"\n✗ Execution error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
