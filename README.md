# Parallel Bitonic Sort Implementation
**SE3082 – Parallel Computing Assignment 03**  
**Student:** Janidu Pabasara (IT23294998)  
**Year 3, BSc (Hons) in Information Technology**  
**Semester 1, 2025**

---

## 📋 Table of Contents
- [Overview](#overview)
- [Algorithm Description](#algorithm-description)
- [Why Bitonic Sort for Parallelization?](#why-bitonic-sort-for-parallelization)
- [Project Structure](#project-structure)
- [Implementation Status](#implementation-status)
- [Compilation and Execution](#compilation-and-execution)
- [Performance Evaluation](#performance-evaluation)
- [Assignment Requirements](#assignment-requirements)
- [References](#references)

---

## 🎯 Overview

This project implements **Bitonic Sort**, a comparison-based sorting algorithm, using four different approaches:
1. **Serial C Implementation** (Baseline)
2. **OpenMP** (Shared-memory parallelization)
3. **MPI** (Distributed-memory parallelization)
4. **CUDA** (GPU parallelization)

**Problem Domain:** Sorting and Searching Algorithms

The goal is to demonstrate the performance improvements achieved through parallelization and compare the effectiveness of different parallel programming paradigms.

---

## 🔍 Algorithm Description

**Bitonic Sort** is a parallel sorting algorithm specifically designed for efficient implementation on architectures that support parallel comparisons, such as:
- Multi-core CPUs
- Computing clusters
- Graphics Processing Units (GPUs)

### How Bitonic Sort Works

The algorithm operates through the following key concepts:

1. **Bitonic Sequence**: A sequence that first increases monotonically and then decreases monotonically, or can be circularly shifted to become so.

2. **Bitonic Merge**: Takes a bitonic sequence and sorts it into either ascending or descending order through recursive compare-and-swap operations.

3. **Sorting Process**: 
   - Starts with individual elements (trivially bitonic)
   - Merges pairs into bitonic sequences of increasing size
   - Continues until the entire array is sorted

### Algorithm Complexity
- **Time Complexity (Serial)**: O(n log²n)
- **Parallel Time Complexity**: O(log²n) with O(n) processors
- **Number of Comparisons**: O(n log²n)
- **Space Complexity**: O(1) auxiliary space

---

## ⚡ Why Bitonic Sort for Parallelization?

Bitonic Sort is exceptionally suitable for parallelization due to several key characteristics:

### 1. **Independent Operations**
All compare-and-swap operations within a single phase are completely independent. They operate on predetermined index pairs, allowing simultaneous execution on different processing units.

### 2. **Regular Communication Pattern**
The algorithm has a fixed, predictable communication pattern. This makes it:
- Easy to map to parallel architectures
- Efficient for GPU implementation
- Suitable for distributed systems

### 3. **Data-Oblivious Nature**
The sequence of comparisons is independent of the input data. This means:
- No conditional branching based on data values
- Predictable memory access patterns
- Optimal for SIMD (Single Instruction, Multiple Data) architectures

### 4. **Scalability**
The algorithm scales well with:
- Number of processing cores (OpenMP)
- Number of compute nodes (MPI)
- GPU threads and blocks (CUDA)

### 5. **Theoretical Speedup**
With sufficient parallel resources:
- **Sequential stages**: O(log²n)
- **Parallel operations per stage**: O(n)
- **Theoretical speedup**: Up to O(n / log²n)

---

## 📁 Project Structure

```
Parallel_Bitonic_Sort_OpenMP_MPI_CUDA/
│
├── README.md                           # This file
│
├── Serial bitonic sort/                # Serial baseline implementation
│   ├── bitonic_sort_serial.c
│   ├── Makefile
│   └── README.md
│
├── OpenMP_Bitonic Sort/                # Shared-memory parallel version
│   ├── bitonic_sort_openmp.c
│   ├── Makefile
│   └── README.md
│
├── MPI parallel bitonic sort/          # Distributed-memory parallel version
│   ├── bitonic_sort_mpi.c
│   ├── Makefile
│   └── README.md
│
├── CUDA bitonic sort for int arrays/   # GPU parallel version
│   ├── bitonic_sort_cuda.cu
│   ├── Makefile
│   └── README.md
│
├── Documentation/                      # Performance analysis and reports
│   ├── Performance_Report.pdf
│   ├── graphs/
│   └── screenshots/
│
└── Data/                              # Test data files
    ├── input_samples/
    └── output_verification/
```

---

## 🚀 Implementation Status

### Phase 1: Algorithm Approval ✅
- [x] Algorithm selected: Bitonic Sort
- [x] Serial C code prepared
- [x] Approval email sent (Deadline: November 5, 2025)
- [ ] Approval received

### Phase 2: Implementation (60 marks)

#### Serial Implementation
- [ ] Core algorithm implemented
- [ ] Testing and verification
- [ ] Documentation

#### OpenMP Implementation (20 marks)
- [ ] Parallel regions identified
- [ ] Thread-level parallelization
- [ ] Load balancing strategy
- [ ] Testing with different thread counts
- [ ] Code documentation

#### MPI Implementation (20 marks)
- [ ] Data distribution strategy
- [ ] Inter-process communication
- [ ] Collective operations
- [ ] Testing with different process counts
- [ ] Code documentation

#### CUDA Implementation (20 marks)
- [ ] Kernel development
- [ ] Memory management (host/device)
- [ ] Thread block configuration
- [ ] Testing with different block sizes
- [ ] Code documentation

### Phase 3: Performance Evaluation (25 marks)
- [ ] OpenMP evaluation (6 marks)
- [ ] MPI evaluation (6 marks)
- [ ] CUDA evaluation (6 marks)
- [ ] Comparative analysis (7 marks)

### Phase 4: Documentation (15 marks)
- [ ] Parallelization strategies (4 marks)
- [ ] Runtime configurations (3 marks)
- [ ] Performance analysis (4 marks)
- [ ] Critical reflection (4 marks)

---

## 🔧 Compilation and Execution

### Prerequisites

#### For Serial and OpenMP:
```bash
# GCC compiler with OpenMP support
gcc --version  # Should be 4.2 or higher
```

#### For MPI:
```bash
# OpenMPI or MPICH
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
mpicc --version
```

#### For CUDA:
```bash
# NVIDIA CUDA Toolkit
nvcc --version  # Should match your GPU architecture
```

### Compilation Instructions

#### Serial Version
```bash
cd "Serial bitonic sort"
gcc -o bitonic_serial bitonic_sort_serial.c -O3
./bitonic_serial
```

#### OpenMP Version
```bash
cd "OpenMP_Bitonic Sort"
gcc -fopenmp -o bitonic_openmp bitonic_sort_openmp.c -O3
./bitonic_openmp
```

#### MPI Version
```bash
cd "MPI parallel bitonic sort"
mpicc -o bitonic_mpi bitonic_sort_mpi.c -O3
mpirun -np 4 ./bitonic_mpi
```

#### CUDA Version
```bash
cd "CUDA bitonic sort for int arrays"
nvcc -o bitonic_cuda bitonic_sort_cuda.cu -O3
./bitonic_cuda
```

---

## 📊 Performance Evaluation

### Metrics to Measure

1. **Execution Time**: Time taken to sort the array
2. **Speedup**: Serial time / Parallel time
3. **Efficiency**: Speedup / Number of processors
4. **Scalability**: Performance with increasing problem size

### Test Configurations

#### OpenMP Testing
- Thread counts: 1, 2, 4, 8, 16
- Array sizes: 2^16, 2^18, 2^20, 2^22
- Graphs required:
  - Threads vs Execution Time
  - Threads vs Speedup

#### MPI Testing
- Process counts: 1, 2, 4, 8, 16
- Array sizes: 2^16, 2^18, 2^20, 2^22
- Graphs required:
  - Processes vs Execution Time
  - Processes vs Speedup

#### CUDA Testing
- Block sizes: 64, 128, 256, 512, 1024
- Array sizes: 2^16, 2^18, 2^20, 2^22
- Graphs required:
  - Configuration vs Execution Time
  - Configuration vs Speedup

#### Comparative Analysis
- Compare all three implementations on same dataset
- Identify best-performing approach
- Analyze strengths and weaknesses

---

## 📝 Assignment Requirements

### Submission Deliverables

1. **Source Code**
   - Separate folders for each implementation
   - Makefiles with compilation instructions
   - Well-commented code

2. **Screenshots**
   - Execution with different configurations
   - Output verification
   - Performance monitoring

3. **Report (PDF, 3-4 pages)**
   - Parallelization strategies
   - Runtime configurations
   - Performance analysis
   - Critical reflection

4. **Data Files**
   - Input test data
   - Output files for verification

5. **Video Recording**
   - Demonstration of all three implementations

### Grading Breakdown
- **Part A**: Parallel Implementations (60 marks)
  - OpenMP: 20 marks
  - MPI: 20 marks
  - CUDA: 20 marks
- **Part B**: Performance Evaluation (25 marks)
- **Part C**: Documentation and Analysis (15 marks)

---

## 📚 References

1. **Batcher, K. E.** (1968). "Sorting networks and their applications". *Proceedings of the Spring Joint Computer Conference*, AFIPS '68 (Spring), pp. 307–314.

2. **Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.** (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.

3. **Knuth, D. E.** (1998). *The Art of Computer Programming, Volume 3: Sorting and Searching* (2nd ed.). Addison-Wesley.

4. **Sanders, P., & Träff, J. L.** (2006). "Parallel prefix (scan) algorithms for MPI". *Recent Advances in Parallel Virtual Machine and Message Passing Interface*.

5. **NVIDIA Corporation** (2023). *CUDA C++ Programming Guide*. Retrieved from https://docs.nvidia.com/cuda/

6. **OpenMP Architecture Review Board** (2021). *OpenMP Application Programming Interface*. Retrieved from https://www.openmp.org/

---

## 👨‍💻 Author

**Janidu Pabasara**  
Student ID: IT23294998  
BSc (Hons) in Information Technology, Year 3  
SLIIT - Sri Lanka Institute of Information Technology

---

## 📄 License

This project is submitted as part of academic coursework for SE3082 – Parallel Computing.  
All code is original work based on the canonical Bitonic Sort algorithm.

---

## 🔗 Quick Links

- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [OpenMP Specifications](https://www.openmp.org/)
- [MPI Forum](https://www.mpi-forum.org/)
- [Course Material - SE3082](mailto:nuwan.k@sliit.lk)

---

*Last Updated: November 25, 2025*
