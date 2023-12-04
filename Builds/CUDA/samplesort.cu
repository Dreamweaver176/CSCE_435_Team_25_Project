#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <iostream>
#include <algorithm>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <thrust/sort.h>

#include <cuda_runtime.h>
#include <cuda.h>

int THREADS;
int BLOCKS;
int NUM_VALS;

/* Define Caliper region names */
const char* whole_computation = "whole_computation";
const char* data_init = "data_init";
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* correctness_check = "correctness_check";

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
};

void array_print(int *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
};

void array_fill(int *arr, int length)
{
    // for (int i = 0; i < length; ++i) {               // Sorted
    //     arr[i] = static_cast<int>(i);
    // }

    // srand(time(NULL));                               // Random
    // int i;
    // for (i = 0; i < length; ++i) {
    //     arr[i] = static_cast<int>(rand() % length);
    // }

    // for (int i = 0; i < length; ++i) {               // Reverse sorted
    //     arr[i] = static_cast<int>(length - i);
    // }

    srand(time(NULL));                             // 1%perturbed
    for(int i = 0; i < length; i++) {
        arr[i] = static_cast<int>(i);
        if (rand() % 100 == 1) {
            arr[i] *= static_cast<int>(rand() % 10 + 0.5);
        }
    }
};

const int RADIX_BITS = 8; // Number of bits to consider in each pass

__device__ int get_radix(int value, int bit) {
    return (value >> bit) & ((1 << RADIX_BITS) - 1);
}

__global__ void counting_sort(int* values, int* temp, int bit, int size, int num_blocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize counters
    int counters[1 << RADIX_BITS] = {0};

    // Count occurrences of each radix
    for (int i = tid; i < size; i += blockDim.x * num_blocks) {
        int radix = get_radix(values[i], bit);
        atomicAdd(&counters[radix], 1);
    }

    __syncthreads();

    // Prefix sum to determine the starting index for each radix
    for (int i = 1; i < (1 << RADIX_BITS); ++i) {
        counters[i] += counters[i - 1];
    }

    __syncthreads();

    // Move values to their sorted positions in temp array
    for (int i = size - 1; i >= 0; --i) {
        int radix = get_radix(values[i], bit);
        int index = counters[radix] - 1;
        temp[index] = values[i];
        atomicSub(&counters[radix], 1);
    }

    __syncthreads();

    // Copy values back to the original array
    for (int i = tid; i < size; i += blockDim.x * num_blocks) {
        values[i] = temp[i];
    }
}

void sample_sort(int* values) {
  CALI_CXX_MARK_FUNCTION;

  CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);

    int* dev_values;
    int* dev_temp;
    const int size = NUM_VALS;

    cudaMalloc((void**)&dev_values, size * sizeof(int));
    cudaMalloc((void**)&dev_temp, size * sizeof(int));

    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);

    cudaMemcpy(dev_values, values, size * sizeof(int), cudaMemcpyHostToDevice);

    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);

    // Iterate through each radix
    for (int bit = 0; bit < sizeof(int) * 8; bit += RADIX_BITS) {
        counting_sort<<<BLOCKS, THREADS>>>(dev_values, dev_temp, bit, size, BLOCKS);
    }

    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);

    cudaMemcpy(values, dev_values, size * sizeof(int), cudaMemcpyDeviceToHost);

    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    cudaFree(dev_values);
    cudaFree(dev_temp);
}

int main(int argc, char *argv[])
{
    CALI_CXX_MARK_FUNCTION;

    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    BLOCKS = NUM_VALS / THREADS;

    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    CALI_MARK_BEGIN(whole_computation);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    CALI_MARK_BEGIN(data_init);

    int *values = (int*) malloc( NUM_VALS * sizeof(int));
    array_fill(values, NUM_VALS);

    CALI_MARK_END(data_init);

    sample_sort(values);

    CALI_MARK_BEGIN(correctness_check);

    bool sorted = true;
    for (int i = 1; i < NUM_VALS; i++) {
      //printf("a[i]: %d\n", values[i]);
        if (values[i] < values[i-1]) {
            //printf("Error. Out of order sequence: %d found\n", values[i]);
            sorted = false;
        }
    }
    if (sorted) {
        printf("Array is in sorted order\n");
    }

    CALI_MARK_END(correctness_check);
    CALI_MARK_END(whole_computation);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "SampleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "Integer"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", "1%Perturbed"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    //adiak::value("num_procs", numtasks); // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 25); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI/Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    //free(values);
};
