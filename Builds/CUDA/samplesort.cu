#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <iostream>
#include <algorithm>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

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

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
};

void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
};

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
};

__global__ void merge(float* values, int num_vals, float* temp) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int left = 2 * idx * num_vals / (2 * blockDim.x * gridDim.x);
    int right = (2 * idx + 1) * num_vals / (2 * blockDim.x * gridDim.x);
    int end = (2 * idx + 2) * num_vals / (2 * blockDim.x * gridDim.x);
    
    if (end > num_vals) end = num_vals;

    int i = left;
    int j = right;
    int k = left;

    while (i < right && j < end) {
        if (values[i] <= values[j]) {
            temp[k++] = values[i++];
        } else {
            temp[k++] = values[j++];
        }
    }

    while (i < right) {
        temp[k++] = values[i++];
    }

    while (j < end) {
        temp[k++] = values[j++];
    }

    // Copy the merged data back to the original array
    for (int i = left; i < end; ++i) {
        values[i] = temp[i];
    }
};

void sample_sort(float* values, int num_vals, int num_threads, int num_blocks) {
    CALI_CXX_MARK_FUNCTION;

    CALI_MARK_BEGIN(comp);

    int block_size = num_threads;
    int grid_size = num_blocks;

    float* d_values;
    cudaMalloc((void**)&d_values, num_vals * sizeof(float));
    cudaMemcpy(d_values, values, num_vals * sizeof(float), cudaMemcpyHostToDevice);

    float* d_temp;
    cudaMalloc((void**)&d_temp, num_vals * sizeof(float));

    // Sort each block independently
    for (int i = 0; i < grid_size; ++i) {
        int offset = i * num_vals / grid_size;
        std::sort(values + offset, values + offset + num_vals / grid_size);
    }

    // Merge sorted blocks
    for (int size = num_vals / grid_size; size < num_vals; size *= 2) {
        for (int i = 0; i < grid_size; i += 2) {
            int offset = i * num_vals / grid_size;
            merge<<<1, block_size>>>(d_values + offset, num_vals, d_temp);
            cudaDeviceSynchronize();
        }
        cudaMemcpy(values, d_values, num_vals * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_values);
    cudaFree(d_temp);

    CALI_MARK_END(comp);
};

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

    float *values = (float*) malloc( NUM_VALS * sizeof(float));
    array_fill(values, NUM_VALS);

    CALI_MARK_END(data_init);

    sample_sort(values, NUM_VALS, THREADS, BLOCKS);

    CALI_MARK_BEGIN(correctness_check);

    bool sorted = true;
    for (int i = 1; i < NUM_VALS; i++) {
        if (values[i] < values[i-1]) {
            printf("Error. Out of order sequence: %f found\n", values[i]);
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
    adiak::value("Datatype", "Float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    //adiak::value("num_procs", numtasks); // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 25); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI/Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
};
