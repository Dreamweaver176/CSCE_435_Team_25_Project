#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <iostream>
#include <algorithm>
#include <thrust/device_vector.h>                           // ADDED
#include <thrust/copy.h>
#include <thrust/sort.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

int THREADS;
int BLOCKS;
int NUM_VALS;

// const char* bitonic_sort_step_region = "bitonic_sort_step";
// const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
// const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";
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

__global__ void sortBlock(float* block, int block_size) {
    thrust::device_ptr<float> dev_ptr(block);
    thrust::sort(dev_ptr, dev_ptr + block_size);
};

void sample_sort(float* values, int num_vals, int num_threads, int num_blocks) {
    int block_size = num_vals / num_blocks;

    // Allocate device memory for the input values
    float* d_values;
    cudaMalloc((void**)&d_values, num_vals * sizeof(float));
    cudaMemcpy(d_values, values, num_vals * sizeof(float), cudaMemcpyHostToDevice);

    // Sort each block in parallel
    for (int i = 0; i < num_blocks; i++) {
        float* d_block = d_values + i * block_size;
        sortBlock<<<1, 1>>>(d_block, block_size);
    }

    // Allocate host memory for the sample values
    float* samples = (float*)malloc(num_blocks * sizeof(float));

    // Select and copy samples from each block to host
    for (int i = 0; i < num_blocks; i++) {
        float* d_block = d_values + i * block_size;
        cudaMemcpy(samples + i, d_block + block_size / 2, sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Sort the samples
    thrust::device_vector<float> dev_samples(samples, samples + num_blocks);
    thrust::sort(dev_samples.begin(), dev_samples.end());

    // Determine split points for merging
    float* split_points = thrust::raw_pointer_cast(dev_samples.data());
    float* d_split_points;
    cudaMalloc((void**)&d_split_points, num_blocks * sizeof(float));
    cudaMemcpy(d_split_points, split_points, num_blocks * sizeof(float), cudaMemcpyHostToDevice);

    // Merge the blocks based on split points
    float* d_result;
    cudaMalloc((void**)&d_result, num_vals * sizeof(float));
    cudaDeviceSynchronize();

    // Implement merge based on split points (not provided here)

    // Copy the sorted result back to host
    cudaMemcpy(values, d_result, num_vals * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup device memory
    cudaFree(d_values);
    cudaFree(d_split_points);
    cudaFree(d_result);
};

int main(int argc, char *argv[])
{
    //CALI_CXX_MARK_FUNCTION;                                                PROBABLY UNNEEDED

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

    sample_sort(values, NUM_VALS, THREADS, BLOCKS); // Inplace                                                  BLOCKS WRONG?

    CALI_MARK_BEGIN(correctness_check);

    bool sorted = true;
    for (int i = 1; i < NUM_VALS; i++) {
        if (values[i] < values[i-1]) {
            printf("Error. Out of order sequence: %d found\n", values[i]);
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
    adiak::value("implementation_source", "AI/Handwritten") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
};
