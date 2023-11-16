#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cuda.h>

//Defining variable inputs for sbatch jobfile
int BLOCKS;
int NUM_VALS;
int THREADS;
int OPTION;

const char* options[4] = {"random", "sorted", "reverse_sorted", "1%perturbed"};

float random_float() {
  return (float)rand()/(float)RAND_MAX;
}

void array_fill(float *array, int length, int choice) {
    if (choice == 1) {
        srand(0);
        for (int i = 0; i < length; ++i) {
            array[i] = random_float();
        }
    } 
    else if (choice == 2) {
        for (int i = 0; i < length; ++i) {
            array[i] = (float)i;
        }
    } 
    else if (choice == 3) {
        for (int i = 0; i < length; ++i) {
            array[i] = (float)length-1-i;
        }
    } 
    else if (choice == 4) {
        for (int i = 0; i < length; ++i) {
            array[i] = (float)i;
        }

        int p_count = length / 100;
        srand(0);
        for (int i = 0; i < p_count; ++i) {
            int index = rand() % length;
            array[index] = random_float();
        }
    }
}

int check(float* arr_values, int length) {
    for (int i = 1; i < length; ++i) {
        if (arr_values[i] < arr_values[i -1]) {
            return 0;
        }
    }
    return 1;
}

__global__ void even_sort(float *arr_values, int num_vals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i%2 == 0 && i < num_vals-1 ) {
        if (arr_values[i] > arr_values[i+1]) {
                float temp = arr_values[i];
                arr_values[i] = arr_values[i + 1];
                arr_values[i + 1] = temp;
        }                   
    }
    __syncthreads();
}

__global__ void odd_sort(float *arr_values, int num_vals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i%2 == 1 && i < num_vals-1 ) {
        if (arr_values[i] > arr_values[i+1]) {
                float temp = arr_values[i];
                arr_values[i] = arr_values[i + 1];
                arr_values[i + 1] = temp;
        }                   
    }
    __syncthreads();
}

int main(int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;

    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    OPTION = atoi(argv[3]);
    BLOCKS = NUM_VALS / THREADS;

    size_t size = NUM_VALS * sizeof(float);
    cali::ConfigManager mgr;
    mgr.start();

    float *values = (float*) malloc( NUM_VALS * sizeof(float));
    CALI_MARK_BEGIN("data_init");
    array_fill(values, NUM_VALS, OPTION);
    CALI_MARK_END("data_init");

    float *dev_values;
    cudaMalloc((void**) &dev_values, size);

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    dim3 threadsPerBlock(THREADS);
    dim3 numBlocks(BLOCKS);

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    for (int i = 0; i < NUM_VALS/2; ++i) {
        even_sort<<<BLOCKS, THREADS>>>(dev_values, NUM_VALS);
        odd_sort<<<BLOCKS, THREADS>>>(dev_values, NUM_VALS);
    }
    cudaDeviceSynchronize();
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");
    
    CALI_MARK_BEGIN("correctness_check");
    int correctness = check(values, NUM_VALS);
    CALI_MARK_END("correctness_check");

    cudaFree(dev_values);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "odd_even_sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", options[OPTION-1]); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 5); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online and Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    adiak::value("correctness", correctness); // Whether the dataset has been sorted (0, 1)

    // Flush Caliper output
    mgr.stop();
    mgr.flush();
}