#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h>

// Define constants
#define MAX_THREADS 1024
#define MAX_BLOCKS 1024

int numThreads;
int numBlocks;
int totalValues;
int inputOption;

const char* inputTypes[4] = {"random", "sorted", "reverse_sorted", "1% perturbed"};

// Function to generate random float
float generateRandomFloat() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

// Function to fill the array
void fillArray(float *array, int size, int option) {
    srand(0);
    int perturb_count = size / 100;

    for (int i = 0; i < size; ++i) {
        switch (option) {
            case 1:
                array[i] = generateRandomFloat();
                break;
            case 2:
                array[i] = static_cast<float>(i);
                break;
            case 3:
                array[i] = static_cast<float>(size - i - 1);
                break;
            case 4:
                array[i] = static_cast<float>(i);
                if (i < perturb_count) {
                    int index = rand() % size;
                    array[index] = generateRandomFloat();
                }
                break;
        }
    }
}

// Function to check if the array is sorted
int isSorted(float* array, int size) {
    for (int i = 1; i < size; ++i) {
        if (array[i] < array[i - 1]) {
            return 0; // Not sorted
        }
    }
    return 1; // Sorted
}

// CUDA kernel for even sort
__global__ void sortEven(float *array, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index % 2 == 0 && index < size - 1) {
        if (array[index] > array[index + 1]) {
            float temp = array[index];
            array[index] = array[index + 1];
            array[index + 1] = temp;
        }
    }
}

// CUDA kernel for odd sort
__global__ void sortOdd(float *array, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index % 2 == 1 && index < size - 1) {
        if (array[index] > array[index + 1]) {
            float temp = array[index];
            array[index] = array[index + 1];
            array[index + 1] = temp;
        }
    }
}

int main(int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;

    // Parsing command line arguments
    numThreads = atoi(argv[1]);
    totalValues = atoi(argv[2]);
    inputOption = atoi(argv[3]);
    numBlocks = totalValues / numThreads;

    size_t arraySize = totalValues * sizeof(float);

    // Caliper ConfigManager setup
    cali::ConfigManager configManager;
    configManager.start();

    // Host array allocation and initialization
    float *hostArray = static_cast<float*>(malloc(arraySize));
    CALI_MARK_BEGIN("data_init");
    fillArray(hostArray, totalValues, inputOption);
    CALI_MARK_END("data_init");

    // Device array allocation
    float *deviceArray;
    cudaMalloc(&deviceArray, arraySize);

    // Copy data from host to device
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(deviceArray, hostArray, arraySize, cudaMemcpyHostToDevice);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");


    dim3 threadsPerBlock(numThreads);
    dim3 blocksPerGrid(numBlocks);

    // Sorting
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    for (int i = 0; i < totalValues / 2; ++i) {
        sortEven<<<blocksPerGrid, threadsPerBlock>>>(deviceArray, totalValues);
        sortOdd<<<blocksPerGrid, threadsPerBlock>>>(deviceArray, totalValues);
    }
    cudaDeviceSynchronize();
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

// Copy data back from device to host
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(hostArray, deviceArray, arraySize, cudaMemcpyDeviceToHost);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // Check if array is sorted
    CALI_MARK_BEGIN("correctness_check");
    int isSortedArray = isSorted(hostArray, totalValues);
    CALI_MARK_END("correctness_check");

    // Free device memory
    cudaFree(deviceArray);

    // Adiak value setup
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

    // Caliper output
    configManager.stop();
    configManager.flush();

    free(hostArray);
    return 0;
}
