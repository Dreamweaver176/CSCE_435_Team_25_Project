#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <mpi.h>
#include <time.h>
#include <adiak.hpp>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>

#include <iostream>
#include <vector>
#include <algorithm>

// Adjustable Threshold
#define threshold 10

// Function to initialize data
void dataInit(int* A, int arr_size) {
    CALI_MARK_BEGIN("data_init");
    srand(time(NULL));
    int i;
    for (i = 0; i < arr_size; ++i) {
        A[i] = rand();
    }
    CALI_MARK_END("data_init");
}

// Function for odd even transposition
void oddEvenTranspositionSort(std::vector<int>& data) {
    bool isSorted = false;
    int n = data.size();

    while (!isSorted) {
        isSorted = true;

        // Perform odd phase
        for (int i = 1; i <= n - 2; i += 2) {
            if (data[i] > data[i + 1]) {
                std::swap(data[i], data[i + 1]);
                isSorted = false;
            }
        }

        // Perform even phase
        for (int i = 0; i <= n - 2; i += 2) {
            if (data[i] > data[i + 1]) {
                std::swap(data[i], data[i + 1]);
                isSorted = false;
            }
        }
    }
}


int main(int argc, char** argv) {

    std::vector<int> data = {34, 21, 45, 12, 11, 9, 67};
    std::cout << "Original array:\n";
    for (int i : data) {
        std::cout << i << " ";
    }
    std::cout << "\n";

    oddEvenTranspositionSort(data);

    std::cout << "Sorted array:\n";
    for (int i : data) {
        std::cout << i << " ";
    }
    std::cout << "\n";

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "SampleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", "4"); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", "1000"); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", size); // The number of processors (MPI ranks)
    //adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
    //adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
    adiak::value("group_num", 1); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI/Handwritten") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    free(A);

    MPI_Finalize();
    CALI_MARK_END("whole_computation");
    printf("9\n");
    return 0;
}
