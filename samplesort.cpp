#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <adiak.hpp>

// Add other necessary headers if required

#define threshold 10 // Adjust this threshold as needed

// Function to initialize data
void data_init(int* A, int arr_size) {
    CALI_MARK_BEGIN("data_init");
    srand(time(NULL));
    int i;
    for (i = 0; i < arr_size; ++i) {
        A[i] = random_float();
    }
    CALI_MARK_END("data_init");
}

// Function for small sorting
void smallSort(int A[], int arr_size) {
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");
    // Implement smallSort logic here
    CALI_MARK_END("comp_small");
    CALI_MARK_END("comp");
}

// Function for sampleSort
void sampleSort(int A[], int arr_size, int k, int num_proc, int rank, int size) {
    // if average bucket size is below a threshold switch to quicksort
    if (arr_size / k < threshold) {
        smallSort(A, arr_size);
        return;
    }

    // Step 1
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");

    int* S = (int*)malloc(num_proc * k * sizeof(int));

    // Select and sort samples
    // (implement code for selecting and sorting samples)

    // Select splitters
    int* splitters = (int*)malloc((num_proc + 1) * sizeof(int));
    splitters[0] = -INT_MAX;
    splitters[num_proc] = INT_MAX;

    // (implement code for selecting splitters)

    CALI_MARK_END("comp_large");

    // Step 2
    for (int i = 0; i < n; i++) {
        int j;
        for (j = 0; j < p; j++) {
            if (splitters[j] < A[i] && A[i] <= splitters[j + 1]) {
                break;
            }
        }
        // place A[i] in bucket bj
        // (implement code to place A[i] in the appropriate bucket)
    }

    free(S);
    free(splitters);

    CALI_MARK_END("comp");

    // Step 3 and concatenation
    // (implement code to concatenate sampleSort(b1), ..., sampleSort(bk))
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            printf("Usage: %s <array_size> <num_processes>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int arr_size = atoi(argv[1]);
    int num_proc = atoi(argv[2]);
    int k = arr_size / (2 * num_proc); // change?

    int* A = (int*)malloc(arr_size * sizeof(int));

    data_init(A, arr_size);

    sampleSort(A, arr_size, k, num_proc, rank, size);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "SampleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    //adiak::value("SizeOfDatatype", sizeOfDatatype); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    //adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
    //adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", size); // The number of processors (MPI ranks)
    //adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
    //adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
    adiak::value("group_num", 1); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI/Handwritten") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    free(A);

    MPI_Finalize();

    return 0;
}
