#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int total_elements;
int fill_option;

const char* fill_methods[4] = {"random", "sorted", "reverse_sorted", "1% perturbed"};

float generate_random_float() {
    return (float)rand() / (float)RAND_MAX;
}

void populate_array(float *array, int size, int seed, int method) {
    srand(seed);
    int i;
    if (method == 1) { // Random
        for (i = 0; i < size; ++i) {
            array[i] = generate_random_float();
        }
    } else if (method == 2) { // Sorted
        for (i = 0; i < size; ++i) {
            array[i] = (float)seed + i;
        }
    } else if (method == 3) { // Reverse Sorted
        for (i = 0; i < size; ++i) {
            array[i] = (float)seed + size - 1 - i;
        }
    } else if (method == 4) { // 1% Perturbed
        for (i = 0; i < size; ++i) {
            array[i] = (float)i;
        }
        int perturbations = size / 100;
        for (i = 0; i < perturbations; ++i) {
            int idx = rand() % size;
            array[idx] = generate_random_float();
        }
    }
}

int float_compare(const void *a, const void *b) {
    return (*(float*)a - *(float*)b);
}

void merge_arrays(float *source, float *temp1, float *temp2, int array_size, int is_high) {
    int x = 0, y = 0, z = 0, i;

    if (is_high) {
        for (i = array_size - 1; i >= 0; --i) {
            if (source[i] >= temp1[i]) {
                temp2[z++] = source[i--];
            } else {
                temp2[z++] = temp1[i--];
            }
        }
    } else {
        for (i = 0; i < array_size; ++i) {
            if (source[x] <= temp1[y]) {
                temp2[z++] = source[x++];
            } else {
                temp2[z++] = temp1[y++];
            }
        }
    }

    memcpy(source, temp2, array_size * sizeof(float));
}

void sorting_iteration(float *data, float *temp1, float *temp2, int array_size, int phase, int even_partner, int odd_partner, int process_rank, int num_processes, MPI_Comm comm) {
    MPI_Status status;

    if (phase % 2 == 0) { // Even phase
        if (even_partner != MPI_PROC_NULL) {
            MPI_Sendrecv(data, array_size, MPI_FLOAT, even_partner, 0, temp1, array_size, MPI_FLOAT, even_partner, 0, comm, &status);
            merge_arrays(data, temp1, temp2, array_size, process_rank % 2);
        }
    } else { // Odd phase
        if (odd_partner != MPI_PROC_NULL) {
            MPI_Sendrecv(data, array_size, MPI_FLOAT, odd_partner, 0, temp1, array_size, MPI_FLOAT, odd_partner, 0, comm, &status);
            merge_arrays(data, temp1, temp2, array_size, !(process_rank % 2));
        }
    }
}

void odd_even_sort(float *data, int array_size, int process_rank, int num_processes, MPI_Comm comm) {
    int even_partner, odd_partner;
    float *temp1 = (float*)malloc(array_size * sizeof(float));
    float *temp2 = (float*)malloc(array_size * sizeof(float));

    even_partner = (process_rank % 2 == 0) ? process_rank + 1 : process_rank - 1;
    odd_partner = (process_rank % 2 == 0) ? process_rank - 1 : process_rank + 1;

    if (even_partner == num_processes || even_partner < 0) even_partner = MPI_PROC_NULL;
    if (odd_partner < 0) odd_partner = MPI_PROC_NULL;

    qsort(data, array_size, sizeof(float), float_compare);

    int phase;
    for (phase = 0; phase < num_processes; ++phase) {
        sorting_iteration(data, temp1, temp2, array_size, phase, even_partner, odd_partner, process_rank, num_processes, comm);
    }

    free(temp1);
    free(temp2);
}

void distribute_data(float *global_data, float *local_data, int segment_size, int rank) {
    for (int i = 0; i < segment_size; ++i) {
        global_data[i + segment_size * rank] = local_data[i];
    }
}

int verify_sorted(float *data, int size) {
    for (int i = 1; i < size; ++i) {
        if (data[i] < data[i - 1]) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;

    int num_processes, process_rank;
    total_elements = atoi(argv[1]);
    fill_option = atoi(argv[2]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    cali::ConfigManager cali_manager;
    cali_manager.start();

    int segment_size = total_elements / num_processes;
    float *local_data = (float*)malloc(segment_size * sizeof(float));

    // Data initialization
    CALI_MARK_BEGIN("data_init");
    populate_array(local_data, segment_size, segment_size * process_rank, fill_option);
    CALI_MARK_END("data_init");

    // Sorting computation
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    odd_even_sort(local_data, segment_size, process_rank, num_processes, MPI_COMM_WORLD);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    float *collected_data;
    MPI_Status status;

    // Data communication
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    if (process_rank == 0) {
        collected_data = (float*)malloc(total_elements * sizeof(float));
        distribute_data(collected_data, local_data, segment_size, 0);

        float *temp_data = (float*)malloc(segment_size * sizeof(float));
        for (int i = 1; i < num_processes; ++i) {
            CALI_MARK_BEGIN("MPI_Recv");
            MPI_Recv(temp_data, segment_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
            CALI_MARK_END("MPI_Recv");
            distribute_data(collected_data, temp_data, segment_size, i);
        }
        free(temp_data);
    } else {
        CALI_MARK_BEGIN("MPI_Send");
        MPI_Send(local_data, segment_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        CALI_MARK_END("MPI_Send");
    }
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // Verification and logging
    if (process_rank == 0) {
        CALI_MARK_BEGIN("correctness_check");
        int is_sorted_correctly = verify_sorted(collected_data, total_elements);
        CALI_MARK_END("correctness_check");

        adiak::init(NULL);
        adiak::launchdate();    // launch date of the job
        adiak::libraries();     // Libraries used
        adiak::cmdline();       // Command line used to launch the job
        adiak::clustername();   // Name of the cluster
        adiak::value("Algorithm", "odd_even_sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
        adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
        adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
        adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
        adiak::value("InputType", options[OPTION-1]); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
        adiak::value("num_procs", numtasks); // The number of processors (MPI ranks)
        adiak::value("group_num", 25); // The number of your group (integer, e.g., 1, 10)
        adiak::value("implementation_source", "Online and Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
        adiak::value("correctness", correctness); // Whether the dataset has been sorted (0, 1)
    }

    // Finalize
    cali_manager.stop();
    cali_manager.flush();
    MPI_Finalize();
}
