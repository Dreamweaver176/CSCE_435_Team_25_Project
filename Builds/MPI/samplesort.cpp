#include "mpi.h"
#include <algorithm>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <mpi.h>
#include <time.h>
#include <adiak.hpp>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

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

void randomFill(double arr[], int n) {
    // for(int i = 0; i < n; i++) {                   // Sorted
    //     arr[i] = static_cast<double>(i);
    // }

    srand(time(NULL));                             // Random
    for(int i = 0; i < n; i++) {
        arr[i] = static_cast<double>(rand() % n);
        // if (n-i < 50) {
        //     printf("arr[i] = %f\n", arr[i]);
        // }
    }

    // for(int i = 0; i < n; i++) {                   // Reverse sorted
    //     arr[i] = static_cast<double>(n - i);
    // }

    // srand(time(NULL));                             // 1%perturbed
    // for(int i = 0; i < n; i++) {
    //     arr[i] = static_cast<double>(i);
    //     if (rand() % 100 == 1) {
    //         arr[i] *= static_cast<double>(rand() % 10 + 0.5);
    //     }
    // }
};

void smallSort(double* arr, int left, int right) {
    if (left < right) {
        // Partition the array
        double pivot = arr[right];
        int i = left - 1;
        for (int j = left; j <= right - 1; j++) {
            if (arr[j] < pivot) {
                i++;
                double temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        double temp = arr[i + 1];
        arr[i + 1] = arr[right];
        arr[right] = temp;
        int pi = i + 1;
        // Recursively sort elements before and after partition
        smallSort(arr, left, pi - 1);
        smallSort(arr, pi + 1, right);
    }
};

void sampleSort(double a[], int n, int k, int p, int taskid, MPI_Comm com) {
    CALI_CXX_MARK_FUNCTION;
    const int threshold = 10; // Define the threshold for smallSort
printf("1\n");
    int mtype = FROM_WORKER;

    // Step 1
    if (n / k < threshold) {
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        smallSort(a, 0, n - 1); // Use smallSort if average bucket size is below threshold
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);
        return;
    }
printf("2\n");
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);

    // Select samples and sort them
    double S[p * k];
    for (int i = 0; i < p * k; i++) {
        S[i] = a[rand() % n]; // Randomly select samples from A
    }
    smallSort(S, 0, p * k - 1); // Sort the samples
printf("3\n");
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);

    // Define splitters
    double splitters[p + 1];
    splitters[0] = INT_MIN;
    for (int i = 1; i < p; i++) {
        splitters[i] = S[i * (k - 1)];
    }
    splitters[p] = INT_MAX;
printf("4\n");
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

    // Step 2
    std::vector<std::vector<double>> buckets(p);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            if (a[i] > splitters[j] && a[i] <= splitters[j + 1]) {
                buckets[j].push_back(a[i]);
                break;
            }
        }
    }
printf("5\n");
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    // Step 3 and concatenation
    int index = 0; // Start index in the original array

    // Use MPI to send data to worker tasks and receive sorted data
    for (int i = 1; i < p; i++) {
        // Send data to worker tasks
        MPI_Send(&buckets[i][0], buckets[i].size(), MPI_DOUBLE, i, mtype, com);

        // Receive sorted data from worker tasks
        MPI_Recv(&a[index], buckets[i].size(), MPI_DOUBLE, i, mtype, com, MPI_STATUS_IGNORE);

        index += buckets[i].size();
    }
};

int main (int argc, char *argv[])
{
CALI_CXX_MARK_FUNCTION;
int sizeOfArray = atoi(argv[1]);

int	numtasks,              /* number of tasks in partition */
	taskid,                /* a task identifier */
	numworkers,            /* number of worker tasks */
	source,                /* task id of message source */
	dest,                  /* task id of message destination */
	mtype,                 /* message type */
	rows,                  /* rows of array A sent to each worker */
	averow, extra, offset, /* used to determine rows sent to each worker */
	i, j, k, rc;           /* misc */

double a[sizeOfArray];     /* array 1 */

MPI_Status status;
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
if (numtasks < 2 ) {
    printf("Need at least two MPI tasks. Quitting...\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
}
numworkers = numtasks-1;
MPI_Comm new_comm;
// MPI_Comm_split(MPI_COMM_WORLD, (taskid != MASTER), taskid, &new_comm);                             TRY IF THINGS GO WRONG
MPI_Comm_split(MPI_COMM_WORLD, (taskid == MASTER) ? MPI_UNDEFINED : 0, 0, &new_comm);

// WHOLE PROGRAM COMPUTATION PART STARTS HERE
CALI_MARK_BEGIN(whole_computation);

// Create caliper ConfigManager object
cali::ConfigManager mgr;
mgr.start();
printf("-1\n");
/**************************** master task ************************************/
if (taskid == MASTER)
{
    printf("sample sort has started with %d tasks.\n",numtasks);
    printf("Initializing arrays...\n");

    CALI_MARK_BEGIN(data_init);

    randomFill(a, sizeOfArray);

    CALI_MARK_END(data_init);

    averow = sizeOfArray/numworkers;
    extra = sizeOfArray%numworkers;
    offset = 0;
    mtype = FROM_MASTER;

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);

    for (dest=1; dest<=numworkers; dest++)
    {
        rows = (dest <= extra) ? averow+1 : averow;   	
        printf("Sending %d rows to task %d offset=%d\n",rows,dest,offset);
        MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
        MPI_Send(&a[offset], rows, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
        //MPI_Send(&b, sizeOfArray*sizeOfArray, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
        offset = offset + rows;
    }

    MPI_Recv(&a, sizeOfArray, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, &status);

    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
}


/**************************** worker task ************************************/
if (taskid > MASTER)
{
    mtype = FROM_MASTER;

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);

    MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&a/*[offset]                       ADD THIS BACK?*/, rows, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
    //MPI_Recv(&b, sizeOfArray*sizeOfArray, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // CALI_MARK_BEGIN(comp);
    // CALI_MARK_BEGIN(comp_small);

    // smallSort(a, 0, sizeof(a)/sizeof(a[0]) - 1); //                                                replace with numtasks-2?

    // CALI_MARK_END(comp_small);
    // CALI_MARK_END(comp);
printf("0\n");
    // CALI_MARK_BEGIN(comp);
    // CALI_MARK_BEGIN(comp_large);
// for (int i = 0; i < sizeOfArray; i++) {
//     if (sizeOfArray-i < 50) {
//             printf("a[i] = %f\n", a[i]);
//         }
// }
    sampleSort(a, sizeOfArray, sizeOfArray/(2*numtasks), numtasks, taskid, new_comm);

    // CALI_MARK_END(comp_large);
    // CALI_MARK_END(comp);
}

CALI_MARK_BEGIN(correctness_check);

bool sorted = true;
for (int i = 1; i < sizeOfArray; i++) {
    if (sizeOfArray-i < 50) {
        printf("a[i]: %f\n", a[i]);
    }
    if (a[i] < a[i-1]) {
        // printf("Error. Out of order sequence: %f found\n", a[i]);
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
adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
adiak::value("Datatype", "Double"); // The datatype of input elements (e.g., double, int, float)
adiak::value("SizeOfDatatype", sizeof(double)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
adiak::value("InputSize", sizeOfArray); // The number of elements in input dataset (1000)
adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
adiak::value("num_procs", numtasks); // The number of processors (MPI ranks)
//adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
//adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
adiak::value("group_num", 25); // The number of your group (integer, e.g., 1, 10)
adiak::value("implementation_source", "AI/Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

MPI_Comm_free(&new_comm);

// Flush Caliper output before finalizing MPI
mgr.stop();
mgr.flush();

MPI_Finalize();
};
