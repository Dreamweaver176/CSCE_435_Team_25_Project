#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <string_view>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "mpi.h"

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

// Cali Regions
const char* main = "main";
const char* data_init = "data_init";
const char* comp = "comp";
const char* comm = "comm";
const char* comp_small = "comp_small";
const char* comm_small = "comm_small";
const char* comp_large = "comp_large";
const char* comm_large = "comm_large";
const char* correctness_check = "correctness_check";

int main(int argc, char *argv[]) {
    int arraySize = atoi(argv[1]);

    int numTasks,
        taskid,                /* a task identifier */
        numworkers,            /* number of worker tasks */
        source,                /* task id of message source */
        dest,                  /* task id of message destination */
        mtype,                 /* message type */
        rows,                  /* rows of array A sent to each worker */
        averow, extra, offset, /* used to determine rows sent to each worker */
        splitv;                /* used to determine splits */
    MPI_Status status;
    int a[arraySize];
    int b[arraySize];

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&numTasks);
    
    numworkers = numTasks - 1;
    averow = arraySize/numworkers;

    CALI_MARK_BEGIN(main);
    
    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();
  
    if (taskid == MASTER) {
        printf("Sample sort has started with %d tasks.\n", numworkers);
        printf("Initializing array...\n");
        
        // initialize master process and generate array values
        CALI_MARK_BEGIN(data_init);
        
        // for(int i = 0; i < arraySize; i++) {                   // Sorted
        //     a[i] = static_cast<int>(i);
        // }

        srand(time(NULL));                             // Random
        for(int i = 0; i < arraySize; i++) {
            a[i] = static_cast<int>(rand() % arraySize);
        }

        // for(int i = 0; i < arraySize; i++) {                   // Reverse sorted
        //     a[i] = static_cast<int>(arraySize - i);
        // }

        // srand(time(NULL));                             // 1%perturbed
        // for(int i = 0; i < arraySize; i++) {
        //     a[i] = static_cast<int>(i);
        //     if (rand() % 100 == 1) {
        //         a[i] *= static_cast<int>(rand() % 10 + 0.5);
        //     }
        // }
        
        CALI_MARK_END(data_init);

        offset = 0;
        extra = arraySize%numworkers;
        mtype = FROM_MASTER;

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);

        for (dest=1; dest<=numworkers; dest++)
        {
            rows = (dest <= extra) ? averow+1 : averow;  
            printf("Sending %d values to task %d offset=%d\n", rows, dest, offset);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&a[offset], rows, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            offset = offset + rows;
        }
        
        // receive chosen samples from workers
        mtype = FROM_WORKER;
        std::vector<int> samp((numworkers-1)*numworkers);
        for (source=1; source<=numworkers; source++)
        {
            MPI_Recv(&samp[(numworkers-1)*(source-1)], numworkers-1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
        }

        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);

        std::sort(samp.begin(), samp.end());
        
        // define splitters
        int allSplitters[numworkers-1];
        int space = std::ceil((float)samp.size()/(float)numworkers);
        int index = space-1;

        for (int i=0; i<numworkers-1; i++) {
            allSplitters[i] = samp[index];
            index += space;
        }

        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        mtype = FROM_MASTER;
        for (dest=1; dest<=numworkers; dest++)
        {
            splitv = allSplitters[dest-1];
            if (dest==numworkers) {
                splitv = INT_MAX;
            }

            CALI_MARK_BEGIN(comm);
            CALI_MARK_BEGIN(comm_small);

            MPI_Send(&allSplitters, numworkers-1, MPI_INT, dest, mtype, MPI_COMM_WORLD);

            CALI_MARK_END(comm_small);

            CALI_MARK_BEGIN(comm_large);

            MPI_Send(&a, arraySize, MPI_INT, dest, mtype, MPI_COMM_WORLD);

            CALI_MARK_END(comm_large);
            CALI_MARK_END(comm);
        }
        
        // receive results from workers
        std::vector<std::vector<int>> vecBuck(numworkers); 
        mtype = FROM_WORKER;

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);
        
        for (source=1; source<=numworkers; source++)
        {
            int buckets[numworkers][averow+1];
            MPI_Recv(&buckets, numworkers*(averow+1), MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            for (int i=0; i<numworkers; i++) {
            for (int j=0; j<averow+1; j++) {
                if (buckets[i][j] != -1)
                vecBuck[i].push_back(buckets[i][j]);
            }
            }
        }

        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);

        for (int i=0; i<numworkers; i++) {
            std::sort(vecBuck[i].begin(), vecBuck[i].end());
        }

        int last = 0;
        for (int i=0; i<numworkers; i++) {
            for (int j=0; j<vecBuck[i].size(); j++) {
                b[last] = vecBuck[i][j];
                last++;
            }
        }

        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(correctness_check);
        
        bool sorted = true;
        for (int i = 1; i < arraySize; i++) {
            if (b[i] < b[i-1]) {
                printf("Error. Out of order sequence: %d found\n", b[i]);
                sorted = false;
            }
        }
        if (sorted) {
            printf("Array is in sorted order\n");
        }

        CALI_MARK_END(correctness_check);
    }
    
    if (taskid != MASTER) {
        int newSamples[numworkers-1];
        int rows;

        mtype = FROM_MASTER;

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);

        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        std::vector<int> arr(rows);
        MPI_Recv(&arr[0], rows, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);

        std::sort(arr.begin(), arr.end());

        CALI_MARK_END(comp_large);

        int space = std::ceil((float)rows/(float)numworkers);
        int index = space-1;

        CALI_MARK_BEGIN(comp_small);

        for (int i=0; i<numworkers-1; i++) {
            if (index > arr.size()-1)
                break;
            newSamples[i] = arr[index];
            index += space;
        }

        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        mtype = FROM_WORKER;

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);

        MPI_Send(&newSamples, numworkers-1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);

        mtype = FROM_MASTER;
        int splitters[numworkers-1];

        MPI_Recv(&splitters, numworkers-1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&a, arraySize, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);

        int buckets[numworkers][averow+1];
        int newIndex[numworkers];

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);

        for (int i=0; i<numworkers; i++) {
            newIndex[i]=0;
            for (int j=0; j<averow+1; j++) {
                buckets[i][j]= -1;
            }
        }
        
        int j;
        for(int num : arr) {
            j = 0;
            while(j < numworkers) {
                if (j == numworkers-1) {
                    buckets[j][newIndex[j]] = num;
                    newIndex[j]++;
                    break;
                }
                if(num < splitters[j]) {
                    buckets[j][newIndex[j]] = num;
                    newIndex[j]++;
                    break;
                }
                j++;
            }
        }

        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);

        mtype = FROM_WORKER;

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);

        MPI_Send(&buckets, numworkers*(averow+1), MPI_INT, MASTER, mtype, MPI_COMM_WORLD);

        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);
    }

    CALI_MARK_END(main);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "SampleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "Integer"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", arraySize); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", numTasks); // The number of processors (MPI ranks)
    //adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
    //adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
    adiak::value("group_num", 25); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI/Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();
    
    MPI_Finalize();
}
