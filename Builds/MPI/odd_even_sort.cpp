#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char* whole_computation = "whole_computation"; 
const char* data_init = "data_init"; 
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comm_small = "comm_small";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comp_small = "comp_small";
const char* check_correctness = "check_correctness";

int merge(double *input_a, int length_a, double *input_b, int length_b, double *out) {
    int outcount = 0;
    int i,j;

    for (i=0,j=0; i<length_a; i++) {
        while ((input_b[j] < input_a[i]) && j < length_b) {
            out[outcount++] = input_b[j++];
        }
        out[outcount++] = input_a[i];
    }
    while (j<length_b)
        out[outcount++] = input_b[j++];

    return 0;
}

bool correctness_check(double A[], int n){
    for(int i = 0; i < n - 1; i++){
        if (A[i] > A[i+1]){
            return false;
        }
    }
    return true;
}

int domerge_sort(double *a, int begin, int finish, double *b) {
    if ((finish - begin) <= 1) return 0;

    int middle = (finish+begin)/2;
    domerge_sort(a, begin, middle, b);
    domerge_sort(a, middle,   finish, b);
    merge(&(a[begin]), middle-begin, &(a[middle]), finish-middle, &(b[begin]));
    for (int i=begin; i<finish; i++)
        a[i] = b[i];

    return 0;
}

int merge_sort(int n, double *a) {
    double b[n];
    domerge_sort(a, 0, n, b);
    return 0;
}

void printstat(int rank, int iter, char *txt, double *la, int n) {
    printf("[%d] %s iter %d: <", rank, txt, iter);
    for (int j=0; j<n-1; j++)
        printf("%6.3lf,",la[j]);
    printf("%6.3lf>\n", la[n-1]);
}

void Value_Exchange(int local_b, double *local_a, int rank_send, int rank_receive, MPI_Comm comm) {

    double remote[local_b];
    double all[2*local_b];
    int rank;
    const int sortedtag = 2;
    const int mergetag = 1;

    MPI_Comm_rank(comm, &rank);

    if (rank == rank_send) {
        CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("comm_large");
        MPI_Send(local_a, local_b, MPI_DOUBLE, rank_receive, mergetag, MPI_COMM_WORLD);
        MPI_Recv(local_a, local_b, MPI_DOUBLE, rank_receive, sortedtag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CALI_MARK_END("comm_large");
        CALI_MARK_END("comm");
    } 
    
    else {
        CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("comm_large");
        MPI_Recv(remote, local_b, MPI_DOUBLE, rank_send, mergetag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CALI_MARK_END("comm_large");
        CALI_MARK_END("comm");
        

        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");
        merge(local_a, local_b, remote, local_b, all);
        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");

        int theirstart = 0, mystart = local_b;
        
        if (rank_send > rank) {
            theirstart = local_b;
            mystart = 0;
        }

        CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("comm_large");
        MPI_Send(&(all[theirstart]), local_b, MPI_DOUBLE, rank_send, sortedtag, MPI_COMM_WORLD);
        CALI_MARK_END("comm_large");
        CALI_MARK_END("comm");
        
        for (int i=mystart; i<mystart+local_b; i++){
            local_a[i-mystart] = all[i];
        }
    
    }
}

int MPI_Odd_Even_Transposition(int n, double *a, int root, MPI_Comm comm){
    int rank, size, i;
    double *local_a;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    local_a = (double *) calloc(n / size, sizeof(double));

    MPI_Scatter(a, n / size, MPI_DOUBLE, local_a, n / size, MPI_DOUBLE, root, comm);

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    merge_sort(n / size, local_a);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    for (i = 1; i <= size; i++) {

        if ((i + rank) % 2 == 0) {
            if (rank < size - 1) {
                Value_Exchange(n / size, local_a, rank, rank + 1, comm);
            }
        } 
        else if (rank > 0) {
            Value_Exchange(n / size, local_a, rank - 1, rank, comm);
        }



    }

    MPI_Gather(local_a, n / size, MPI_DOUBLE, a, n / size, MPI_DOUBLE, root, comm);

    if (rank == root){
        CALI_MARK_BEGIN("check_correctness");
        bool cc = correctness_check(a, n);
        CALI_MARK_END("check_correctness");
        if(cc == true){
            printf("Least to Greatest \n");

        }
        else{
            printf("Unsorted Array \n");
        }
    }

    return MPI_SUCCESS;
}

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int array_size = atoi(argv[1]);
    double a[array_size];
    int input = atoi(argv[2]);
    int num_procsesses = atoi(argv[3]);
    int upperlimit = 250;


    CALI_MARK_BEGIN("whole_computation");
    CALI_MARK_BEGIN("data_init");

    if(input == 0){
        for(int i = 0; i<array_size; i++){
            a[i] = i;
        }

    }
    else if(input == 1){
        srand(time(NULL));
        for (int i = 0; i < array_size; i++) {
            a[i] = rand()%upperlimit;
        }
    }
    else if(input == 2){
        for(int i = 0; i<array_size; i++){
            a[i] = array_size - i;
        }
    }
    else{
        std::srand(static_cast<unsigned>(std::time(0)));
        for (int i = 0; i < array_size; ++i) {
            double randomValue = static_cast<double>(rand()) / RAND_MAX;
            if (randomValue < 0.01) {
                a[i] = static_cast<double>(rand() % 10);
            } else {
                a[i] = i + 1;
            }
        }
    }
    CALI_MARK_END("data_init");

    MPI_Odd_Even_Transposition(array_size, a, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    CALI_MARK_END("whole_computation");
    
    CALI_MARK_BEGIN("comm_small");
    CALI_MARK_END("comm_small");
    CALI_MARK_BEGIN("comp_small");
    CALI_MARK_END("comp_small");


   std::string temp_string; 
   if(input == 0){
      temp_string = "Sorted";
   }
   else if(input == 1){
      temp_string = "Random";
   }
   else if(input == 2){
      temp_string = "ReverseSorted";
   }
   else{
      temp_string = "1%%perturbed";
   }


   adiak::init(NULL);
   adiak::launchdate();
   adiak::libraries();
   adiak::cmdline();
   adiak::clustername();
   adiak::value("Algorithm", "Odd Even Transposition Sort");
   adiak::value("ProgrammingModel", "MPI");
   adiak::value("Datatype", "double");
   adiak::value("SizeOfDatatype", sizeof(double));
   adiak::value("InputSize", array_size); 
   adiak::value("InputType", temp_string); 
   adiak::value("num_procs", num_procsesses); 
   adiak::value("group_num", 9); 
   adiak::value("implementation_source", "Online");

    return 0;
}
