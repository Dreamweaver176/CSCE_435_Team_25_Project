#include<stdio.h>
#include<cuda.h>
#include <iostream>
#include <cstdlib>
#include <ctime>    

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

//CALI Regions
const char* whole_computation = "whole_computation"; 
const char* data_init = "data_init"; 
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comm_small = "comm_small";
const char* check_correctness = "check_correctness";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comp_small = "comp_small";

int threads;

__global__ void oddeven(int* x, int I, int n)
{
    //Even Phase
	int id=blockIdx.x;
	if(I==0 && ((id*2+1)< n)){
		if(x[id*2]>x[id*2+1]){
			int X=x[id*2];
			x[id*2]=x[id*2+1];
			x[id*2+1]=X;
		}
	}

    //Odd Phase
	if(I==1 && ((id*2+2)< n)){
		if(x[id*2+1]>x[id*2+2]){
			int X=x[id*2+1];
			x[id*2+1]=x[id*2+2];
			x[id*2+2]=X;
		}
	}
}


bool correctness_check(int A[], int n){
    for(int i = 0; i < n - 1; i++){
        if (A[i] > A[i+1]){
            return false;
        }
    }
    return true;
}


int main(int argc, char *argv[])
{

    int n = atoi(argv[1]);
    threads = atoi(argv[2]);
    int input = atoi(argv[3]);
    int c[n];
    int a[n]; 

    const int limit_upper = 250;

    CALI_MARK_BEGIN("whole_computation");
    CALI_MARK_BEGIN("data_init");


    if(input == 0){
        for(int i = 0; i<n; i++){
            a[i] = i;
        }

    }
    else if(input == 1){
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            a[i] = rand()%limit_upper;
        }
    }
    else if(input == 2){
        for(int i = 0; i<n; i++){
            a[i] = n - i;
        }
    }
    else{
        std::srand(static_cast<unsigned>(std::time(0)));
        for (int i = 0; i < n; ++i) {
            double randomValue = static_cast<double>(rand()) / RAND_MAX;
            if (randomValue < 0.01) {
                a[i] = static_cast<int>(rand() % 10);
            } 
            else {
                a[i] = i + 1;
            }
        }
    }

    CALI_MARK_END("data_init");

    int *d; 
	cudaMalloc((void**)&d, n*sizeof(int));

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
	cudaMemcpy(d,a,n*sizeof(int),cudaMemcpyHostToDevice);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
	for(int i=0;i<n;i++){
		oddeven<<<n/2, threads>>>(d,i%2,n);
	}
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");


    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
	cudaMemcpy(c,d,n*sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");


    bool correct_check= correctness_check(c,n);

    CALI_MARK_END("whole_computation");

    printf("Sorted Array is:\t");
	for(int i=0; i<n; i++)
	{
		printf("%d\t",c[i]);
	}
    printf("\n");

    if( correct_check== true){
        printf("The array is sorted from least to greatest \n");
    }
    else{
        printf("The array is not sorted \n");
    }

	cudaFree(d);


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
        temp_string = "1%%pertubed";
    }

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "Odd Even Transposition Sort");
    adiak::value("ProgrammingModel", "CUDA");
    adiak::value("Datatype", "int"); 
    adiak::value("SizeOfDatatype", sizeof(int)); 
    adiak::value("InputSize", n); 
    adiak::value("InputType", temp_string);
    adiak::value("num_threads", threads);
    adiak::value("num_blocks", n/2);
    adiak::value("group_num", 9);
    adiak::value("implementation_source", "Online");

	return 0;
}