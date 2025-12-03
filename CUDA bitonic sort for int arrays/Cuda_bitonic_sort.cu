//make the arry in cpu then copy to gpu then sort them after copy
// back to cpu then print and cleen up

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

void make_arry( int a[] , int num ){
    for( int i =0 ; i<num ;i++){
        a[i] = srand()%1000;
    }
}

__global__ void bitonic_marge_kernel(int arr , int n , int  j ,int k){

}

void bitonic_sort_cuda(int *d_arr , int n ){

}



int main(){

    int n = 16;
    int *arr = (int*)malloc(n * sizeof(int));   //arr is made (this will be
                                             // filed with random values)

    if(arr == NULL ){
        printf("mamoery alocation failed !! ");
        return -1;
    }

    srand(time(NULL));
    make_arry(arr , n);

    printf("before sorting :\n");
    for(int i = 0 ; i<100 ;i++){
        printf("%d ",arr[i]);
    }
    printf("\n");

    //now need to store it in gpu for that needs a arry just for gpu
    //to do this use pointers

    int *d_arr;
    cudaMalloc( &d_arr ,n * sizeof(int)); // for ther is  a arry in gpu
    //now ther is two arrys for cpu and the gpu 
    //now we copy the cpu arry to the gpu for sorting 

    cudaMemcpy( d_arr , arr , n * sizeof(int), cudaMemcpyHostToDevice);
    // now it has been copyed 


    //add the time caclulations
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEvrnyCreate(&end);

    cudaEventRecord(start);

    //add the code you nee to time
    //sorting code 

    cudaEventRecord(end);
    float milsec=0;
    cudaEventElapsedTime(&milsec, start ,end );

    //take back the soreted arry form the gpu
    cudaMemcpy( arr , d_arr , n * sizeof(int) , cudaMemcpyDeviceToHost);

    printf("time taken : %.6f sec\n" ,milsec/1000.0);

    //print the reuslt
    printf("sorted arry:\n");
    for(int i = 0 ;i<100 ;i++){
        printf("%d " ,arr[i]);
    }
    printf("\n");

    //now needs to cleen up everything
    cudaFree(d_arr);
    //need to remove the events in gpu to 
    cudaEventDestory(start);
    cudaEventDestory(end);

    free(arr);
    return 0 ;
}


// //    clock_t start = clock();
//     kernel<<<blocks, threads>>>();
//     clock_t end = clock();  ← Measures kernel LAUNCH time only! 
//     Problem: Kernel launches are ASYNCHRONOUS
//     CPU continues immediately, doesn't wait for GPU to finish

//so use a cuda event
// cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
    
//     cudaEventRecord(start);      ← Insert event in GPU stream
//     kernel<<<blocks, threads>>>();
//     cudaEventRecord(stop);       ← Insert event in GPU stream
    
//     cudaEventSynchronize(stop);  ← Wait for GPU to finish
    
//     cudaEventElapsedTime(&ms, start, stop);  ← Accurate GPU time! 