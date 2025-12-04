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

    //need a way to add the global index uniqe
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //need the XOR gate to find out of accending or decensding
    int ixj = i^ j; //find the index of the morrored pert to compair 
    //i : 110 (6) | j : 010 (2) XOR : 100  -> 4 

    //need to stop relationship is symmetric: so make a gard raill
    if(ixj > i){
        //must be udner the n and the ixk must be under n to 
        //both must be true it must be xor must be in arry 
        if( i<n && ixj <n){
            //set derection up or down
            int accending = ((i & k) == 0 );
            if ((ascending && arr[i] > arr[ixj]) || (!ascending && arr[i] < arr[ixj])){
                //swap
                int temp = arr[i];
                arr[i] = arr[ixj];
                arr[ixj] = temp;
            }
        }
    }

    //the code do the marging (with swap)

}

void bitonic_sort_cuda(int *d_arr , int n ){

    int threads =256;
    int blocks = (n+ threads -1)/ threads;

    // k = 2,4,8,16,...,n need to go liek this 1st k is 2 then j must be 1 
    for(int k =2 ; k<=n ; k *=2){
        // // j = k/2, k/4, k/8, ..., 1 make like this
        for(int j = k/2 ; j> 0; j /=2){
            //need to magre the parires up up 
            bitonic_marge_kernel<<<block, threads >>>(d_arr , n, j, k);
            cudaDeciceSynchronize();
            }

        }

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