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
    int *arr = (int*)malloc(n * sizeof(int));

    if(arr == NULL ){
        printf("mamoery alocation failed !! ");
        return -1;
    }



    free(arr);
    return 0 ;
}