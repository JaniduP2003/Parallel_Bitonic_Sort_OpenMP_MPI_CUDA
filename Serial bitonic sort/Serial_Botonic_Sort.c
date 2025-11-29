#include <studio.h>
#include <stdlib.h>


void swap( int arr[] ,int i ,int j){
    int temp = arr[j];
    arr[j] =arr[i];
    arr[j] =temp
}

void bitonic_sort(int a[] , int low ,int count  ,int dir){

    if(n>1){
        k =count/2;
        bitonic_sort(a[] , low ,k ,1);
        bitonic_sort(a[] ,low+k ,k ,0);

        bitonic_marge(a ,low ,count ,dir);
    }
    return
}

void bitoninc+murge (arr[] , ){

    return 
}



int main(){

    int arr[] = {3,7,4,8,6,2,1,5};
    int n = sizeof(arr)/sizeof(arr[0]);

    for (int i =0 ; i <n ;i++){
        printf("%d" ,arr[i]);
    }
    printf("/n");

    bitonic_sort(arr ,0 ,n,1);


    return 0;
}