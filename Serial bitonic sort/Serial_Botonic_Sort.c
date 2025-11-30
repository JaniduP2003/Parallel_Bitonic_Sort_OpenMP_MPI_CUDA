#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void swap( int arr[] ,int i ,int j){
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

void make_arry(int a[] , int n){
    for(int i =0 ; i<n ;i++){
        a[i] = rand()%1000;
    }
}

void bitonic_marge(int a[] , int low ,int count ,int dir ){

    if(count >1 ){
        int k =count /2;

        for (int i =low ; i <low+k ;i++){
            if(( dir == 1  && a[i] > a[i+k] ) ||
               ( dir == 0  && a[i] < a[i+k] )   ){
                swap(a,i,i+k);
               }
        }

        bitonic_marge(a ,low    ,k ,dir );
        bitonic_marge(a ,low +k ,k ,dir  );
    }

    
}

void bitonic_sort(int a[] , int low ,int count  ,int dir){

    if(count >1){
        int k =count/2;
        bitonic_sort(a , low ,k ,1);
        bitonic_sort(a ,low+k ,k ,0);

        bitonic_marge(a ,low ,count ,dir);
    }
    
}

int main(){

    // int arr[] = {3,7,4,8,6,2,1,5};
    // int n = sizeof(arr)/sizeof(arr[0]);

    int n = 65536;
    int *arr =(int*)malloc(n*sizeof(int));

    if(arr ==NULL){
        printf("memeory alocation failed ");
        return 1;
    }

    srand(time(NULL));
    make_arry( arr , n);

    printf("Before sorting (first 20 elements):\n");
    for (int i =0 ; i <200 ;i++){
        printf("%d " ,arr[i]);
    }
    printf("\n\n");

    clock_t start =clock();
    bitonic_sort(arr ,0 ,n,1);
    clock_t end =clock();

    double time_taken = ((double)end-(double)start )/CLOCKS_PER_SEC;
    printf("time it took to mkae the caclulations : %.6f sec\n",time_taken);
    
     printf("After sorting (first 20 elements):\n");
    for (int i =0 ; i <200 ;i++){
        printf("%d " ,arr[i]);
    }
    printf("\n");

    free(arr);
    return 0;
}


// gcc Serial_Botonic_Sort.c -o Serial_Botonic_Sort && ./Serial_Botonic_Sort