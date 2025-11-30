#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>


void swap(int a[], int i,int j){
    int temp = a[i];
    a[i] =a[j];
    a[j] =temp;
}

void make_arry( int a[] ,int n){
    for(int i =0 ; i<n ;i++){
        a[i] = rand()%1000;
    }
}

void bitonic_marge( int a[] ,int low ,int count ,int dir ){
    if(count>1){
        int k =count/2;

        #pragma omp parallel for 
        for(int i =low ; i<low+k ;i++){
            if((dir == 1 && a[i] > a[i+k]) ||
               (dir == 0 && a[i] < a[i+k])){
                swap(a,i,i+k);
            }
        }
        bitonic_marge(a , low   , k , dir);
        bitonic_marge(a , low+k , k , dir);
    }
}

void bitonic_sort( int a[] , int low , int count ,int dir){
    if(count >1){
        int k = count /2;

        #pragma omp task shared(a)
        bitonic_sort(a,low   ,k ,1);

        #pragma omp task shared(a)
        bitonic_sort(a,low+k ,k ,0);
        
        #pragma omp taskwait 
        bitonic_marge(a , low, count , dir);    
    }
}

//Why is shared(a) there? becose 
//all the sorting ahappens in the SAME ARRRY
//#pragma omp taskwait — “Wait until both tasks finish”


int main (){

 int n = 8388608;
 int *arr = (int* )malloc(n*sizeof(int));
 
 if(arr == NULL){
    printf("memeory alocation failed ");
    return 1;
}

 srand(time(NULL));
 make_arry(arr , n);

 printf("befor arry \n");
 for(int i =0 ;i<200 ;i++){
    printf("%d ",arr[i]);
 }
 printf("\n");

 clock_t start =clock();
 bitonic_sort(arr , 0 , n ,1 );
 clock_t end =clock();

double clock_dif = ((double)(end - start)) / CLOCKS_PER_SEC;
 printf(" rime taken for comutation of sorted arry : %.6f sec" ,clock_dif);

printf("after sorting ");
for(int i =0 ;i <200 ;i++){
    printf("%d ",arr[i]);
}
printf("\n");

    free(arr);
    return 0;
}

//gcc -fopenmp openMp_bitonic_sort.c -o openMp_bitonic_sort && ./openMp_bitonic_sort