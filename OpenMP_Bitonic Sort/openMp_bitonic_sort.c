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

void bitonic_sort( int a[] , int low , int count ,int dir){
    if(count >1){
        int k = count /2;
        bitonic_sort(a,low   ,k ,1);
        bitonic_sort(a,low+k ,k ,0);

        bitonic_marge(a , low, count , dir);
    }
}

void bitonic_marge( int a[] ,int low ,int count ,int dir ){
    if(count>1){
        int k =count/2;

        for(int i =low ; i<low+k ;i++){
            if(dir == 1 && a[i] > a[i+k] ||
               dir == 0 && a[i] < a[i+k]   ){
                swap(a,i,i+k);
               }
        }
        bitonic_marge(a , low   , k , dir);
        bitonic_marge(a , low+k , k , dir);
    }
}



int main (){

 int n = 65536;
 int *arr = (int* )malloc(n*sizeof(int));
 
 if(arr == NULL)[
    printf("memeory alocation failed ");
    return 1;
 ]
 
 void make_arry(arr , n);

 void bitonic_sort(arr , 0 , n ,1 );



    return 0;
}