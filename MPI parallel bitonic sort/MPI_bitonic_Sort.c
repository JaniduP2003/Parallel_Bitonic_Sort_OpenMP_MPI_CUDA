#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>


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

    MPI_Init(&argc, &argv);

    int rank, numproc
    MPI_comm_rank(MPI_COMM_WORLD , &rank);
    MPI_Comm_size(MPI_COMM_WORLD , &numproc);

    int n =8388608;

    int chank = n /numproc;

    int *arr =NULL; //good practice in C to make this NULL



    //Allocate memory | int *arr =(int*)malloc(n*sizeof(int));
    // Seed the random number generator |  srand(time(NULL));
    // then  Generate the random arry | make_arry(arr, n);
    
    // Only rank 0 does this, because: In MPI, you don't want every rank 
    //if(rank == 0){  add evrythinh here }
    //creating its own array—only one master array is needed.

    if(rank == 0){
        arr = malloc(n*sizeof(int));
        srand(time(NULL));
        make_arry(arr , n);
    }

    if(arr ==NULL){
        printf("memeory alocation failed ");
        return 1;
    }

    int local_buffer =malloc(chank * sizeof(int));
    //why: chanck becose it is broken to the n/numberof proc thats the size of one part

    //now need to sactter the main big arry
    //the scatterd parts can be sored in locak_buffer arry i created 
    //each broken part must be sorted (per chank only sort ) | bitonic sort();
    //now each part is in asccending oder 
        //# problem : thes wrong need to be in bitionc seq to be sorted so need a way to 
        // organize this each accending parts up up up up to
        // up down up down up 
        //how :::???? 
        // to do this sue XOR | (rank & size) == 0 if its 00 then up if 10 then down 
    //now you have groupes
    //now pari them and compair 
    //need a way to pair teh groupes to work to gether???
    //how ::: ????
    //when you exange the arry DONT OVERRIDE THE LOCAL_BUFFER add the resived one to
    // RESIVE_BUFFER
    // now in each local_arry <=> resive_arry compair them 
    // add the reuslt to the local buffer 

    srand(time(NULL));
    make_arry( arr , n);

    printf("Before sorting (first 20 elements):\n");
    for (int i =0 ; i <400 ;i++){
        printf("%d " ,arr[i]);
    }
    printf("\n\n");

    clock_t start =clock();
    bitonic_sort(arr ,0 ,n,1);
    clock_t end =clock();

    double time_taken = ((double)end-(double)start )/CLOCKS_PER_SEC;
    printf("time it took to mkae the caclulations : %.6f sec\n",time_taken);
    
     printf("After sorting (first 20 elements):\n");
    for (int i =0 ; i <100000 ;i++){
        printf("%d " ,arr[i]);
    }
    printf("\n");

    free(arr);
    return 0;
}


// gcc Serial_Botonic_Sort.c -o Serial_Botonic_Sort && ./Serial_Botonic_Sort


//2^26 = 67108864 when go this out put the fost are 4000 or so are 0 so dont 
//need to warry the code is not broken  not stack overflow 