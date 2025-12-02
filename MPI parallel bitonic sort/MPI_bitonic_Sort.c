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

int main( int argc , char** argv){

    // int arr[] = {3,7,4,8,6,2,1,5};
    // int n = sizeof(arr)/sizeof(arr[0]);

    MPI_Init(&argc, &argv);

    int rank, numproc;
    MPI_Comm_rank(MPI_COMM_WORLD , &rank);
    MPI_Comm_size(MPI_COMM_WORLD , &numproc);

    // int n =8388608;
     int n =16;

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
        if(arr == NULL){ 
            printf("memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        srand(time(NULL));
        make_arry(arr , n);
    }

    

    int *local_buffer =malloc(chank * sizeof(int));
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

    //Right Shift (>> 1) → Divide by 2
    //Left Shift (<< 1) → Multiply by 2

    //scahter func 
    MPI_Scatter(arr,
                chank,
                MPI_INT,
                local_buffer,
                chank,
                MPI_INT,
                0,
                MPI_COMM_WORLD);
    
   bitonic_sort( local_buffer , 0 ,chank , 1);

   //place to keep the sorted parts in local arrry
   int *recv_buffer =malloc( chank * sizeof(int));

   //slider to go 2,4,8,16
   for(int size = 2; size <= numproc ;size <<=1){
    //add the diraction to local_bufferd arrys
    int groupDir = ((rank & size ) == 0);

    for (int step = size >>1 ;step >0 ;step >>= 1){
        int partner = rank ^ step ;

        MPI_Sendrecv( local_buffer ,
                      chank,
                      MPI_INT,
                      partner,
                      0,
                      recv_buffer,
                      chank,
                      MPI_INT,
                      partner,
                      0,
                      MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE
                    );

         // now in each local_arry <=> resive_arry compair them 
            if((rank < partner && groupDir == 1) || (rank > partner && groupDir == 0)){
                // Keep SMALLER values
                for(int i = 0; i < chank; i++)
                    if(local_buffer[i] > recv_buffer[i]) 
                        local_buffer[i] = recv_buffer[i];
            } else {
                // Keep LARGER values
                for(int i = 0; i < chank; i++)
                    if(local_buffer[i] < recv_buffer[i]) 
                        local_buffer[i] = recv_buffer[i];
            }
        }

   }

   MPI_Gather(local_buffer,
              chank,
              MPI_INT,
              arr,
              chank,
              MPI_INT,
              0,
              MPI_COMM_WORLD
            );


   if(rank == 0 ){
    printf("sorted arry :\n");
        for(int i=0 ; i<17 ;i++)
            printf("%d " ,arr[i]);
    printf("\n");
        
   }

   free(local_buffer);
   free(recv_buffer);

   if(rank == 0) free(arr); // in the begiing we give mmeory to the arr in rank 0
   //now no need for the memeory so free the rank 0 so NO MEMEORY LEEK 

   MPI_Finalize();
   return 0;

}


// gcc Serial_Botonic_Sort.c -o Serial_Botonic_Sort && ./Serial_Botonic_Sort


//2^26 = 67108864 when go this out put the fost are 4000 or so are 0 so dont 
//need to warry the code is not broken  not stack overflow 


//  //  // ((rank & size) == 0) this works like below
// rank = 2  (binary: 010)
// size = 2  (binary: 010)
//          &  (bitwise AND)
//          ─────────────
// Result:      010  (= 2 in decimal)

// (2 == 0)  // Is 2 equal to 0?
// → False   // NO! 
// → 0       // In C, false = 0

// Rank 0:
//   rank & size = 000 & 010 = 000 (= 0)
//   (0 == 0) → True → 1
//   groupDir = 1 (ascending ↑)

// Rank 1:
//   rank & size = 001 & 010 = 000 (= 0)
//   (0 == 0) → True → 1
//   groupDir = 1 (ascending ↑)

// thos code slider for (int step = size >> 1; step > 0; step >>= 1) {
// step = size >> 1 = 8 >> 1 = 4

// Iteration 1: step = 4  ✓ (4 > 0, continue)
// Iteration 2: step = 4 >> 1 = 2  ✓ (2 > 0, continue)
// Iteration 3: step = 2 >> 1 = 1  ✓ (1 > 0, continue)
// Iteration 4: step = 1 >> 1 = 0  ✗ (0 is not > 0, STOP)

// Result: step goes through: 4, 2, 1


//HOW TO RUN
// > mpicc MPI_bitonic_Sort.c -o MPI_bitonic_Sort

// # Run with 2 processes
//mpirun -np 2 ./MPI_bitonic_Sort

// # Run with 4 processes
// mpirun -np 4 ./MPI_bitonic_Sort

// # Run with 8 processes
// mpirun -np 8 ./MPI_bitonic_Sort

// # Run with 16 processes (if you have enough cores)
// mpirun -np 16 ./MPI_bitonic_Sort