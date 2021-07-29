
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

__global__ void get_negative_columns(float  * d_matrix, int *d_result, int size);
void fillUpMatrix(float *matrix, int size);
__device__ int d_next_idx = 0;
__device__ int *d_result_cnt = 0;
//__device__ int *d_result;


int main(int argc, char *argv[])
{
    long numElement = 0;
    if (argc < 2)
    {
        numElement = 1000000;
        printf("no arg given, %li elements will be generated.\n", numElement);
    }
    else
    {
        numElement = (long)strtod(argv[1], NULL);
    }

    const long size = (long)sqrt(numElement);
    printf("\nWe will be working with %li X %li matrix.\n", size, size);
    numElement = size * size;

    const int MATRIX_BYTE_SIZE = numElement * sizeof(float);
    const int RESULT_BYTE_SIZE = size * sizeof(int);
 
    float *h_matrix = (float *)malloc(MATRIX_BYTE_SIZE);
    fillUpMatrix(h_matrix, size);


    // declare, allocate, and zero out GPU memory
    float * d_matrix;
    cudaMalloc((void **)&d_matrix, MATRIX_BYTE_SIZE);
        
    // copy the matrix from Host to GPU
    cudaMemcpy(d_matrix, h_matrix, MATRIX_BYTE_SIZE, cudaMemcpyHostToDevice);
 
    //DEBUG
    float *h_matrix_1 = (float *)malloc(MATRIX_BYTE_SIZE);
   
    int *d_result;
    cudaMalloc((void **)&d_result, RESULT_BYTE_SIZE);
    //cudaMemset((void *)d_result, 11, RESULT_BYTE_SIZE);

    int *h_result = (int *)malloc(RESULT_BYTE_SIZE);
    // copy the result of process from GPU
    //cudaMemcpy(h_result, d_result, RESULT_BYTE_SIZE, cudaMemcpyDeviceToHost);

    
    //printf("\nTHIS IS BEFORE:\n");

    //for(int i=0; i<size; ++i){
    //  *(h_result + i) = 12;
    //    printf("%d, ",*(h_result + i));
    //}

    // launch the kernel
    const int NUM_THREAD = size;
    const int BLOCK_WIDTH = 1000;
    
    if(NUM_THREAD > 1000)
      get_negative_columns<<<NUM_THREAD / BLOCK_WIDTH, BLOCK_WIDTH>>>(d_matrix, d_result, size);
    else get_negative_columns<<<1, NUM_THREAD>>>(d_matrix, d_result, size);
    
    // force the printf()s to flush
    //cudaDeviceSynchronize();
 
    int h_result_cnt = 0;
    // copy the result of process from GPU
    cudaMemcpy(h_result, d_result, RESULT_BYTE_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_result_cnt, &d_result_cnt, sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_matrix_1, d_matrix, MATRIX_BYTE_SIZE, cudaMemcpyDeviceToHost);


/*
    // PRINT MATRIX
    for(int i=0; i<size; ++i){
      for(int j=0; j<size; ++j){
        printf("%f",*(h_matrix_1 + (i * size) + j));
        printf((j%5 ? ",\t": "\n"));
      }
    }

    printf("\nFROM HOST:\n");

    for(int i=0; i<size; ++i){
      for(int j=0; j<size; ++j){
        printf("%f",*(h_matrix + (i * size) + j));
        printf((j%5 ? ",\t": "\n"));
      }
    }
 
*/
 
    printf("\nTHIS IS THE RESULT: %d \n", h_result_cnt);

    for(int i=0; i<size; ++i){      
        printf("%d\n",*(h_result + i));
        //printf((i%50)? ", ": "\n");      
    }

    // free GPU memory allocation
    cudaFree(d_matrix);
    cudaFree(d_result);

  return 0;
}

//expects "matrix" to of shape (size X size)
void fillUpMatrix(float *matrix, int size)
{  
	srand((unsigned int)time(NULL));
	
  for(int i=0; i<size; ++i){
		for(int j=0; j<size; ++j){
			float rnd1 = (float)rand()/(float)(RAND_MAX/2.55);
      float rnd2 = (float)rand()/RAND_MAX;

      float randValue = (rnd2<0.1) ? 0.0 : rnd1-rnd2;
      *(matrix + (i * size) + j) = randValue;
		}
	}
}

__device__ int getGlobalIdx(){
  return blockIdx.x *blockDim.x + threadIdx.x;
}

__global__ void get_negative_columns(float *d_matrix, int *d_result, int size){
    
    int idx = getGlobalIdx();
    //d_result[idx] = 3030;
    //printf("Hello World! I'm a thread in thread %d\n", idx);


    ///*
    //int idx = vlockIdx.x * blockDim.x + threadIdx.x;
    //int idx = blockDim.x + threadIdx.x;
    
    int zeros = 0;
    int negs = 0;
    
    for (size_t i = 0; i < size; i++)
    {
        float value = *(d_matrix + (idx * size) +i);
        if (value == 0.0){ zeros++; }
        else if (value < 0.0){ negs++; }
    }

    if (zeros * 2 >= negs)
    {
        int my_idx = atomicAdd(&d_next_idx, 1);
        if ((my_idx + 1) < size)
        {
            d_result[my_idx++] = (idx==0) ? -2 : idx; //save the column idx; 0 == -2
//            atomicAdd(d_result_cnt, 1); //increment the count
        }
    /*}else{
        int my_idx = atomicAdd(&d_next_idx, 1);
        if ((my_idx + 1) < size)
        {
            d_result[my_idx++] = -1;
        }
    */
    }
    
    //
    //*/
}
