#include <stdio.h>

#define N  64


inline cudaError_t checkCudaErr(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime error at %s: %s\n", msg, cudaGetErrorString(err));
  }
  return err;
}

//__global__ void matrixMulGPU( int * a, int * b, int * c )
//{
//  /*
//   * Build out this kernel.
//   */
//    int row = threadIdx.y + blockIdx.y * blockDim.y;
//    int col = threadIdx.x + blockIdx.x * blockDim.x;
//    
//    int val = 0;
//    if (row < N && col < N) {
//      for (int i = 0; i < N; ++i) {
//         val += a[row * N + i] * b[i * N + col];
//       }
//    
//      c[row * N + col] = val;
//    }
//}

__global__ void square( float * d_out, float * d_in){
    int idx = threadIdx.x;
    float value = d_in[idx];
    d_out[idx] = value * value *value;
}

/*
 * This CPU function already works, and will run to create a solution matrix
 * against which to verify your work building out the matrixMulGPU kernel.
 */
void matrixMulCPU( int * a, int * b, int * c )
{
  int val = 0;
  for( int row = 0; row < N; ++row )
      for( int col = 0; col < N; ++col )
      {
        val = 0;
        for ( int k = 0; k < N; ++k )
          val += a[row * N + k] * b[k * N + col];
        c[row * N + col] = val;
      }
}

int main()
{
    
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
    
    // generate the input array on host
    float h_in[ARRAY_SIZE];
    for (size_t i = 0; i < ARRAY_SIZE; i++)
    {
        h_in[i] = float(i);
    }
    float h_out[ARRAY_SIZE];

    // declare GPU memory pointers
    float * d_in;
    float * d_out;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);
    
    // transfer the array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // launch the kernel
    square<<<1, ARRAY_SIZE>>>(d_out, d_in);

    // copy back the result array to the CPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // print out the array
    for (size_t i = 0; i < ARRAY_SIZE; i++)
    {
        printf("%f", h_out[i]);
        printf(((i % 4) != 3) ? "\t" : "\n");
    }

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
