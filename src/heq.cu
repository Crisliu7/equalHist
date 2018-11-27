
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <math.h>
#include <iostream>

#define TIMER_CREATE(t)           \
  cudaEvent_t t##_start, t##_end; \
  cudaEventCreate(&t##_start);    \
  cudaEventCreate(&t##_end);

#define TIMER_START(t)        \
  cudaEventRecord(t##_start); \
  cudaEventSynchronize(t##_start);

#define TIMER_END(t)                            \
  cudaEventRecord(t##_end);                     \
  cudaEventSynchronize(t##_end);                \
  cudaEventElapsedTime(&t, t##_start, t##_end); \
  cudaEventDestroy(t##_start);                  \
  cudaEventDestroy(t##_end);

#define TILE_SIZE 16
#define BLOCK_SIZE_1D 256
#define NUM_BINS 256

#define CUDA_TIMING
#define DEBUG

#define WARP_SIZE 32
#define R 9

#define INTDIVIDE_CEILING(i, N) (((i) + (N)-1) / (N))

unsigned char *input_gpu;
unsigned char *output_gpu;

double CLOCK()
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess)
  {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    exit(-1);
  }
#endif
  return result;
}

// Add GPU kernel and functions
// HERE!!!

__global__ void kernel(unsigned char *input, unsigned int *output_cdf,
                       unsigned char *output,
                       unsigned int im_size, unsigned int cdf_min)
{

  int x = blockIdx.x * TILE_SIZE + threadIdx.x;
  int y = blockIdx.y * TILE_SIZE + threadIdx.y;

  int location = y * TILE_SIZE * gridDim.x + x;

  //float temp = float(output_cdf[input[location]] - cdf_min)/float(im_size - cdf_min) * (NUM_BINS - 1);
  //output[location] = round(temp);
  output[location] = (unsigned char)(float(output_cdf[input[location]] - cdf_min) / float(im_size / 4 - cdf_min) * (NUM_BINS - 1));
  //printf("the final: %d .", int(output[location]));
}

__global__ void get_histogram(unsigned char *input,
                              unsigned int *output_histogram)
{

  int x = blockIdx.x * TILE_SIZE + threadIdx.x;
  int y = blockIdx.y * TILE_SIZE + threadIdx.y;

  if (x & 1 && y & 1)
  {
    int location = y * TILE_SIZE * gridDim.x + x;

    atomicAdd(&(output_histogram[input[location]]), 1);
  }

  //__syncthreads();
}

// __global__ void histogram_gmem_atomics(unsigned char *input, int width, int height, unsigned int *output)
// {
//   // pixel coordinates
//   int x = blockIdx.x * blockDim.x + threadIdx.x;
//   int y = blockIdx.y * blockDim.y + threadIdx.y;

//   // grid dimensions
//   int nx = blockDim.x * gridDim.x;
//   int ny = blockDim.y * gridDim.y;

//   // linear thread index within 2D block
//   int t = threadIdx.x + threadIdx.y * blockDim.x;

//   // total threads in 2D block
//   int nt = blockDim.x * blockDim.y;

//   // linear block index within 2D grid
//   int g = blockIdx.x + blockIdx.y * gridDim.x;

//   // initialize temporary accumulation array in global memory
//   unsigned int *gmem = out + g * NUM_PARTS;
//   for (int i = t; i < NUM_BINS; i += nt)
//     gmem[i] = 0;

//   // process pixels
//   // updates our block's partial histogram in global memory
//   for (int col = x; col < width; col += nx)
//     for (int row = y; row < height; row += ny)
//     {

//       atomicAdd(&gmem[input[row * width + col]], 1);
//     }
// }

inline __device__ void
incPrivatized32Element(unsigned char pixval)
{
  extern __shared__ unsigned int privHist[];
  const int blockDimx = 64;
  unsigned int increment = 1 << 8 * (pixval & 3);
  int index = pixval >> 2;
  privHist[index * blockDimx + threadIdx.x] += increment;
}

template <bool bClear>
__device__ void
merge64HistogramsToOutput(unsigned int *output_histogram)
{
  extern __shared__ unsigned int privHist[];

  unsigned int sum02 = 0;
  unsigned int sum13 = 0;
  for (int i = 0; i < 64; i++)
  {
    int index = (i + threadIdx.x) & 63;
    unsigned int myValue = privHist[threadIdx.x * 64 + index];
    if (bClear)
      privHist[threadIdx.x * 64 + index] = 0;
    sum02 += myValue & 0xff00ff;
    myValue >>= 8;
    sum13 += myValue & 0xff00ff;
  }

  atomicAdd(&output_histogram[threadIdx.x * 4 + 0], sum02 & 0xffff);
  sum02 >>= 16;
  atomicAdd(&output_histogram[threadIdx.x * 4 + 2], sum02);

  atomicAdd(&output_histogram[threadIdx.x * 4 + 1], sum13 & 0xffff);
  sum13 >>= 16;
  atomicAdd(&output_histogram[threadIdx.x * 4 + 3], sum13);
}

__global__ void
histogram1DPerThread4x64(
    unsigned int *output_histogram,
    const unsigned char *input, int N)
{
  extern __shared__ unsigned int privHist[];
  const int blockDimx = 64;

  if (blockDim.x != blockDimx)
    return;

  for (int i = threadIdx.x;
       i < 64 * blockDimx;
       i += blockDimx)
  {
    privHist[i] = 0;
  }
  __syncthreads();
  int cIterations = 0;
  for (int i = blockIdx.x * blockDimx + threadIdx.x;
       i < N / 4;
       i += blockDimx * gridDim.x)
  {
    unsigned int value = ((unsigned int *)input)[i];
    incPrivatized32Element(value & 0xff);
    value >>= 8;
    incPrivatized32Element(value & 0xff);
    value >>= 8;
    incPrivatized32Element(value & 0xff);
    value >>= 8;
    incPrivatized32Element(value);
    cIterations += 1;
    if (false && cIterations >= 252 / 4)
    {
      cIterations = 0;
      __syncthreads();
      merge64HistogramsToOutput<true>(output_histogram);
    }
  }
  __syncthreads();

  merge64HistogramsToOutput<false>(output_histogram);
}

__global__ void
histogram1DPerBlock(
    unsigned int *pHist,
    const unsigned char *base, int N )
{
    __shared__ int sHist[256];
    for ( int i = threadIdx.x;
              i < 256;
              i += blockDim.x ) {
        sHist[i] = 0;
    }
    __syncthreads();
    for ( int i = blockIdx.x*blockDim.x+threadIdx.x;
              i < N;
              i += blockDim.x*gridDim.x ) {
            unsigned int value = ((unsigned int *) base)[i];

            atomicAdd( &sHist[ value & 0xff ], 1 ); value >>= 8;
            atomicAdd( &sHist[ value & 0xff ], 1 ); value >>= 8;
            atomicAdd( &sHist[ value & 0xff ], 1 ); value >>= 8;
            atomicAdd( &sHist[ value ]       , 1 );
    }
    __syncthreads();
    for ( int i = threadIdx.x;
              i < 256;
              i += blockDim.x ) {
        atomicAdd( &pHist[i], sHist[ i ] );
    }
}

__global__ void get_cdf(unsigned int *output_histogram,
                        unsigned int *output_cdf,
                        int n)
{
  unsigned int d_hist_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (d_hist_idx >= n)
  {
    return;
  }
  unsigned int cdf_val = 0;
  for (int i = 0; i <= d_hist_idx; ++i)
  {
    cdf_val = cdf_val + output_histogram[i];
  }
  output_cdf[d_hist_idx] = cdf_val;
}

/*
__global__ void get_cdf_min(unsigned int *output_cdf, unsigned int cdf_min)
{


}
*/
__global__ void kernel_warmup(unsigned char *input,
                              unsigned char *output)
{

  int x = blockIdx.x * TILE_SIZE + threadIdx.x;
  int y = blockIdx.y * TILE_SIZE + threadIdx.y;

  int location = y * TILE_SIZE * gridDim.x + x;
  output[location] = x % 255;
}

void histogram_gpu(unsigned char *data,
                   unsigned int height,
                   unsigned int width)
{

  int gridXSize = 1 + ((width - 1) / TILE_SIZE);
  int gridYSize = 1 + ((height - 1) / TILE_SIZE);
  int gridSize_1D = 1 + (NUM_BINS - 1) / BLOCK_SIZE_1D;

  int XSize = gridXSize * TILE_SIZE;
  int YSize = gridYSize * TILE_SIZE;

  // Both are the same size (CPU/GPU).
  int size = XSize * YSize;

  // CPU
  //unsigned int *probability_gpu = new unsigned int [NUM_BINS];
  unsigned int *cdf_gpu = new unsigned int[NUM_BINS];

  // Pinned
  //unsigned char *data_pinned;

  // GPU
  unsigned int *output_histogram;
  unsigned int *output_cdf;

  bool bPeriodicMerge = false;
  dim3 threads(16, 4, 1);
  int numthreads = threads.x * threads.y;
  int numblocks = bPeriodicMerge ? 256 : INTDIVIDE_CEILING(size, numthreads * (255 / 4));

  // Pageable to Pinned memory
  //cudaMallocHost((void**)&data_pinned, size*sizeof(unsigned char));
  //memcpy(data_pinned, data, size*sizeof(unsigned char));

  // Allocate arrays in GPU memory
  checkCuda(cudaMalloc((void **)&input_gpu, size * sizeof(unsigned char)));
  checkCuda(cudaMalloc((void **)&output_gpu, size * sizeof(unsigned char)));
  checkCuda(cudaMalloc((void **)&output_histogram, NUM_BINS * sizeof(unsigned int)));
  checkCuda(cudaMalloc((void **)&output_cdf, NUM_BINS * sizeof(unsigned int)));

  checkCuda(cudaMemset(output_histogram, 0, NUM_BINS * sizeof(unsigned int)));
  checkCuda(cudaMemset(output_cdf, 0, NUM_BINS * sizeof(unsigned int)));
  checkCuda(cudaMemset(output_gpu, 0, size * sizeof(unsigned char)));

  // Copy data to GPU
  checkCuda(cudaMemcpy(input_gpu,
                       data,
                       size * sizeof(char),
                       cudaMemcpyHostToDevice));

  checkCuda(cudaDeviceSynchronize());

  // Execute algorithm

  dim3 dimGrid2D(gridXSize, gridYSize);
  dim3 dimBlock2D(TILE_SIZE, TILE_SIZE);

  dim3 dimGrid1D(gridSize_1D);
  dim3 dimBlock1D(BLOCK_SIZE_1D);

// Kernel Call
#if defined(CUDA_TIMING)
  float Ktime;
  TIMER_CREATE(Ktime);
  TIMER_START(Ktime);
#endif
  //histogram_generation<<<5,256>>>(output_histogram, input_gpu, width*height);
  //histogram256Kernel<<<gridXSize*gridYSize, 256>>>(output_histogram, input_gpu, width*height);
  // get_histogram<<<dimGrid2D, dimBlock2D>>>(input_gpu, output_histogram);
  // histogram1DPerThread4x64<<<numblocks, numthreads, numthreads * 256>>>(output_histogram, input_gpu, size);
  histogram1DPerBlock<<<400,256/*threads.x*threads.y*/>>>( output_histogram, input_gpu, width * height / 4);
  get_cdf<<<dimGrid1D, dimBlock1D>>>(output_histogram, output_cdf, NUM_BINS);

  checkCuda(cudaPeekAtLastError());
  checkCuda(cudaDeviceSynchronize());

  // Retrieve results from the GPU

  /*
	checkCuda(cudaMemcpy(probability_gpu, 
			output_histogram, 
			NUM_BINS*sizeof(unsigned int), 
			cudaMemcpyDeviceToHost));
  */
  checkCuda(cudaFree(output_histogram));

  checkCuda(cudaMemcpy(cdf_gpu,
                       output_cdf,
                       NUM_BINS * sizeof(unsigned int),
                       cudaMemcpyDeviceToHost));
  // Free resources and end the program

  unsigned int cdf_min = INT_MAX;
  for (int i = 0; i < NUM_BINS; i++)
  {
    if (cdf_gpu[i] != 0 && cdf_gpu[i] < cdf_min)
    {
      cdf_min = cdf_gpu[i];
    }
  }

  // std::cout << "cdf min : " << cdf_min << std::endl;
  //kernel<<<dimGrid2D, dimBlock2D>>>(input_gpu, output_cdf, width*height, cdf_min);
  kernel<<<dimGrid2D, dimBlock2D>>>(input_gpu, output_cdf, output_gpu, width * height, cdf_min);
  checkCuda(cudaPeekAtLastError());
  checkCuda(cudaDeviceSynchronize());

#if defined(CUDA_TIMING)
  TIMER_END(Ktime);
  printf("Kernel Execution Time: %f ms\n", Ktime);
#endif

  checkCuda(cudaMemcpy(data,
                       output_gpu,
                       size * sizeof(unsigned char),
                       cudaMemcpyDeviceToHost));
  //memcpy(data, data_pinned, size*sizeof(unsigned char));

  //checkCuda(cudaFreeHost(data_pinned));
  checkCuda(cudaFree(output_cdf));
  checkCuda(cudaFree(output_gpu));
  checkCuda(cudaFree(input_gpu));

  /*
  for(int i = 0; i < NUM_BINS; i++){
    std::cout << "Value " << i << " : " << probability_gpu[i] << "  " << cdf_gpu[i] << std::endl;
  }*/

  /*
  for (long int i = 0; i < 4990464; i++){
    std::cout << data[i] << "  ";
  }*/
}

void histogram_gpu_warmup(unsigned char *data,
                          unsigned int height,
                          unsigned int width)
{

  int gridXSize = 1 + ((width - 1) / TILE_SIZE);
  int gridYSize = 1 + ((height - 1) / TILE_SIZE);

  int XSize = gridXSize * TILE_SIZE;
  int YSize = gridYSize * TILE_SIZE;

  // Both are the same size (CPU/GPU).
  int size = XSize * YSize;

  // Allocate arrays in GPU memory
  checkCuda(cudaMalloc((void **)&input_gpu, size * sizeof(unsigned char)));
  checkCuda(cudaMalloc((void **)&output_gpu, size * sizeof(unsigned char)));

  checkCuda(cudaMemset(output_gpu, 0, size * sizeof(unsigned char)));

  // Copy data to GPU
  checkCuda(cudaMemcpy(input_gpu,
                       data,
                       size * sizeof(char),
                       cudaMemcpyHostToDevice));

  checkCuda(cudaDeviceSynchronize());

  // Execute algorithm

  dim3 dimGrid(gridXSize, gridYSize);
  dim3 dimBlock(TILE_SIZE, TILE_SIZE);

  kernel_warmup<<<dimGrid, dimBlock>>>(input_gpu,
                                       output_gpu);

  checkCuda(cudaDeviceSynchronize());

  // Retrieve results from the GPU
  checkCuda(cudaMemcpy(data,
                       output_gpu,
                       size * sizeof(unsigned char),
                       cudaMemcpyDeviceToHost));

  // Free resources and end the program
  checkCuda(cudaFree(output_gpu));
  checkCuda(cudaFree(input_gpu));
}
