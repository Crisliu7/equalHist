#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
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
#define R 32

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
/* Cuda Error Function */
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

/**
 * The basic algorithm for Histagram Equalization can be divided into four steps:
 * 1. Calculate the histogram of the image.
 *    a. Considering to split one big image into multi small images and parallelly caluculate that. (not used this)
 *    b. Can comprise the image from the CPU, so less GPU memory will be malloced by calling cudamalloc(), which will save more time. (Not used this, this technology is called RLC, Run Length Coding)
 *    c. Atomicadd method is initially considered to use, but it will reduce the performance. But we have no choice, it should definitely be used for histogram.
 *    d. Better way to do is do per-thread histogrms parallelly, sort each gray value and reduce by key, then reduce all histograms. This is a per-block or per-thread histogram generation algorithm.
 *       This method can work well on GTX 1060(Pascal Architecture), not well on NEU Discovery cluster, so it is not used.
 *    e. Used down-sampling method to reduce the whole work of calculating histogram
 * 
 * 2. Calculate the cumulative distribution function(CDF). Using prefix sum to parallely calculate. The algorithm is called Hillis and Steele scan algorithm
 *
 * 3. Calculate the cdfmin, maybe using the reduction tree method
 *
 * 4. Calculate the histogram equalization value with the given formula
 * 
 * 5. Put the calculated value back to generate new image data, and transfer it back to host memory
 */

/** 
 * This method will calculate the histogram of the image. It does not behave well, so it is not used. So reference: https://www.researchgate.net/publication/256674650_An_optimized_approach_to_histogram_computation_on_GPU
 * Three main steps:
 * 1. Replication
 * 2. Padding
 * 3. Interleaved read access
 */
// __global__ void histogram_generation(unsigned int *histogram, unsigned char *input, int size)
// {

//   // allocate shared memory
//   __shared__ int Hs[(BLOCK_SIZE_1D + 1) * R];

//   // warp index, lane and number of warps per block
//   const int warpid = (int)(threadIdx.x / WARP_SIZE);
//   const int lane = threadIdx.x % WARP_SIZE;
//   const int warps_block = blockDim.x / WARP_SIZE;
//   // printf("warpid: %d lane: %d warps_block: %d", warpid, lane, warps_block);

//   // Offset to per-block sub-histogram
//   const int off_rep = (BLOCK_SIZE_1D + 1) * (threadIdx.x % R);

//   // constants for interleaved read access
//   const int begin = (size / warps_block) * warpid + WARP_SIZE * blockIdx.x + lane;
//   const int end = (size / warps_block) * (warpid + 1);
//   const int step = WARP_SIZE * gridDim.x;

//   // initialization
//   for (int pos = threadIdx.x; pos < (BLOCK_SIZE_1D + 1) * R; pos += blockDim.x)
//     Hs[pos] = 0;

//   __syncthreads(); // intra-block synchronization

//   // main loop
//   for (int i = begin; i < end; i += step)
//   {
//     int d = input[i]; // global memory read

//     atomicAdd(&Hs[off_rep + d], 1); // vote in shared memory
//   }

//   __syncthreads(); // intra-block synchronization

//   for (int pos = threadIdx.x; pos < BLOCK_SIZE_1D; pos += blockDim.x)
//   {
//     int sum = 0;
//     for (int base = 0; base < (BLOCK_SIZE_1D + 1) * R; base += BLOCK_SIZE_1D + 1)
//     {
//       sum += Hs[base + pos];
//     }
//     atomicAdd(histogram + pos, sum);
//   }
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
merge64HistogramsToOutput(unsigned int *histogram)
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

  atomicAdd(&histogram[threadIdx.x * 4 + 0], sum02 & 0xffff);
  sum02 >>= 16;
  atomicAdd(&histogram[threadIdx.x * 4 + 2], sum02);

  atomicAdd(&histogram[threadIdx.x * 4 + 1], sum13 & 0xffff);
  sum13 >>= 16;
  atomicAdd(&histogram[threadIdx.x * 4 + 3], sum13);
}

__global__ void
histogram1DPerThread4x64(
    unsigned int *histogram,
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
      merge64HistogramsToOutput<true>(histogram);
    }
  }
  __syncthreads();

  merge64HistogramsToOutput<false>(histogram);
}

  __global__ void
  histogram1DPerBlock(
      unsigned int *pHist,
      const unsigned char *base, int N)
  {
    __shared__ int sHist[256];
    for (int i = threadIdx.x;
         i < 256;
         i += blockDim.x)
    {
      sHist[i] = 0;
    }
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += blockDim.x * gridDim.x)
    {
      unsigned int value = ((unsigned int *)base)[i];

      atomicAdd(&sHist[value & 0xff], 1);
      value >>= 8;
      atomicAdd(&sHist[value & 0xff], 1);
      value >>= 8;
      atomicAdd(&sHist[value & 0xff], 1);
      value >>= 8;
      atomicAdd(&sHist[value], 1);
    }
    __syncthreads();
    for (int i = threadIdx.x;
         i < 256;
         i += blockDim.x)
    {
      atomicAdd(&pHist[i], sHist[i]);
    }
  }

  __global__ void
  histogram1DPerGrid(
      unsigned int *pHist,
      const unsigned char *base, int N)
  {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += blockDim.x * gridDim.x)
    {
      unsigned int value = ((unsigned int *)base)[i];
      atomicAdd(&pHist[value & 0xff], 1);
      value >>= 8;
      atomicAdd(&pHist[value & 0xff], 1);
      value >>= 8;
      atomicAdd(&pHist[value & 0xff], 1);
      value >>= 8;
      atomicAdd(&pHist[value], 1);
    }
  }

  __global__ void kernel(unsigned char *input, unsigned int *output_cdf,
                         unsigned int im_size, unsigned int *cdf_min)
  {

    int location = blockIdx.x * blockDim.x + threadIdx.x;
    input[location] = float(output_cdf[input[location]] - *cdf_min) / float(im_size / 64 - *cdf_min) * (NUM_BINS - 1);
  }

  __global__ void get_histogram(unsigned char *input,
                                unsigned int *output_histogram)
  {
    if (!(threadIdx.x & 63))
    {

      int location = blockIdx.x * blockDim.x + threadIdx.x;
      atomicAdd(&(output_histogram[input[location]]), 1);
    }

    __syncthreads();
  }

  __global__ void get_cdf_prefixSum(unsigned int *histogram)
  {
    int tid = threadIdx.x;

    //USE SHARED MEMORY - COMON WE ARE EXPERIENCED PROGRAMMERS
    __shared__ int Cache[256];
    Cache[tid] = histogram[tid];
    __syncthreads();
    int space = 1;

    //BEGIN
    for (int i = 0; i < 8; i++)
    {
      int temp = Cache[tid];
      int neighbor = 0;
      if ((tid - space) >= 0)
      {
        neighbor = Cache[tid - space];
      }
      __syncthreads(); //AFTER LOADING

      if (tid < space)
      {
        //DO NOTHING
      }
      else
      {
        Cache[tid] = temp + neighbor;
      }

      space = space * 2;
      __syncthreads();
    }

    //REWRITE RESULTS TO MAIN MEMORY
    histogram[tid] = Cache[tid];
  }

  __global__ void reductionMin(unsigned int *sdata, unsigned int *results, int n)
  {
    // extern __shared__ int sdata[];
    unsigned int tx = threadIdx.x;

    // block-wide reduction
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
      __syncthreads();
      if (tx < offset)
      {
        if (sdata[tx + offset] < sdata[tx] || sdata[tx] == 0)
          sdata[tx] = sdata[tx + offset];
      }
    }
    // finally, thread 0 writes the result
    if (threadIdx.x == 0)
    {
      // the result is per-block
      *results = sdata[0];
    }
  }

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

    int gridSize1D_2D = 1 + ((width * height - 1) / BLOCK_SIZE_1D);

    int XSize = gridXSize * TILE_SIZE;
    int YSize = gridYSize * TILE_SIZE;

    // Both are the same size (CPU/GPU).
    int size = XSize * YSize;

    // CPU
    unsigned int *cdf_gpu = new unsigned int[NUM_BINS];

    // GPU
    unsigned int *histogram;
    unsigned int *cdf_min;

    // bool bPeriodicMerge = false;
    // dim3 threads(16, 4, 1);
    // int numthreads = threads.x * threads.y;
    // int numblocks = bPeriodicMerge ? 256 : INTDIVIDE_CEILING(size, numthreads * (255 / 4));

    // Allocate arrays in GPU memory
    checkCuda(cudaMalloc((void **)&input_gpu, size * sizeof(unsigned char)));
    checkCuda(cudaMalloc((void **)&histogram, NUM_BINS * sizeof(unsigned int)));
    checkCuda(cudaMalloc((void **)&cdf_min, sizeof(unsigned int)));

    checkCuda(cudaMemset(histogram, 0, NUM_BINS * sizeof(unsigned int)));
    checkCuda(cudaMemset(cdf_min, 0, sizeof(unsigned int)));

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

    dim3 dimGrid1D_2D(gridSize1D_2D);
    dim3 dimBlock1D_2D(BLOCK_SIZE_1D);

// Kernel Call
#if defined(CUDA_TIMING)
    float Ktime;
    TIMER_CREATE(Ktime);
    TIMER_START(Ktime);
#endif
    //histogram_generation<<<5,256>>>(histogram, input_gpu, width*height);
    //histogram256Kernel<<<gridXSize*gridYSize, 256>>>(histogram, input_gpu, width*height);
    // histogram1DPerThread4x64<<<numblocks, numthreads, numthreads * 256>>>(histogram, input_gpu, size);
    // histogram1DPerBlock<<<400,256/*threads.x*threads.y*/>>>( histogram, input_gpu, width * height / 4);
    // histogram1DPerGrid<<<400,256/*threads.x*threads.y*/>>>( histogram, input_gpu, width * height / 4);
    // get_cdf<<<dimGrid1D, dimBlock1D>>>(histogram, histogram, NUM_BINS);
    get_histogram<<<dimGrid1D_2D, dimBlock1D_2D>>>(input_gpu, histogram);
    get_cdf_prefixSum<<<1, 256>>>(histogram);

    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaDeviceSynchronize());

    reductionMin<<<1, 256>>>(histogram, cdf_min, 256);
    kernel<<<dimGrid1D_2D, dimBlock1D_2D>>>(input_gpu, histogram, width * height, cdf_min);
    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaDeviceSynchronize());

#if defined(CUDA_TIMING)
    TIMER_END(Ktime);
    printf("Kernel Execution Time: %f ms\n", Ktime);
#endif

    checkCuda(cudaMemcpy(data,
                         input_gpu,
                         size * sizeof(unsigned char),
                         cudaMemcpyDeviceToHost));

    checkCuda(cudaFree(histogram));
    checkCuda(cudaFree(cdf_min));
    checkCuda(cudaFree(input_gpu));
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
