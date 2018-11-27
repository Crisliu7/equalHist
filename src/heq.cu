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

#define TILE_SIZE 256
#define NUM_GRAY_LEVELS 256
#define CUDA_TIMING
#define DEBUG
#define WARP_SIZE 32
#define R 32

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
 * 1. Calculate the histogram of the image. Considering to split one big image into multi small images and
 * parallelly caluculate that. 
 *
 * Atomicadd method is initially considered to use, but it will reduce the performance.
 *
 * Better way to do is do per-thread histogrms parallelly, sort each gray value and reduce by key, then reduce all histograms.
 * 
 * 2. Calculate the cumulative distribution function(CDF). Using prefix sum to parallely calculate.
 *
 * 3. Calculate the cdfmin, maybe using the reduction tree method? Or this step may combine with the step 2?
 *
 * 4. Calculate the histogram equalization value with the given formula
 * 
 * 5. Put the calculated value back to 
 */

/** 
 * This method will calculate the histogram of the image. 
 * Three main steps:
 * 1. Replication
 * 2. Padding
 * 3. Interleaved read access
 */

__global__ void histogram_generation(unsigned int *histogram, unsigned char *input, int size)
{

  // allocate shared memory
  __shared__ int Hs[(NUM_GRAY_LEVELS + 1) * R];

  // warp index, lane and number of warps per block
  const int warpid = (int)(threadIdx.x / WARP_SIZE);
  const int lane = threadIdx.x % WARP_SIZE;
  const int warps_block = blockDim.x / WARP_SIZE;
  // printf("warpid: %d lane: %d warps_block: %d", warpid, lane, warps_block);

  // Offset to per-block sub-histogram
  const int off_rep = (NUM_GRAY_LEVELS + 1) * (threadIdx.x % R);

  // constants for interleaved read access
  const int begin = (size / warps_block) * warpid + WARP_SIZE * blockIdx.x + lane;
  const int end = (size / warps_block) * (warpid + 1);
  const int step = WARP_SIZE * gridDim.x;

  // initialization
  for (int pos = threadIdx.x; pos < (NUM_GRAY_LEVELS + 1) * R; pos += blockDim.x)
    Hs[pos] = 0;

  __syncthreads(); // intra-block synchronization

  // main loop
  for (int i = begin; i < end; i += step)
  {
    int d = input[i]; // global memory read

    atomicAdd(&Hs[off_rep + d], 1); // vote in shared memory
  }

  __syncthreads(); // intra-block synchronization

  for (int pos = threadIdx.x; pos < NUM_GRAY_LEVELS; pos += blockDim.x)
  {
    int sum = 0;
    for (int base = 0; base < (NUM_GRAY_LEVELS + 1) * R; base += NUM_GRAY_LEVELS + 1)
    {
      sum += Hs[base + pos];
    }
    atomicAdd(histogram + pos, sum);
  }
  // int x = blockIdx.x*TILE_SIZE+threadIdx.x;
  // int y = blockIdx.y*TILE_SIZE+threadIdx.y;

  // int location =   y*TILE_SIZE*gridDim.x+x;

  // unsigned int location = blockDim.x * blockIdx.x + threadIdx.x;
  // atomicAdd(&(histogram[input[location]]), 1);
  // __syncthreads();
}

// __global__ void get_cdf(unsigned int *histogram, unsigned long int *intensity_probability, int n)
// {
//   unsigned int d_hist_idx = blockDim.x * blockIdx.x + threadIdx.x;
//   if (d_hist_idx == 0 || d_hist_idx >= n)
//   {
//     return;
//   }
//   unsigned int cdf_val = 0;
//   for (int i = 0; i <= d_hist_idx; ++i)
//   {
//     cdf_val = cdf_val + histogram[i];
//   }
//   intensity_probability[d_hist_idx] = cdf_val;
// }

/** TODO: 
 * prefix sum using hillis and steele algorithm
 */

__global__ void prefixSum(unsigned int *histogram, int n, unsigned int *cdf_min)
{
  // extern __shared__ unsigned int temp[]; // allocated on invocation
  // int thid = threadIdx.x;
  // int pout = 0, pin = 1;
  // // Load input into shared memory.
  // // This is exclusive scan, so shift right by one
  // // and set first element to 0
  // temp[pout * n + thid] = (thid > 0) ? histogram[thid - 1] : 0;
  // __syncthreads();
  // for (int offset = 1; offset < n; offset *= 2)
  // {
  //   pout = 1 - pout; // swap double buffer indices
  //   pin = 1 - pout;
  //   if (thid >= offset)
  //     temp[pout * n + thid] += temp[pin * n + thid - offset];
  //   else
  //     temp[pout * n + thid] = temp[pin * n + thid];
  //   __syncthreads();
  // }
  // histogram[thid] = temp[pout * n + thid]; // write output

  int tid = threadIdx.x;

  //USE SHARED MEMORY - COMON WE ARE EXPERIENCED PROGRAMMERS
  __shared__ int Cache[1024];
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
  if (histogram[tid] < intensity_num[i - 1])
  {
    *min_index = intensity_num[i];
  }
}

// /**
//  * find minimum value of cdf
//  */
// __global__ void get_minimum_cdf(unsigned int *cdf,
//                                 unsigned int *cdf_min)
// {
//   extern __shared__ float sdata[];
//   unsigned int thid = threadIdx.x;
//   unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
//   sdata[thid] = cdf[i] + cdf[i + blockDim.x];
//   if (sdata[thid] == 0)
//     sdata[thid] = 1;
//   __syncthreads();

//   for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
//   {
//     if (thid < s)
//     {
//       if (sdata[thid] > sdata[thid + s])
//       {
//         sdata[thid] = sdata[thid + s];
//       }
//     }
//     __syncthreads();
//   }
//   if (thid == 0)
//   {
//     *cdf_min = sdata[0];
//   }
// }

// __global__ void prefixSum(unsigned int *intensity_num,
//                           unsigned int *min_index)
// {

//     for (int i = 1; i < NUM_GRAY_LEVELS; ++i)
//     {
//         intensity_num[i] += intensity_num[i - 1];
//         if (intensity_num[i] < intensity_num[i - 1])
//         {
//             *min_index = intensity_num[i];
//         }
//     }
// }

/**
 * Calculate the probability of each bin's intensity based on the given cdf array
 * @param {unsighed int*} cdf: the array of cdf
 * @param {float*} intensity_probability: the array of probability of each bin's intensity
 * @param {unsigned int} size: image size
 * @param {unsighed int} cdf_min: the minimum value of the cdf
 */
__global__ void calculate_probability(unsigned int *cdf,
                                      float *intensity_probability,
                                      unsigned int img_size,
                                      unsigned int *cdf_min)
{
  unsigned int index = threadIdx.x;
  if (index < NUM_GRAY_LEVELS)
  {
    intensity_probability[index] = ((float)(cdf[index] - *cdf_min)) / (img_size - *cdf_min);
  }
}

/**
 * Finish the histogram equalization by calculating the normalized value, and set them in the output array
 * @param {unsighed char*} input: the array of original grayscale value of the image
 * @param {unsigned char*} output: the array of after histogram equalization grayscale value of the image
 * @param {unsigned int} size: image size
 * @param {float*} intensity_probability: the array of probability of each bin's intensity
 */
__global__ void historam_equalization(unsigned char *input,
                                      unsigned char *output,
                                      unsigned int img_size,
                                      float *intensity_probability)
{

  unsigned int location = blockIdx.x * blockDim.x + threadIdx.x;
  if (location < img_size)
  {
    output[location] = (unsigned char)(intensity_probability[input[location]] * (NUM_GRAY_LEVELS - 1));
  }
  // printf("the final: %d .", output[location]);
}

void histogram_gpu(unsigned char *data,
                   unsigned int height,
                   unsigned int width)
{

  // int gridXSize = 1 + ((width - 1) / TILE_SIZE);
  // int gridYSize = 1 + ((height - 1) / TILE_SIZE);

  // int XSize = gridXSize * TILE_SIZE;
  // int YSize = gridYSize * TILE_SIZE;

  // Both are the same size (CPU/GPU).
  // int size = XSize * YSize;
  int size = width * height;
  int gridSize = 1 + ((size - 1) / TILE_SIZE);

  // // CPU
  // unsigned int *intensity_probability_cpu = new unsigned int[NUM_GRAY_LEVELS];
  // float *cdf_gpu = new unsigned long int[NUM_GRAY_LEVELS];

  // GPU
  unsigned int *histogram;
  float *intensity_probability;
  unsigned int *cdf_min;

  // Allocate arrays in GPU memory
  checkCuda(cudaMalloc((void **)&input_gpu, size * sizeof(unsigned char)));
  checkCuda(cudaMalloc((void **)&output_gpu, size * sizeof(unsigned char)));
  checkCuda(cudaMalloc((void **)&histogram, NUM_GRAY_LEVELS * sizeof(unsigned int)));
  checkCuda(cudaMalloc((void **)&intensity_probability, NUM_GRAY_LEVELS * sizeof(float)));
  checkCuda(cudaMalloc((void **)&cdf_min, sizeof(unsigned int)));

  // Copy data to GPU, data initialization
  checkCuda(cudaMemcpy(input_gpu,
                       data,
                       size * sizeof(char),
                       cudaMemcpyHostToDevice));
  checkCuda(cudaMemset(histogram, 0, NUM_GRAY_LEVELS * sizeof(unsigned int)));
  checkCuda(cudaMemset(intensity_probability, 0, NUM_GRAY_LEVELS * sizeof(float)));
  checkCuda(cudaMemset(cdf_min, 0, sizeof(unsigned int)));

  checkCuda(cudaDeviceSynchronize());

  // Execute algorithm

  // dim3 dimGrid(gridXSize, gridYSize);
  // dim3 dimBlock(TILE_SIZE, TILE_SIZE);
  dim3 dimGrid(gridSize);
  dim3 dimBlock(TILE_SIZE);

// Kernel Call
#if defined(CUDA_TIMING)
  float Ktime;
  TIMER_CREATE(Ktime);
  TIMER_START(Ktime);
#endif

  //step1
  histogram_generation<<<dimGrid, dimBlock>>>(histogram, input_gpu, size);
  // //step2
  // prefixSum<<<1, dimBlock>>>(histogram, NUM_GRAY_LEVELS);
  // //step3-find minimum value
  // get_minimum_cdf<<<1, dimBlock>>>(histogram, cdf_min);
  prefixSum<<<1, dimBlock>>>(histogram, cdf_min);
  //step4
  calculate_probability<<<1, 256>>>(histogram, intensity_probability, size, cdf_min);
  //step5
  historam_equalization<<<dimGrid, dimBlock>>>(input_gpu, output_gpu, size, intensity_probability);

  checkCuda(cudaPeekAtLastError());
  checkCuda(cudaDeviceSynchronize());

#if defined(CUDA_TIMING)
  TIMER_END(Ktime);
  printf("Kernel Execution Time: %f ms\n", Ktime);
#endif

  // Retrieve results from the GPU
  checkCuda(cudaMemcpy(data,
                       output_gpu,
                       size * sizeof(unsigned char),
                       cudaMemcpyDeviceToHost));

  checkCuda(cudaFree(output_gpu));
  checkCuda(cudaFree(input_gpu));
  checkCuda(cudaFree(histogram));
  checkCuda(cudaFree(intensity_probability));
  checkCuda(cudaFree(cdf_min));

  // checkCuda(cudaMemcpy(intensity_probability_cpu,
  //                      histogram,
  //                      NUM_GRAY_LEVELS * sizeof(unsigned int),
  //                      cudaMemcpyDeviceToHost));

  // checkCuda(cudaMemcpy(cdf_gpu,
  //                      intensity_probability,
  //                      NUM_GRAY_LEVELS * sizeof(unsigned long int),
  //                      cudaMemcpyDeviceToHost));

  // Free resources and end the program

  // int cdf_min;
  // for (int i = 0; i < NUM_GRAY_LEVELS; i++)
  // {
  //   if (cdf_gpu[i] != 0)
  //   {
  //     cdf_min = cdf_gpu[i];
  //   }
  // }

  // std::cout << "cdf min : " << cdf_min << std::endl;

  // for (int i = 0; i < NUM_GRAY_LEVELS; i++)
  // {
  //   std::cout << "Value " << i << " : " << intensity_probability_cpu[i] << " " << cdf_gpu[i] << std::endl;
  // }

  // for (long int i = 0; i < 4990464; i++)
  // {
  //   std::cout << data[i] << " ";
  // }
}

/*
void histogram_gpu_warmup(unsigned char *data, 
 unsigned int height, 
 unsigned int width){
 
  int gridXSize = 1 + (( width - 1) / TILE_SIZE);
  int gridYSize = 1 + ((height - 1) / TILE_SIZE);
  
  int XSize = gridXSize*TILE_SIZE;
  int YSize = gridYSize*TILE_SIZE;
  
  // Both are the same size (CPU/GPU).
  int size = XSize*YSize;
  
  // Allocate arrays in GPU memory
  checkCuda(cudaMalloc((void**)&input_gpu , size*sizeof(unsigned char)));
  checkCuda(cudaMalloc((void**)&output_gpu , size*sizeof(unsigned char)));
  
 checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(unsigned char)));
 
 // Copy data to GPU
 checkCuda(cudaMemcpy(input_gpu, 
 data, 
 size*sizeof(char), 
 cudaMemcpyHostToDevice));

  checkCuda(cudaDeviceSynchronize());
 
 // Execute algorithm
 
  dim3 dimGrid(gridXSize, gridYSize);
 dim3 dimBlock(TILE_SIZE, TILE_SIZE);
 
 kernel<<<dimGrid, dimBlock>>>(input_gpu, 
 output_gpu);
 
 checkCuda(cudaDeviceSynchronize());
 
  // Retrieve results from the GPU
  checkCuda(cudaMemcpy(data, 
      output_gpu, 
      size*sizeof(unsigned char), 
      cudaMemcpyDeviceToHost));
 
 // Free resources and end the program
  checkCuda(cudaFree(output_gpu));
  checkCuda(cudaFree(input_gpu));

}*/