
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <math.h>
#include <iostream>

#define TIMER_CREATE(t)               \
  cudaEvent_t t##_start, t##_end;     \
  cudaEventCreate(&t##_start);        \
  cudaEventCreate(&t##_end);               
 
 
#define TIMER_START(t)                \
  cudaEventRecord(t##_start);         \
  cudaEventSynchronize(t##_start);    \
 
 
#define TIMER_END(t)                             \
  cudaEventRecord(t##_end);                      \
  cudaEventSynchronize(t##_end);                 \
  cudaEventElapsedTime(&t, t##_start, t##_end);  \
  cudaEventDestroy(t##_start);                   \
  cudaEventDestroy(t##_end);     
  
#define TILE_SIZE 16
#define NUM_GRAY_LEVELS 256
#define CUDA_TIMING
#define DEBUG

unsigned char *input_gpu;
unsigned char *output_gpu;

double CLOCK() {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC,  &t);
	return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
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
  *   1. Calculate the histogram of the image. Considering to split one big image into multi small images and
  *      parallelly caluculate that. 
  *
  *      Atomicadd method is initially considered to use, but it will reduce the performance.
  *
  *      Better way to do is do per-thread histogrms parallelly, sort each gray value and reduce by key, then reduce all histograms.
  *   
  *   2. Calculate the cumulative distribution function(CDF). Using prefix sum to parallely calculate.
  *
  *   3. Calculate the cdfmin, maybe using the reduction tree method? Or this step may combine with the step 2?
  *
  *   4. Calculate the histogram equalization value with the given formula
  */


__global__ void kernel(unsigned char *input, unsigned long int *output_cdf, 
                       unsigned char *output, unsigned long int im_size, unsigned long int cdf_min){
                       
    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;

    int location = 	y*TILE_SIZE*gridDim.x+x;
    
    float temp = (output_cdf[input[location]] - cdf_min)/(im_size - cdf_min)*(NUM_GRAY_LEVELS - 1);
    float temp2 = round(temp) ;
    output[location] = int(temp2);
    
    printf("the first: %f  . the seond: %f  . the final: %d .", temp, temp2, output[location]);

}

__global__ void cal(unsigned char *input, 
                             unsigned int *output_probability){
    
    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;

    int location = 	y*TILE_SIZE*gridDim.x+x;
    
    atomicAdd(&(output_probability[input[location]]), 1);
    __syncthreads();
    
}

   __global__ void get_cdf(unsigned int *output_probability, unsigned long int *output_cdf, int n)
{
    unsigned int d_hist_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (d_hist_idx == 0 || d_hist_idx >= n)
    {
    	return;
    }
    unsigned int cdf_val = 0;
    for (int i = 0; i <= d_hist_idx; ++i)
    {
    	cdf_val = cdf_val + output_probability[i];
    }
    output_cdf[d_hist_idx] = cdf_val;
}


void histogram_gpu(unsigned char *data, 
                   unsigned int height, 
                   unsigned int width){
    
	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	// Both are the same size (CPU/GPU).
	int size = XSize*YSize;
 
  // CPU
  unsigned int *probability_gpu = new unsigned int [NUM_GRAY_LEVELS];
  unsigned long int *cdf_gpu = new unsigned long int [NUM_GRAY_LEVELS];
  
  // GPU
  unsigned int *output_probability;
  unsigned long int *output_cdf;
  
	
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&input_gpu   , size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(unsigned char)));
 checkCuda(cudaMalloc((void**)&output_probability  , NUM_GRAY_LEVELS*sizeof(unsigned int)));
 checkCuda(cudaMalloc((void**)&output_cdf  , NUM_GRAY_LEVELS*sizeof(unsigned long int)));
	
   checkCuda(cudaMemset(output_probability , 0 , NUM_GRAY_LEVELS*sizeof(unsigned int)));
   checkCuda(cudaMemset(output_cdf , 0 , NUM_GRAY_LEVELS*sizeof(unsigned long int)));
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

	// Kernel Call
	#if defined(CUDA_TIMING)
		float Ktime;
		TIMER_CREATE(Ktime);
		TIMER_START(Ktime);
	#endif
        
   get_probability<<<dimGrid, dimBlock>>>(input_gpu, output_probability);
   get_cdf<<<dimGrid, dimBlock>>>(output_probability, output_cdf, NUM_GRAY_LEVELS);

        
        
        checkCuda(cudaPeekAtLastError());                                     
        checkCuda(cudaDeviceSynchronize());
	
	#if defined(CUDA_TIMING)
		TIMER_END(Ktime);
		printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif
        
	// Retrieve results from the GPU


	checkCuda(cudaMemcpy(probability_gpu, 
			output_probability, 
			NUM_GRAY_LEVELS*sizeof(unsigned int), 
			cudaMemcpyDeviceToHost));
      
	checkCuda(cudaMemcpy(cdf_gpu, 
			output_cdf, 
			NUM_GRAY_LEVELS*sizeof(unsigned long int), 
			cudaMemcpyDeviceToHost));
    // Free resources and end the program
    
   int cdf_min;
   for (int i = 0; i < NUM_GRAY_LEVELS; i++){
     if(cdf_gpu[i] != 0){
       cdf_min = cdf_gpu[i];
     }
   }
   
   std::cout << "cdf min : " << cdf_min << std::endl;

  kernel<<<dimGrid, dimBlock>>>(input_gpu, output_cdf, output_gpu, width*height, cdf_min);
  checkCuda(cudaPeekAtLastError());                                     
  checkCuda(cudaDeviceSynchronize());     
	checkCuda(cudaMemcpy(data, 
			output_gpu, 
			size*sizeof(unsigned char), 
			cudaMemcpyDeviceToHost));
    
  checkCuda(cudaFree(output_probability));
  checkCuda(cudaFree(output_cdf));
	checkCuda(cudaFree(output_gpu));
	checkCuda(cudaFree(input_gpu));
 
  for(int i = 0; i < NUM_GRAY_LEVELS; i++){
    std::cout << "Value " << i << " : " << probability_gpu[i] << "  " << cdf_gpu[i] << std::endl;
  }
  
  for (long int i = 0; i < 4990464; i++){
    std::cout << data[i] << "  ";
  }

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
	checkCuda(cudaMalloc((void**)&input_gpu   , size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(unsigned char)));
	
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

