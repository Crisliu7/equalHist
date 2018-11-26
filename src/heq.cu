
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
#define BLOCK_SIZE_1D 256
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


__global__ void kernel(unsigned char *input, unsigned int *output_cdf, 
                       //unsigned char *output, 
                       unsigned int im_size, unsigned int cdf_min){
                       
    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;

    int location = 	y*TILE_SIZE*gridDim.x+x;
    

    
    //float temp = float(output_cdf[input[location]] - cdf_min)/float(im_size - cdf_min) * (NUM_GRAY_LEVELS - 1);
    //output[location] = round(temp);
    input[location] = round(float(output_cdf[input[location]] - cdf_min)/float(im_size/4 - cdf_min) * (NUM_GRAY_LEVELS - 1));
    //printf("the final: %d .", int(output[location]));  


}


__global__ void get_histogram(unsigned char *input, 
                             unsigned int *output_histogram){
    
    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;
    
    if(x & 1 && y & 1){
      int location = 	y*TILE_SIZE*gridDim.x+x;
    
      atomicAdd(&(output_histogram[input[location]]), 1);   
    
    }

    //__syncthreads();
    
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
                       unsigned char *output){

    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;

    int location = 	y*TILE_SIZE*gridDim.x+x;
    output[location] = x%255;

}


void histogram_gpu(unsigned char *data, 
                   unsigned int height, 
                   unsigned int width){
    
	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
  int gridSize_1D = 1 + (NUM_GRAY_LEVELS - 1)/ BLOCK_SIZE_1D;
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	// Both are the same size (CPU/GPU).
	int size = XSize*YSize;
 
  // CPU
  //unsigned int *probability_gpu = new unsigned int [NUM_GRAY_LEVELS];
  unsigned int *cdf_gpu = new unsigned int [NUM_GRAY_LEVELS];
  
  // Pinned
  unsigned char *data_pinned;
  
  // GPU
  unsigned int *output_histogram;
  unsigned int *output_cdf;
  
  // Pageable to Pinned memory
  cudaMallocHost((void**)&data_pinned, size*sizeof(unsigned char));
  memcpy(data_pinned, data, size*sizeof(unsigned char));
    
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&input_gpu   , size*sizeof(unsigned char)));
	//checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(unsigned char)));
  checkCuda(cudaMalloc((void**)&output_histogram  , NUM_GRAY_LEVELS*sizeof(unsigned int)));
  checkCuda(cudaMalloc((void**)&output_cdf  , NUM_GRAY_LEVELS*sizeof(unsigned int)));
	
  checkCuda(cudaMemset(output_histogram , 0 , NUM_GRAY_LEVELS*sizeof(unsigned int)));
  checkCuda(cudaMemset(output_cdf , 0 , NUM_GRAY_LEVELS*sizeof(unsigned int)));
  //checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(unsigned char)));
	
  // Copy data to GPU
  checkCuda(cudaMemcpy(input_gpu, 
      data_pinned, 
      size*sizeof(char), 
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
        
  get_histogram<<<dimGrid2D, dimBlock2D>>>(input_gpu, output_histogram);
  get_cdf<<<dimGrid1D, dimBlock1D>>>(output_histogram, output_cdf, NUM_GRAY_LEVELS);

        
        
  checkCuda(cudaPeekAtLastError());                                     
  checkCuda(cudaDeviceSynchronize());
	

	// Retrieve results from the GPU

  /*
	checkCuda(cudaMemcpy(probability_gpu, 
			output_histogram, 
			NUM_GRAY_LEVELS*sizeof(unsigned int), 
			cudaMemcpyDeviceToHost));
  */
  checkCuda(cudaFree(output_histogram));
     
	checkCuda(cudaMemcpy(cdf_gpu, 
			output_cdf, 
			NUM_GRAY_LEVELS*sizeof(unsigned int), 
			cudaMemcpyDeviceToHost));
    // Free resources and end the program
    
   unsigned int cdf_min = INT_MAX;
   for (int i = 0; i < NUM_GRAY_LEVELS; i++){
     if(cdf_gpu[i] != 0 && cdf_gpu[i] < cdf_min){
       cdf_min = cdf_gpu[i];
     }
   }
   
  // std::cout << "cdf min : " << cdf_min << std::endl;
  kernel<<<dimGrid2D, dimBlock2D>>>(input_gpu, output_cdf, width*height, cdf_min);
  //kernel<<<dimGrid2D, dimBlock2D>>>(input_gpu, output_cdf, output_gpu, width*height, cdf_min);
  checkCuda(cudaPeekAtLastError());                                     
  checkCuda(cudaDeviceSynchronize()); 
  
  
 	#if defined(CUDA_TIMING)
		TIMER_END(Ktime);
		printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif
        
            
	checkCuda(cudaMemcpy(data_pinned, 
			input_gpu, 
			size*sizeof(unsigned char), 
			cudaMemcpyDeviceToHost));
  memcpy(data, data_pinned, size*sizeof(unsigned char));  
  
  
  checkCuda(cudaFreeHost(data_pinned));
  checkCuda(cudaFree(output_cdf));
	//checkCuda(cudaFree(output_gpu));
	checkCuda(cudaFree(input_gpu));
 
 /*
  for(int i = 0; i < NUM_GRAY_LEVELS; i++){
    std::cout << "Value " << i << " : " << probability_gpu[i] << "  " << cdf_gpu[i] << std::endl;
  }*/
  
  /*
  for (long int i = 0; i < 4990464; i++){
    std::cout << data[i] << "  ";
  }*/

}


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
    
    kernel_warmup <<<dimGrid, dimBlock>>>(input_gpu, 
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

}

