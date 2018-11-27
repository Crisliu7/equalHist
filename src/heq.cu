
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
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
#define NUM_BINS 256

#define CUDA_TIMING
#define DEBUG

#define WARP_SIZE 32
#define R 9

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
                       unsigned int im_size, unsigned int *cdf_min){
   /*              
    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;
    int location = 	y*TILE_SIZE*gridDim.x+x;
    input[location] = round(float(output_cdf[input[location]] - *cdf_min)/float(im_size/4 - *cdf_min) * (NUM_BINS - 1));
  */

    int location = blockIdx.x * blockDim.x+threadIdx.x;
    input[location] = float(output_cdf[input[location]] - *cdf_min)/float(im_size/64 - *cdf_min) * (NUM_BINS - 1);    
    //printf("the final: %d .", int(output[location]));  

}

__global__ void get_histogram(unsigned char *input, 
                             unsigned int *output_histogram
                             //int offset
                             ){
    /*
    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;
    //if(x % 4 == 0 && y % 4 == 0){
    if(x & 1 && y & 1){
      int location = 	offset + y*TILE_SIZE*gridDim.x+x;
      atomicAdd(&(output_histogram[input[location]]), 1);   
    
    }*/
    if( !(threadIdx.x & 63)){
    
      int location = 	blockIdx.x * blockDim.x+threadIdx.x;
      atomicAdd(&(output_histogram[input[location]]), 1); 
        
    }
 
    //__syncthreads();  
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

/*
__global__ void get_cdf_naive(unsigned int *output_histogram, 
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
}*/

__global__ void reductionMin(unsigned int *sdata, unsigned int *results, int n)
{	
	// extern __shared__ int sdata[]; 
	unsigned int tx = threadIdx.x; 

	// block-wide reduction
	for(unsigned int offset = blockDim.x>>1; offset > 0; offset >>= 1)
	{
		__syncthreads();
		if(tx < offset)
	    {
			if(sdata[tx + offset] < sdata[tx] || sdata[tx] == 0)
				sdata[tx] = sdata[tx + offset];
		}

	}
	// finally, thread 0 writes the result 
	if(threadIdx.x == 0) 
	{ 
		// the result is per-block 
		*results = sdata[0]; 
	} 
}


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
  int gridSize_1D = 1 + (NUM_BINS - 1)/ BLOCK_SIZE_1D;
  
  int gridSize1D_2D = 1 + (( width*height - 1) / BLOCK_SIZE_1D);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
 
	
	// Both are the same size (CPU/GPU).
	int size = XSize*YSize;
 
  // CPU
  unsigned int *cdf_gpu = new unsigned int [NUM_BINS];
  
  // Pinned
  //unsigned char *data_pinned;
  
  // GPU
  unsigned int *output_histogram;
  //unsigned int *output_cdf;
  unsigned int *cdf_min;
  
  // Pageable to Pinned memory
  //cudaMallocHost((void**)&data_pinned, size*sizeof(unsigned char));
  //memcpy(data_pinned, data, size*sizeof(unsigned char));
    
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&input_gpu   , size*sizeof(unsigned char)));
  checkCuda(cudaMalloc((void**)&output_histogram  , NUM_BINS*sizeof(unsigned int)));
  //checkCuda(cudaMalloc((void**)&output_cdf  , NUM_BINS*sizeof(unsigned int)));
  checkCuda(cudaMalloc((void**)&cdf_min  , sizeof(unsigned int)));
	
  checkCuda(cudaMemset(output_histogram , 0 , NUM_BINS*sizeof(unsigned int)));
  //checkCuda(cudaMemset(output_cdf , 0 , NUM_BINS*sizeof(unsigned int)));
  checkCuda(cudaMemset(cdf_min, 0, sizeof(unsigned int)));
	

  // Grid & Block Size
  //dim3 dimGrid2D(gridXSize, gridYSize);
  //dim3 dimBlock2D(TILE_SIZE, TILE_SIZE);

  
  
 // create streams
 /*
   const int nStreams = 2;
   const int streamSize = width*height/nStreams;
   std:: cout << "stream size: " << streamSize<<std::endl;
   const int streamBytes = streamSize * sizeof(unsigned char);
   cudaStream_t stream[nStreams];
  for (int i = 0; i < nStreams; ++i){
   checkCuda(cudaStreamCreate(&stream[i]));
  }
   */
 
  // Copy data to GPU
  
  checkCuda(cudaMemcpy(input_gpu, 
      data, 
      size*sizeof(unsigned char), 
      cudaMemcpyHostToDevice)); 

  //checkCuda(cudaDeviceSynchronize());

  // Execute algorithm
    
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
 
  get_histogram<<<dimGrid1D_2D, dimBlock1D_2D>>>(input_gpu, output_histogram);
  //get_cdf_naive<<<dimGrid1D, dimBlock1D>>>(output_histogram, output_cdf, NUM_BINS);
  get_cdf_prefixSum<<<1, 256>>>(output_histogram);
  reductionMin<<<1, 256>>>(output_histogram, cdf_min, 256);    
        
  //checkCuda(cudaPeekAtLastError());                                     
  //checkCuda(cudaDeviceSynchronize());
	

	// Retrieve results from the GPU
  /* 
	checkCuda(cudaMemcpy(cdf_gpu, 
			output_cdf, 
			NUM_BINS*sizeof(unsigned int), 
			cudaMemcpyDeviceToHost));
    // Free resources and end the program
  */
    /*
   unsigned int cdf_min = INT_MAX;
   for (int i = 0; i < NUM_BINS; i++){
     if(cdf_gpu[i] != 0 && cdf_gpu[i] < cdf_min){
       cdf_min = cdf_gpu[i];
     }
   }*/
   
  // std::cout << "cdf min : " << cdf_min << std::endl;
  kernel<<<dimGrid1D_2D, dimBlock1D_2D>>>(input_gpu, output_histogram, width*height, cdf_min);



  checkCuda(cudaPeekAtLastError());                                     
  checkCuda(cudaDeviceSynchronize()); 
  
  
 	#if defined(CUDA_TIMING)
		TIMER_END(Ktime);
		printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif
        
          
	checkCuda(cudaMemcpy(data, 
			input_gpu, 
			size*sizeof(unsigned char), 
			cudaMemcpyDeviceToHost));
  //memcpy(data, data_pinned, size*sizeof(unsigned char));  
  
  
  //checkCuda(cudaFreeHost(data_pinned));
  checkCuda(cudaFree(output_histogram));
  checkCuda(cudaFree(cdf_min));
  //checkCuda(cudaFree(output_cdf));
	//checkCuda(cudaFree(output_gpu));
	checkCuda(cudaFree(input_gpu));
 
 /*
  for(int i = 0; i < NUM_BINS; i++){
    std::cout << "Value " << i << " : " << probability_gpu[i] << "  " << cdf_gpu[i] << std::endl;
  }*/
 /*
  for(int i = 0; i < NUM_BINS*NUM_PARTS; i++){
    std::cout << "Value " << i << " : " << hist_local_gpu[i] << "  "  << std::endl;
  }  */
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

