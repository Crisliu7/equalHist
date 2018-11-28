__global__ void histogram_gmem_atomics(unsigned char *input, unsigned int width, unsigned int height, unsigned int *output_histogram_local)
{
    // pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // grid dimensions
    int nx = blockDim.x * gridDim.x; 
    int ny = blockDim.y * gridDim.y;

    // linear thread index within 2D block
    int t = threadIdx.x + threadIdx.y * blockDim.x; 

    // total threads in 2D block
    int nt = blockDim.x * blockDim.y; 

    // linear block index within 2D grid
    int g = blockIdx.x + blockIdx.y * gridDim.x;

    //int NUM_PARTS = nx * ny;
    // initialize temporary accumulation array in global memory
    unsigned int *gmem = output_histogram_local + g * NUM_PARTS;
    for (int i = t; i < NUM_BINS; i += nt) {
        gmem[i] = 0;
    }
    

    // process pixels
    // updates our block's partial histogram in global memory
    for (int col = x; col < width; col += nx) {
        for (int row = y; row < height; row += ny) { 
            atomicAdd(&gmem[input[row * width + col]], 1);
        }
    }
    
}

__global__ void histogram_final_accum(const unsigned int *output_histogram_local, int n, unsigned int *output_histogram)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NUM_BINS) {
        unsigned int total = 0;
        for (int j = 0; j < n; j++){
            total += output_histogram_local[i + NUM_PARTS * j];
        } 
        output_histogram[i] = total;
    }
}


__global__ void histogram1DPerBlock(
    unsigned int *output_histogram,
    const unsigned char *input_gpu, int N )
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
        atomicAdd( &output_histogram[i], sHist[ i ] );
    }
}

__global__ void ReductionMin(unsigned int *sdata, unsigned int *results, int n)    //take thread divergence into account
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