__global__ void histogram_generation(unsigned int *output_histogram, unsigned char *input_gpu, int size)
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
    atomicAdd(output_histogram + pos, sum);
  }
  // int x = blockIdx.x*TILE_SIZE+threadIdx.x;
  // int y = blockIdx.y*TILE_SIZE+threadIdx.y;

  // int location =   y*TILE_SIZE*gridDim.x+x;

  // unsigned int location = blockDim.x * blockIdx.x + threadIdx.x;
  // atomicAdd(&(histogram[input[location]]), 1);
  // __syncthreads();
}