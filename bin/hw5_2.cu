#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 1024
#define HEIGHT 1024
#define RGB_SIZE (WIDTH * HEIGHT * 3)
#define BLOCK_DIM 32

// cpu transposition
void cpu_transpose(unsigned char *input, unsigned char *output, int width,
                   int height) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int c = 0; c < 3; c++) {
        // 24 bit rgb so stored sequentially
        output[(x * height + y) * 3 + c] = input[(y * width + x) * 3 + c];
      }
    }
  }
}

// global memory transpose which is basically the same as the cpu transpose
__global__ void transpose_global(unsigned char *input, unsigned char *output,
                                 int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // process the rgb the same as in the cpu transpose
    for (int c = 0; c < 3; c++) {
      output[(x * height + y) * 3 + c] = input[(y * width + x) * 3 + c];
    }
  }
}

// shared memory transpose with tiling
__global__ void transpose_shared(unsigned char *input, unsigned char *output,
                                 int width, int height) {
  __shared__ unsigned char tile[BLOCK_DIM][BLOCK_DIM][3];

  // input and output indices
  int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
  int y = blockIdx.y * BLOCK_DIM + threadIdx.y;

  // load into shared memory
  if (x < width && y < height) {
    for (int c = 0; c < 3; c++) {
      tile[threadIdx.y][threadIdx.x][c] = input[(y * width + x) * 3 + c];
    }
  }

  // wait for threads to finish
  __syncthreads();

  // transposed coordinates
  int out_x = blockIdx.y * BLOCK_DIM + threadIdx.x;
  int out_y = blockIdx.x * BLOCK_DIM + threadIdx.y;

  // write from shared memory to output
  if (out_x < height && out_y < width) {
    for (int c = 0; c < 3; c++) {
      // using height here instead of width for the transposing if not square
      // which I guess doesn't really matter
      output[(out_y * height + out_x) * 3 + c] =
          tile[threadIdx.x][threadIdx.y][c];
    }
  }
}

// validate results
int validate_results(unsigned char *gpu_result, unsigned char *cpu_result,
                     int size) {
  for (int i = 0; i < size; i++) {
    if (gpu_result[i] != cpu_result[i]) {
      printf("validation failed");
      return 0;
    }
  }
  printf("both match\n");
  return 1;
}

// compute and report bandwidth
void bandwidth(float time_in_ms, int size_in_bytes) {
  float bandwidth = size_in_bytes / (time_in_ms * 1e-3) / 1e9;
  printf("Bandwidth: %.2f GB/s\n", bandwidth);
}

int main() {

  // device pointers
  unsigned char *d_input, *d_output_global, *d_output_shared;

  // host memory
  unsigned char *h_input = (unsigned char *)malloc(RGB_SIZE);
  unsigned char *h_output_global = (unsigned char *)malloc(RGB_SIZE);
  unsigned char *h_output_shared = (unsigned char *)malloc(RGB_SIZE);
  unsigned char *h_output_cpu = (unsigned char *)malloc(RGB_SIZE);

  // read input
  FILE *input_image = fopen("gc_1024x1024.raw", "rb");

  fread(h_input, sizeof(unsigned char), RGB_SIZE, input_image);

  // device memory
  cudaMalloc((void **)&d_input, RGB_SIZE);
  cudaMalloc((void **)&d_output_global, RGB_SIZE);
  cudaMalloc((void **)&d_output_shared, RGB_SIZE);

  // copy data to GPU
  cudaMemcpy(d_input, h_input, RGB_SIZE, cudaMemcpyHostToDevice);

  // grid and block dimensions
  dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
  dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x,
               (HEIGHT + blockDim.y - 1) / blockDim.y);

  // events for timing referenced from
  // https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsed_time_global, elapsed_time_shared;

  // run global memory and measure performance
  cudaEventRecord(start);
  transpose_global<<<gridDim, blockDim>>>(d_input, d_output_global, WIDTH,
                                          HEIGHT);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_global, start, stop);

  printf("global memory time: %.3f ms\n", elapsed_time_global);
  bandwidth(elapsed_time_global,
            2 * RGB_SIZE); // using 2 * RGB_SIZE because this is two copy
                           // operations, one read one write.

  // run shared memory and measure performance
  cudaEventRecord(start);
  transpose_shared<<<gridDim, blockDim>>>(d_input, d_output_shared, WIDTH,
                                          HEIGHT);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_shared, start, stop);

  printf("shared memory time: %.3f ms\n", elapsed_time_shared);
  bandwidth(elapsed_time_shared, 2 * RGB_SIZE);

  // copy results back from GPU
  cudaMemcpy(h_output_global, d_output_global, RGB_SIZE,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_output_shared, d_output_shared, RGB_SIZE,
             cudaMemcpyDeviceToHost);

  // CPU transpose
  cpu_transpose(h_input, h_output_cpu, WIDTH, HEIGHT);

  // validate
  printf("validate global memory results\n");
  validate_results(h_output_global, h_output_cpu, RGB_SIZE);

  printf("validate shared memory results\n");
  validate_results(h_output_shared, h_output_cpu, RGB_SIZE);

  // output file
  FILE *output_file = fopen("transposed_rgb.raw", "wb");

  fwrite(h_output_shared, sizeof(unsigned char), RGB_SIZE, output_file);

  return 0;
}

// NOTE:
// The measured bandwidth was 25GB/s on the global and 130 GB/s on the shared
// memory.
