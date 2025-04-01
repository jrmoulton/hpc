#include <stdio.h>
#include <stdlib.h>

#define WIDTH 1024
#define HEIGHT 1024
#define RGB_SIZE (WIDTH * HEIGHT * 3)
#define GRAY_SIZE (WIDTH * HEIGHT)

// cUDA kernel for RGB to grayscale conversion
__global__ void rgb_to_gray(unsigned char *rgb, unsigned char *gray, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // calculate 1D position
        int gray_pos = row * width + col;
        int rgb_pos = gray_pos * 3;

        // get RGB values
        unsigned char r = rgb[rgb_pos];
        unsigned char g = rgb[rgb_pos + 1];
        unsigned char b = rgb[rgb_pos + 2];

        // https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/
        // convert to grayscale using luminance method
        gray[gray_pos] = (unsigned char)(0.21f * r + 0.72f * g + 0.07f * b);
    }
}

int main() {
    // host pointers
    unsigned char *h_rgb, *h_gray;
    // device pointers
    unsigned char *d_rgb, *d_gray;

    // allocate memory
    h_rgb = (unsigned char *)malloc(RGB_SIZE);
    h_gray = (unsigned char *)malloc(GRAY_SIZE);

    // read input
    FILE *in_file = fopen("gc_conv_1024x1024.raw", "rb");

    fread(h_rgb, sizeof(unsigned char), RGB_SIZE, in_file);
    fclose(in_file);

    // allocate device memory
    cudaMalloc((void **)&d_rgb, RGB_SIZE);
    cudaMalloc((void **)&d_gray, GRAY_SIZE);

    // copy data go gpu
    cudaMemcpy(d_rgb, h_rgb, RGB_SIZE, cudaMemcpyHostToDevice);

    // grid and block dimensions
    int threads_per_block = 256;
    int blocks_per_grid = (WIDTH * HEIGHT + threads_per_block - 1) / threads_per_block;

    // launch gpu program
    rgb_to_gray<<<blocks_per_grid, threads_per_block>>>(d_rgb, d_gray, WIDTH, HEIGHT);

    // sync to ensure that kernel has run
    cudaDeviceSynchronize();

    // copy result back from gpu
    cudaMemcpy(h_gray, d_gray, GRAY_SIZE, cudaMemcpyDeviceToHost);

    // write grayscale image to output file
    FILE *out_file = fopen("gc.raw", "wb");
    fwrite(h_gray, sizeof(unsigned char), GRAY_SIZE, out_file);

    return 0;
}
