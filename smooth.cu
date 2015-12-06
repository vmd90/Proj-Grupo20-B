#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define DIM  5
#define OFFSET ((int)DIM/2)
#define N (1.0f/25.0f)

__global__ void smooth(const unsigned char *img_in, unsigned char *img_out, int rows, int cols)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    // threads devem se limitar ao tamanho da imagem
	if(ix < cols && iy < rows)
	{
		// calcula a media
		for (int j = iy-OFFSET; j <= iy+OFFSET; ++j)
		{
			for (int k = ix-OFFSET; k <= ix+OFFSET; ++k)
			{
				img_out[(j * cols) + k] += img_in[j * cols + k] * N;
			}
		}
	}
}


void exec_smooth(const unsigned char* img, unsigned char* out, int rows, int cols, int step)
{
	unsigned char *gpu_in, *gpu_out;
	cudaMalloc(&gpu_in, sizeof(unsigned char) * rows * cols);
	cudaMalloc(&gpu_out, sizeof(unsigned char) * rows * cols);

	cudaMemcpy(gpu_in, img, sizeof(unsigned char) * rows * cols, cudaMemcpyHostToDevice);
	
	//dim3 grid(cols/BLOCK_W, rows/BLOCK_H);
	//dim3 block(BLOCK_W, BLOCK_H);
	int blockSize, gridSize, minGridSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, smooth, 0, rows*cols);
	gridSize = (rows*cols + blockSize - 1) / blockSize;
	printf("\n%d, %d\n", gridSize, blockSize);
	smooth<<<gridSize, blockSize>>>(img, gpu_out, cols, rows);

	cudaDeviceSynchronize();

	cudaMemcpy(out, gpu_out, sizeof(unsigned char) * rows * cols, cudaMemcpyDeviceToHost);
	cudaFree(gpu_out);
	cudaFree(gpu_in);
	printf("Smooth completo\n");
}