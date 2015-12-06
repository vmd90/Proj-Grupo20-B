#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define DIM  5
#define OFFSET ((int)DIM/2)
#define N (1.0f/25.0f)
#define BLOCK_H 16
#define BLOCK_W 16

__global__ void smooth(const unsigned char *img_in, unsigned char *img_out, int rows, int cols)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    // threads devem se limitar ao tamanho da imagem
	//if(ix >= 2 && ix < cols-2 && iy >= 2 && iy < rows-2)
	if(ix < cols && iy < rows)
	{
		printf("Passou 1\n");
		// calcula a media
		for (int j = iy-OFFSET; j <= iy+OFFSET; ++j)
		{
			for (int k = ix-OFFSET; k <= ix+OFFSET; ++k)
			{
				//int ji = max(j + ix, 0);
				//ji = min(ji, cols-OFFSET);
				//int ki = max(k + iy, 0);
				//ki = min(ki, rows-OFFSET);
				if((j * cols + k) < rows*cols)
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
	
	dim3 grid(cols/BLOCK_W, rows/BLOCK_H);
	dim3 block(BLOCK_W, BLOCK_H);
	smooth<<<grid, block>>>(img, gpu_out, cols, rows);

	cudaDeviceSynchronize();

	cudaMemcpy(out, gpu_out, sizeof(unsigned char) * rows * cols, cudaMemcpyDeviceToHost);
	cudaFree(gpu_out);
	cudaFree(gpu_in);
	printf("Smooth completo\n");
}