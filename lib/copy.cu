#include <stdio.h>

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void
copy_kernel(const cuFloatComplex* in, cuFloatComplex* out, int batch_size, int load = 1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = batch_size;
    
    if (i < n) 
    {
        for (int x = 0; x < load; x++) 
        {
            out[i].x = in[i].x;
            out[i].y = in[i].y;
        }
    }
}

void apply_copy(const cuFloatComplex* in,
                cuFloatComplex* out,
                int grid_size,
                int block_size,
                int load,
                cudaStream_t stream)
{
    int batch_size = block_size * grid_size;
    
    copy_kernel<<<grid_size, block_size, 0, stream>>>(in, out, batch_size, load);
    
#if 1
    cudaError_t cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess)
    {
        fprintf(stderr, "ERROR: kernel launch failed - \"%s\" (%d).\n", 
                cudaGetErrorString(cudaerr), int(cudaerr));
    }
#endif
}

void get_block_and_grid(int* minGrid, int* minBlock)
{
    cudaError_t rc = cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, copy_kernel, 0, 0);
    if (rc != cudaSuccess)
    {
        fprintf(stderr, "ERROR: - \"%s\" (%d).\n", 
                cudaGetErrorString(rc), int(rc));
    }
}
