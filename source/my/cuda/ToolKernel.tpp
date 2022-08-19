#include "ToolKernel.cuh"

#ifdef __NVCC__
template <typename value_t, typename result_t>
__global__ void my::cuda::count_if(value_t * data, result_t * result, value_t key,
                                   size_t length_data)
{
    const size_t thid = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t num_threads = gridDim.x * blockDim.x;
    if (thid == 0) *result = 0;
    __syncthreads();
    {
        result_t accum{0};
        for (size_t i = thid; i < length_data; i += num_threads)
        {
            if (data[i] == key) accum += 1;
        }

        // reduce all values within a warp
        for (size_t offset = WARP_SIZE / 2; offset > 0; offset /= 2)
            accum += result_t(warp_shfl_down(accum, offset));
        if (thid % WARP_SIZE == 0)
            atomicAdd(result, accum); // first thread of every warp writes result
    }
}
#endif