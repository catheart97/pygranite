#pragma once

namespace my
{

namespace cuda
{

template <typename value_t, typename result_t>
#ifdef __NVCC__
__global__
#endif
    void
    count_if(value_t * data, result_t * result, value_t key, size_t length_data);


} // namespace cuda

} // namespace my

#include "ToolKernel.tpp"

