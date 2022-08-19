#pragma once

namespace my
{

namespace math
{

template <typename value_t>
#ifdef __NVCC__
__device__ __host__ __forceinline__
#endif
    constexpr value_t
    sdiv(const value_t x, const value_t y)
{
    return (x + y - 1) / (y);
}

template <typename value_t>
#ifdef __NVCC__
__device__ __host__ __forceinline__
#endif
    constexpr value_t
    sq(value_t sq)
{
    return sq * sq;
}

} // namespace math

} // namespace my