#pragma once

namespace my::util
{

/**
 * @brief This class implements a SwapChain. (value_t should be a pointer or reference for best
 *        results)
 *
 * @author Ronja Schnur (catheart97@outlook.com)
 */
template <typename value_t> struct SwapChain
{
    // data
private:
    value_t _buffer[2]{nullptr, nullptr};
    int _index{0};

    // properties
public:
    /**
     * @brief Swaps back and front.
     */
#ifdef __NVCC__
    __device__ __host__ __forceinline__
#else
    inline
#endif
        void
        swap()
    {
        _index = (_index + 1) % 2;
    }

    /**
     * @brief Returns the current front value.
     */
#ifdef __NVCC__
    __device__ __host__ __forceinline__
#else
    inline
#endif
        value_t &
        front()
    {
        return _buffer[_index];
    }

    /**
     * @brief Returns the current back value.
     */
#ifdef __NVCC__
    __device__ __host__ __forceinline__
#else
    inline
#endif
        value_t &
        back()
    {
        return _buffer[(_index + 1) % 2];
    }

    /**
     * @brief Sets the initially front as current front.
     */
#ifdef __NVCC__
    __device__ __host__ __forceinline__
#else
    inline
#endif
        void
        reset()
    {
        _index = 0;
    }

    // constructors
public:
    SwapChain() = default;

    SwapChain(value_t fst, value_t snd)
    {
        _buffer[0] = fst;
        _buffer[1] = snd;
    }
};

} // namespace my::util