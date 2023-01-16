#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace granite
{

/**
 * @brief This struct capsules texture information (3D and 2D)
 *
 * @author Ronja Schnur (ronjaschnur@uni-mainz.de)
 */
struct TextureInfo
{
    cudaTextureObject_t Object{0};
    cudaArray_t CudaArray{nullptr};
    bool Initialized{false};
    float * LinearArray{nullptr};
    size_t Pitch{0};
};

} // namespace granite