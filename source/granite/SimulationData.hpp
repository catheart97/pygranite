#pragma once

#include <cuda.h>

#include "granite/Constants.hpp"
#include "granite/Info.hpp"
#include "granite/Language.hpp"
#include "granite/TextureInfo.hpp"
#include "granite/AbortReason.hpp"

namespace granite
{

/**
 * @brief Struct bundeling the data passed to the gpu, required for TrajectoryIntegrator
 *        Note that there is an overhead here, as some values (usually pointers) might not be used,
 *        This is intentionally to reduce code duplication.
 *
 * @author Ronja Schnur (catheart97@outlook.com)
 */
template <typename VEC_T> struct SimulationData
{
    // coordinate information
    // granite::GridInfo GridInfo;
    granite::ScaleInfo GridScale;
    granite::OffsetInfo Offset;
    granite::DomainInfo Domain;

    // textures
    cudaTextureObject_t WindfieldBack[3];
    cudaTextureObject_t WindfieldFront[3];
    cudaTextureObject_t Topography;
    cudaTextureObject_t AdditionalVolumes[MAX_ADDITIONAL_VOLUMES];
    size_t NumAdditionalVolumes{0};
    size_t NumAdditionalCompute{0};

    // time stamps of front and back + current timestamp
    float TimeBack{0}, TimeFront{0}, Time{0};

    // particle
    size_t NumParticles{0};
    VEC_T * BackParticles{nullptr};
    VEC_T * FrontParticles{nullptr};
    AbortReason * Status{nullptr};
    VEC_T * LastDirections{nullptr};
    VEC_T * Lift{nullptr};
    float * TotalCurvature{nullptr};
    float * Curvature{nullptr};
    float * ParticleVolumeInfo[MAX_ADDITIONAL_VOLUMES];
    float * ParticleComputeInfo[MAX_ADDITIONAL_COMPUTE];
    float * CurrentLength{nullptr};
    float * ComputeData[NUM_CONSTANT_ADDITIONAL_COMPUTE]{0, 0, 0, 0};

    // maximum distance particles are allowed to move
    float MaximumLength{0.f};

    // integrator step size
    float DeltaT;

    // sphere transform parameters
    float Radius;

    // dynamic computation
    granite::ASTANode * ASTNodes[MAX_ADDITIONAL_COMPUTE];
    size_t ASTSizes[MAX_ADDITIONAL_COMPUTE];
};

} // namespace granite
