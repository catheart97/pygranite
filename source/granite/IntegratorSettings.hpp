#pragma once

// need to be defined when using numpy
#define PY_ARRAY_UNIQUE_SYMBOL MAGSENSE_ARRAY_API
#define NO_IMPORT_ARRAY

#include <iostream>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <granite/ComputeLoader.hpp>

namespace granite
{

enum class Space : uint32_t
{
    Space3D = 0,
    Space2D = 1
};

enum class Integrator : uint32_t
{
    ExplicitEuler = 0,
    ClassicRungeKutta = 1
};

enum class BorderMode : uint32_t
{
    Block = 0,
    LoopX = 1,
    LoopY = 2,
    LoopXY = 3
};

enum class CurvatureMode : uint32_t
{
    Off = 0,
    TotalCurvature = 1,
    FastTotalCurvature = 2,
    IndividualAndTotalCurvature = 3
};

enum class AbortMode : uint32_t
{
    Time = 0,
    Length = 1,
    FitLength = 2
};

enum class UpLiftMode : uint32_t
{
    Off = 0,
    Constant = 1,
    Dynamic = 2
};

/**
 * @brief Settings class for @ref TrajectoryIntegrator.
 *
 * @author Ronja Schnur (catheart97@outlook.com)
 */
struct IntegratorSettings
{
    // data
public:
    // Radius for spherical coordinate transform, when set to <= zero, spherical coordinates
    // are disabled. (Default is 0)
    float SphereRadius{0.f};

    // Determines the integrator space. Must match with supplied data. (Default is Space3D)
    granite::Space Space{granite::Space::Space3D};

    // Determines which integrator engine should be used to compute the trajectories.
    // (Default is ClassicRungeKutta)
    granite::Integrator Integrator{granite::Integrator::ClassicRungeKutta};

    // Determines if the simulation should interpolate between windfield timestamps or use
    // a fixed one (the first given).
    // (Default is true)
    bool InterpolateWindfields{true};

    // The integration delta used.
    // (Default is 1)
    float DeltaT{1};

    // The time that needs to pass between to following windfields.
    // (Default is 60)
    float WindfieldTimeDistance{60};

    // Defines the scale of the windfield data.
    // Must not match dimension although in 3D case 3 values are assumed
    std::vector<float> GridScale{25, 25, 25};

    // Specifies the offset which is applied to particles (aka. moves the domain accordingly)
    std::vector<float> Offset{0, 0, 0};

    // Determines in which steps particle positions should be saved.
    // 0 => Save only last
    // 1 => Save all
    // 2 => Save every second
    // etc.
    size_t SaveInterval{1};

    // If set to true, all windfield data is multiplied by -1. Resulting in reversed
    // computation. (p + dt * v => p - dt * v)
    bool Reverse{false};

    // Determines how borders are handled (Only 2D).
    // Block : Stops execution when particle hits or outmoves domain. (default)
    // LoopX : Loops X direction.
    // LoopY : Loops Y direction (Same as Block when using sphere coordinates.).
    // LoopXY: Loops X and Y direction.
    // x - Loop is computed using simple modulo looping.
    // y - Loop is computed using a reflect pattern.
    granite::BorderMode BorderMode{granite::BorderMode::Block};

    // Determines the behaviour of curvature computation.
    // Off                         : No curvature is computed at all
    // TotalCurvature              : TotalCurvature is computed in the standard fashion using
    //                               the second derivative (3D only)
    // FastTotalCurvature          : Total curvature is computed using the angle and method.
    // IndividualAndTotalCurvature : Total curvature and curvature values for each trajectory
    //                               point are computed (3D only)
    granite::CurvatureMode CurvatureMode{granite::CurvatureMode::Off};

    // Determines after which distance computation should stop.
    float MaximumLength{0.f};

    // The maximum Time the simulation will run.
    // Default is 100.
    float MaximumSimulationTime{100.f};

    // Specifies what simulation stop criteria is used.
    //        Time      : No additional stop checks
    //        Length    : Stops the particle, when its trajectory's length exceeds MaximumLength.
    //        FitLength : Stops the particle, when its trajectory's length exceeds MaximumLength,
    //                    while trying to fit it exactly if possible.
    // In all cases the simulation will stop when, no windfield is left for computation (when using
    // InterpolateWindfields) or when the time exceeds MaximumSimulationTime.
    granite::AbortMode AbortMode{granite::AbortMode::Time};

    // The minimum amount of alive particles.
    // Default is 0, which disables checks.
    int32_t MinimumAliveParticles{0};

    // If set, the algorithim will stop particles inside the provided topography. (Only 3D!)
    pybind11::array_t<float> Topography = pybind11::array_t<float>(0);

    // If set, the algorithim will interpolate the value of the provided volume for each
    // particle position.
    std::unordered_map<std::string, pybind11::array_t<float>> AdditionalVolumes;

    // maps the identifiers to the corresponding loader
    std::unordered_map<std::string, ComputeLoader &> AdditionalConstants;

    // code that will be interpreted
    std::vector<std::string> AdditionalCompute;

    // determines how th
    granite::UpLiftMode UpLiftMode{granite::UpLiftMode::Off};
};

} // namespace granite