#pragma once

#define PY_ARRAY_UNIQUE_SYMBOL MAGSENSE_ARRAY_API
#define NO_IMPORT_ARRAY

#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <thread>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>


#define MY_USE_CUDA_TYPES
#include "my/math/Constants.hpp"
#include "my/math/Arithmetic.hpp"
#include "my/math/LinearAlgebra.hpp"

#include "my/util/Util.cuh"

#include "granite/IntegratorSettings.hpp"
#include "granite/SimulationData.hpp"
#include "granite/TrajectoryIntegrator.hpp"
#include "granite/TrajectorySet.hpp"
#include "granite/WindfieldLoader.hpp"

namespace granite
{

/**
 * @brief This struct bundels the additional settings needed for the CauchyGreenIntegrator class.
 *
 * @author Ronja Schnur (catheart97@outlook.com)
 */
struct CauchyGreenProperties
{
    float DLON{10e-4}, DLAT{10e-4}; // dlambda, dphi
    std::array<float, 2> MIN, MAX;  // GRID Position
    std::array<size_t, 2> DIM;      // GridSize
};

/**
 * @brief This class realizes the computation of the Cauchy Green Tensor using CUDA.
 *
 * @author Ronja Schnur (catheart97@outlook.com)
 */
class CauchyGreenIntegrator : public TrajectoryIntegrator
{
    // data
private:
    CauchyGreenProperties _cgproperties;

    // constructors etc.
public:
    CauchyGreenIntegrator(IntegratorSettings & settings, //
                          WindfieldLoader & loader,      //
                          CauchyGreenProperties & cgproperties)
        : TrajectoryIntegrator(CauchyGreenIntegrator::EnsureSettings(settings), loader),
          _cgproperties{cgproperties}
    {}

    // methods
private:
    // This method modifies the provided IntegratorSettings such that these are optimal for
    // computing the cauchy green tensor (e.g. setting SaveInterval to 0 as only first and
    // last point are required).
    static IntegratorSettings & EnsureSettings(IntegratorSettings & settings)
    {
        if (settings.Space != Space::Space2D)
            MY_USER_ERROR("Cauchy Green computation does not support 3D spaces.");
        settings.SaveInterval = 0;
        return settings;
    }

    // This method test whether the provided min and max values in the CauchyGreenProperties
    // are within the domain.
    CauchyGreenProperties ensureProperties(CauchyGreenProperties & props) const;

    // This function maps the provided values back into the domain.
    my::math::Vec2 ensureDomain(my::math::Vec2 & o, my::math::Vec2 v);

public:
    // Compute cauchy green tensor for each specified point in the grid.
    pybind11::array_t<float> computeCG();
};

} // namespace granite