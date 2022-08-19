#pragma once

#define PY_ARRAY_UNIQUE_SYMBOL MAGSENSE_ARRAY_API
#define NO_IMPORT_ARRAY

#include <algorithm>
#include <iostream>
#include <vector>

#define MY_USE_CUDA_TYPES
#include "my/math/LinearAlgebra.hpp"
#include "my/util/Util.cuh"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "granite/Constants.hpp"
#include "granite/IntegratorSettings.hpp"
#include "granite/AbortReason.hpp"

namespace granite
{

class TrajectoryIntegrator;
class CauchyGreenIntegrator;

/**
 * @brief Class wrapping a set of particle trajectories (either 2D or 3D)
 *
 * @author Ronja Schnur (catheart97@outlook.com)
 */
class TrajectorySet
{
    // data
private:
    Space _space{Space::Space2D};
    size_t _num_trajectories{0};

    std::vector<std::vector<my::math::Vec3>> _trajectories3d;
    std::vector<std::vector<my::math::Vec2>> _trajectories2d;

    std::vector<my::math::Vec3> _lift_constant;
    std::vector<my::math::Vec3> _lift_dynamic;

    std::vector<float> _total_curvature;
    std::vector<float> _individual_lengths;
    std::vector<std::vector<float>> _curvature;

    std::vector<AbortReason> _abort_reasons;

    std::unordered_map<std::string, std::vector<std::vector<float>>> _volume_data;
    std::array<std::vector<std::vector<float>>, MAX_ADDITIONAL_COMPUTE> _compute_data;
    size_t _num_additional_volumes{0};
    size_t _num_additional_compute{0};

    // properties
public:
    void add_(const std::vector<my::math::Vec3> & data_point);

    void add_(const std::vector<my::math::Vec2> & data_point);

    /**
     * @brief Add another point of trajectories (must match with previous given in size).
     * @param data_point
     */
    void add(pybind11::array_t<float> data_point);

    /**
     * @brief Gives the length of each stored trajectory.
     * @return size_t
     */
    size_t lengthTrajectories() const
    {
        if (_space == Space::Space3D) return _trajectories3d.size();
        return _trajectories2d.size();
    }

    /**
     * @brief Returns the number of trajectories stored (const will always be the same value)
     * @return size_t
     */
    constexpr size_t numberTrajectories() const { return _num_trajectories; }

    std::vector<AbortReason> abortReasons() const { return _abort_reasons; }

    // constructors etc
public:
    TrajectorySet() = default;

    TrajectorySet(Space space, size_t num_trajectories);

    TrajectorySet(pybind11::array_t<float> points);

    // methods
public:
    /**
     * @brief Copys the trajectory with provided index to an numpy array.
     */
    pybind11::array_t<float> trajectory(size_t particle_id) const;

    /**
     * @brief Copies the volume info into a numpy array.
     */
    pybind11::array_t<float> volumeInfo(std::string volume_key) const;

    pybind11::array_t<float> computeInfo(size_t compute_id) const;

    /**
     * @brief Copies the individual trajectory lengths (in m) to a numpy array (requires abort
     *        on length to be true)
     */
    pybind11::array_t<float> individualLengths() const;

    /**
     * @brief Creates a numpy point cloud of the trajectory positions (can be used for
     *        visualizations)
     */
    pybind11::array_t<float> cloud() const;

    /**
     * @brief Creates a numpy array containing all trajectories.
     */
    pybind11::array_t<float> trajectories() const;

    /**
     * @brief Creates a numpy array containing the total curvature data.
     *        A value of -1 indicates that this trajectory was aborted due to some reason.
     */
    pybind11::array_t<float> totalCurvatures() const;

    /**
     * @brief Creates a numpy array containing all curvature data.
     */
    pybind11::array_t<float> curvatures() const;

private:
    // trajectory templated by vector type
    template <typename VEC_T>
    pybind11::array_t<float> trajectory_(const std::vector<std::vector<VEC_T>> & trajectories,
                                         size_t particle_id) const;

    // cloud templated by vector type
    template <typename VEC_T>
    pybind11::array_t<float> cloud_(const std::vector<std::vector<VEC_T>> & trajectories) const;

    // trajectories templated by
    template <typename VEC_T>
    pybind11::array_t<float>
    trajectories_(const std::vector<std::vector<VEC_T>> & trajectories) const;

    // ? internal helper functions to reduce code duplication
    template <typename VEC_T>
    void push(std::vector<std::vector<VEC_T>> & target, //
              void * bufferptr) const;

    template <typename VEC_T>
    pybind11::array_t<float> numpy_(pybind11::array::ShapeContainer container,
                                    const std::vector<std::vector<VEC_T>> & trajectories) const;

    // additional declarations
    friend TrajectoryIntegrator;
    friend CauchyGreenIntegrator;
};

} // namespace granite