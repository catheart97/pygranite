#pragma once

#define PY_ARRAY_UNIQUE_SYMBOL MAGSENSE_ARRAY_API
#define NO_IMPORT_ARRAY

#include <string>
#include <vector>
#include <unordered_map>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace granite
{

using map_t = std::unordered_map<std::string, pybind11::array_t<float>>;

/**
 * @brief This class represents the interface for the data loader which replaces DataLoader and
 * ComputeLoader as of version 1.5.0. This class should be overwritten from python to allow dynamic
 * loading during computation, to reduce the amout of required memory.
 *
 * @author Ronja Schnur (ronjaschnur@uni-mainz.de)
 */
class DataLoader
{
    // constructors etc.
public:
    DataLoader() = default;

    virtual ~DataLoader() = default;

    // methods
public:
    /**
     * @brief This method should update the internal structure of your implementation to upgrade to
     * the next step (where necessary).
     */
    virtual bool step() = 0;

    /**
     * @brief Should return depending on your setting, either
     *        a python list with 3 numpy arrays 3 (dimensions in (z, y, x)), representing
     *        the individual u, v, w maps,
     *        or a python list with 2 numpy arrays (dimensions in (y, x)), representing the
     *        individual u and v maps.
     *
     * @return std::vector<pybind11::array_t<float>>
     */
    virtual std::vector<pybind11::array_t<float>> windfield() = 0;

    /**
     * @brief Should return (at every datastep) a list with the length of particles in computation
     *        of offsets (either vec3 or vec2) to be applied in each integration step.
     * @return pybind11::array_t<float>
     */
    virtual pybind11::array_t<float> uplift() { return pybind11::array_t<float>(); }

    /**
     * @brief Dict of variable names mapping to a list of length of particles in computation which
     * will be used in additional compute. (! It is assumed that the variable names are always the
     * same)
     * @return std::unordered_map<std::string, pybind11::array_t<float>>
     */
    virtual map_t constants()
    {
        return map_t();
    }

    /**
     * @brief Dict of variable names mapping to additional volumes to be interpolated during
     * computation.
     *        (! It is assumed that the variable names are always the same)
     * @return std::unordered_map<std::string, pybind11::array_t<float>>
     */
    virtual map_t additionalVolumes()
    {
        return map_t();
    }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/**
 * @brief This class is the required "trampolin" class for the pybind11 inheritance.
 *        See <a href="https://pybind11.readthedocs.io/en/stable/advanced/classes.html">here</a>.
 *
 *
 * @author Ronja Schnur (ronjaschnur@uni-mainz.de)
 */
class PyDataLoader : public DataLoader
{
    // constructors
public:
    using DataLoader::DataLoader;

    // methods
public:
    bool step() override
    {
        PYBIND11_OVERRIDE_PURE( //
            bool,               //
            DataLoader,         //
            step                //
        );
    }

    std::vector<pybind11::array_t<float>> windfield() override
    {
        PYBIND11_OVERRIDE_PURE(                    //
            std::vector<pybind11::array_t<float>>, //
            DataLoader,                            //
            windfield                              //
        );
    }

    pybind11::array_t<float> uplift() override
    {
        PYBIND11_OVERRIDE(            //
            pybind11::array_t<float>, //
            DataLoader,               //
            uplift                    //
        );
    }

    map_t constants() override
    {
        PYBIND11_OVERRIDE(                                             //
            map_t, //
            DataLoader,                                                //
            constants                                                  //
        );
    }

    map_t additionalVolumes() override
    {
        PYBIND11_OVERRIDE(                                             //
            map_t, //
            DataLoader,                                                //
            additionalVolumes,                                         //
        );
    }
};

#endif

} // namespace granite