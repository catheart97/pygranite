#pragma once

#define PY_ARRAY_UNIQUE_SYMBOL MAGSENSE_ARRAY_API
#define NO_IMPORT_ARRAY

#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace granite
{

/**
 * @brief This class represents the interface for the windfield loader.
 *        This class should be overwritten from python to allow dynamic loading during computation,
 *        to reduce the amout of required memory.
 *
 * @author Ronja Schnur (catheart97@outlook.com)
 */
class WindfieldLoader
{
    // constructors etc.
public:
    WindfieldLoader() = default;

    virtual ~WindfieldLoader() = default;

    // methods
public:
    /**
     * @brief Override if you have another windfield remainding that needs to be loaded.
     *
     * @return true next() will be called once more.
     * @return false next() won't be called anymore.
     */
    virtual bool hasNext() = 0;

    /**
     * @brief Should return depending on your setting, either
     *        a python list with 3 numpy arrays 3 (dimensions in (z, y, x)), representing
     *        the individual u, v, w maps,
     *        or a python list with 2 numpy arrays (dimensions in (y, x)), representing the
     *        individual u and v maps.
     *
     * @return std::vector<pybind11::array_t<float>>
     */
    virtual std::vector<pybind11::array_t<float>> next() = 0;

    virtual pybind11::array_t<float> uplift() { return pybind11::array_t<float>(); }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/**
 * @brief This class is the required "trampolin" class for the pybind11 inheritance.
 *        See <a href="https://pybind11.readthedocs.io/en/stable/advanced/classes.html">here</a>.
 *
 *
 * @author Ronja Schnur (catheart97@outlook.com)
 */
class PyWindfieldLoader : public WindfieldLoader
{
    // constructors
public:
    PyWindfieldLoader() = default;

    // methods
public:
    bool hasNext() override
    {
        PYBIND11_OVERRIDE_PURE( //
            bool,               //
            WindfieldLoader,    //
            hasNext);
    }

    std::vector<pybind11::array_t<float>> next() override
    {
        PYBIND11_OVERRIDE_PURE(                    //
            std::vector<pybind11::array_t<float>>, //
            WindfieldLoader,                       //
            next                                   //
        );
    }

    pybind11::array_t<float> uplift() override
    {
        PYBIND11_OVERRIDE(            //
            pybind11::array_t<float>, //
            WindfieldLoader,          //
            uplift, );
    }
};

#endif

} // namespace granite