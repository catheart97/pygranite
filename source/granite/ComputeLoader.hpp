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

class ComputeLoader
{
    // constructors etc.
public:
    ComputeLoader() = default;

    virtual ~ComputeLoader() = default;

    // methods
public:
    virtual bool hasNext() = 0;
    virtual pybind11::array_t<float> next() = 0;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/**
 * @brief This class is the required "trampolin" class for the pybind11 inheritance.
 *        See <a href="https://pybind11.readthedocs.io/en/stable/advanced/classes.html">here</a>.
 *
 *
 * @author Ronja Schnur (catheart97@outlook.com)
 */
class PyComputeLoader : public ComputeLoader
{
    // constructors
public:
    PyComputeLoader() = default;

    // methods
public:
    bool hasNext() override
    {
        PYBIND11_OVERRIDE_PURE( //
            bool,               //
            ComputeLoader,      //
            hasNext);
    }

    pybind11::array_t<float> next() override
    {
        PYBIND11_OVERRIDE_PURE(       //
            pybind11::array_t<float>, //
            ComputeLoader,            //
            next                      //
        );
    }
};

#endif

} // namespace granite