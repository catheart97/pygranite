#define PY_ARRAY_UNIQUE_SYMBOL MAGSENSE_ARRAY_API  /// This macro is necessary if numpy is used

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "granite/IntegratorSettings.hpp"
#include "granite/WindfieldLoader.hpp"
#include "granite/ComputeLoader.hpp"
#include "granite/TrajectoryIntegrator.hpp"
#include "granite/TrajectorySet.hpp"
#include "granite/CauchyGreenIntegrator.hpp"

#ifdef PYGRANITE_TEST
#include "test/Test.hpp"
#endif