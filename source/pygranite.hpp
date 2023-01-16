#define PY_ARRAY_UNIQUE_SYMBOL MAGSENSE_ARRAY_API  /// This macro is necessary if numpy is used

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "granite/IntegratorSettings.hpp"
#include "granite/DataLoader.hpp"
#include "granite/TrajectoryIntegrator.hpp"
#include "granite/TrajectorySet.hpp"

#ifdef PYGRANITE_TEST
#include "test/Test.hpp"
#endif