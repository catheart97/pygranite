#define PY_ARRAY_UNIQUE_SYMBOL MAGSENSE_ARRAY_API
#define NO_IMPORT_ARRAY

#include <cmath>
#include <cuda.h>

#include "granite/IntegratorSettings.hpp"
#include "granite/SimulationData.hpp"
#include "granite/TrajectoryIntegrator.hpp"

#define MY_USE_CUDA_TYPES
#include "my/math/LinearAlgebra.hpp"
#include "my/util/Util.cuh"

#define KERNEL_3D(                           /**/                                                  \
                  __integrator,              /**/                                                  \
                  __curvature_mode,          /**/                                                  \
                  __abort_mode,              /**/                                                  \
                  __uplift_mode,             /**/                                                  \
                  __windfield_mode,          /**/                                                  \
                  __constants_mode,          /**/                                                  \
                  __additional_volume_mode,  /**/                                                  \
                  __use_topography,          /**/                                                  \
                  __use_reverse_computation, /**/                                                  \
                  __comp_additional         /**/                                                  \
)                                                                                                  \
    MY_VLOG("Kernel selected.");                                                                   \
    compute_<                                                               /**/                   \
             my::math::Vec3,                                                /**/                   \
             granite::Space::Space3D,                                       /**/                   \
             __integrator,                                                  /**/                   \
             granite::BorderMode::Block,                                    /**/                   \
             __curvature_mode,                                              /**/                   \
             __abort_mode,                                                  /**/                   \
             __uplift_mode,                                                 /**/                   \
             __windfield_mode,                                              /**/                   \
             __constants_mode,                                              /**/                   \
             __additional_volume_mode,                                      /**/                   \
             __use_topography,                                              /**/                   \
             __use_reverse_computation,                                     /**/                   \
             false,                                                         /**/                   \
             __comp_additional,                                             /**/                   \
             __additional_volume_mode != granite::AdditionalVolumeMode::Off /**/                   \
             >(_set->_trajectories3d);                                                             \
    return std::move(_set);

#define KERNEL_2D(                           /**/                                                  \
                  __integrator,              /**/                                                  \
                  __border_mode,             /**/                                                  \
                  __curvature_mode,          /**/                                                  \
                  __abort_mode,              /**/                                                  \
                  __windfield_mode,          /**/                                                  \
                  __constants_mode,          /**/                                                  \
                  __additional_volume_mode,  /**/                                                  \
                  __use_reverse_computation, /**/                                                  \
                  __use_sphere_coordinates,  /**/                                                  \
                  __comp_additional         /**/                                                  \
)                                                                                                  \
    MY_VLOG("Kernel selected.");                                                                   \
    compute_<                                                               /**/                   \
             my::math::Vec2,                                                /**/                   \
             Space::Space2D,                                                /**/                   \
             __integrator,                                                  /**/                   \
             __border_mode,                                                 /**/                   \
             __curvature_mode,                                              /**/                   \
             __abort_mode,                                                  /**/                   \
             granite::UpLiftMode::Off,                                      /**/                   \
             __windfield_mode,                                              /**/                   \
             __constants_mode,                                              /**/                   \
             __additional_volume_mode,                                      /**/                   \
             false,                                                         /**/                   \
             __use_reverse_computation,                                     /**/                   \
             __use_sphere_coordinates,                                      /**/                   \
             __comp_additional,                                             /**/                   \
             __additional_volume_mode != granite::AdditionalVolumeMode::Off /**/                   \
             >(_set->_trajectories2d);                                                             \
    return std::move(_set);