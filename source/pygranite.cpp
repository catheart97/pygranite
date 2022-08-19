#include "pygranite.hpp"

PYBIND11_MODULE(pygranite, m)
{
    using namespace pybind11::literals;

    m.doc() =
        "pygranite is a library for fast trajectory computation of particles inside of "
        "windfields using cuda hardware acceleration."
        ""
#ifdef PYGRANITE_TEST
        "Build includes test suite."
        ""
#endif
        "Licensing Information:"
        "This software contains source code provided by NVIDIA Corporation."
        "pybind11 Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved."
        "catch2 licenced under Boost Software License."
        "cpp-peglib Copyright (c) 2021 yhirose licenced under MIT license.";

#ifdef PYGRANITE_TEST
    m.def("test_run", &test::test_run);
#endif

    pybind11::enum_<granite::Space>(m, "Space")
        .value("Space2D", granite::Space::Space2D)
        .value("Space3D", granite::Space::Space3D)
        .export_values();

    pybind11::enum_<granite::Integrator>(m, "Integrator")
        .value("ExplicitEuler", granite::Integrator::ExplicitEuler)
        .value("ClassicRungeKutta", granite::Integrator::ClassicRungeKutta)
        .export_values();

    pybind11::enum_<granite::AbortReason>(m, "AbortReason")
        .value("Time", granite::AbortReason::Time)
        .value("Domain", granite::AbortReason::Domain)
        .value("Length", granite::AbortReason::Length)
        .value("Topography", granite::AbortReason::Topography)
        .value("Wind", granite::AbortReason::Wind)
        .export_values();
    
    pybind11::enum_<granite::BorderMode>(m, "BorderMode")
        .value("Block", granite::BorderMode::Block)
        .value("LoopX", granite::BorderMode::LoopX)
        .value("LoopY", granite::BorderMode::LoopY)
        .value("LoopXY", granite::BorderMode::LoopXY)
        .export_values();

    pybind11::enum_<granite::CurvatureMode>(m, "CurvatureMode")
        .value("Off", granite::CurvatureMode::Off)
        .value("FastTotalCurvature", granite::CurvatureMode::FastTotalCurvature)
        .value("IndividualAndTotalCurvature", granite::CurvatureMode::IndividualAndTotalCurvature)
        .value("TotalCurvature", granite::CurvatureMode::TotalCurvature)
        .export_values();

    pybind11::enum_<granite::AbortMode>(m, "AbortMode")
        .value("Time", granite::AbortMode::Time)
        .value("Length", granite::AbortMode::Length)
        .value("FitLength", granite::AbortMode::FitLength)
        .export_values();

    pybind11::enum_<granite::UpLiftMode>(m, "UpLiftMode")
        .value("Off", granite::UpLiftMode::Off)
        .value("Constant", granite::UpLiftMode::Constant)
        .value("Dynamic", granite::UpLiftMode::Dynamic)
        .export_values();

    pybind11::class_<granite::IntegratorSettings>(m, "IntegratorSettings")
        .def(pybind11::init<>())
        .def_readwrite("Space", &granite::IntegratorSettings::Space)
        .def_readwrite("Integrator", &granite::IntegratorSettings::Integrator)
        .def_readwrite("DeltaT", &granite::IntegratorSettings::DeltaT)
        .def_readwrite("SaveInterval", &granite::IntegratorSettings::SaveInterval)
        .def_readwrite("InterpolateWindfields", &granite::IntegratorSettings::InterpolateWindfields)
        .def_readwrite("MinimumAliveParticles", &granite::IntegratorSettings::MinimumAliveParticles)
        .def_readwrite("GridScale", &granite::IntegratorSettings::GridScale)
        .def_readwrite("WindfieldTimeDistance", &granite::IntegratorSettings::WindfieldTimeDistance)
        .def_readwrite("Topography", &granite::IntegratorSettings::Topography)
        .def_readwrite("MaximumLength", &granite::IntegratorSettings::MaximumLength)
        .def_readwrite("AdditionalVolumes", &granite::IntegratorSettings::AdditionalVolumes)
        .def_readwrite("AdditionalCompute", &granite::IntegratorSettings::AdditionalCompute)
        .def_readwrite("Offset", &granite::IntegratorSettings::Offset)
        .def_readwrite("Reverse", &granite::IntegratorSettings::Reverse)
        .def_readwrite("SphereRadius", &granite::IntegratorSettings::SphereRadius)
        .def_readwrite("BorderMode", &granite::IntegratorSettings::BorderMode)
        .def_readwrite("CurvatureMode", &granite::IntegratorSettings::CurvatureMode)
        .def_readwrite("AbortMode", &granite::IntegratorSettings::AbortMode)
        .def_readwrite("UpLiftMode", &granite::IntegratorSettings::UpLiftMode)
        .def_readwrite("AdditionalConstants", &granite::IntegratorSettings::AdditionalConstants)
        .def_readwrite("MaximumSimulationTime",
                       &granite::IntegratorSettings::MaximumSimulationTime);

    pybind11::class_<granite::CauchyGreenProperties>(m, "CauchyGreenProperties")
        .def(pybind11::init<>())
        .def_readwrite("MIN", &granite::CauchyGreenProperties::MIN)
        .def_readwrite("MAX", &granite::CauchyGreenProperties::MAX)
        .def_readwrite("DLON", &granite::CauchyGreenProperties::DLON)
        .def_readwrite("DLAT", &granite::CauchyGreenProperties::DLAT)
        .def_readwrite("DIM", &granite::CauchyGreenProperties::DIM);

    pybind11::class_<granite::TrajectoryIntegrator>(m, "TrajectoryIntegrator")
        .def(pybind11::init<granite::IntegratorSettings &, granite::WindfieldLoader &,
                            granite::TrajectorySet &>())
        .def("compute", &granite::TrajectoryIntegrator::compute);

    pybind11::class_<granite::CauchyGreenIntegrator>(m, "CauchyGreenIntegrator")
        .def(pybind11::init<granite::IntegratorSettings &, granite::WindfieldLoader &,
                            granite::CauchyGreenProperties &>())
        .def("compute", &granite::CauchyGreenIntegrator::computeCG);

    pybind11::class_<granite::ComputeLoader, granite::PyComputeLoader>(m, "ComputeLoader")
        .def(pybind11::init<>())
        .def("hasNext", &granite::ComputeLoader::hasNext)
        .def("next", &granite::ComputeLoader::next);

    pybind11::class_<granite::WindfieldLoader, granite::PyWindfieldLoader>(m, "WindfieldLoader")
        .def(pybind11::init<>())
        .def("hasNext", &granite::WindfieldLoader::hasNext)
        .def("next", &granite::WindfieldLoader::next)
        .def("uplift", &granite::WindfieldLoader::uplift);

    pybind11::class_<granite::TrajectorySet>(m, "TrajectorySet")
        .def(pybind11::init<pybind11::array_t<float>>())
        .def("trajectory", &granite::TrajectorySet::trajectory)
        .def("trajectories", &granite::TrajectorySet::trajectories)
        .def("cloud", &granite::TrajectorySet::cloud)
        .def("totalCurvatures", &granite::TrajectorySet::totalCurvatures)
        .def("curvatures", &granite::TrajectorySet::curvatures)
        .def("individualLengths", &granite::TrajectorySet::individualLengths)
        .def("volumeInfo", &granite::TrajectorySet::volumeInfo)
        .def("computeInfo", &granite::TrajectorySet::computeInfo)
        .def("lengthTrajectories", &granite::TrajectorySet::lengthTrajectories)
        .def("abortReasons", &granite::TrajectorySet::abortReasons)
        .def("numberTrajectories", &granite::TrajectorySet::numberTrajectories);
}