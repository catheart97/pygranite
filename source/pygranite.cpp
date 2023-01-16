#include "pygranite.hpp"

PYBIND11_MODULE(pygranite, m)
{
    using namespace pybind11::literals;

    m.doc() =
        "pygranite is a library for fast trajectory computation of particles inside of "
        "windfields using cuda hardware acceleration."
        ""
        "Licensing Information:\n\n"
        "This software contains source code provided by NVIDIA Corporation.\n\n"
        "pybind11 Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved."
        "Redistribution and use in source and binary forms, with or without"
        "modification, are permitted provided that the following conditions are met:"
        "1. Redistributions of source code must retain the above copyright notice, this"
        "list of conditions and the following disclaimer."
        "2. Redistributions in binary form must reproduce the above copyright notice,"
        "this list of conditions and the following disclaimer in the documentation"
        "and/or other materials provided with the distribution."
        "3. Neither the name of the copyright holder nor the names of its contributors"
        "may be used to endorse or promote products derived from this software"
        "without specific prior written permission."
        "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND"
        "ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED"
        "WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE"
        "DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE"
        "FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL"
        "DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR"
        "SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER"
        "CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,"
        "OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE"
        "OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\n\n"
        "catch2 licenced under Boost Software License.\n\n"
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

    pybind11::enum_<granite::WindfieldMode>(m, "WindfieldMode")
        .value("Constant", granite::WindfieldMode::Constant)
        .value("Dynamic", granite::WindfieldMode::Dynamic)
        .export_values();

    pybind11::enum_<granite::AdditionalVolumeMode>(m, "AdditionalVolumeMode")
        .value("Off", granite::AdditionalVolumeMode::Off)
        .value("Constant", granite::AdditionalVolumeMode::Constant)
        .value("Dynamic", granite::AdditionalVolumeMode::Dynamic)
        .export_values();

    pybind11::enum_<granite::ConstantsMode>(m, "ConstantsMode")
        .value("Off", granite::ConstantsMode::Off)
        .value("Constant", granite::ConstantsMode::Constant)
        .value("Dynamic", granite::ConstantsMode::Dynamic)
        .export_values();

    pybind11::class_<granite::IntegratorSettings>(m, "IntegratorSettings")
        .def(pybind11::init<>())
        .def_readwrite("Space", &granite::IntegratorSettings::Space)
        .def_readwrite("Integrator", &granite::IntegratorSettings::Integrator)
        .def_readwrite("DeltaT", &granite::IntegratorSettings::DeltaT)
        .def_readwrite("SaveInterval", &granite::IntegratorSettings::SaveInterval)
        .def_readwrite("MinimumAliveParticles", &granite::IntegratorSettings::MinimumAliveParticles)
        .def_readwrite("GridScale", &granite::IntegratorSettings::GridScale)
        .def_readwrite("DataTimeDistance", &granite::IntegratorSettings::DataTimeDistance)
        .def_readwrite("Topography", &granite::IntegratorSettings::Topography)
        .def_readwrite("MaximumLength", &granite::IntegratorSettings::MaximumLength)
        .def_readwrite("AdditionalCompute", &granite::IntegratorSettings::AdditionalCompute)
        .def_readwrite("Offset", &granite::IntegratorSettings::Offset)
        .def_readwrite("Reverse", &granite::IntegratorSettings::Reverse)
        .def_readwrite("SphereRadius", &granite::IntegratorSettings::SphereRadius)
        .def_readwrite("BorderMode", &granite::IntegratorSettings::BorderMode)
        .def_readwrite("CurvatureMode", &granite::IntegratorSettings::CurvatureMode)
        .def_readwrite("AbortMode", &granite::IntegratorSettings::AbortMode)
        .def_readwrite("UpLiftMode", &granite::IntegratorSettings::UpLiftMode)
        .def_readwrite("WindfieldMode", &granite::IntegratorSettings::WindfieldMode)
        .def_readwrite("AdditionalVolumeMode", &granite::IntegratorSettings::AdditionalVolumeMode)
        .def_readwrite("ConstantsMode", &granite::IntegratorSettings::ConstantsMode)
        .def_readwrite("MaximumSimulationTime",
                       &granite::IntegratorSettings::MaximumSimulationTime);

    pybind11::class_<granite::TrajectoryIntegrator>(m, "TrajectoryIntegrator")
        .def(pybind11::init<granite::IntegratorSettings &, granite::DataLoader &,
                            granite::TrajectorySet &>())
        .def("compute", &granite::TrajectoryIntegrator::compute);

    pybind11::class_<granite::DataLoader, granite::PyDataLoader>(m, "DataLoader")
        .def(pybind11::init<>())
        .def("step", &granite::DataLoader::step)
        .def("windfield", &granite::DataLoader::windfield)
        .def("uplift", &granite::DataLoader::uplift)
        .def("constants", &granite::DataLoader::constants)
        .def("additionalVolumes", &granite::DataLoader::additionalVolumes);

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