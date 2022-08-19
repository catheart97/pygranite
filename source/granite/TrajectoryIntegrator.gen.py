#!/usr/bin/env python3

"""
This script generates the kernel call hierachy. 
This is done to reduce code duplication and performance.
"""

import os
import glob
import sys

INCLUDES = """
# define PY_ARRAY_UNIQUE_SYMBOL MAGSENSE_ARRAY_API
# define NO_IMPORT_ARRAY

# include <cmath>
# include <cuda.h>

# include "granite/IntegratorSettings.hpp"
# include "granite/SimulationData.hpp"
# include "granite/TrajectoryIntegrator.hpp"

# define MY_USE_CUDA_TYPES
# include "my/math/LinearAlgebra.hpp"
# include "my/util/Util.cuh"

# define KERNEL_3D(                                                                           \\
            __integrator,                                                                     \\
            __curvature_mode,                                                                 \\
            __abort_mode,                                                                     \\
            __uplift_mode,                                                                    \\
            __use_interpolate,                                                                \\
            __use_topography,                                                                 \\
            __use_reverse_computation,                                                        \\
            __comp_additional,                                                                \\
            __comp_additional_volume                                                          \\
            )                                                                                 \\
        MY_VLOG("Kernel selected.");                                                          \\
        compute_<                                                                             \\
            my::math::Vec3,                                                                   \\
            granite::Space::Space3D,                                                          \\
            __integrator,                                                                     \\
            granite::BorderMode::Block,                                                       \\
            __curvature_mode,                                                                 \\
            __abort_mode,                                                                     \\
            __uplift_mode,                                                                    \\
            __use_interpolate,                                                                \\
            __use_topography,                                                                 \\
            __use_reverse_computation,                                                        \\
            false,                                                                            \\
            __comp_additional,                                                                \\
            __comp_additional_volume                                                          \\
        >(_set->_trajectories3d);                                                             \\
        return std::move(_set);

# define KERNEL_2D(                                                                           \\
            __integrator,                                                                     \\
            __border_mode,                                                                    \\
            __curvature_mode,                                                                 \\
            __abort_mode,                                                                     \\
            __use_interpolate,                                                                \\
            __use_reverse_computation,                                                        \\
            __use_sphere_coordinates,                                                         \\
            __comp_additional,                                                                \\
            __comp_additional_volume                                                          \\
            )                                                                                 \\
        MY_VLOG("Kernel selected.");                                                          \\
        compute_<                                                                             \\
            my::math::Vec2,                                                                   \\
            Space::Space2D,                                                                   \\
            __integrator,                                                                     \\
            __border_mode,                                                                    \\
            __curvature_mode,                                                                 \\
            __abort_mode,                                                                     \\
            granite::UpLiftMode::Off,                                                         \\
            __use_interpolate,                                                                \\
            false,                                                                            \\
            __use_reverse_computation,                                                        \\
            __use_sphere_coordinates,                                                         \\
            __comp_additional,                                                                \\
            __comp_additional_volume                                                          \\
        >(_set->_trajectories2d);                                                             \\
        return std::move(_set);
"""


class KernelArgument:
    def __init__(self, cases, comparison):
        self._cases = cases
        self._comparison = comparison

    def __eq__(self, case):
        if case in self._cases:
            return f"""{self._comparison} == {case}"""
        raise Exception("Invalid case supplied!")


class KernelCall:
    def __init__(self, function_name, args):
        self.count = 0
        self._calls = []
        self._function_name = function_name
        self._marco_args = args

    def stat(self):
        return f"-- {self._count} '{self._function_name}' variants generated."

    def _flatten(self, macro_args, args=[]):
        if (len(macro_args) > 0):
            result = ""
            macro_arg = macro_args[0]
            cases = macro_arg._cases
            for i in range(len(cases)):
                case = cases[i]
                prefix = "if" if i == 0 else "else if"
                content = self._flatten(macro_args[1:], [case] + args)
                result += f"""{prefix} ({macro_arg == case}) {"{"} {content} {"}"}"""
            return result
        else:
            id = self._count
            self._count += 1
            self._calls.append(self._function_name +
                               "(" + ",".join(reversed(args)) + ")")
            return f"return compute_{self._function_name}_{id}();"

    def compile_files(self):
        result = []
        PER_FILE = 256

        counter = 0
        code = ""
        for i, call in enumerate(self._calls):
            fn = f"TrajectoryIntegratorCompute_{self._function_name}_{i}.cu"

            if (counter == 0):
                code = INCLUDES

            code += f"""
std::unique_ptr<granite::TrajectorySet> granite::TrajectoryIntegrator::compute_{self._function_name}_{i}()
{"{"}
    using namespace my::math;
    {call}
{"}"}
"""
            if (counter == PER_FILE - 1 or i == len(self._calls) - 1):
                result.append([fn, code])
                counter = 0
            else:
                counter += 1

        return result

    def genMacro(self):
        code = ""
        for i, call in enumerate(self._calls):
            code += f"std::unique_ptr<granite::TrajectorySet> compute_{self._function_name}_{i}();"
            if i != len(self._calls) - 1:
                code += "\\\n"
        return code

    def __str__(self):
        self._count = 0
        macro_args = self._marco_args
        return self._flatten(macro_args)


def shorten(call_args):
    for arg in call_args:
        arg._cases = arg._cases[:1]


if __name__ == "__main__":

    # Configuration !
    SHORT_VERSION = bool(int(sys.argv[1]))

    args_3D = [
        KernelArgument([
            "granite::Integrator::ClassicRungeKutta",
            "granite::Integrator::ExplicitEuler"
        ], "_settings.Integrator"),
        KernelArgument([
            "granite::CurvatureMode::Off",
            "granite::CurvatureMode::TotalCurvature",
            "granite::CurvatureMode::FastTotalCurvature",
            "granite::CurvatureMode::IndividualAndTotalCurvature"
        ], "_settings.CurvatureMode"),
        KernelArgument([
            "granite::AbortMode::Time",
            "granite::AbortMode::Length",
            "granite::AbortMode::FitLength"
        ], "_settings.AbortMode"),
        KernelArgument([
            "granite::UpLiftMode::Off",
            "granite::UpLiftMode::Constant",
            "granite::UpLiftMode::Dynamic"
        ], "_settings.UpLiftMode"),
        KernelArgument([
            "false",
            "true"
        ], "_settings.InterpolateWindfields"),
        KernelArgument([
            "false",
            "true"
        ], "_topography_texture.Initialized"),
        KernelArgument([
            "false",
            "true"
        ], "_settings.Reverse"),
        KernelArgument([
            "false",
            "true"
        ], "(_settings.AdditionalCompute.size() > 0)"),
        KernelArgument([
            "false",
            "true"
        ], "(_settings.AdditionalVolumes.size() > 0)")
    ]

    if SHORT_VERSION:
        shorten(args_3D)

    call_3D = KernelCall("KERNEL_3D", args_3D)

    args_2D = [
        KernelArgument([
            "granite::Integrator::ClassicRungeKutta",
            "granite::Integrator::ExplicitEuler"
        ], "_settings.Integrator"),
        KernelArgument([
            "granite::BorderMode::Block",
            "granite::BorderMode::LoopX",
            "granite::BorderMode::LoopY",
            "granite::BorderMode::LoopXY"
        ], "_settings.BorderMode"),
        KernelArgument([
            "granite::CurvatureMode::Off",
            "granite::CurvatureMode::FastTotalCurvature"
        ], "_settings.CurvatureMode"),
        KernelArgument([
            "granite::AbortMode::Time",
            "granite::AbortMode::Length",
            "granite::AbortMode::FitLength"
        ], "_settings.AbortMode"),
        KernelArgument([
            "false", 
            "true"
        ], "_settings.InterpolateWindfields"),
        KernelArgument([
            "false", 
            "true"
        ], "_settings.Reverse"),
        KernelArgument([
            "false", 
            "true"
        ], "(_settings.SphereRadius > 0.f)"),
        KernelArgument([
            "false", 
            "true"
        ], "(_settings.AdditionalCompute.size() > 0)"),
        KernelArgument([
            "false", 
            "true"
        ], "(_settings.AdditionalVolumes.size() > 0)")
    ]

    if SHORT_VERSION:
        shorten(args_2D)

    call_2D = KernelCall("KERNEL_2D", args_2D)

    code = INCLUDES + f"""
std::unique_ptr<granite::TrajectorySet> granite::TrajectoryIntegrator::compute3D()
{"{"}

    {call_3D}

    MY_USER_ERROR(
        "No kernel could be picked for your settings configuration.");

    return std::move(_set);

{"}"}

std::unique_ptr<granite::TrajectorySet> granite::TrajectoryIntegrator::compute2D()
{"{"}

    {call_2D}

    MY_USER_ERROR(
        "No kernel could be picked for your settings configuration.");

    return std::move(_set);

{"}"}
"""

    root = os.path.abspath(os.path.dirname(__file__))
    # clear old files.
    files = glob.glob(root + '/TrajectoryIntegratorCompute/*.cu')
    for f in files: 
        try: 
            os.remove(f)
        except:
            print("-- ## ERROR: Could not delete file:", f)

    # write new files
    with open(root + "/TrajectoryIntegratorCompute.cpp", "w+") as f:
        f.write(code)

    for c in call_2D.compile_files():
        with open(root + "/TrajectoryIntegratorCompute/" + c[0], "w+") as f:
            f.write(c[1])

    for c in call_3D.compile_files():
        with open(root + "/TrajectoryIntegratorCompute/" + c[0], "w+") as f:
            f.write(c[1])

    with open(root + "/TrajectoryIntegratorKernelSelection.hpp", "w+") as f:
        f.write("#pragma once\n#define PYGRANITE_KERNEL_SELECTION()\\\n" +
                call_2D.genMacro() + "\\\n" + call_3D.genMacro())

    print(call_3D.stat())
    print(call_2D.stat())
