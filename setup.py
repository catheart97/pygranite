from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

import os
import platform
import subprocess
import sys
import multiprocessing

from distutils.core import setup
from pathlib import Path


class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        build_directory = os.path.abspath(self.build_temp)

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + build_directory,
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg, '--parallel', str(multiprocessing.cpu_count())]
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg, '-G', 'Ninja']

        self.build_args = build_args

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # CMakeLists.txt is in the same directory as this setup.py file
        cmake_list_dir = os.path.abspath(os.path.dirname(__file__))
        print('-'*10, 'Running CMake prepare', '-'*40)
        subprocess.check_call(['cmake', cmake_list_dir] + cmake_args,
                                  cwd=self.build_temp, env=env)

        print('-'*10, 'Building extensions', '-'*40)
        cmake_cmd = ['cmake', '--build', '.'] + self.build_args
        print('*'*5, cmake_cmd)
        subprocess.check_call(cmake_cmd,
                              cwd=self.build_temp)

        # Move from build temp to final position
        for ext in self.extensions:
            self.move_output(ext, cfg)

    def move_output(self, ext, cfg):
        build_temp = Path(self.build_temp).resolve()
        dest_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        source_path = build_temp / self.get_ext_filename(ext.name)
        dest_directory = dest_path.parents[0]
        dest_directory.mkdir(parents=True, exist_ok=True)
        self.copy_file(source_path, dest_path)


setup(
    name="pygranite",
    author="Ronja Schnur",
    author_email="ronjaschnur@uni-mainz.de",
    description="pygranite is a library for fast trajectory computation of particles inside of " +
                "windfields using cuda hardware acceleration.",
    keywords="windfield trajectory trajectories gpu computation cuda meteorology atmospheric",
    version="1.5.0",
    url="https://github.com/catheart97/pygranite",
    packages=find_packages(),
    ext_modules=[
        CMakeExtension('pygranite')
    ],
    install_requires=[
        'numpy', 'netCDF4'
    ],
    cmdclass={
        'build_ext': CMakeBuild
    },
    platforms=[
        'nt',
        'posix'
    ],
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: Win32 (MS Windows)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Intended Audience :: Science/Research",
        'Programming Language :: Python :: 3'
    ],
    zip_safe=False
)
