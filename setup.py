import os
from pathlib import Path
import sys
import platform
import subprocess

from pprint import pprint

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError('Cannot find CMake executable')

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"
        build_args = ['--config', cfg]

        cmake_args = [
            '-DDF_BUILD_PYTHON_WRAPPER=ON',
            '-DDF_BUILD_EXECUTABLE=OFF',
            '-DDF_BUILD_TESTS=OFF',
            '-DDF_BUILD_WITH_DEBUG_INFO=%s' % ('ON' if cfg == 'Debug' else 'OFF'),
            '-DCMAKE_BUILD_TYPE=%s' % cfg,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
        ]

        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]

        pprint(cmake_args)

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(['cmake', ext.sourcedir, *cmake_args], cwd=build_temp, check=True)
        subprocess.run(['cmake', '--build', '.', *build_args], cwd=build_temp, check=True)


setup(
    packages=find_packages(),
    ext_modules=[CMakeExtension('_pydeform', '.')],
    cmdclass={'build_ext': CMakeBuild},
)

