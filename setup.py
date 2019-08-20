import os
import sys
import platform
import subprocess

from pprint import pprint

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

with open("README.md", encoding="utf-8") as f:
    readme = f.read()

# Parse command line flags
flags = {k: 'OFF' for k in ['--debug', '--use-cuda', '--use-ispc', '--use-itk']}
for flag in flags.keys():
    if flag in sys.argv:
        flags[flag] = 'ON'
        sys.argv.remove(flag)

# Command line flags forwarded to CMake
cmake_cmd_args = []
for f in sys.argv:
    if f.startswith('-D'):
        cmake_cmd_args.append(f)

for f in cmake_cmd_args:
    sys.argv.remove(f)


class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir=''):
        Extension.__init__(self, name, sources=[])
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


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
        cfg = 'Debug' if flags['--debug'] == 'ON' else 'Release'
        build_args = ['--config', cfg]

        cmake_args = [
            '-DDF_BUILD_PYTHON_WRAPPER=ON',
            '-DDF_BUILD_EXECUTABLE=OFF',
            '-DDF_BUILD_UTILS=OFF',
            '-DDF_BUILD_TESTS=OFF',
            '-DDF_BUILD_WITH_DEBUG_INFO=%s' % ('ON' if cfg == 'Debug' else 'OFF'),
            '-DCMAKE_BUILD_TYPE=%s' % cfg,
            '-DDF_USE_CUDA=%s' % flags['--use-cuda'],
            '-DDF_USE_ISPC=%s' % flags['--use-ispc'],
            '-DDF_ITK_BRIDGE=%s' % flags['--use-itk'],
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
        ]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]

        cmake_args += cmake_cmd_args
        pprint(cmake_args)

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.cmake_lists_dir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(
    name='pydeform',
    version='0.4',
    author='Simon Ekström',
    author_email='',
    description='',
    long_description=readme,
    long_description_content_type='text/markdown',
    install_requires=['numpy', 'PyYAML'],
    packages=['pydeform'],
    ext_modules=[CMakeExtension('_pydeform', '.')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
)

