import os, sys
import re
import sysconfig
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

DF_DEBUG = False
if '--debug' in sys.argv:
    sys.argv.remove('--debug')
    DF_DEBUG = True

DF_USE_CUDA = 'OFF'
if '--use-cuda' in sys.argv:
    sys.argv.remove('--use-cuda')
    DF_USE_CUDA = 'ON'

DF_USE_ISPC = 'OFF'
if '--use-ispc' in sys.argv:
    sys.argv.remove('--use-ispc')
    DF_USE_ISPC = 'ON'

class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir=''):
        Extension.__init__(self, name, sources=[])
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError('Cannot find CMake executable')

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = 'Debug' if DF_DEBUG else 'Release'
        build_args = ['--config', cfg]

        cmake_args = [
            '-DDF_BUILD_PYTHON_WRAPPER=ON',
            '-DDF_BUILD_TESTS=OFF',
            '-DF_BUILD_WITH_DEBUG_INFO=%s' % ('ON' if cfg == 'Debug' else 'OFF'),
            '-DCMAKE_BUILD_TYPE=%s' % cfg,
            '-DDF_USE_CUDA=%s' % DF_USE_CUDA,
            '-DDF_USE_ISPC=%s' % DF_USE_ISPC,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
        ]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.cmake_lists_dir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='pydeform',
    version='0.0.1',
    author='Simon Ekström',
    author_email='',
    description='',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['numpy', 'PyYAML'],
    packages=['pydeform'],
    ext_modules=[CMakeExtension('_pydeform', '.')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
)

