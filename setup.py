#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
from os.path import join as pjoin
import runpy
import sys
import warnings
from typing import List, Optional
import setuptools
import distutils
# from Cython.Distutils import build_ext

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CppExtension, CUDA_HOME, CUDAExtension
from torch.torch_version import TorchVersion
from typing import Dict, List, Optional, Union, Tuple

BASE_DIR = os.path.abspath('/home/torch_musa')
IS_MUSA_EXTENSION = True
os.putenv("FORCE_MUSA", "1")
os.putenv("FORCE_CUDA", "0")

def get_existing_ccbin(mcc_args: List[str]) -> Optional[str]:
    """
    Given a list of mcc arguments, return the compiler if specified.

    Note from CUDA doc: Single value options and list options must have
    arguments, which must follow the name of the option itself by either
    one of more spaces or an equals character.
    """
    last_arg = None
    for arg in reversed(mcc_args):
        if arg == "-ccbin":
            return last_arg
        if arg.startswith("-ccbin="):
            return arg[7:]
        last_arg = arg
    return None

_HERE = os.path.abspath(__file__)
_TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))
TORCH_MUSA_LIB_PATH = "/opt/conda/envs/py38/lib/python3.8/site-packages/torch_musa-2.0.0-py3.8-linux-x86_64.egg/torch_musa/lib"

def MUSAExtension(name, sources, *args, **kwargs):
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += torch.utils.cpp_extension.library_paths(cuda=False)
    # library_dirs.append(os.path.join(BASE_DIR, "torch_musa/lib"))
    library_dirs.append(TORCH_MUSA_LIB_PATH)
    library_dirs.append(MUSA['lib'])

    libraries = kwargs.get('libraries', [])
    # libraries.append('c10')
    # libraries.append('torch')
    # libraries.append('torch_cpu')
    # libraries.append('torch_python')
    if IS_MUSA_EXTENSION:
        libraries.append('musa_python')
        # libraries.append('torch_musa')
        libraries.append('mudnn')
        libraries.append('mublas')
        libraries.append('musart')
    runtime_library_dirs = kwargs.get('runtime_library_dirs', [])
    runtime_library_dirs.append(MUSA['lib'])

    include_dirs = kwargs.get('include_dirs', [])
    include_dirs.append(MUSA['include'])

    # include_dirs += torch.utils.cpp_extension.include_paths(cuda=False)
    kwargs['include_dirs'] = include_dirs
    kwargs['library_dirs'] = library_dirs
    kwargs['runtime_library_dirs'] = library_dirs

    kwargs['language'] = 'c++'
    kwargs['libraries'] = libraries

    return setuptools.Extension(name, sources, *args, **kwargs)


def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_musa():
    """Locate the MUSA environment on the system

    Returns a dict with keys 'home', 'mcc', 'include', and 'lib'
    and values giving the absolute path to each directory.

    Starts by looking for the MUSA_HOME env variable. If not found,
    everything is based on finding 'mcc' in the PATH.
    """

    # First check if the MUSA_HOME env variable is in use
    if 'MUSA_HOME' in os.environ:
        home = os.environ['MUSA_HOME']
        mcc = pjoin(home, 'bin', 'mcc')
    else:
        # Otherwise, search the PATH for NVCC
        mcc = find_in_path('mcc', os.environ['PATH'])
        if mcc is None:
            raise EnvironmentError('The mcc binary could not be '
                'located in your $PATH. Either add it to your path, '
                'or set $MUSA_HOME')
        home = os.path.dirname(os.path.dirname(mcc))

    musaconfig = {'home': home, 'mcc': mcc,
                  'include': pjoin(home, 'include'),
                  'lib': pjoin(home, 'lib')}
    for k, v in iter(musaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The MUSA %s path could not be '
                                   'located in %s' % (k, v))

    return musaconfig


def customize_compiler_for_mcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/mcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .mu
    self.src_extensions.append('.mu')

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.mu':
            # use the musa for .mu files
            self.set_executable('compiler_so', MUSA['mcc'])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs['mcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for musa
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile



# Run the customize_compiler
class custom_build_ext(setuptools.command.build_ext.build_ext):
    def build_extensions(self):
        customize_compiler_for_mcc(self.compiler)
        setuptools.command.build_ext.build_ext.build_extensions(self)



MUSA = locate_musa()

import numpy
# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


ext = distutils.extension.Extension('gpuadder',
        sources = [],
        library_dirs = [],
        libraries = [],
        language = 'c++',
        runtime_library_dirs = [],
        # This syntax is specific to this build system
        # we're only going to use certain compiler args with mcc
        # and not with gcc the implementation of this trick is in
        # customize_compiler()
        extra_compile_args= {
            'gcc': [],
            'mcc': [
                '-arch=sm_80', '--ptxas-options=-v', '-c',
                '--compiler-options', "'-fPIC'"
                ]
            },
            include_dirs = [numpy_include, MUSA['include'], 'src']
        )


def get_extensions():
    no_extension = os.getenv("PYTORCH3D_NO_EXTENSION", "0") == "1"
    if no_extension:
        msg = "SKIPPING EXTENSION BUILD. PYTORCH3D WILL NOT WORK!"
        print(msg, file=sys.stderr)
        warnings.warn(msg)
        return []

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "pytorch3d", "csrc")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"), recursive=True)
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True)
    source_musa = glob.glob(os.path.join(extensions_dir, "**", "*.mu"), recursive=True)
    extension = CppExtension

    extra_compile_args = {"gcc": ["-std=c++14", "-fPIC"]}
    define_macros = []
    include_dirs = [extensions_dir]

    force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
    force_musa = os.getenv("FORCE_MUSA", "0") == "1"
    force_no_cuda = os.getenv("PYTORCH3D_FORCE_NO_CUDA", "0") == "1"
    force_no_musa = os.getenv("PYTORCH3D_FORCE_NO_MUSA", "0") == "1"
    if (
        not force_no_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    ) or force_cuda:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        # Thrust is only used for its tuple objects.
        # With CUDA 11.0 we can't use the cudatoolkit's version of cub.
        # We take the risk that CUB and Thrust are incompatible, because
        # we aren't using parts of Thrust which actually use CUB.
        define_macros += [("THRUST_IGNORE_CUB_VERSION_CHECK", None)]
        cub_home = os.environ.get("CUB_HOME", None)
        mcc_args = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        if os.name != "nt":
            mcc_args.append("-std=c++14")
        if cub_home is None:
            prefix = os.environ.get("CONDA_PREFIX", None)
            if prefix is not None and os.path.isdir(prefix + "/include/cub"):
                cub_home = prefix + "/include"

        if cub_home is None:
            warnings.warn(
                "The environment variable `CUB_HOME` was not found. "
                "NVIDIA CUB is required for compilation and can be downloaded "
                "from `https://github.com/NVIDIA/cub/releases`. You can unpack "
                "it to a location of your choice and set the environment variable "
                "`CUB_HOME` to the folder containing the `CMakeListst.txt` file."
            )
        else:
            include_dirs.append(os.path.realpath(cub_home).replace("\\ ", " "))
        mcc_flags_env = os.getenv("NVCC_FLAGS", "")
        if mcc_flags_env != "":
            mcc_args.extend(mcc_flags_env.split(" "))

        # This is needed for pytorch 1.6 and earlier. See e.g.
        # https://github.com/facebookresearch/pytorch3d/issues/436
        # It is harmless after https://github.com/pytorch/pytorch/pull/47404 .
        # But it can be problematic in torch 1.7.0 and 1.7.1
        if torch.__version__[:4] != "1.7.":
            CC = os.environ.get("CC", None)
            if CC is not None:
                existing_CC = get_existing_ccbin(mcc_args)
                if existing_CC is None:
                    CC_arg = "-ccbin={}".format(CC)
                    mcc_args.append(CC_arg)
                elif existing_CC != CC:
                    msg = f"Inconsistent ccbins: {CC} and {existing_CC}"
                    raise ValueError(msg)

        extra_compile_args["mcc"] = mcc_args
    # elif os.getenv('FORCE_MUSA', '0') == '1':
    elif True:
            print("------------build MUSA source code")
            include_dirs.append(BASE_DIR)
            include_dirs.append(os.path.abspath('/usr/local/musa/include'))
            # sources += source_musa
            source_musa += sources
            sources = source_musa
            define_macros += [("WITH_MUSA", None)]

            if True:
                import sys
                import sysconfig
                sys.path.append(os.path.join("/home/pytorch", "tools"))
                from setup_helpers.env import build_type, check_negative_env_flag
                from tools.setup_helper.cmake_manager import CMakeManager

                RERUN_CMAKE = False
                build_dir = "build"
                gen_porting_dir = "generated_cuda_compatible"
                cuda_compatiable_include_path = os.path.join(BASE_DIR, build_dir, gen_porting_dir, "include")
                include_dirs.append(os.path.abspath(cuda_compatiable_include_path))
                include_dirs.append(os.path.join(BASE_DIR, build_dir, gen_porting_dir, "include/torch/csrc/api/include"))

                def get_pytorch_install_path():
                    import torch
                    pytorch_install_root = os.path.dirname(os.path.abspath(torch.__file__))
                    return pytorch_install_root

                def build_musa_lib():
                    # generate code for CUDA porting
                    # if not os.path.isdir(cuda_compatiable_path):
                    #     port_cuda(pytorch_root, get_pytorch_install_path(), cuda_compatiable_path)

                    cmake = CMakeManager(build_dir)
                    env = os.environ.copy()
                    env["GENERATED_PORTING_DIR"] = gen_porting_dir
                    build_test = not check_negative_env_flag("BUILD_TEST")
                    cmake_python_library = "{}/{}".format(
                        sysconfig.get_config_var("LIBDIR"), sysconfig.get_config_var("INSTSONAME")
                    )
                    cmake.generate("unknown", cmake_python_library, True, build_test, env, RERUN_CMAKE)
                    cmake.build(env)

                # build_musa_lib()
                extra_link_args = []
                # extra_compile_args = {'gcc': []}
                extra_compile_args['mcc'] = [
                    "-std=c++14",
                    "-Wall", 
                    "-Wextra",
                    "-fno-strict-aliasing",
                    "-fPIC"
                    ]

                # if build_type.is_debug():
                extra_compile_args['mcc'] += ["-O2", "-ggdb"]
                extra_link_args += ["-O2", "-ggdb"]

                if build_type.is_rel_with_deb_info():
                    extra_compile_args['mcc'] += ["-g"]
                    extra_link_args += ["-g"]

                use_asan = os.getenv("USE_ASAN", default="").upper() in [
                    "ON",
                    "1",
                    "YES",
                    "TRUE",
                    "Y",
                ]

                if use_asan:
                    extra_compile_args['mcc'] += ["-fsanitize=address"]
                    extra_link_args += ["-fsanitize=address"]
                extra_link_args += ["-Wl,-rpath,$ORIGIN/lib"]
                # extra_link_args += ["-L/home/mmcv/torch_musa/lib/"]
                library_dirs=["pytorch3d/"]
                libraries=[]
                extension = MUSAExtension

    sources = [os.path.join(extensions_dir, s) for s in sources]

    ext_modules = [
        extension(
            "pytorch3d._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


# Retrieve __version__ from the package.
__version__ = runpy.run_path("pytorch3d/__init__.py")["__version__"]
# from distutils.ccompiler import new_compiler
# compiler = new_compiler(compiler=None, verbose=True)

if os.getenv("PYTORCH3D_NO_NINJA", "0") == "1":

    class BuildExtension(torch.utils.cpp_extension.BuildExtension):
        def __init__(self, *args, **kwargs):
            super().__init__(use_ninja=False, *args, **kwargs)

elif os.getenv('FORCE_MUSA', '0') == '1':
    class BuildExtension(setuptools.command.build_ext.build_ext):
        def __init__(self, *args, **kwargs):
            super().__init__(use_ninja=False, *args, **kwargs)
            from distutils.ccompiler import new_compiler
            self.compiler = new_compiler()
            self.compiler.compiler_type == 'mcc'
            self.compiler._cpp_extensions += ['.mu', '.muh']

        def _check_abi(self) -> Tuple[str, TorchVersion]:
            compiler = os.environ.get('mcc', 'cl')
            version = TorchVersion('1.4.0')
            return compiler, version

else:
    BuildExtension = torch.utils.cpp_extension.BuildExtension

trainer = "pytorch3d.implicitron_trainer"

setup(
    name="pytorch3d",
    version=__version__,
    author="FAIR",
    url="https://github.com/facebookresearch/pytorch3d",
    description="PyTorch3D is FAIR's library of reusable components "
    "for deep Learning with 3D data.",
    packages=find_packages(
        exclude=("configs", "tests", "tests.*", "docs.*", "projects.*")
    )
    + [trainer],
    package_dir={trainer: "projects/implicitron_trainer"},
    install_requires=["fvcore", "iopath"],
    extras_require={
        "all": ["matplotlib", "tqdm>4.29.0", "imageio", "ipywidgets"],
        "dev": ["flake8", "usort"],
        "implicitron": [
            "hydra-core>=1.1",
            "visdom",
            "lpips",
            "tqdm>4.29.0",
            "matplotlib",
            "accelerate",
            "sqlalchemy>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            f"pytorch3d_implicitron_runner={trainer}.experiment:experiment",
            f"pytorch3d_implicitron_visualizer={trainer}.visualize_reconstruction:main",
        ]
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": custom_build_ext},
    package_data={
        "": ["*.json"],
    },
)
