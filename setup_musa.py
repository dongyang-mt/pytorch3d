#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import runpy
import sys
import warnings
from typing import List, Optional

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CppExtension, CUDA_HOME, CUDAExtension


def get_existing_ccbin(nvcc_args: List[str]) -> Optional[str]:
    """
    Given a list of nvcc arguments, return the compiler if specified.

    Note from CUDA doc: Single value options and list options must have
    arguments, which must follow the name of the option itself by either
    one of more spaces or an equals character.
    """
    last_arg = None
    for arg in reversed(nvcc_args):
        if arg == "-ccbin":
            return last_arg
        if arg.startswith("-ccbin="):
            return arg[7:]
        last_arg = arg
    return None


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
    # force_musa = True
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
        nvcc_args = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        if os.name != "nt":
            nvcc_args.append("-std=c++14")
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
        nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags_env != "":
            nvcc_args.extend(nvcc_flags_env.split(" "))

        # This is needed for pytorch 1.6 and earlier. See e.g.
        # https://github.com/facebookresearch/pytorch3d/issues/436
        # It is harmless after https://github.com/pytorch/pytorch/pull/47404 .
        # But it can be problematic in torch 1.7.0 and 1.7.1
        if torch.__version__[:4] != "1.7.":
            CC = os.environ.get("CC", None)
            if CC is not None:
                existing_CC = get_existing_ccbin(nvcc_args)
                if existing_CC is None:
                    CC_arg = "-ccbin={}".format(CC)
                    nvcc_args.append(CC_arg)
                elif existing_CC != CC:
                    msg = f"Inconsistent ccbins: {CC} and {existing_CC}"
                    raise ValueError(msg)

        extra_compile_args["nvcc"] = nvcc_args
    elif force_musa:
        print("== build for MUSA ==")
        # sources += source_musa
        # define_macros += [("WITH_MUSA", None)]
        # op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp') + \
        #     glob.glob('./mmcv/ops/csrc/pytorch/cpu/*.cpp') + \
        #     glob.glob('./mmcv/ops/csrc/pytorch/musa/*.cpp')
        # include_dirs.append(os.path.abspath('./mmcv/ops/csrc/pytorch'))
        # include_dirs.append(os.path.abspath('./'))
        include_dirs.append(os.path.abspath('/usr/local/musa/include'))
        # include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common'))
        # include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common/musa'))
        include_dirs.append(os.path.abspath('/home/torch_musa'))
        # include_dirs.append(os.path.abspath('/home/torch_musa/torch_musa/csrc/core'))
        # include_dirs.append(os.path.abspath('/home/torch_musa/torch_musa/csrc/aten/musa'))
        # include_dirs.append(os.path.abspath('/home/torch_musa/torch_musa/csrc/aten/mudnn'))
        # include_dirs.append(os.path.abspath('/home/torch_musa/torch_musa/csrc/aten/ops'))
        # include_dirs.append(os.path.abspath('/home/torch_musa/torch_musa/csrc/aten/ops/musa'))
        # include_dirs.append(os.path.abspath('/home/torch_musa/torch_musa/csrc/aten/utils'))
        # include_dirs.append(os.path.abspath('/home/torch_musa/torch_musa/csrc/utils'))
        op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp') + \
            glob.glob('./mmcv/ops/csrc/pytorch/cpu/*.cpp')
        include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common'))
        try:
            import sys
            import sysconfig
            sys.path.append(os.path.join("/home/pytorch", "tools"))
            from setup_helpers.env import build_type, check_negative_env_flag
            from tools.setup_helper.cmake_manager import CMakeManager

            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            RERUN_CMAKE = False

            def get_pytorch_install_path():
                try:
                    import torch

                    pytorch_install_root = os.path.dirname(os.path.abspath(torch.__file__))
                except Exception:
                    raise RuntimeError("Building error: import torch failed when building!")
                return pytorch_install_root


            def build_musa_lib():
                # generate code for CUDA porting
                build_dir = "build"
                gen_porting_dir = "generated_cuda_compatible"
                cuda_compatiable_path = os.path.join(BASE_DIR, build_dir, gen_porting_dir)
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

            build_musa_lib()
            extra_link_args = []
            extra_compile_args = [
                "-std=c++14",
                "-Wall", 
                "-Wextra",
                "-fno-strict-aliasing",
                "-fstack-protector-all",
            ]

            # if build_type.is_debug():
            extra_compile_args += ["-O0", "-ggdb"]
            extra_link_args += ["-O0", "-ggdb"]

            if build_type.is_rel_with_deb_info():
                extra_compile_args += ["-g"]
                extra_link_args += ["-g"]

            use_asan = os.getenv("USE_ASAN", default="").upper() in [
                "ON",
                "1",
                "YES",
                "TRUE",
                "Y",
            ]

            if use_asan:
                extra_compile_args += ["-fsanitize=address"]
                extra_link_args += ["-fsanitize=address"]

            extra_link_args=extra_link_args + ["-Wl,-rpath,$ORIGIN/lib"]
            library_dirs=["/home/mmcv/torch_musa/lib"]
            libraries=["mmcv_musa"]
            extension = CppExtension
        except ImportError:
            raise
    else:
        print("== build for CPU ==")

    print("== start building ==")
    sources = [os.path.join(extensions_dir, s) for s in sources]

    ext_modules = [
        extension(
            "pytorch3d._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


# Retrieve __version__ from the package.
__version__ = runpy.run_path("pytorch3d/__init__.py")["__version__"]


if os.getenv("PYTORCH3D_NO_NINJA", "0") == "1":

    class BuildExtension(torch.utils.cpp_extension.BuildExtension):
        def __init__(self, *args, **kwargs):
            super().__init__(use_ninja=False, *args, **kwargs)

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
    cmdclass={"build_ext": BuildExtension},
    package_data={
        "": ["*.json"],
    },
)