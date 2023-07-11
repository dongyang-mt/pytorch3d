import glob
import os
import sys
import warnings
import setuptools
import distutils
import torch
import copy
import shlex
import re

from torch.torch_version import TorchVersion
from typing import Dict, List, Optional, Union, Tuple

IS_MUSA_EXTENSION = True
TORCH_MUSA_BASE_DIR = os.path.abspath('/home/torch_musa')
TORCH_MUSA_LIB_PATH = "/opt/conda/envs/py38/lib/python3.8/site-packages/torch_musa-2.0.0-py3.8-linux-x86_64.egg/torch_musa/lib"


def find_in_path(name, path):
    """Find a file in a search path"""
    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
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
        mcc = os.path.join(home, 'bin', 'mcc')
    else:
        # Otherwise, search the PATH for NVCC
        mcc = find_in_path('mcc', os.environ['PATH'])
        if mcc is None:
            raise EnvironmentError('The mcc binary could not be '
                'located in your $PATH. Either add it to your path, '
                'or set $MUSA_HOME')
        home = os.path.dirname(os.path.dirname(mcc))

    musaconfig = {'home': home, 'mcc': mcc,
                  'include': os.path.join(home, 'include'),
                  'lib': os.path.join(home, 'lib')}
    for k, v in iter(musaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The MUSA %s path could not be '
                                   'located in %s' % (k, v))
    return musaconfig

# reference https://github.com/dongyang-mt/pytorch3d/blob/dev-musa-porting/setup.py
# TODO(dong.yang): need more comment for MUSAExtension
def MUSAExtension(name, sources, *args, **kwargs):
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += torch.utils.cpp_extension.library_paths(cuda=False)
    # library_dirs.append(os.path.join(BASE_DIR, "torch_musa/lib"))
    library_dirs.append(TORCH_MUSA_LIB_PATH)
    library_dirs.append(MUSA['lib'])

    libraries = kwargs.get('libraries', [])
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
    include_dirs.append(TORCH_MUSA_BASE_DIR)
    include_dirs.append(os.path.abspath('/usr/local/musa/include'))
    build_dir = "build"
    gen_porting_dir = "generated_cuda_compatible"
    cuda_compatiable_include_path = os.path.join(TORCH_MUSA_BASE_DIR, build_dir, gen_porting_dir, "include")
    include_dirs.append(os.path.abspath(cuda_compatiable_include_path))
    include_dirs.append(os.path.join(TORCH_MUSA_BASE_DIR, build_dir, gen_porting_dir, "include/torch/csrc/api/include"))

    extra_link_args = kwargs.get('extra_link_args', [])
    extra_link_args += ["-O2", "-ggdb"]

    extra_compile_args = kwargs.get('extra_compile_args', {})
    if 'mcc' not in extra_compile_args:
        extra_compile_args['mcc'] = []
    extra_compile_args['mcc'] += [
        "-std=c++14",
        "-Wall", 
        "-Wextra",
        "-fno-strict-aliasing",
        "-fPIC"
        ]
    extra_compile_args['mcc'] += ["-O2", "-ggdb"]

    # if build_type.is_debug():
    #     extra_compile_args['mcc'] += ["-O0", "-ggdb"]
    #     extra_link_args += ["-O0", "-ggdb"]

    # if build_type.is_rel_with_deb_info():
    #     extra_compile_args['mcc'] += ["-g"]
    #     extra_link_args += ["-g"]

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

    # include_dirs += torch.utils.cpp_extension.include_paths(cuda=False)
    kwargs['include_dirs'] = include_dirs
    kwargs['library_dirs'] = library_dirs
    kwargs['runtime_library_dirs'] = library_dirs

    kwargs['language'] = 'c++'
    kwargs['libraries'] = libraries
    kwargs['extra_link_args'] = extra_link_args
    kwargs['extra_compile_args'] = extra_compile_args

    return setuptools.Extension(name, sources, *args, **kwargs)


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
class musa_build_ext(setuptools.command.build_ext.build_ext):
    def build_extensions(self):
        customize_compiler_for_mcc(self.compiler)
        setuptools.command.build_ext.build_ext.build_extensions(self)

MUSA = locate_musa()
# TODO(dong.yang): I dont know why need ext which used by customize_compiler_for_mcc
ext = distutils.extension.Extension('gpuadder',
        sources = [],
        language = 'c++',
        extra_compile_args= {
            'gcc': [],
            'mcc': []
            }
        )


class MUSA_BuildExtension(setuptools.command.build_ext.build_ext):
    @classmethod
    def with_options(cls, **options):
        r'''
        Returns a subclass with alternative constructor that extends any original keyword
        arguments to the original constructor with the given options.
        '''
        class cls_with_options(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                kwargs.update(options)
                super().__init__(*args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

        self.use_ninja = kwargs.get('use_ninja', True)
        if self.use_ninja:
            # Test if we can use ninja. Fallback otherwise.
            msg = ('Attempted to use ninja as the BuildExtension backend but '
                   '{}. Falling back to using the slow distutils backend.')
            if not torch.utils.cpp_extension.is_ninja_available():
                warnings.warn(msg.format('we could not find ninja.'))
                self.use_ninja = False

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.use_ninja:
            self.force = True

    def build_extensions(self) -> None:
        compiler_name, compiler_version = self._check_abi()

        cuda_ext = False
        extension_iter = iter(self.extensions)
        extension = next(extension_iter, None)
        while not cuda_ext and extension:
            for source in extension.sources:
                _, ext = os.path.splitext(source)
                if ext == '.cu':
                    cuda_ext = True
                    break
            extension = next(extension_iter, None)

        if cuda_ext and not torch.utils.cpp_extension.IS_HIP_EXTENSION:
            torch.utils.cpp_extension._check_cuda_version(compiler_name, compiler_version)

        for extension in self.extensions:
            # Ensure at least an empty list of flags for 'cxx' and 'nvcc' when
            # extra_compile_args is a dict. Otherwise, default torch flags do
            # not get passed. Necessary when only one of 'cxx' and 'nvcc' is
            # passed to extra_compile_args in CUDAExtension, i.e.
            #   CUDAExtension(..., extra_compile_args={'cxx': [...]})
            # or
            #   CUDAExtension(..., extra_compile_args={'nvcc': [...]})
            if isinstance(extension.extra_compile_args, dict):
                for ext in ['cxx', 'nvcc']:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []

            self._add_compile_flag(extension, '-DTORCH_API_INCLUDE_EXTENSION_H')
            # See note [Pybind11 ABI constants]
            for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
                val = getattr(torch._C, f"_PYBIND11_{name}")
                if val is not None and not torch.utils.cpp_extension.IS_WINDOWS:
                    self._add_compile_flag(extension, f'-DPYBIND11_{name}="{val}"')
            self._define_torch_extension_name(extension)
            self._add_gnu_cpp_abi_flag(extension)

            if 'nvcc_dlink' in extension.extra_compile_args:
                assert self.use_ninja, f"With dlink=True, ninja is required to build cuda extension {extension.name}."

        # Register .cu, .cuh and .hip as valid source extensions.
        self.compiler.src_extensions += ['.cu', '.cuh', '.hip']
        # Save the original _compile method for later.
        if self.compiler.compiler_type == 'msvc':
            self.compiler._cpp_extensions += ['.cu', '.cuh']
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile

        def append_std17_if_no_std_present(cflags) -> None:
            # NVCC does not allow multiple -std to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            cpp_format_prefix = '/{}:' if self.compiler.compiler_type == 'msvc' else '-{}='
            cpp_flag_prefix = cpp_format_prefix.format('std')
            cpp_flag = cpp_flag_prefix + 'c++17'
            if not any(flag.startswith(cpp_flag_prefix) for flag in cflags):
                cflags.append(cpp_flag)

        def unix_cuda_flags(cflags):
            cflags = (torch.utils.cpp_extension.COMMON_NVCC_FLAGS +
                      ['--compiler-options', "'-fPIC'"] +
                      cflags + torch.utils.cpp_extension._get_cuda_arch_flags(cflags))

            # NVCC does not allow multiple -ccbin/--compiler-bindir to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            _ccbin = os.getenv("CC")
            if (
                _ccbin is not None
                and not any([flag.startswith('-ccbin') or flag.startswith('--compiler-bindir') for flag in cflags])
            ):
                cflags.extend(['-ccbin', _ccbin])

            return cflags

        def convert_to_absolute_paths_inplace(paths):
            # Helper function. See Note [Absolute include_dirs]
            if paths is not None:
                for i in range(len(paths)):
                    if not os.path.isabs(paths[i]):
                        paths[i] = os.path.abspath(paths[i])

        def unix_wrap_single_compile(obj, src, ext, cc_args, extra_postargs, pp_opts) -> None:
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if torch.utils.cpp_extension._is_cuda_file(src):
                    nvcc = [torch.utils.cpp_extension._join_rocm_home('bin', 'hipcc') if torch.utils.cpp_extension.IS_HIP_EXTENSION else torch.utils.cpp_extension._join_cuda_home('bin', 'nvcc')]
                    self.compiler.set_executable('compiler_so', nvcc)
                    if isinstance(cflags, dict):
                        cflags = cflags['nvcc']
                    if torch.utils.cpp_extension.IS_HIP_EXTENSION:
                        cflags = torch.utils.cpp_extension.COMMON_HIPCC_FLAGS + cflags + torch.utils.cpp_extension._get_rocm_arch_flags(cflags)
                    else:
                        cflags = unix_cuda_flags(cflags)
                elif isinstance(cflags, dict):
                    cflags = cflags['cxx']
                if torch.utils.cpp_extension.IS_HIP_EXTENSION:
                    cflags = torch.utils.cpp_extension.COMMON_HIP_FLAGS + cflags
                append_std17_if_no_std_present(cflags)

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable('compiler_so', original_compiler)

        def unix_wrap_ninja_compile(sources,
                                    output_dir=None,
                                    macros=None,
                                    include_dirs=None,
                                    debug=0,
                                    extra_preargs=None,
                                    extra_postargs=None,
                                    depends=None):
            r"""Compiles sources by outputting a ninja file and running it."""
            # NB: I copied some lines from self.compiler (which is an instance
            # of distutils.UnixCCompiler). See the following link.
            # https://github.com/python/cpython/blob/f03a8f8d5001963ad5b5b28dbd95497e9cc15596/Lib/distutils/ccompiler.py#L564-L567
            # This can be fragile, but a lot of other repos also do this
            # (see https://github.com/search?q=_setup_compile&type=Code)
            # so it is probably OK; we'll also get CI signal if/when
            # we update our python version (which is when distutils can be
            # upgraded)

            # Use absolute path for output_dir so that the object file paths
            # (`objects`) get generated with absolute paths.
            output_dir = os.path.abspath(output_dir)

            # See Note [Absolute include_dirs]
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)

            _, objects, extra_postargs, pp_opts, _ = \
                self.compiler._setup_compile(output_dir, macros,
                                             include_dirs, sources,
                                             depends, extra_postargs)
            common_cflags = self.compiler._get_cc_args(pp_opts, debug, extra_preargs)
            extra_cc_cflags = self.compiler.compiler_so[1:]
            with_cuda = any(map(torch.utils.cpp_extension._is_cuda_file, sources))

            # extra_postargs can be either:
            # - a dict mapping cxx/nvcc to extra flags
            # - a list of extra flags.
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs['cxx']
            else:
                post_cflags = list(extra_postargs)
            if torch.utils.cpp_extension.IS_HIP_EXTENSION:
                post_cflags = torch.utils.cpp_extension.COMMON_HIP_FLAGS + post_cflags
            append_std17_if_no_std_present(post_cflags)

            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = common_cflags
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs['nvcc']
                else:
                    cuda_post_cflags = list(extra_postargs)
                if torch.utils.cpp_extension.IS_HIP_EXTENSION:
                    cuda_post_cflags = cuda_post_cflags + torch.utils.cpp_extension._get_rocm_arch_flags(cuda_post_cflags)
                    cuda_post_cflags = torch.utils.cpp_extension.COMMON_HIP_FLAGS + torch.utils.cpp_extension.COMMON_HIPCC_FLAGS + cuda_post_cflags
                else:
                    cuda_post_cflags = unix_cuda_flags(cuda_post_cflags)
                append_std17_if_no_std_present(cuda_post_cflags)
                cuda_cflags = [shlex.quote(f) for f in cuda_cflags]
                cuda_post_cflags = [shlex.quote(f) for f in cuda_post_cflags]

            if isinstance(extra_postargs, dict) and 'nvcc_dlink' in extra_postargs:
                cuda_dlink_post_cflags = unix_cuda_flags(extra_postargs['nvcc_dlink'])
            else:
                cuda_dlink_post_cflags = None
            torch.utils.cpp_extension._write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                post_cflags=[shlex.quote(f) for f in post_cflags],
                cuda_cflags=cuda_cflags,
                cuda_post_cflags=cuda_post_cflags,
                cuda_dlink_post_cflags=cuda_dlink_post_cflags,
                build_directory=output_dir,
                verbose=True,
                with_cuda=with_cuda)

            # Return *all* object filenames, not just the ones we just built.
            return objects

        def win_cuda_flags(cflags):
            return (torch.utils.cpp_extension.COMMON_NVCC_FLAGS +
                    cflags + torch.utils.cpp_extension._get_cuda_arch_flags(cflags))

        def win_wrap_single_compile(sources,
                                    output_dir=None,
                                    macros=None,
                                    include_dirs=None,
                                    debug=0,
                                    extra_preargs=None,
                                    extra_postargs=None,
                                    depends=None):

            self.cflags = copy.deepcopy(extra_postargs)
            extra_postargs = None

            def spawn(cmd):
                # Using regex to match src, obj and include files
                src_regex = re.compile('/T(p|c)(.*)')
                src_list = [
                    m.group(2) for m in (src_regex.match(elem) for elem in cmd)
                    if m
                ]

                obj_regex = re.compile('/Fo(.*)')
                obj_list = [
                    m.group(1) for m in (obj_regex.match(elem) for elem in cmd)
                    if m
                ]

                include_regex = re.compile(r'((\-|\/)I.*)')
                include_list = [
                    m.group(1)
                    for m in (include_regex.match(elem) for elem in cmd) if m
                ]

                if len(src_list) >= 1 and len(obj_list) >= 1:
                    src = src_list[0]
                    obj = obj_list[0]
                    if torch.utils.cpp_extension._is_cuda_file(src):
                        nvcc = torch.utils.cpp_extension._join_cuda_home('bin', 'nvcc')
                        if isinstance(self.cflags, dict):
                            cflags = self.cflags['nvcc']
                        elif isinstance(self.cflags, list):
                            cflags = self.cflags
                        else:
                            cflags = []

                        cflags = win_cuda_flags(cflags) + ['--use-local-env']
                        for flag in torch.utils.cpp_extension.COMMON_MSVC_FLAGS:
                            cflags = ['-Xcompiler', flag] + cflags
                        for ignore_warning in torch.utils.cpp_extension.MSVC_IGNORE_CUDAFE_WARNINGS:
                            cflags = ['-Xcudafe', '--diag_suppress=' + ignore_warning] + cflags
                        cmd = [nvcc, '-c', src, '-o', obj] + include_list + cflags
                    elif isinstance(self.cflags, dict):
                        cflags = torch.utils.cpp_extension.COMMON_MSVC_FLAGS + self.cflags['cxx']
                        cmd += cflags
                    elif isinstance(self.cflags, list):
                        cflags = torch.utils.cpp_extension.COMMON_MSVC_FLAGS + self.cflags
                        cmd += cflags

                return original_spawn(cmd)

            try:
                self.compiler.spawn = spawn
                return original_compile(sources, output_dir, macros,
                                        include_dirs, debug, extra_preargs,
                                        extra_postargs, depends)
            finally:
                self.compiler.spawn = original_spawn

        def win_wrap_ninja_compile(sources,
                                   output_dir=None,
                                   macros=None,
                                   include_dirs=None,
                                   debug=0,
                                   extra_preargs=None,
                                   extra_postargs=None,
                                   depends=None):

            if not self.compiler.initialized:
                self.compiler.initialize()
            output_dir = os.path.abspath(output_dir)

            # Note [Absolute include_dirs]
            # Convert relative path in self.compiler.include_dirs to absolute path if any,
            # For ninja build, the build location is not local, the build happens
            # in a in script created build folder, relative path lost their correctness.
            # To be consistent with jit extension, we allow user to enter relative include_dirs
            # in setuptools.setup, and we convert the relative path to absolute path here
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)

            _, objects, extra_postargs, pp_opts, _ = \
                self.compiler._setup_compile(output_dir, macros,
                                             include_dirs, sources,
                                             depends, extra_postargs)
            common_cflags = extra_preargs or []
            cflags = []
            if debug:
                cflags.extend(self.compiler.compile_options_debug)
            else:
                cflags.extend(self.compiler.compile_options)
            common_cflags.extend(torch.utils.cpp_extension.COMMON_MSVC_FLAGS)
            cflags = cflags + common_cflags + pp_opts
            with_cuda = any(map(torch.utils.cpp_extension._is_cuda_file, sources))

            # extra_postargs can be either:
            # - a dict mapping cxx/nvcc to extra flags
            # - a list of extra flags.
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs['cxx']
            else:
                post_cflags = list(extra_postargs)
            append_std17_if_no_std_present(post_cflags)

            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = ['--use-local-env']
                for common_cflag in common_cflags:
                    cuda_cflags.append('-Xcompiler')
                    cuda_cflags.append(common_cflag)
                for ignore_warning in torch.utils.cpp_extension.MSVC_IGNORE_CUDAFE_WARNINGS:
                    cuda_cflags.append('-Xcudafe')
                    cuda_cflags.append('--diag_suppress=' + ignore_warning)
                cuda_cflags.extend(pp_opts)
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs['nvcc']
                else:
                    cuda_post_cflags = list(extra_postargs)
                cuda_post_cflags = win_cuda_flags(cuda_post_cflags)

            cflags = torch.utils.cpp_extension._nt_quote_args(cflags)
            post_cflags = torch.utils.cpp_extension._nt_quote_args(post_cflags)
            if with_cuda:
                cuda_cflags = torch.utils.cpp_extension._nt_quote_args(cuda_cflags)
                cuda_post_cflags = torch.utils.cpp_extension._nt_quote_args(cuda_post_cflags)
            if isinstance(extra_postargs, dict) and 'nvcc_dlink' in extra_postargs:
                cuda_dlink_post_cflags = win_cuda_flags(extra_postargs['nvcc_dlink'])
            else:
                cuda_dlink_post_cflags = None

            torch.utils.cpp_extension._write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=cflags,
                post_cflags=post_cflags,
                cuda_cflags=cuda_cflags,
                cuda_post_cflags=cuda_post_cflags,
                cuda_dlink_post_cflags=cuda_dlink_post_cflags,
                build_directory=output_dir,
                verbose=True,
                with_cuda=with_cuda)

            # Return *all* object filenames, not just the ones we just built.
            return objects

        # Monkey-patch the _compile or compile method.
        # https://github.com/python/cpython/blob/dc0284ee8f7a270b6005467f26d8e5773d76e959/Lib/distutils/ccompiler.py#L511
        if self.compiler.compiler_type == 'msvc':
            if self.use_ninja:
                self.compiler.compile = win_wrap_ninja_compile
            else:
                self.compiler.compile = win_wrap_single_compile
        else:
            if self.use_ninja:
                self.compiler.compile = unix_wrap_ninja_compile
            else:
                self.compiler._compile = unix_wrap_single_compile

        setuptools.command.build_ext.build_ext.build_extensions(self)

    def get_ext_filename(self, ext_name):
        # Get the original shared library name. For Python 3, this name will be
        # suffixed with "<SOABI>.so", where <SOABI> will be something like
        # cpython-37m-x86_64-linux-gnu.
        ext_filename = super().get_ext_filename(ext_name)
        # If `no_python_abi_suffix` is `True`, we omit the Python 3 ABI
        # component. This makes building shared libraries with setuptools that
        # aren't Python modules nicer.
        if self.no_python_abi_suffix:
            # The parts will be e.g. ["my_extension", "cpython-37m-x86_64-linux-gnu", "so"].
            ext_filename_parts = ext_filename.split('.')
            # Omit the second to last element.
            without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
            ext_filename = '.'.join(without_abi)
        return ext_filename

    def _check_abi(self) -> Tuple[str, TorchVersion]:
        # On some platforms, like Windows, compiler_cxx is not available.
        if hasattr(self.compiler, 'compiler_cxx'):
            compiler = self.compiler.compiler_cxx[0]
        elif torch.utils.cpp_extension.IS_WINDOWS:
            compiler = os.environ.get('CXX', 'cl')
        else:
            compiler = os.environ.get('CXX', 'c++')
        _, version = torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version(compiler)
        # Warn user if VC env is activated but `DISTUILS_USE_SDK` is not set.
        if torch.utils.cpp_extension.IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' in os.environ and 'DISTUTILS_USE_SDK' not in os.environ:
            msg = ('It seems that the VC environment is activated but DISTUTILS_USE_SDK is not set.'
                   'This may lead to multiple activations of the VC env.'
                   'Please set `DISTUTILS_USE_SDK=1` and try again.')
            raise UserWarning(msg)
        return compiler, version

    def _add_compile_flag(self, extension, flag):
        extension.extra_compile_args = copy.deepcopy(extension.extra_compile_args)
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(flag)
        else:
            extension.extra_compile_args.append(flag)

    def _define_torch_extension_name(self, extension):
        # pybind11 doesn't support dots in the names
        # so in order to support extensions in the packages
        # like torch._C, we take the last part of the string
        # as the library name
        names = extension.name.split('.')
        name = names[-1]
        define = f'-DTORCH_EXTENSION_NAME={name}'
        self._add_compile_flag(extension, define)

    def _add_gnu_cpp_abi_flag(self, extension):
        # use the same CXX ABI as what PyTorch was compiled with
        self._add_compile_flag(extension, '-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)))
