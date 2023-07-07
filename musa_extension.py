import glob
import os
import sys
import warnings
import setuptools
import distutils
import torch


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
