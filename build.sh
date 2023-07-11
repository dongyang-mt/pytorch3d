export MAX_JOBS=64
ENABLE_COMPILE_FP64=1 FORCE_MUSA=1 FORCE_CUDA=0 python setup_musa.py install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/pytorch3d/torch_musa/lib:/home/torch_musa/build/lib64