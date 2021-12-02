#/bin/bash
CUDA=/home/qianyue/cuda/cuda-10.0
TF=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

$CUDA/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I $TF/include -I $TF/include/external/nsync/public -L $TF -ltensorflow_framework -I$CUDA/include -lcudart -L$CUDA/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

