DEEPMD_ROOT=/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd_root
TENSORFLOW_INCLUDE_DIRS="/home/xuben/soft/tensorflow1.14_root/include;/home/xuben/soft/tensorflow1.14_root/include"
TENSORFLOW_LIBRARY_PATH="/home/xuben/soft/tensorflow1.14_root/lib;/home/xuben/soft/tensorflow1.14_root/lib"

TF_INCLUDE_DIRS=`echo $TENSORFLOW_INCLUDE_DIRS | sed "s/;/ -I/g"`
TF_LIBRARY_PATH=`echo $TENSORFLOW_LIBRARY_PATH | sed "s/;/ -L/g"`
TF_RPATH=`echo $TENSORFLOW_LIBRARY_PATH | sed "s/;/ -Wl,-rpath=/g"`

NNP_INC=" -std=c++11 -DHIGH_PREC   -I$TF_INCLUDE_DIRS -I$DEEPMD_ROOT/include/deepmd "
NNP_PATH=" -L$TF_LIBRARY_PATH -L$DEEPMD_ROOT/lib"
NNP_LIB=" -Wl,--no-as-needed -ldeepmd_op_cuda -ldeepmd_op -ldeepmd -ltensorflow_cc -ltensorflow_framework -Wl,-rpath=$TF_RPATH -Wl,-rpath=$DEEPMD_ROOT/lib"
