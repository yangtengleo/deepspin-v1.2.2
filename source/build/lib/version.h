#pragma once

#include <string>
using namespace std;

#ifdef HIGH_PREC
const string global_float_prec="double";
#else 
const string global_float_prec="float";
#endif

const string global_install_prefix="/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd_root";
const string global_git_summ="v1.2.2-dirty";
const string global_git_hash="30922e7";
const string global_git_date="2020-09-14 22:49:07 +0800";
const string global_git_branch="HEAD";
const string global_tf_include_dir="/home/xuben/soft/tensorflow1.14_root/include;/home/xuben/soft/tensorflow1.14_root/include";
const string global_tf_lib="/home/xuben/soft/tensorflow1.14_root/lib/libtensorflow_cc.so;/home/xuben/soft/tensorflow1.14_root/lib/libtensorflow_framework.so";
