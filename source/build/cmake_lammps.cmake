set (LMP_INSTALL_PREFIX "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/build/USER-DEEPMD")
file(READ "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/build/lmp/lammps_install_list.txt" files)
string(REGEX REPLACE "\n" "" files "${files}")

foreach (cur_file ${files})
  file (
    INSTALL DESTINATION "${LMP_INSTALL_PREFIX}" 
    TYPE FILE
    FILES "${cur_file}"
    )
endforeach ()

file (
  INSTALL DESTINATION "${LMP_INSTALL_PREFIX}" 
  TYPE FILE 
  FILES "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/build/lmp/env.sh"
)

file (
  INSTALL DESTINATION "${LMP_INSTALL_PREFIX}" 
  TYPE FILE 
  FILES "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/build/lmp/pair_nnp.h"
)
