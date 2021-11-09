# Install script for directory: /home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/lib

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd_root")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/build/lib/libdeepmd.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/deepmd" TYPE FILE FILES
    "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/build/lib/version.h"
    "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/lib/include/ComputeDescriptor.h"
    "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/lib/include/DataModifier.h"
    "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/lib/include/DeepTensor.h"
    "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/lib/include/Ewald.h"
    "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/lib/include/MathUtilities.h"
    "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/lib/include/MathUtilities_Impl.h"
    "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/lib/include/NNPAtomMap.h"
    "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/lib/include/NNPInter.h"
    "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/lib/include/NeighborList.h"
    "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/lib/include/SimulationRegion.h"
    "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/lib/include/SimulationRegion_Impl.h"
    "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/lib/include/Stopwatch.h"
    "/home/xuben/soft/deepmd-kit-1.2.2-gpu/deepmd-kit/source/lib/include/common.h"
    )
endif()

