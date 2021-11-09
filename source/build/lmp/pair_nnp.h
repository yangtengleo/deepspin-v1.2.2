/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(deepmd,PairNNP)

#else

#ifndef LMP_PAIR_NNP_H
#define LMP_PAIR_NNP_H

#include "pair.h"
#include "NNPInter.h"
#include <iostream>
#include <fstream>
#include <map>

#define GIT_SUMM v1.2.2-dirty
#define GIT_HASH 30922e7
#define GIT_BRANCH spin
#define GIT_DATE 2020-09-14 22:49:07 +0800
#ifdef HIGH_PREC
#define FLOAT_PREC double
#else
#define FLOAT_PREC float
#endif
#define DEEPMD_ROOT /root/yangteng/dpspin-root
#define TensorFlow_INCLUDE_DIRS /root/tensorflow1.14_root/include;/root/tensorflow1.14_root/include
#define TensorFlow_LIBRARY /root/tensorflow1.14_root/lib/libtensorflow_cc.so;/root/tensorflow1.14_root/lib/libtensorflow_framework.so
#define DPMD_CVT_STR(x) #x
#define DPMD_CVT_ASSTR(X) DPMD_CVT_STR(X)
#define STR_GIT_SUMM DPMD_CVT_ASSTR(GIT_SUMM)
#define STR_GIT_HASH DPMD_CVT_ASSTR(GIT_HASH)
#define STR_GIT_BRANCH DPMD_CVT_ASSTR(GIT_BRANCH)
#define STR_GIT_DATE DPMD_CVT_ASSTR(GIT_DATE)
#define STR_FLOAT_PREC DPMD_CVT_ASSTR(FLOAT_PREC)
#define STR_DEEPMD_ROOT DPMD_CVT_ASSTR(DEEPMD_ROOT)
#define STR_TensorFlow_INCLUDE_DIRS DPMD_CVT_ASSTR(TensorFlow_INCLUDE_DIRS)
#define STR_TensorFlow_LIBRARY DPMD_CVT_ASSTR(TensorFlow_LIBRARY)

namespace LAMMPS_NS {

class PairNNP : public Pair {
public:
  PairNNP(class LAMMPS *);
  virtual ~PairNNP();
  virtual void compute(int, int);
  virtual void *extract(const char *, int &);
  void settings(int, char **);
  virtual void coeff(int, char **);
  void init_style();
  double init_one(int i, int j);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  void print_summary(const string pre) const;
  int get_node_rank();
  void extend (int &                            extend_inum,
              std::vector<int> &                extend_ilist,
              std::vector<int> &                extend_numneigh,
              std::vector<std::vector<int>> &   extend_neigh,
              std::vector<int *> &              extend_firstneigh,
              std::vector<VALUETYPE> &	        extend_coord,
              std::vector<int> &		            extend_atype,
              int &			                        extend_nghost,
              std::map<int, int> &              new_idx_map,
              std::map<int, int> &              old_idx_map,
              const LammpsNeighborList &	      lmp_list,
              const std::vector<VALUETYPE> &	  coord,
              const std::vector<int> &		      atype,
              const int			                    nghost,
              const std::vector<VALUETYPE> &	  spin,
              const int                         numb_types,
              const int                         numb_types_spin,
              const double                      spin_len);
  void cum_sum (std::map<int, int> & sec, std::map<int, int> & sel);
protected:  
  virtual void allocate();
  double **scale;

private:  
  NNPInter nnp_inter;
  NNPInterModelDevi nnp_inter_model_devi;
  unsigned numb_models;
  double cutoff;
  int numb_types;
  int numb_types_spin;
  vector<vector<double > > all_force;
  ofstream fp;
  int out_freq;
  string out_file;
  int dim_fparam;
  int dim_aparam;
  int out_each;
  int out_rel;
  bool single_model;
  bool multi_models_mod_devi;
  bool multi_models_no_mod_devi;
  int extend_inum; 
  vector<int> extend_ilist;
  vector<int> extend_numneigh;
  vector<vector<int>> extend_neigh;
  vector<int *> extend_firstneigh;
  vector<double> extend_dcoord;
  vector<int> extend_dtype;
  int extend_nghost;
  // for spin systems, search new index of atoms by their old index
  map<int, int> new_idx_map;
  map<int, int> old_idx_map;
#ifdef HIGH_PREC
  vector<double > fparam;
  vector<double > aparam;
  double eps;
#else
  vector<float > fparam;
  vector<float > aparam;
  float eps;
#endif
  void make_ttm_aparam(
#ifdef HIGH_PREC
      vector<double > & dparam
#else
      vector<float > & dparam
#endif
      );
  bool do_ttm;
  string ttm_fix_id;
};

}

#endif
#endif
