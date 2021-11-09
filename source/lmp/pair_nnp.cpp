#include <iostream>
#include <string.h>
#include <iomanip>
#include <limits>
#include <malloc.h>
#include "atom.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "update.h"
#include "output.h"
#include "error.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "pair_nnp.h"
#include <map>
#include <algorithm>
#include "modify.h"
#include "fix.h"
#include "citeme.h"
#ifdef USE_TTM
#include "fix_ttm_mod.h"
#endif

using namespace LAMMPS_NS;
using namespace std;

static const char cite_user_deepmd_package[] =
	"USER-DEEPMD package:\n\n"
    "@article{Wang_ComputPhysCommun_2018_v228_p178,\n"
    "  author = {Wang, Han and Zhang, Linfeng and Han, Jiequn and E, Weinan},\n"
    "  doi = {10.1016/j.cpc.2018.03.016},\n"
    "  url = {https://doi.org/10.1016/j.cpc.2018.03.016},\n"
    "  year = 2018,\n"
    "  month = {jul},\n"
    "  publisher = {Elsevier {BV}},\n"
    "  volume = 228,\n"
    "  journal = {Comput. Phys. Commun.},\n"
    "  title = {{DeePMD-kit: A deep learning package for many-body potential energy representation and molecular dynamics}},\n"
    "  pages = {178--184}\n"
	"}\n\n";


static int stringCmp(const void *a, const void* b)
{
    char* m = (char*)a;
    char* n = (char*)b;
    int i, sum = 0;

    for(i = 0; i < MPI_MAX_PROCESSOR_NAME; i++)
        if (m[i] == n[i])
            continue;
        else
        {
            sum = m[i] - n[i];
            break;
        }
    return sum;
}

int PairNNP::get_node_rank() {
    char host_name[MPI_MAX_PROCESSOR_NAME];
    memset(host_name, '\0', sizeof(char) * MPI_MAX_PROCESSOR_NAME);
    char (*host_names)[MPI_MAX_PROCESSOR_NAME];
    int n, namelen, color, rank, nprocs, myrank;
    size_t bytes;
    MPI_Comm nodeComm;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Get_processor_name(host_name,&namelen);

    bytes = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
    host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);
    for (int ii = 0; ii < nprocs; ii++) {
        memset(host_names[ii], '\0', sizeof(char) * MPI_MAX_PROCESSOR_NAME);
    }
    
    strcpy(host_names[rank], host_name);

    for (n=0; n<nprocs; n++)
        MPI_Bcast(&(host_names[n]),MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD);
    qsort(host_names, nprocs,  sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);

    color = 0;
    for (n=0; n<nprocs-1; n++)
    {
        if(strcmp(host_name, host_names[n]) == 0)
        {
            break;
        }
        if(strcmp(host_names[n], host_names[n+1]))
        {
            color++;
        }
    }

    MPI_Comm_split(MPI_COMM_WORLD, color, 0, &nodeComm);
    MPI_Comm_rank(nodeComm, &myrank);

    MPI_Barrier(MPI_COMM_WORLD);
    int looprank=myrank;
    // printf (" Assigning device %d  to process on node %s rank %d, OK\n",looprank,  host_name, rank );
    free(host_names);
    return looprank;
}

static void 
ana_st (double & max, 
	double & min, 
	double & sum, 
	const vector<double> & vec,
    const int & start, 
	const int & end) 
{
  if ((end - start) == 0) return;
  max = vec[start];
  min = vec[start];
  sum = vec[start];
  for (unsigned ii = start+1; ii < end; ++ii){
    if (vec[ii] > max) max = vec[ii];
    if (vec[ii] < min) min = vec[ii];
    sum += vec[ii];
  }
}

static void 
make_uniform_aparam(
#ifdef HIGH_PREC    
    vector<double > & daparam,
    const vector<double > & aparam,
    const int & nlocal
#else
    vector<float > & daparam,
    const vector<float > & aparam,
    const int & nlocal
#endif
    )
{
  unsigned dim_aparam = aparam.size();
  daparam.resize(dim_aparam * nlocal);
  for (int ii = 0; ii < nlocal; ++ii){
    for (int jj = 0; jj < dim_aparam; ++jj){
      daparam[ii*dim_aparam+jj] = aparam[jj];
    }
  }
}

#ifdef USE_TTM
void PairNNP::make_ttm_aparam(
#ifdef HIGH_PREC
    vector<double > & daparam
#else
    vector<float > & daparam
#endif
    )
{
  assert(do_ttm);
  // get ttm_fix
  const FixTTMMod * ttm_fix = NULL;
  for (int ii = 0; ii < modify->nfix; ii++) {
    if (string(modify->fix[ii]->id) == ttm_fix_id){
      ttm_fix = dynamic_cast<FixTTMMod*>(modify->fix[ii]);
    }
  }
  assert(ttm_fix);
  // modify
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  vector<int> nnodes = ttm_fix->get_nodes();
  int nxnodes = nnodes[0];
  int nynodes = nnodes[1];
  int nznodes = nnodes[2];
  double *** const T_electron = ttm_fix->get_T_electron();
  double dx = domain->xprd/nxnodes;
  double dy = domain->yprd/nynodes;
  double dz = domain->zprd/nynodes;
  // resize daparam
  daparam.resize(nlocal);
  // loop over atoms to assign aparam
  for (int ii = 0; ii < nlocal; ii++) {
    if (mask[ii] & ttm_fix->groupbit) {
      double xscale = (x[ii][0] - domain->boxlo[0])/domain->xprd;
      double yscale = (x[ii][1] - domain->boxlo[1])/domain->yprd;
      double zscale = (x[ii][2] - domain->boxlo[2])/domain->zprd;
      int ixnode = static_cast<int>(xscale*nxnodes);
      int iynode = static_cast<int>(yscale*nynodes);
      int iznode = static_cast<int>(zscale*nznodes);
      while (ixnode > nxnodes-1) ixnode -= nxnodes;
      while (iynode > nynodes-1) iynode -= nynodes;
      while (iznode > nznodes-1) iznode -= nznodes;
      while (ixnode < 0) ixnode += nxnodes;
      while (iynode < 0) iynode += nynodes;
      while (iznode < 0) iznode += nznodes;
      daparam[ii] = T_electron[ixnode][iynode][iznode];
    }
  }
}
#endif

void PairNNP::cum_sum (std::map<int, int> & sec, std::map<int, int> & sel)
{
  sec[0] = 0;
  for (int ii = 1; ii < sel.size(); ++ii){
    sec[ii] = sec[ii-1] + sel[ii-1];
  }
}

PairNNP::PairNNP(LAMMPS *lmp) 
    : Pair(lmp)
{
  if (lmp->citeme) lmp->citeme->add(cite_user_deepmd_package);
  if (strcmp(update->unit_style,"metal") != 0) {
    error->all(FLERR,"Pair deepmd requires metal unit, please set it by \"units metal\"");
  }
  pppmflag = 1;
  respa_enable = 0;
  writedata = 0;
  cutoff = 0.;
  numb_types = 0;
  numb_types_spin = 0;
  numb_models = 0;
  out_freq = 0;
  out_each = 0;
  out_rel = 0;
  eps = 0.;
  scale = NULL;
  do_ttm = false;
  single_model = false;
  multi_models_mod_devi = false;
  multi_models_no_mod_devi = false;
  // set comm size needed by this Pair
  comm_reverse = 1;

  print_summary("  ");
}

void
PairNNP::print_summary(const string pre) const
{
  if (comm->me == 0){
    cout << "Summary of lammps deepmd module ..." << endl;
    cout << pre << ">>> Info of deepmd-kit:" << endl;
    nnp_inter.print_summary(pre);
    cout << pre << ">>> Info of lammps module:" << endl;
    cout << pre << "use deepmd-kit at:  " << STR_DEEPMD_ROOT << endl;
    cout << pre << "source:             " << STR_GIT_SUMM << endl;
    cout << pre << "source branch:      " << STR_GIT_BRANCH << endl;
    cout << pre << "source commit:      " << STR_GIT_HASH << endl;
    cout << pre << "source commit at:   " << STR_GIT_DATE << endl;
    cout << pre << "build float prec:   " << STR_FLOAT_PREC << endl;
    cout << pre << "build with tf inc:  " << STR_TensorFlow_INCLUDE_DIRS << endl;
    cout << pre << "build with tf lib:  " << STR_TensorFlow_LIBRARY << endl;
  }
}


PairNNP::~PairNNP()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(scale);
  }
}

void PairNNP::compute(int eflag, int vflag)
{
  if (numb_models == 0) return;
  if (eflag || vflag) ev_setup(eflag,vflag);
  bool do_ghost = true;
  
  // assure atom_style is spin
  if (!atom->sp_flag)
      error->all(FLERR,"Compute compute/spin requires atom/spin style");
  
  double **x = atom->x;
  double **f = atom->f;
  // spin pointers
  double **sp = atom->sp;
  double **fm = atom->fm;

  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = 0;
  if (do_ghost) {
    nghost = atom->nghost;
  }
  int nall = nlocal + nghost;
  int newton_pair = force->newton_pair;

  vector<int > dtype (nall);
  for (int ii = 0; ii < nall; ++ii){
    dtype[ii] = type[ii] - 1;
  }  

  double dener (0.);
  vector<double > dforce (nall * 3);
  vector<double > dvirial (9, 0);
  vector<double > dcoord (nall * 3, 0.);
  vector<double > dbox (9, 0);
#ifdef HIGH_PREC
  vector<double > daparam;
#else 
  vector<float > daparam;
#endif
  // new arrays for spin
  double spin_len (0.3);
  vector<double > dspin (nall * 3, 0.);
  vector<double > dfm (nall * 3, 0.);

  // get box
  dbox[0] = domain->h[0];	// xx
  dbox[4] = domain->h[1];	// yy
  dbox[8] = domain->h[2];	// zz
  dbox[7] = domain->h[3];	// zy
  dbox[6] = domain->h[4];	// zx
  dbox[3] = domain->h[5];	// yx

  // get coord
  for (int ii = 0; ii < nall; ++ii){
    for (int dd = 0; dd < 3; ++dd){
      dcoord[ii*3+dd] = x[ii][dd] - domain->boxlo[dd];
    }
  }

  // get spin 
  for (int ii = 0; ii < nall; ++ii){
    for (int dd = 0; dd < 3; ++dd){
      dspin[ii*3+dd] = sp[ii][dd];
    }
  }

  // uniform aparam
  if (aparam.size() > 0){
    make_uniform_aparam(daparam, aparam, nlocal);
  }
  else if (do_ttm) {
#ifdef USE_TTM
    make_ttm_aparam(daparam);
#endif
  }

  // int ago = numb_models > 1 ? 0 : neighbor->ago;
  int ago = neighbor->ago;
  if (numb_models > 1) {
      if (multi_models_no_mod_devi && (out_freq > 0 && update->ntimestep % out_freq == 0)) {
        ago = 0;
      }
      else if (multi_models_mod_devi && (out_freq == 0 || update->ntimestep % out_freq != 0)) {
        ago = 0;
      }
  }
  // compute
  single_model = (numb_models == 1);
  multi_models_no_mod_devi = (numb_models > 1 && (out_freq == 0 || update->ntimestep % out_freq != 0));
  multi_models_mod_devi = (numb_models > 1 && (out_freq > 0 && update->ntimestep % out_freq == 0));
  if (do_ghost) {
    LammpsNeighborList lmp_list (list->inum, list->ilist, list->numneigh, list->firstneigh);
    extend(extend_inum, extend_ilist, extend_numneigh, extend_neigh, extend_firstneigh, extend_dcoord, extend_dtype, extend_nghost,
           new_idx_map, old_idx_map, lmp_list, dcoord, dtype, nghost, dspin, numb_types, numb_types_spin, spin_len);
    LammpsNeighborList extend_lmp_list (extend_inum, &extend_ilist[0], &extend_numneigh[0], &extend_firstneigh[0]);

    if (single_model || multi_models_no_mod_devi) {
      if ( ! (eflag_atom || vflag_atom) ) {      
        #ifdef HIGH_PREC
          if (!atom->sp_flag){
            nnp_inter.compute (dener, dforce, dvirial, dcoord, dtype, dbox, nghost, lmp_list, ago, fparam, daparam);
          }
          else{
            dforce.resize((extend_inum + extend_nghost) * 3);
            nnp_inter.compute (dener, dforce, dvirial, extend_dcoord, extend_dtype, dbox, extend_nghost, extend_lmp_list, ago, fparam, daparam);
          }
        #else
          vector<float> dcoord_(dcoord.size());
          vector<float> dbox_(dbox.size());
          for (unsigned dd = 0; dd < dcoord.size(); ++dd) dcoord_[dd] = dcoord[dd];
          for (unsigned dd = 0; dd < dbox.size(); ++dd) dbox_[dd] = dbox[dd];
          vector<float> dforce_(dforce.size(), 0);
          vector<float> dvirial_(dvirial.size(), 0);
          double dener_ = 0;
          nnp_inter.compute (dener_, dforce_, dvirial_, dcoord_, dtype, dbox_, nghost, lmp_list, ago, fparam, daparam);
          for (unsigned dd = 0; dd < dforce.size(); ++dd) dforce[dd] = dforce_[dd];	
          for (unsigned dd = 0; dd < dvirial.size(); ++dd) dvirial[dd] = dvirial_[dd];	
          dener = dener_;
        #endif
      }
      // do atomic energy and virial
      else {
          vector<double > deatom (nall * 1, 0);
          vector<double > dvatom (nall * 9, 0);
        #ifdef HIGH_PREC
          nnp_inter.compute (dener, dforce, dvirial, deatom, dvatom, dcoord, dtype, dbox, nghost, lmp_list, ago, fparam, daparam);
        #else 
          vector<float> dcoord_(dcoord.size());
          vector<float> dbox_(dbox.size());
          for (unsigned dd = 0; dd < dcoord.size(); ++dd) dcoord_[dd] = dcoord[dd];
          for (unsigned dd = 0; dd < dbox.size(); ++dd) dbox_[dd] = dbox[dd];
          vector<float> dforce_(dforce.size(), 0);
          vector<float> dvirial_(dvirial.size(), 0);
          vector<float> deatom_(dforce.size(), 0);
          vector<float> dvatom_(dforce.size(), 0);
          double dener_ = 0;
          nnp_inter.compute (dener_, dforce_, dvirial_, deatom_, dvatom_, dcoord_, dtype, dbox_, nghost, lmp_list, ago, fparam, daparam);
          for (unsigned dd = 0; dd < dforce.size(); ++dd) dforce[dd] = dforce_[dd];	
          for (unsigned dd = 0; dd < dvirial.size(); ++dd) dvirial[dd] = dvirial_[dd];	
          for (unsigned dd = 0; dd < deatom.size(); ++dd) deatom[dd] = deatom_[dd];	
          for (unsigned dd = 0; dd < dvatom.size(); ++dd) dvatom[dd] = dvatom_[dd];	
          dener = dener_;
        #endif	
          if (eflag_atom) {
            for (int ii = 0; ii < nlocal; ++ii) eatom[ii] += deatom[ii];
          }
          if (vflag_atom) {
            for (int ii = 0; ii < nall; ++ii){
              vatom[ii][0] += 1.0 * dvatom[9*ii+0];
              vatom[ii][1] += 1.0 * dvatom[9*ii+4];
              vatom[ii][2] += 1.0 * dvatom[9*ii+8];
              vatom[ii][3] += 1.0 * dvatom[9*ii+3];
              vatom[ii][4] += 1.0 * dvatom[9*ii+6];
              vatom[ii][5] += 1.0 * dvatom[9*ii+7];
            }
          }
      }
    }
    else if (multi_models_mod_devi) {
      vector<double > deatom (nall * 1, 0);
      vector<double > dvatom (nall * 9, 0);
    #ifdef HIGH_PREC
      vector<double> 		    all_energy;
      vector<vector<double>> 	all_virial;	       
      vector<vector<double>> 	all_atom_energy;
      vector<vector<double>> 	all_atom_virial;
      nnp_inter_model_devi.compute(all_energy, all_force, all_virial, all_atom_energy, all_atom_virial, 
                                   extend_dcoord, extend_dtype, dbox, extend_nghost, extend_lmp_list, ago, fparam, daparam);
      // nnp_inter_model_devi.compute_avg (dener, all_energy);
      // nnp_inter_model_devi.compute_avg (dforce, all_force);
      // nnp_inter_model_devi.compute_avg (dvirial, all_virial);
      // nnp_inter_model_devi.compute_avg (deatom, all_atom_energy);
      // nnp_inter_model_devi.compute_avg (dvatom, all_atom_virial);
      dener = all_energy[0];
      dforce = all_force[0];
      dvirial = all_virial[0];
      deatom = all_atom_energy[0];
      dvatom = all_atom_virial[0];
    #else 
      vector<float> dcoord_(dcoord.size());
      vector<float> dbox_(dbox.size());
      for (unsigned dd = 0; dd < dcoord.size(); ++dd) dcoord_[dd] = dcoord[dd];
      for (unsigned dd = 0; dd < dbox.size(); ++dd) dbox_[dd] = dbox[dd];
      vector<float> dforce_(dforce.size(), 0);
      vector<float> dvirial_(dvirial.size(), 0);
      vector<float> deatom_(dforce.size(), 0);
      vector<float> dvatom_(dforce.size(), 0);
      double dener_ = 0;
      vector<double> 		all_energy_;
      vector<vector<float>>	all_force_;
      vector<vector<float>> 	all_virial_;	       
      vector<vector<float>> 	all_atom_energy_;
      vector<vector<float>> 	all_atom_virial_;
      nnp_inter_model_devi.compute(all_energy_, all_force_, all_virial_, all_atom_energy_, all_atom_virial_, dcoord_, dtype, dbox_, nghost, lmp_list, ago, fparam, daparam);
      // nnp_inter_model_devi.compute_avg (dener_, all_energy_);
      // nnp_inter_model_devi.compute_avg (dforce_, all_force_);
      // nnp_inter_model_devi.compute_avg (dvirial_, all_virial_);
      // nnp_inter_model_devi.compute_avg (deatom_, all_atom_energy_);
      // nnp_inter_model_devi.compute_avg (dvatom_, all_atom_virial_);
      dener_ = all_energy_[0];
      dforce_ = all_force_[0];
      dvirial_ = all_virial_[0];
      deatom_ = all_atom_energy_[0];
      dvatom_ = all_atom_virial_[0];
      dener = dener_;
      for (unsigned dd = 0; dd < dforce.size(); ++dd) dforce[dd] = dforce_[dd];	
      for (unsigned dd = 0; dd < dvirial.size(); ++dd) dvirial[dd] = dvirial_[dd];	
      for (unsigned dd = 0; dd < deatom.size(); ++dd) deatom[dd] = deatom_[dd];	
      for (unsigned dd = 0; dd < dvatom.size(); ++dd) dvatom[dd] = dvatom_[dd];	
      all_force.resize(all_force_.size());
      for (unsigned ii = 0; ii < all_force_.size(); ++ii){
        all_force[ii].resize(all_force_[ii].size());
        for (unsigned jj = 0; jj < all_force_[ii].size(); ++jj){
          all_force[ii][jj] = all_force_[ii][jj];
        }
      }
    #endif
      if (eflag_atom) {
        for (int ii = 0; ii < nlocal; ++ii) eatom[ii] += deatom[ii];
      }
      if (vflag_atom) {
        for (int ii = 0; ii < nall; ++ii){
          vatom[ii][0] += 1.0 * dvatom[9*ii+0];
          vatom[ii][1] += 1.0 * dvatom[9*ii+4];
          vatom[ii][2] += 1.0 * dvatom[9*ii+8];
          vatom[ii][3] += 1.0 * dvatom[9*ii+3];
          vatom[ii][4] += 1.0 * dvatom[9*ii+6];
          vatom[ii][5] += 1.0 * dvatom[9*ii+7];
        }
      }      
      if (out_freq > 0 && update->ntimestep % out_freq == 0) {
        int rank = comm->me;
        // std force 
        if (newton_pair) {
          comm->reverse_comm_pair(this);
        }
        vector<double> std_f;
      #ifdef HIGH_PREC
        vector<double> tmp_avg_f;
        nnp_inter_model_devi.compute_avg (tmp_avg_f, all_force);  
        nnp_inter_model_devi.compute_std_f (std_f, tmp_avg_f, all_force);
        if (out_rel == 1){
          nnp_inter_model_devi.compute_relative_std_f (std_f, tmp_avg_f, eps);
        }
      #else 
        vector<float> tmp_avg_f_, std_f_;
        for (unsigned ii = 0; ii < all_force_.size(); ++ii){
          for (unsigned jj = 0; jj < all_force_[ii].size(); ++jj){
            all_force_[ii][jj] = all_force[ii][jj];
          }
        }
        nnp_inter_model_devi.compute_avg (tmp_avg_f_, all_force_);  
        nnp_inter_model_devi.compute_std_f (std_f_, tmp_avg_f_, all_force_);
        std_f.resize(std_f_.size());
        for (int dd = 0; dd < std_f_.size(); ++dd) std_f[dd] = std_f_[dd];
        if (out_rel == 1){
            nnp_inter_model_devi.compute_relative_std_f (std_f_, tmp_avg_f_, eps);
        }
      #endif
      double min_fa = numeric_limits<double>::max(), max_fa = 0, avg_fa = 0;
      double min_fs = numeric_limits<double>::max(), max_fs = 0, avg_fs = 0;
      ana_st(max_fa, min_fa, avg_fa, std_f, 0, nlocal);
      ana_st(max_fs, min_fs, avg_fs, std_f, nlocal, extend_inum);
      int all_nlocal_atom = 0;
      int all_nlocal_spin = 0;
      int nlocal_spin = extend_inum - nlocal;
      MPI_Reduce (&nlocal, &all_nlocal_atom, 1, MPI_INT, MPI_SUM, 0, world);
      MPI_Reduce (&(nlocal_spin), &all_nlocal_spin, 1, MPI_INT, MPI_SUM, 0, world);
      double all_fa_min = 0, all_fa_max = 0, all_fa_avg = 0;
      double all_fs_min = 0, all_fs_max = 0, all_fs_avg = 0;  
      MPI_Reduce (&min_fa, &all_fa_min, 1, MPI_DOUBLE, MPI_MIN, 0, world);
      MPI_Reduce (&max_fa, &all_fa_max, 1, MPI_DOUBLE, MPI_MAX, 0, world);
      MPI_Reduce (&avg_fa, &all_fa_avg, 1, MPI_DOUBLE, MPI_SUM, 0, world);
      MPI_Reduce (&min_fs, &all_fs_min, 1, MPI_DOUBLE, MPI_MIN, 0, world);
      MPI_Reduce (&max_fs, &all_fs_max, 1, MPI_DOUBLE, MPI_MAX, 0, world);
      MPI_Reduce (&avg_fs, &all_fs_avg, 1, MPI_DOUBLE, MPI_SUM, 0, world);
      all_fa_avg /= double(all_nlocal_atom);
      all_fs_avg /= double(all_nlocal_spin);
      // std energy
      vector<double > std_e;
      #ifdef HIGH_PREC
        vector<double > tmp_avg_e;
        nnp_inter_model_devi.compute_avg (tmp_avg_e, all_atom_energy);
        nnp_inter_model_devi.compute_std_e (std_e, tmp_avg_e, all_atom_energy);
      #else 
        vector<float> tmp_avg_e_, std_e_;
        nnp_inter_model_devi.compute_avg (tmp_avg_e_, all_atom_energy_);
        nnp_inter_model_devi.compute_std_e (std_e_, tmp_avg_e_, all_atom_energy_);
        std_e.resize(std_e_.size());
        for (int dd = 0; dd < std_e_.size(); ++dd) std_e[dd] = std_e_[dd];
      #endif	
        double min_e = numeric_limits<double>::max(), max_e = 0, avg_e = 0;
        ana_st(max_e, min_e, avg_e, std_e, 0, nlocal);
        double all_e_min = 0, all_e_max = 0, all_e_avg = 0;
        MPI_Reduce (&min_e, &all_e_min, 1, MPI_DOUBLE, MPI_MIN, 0, world);
        MPI_Reduce (&max_e, &all_e_max, 1, MPI_DOUBLE, MPI_MAX, 0, world);
        MPI_Reduce (&avg_e, &all_e_avg, 1, MPI_DOUBLE, MPI_SUM, 0, world);
        all_e_avg /= double(all_nlocal_atom);
        // // total e
        // vector<double > sum_e(numb_models, 0.);
        // MPI_Reduce (&all_energy[0], &sum_e[0], numb_models, MPI_DOUBLE, MPI_SUM, 0, world);
        if (rank == 0) {
          // double avg_e = 0;
          // nnp_inter_model_devi.compute_avg(avg_e, sum_e);
          // double std_e_1 = 0;
          // nnp_inter_model_devi.compute_std(std_e_1, avg_e, sum_e);	
          fp << setw(12) << update->ntimestep 
            << " " << setw(18) << all_e_max 
            << " " << setw(18) << all_e_min
            << " " << setw(18) << all_e_avg
            << " " << setw(18) << all_fa_max 
            << " " << setw(18) << all_fa_min
            << " " << setw(18) << all_fa_avg
            << " " << setw(18) << all_fs_max 
            << " " << setw(18) << all_fs_min
            << " " << setw(18) << all_fs_avg;
            // << " " << setw(18) << avg_e
            // << " " << setw(18) << std_e_1 / all_nlocal
          if (out_each == 1){
              // TODO: Fix two problems:
              // 1. If the atom_style is not atomic (e.g. charge), the order of std_f is different from that of atom ids.
                    // 2. std_f is not gathered by MPI.
              for (int dd = 0; dd < all_nlocal_atom; ++dd) {
                  fp << " " << setw(18) << std_f[dd];	
              }
          }
          fp << endl;
        }
      }
    }
    else {
      error->all(FLERR,"unknown computational branch");
    }
  }
  else {
    if (numb_models == 1) {
    #ifdef HIGH_PREC
      nnp_inter.compute (dener, dforce, dvirial, dcoord, dtype, dbox, nghost);
    #else
      vector<float> dcoord_(dcoord.size());
      vector<float> dbox_(dbox.size());
      for (unsigned dd = 0; dd < dcoord.size(); ++dd) dcoord_[dd] = dcoord[dd];
      for (unsigned dd = 0; dd < dbox.size(); ++dd) dbox_[dd] = dbox[dd];
      vector<float> dforce_(dforce.size(), 0);
      vector<float> dvirial_(dvirial.size(), 0);
      double dener_ = 0;
      nnp_inter.compute (dener_, dforce_, dvirial_, dcoord_, dtype, dbox_, nghost);
      for (unsigned dd = 0; dd < dforce.size(); ++dd) dforce[dd] = dforce_[dd];	
      for (unsigned dd = 0; dd < dvirial.size(); ++dd) dvirial[dd] = dvirial_[dd];	
      dener = dener_;      
    #endif
    }
    else {
      error->all(FLERR,"Serial version does not support model devi");
    }
  }

  // get force
  if (!atom->sp_flag){
    for (int ii = 0; ii < nall; ++ii){
      for (int dd = 0; dd < 3; ++dd){
        f[ii][dd] += scale[1][1] * dforce[3*ii+dd];
      }
    }
  }
  else{
    // unit_factor = hbar / spin_norm;
    double unit_factor = 6.5821191e-04 / 1.549;
    for (int ii = 0; ii < nall; ++ii){
      for (int dd = 0; dd < 3; ++dd){
        int new_idx = new_idx_map[ii];
        f[ii][dd] += scale[1][1] * dforce[3*new_idx+dd];
        if (dtype[ii] < numb_types_spin && ii < nlocal){
          fm[ii][dd] += scale[1][1] * dforce[3*(new_idx+nlocal)+dd] / unit_factor;
        }
        else if (dtype[ii] < numb_types_spin){
          fm[ii][dd] += scale[1][1] * dforce[3*(new_idx+nghost)+dd] / unit_factor;
        }
      }
    }
  }

  map<int, int>().swap(new_idx_map);
  map<int, int>().swap(old_idx_map);
  malloc_trim(0);
  
  // accumulate energy and virial
  if (eflag) eng_vdwl += scale[1][1] * dener;
  if (vflag) {
    virial[0] += 1.0 * dvirial[0] * scale[1][1];
    virial[1] += 1.0 * dvirial[4] * scale[1][1];
    virial[2] += 1.0 * dvirial[8] * scale[1][1];
    virial[3] += 1.0 * dvirial[3] * scale[1][1];
    virial[4] += 1.0 * dvirial[6] * scale[1][1];
    virial[5] += 1.0 * dvirial[7] * scale[1][1];
  }
}


void PairNNP::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(scale,n+1,n+1,"pair:scale");

  for (int i = 1; i <= n; i++){
    for (int j = i; j <= n; j++){
      setflag[i][j] = 0;
      scale[i][j] = 0;
    }
  }
  for (int i = 1; i <= numb_types; ++i) {
    if (i > n) continue;
    for (int j = i; j <= numb_types; ++j) {
      if (j > n) continue;
      setflag[i][j] = 1;
      scale[i][j] = 1;
    }
  }
}


static bool 
is_key (const string& input) 
{
  vector<string> keys ;
  keys.push_back("out_freq");
  keys.push_back("out_file");
  keys.push_back("fparam");
  keys.push_back("aparam");
  keys.push_back("ttm");
  keys.push_back("atomic");
  keys.push_back("relative");

  for (int ii = 0; ii < keys.size(); ++ii){
    if (input == keys[ii]) {
      return true;
    }
  }
  return false;
}


void PairNNP::settings(int narg, char **arg)
{
  if (narg <= 0) error->all(FLERR,"Illegal pair_style command");

  vector<string> models;
  int iarg = 0;
  while (iarg < narg){
    if (is_key(arg[iarg])) {
      break;
    }
    iarg ++;
  }
  for (int ii = 0; ii < iarg; ++ii){
    models.push_back(arg[ii]);
  }
  numb_models = models.size();
  if (numb_models == 1) {
    nnp_inter.init (arg[0], get_node_rank());
    cutoff = nnp_inter.cutoff ();
    numb_types = nnp_inter.numb_types();
    numb_types_spin = nnp_inter.numb_types_spin();
    dim_fparam = nnp_inter.dim_fparam();
    dim_aparam = nnp_inter.dim_aparam();
  }
  else {
    nnp_inter.init (arg[0], get_node_rank());
    nnp_inter_model_devi.init(models, get_node_rank());
    cutoff = nnp_inter_model_devi.cutoff();
    numb_types = nnp_inter_model_devi.numb_types();
    numb_types_spin = nnp_inter_model_devi.numb_types_spin();
    dim_fparam = nnp_inter_model_devi.dim_fparam();
    dim_aparam = nnp_inter_model_devi.dim_aparam();
    assert(cutoff == nnp_inter.cutoff());
    assert(numb_types == nnp_inter.numb_types());
    assert(numb_types_spin == nnp_inter.numb_types_spin());
    assert(dim_fparam == nnp_inter.dim_fparam());
    assert(dim_aparam == nnp_inter.dim_aparam());
  }

  out_freq = 100;
  out_file = "model_devi.out";
  out_each = 0;
  out_rel = 0;
  eps = 0.;
  fparam.clear();
  aparam.clear();
  while (iarg < narg) {
    if (! is_key(arg[iarg])) {
      error->all(FLERR,"Illegal pair_style command\nwrong number of parameters\n");
    }
    if (string(arg[iarg]) == string("out_freq")) {
      if (iarg+1 >= narg) error->all(FLERR,"Illegal out_freq, not provided");
      out_freq = atoi(arg[iarg+1]);
      iarg += 2;
    }
    else if (string(arg[iarg]) == string("out_file")) {
      if (iarg+1 >= narg) error->all(FLERR,"Illegal out_file, not provided");
      out_file = string(arg[iarg+1]);	
      iarg += 2;
    }
    else if (string(arg[iarg]) == string("fparam")) {
      for (int ii = 0; ii < dim_fparam; ++ii){
	if (iarg+1+ii >= narg || is_key(arg[iarg+1+ii])) {
	  char tmp[1024];
	  sprintf(tmp, "Illegal fparam, the dimension should be %d", dim_fparam);		  
	  error->all(FLERR, tmp);
	}
	fparam.push_back(atof(arg[iarg+1+ii]));
      }
      iarg += 1 + dim_fparam ;
    }
    else if (string(arg[iarg]) == string("aparam")) {
      for (int ii = 0; ii < dim_aparam; ++ii){
	if (iarg+1+ii >= narg || is_key(arg[iarg+1+ii])) {
	  char tmp[1024];
	  sprintf(tmp, "Illegal aparam, the dimension should be %d", dim_aparam);		  
	  error->all(FLERR, tmp);
	}
	aparam.push_back(atof(arg[iarg+1+ii]));
      }      
      iarg += 1 + dim_aparam ;
    }
    else if (string(arg[iarg]) == string("ttm")) {
#ifdef USE_TTM
      for (int ii = 0; ii < 1; ++ii){
	if (iarg+1+ii >= narg || is_key(arg[iarg+1+ii])) {
	  error->all(FLERR, "invalid ttm key: should be ttm ttm_fix_id(str)");
	}
      }	
      do_ttm = true;
      ttm_fix_id = arg[iarg+1];
      iarg += 1 + 1;
#else
      error->all(FLERR, "The deepmd-kit was compiled without support for TTM, please rebuild it with -DUSE_TTM");
#endif      
    }
    else if (string(arg[iarg]) == string("atomic")) {
      out_each = 1;
      iarg += 1;
    }
    else if (string(arg[iarg]) == string("relative")) {
      out_rel = 1;
#ifdef HIGH_PREC
      eps = atof(arg[iarg+1]);
#else
      eps = strtof(arg[iarg+1], NULL);
#endif
      iarg += 2;
    }
  }
  if (out_freq < 0) error->all(FLERR,"Illegal out_freq, should be >= 0");
  if (do_ttm && aparam.size() > 0) {
    error->all(FLERR,"aparam and ttm should NOT be set simultaneously");
  }
  
  if (comm->me == 0){
    if (numb_models > 1 && out_freq > 0){
      fp.open (out_file);
      fp << scientific;
      fp << "#"
	 << setw(12-1) << "step" 
	 << setw(18+1) << "max_devi_e"
	 << setw(18+1) << "min_devi_e"
	 << setw(18+1) << "avg_devi_e"
	 << setw(18+1) << "max_devi_fa"
	 << setw(18+1) << "min_devi_fa"
	 << setw(18+1) << "avg_devi_fa"
     << setw(18+1) << "max_devi_fs"
	 << setw(18+1) << "min_devi_fs"
	 << setw(18+1) << "avg_devi_fs"
	 << endl;
    }
    string pre = "  ";
    cout << pre << ">>> Info of model(s):" << endl
	 << pre << "using " << setw(3) << numb_models << " model(s): ";
    if (narg == 1) {
      cout << arg[0] << " ";
    }
    else {
      for (int ii = 0; ii < models.size(); ++ii){
      	cout << models[ii] << " ";
      }
    }
    cout << endl
	 << pre << "rcut in model:      " << cutoff << endl
	 << pre << "ntypes in model:    " << numb_types << endl;
    if (dim_fparam > 0) {
      cout << pre << "using fparam(s):    " ;
      for (int ii = 0; ii < dim_fparam; ++ii){
	cout << fparam[ii] << "  " ;
      }
      cout << endl;
    }
    if (aparam.size() > 0) {
      cout << pre << "using aparam(s):    " ;
      for (int ii = 0; ii < aparam.size(); ++ii){
	cout << aparam[ii] << "  " ;
      }
      cout << endl;
    }
  }
  
  comm_reverse = numb_models * 3;
  all_force.resize(numb_models);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairNNP::coeff(int narg, char **arg)
{
  if (!allocated) {
    allocate();
  }

  int n = atom->ntypes;
  int ilo,ihi,jlo,jhi;
  ilo = 0;
  jlo = 0;
  ihi = n;
  jhi = n;
  if (narg == 2) {
    force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
    force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);
    if (ilo != 1 || jlo != 1 || ihi != n || jhi != n) {
      error->all(FLERR,"deepmd requires that the scale should be set to all atom types, i.e. pair_coeff * *.");
    }
  }  
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      setflag[i][j] = 1;
      scale[i][j] = 1.0;
      if (i > numb_types || j > numb_types) {
	char warning_msg[1024];
	sprintf(warning_msg, "Interaction between types %d and %d is set with deepmd, but will be ignored.\n Deepmd model has only %d types, it only computes the mulitbody interaction of types: 1-%d.", i, j, numb_types, numb_types);
	error->warning(FLERR, warning_msg);
      }
    }
  }
}


void PairNNP::init_style()
{
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  // neighbor->requests[irequest]->full = 1;  
  // neighbor->requests[irequest]->newton = 2;  
}


double PairNNP::init_one(int i, int j)
{
  if (i > numb_types || j > numb_types) {
    char warning_msg[1024];
    sprintf(warning_msg, "Interaction between types %d and %d is set with deepmd, but will be ignored.\n Deepmd model has only %d types, it only computes the mulitbody interaction of types: 1-%d.", i, j, numb_types, numb_types);
    error->warning(FLERR, warning_msg);
  }

  if (setflag[i][j] == 0) scale[i][j] = 1.0;
  scale[j][i] = scale[i][j];

  return cutoff;
}


/* ---------------------------------------------------------------------- */

int PairNNP::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    for (int dd = 0; dd < numb_models; ++dd){
      buf[m++] = all_force[dd][3*i+0];
      buf[m++] = all_force[dd][3*i+1];
      buf[m++] = all_force[dd][3*i+2];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairNNP::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    for (int dd = 0; dd < numb_models; ++dd){
      all_force[dd][3*j+0] += buf[m++];
      all_force[dd][3*j+1] += buf[m++];
      all_force[dd][3*j+2] += buf[m++];
    }
  }
}

void *PairNNP::extract(const char *str, int &dim)
{
  if (strcmp(str,"cut_coul") == 0) {
    dim = 0;
    return (void *) &cutoff;
  }
  if (strcmp(str,"scale") == 0) {
    dim = 2;
    return (void *) scale;
  }
  return NULL;
}

void PairNNP::extend(int &	                     extend_inum,
                    vector<int> &                extend_ilist,
                    vector<int> &                extend_numneigh,
                    vector<vector<int>> &        extend_neigh,
                    vector<int *> &              extend_firstneigh,
                    vector<double> &	           extend_dcoord,
                    vector<int> &		             extend_atype,
                    int &			                   extend_nghost,
                    map<int, int> &              new_idx_map,
                    map<int, int> &              old_idx_map,
                    const LammpsNeighborList &	 lmp_list,
                    const vector<double> &	     dcoord,
                    const vector<int> &		       atype,
                    const int			               nghost,
                    const vector<double> &	     spin,
                    const int                    numb_types,
                    const int                    numb_types_spin,
                    const double                 spin_len)
{
  int nall = dcoord.size() / 3;
  int nloc = nall - nghost;
  assert(nloc == lmp_list.inum);

  // record numb_types_real and nloc_virt
  int numb_types_real = numb_types - numb_types_spin;
  map<int, int> loc_type_count;
	map<int, int>::iterator iter = loc_type_count.begin();
	for (int i = 0; i < nloc; i++){
		iter = loc_type_count.find(atype[i]);
		if (iter != loc_type_count.end())
			iter->second += 1;
		else
			loc_type_count.insert(pair<int, int>(atype[i], 1));
	}
  assert(numb_types_real-1 == loc_type_count.rbegin()->first);
  int nloc_virt = 0;
  for (int i = 0; i < numb_types_spin; i++){
    nloc_virt += loc_type_count[i];
  }
  
  // record nghost_virt
  map<int, int> ghost_type_count;
  for (int i = nloc; i < nall; i++){
		iter = ghost_type_count.find(atype[i]);
		if (iter != ghost_type_count.end())
			iter->second += 1;
		else
			ghost_type_count.insert(pair<int, int>(atype[i], 1));
	}
  int nghost_virt = 0;
  for (int i = 0; i < numb_types_spin; i++){
    nghost_virt += ghost_type_count[i];
  }

  // for extended system, search new index by old index, and vice versa
  extend_nghost = nghost + nghost_virt;
  int extend_nloc = nloc + nloc_virt;
  int extend_nall = extend_nloc + extend_nghost;
  map<int, int> cum_loc_type_count;
  map<int, int> cum_ghost_type_count;
  cum_sum(cum_loc_type_count, loc_type_count);
  cum_sum(cum_ghost_type_count, ghost_type_count);
  vector<int>  loc_type_reset (numb_types_real, 0);
  vector<int>  ghost_type_reset (numb_types_real, 0);

  new_idx_map.clear();
  old_idx_map.clear();
  for (int ii = 0; ii < nloc; ii++){
    int new_idx = cum_loc_type_count[atype[ii]] + loc_type_reset[atype[ii]];
    new_idx_map[ii] = new_idx;
    old_idx_map[new_idx] = ii;
    loc_type_reset[atype[ii]]++;
  }
  for (int ii = nloc; ii < nall; ii++){
    int new_idx = cum_ghost_type_count[atype[ii]] + ghost_type_reset[atype[ii]] + extend_nloc;
    new_idx_map[ii] = new_idx;
    old_idx_map[new_idx] = ii;
    ghost_type_reset[atype[ii]]++;
  }

  // extend lmp_list
  extend_inum = extend_nloc;

  extend_ilist.resize(extend_nloc);
  for (int ii = 0; ii < extend_nloc; ii++){
    extend_ilist[ii] = ii;
  }

  extend_neigh.resize(extend_nloc);
  for (int ii = 0; ii < nloc; ii++){
    int jnum = lmp_list.numneigh[old_idx_map[ii]];
    const int * jlist = lmp_list.firstneigh[old_idx_map[ii]];
    extend_neigh[ii].push_back(ii+nloc);
    for (int jj = 0; jj < jnum; jj++){
      int new_idx = new_idx_map[jlist[jj]];
      extend_neigh[ii].push_back(new_idx);
      if (atype[jlist[jj]] < numb_types_spin && jlist[jj] < nloc){
        extend_neigh[ii].push_back(new_idx + nloc);
      }
      else if (atype[jlist[jj]] < numb_types_spin && jlist[jj] < nall){
        extend_neigh[ii].push_back(new_idx + nghost);
      }
    }
  }
  for (int ii = nloc; ii < extend_nloc; ii++){
    extend_neigh[ii].assign(extend_neigh[ii-nloc].begin(), extend_neigh[ii-nloc].end());
    vector<int>::iterator it = find(extend_neigh[ii].begin(), extend_neigh[ii].end(), ii);
    *it = ii - nloc;
  }

  extend_firstneigh.resize(extend_nloc);
  extend_numneigh.resize(extend_nloc);
  for (int ii = 0; ii < extend_nloc; ii++){
    extend_firstneigh[ii] = &extend_neigh[ii][0];
    extend_numneigh[ii] = extend_neigh[ii].size();
  }

  // extend coord
  extend_dcoord.resize(extend_nall * 3);
  for (int ii = 0; ii < nloc; ii++){
    for (int jj = 0; jj < 3; jj++){
      extend_dcoord[new_idx_map[ii] * 3 + jj] = dcoord[ii * 3 + jj];
      if (atype[ii] < numb_types_spin){
        double temp_dcoord = dcoord[ii * 3 + jj] + spin[ii * 3 + jj] * spin_len;
        extend_dcoord[(new_idx_map[ii] + nloc) * 3 + jj] = temp_dcoord;
      }
    }
  }
  for (int ii = nloc; ii < nall; ii++){
    for (int jj = 0; jj < 3; jj++){
      extend_dcoord[new_idx_map[ii] * 3 + jj] = dcoord[ii * 3 + jj];
      if (atype[ii] < numb_types_spin){
        double temp_dcoord = dcoord[ii * 3 + jj] + spin[ii * 3 + jj] * spin_len;
        extend_dcoord[(new_idx_map[ii] + nghost) * 3 + jj] = temp_dcoord;
      }
    }
  }

  // extend atype
  extend_atype.resize(extend_nall);
  for (int ii = 0; ii < nall; ii++){
    extend_atype[new_idx_map[ii]] = atype[ii];
    if (atype[ii] < numb_types_spin){
      if (ii < nloc){
        extend_atype[new_idx_map[ii] + nloc] = atype[ii] + numb_types_real;
      }
      else{
        extend_atype[new_idx_map[ii] + nghost] = atype[ii] + numb_types_real;
      }
    }
  }

}
