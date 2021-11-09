#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>

using namespace tensorflow;
using namespace std;

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else
typedef float  VALUETYPE;
#endif

#ifdef HIGH_PREC
REGISTER_OP("ProdForceSeASpin")
.Input("net_deriv: double")
.Input("in_deriv: double")
.Input("nlist: int32")
.Input("natoms: int32")
.Input("atype: int32")
.Attr("n_a_sel: int")
.Attr("n_r_sel: int")
.Attr("use_spin: list(bool)")
.Attr("spin_len: list(double)")
.Attr("spin_norm: list(double)")
.Output("force: double");
#else
REGISTER_OP("ProdForceSeASpin")
.Input("net_deriv: float")
.Input("in_deriv: float")
.Input("nlist: int32")
.Input("natoms: int32")
.Input("atype: int32")
.Attr("n_a_sel: int")
.Attr("n_r_sel: int")
.Attr("use_spin: list(bool)")
.Attr("spin_len: list(float)")
.Attr("spin_norm: list(float)")
.Output("force: float");
#endif

using namespace tensorflow;

class ProdForceSeASpinOp : public OpKernel {
public:
  explicit ProdForceSeASpinOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_a_sel", &n_a_sel));
    OP_REQUIRES_OK(context, context->GetAttr("n_r_sel", &n_r_sel));
    OP_REQUIRES_OK(context, context->GetAttr("use_spin", &use_spin));
    OP_REQUIRES_OK(context, context->GetAttr("spin_len", &spin_len));
    OP_REQUIRES_OK(context, context->GetAttr("spin_norm", &spin_norm));
    n_a_shift = n_a_sel * 4;
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& net_deriv_tensor	= context->input(context_input_index++);
    const Tensor& in_deriv_tensor	= context->input(context_input_index++);
    const Tensor& nlist_tensor		= context->input(context_input_index++);
    const Tensor& natoms_tensor		= context->input(context_input_index++);
    const Tensor& atype_tensor		= context->input(context_input_index++);

    // set size of the sample
    OP_REQUIRES (context, (net_deriv_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of net deriv should be 2"));
    OP_REQUIRES (context, (in_deriv_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of input deriv should be 2"));
    OP_REQUIRES (context, (nlist_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of nlist should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),		errors::InvalidArgument ("Dim of natoms should be 1"));
    OP_REQUIRES (context, (atype_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of atype should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3),	errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
    auto natoms	= natoms_tensor.flat<int>();

    int nframes = net_deriv_tensor.shape().dim_size(0);
    int nloc = natoms(0);
    int nall = natoms(1);
    int ndescrpt = net_deriv_tensor.shape().dim_size(1) / nloc;
    int nnei = nlist_tensor.shape().dim_size(1) / nloc;
    int nloc_real = 0;
    int ntypes_real = use_spin.size();
    for (int i = 0; i < ntypes_real; ++i){
      nloc_real += natoms(2 + i);
    }
    // check the sizes
    OP_REQUIRES (context, (nframes == in_deriv_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nframes == nlist_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nframes == atype_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nall == atype_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of atoms should match"));
    OP_REQUIRES (context, (nloc * ndescrpt * 3 == in_deriv_tensor.shape().dim_size(1)), errors::InvalidArgument ("number of descriptors should match"));
    OP_REQUIRES (context, (nnei == n_a_sel + n_r_sel),				errors::InvalidArgument ("number of neighbors should match"));
    OP_REQUIRES (context, (0 == n_r_sel),					errors::InvalidArgument ("Rotational free only support all-angular information"));

    // Create an output tensor
    TensorShape force_shape ;
    force_shape.AddDim (nframes);
    force_shape.AddDim (3 * nall);
    Tensor* force_tensor = NULL;
    int context_output_index = 0;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                  force_shape, &force_tensor));
    
    // flat the tensors
    auto net_deriv = net_deriv_tensor.flat<VALUETYPE>();
    auto in_deriv = in_deriv_tensor.flat<VALUETYPE>();
    auto nlist = nlist_tensor.flat<int>();
    auto atype = atype_tensor.matrix<int>();
    auto force = force_tensor->flat<VALUETYPE>();

    assert (nframes == force_shape.dim_size(0));
    assert (nframes == net_deriv_tensor.shape().dim_size(0));
    assert (nframes == in_deriv_tensor.shape().dim_size(0));
    assert (nframes == nlist_tensor.shape().dim_size(0));
    assert (nall * 3 == force_shape.dim_size(1));
    assert (nloc * ndescrpt == net_deriv_tensor.shape().dim_size(1));
    assert (nloc * ndescrpt * 3 == in_deriv_tensor.shape().dim_size(1));
    assert (nloc * nnei == nlist_tensor.shape().dim_size(1));
    assert (nnei * 4 == ndescrpt);	    
    
    // loop over samples
#pragma omp parallel for
    for (int kk = 0; kk < nframes; ++kk){
      int force_iter	= kk * nall * 3;
      int net_iter	= kk * nloc * ndescrpt;
      int in_iter	= kk * nloc * ndescrpt * 3;
      int nlist_iter	= kk * nloc * nnei;

      vector<int> dtype (nall);
      for (int ii = 0; ii < nall; ++ii){
        dtype[ii] = atype(kk, ii);
      }

      int nghost_real_atom = 0;
      if (nloc < nall){
        for (int ii = nloc; ii < nall; ++ii){
          nghost_real_atom += (dtype[ii] < ntypes_real) ? 1 : 0;
        }
      }
      
      for (int ii = 0; ii < nall; ++ii){
        int i_idx = ii;
        force (force_iter + i_idx * 3 + 0) = 0;
        force (force_iter + i_idx * 3 + 1) = 0;
        force (force_iter + i_idx * 3 + 2) = 0;
      }

      // compute force of a frame
      for (int ii = 0; ii < nloc_real; ++ii){
        int i_idx = ii;	
        // deriv wrt center atom
        for (int aa = 0; aa < ndescrpt; ++aa){
	        force (force_iter + i_idx * 3 + 0) -= net_deriv (net_iter + i_idx * ndescrpt + aa) * in_deriv (in_iter + i_idx * ndescrpt * 3 + aa * 3 + 0);
          force (force_iter + i_idx * 3 + 1) -= net_deriv (net_iter + i_idx * ndescrpt + aa) * in_deriv (in_iter + i_idx * ndescrpt * 3 + aa * 3 + 1);
          force (force_iter + i_idx * 3 + 2) -= net_deriv (net_iter + i_idx * ndescrpt + aa) * in_deriv (in_iter + i_idx * ndescrpt * 3 + aa * 3 + 2);
        }
        // deriv wrt neighbors
        for (int jj = 0; jj < nnei; ++jj){
          int j_idx = nlist (nlist_iter + i_idx * nnei + jj);
          if (j_idx < 0) continue;
          else if (dtype[j_idx] >= ntypes_real) continue;
          int aa_start, aa_end;
          make_descript_range (aa_start, aa_end, jj);
          for (int aa = aa_start; aa < aa_end; ++aa) {
            force (force_iter + j_idx * 3 + 0) += net_deriv (net_iter + i_idx * ndescrpt + aa) * in_deriv (in_iter + i_idx * ndescrpt * 3 + aa * 3 + 0);
	          force (force_iter + j_idx * 3 + 1) += net_deriv (net_iter + i_idx * ndescrpt + aa) * in_deriv (in_iter + i_idx * ndescrpt * 3 + aa * 3 + 1);
	          force (force_iter + j_idx * 3 + 2) += net_deriv (net_iter + i_idx * ndescrpt + aa) * in_deriv (in_iter + i_idx * ndescrpt * 3 + aa * 3 + 2);
          }
        }
        // deriv wrt spin neighbors
        for (int jj = 0; jj < nnei; ++jj){
          int j_idx = nlist (nlist_iter + i_idx * nnei + jj);
          if (j_idx < 0) continue;
          else if (dtype[j_idx] < ntypes_real) continue;
          int j_atom_idx = (j_idx < nloc) ? (j_idx - nloc_real) : (j_idx - nghost_real_atom);
          int aa_start, aa_end;
          make_descript_range (aa_start, aa_end, jj);
          for (int aa = aa_start; aa < aa_end; ++aa) {
	          force (force_iter + j_atom_idx * 3 + 0) += net_deriv (net_iter + i_idx * ndescrpt + aa) * in_deriv (in_iter + i_idx * ndescrpt * 3 + aa * 3 + 0);
	          force (force_iter + j_atom_idx * 3 + 1) += net_deriv (net_iter + i_idx * ndescrpt + aa) * in_deriv (in_iter + i_idx * ndescrpt * 3 + aa * 3 + 1);
	          force (force_iter + j_atom_idx * 3 + 2) += net_deriv (net_iter + i_idx * ndescrpt + aa) * in_deriv (in_iter + i_idx * ndescrpt * 3 + aa * 3 + 2);
          }
        }

        // deriv magnetic force
        const double coeff = spin_len[dtype[ii]] / spin_norm[dtype[ii]];  // (partial_ri / partial_si) = 0.3 / 1.549
        for (int jj = 0; jj < nnei; ++jj){
          int j_idx = nlist (nlist_iter + i_idx * nnei + jj);
          if (j_idx < 0) continue;
          else if (dtype[j_idx] < ntypes_real) continue;
          int aa_start, aa_end;
          make_descript_range (aa_start, aa_end, jj); 
          for (int aa = aa_start; aa < aa_end; ++aa) {
            force (force_iter + j_idx * 3 + 0) += coeff * net_deriv (net_iter + i_idx * ndescrpt + aa) * in_deriv (in_iter + i_idx * ndescrpt * 3 + aa * 3 + 0);
            force (force_iter + j_idx * 3 + 1) += coeff * net_deriv (net_iter + i_idx * ndescrpt + aa) * in_deriv (in_iter + i_idx * ndescrpt * 3 + aa * 3 + 1);
            force (force_iter + j_idx * 3 + 2) += coeff * net_deriv (net_iter + i_idx * ndescrpt + aa) * in_deriv (in_iter + i_idx * ndescrpt * 3 + aa * 3 + 2);
          }
        }
      }
      
    }
  }
private:
  int n_r_sel, n_a_sel, n_a_shift;
  vector<bool> use_spin;
  vector<float> spin_len;
  vector<float> spin_norm;
  
  inline void
  make_descript_range (int & idx_start,
		       int & idx_end,
		       const int & nei_idx) {
    if (nei_idx < n_a_sel) {
      idx_start = nei_idx * 4;
      idx_end   = nei_idx * 4 + 4;
    }
    else {
      idx_start = n_a_shift + (nei_idx - n_a_sel);
      idx_end   = n_a_shift + (nei_idx - n_a_sel) + 1;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("ProdForceSeASpin").Device(DEVICE_CPU), ProdForceSeASpinOp);



