#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace tensorflow;
using namespace std;

#define cudaErrcheck(res) { cudaAssert((res), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"cuda assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

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

void ProdForceSeASpinLauncher ( VALUETYPE * force, 
                                const VALUETYPE * net_deriv,
                                const VALUETYPE * in_deriv,
                                const int * nlist,
                                const int * atype,
                                const int nloc,
                                const int nall,
                                const int ndescrpt,
                                const int nnei,
                                const int ntypes_real,
                                const int nloc_real,
                                const int nghost_real_atom,
                                const int n_a_sel,
                                const int n_a_shift,
                                const VALUETYPE * spin_len_ptr,
                                const VALUETYPE * spin_norm_ptr);

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
        int * natoms = new int[natoms_tensor.shape().dim_size(0)];
        cudaErrcheck(cudaMemcpy(natoms, natoms_tensor.flat<int>().data(), sizeof(int) * natoms_tensor.shape().dim_size(0), cudaMemcpyDeviceToHost));
        int nloc = natoms[0];
        int nall = natoms[1];
        int nframes = net_deriv_tensor.shape().dim_size(0);
        int ndescrpt = net_deriv_tensor.shape().dim_size(1) / nloc;
        int nnei = nlist_tensor.shape().dim_size(1) / nloc;
        int nloc_real = 0;
        int ntypes_real = use_spin.size();
        for (int i = 0; i < ntypes_real; ++i){
            nloc_real += natoms[2 + i];
        }
        VALUETYPE * spin_len_ptr;
        VALUETYPE * spin_norm_ptr;
        cudaMalloc((void**)& spin_len_ptr, sizeof(VALUETYPE) * spin_len.size());
        cudaMalloc((void**)& spin_norm_ptr, sizeof(VALUETYPE) * spin_norm.size());
        cudaErrcheck(cudaMemcpy(spin_len_ptr, &spin_len[0], sizeof(VALUETYPE) * spin_len.size(), cudaMemcpyHostToDevice));
        cudaErrcheck(cudaMemcpy(spin_norm_ptr, &spin_norm[0], sizeof(VALUETYPE) * spin_norm.size(), cudaMemcpyHostToDevice));

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
        OP_REQUIRES_OK (context, context->allocate_output(context_output_index++,
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

        for (int II = 0; II < nframes; II++) {
            int nghost_real_atom = 0;
            int * dtype = new int[nall];
            cudaErrcheck(cudaMemcpy(dtype, atype_tensor.flat<int>().data() + II * nall, sizeof(int) * nall, cudaMemcpyDeviceToHost));
            if (nloc < nall){
                for (int ii = nloc; ii < nall; ++ii){
                    nghost_real_atom += (dtype[ii] < ntypes_real) ? 1 : 0;
                }
            }
            delete[] dtype;
            ProdForceSeASpinLauncher(force_tensor->flat<VALUETYPE>().data() + II * (nall * 3),
                                    net_deriv_tensor.flat<VALUETYPE>().data() + II * (nloc * ndescrpt),
                                    in_deriv_tensor.flat<VALUETYPE>().data() + II * (nloc * ndescrpt * 3),
                                    nlist_tensor.flat<int>().data() + II * (nloc * nnei),
                                    atype_tensor.flat<int>().data() + II * (nall),
                                    nloc,
                                    nall, 
                                    ndescrpt,
                                    nnei,
                                    ntypes_real,
                                    nloc_real,
                                    nghost_real_atom,
                                    n_a_sel,
                                    n_a_shift,
                                    spin_len_ptr,
                                    spin_norm_ptr);
        }
        delete[] natoms;
        delete[] spin_len_ptr;
        delete[] spin_norm_ptr;
    }
private:
    int n_r_sel, n_a_sel, n_a_shift;
    std::vector<bool> use_spin;
    std::vector<float> spin_len;
    std::vector<float> spin_norm;
};

REGISTER_KERNEL_BUILDER(Name("ProdForceSeASpin").Device(DEVICE_GPU), ProdForceSeASpinOp);