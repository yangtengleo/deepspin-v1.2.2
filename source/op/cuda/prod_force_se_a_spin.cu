#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#ifdef HIGH_PREC
    typedef double VALUETYPE;
#else
    typedef float  VALUETYPE;
#endif

#define cudaErrcheck(res) { cudaAssert((res), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"cuda assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__global__ void deriv_wrt_center_atom_se_a_spin(VALUETYPE * force, 
                                                const VALUETYPE * net_deriv,
                                                const VALUETYPE * in_deriv,
                                                const int ndescrpt,
                                                const int nloc_real)
{
    const unsigned int idx = blockIdx.y;
    const unsigned int idy = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idz = threadIdx.y;

    if (idx >= nloc_real) {
        return;
    }
    
    if (idy >= ndescrpt) {
        return;
    }
    
    atomicAdd(force + idx * 3 + idz, -1.0 * net_deriv[idx * ndescrpt + idy] * in_deriv[idx * ndescrpt * 3 + idy * 3 + idz]);
}

__global__ void deriv_wrt_neighbors_se_a_spin ( VALUETYPE * force, 
                                                const VALUETYPE * net_deriv,
                                                const VALUETYPE * in_deriv,
                                                const int * nlist,
                                                const int * atype,
                                                const int nloc,
                                                const int nnei,
                                                const int ndescrpt,
                                                const int ntypes_real,
                                                const int nloc_real,
                                                const int n_a_sel,
                                                const int n_a_shift)
{  
    // idy -> nnei
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y;
    const unsigned int idz = threadIdx.y;
    const unsigned int idw = threadIdx.z;
    
    if (idx >= nloc_real) {
        return;
    }
    // deriv wrt neighbors
    int j_idx = nlist[idx * nnei + idy];
    if (j_idx < 0) {
        return;
    }
    else if (atype[j_idx] >= ntypes_real) {
        return;
    }
    atomicAdd(force + j_idx * 3 + idz, net_deriv[idx * ndescrpt + idy * 4 + idw] * in_deriv[idx * ndescrpt * 3 + (idy * 4 + idw) * 3 + idz]);
}

__global__ void deriv_wrt_spin_neighbors_se_a_spin (VALUETYPE * force, 
                                                    const VALUETYPE * net_deriv,
                                                    const VALUETYPE * in_deriv,
                                                    const int * nlist,
                                                    const int * atype,
                                                    const int nloc,
                                                    const int nnei,
                                                    const int ndescrpt,
                                                    const int ntypes_real,
                                                    const int nloc_real,
                                                    const int nghost_real_atom,
                                                    const int n_a_sel,
                                                    const int n_a_shift)
{  
    // idy -> nnei
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y;
    const unsigned int idz = threadIdx.y;
    const unsigned int idw = threadIdx.z;

    if (idx >= nloc_real) {
        return;
    }
    // deriv wrt neighbors
    int j_idx = nlist[idx * nnei + idy];
    if (j_idx < 0) {
        return;
    }
    else if (atype[j_idx] < ntypes_real) {
        return;
    }
    int j_atom_idx = (j_idx < nloc) ? (j_idx - nloc_real) : (j_idx - nghost_real_atom);
    atomicAdd(force + j_atom_idx * 3 + idz, net_deriv[idx * ndescrpt + idy * 4 + idw] * in_deriv[idx * ndescrpt * 3 + (idy * 4 + idw) * 3 + idz]);
}

__global__ void deriv_wrt_mag_force_se_a_spin ( VALUETYPE * force, 
                                                const VALUETYPE * net_deriv,
                                                const VALUETYPE * in_deriv,
                                                const int * nlist,
                                                const int * atype,
                                                const int nloc,
                                                const int nnei,
                                                const int ndescrpt,
                                                const int ntypes_real,
                                                const int nloc_real,
                                                const int n_a_sel,
                                                const int n_a_shift,
                                                const VALUETYPE * spin_len_ptr,
                                                const VALUETYPE * spin_norm_ptr)
{  
    // idy -> nnei
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y;
    const unsigned int idz = threadIdx.y;
    const unsigned int idw = threadIdx.z;

    // (partial_ri / partial_si) = 0.3 / 1.549
    double coeff = spin_len_ptr[atype[idx]] / spin_norm_ptr[atype[idx]];
    if (idx >= nloc_real) {
        return;
    }
    // deriv wrt neighbors
    int j_idx = nlist[idx * nnei + idy];
    if (j_idx < 0) {
        return;
    }
    else if (atype[j_idx] < ntypes_real) {
        return;
    }
    atomicAdd(force + j_idx * 3 + idz, coeff * net_deriv[idx * ndescrpt + idy * 4 + idw] * in_deriv[idx * ndescrpt * 3 + (idy * 4 + idw) * 3 + idz]);
}


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
                                const VALUETYPE * spin_norm_ptr)
{   
    cudaErrcheck(cudaMemset(force, 0.0, sizeof(VALUETYPE) * nall * 3));
    const int LEN1 = 256;
    const int nblock1 = (ndescrpt + LEN1 -1) / LEN1;
    dim3 grid(nblock1, nloc);
    dim3 thread(LEN1, 3);
    deriv_wrt_center_atom_se_a_spin <<<grid, thread>>> (force, 
                                                        net_deriv, 
                                                        in_deriv, 
                                                        ndescrpt,
                                                        nloc_real);
    const int LEN = 64;
    int nblock = (nloc + LEN -1) / LEN;
    dim3 block_grid(nblock, nnei);
    dim3 thread_grid(LEN, 3, 4);
    deriv_wrt_neighbors_se_a_spin <<<block_grid, thread_grid>>>(force,
                                                                net_deriv, 
                                                                in_deriv, 
                                                                nlist,
                                                                atype, 
                                                                nloc, 
                                                                nnei, 
                                                                ndescrpt,
                                                                ntypes_real,
                                                                nloc_real,
                                                                n_a_sel, 
                                                                n_a_shift);
    deriv_wrt_spin_neighbors_se_a_spin <<<block_grid, thread_grid>>>(force,
                                                                    net_deriv, 
                                                                    in_deriv, 
                                                                    nlist,
                                                                    atype, 
                                                                    nloc, 
                                                                    nnei, 
                                                                    ndescrpt,
                                                                    ntypes_real,
                                                                    nloc_real,
                                                                    nghost_real_atom,
                                                                    n_a_sel, 
                                                                    n_a_shift);
    deriv_wrt_mag_force_se_a_spin <<<block_grid, thread_grid>>>(force,
                                                                net_deriv, 
                                                                in_deriv, 
                                                                nlist,
                                                                atype, 
                                                                nloc, 
                                                                nnei, 
                                                                ndescrpt,
                                                                ntypes_real,
                                                                nloc_real,
                                                                n_a_sel, 
                                                                n_a_shift,
                                                                spin_len_ptr,
                                                                spin_norm_ptr);
}