#include "./layer_norm/ln.h"
#include "./layer_norm/ln_utils.cuh"
#include "./layer_norm/ln_kernel_traits.h"
using namespace layer_norm;
template <typename T, int N>
struct Tensor {
  __device__ T& operator[](int ind) {
    return data[ind];
  };

  T* data;
  int size[N];
  int stride[N];
};

__global__ void kernel1(Tensor<__half, 2> T0, Tensor<__half, 1> T1, Tensor<__half, 1> T2, Tensor<__half, 2> T20, Tensor<float, 2> T29, Tensor<float, 2> T31)
{
    FwdParams params;
    params.mu = &T29[0];
    params.rs = &T31[0];
    params.gamma = &T1[0];
    params.beta = &T2[0];
    params.x = &T0[0];

    using Ktraits = Kernel_traits<__half,
                                        __half,
                                        __half,
                                        float,
                                        int,
                                        10240,
                                        1,
                                        1,
                                        4,
                                        16
                                        >;
    enum { ROWS_PER_CTA = Ktraits::ROWS_PER_CTA };
    enum { WARPS_N = Ktraits::WARPS_N };
    enum { WARPS_M = Ktraits::WARPS_M };
    enum { THREADS_PER_ROW = Ktraits::THREADS_PER_ROW };
    enum { VEC_COLS_PER_LDG = Ktraits::VEC_COLS_PER_LDG };
    enum { BYTES_PER_ROW = Ktraits::BYTES_PER_ROW };
    enum { LDGS = Ktraits::LDGS };
    enum { NUM_ELTS = Ktraits::NUM_ELTS };
    enum { CTAS_PER_ROW = Ktraits::CTAS_PER_ROW };

    using output_t = typename Ktraits::output_t;
    using index_t = typename Ktraits::index_t;
    using compute_t = typename Ktraits::compute_t;
    using Ivec = typename Ktraits::Ivec;
    using Ovec = typename Ktraits::Ovec;
    using Wvec = typename Ktraits::Wvec;
    using Cvec = typename Ktraits::Cvec;

    using Stats = typename Ktraits::Stats;
    using stats_t = typename Stats::stats_t;

    extern __shared__ char smem_[];

    const index_t tidx = threadIdx.x;
    const index_t bidn = blockIdx.x % CTAS_PER_ROW;
    const index_t bidm = blockIdx.x / CTAS_PER_ROW;
    const index_t lane = tidx % THREADS_PER_WARP;
    const index_t warp = tidx / THREADS_PER_WARP;
    const index_t warp_m = warp / WARPS_N;
    const index_t warp_n = warp % WARPS_N;

    const index_t r = bidm * ROWS_PER_CTA + warp_m;
    const index_t c = bidn * THREADS_PER_ROW + warp_n * THREADS_PER_WARP + lane;

    Stats stats(params, bidm, bidn, warp_m, warp_n, lane, smem_);

    compute_t *mu_ptr = static_cast<compute_t *>(params.mu);
    compute_t *rs_ptr = static_cast<compute_t *>(params.rs);

    Wvec gamma[LDGS];
    Wvec beta[LDGS];
    index_t idx = c;
    #pragma unroll
    for( int it = 0; it < LDGS; it++ ) {
        gamma[it].load_from(params.gamma, idx);
        beta[it].load_from(params.beta, idx);
        idx += VEC_COLS_PER_LDG;
    }

    constexpr compute_t rn = 1.f / compute_t(Ktraits::COLS);

    for( int row = r; row < params.rows; row += params.ctas_per_col * ROWS_PER_CTA ) {
        Ivec x[LDGS];
        index_t idx = row * Ktraits::VEC_COLS + c;
        compute_t xf[LDGS * NUM_ELTS];
        #pragma unroll
        for( int it = 0; it < LDGS; it++ ) {
            x[it].load_from(params.x, idx);
            #pragma unroll
            for( int jt = 0; jt < NUM_ELTS; jt++ ) {
                compute_t x_ij = compute_t(x[it].data.elt[jt]);
                xf[it * NUM_ELTS + jt] =  x_ij;
            }
            idx += VEC_COLS_PER_LDG;
        }

        stats_t s = stats.compute(xf, rn);

        compute_t mu = layer_norm::Get<0>::of<stats_t, compute_t>(s);
        compute_t m2 = layer_norm::Get<1>::of<stats_t, compute_t>(s);

        if( bidn == 0 && warp_n == 0 && lane == 0 ) {
            mu_ptr[row] = mu;
        }

        compute_t rs = rsqrtf(rn * m2 + params.epsilon);

        if( bidn == 0 && warp_n == 0 && lane == 0 ) {
            rs_ptr[row] = rs;
        }

        Ovec z[LDGS];
        idx = row * Ktraits::VEC_COLS + c;
        #pragma unroll
        for( int it = 0; it < LDGS; it++ ) {
            #pragma unroll
            for( int jt = 0; jt < NUM_ELTS; jt++ ) {
                output_t y_ij = output_t(rs * (xf[it * NUM_ELTS + jt] - mu));
                output_t g_ij = gamma[it].data.elt[jt];
                output_t b_ij = beta[it].data.elt[jt];
                z[it].data.elt[jt] = (g_ij * y_ij + b_ij);
            }
            z[it].store_to(params.z, idx);
            idx += VEC_COLS_PER_LDG;
        }

    }
}
// Create forward launch function and register. Macro signature:
// HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG
// REGISTER_FWD_LAUNCHER(10240, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
