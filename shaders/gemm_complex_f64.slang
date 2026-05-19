#include "gemm_common.slang"

struct GemmParamsComplexF64
{
    uint op_a;
    uint op_b;
    uint m;
    uint n;
    uint k;
    uint batch_count;

    uint a_offset;
    uint b_offset;
    uint c_offset;
    uint a_row_stride;
    uint a_col_stride;
    uint b_row_stride;
    uint b_col_stride;
    uint c_row_stride;
    uint c_col_stride;
    uint stride_a;
    uint stride_b;
    uint stride_c;

    double alpha_re;
    double alpha_im;
    double beta_re;
    double beta_im;
};

struct ComplexF64
{
    double re;
    double im;
};

ComplexF64 makeComplex(double re, double im)
{
    ComplexF64 z;
    z.re = re;
    z.im = im;
    return z;
}

ComplexF64 cadd(ComplexF64 a, ComplexF64 b)
{
    return makeComplex(a.re + b.re, a.im + b.im);
}

ComplexF64 cmul(ComplexF64 a, ComplexF64 b)
{
    return makeComplex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

ComplexF64 cconj(ComplexF64 a)
{
    return makeComplex(a.re, -a.im);
}

[[vk::binding(0, 0)]] StructuredBuffer<GemmParamsComplexF64> params_buffer;
[[vk::binding(1, 0)]] StructuredBuffer<ComplexF64> a_buffer;
[[vk::binding(2, 0)]] StructuredBuffer<ComplexF64> b_buffer;
[[vk::binding(3, 0)]] RWStructuredBuffer<ComplexF64> c_buffer;

static const uint OUT_TILE = 16;
static const uint WG_M = 16;
static const uint WG_N = 16;
static const uint THREAD_M = 1;
static const uint THREAD_N = 1;
static const uint K_TILE = 16;

groupshared ComplexF64 tile_a[16][16];
groupshared ComplexF64 tile_b[16][16];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
    uint3 dispatch_id : SV_DispatchThreadID,
    uint3 group_id : SV_GroupID,
    uint3 group_thread_id : SV_GroupThreadID)
{
    GemmParamsComplexF64 p = params_buffer[0];
    uint tx = group_thread_id.x;
    uint ty = group_thread_id.y;
    uint batch = dispatch_id.z;
    if (batch >= p.batch_count)
    {
        return;
    }

    uint a_base = batchBase(p.a_offset, p.stride_a, batch);
    uint b_base = batchBase(p.b_offset, p.stride_b, batch);
    uint c_base = batchBase(p.c_offset, p.stride_c, batch);
    ComplexF64 sum = makeComplex(0.0, 0.0);
    uint row = group_id.y * OUT_TILE + tx;
    uint col = group_id.x * OUT_TILE + ty;

    for (uint kk_base = 0; kk_base < p.k; kk_base += K_TILE)
    {
        uint a_k = kk_base + ty;
        ComplexF64 av = makeComplex(0.0, 0.0);
        if (row < p.m && a_k < p.k)
        {
            av = a_buffer[opIndex(p.op_a, a_base, row, a_k, p.a_row_stride, p.a_col_stride)];
            if (p.op_a == GEMM_OP_C) av = cconj(av);
        }
        tile_a[tx][ty] = av;

        uint b_k = kk_base + tx;
        ComplexF64 bv = makeComplex(0.0, 0.0);
        if (b_k < p.k && col < p.n)
        {
            bv = b_buffer[opIndex(p.op_b, b_base, b_k, col, p.b_row_stride, p.b_col_stride)];
            if (p.op_b == GEMM_OP_C) bv = cconj(bv);
        }
        tile_b[tx][ty] = bv;

        GroupMemoryBarrierWithGroupSync();

        [ForceUnroll]
        for (uint kk = 0; kk < K_TILE; kk++)
        {
            sum = cadd(sum, cmul(tile_a[tx][kk], tile_b[kk][ty]));
        }

        GroupMemoryBarrierWithGroupSync();
    }

    ComplexF64 alpha = makeComplex(p.alpha_re, p.alpha_im);
    ComplexF64 beta = makeComplex(p.beta_re, p.beta_im);
    if (row < p.m && col < p.n)
    {
        uint c_index = matrixIndex(c_base, row, col, p.c_row_stride, p.c_col_stride);
        c_buffer[c_index] = cadd(cmul(alpha, sum), cmul(beta, c_buffer[c_index]));
    }
}
