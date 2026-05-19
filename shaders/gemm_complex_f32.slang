#include "gemm_common.slang"

struct ComplexF32
{
    float re;
    float im;
};

ComplexF32 makeComplex(float re, float im)
{
    ComplexF32 z;
    z.re = re;
    z.im = im;
    return z;
}

ComplexF32 cadd(ComplexF32 a, ComplexF32 b)
{
    return makeComplex(a.re + b.re, a.im + b.im);
}

ComplexF32 cmul(ComplexF32 a, ComplexF32 b)
{
    return makeComplex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

ComplexF32 cconj(ComplexF32 a)
{
    return makeComplex(a.re, -a.im);
}

[[vk::binding(0, 0)]] StructuredBuffer<GemmParams> params_buffer;
[[vk::binding(1, 0)]] StructuredBuffer<ComplexF32> a_buffer;
[[vk::binding(2, 0)]] StructuredBuffer<ComplexF32> b_buffer;
[[vk::binding(3, 0)]] RWStructuredBuffer<ComplexF32> c_buffer;

static const uint OUT_TILE = 32;
static const uint WG_M = 16;
static const uint WG_N = 16;
static const uint THREAD_M = 2;
static const uint THREAD_N = 2;
static const uint K_TILE = 16;

groupshared ComplexF32 tile_a[32][16];
groupshared ComplexF32 tile_b[16][32];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
    uint3 dispatch_id : SV_DispatchThreadID,
    uint3 group_id : SV_GroupID,
    uint3 group_thread_id : SV_GroupThreadID)
{
    GemmParams p = params_buffer[0];
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

    ComplexF32 acc[2][2];
    [ForceUnroll]
    for (uint ii = 0; ii < THREAD_M; ii++)
    {
        [ForceUnroll]
        for (uint jj = 0; jj < THREAD_N; jj++)
        {
            acc[ii][jj] = makeComplex(0.0f, 0.0f);
        }
    }

    uint linear_tid = ty * WG_M + tx;
    for (uint kk_base = 0; kk_base < p.k; kk_base += K_TILE)
    {
        [ForceUnroll]
        for (uint load_i = 0; load_i < 2; load_i++)
        {
            uint a_linear = linear_tid + load_i * 256;
            uint a_row_offset = a_linear & 31;
            uint a_k_offset = a_linear >> 5;
            uint a_row = group_id.y * OUT_TILE + a_row_offset;
            uint a_k = kk_base + a_k_offset;
            ComplexF32 av = makeComplex(0.0f, 0.0f);
            if (a_row < p.m && a_k < p.k)
            {
                av = a_buffer[opIndex(p.op_a, a_base, a_row, a_k, p.a_row_stride, p.a_col_stride)];
                if (p.op_a == GEMM_OP_C) av = cconj(av);
            }
            tile_a[a_row_offset][a_k_offset] = av;

            uint b_linear = linear_tid + load_i * 256;
            uint b_k_offset = b_linear & 15;
            uint b_col_offset = b_linear >> 4;
            uint b_k = kk_base + b_k_offset;
            uint b_col = group_id.x * OUT_TILE + b_col_offset;
            ComplexF32 bv = makeComplex(0.0f, 0.0f);
            if (b_k < p.k && b_col < p.n)
            {
                bv = b_buffer[opIndex(p.op_b, b_base, b_k, b_col, p.b_row_stride, p.b_col_stride)];
                if (p.op_b == GEMM_OP_C) bv = cconj(bv);
            }
            tile_b[b_k_offset][b_col_offset] = bv;
        }

        GroupMemoryBarrierWithGroupSync();

        [ForceUnroll]
        for (uint kk = 0; kk < K_TILE; kk++)
        {
            ComplexF32 a_frag[2];
            ComplexF32 b_frag[2];
            [ForceUnroll]
            for (uint ii = 0; ii < THREAD_M; ii++)
            {
                a_frag[ii] = tile_a[tx + ii * WG_M][kk];
            }
            [ForceUnroll]
            for (uint jj = 0; jj < THREAD_N; jj++)
            {
                b_frag[jj] = tile_b[kk][ty + jj * WG_N];
            }
            [ForceUnroll]
            for (uint ii = 0; ii < THREAD_M; ii++)
            {
                [ForceUnroll]
                for (uint jj = 0; jj < THREAD_N; jj++)
                {
                    acc[ii][jj] = cadd(acc[ii][jj], cmul(a_frag[ii], b_frag[jj]));
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    ComplexF32 alpha = makeComplex(p.alpha_re, p.alpha_im);
    ComplexF32 beta = makeComplex(p.beta_re, p.beta_im);
    [ForceUnroll]
    for (uint ii = 0; ii < THREAD_M; ii++)
    {
        uint row = group_id.y * OUT_TILE + tx + ii * WG_M;
        [ForceUnroll]
        for (uint jj = 0; jj < THREAD_N; jj++)
        {
            uint col = group_id.x * OUT_TILE + ty + jj * WG_N;
            if (inBounds(p, row, col, batch))
            {
                uint c_index = matrixIndex(c_base, row, col, p.c_row_stride, p.c_col_stride);
                c_buffer[c_index] = cadd(cmul(alpha, acc[ii][jj]), cmul(beta, c_buffer[c_index]));
            }
        }
    }
}
