#include "gemm_common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<GemmParams> params_buffer;
[[vk::binding(1, 0)]] StructuredBuffer<float> a_buffer;
[[vk::binding(2, 0)]] StructuredBuffer<float> b_buffer;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> c_buffer;

static const uint OUT_TILE = 64;
static const uint WG_M = 16;
static const uint WG_N = 16;
static const uint THREAD_M = 4;
static const uint THREAD_N = 4;
static const uint K_TILE = 16;

groupshared float tile_a[64][16];
groupshared float tile_b[16][64];

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
    uint a_base = batchBase(p.a_offset, p.stride_a, batch);
    uint b_base = batchBase(p.b_offset, p.stride_b, batch);
    uint c_base = batchBase(p.c_offset, p.stride_c, batch);

    float acc[4][4];
    [ForceUnroll]
    for (uint ii = 0; ii < THREAD_M; ii++)
    {
        [ForceUnroll]
        for (uint jj = 0; jj < THREAD_N; jj++)
        {
            acc[ii][jj] = 0.0f;
        }
    }

    uint linear_tid = ty * WG_M + tx;
    for (uint kk_base = 0; kk_base < p.k; kk_base += K_TILE)
    {
        [ForceUnroll]
        for (uint load_i = 0; load_i < 4; load_i++)
        {
            uint a_linear = linear_tid + load_i * 256;
            uint a_row_offset = a_linear & 63;
            uint a_k_offset = a_linear >> 6;
            uint a_row = group_id.y * OUT_TILE + a_row_offset;
            uint a_k = kk_base + a_k_offset;
            tile_a[a_row_offset][a_k_offset] =
                (a_row < p.m && a_k < p.k && batch < p.batch_count)
                ? a_buffer[opIndex(p.op_a, a_base, a_row, a_k, p.a_row_stride, p.a_col_stride)]
                : 0.0f;

            uint b_linear = linear_tid + load_i * 256;
            uint b_k_offset = b_linear & 15;
            uint b_col_offset = b_linear >> 4;
            uint b_k = kk_base + b_k_offset;
            uint b_col = group_id.x * OUT_TILE + b_col_offset;
            tile_b[b_k_offset][b_col_offset] =
                (b_k < p.k && b_col < p.n && batch < p.batch_count)
                ? b_buffer[opIndex(p.op_b, b_base, b_k, b_col, p.b_row_stride, p.b_col_stride)]
                : 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        [ForceUnroll]
        for (uint kk = 0; kk < K_TILE; kk++)
        {
            float a_frag[4];
            float b_frag[4];
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
                    acc[ii][jj] += a_frag[ii] * b_frag[jj];
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

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
                c_buffer[c_index] = p.alpha_re * acc[ii][jj] + p.beta_re * c_buffer[c_index];
            }
        }
    }
}
