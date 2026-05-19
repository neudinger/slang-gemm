#include "gemm_common.slang"

static const uint GEMM_INT_S4 = 16;
static const uint GEMM_INT_U4 = 18;
static const uint GEMM_INT_S8 = 3;
static const uint GEMM_INT_U8 = 8;
static const uint GEMM_INT_S16 = 20;
static const uint GEMM_INT_U16 = 22;
static const uint GEMM_INT_S32 = 10;
static const uint GEMM_INT_U32 = 12;

struct GemmParamsI32
{
    GemmParams base;
    uint a_type;
    uint b_type;
    uint c_type;
    int alpha;
    int beta;
};

[[vk::binding(0, 0)]] StructuredBuffer<GemmParamsI32> params_buffer;
// One uint lane per logical scalar. Int4/8/16 values use the low bits.
[[vk::binding(1, 0)]] StructuredBuffer<uint> a_buffer;
[[vk::binding(2, 0)]] StructuredBuffer<uint> b_buffer;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> c_buffer;

static const uint OUT_TILE = 32;
static const uint WG_M = 16;
static const uint WG_N = 16;
static const uint THREAD_M = 2;
static const uint THREAD_N = 2;
static const uint K_TILE = 16;

groupshared int tile_a[32][16];
groupshared int tile_b[16][32];

int signExtend(uint value, uint bits)
{
    uint shift = 32 - bits;
    return int(value << shift) >> int(shift);
}

int decodeInt(uint raw, uint data_type)
{
    if (data_type == GEMM_INT_S4) return signExtend(raw & 0xfu, 4);
    if (data_type == GEMM_INT_U4) return int(raw & 0xfu);
    if (data_type == GEMM_INT_S8) return signExtend(raw & 0xffu, 8);
    if (data_type == GEMM_INT_U8) return int(raw & 0xffu);
    if (data_type == GEMM_INT_S16) return signExtend(raw & 0xffffu, 16);
    if (data_type == GEMM_INT_U16) return int(raw & 0xffffu);
    return int(raw);
}

uint encodeInt(int value, uint data_type)
{
    if (data_type == GEMM_INT_S4 || data_type == GEMM_INT_U4) return uint(value) & 0xfu;
    if (data_type == GEMM_INT_S8 || data_type == GEMM_INT_U8) return uint(value) & 0xffu;
    if (data_type == GEMM_INT_S16 || data_type == GEMM_INT_U16) return uint(value) & 0xffffu;
    return uint(value);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
    uint3 dispatch_id : SV_DispatchThreadID,
    uint3 group_id : SV_GroupID,
    uint3 group_thread_id : SV_GroupThreadID)
{
    GemmParamsI32 all = params_buffer[0];
    GemmParams p = all.base;
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

    int acc[2][2];
    [ForceUnroll]
    for (uint ii = 0; ii < THREAD_M; ii++)
    {
        [ForceUnroll]
        for (uint jj = 0; jj < THREAD_N; jj++)
        {
            acc[ii][jj] = 0;
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
            tile_a[a_row_offset][a_k_offset] =
                (a_row < p.m && a_k < p.k)
                ? decodeInt(a_buffer[opIndex(p.op_a, a_base, a_row, a_k, p.a_row_stride, p.a_col_stride)], all.a_type)
                : 0;

            uint b_linear = linear_tid + load_i * 256;
            uint b_k_offset = b_linear & 15;
            uint b_col_offset = b_linear >> 4;
            uint b_k = kk_base + b_k_offset;
            uint b_col = group_id.x * OUT_TILE + b_col_offset;
            tile_b[b_k_offset][b_col_offset] =
                (b_k < p.k && b_col < p.n)
                ? decodeInt(b_buffer[opIndex(p.op_b, b_base, b_k, b_col, p.b_row_stride, p.b_col_stride)], all.b_type)
                : 0;
        }

        GroupMemoryBarrierWithGroupSync();

        [ForceUnroll]
        for (uint kk = 0; kk < K_TILE; kk++)
        {
            int a_frag[2];
            int b_frag[2];
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
                int old = decodeInt(c_buffer[c_index], all.c_type);
                c_buffer[c_index] = encodeInt(all.alpha * acc[ii][jj] + all.beta * old, all.c_type);
            }
        }
    }
}
