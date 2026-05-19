#include "gemm_common.slang"

static const uint GEMM_R_16F = 2;
static const uint GEMM_R_16BF = 14;
static const uint GEMM_R_8F_E4M3 = 28;
static const uint GEMM_R_8F_E5M2 = 29;
static const uint GEMM_R_8F_UE8M0 = 30;
static const uint GEMM_R_6F_E2M3 = 31;
static const uint GEMM_R_6F_E3M2 = 32;
static const uint GEMM_R_4F_E2M1 = 33;

struct GemmParamsLowp
{
    GemmParams base;
    uint a_type;
    uint b_type;
};

[[vk::binding(0, 0)]] StructuredBuffer<GemmParamsLowp> params_buffer;
// One uint lane per logical scalar. Packed CUDA low-precision formats use the low bits.
[[vk::binding(1, 0)]] StructuredBuffer<uint> a_buffer;
[[vk::binding(2, 0)]] StructuredBuffer<uint> b_buffer;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> c_buffer;

static const uint OUT_TILE = 32;
static const uint WG_M = 16;
static const uint WG_N = 16;
static const uint THREAD_M = 2;
static const uint THREAD_N = 2;
static const uint K_TILE = 16;

groupshared float tile_a[32][16];
groupshared float tile_b[16][32];

float pow2i(int exponent)
{
    return exp2(float(exponent));
}

float decodeE4M3(uint raw)
{
    uint sign = raw >> 7;
    uint exponent = (raw >> 3) & 0xfu;
    uint mantissa = raw & 0x7u;
    float value = 0.0f;
    if (exponent == 0)
    {
        value = (float(mantissa) / 8.0f) * pow2i(-6);
    }
    else
    {
        value = (1.0f + float(mantissa) / 8.0f) * pow2i(int(exponent) - 7);
    }
    return sign == 0 ? value : -value;
}

float decodeE5M2(uint raw)
{
    uint sign = raw >> 7;
    uint exponent = (raw >> 2) & 0x1fu;
    uint mantissa = raw & 0x3u;
    float value = 0.0f;
    if (exponent == 0)
    {
        value = (float(mantissa) / 4.0f) * pow2i(-14);
    }
    else
    {
        value = (1.0f + float(mantissa) / 4.0f) * pow2i(int(exponent) - 15);
    }
    return sign == 0 ? value : -value;
}

float decodeTinyFloat(uint raw, uint exponent_bits, uint mantissa_bits, int bias)
{
    uint sign = raw >> (exponent_bits + mantissa_bits);
    uint exponent_mask = (1u << exponent_bits) - 1u;
    uint mantissa_mask = (1u << mantissa_bits) - 1u;
    uint exponent = (raw >> mantissa_bits) & exponent_mask;
    uint mantissa = raw & mantissa_mask;
    float denom = float(1u << mantissa_bits);
    float value = 0.0f;
    if (exponent == 0)
    {
        value = (float(mantissa) / denom) * pow2i(1 - bias);
    }
    else
    {
        value = (1.0f + float(mantissa) / denom) * pow2i(int(exponent) - bias);
    }
    return sign == 0 ? value : -value;
}

float decodeLowp(uint raw, uint data_type)
{
    if (data_type == GEMM_R_16F) return f16tof32(raw & 0xffffu);
    if (data_type == GEMM_R_16BF) return asfloat((raw & 0xffffu) << 16);
    if (data_type == GEMM_R_8F_E4M3) return decodeE4M3(raw & 0xffu);
    if (data_type == GEMM_R_8F_E5M2) return decodeE5M2(raw & 0xffu);
    if (data_type == GEMM_R_8F_UE8M0) return pow2i(int(raw & 0xffu) - 127);
    if (data_type == GEMM_R_6F_E2M3) return decodeTinyFloat(raw & 0x3fu, 2, 3, 1);
    if (data_type == GEMM_R_6F_E3M2) return decodeTinyFloat(raw & 0x3fu, 3, 2, 3);
    if (data_type == GEMM_R_4F_E2M1) return decodeTinyFloat(raw & 0xfu, 2, 1, 1);
    return asfloat(raw);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
    uint3 dispatch_id : SV_DispatchThreadID,
    uint3 group_id : SV_GroupID,
    uint3 group_thread_id : SV_GroupThreadID)
{
    GemmParamsLowp all = params_buffer[0];
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

    float acc[2][2];
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
        for (uint load_i = 0; load_i < 2; load_i++)
        {
            uint a_linear = linear_tid + load_i * 256;
            uint a_row_offset = a_linear & 31;
            uint a_k_offset = a_linear >> 5;
            uint a_row = group_id.y * OUT_TILE + a_row_offset;
            uint a_k = kk_base + a_k_offset;
            tile_a[a_row_offset][a_k_offset] =
                (a_row < p.m && a_k < p.k)
                ? decodeLowp(a_buffer[opIndex(p.op_a, a_base, a_row, a_k, p.a_row_stride, p.a_col_stride)], all.a_type)
                : 0.0f;

            uint b_linear = linear_tid + load_i * 256;
            uint b_k_offset = b_linear & 15;
            uint b_col_offset = b_linear >> 4;
            uint b_k = kk_base + b_k_offset;
            uint b_col = group_id.x * OUT_TILE + b_col_offset;
            tile_b[b_k_offset][b_col_offset] =
                (b_k < p.k && b_col < p.n)
                ? decodeLowp(b_buffer[opIndex(p.op_b, b_base, b_k, b_col, p.b_row_stride, p.b_col_stride)], all.b_type)
                : 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        [ForceUnroll]
        for (uint kk = 0; kk < K_TILE; kk++)
        {
            float a_frag[2];
            float b_frag[2];
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
