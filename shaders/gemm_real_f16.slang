#include "gemm_common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<GemmParams> params_buffer;
[[vk::binding(1, 0)]] StructuredBuffer<half> a_buffer;
[[vk::binding(2, 0)]] StructuredBuffer<half> b_buffer;
[[vk::binding(3, 0)]] RWStructuredBuffer<half> c_buffer;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    GemmParams p = params_buffer[0];
    uint col = id.x;
    uint row = id.y;
    uint batch = id.z;
    if (!inBounds(p, row, col, batch))
    {
        return;
    }

    uint a_base = batchBase(p.a_offset, p.stride_a, batch);
    uint b_base = batchBase(p.b_offset, p.stride_b, batch);
    uint c_base = batchBase(p.c_offset, p.stride_c, batch);
    float sum = 0.0f;
    for (uint kk = 0; kk < p.k; kk++)
    {
        float av = float(a_buffer[opIndex(p.op_a, a_base, row, kk, p.a_row_stride, p.a_col_stride)]);
        float bv = float(b_buffer[opIndex(p.op_b, b_base, kk, col, p.b_row_stride, p.b_col_stride)]);
        sum += av * bv;
    }
    uint c_index = matrixIndex(c_base, row, col, p.c_row_stride, p.c_col_stride);
    c_buffer[c_index] = half(p.alpha_re * sum + p.beta_re * float(c_buffer[c_index]));
}
