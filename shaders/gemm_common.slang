// Shared GEMM descriptor and indexing helpers for scalar correctness kernels.

static const uint GEMM_OP_N = 0;
static const uint GEMM_OP_T = 1;
static const uint GEMM_OP_C = 2;

struct GemmParams
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

    float alpha_re;
    float alpha_im;
    float beta_re;
    float beta_im;
};

uint matrixIndex(uint base, uint row, uint col, uint row_stride, uint col_stride)
{
    return base + row * row_stride + col * col_stride;
}

uint batchBase(uint offset, uint stride, uint batch)
{
    return offset + stride * batch;
}

uint opIndex(uint op, uint base, uint row, uint col, uint row_stride, uint col_stride)
{
    if (op == GEMM_OP_N)
    {
        return matrixIndex(base, row, col, row_stride, col_stride);
    }
    return matrixIndex(base, col, row, row_stride, col_stride);
}

bool inBounds(GemmParams p, uint row, uint col, uint batch)
{
    return row < p.m && col < p.n && batch < p.batch_count;
}
