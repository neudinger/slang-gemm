using namespace linalg;

#ifndef K_UNROLL
#define K_UNROLL 1
#endif

#ifndef N_TILES
#define N_TILES 1
#endif

struct PushConstants
{
    uint m;
    uint n;
    uint k;
    uint a_stride;
    uint b_stride;
    uint c_stride;
    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_c;
    uint op_a;
    uint op_b;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> a_buffer;
[[vk::binding(1, 0)]] StructuredBuffer<float> b_buffer;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> c_buffer;
[[vk::push_constant]] PushConstants push_constants;

typealias AccMat = CoopMat<float, MemoryScope.Workgroup, 128, 128, CoopMatMatrixUse.MatrixAccumulator>;
typealias AMat = CoopMat<float, MemoryScope.Workgroup, 128, 16, CoopMatMatrixUse.MatrixA>;
typealias BMat = CoopMat<float, MemoryScope.Workgroup, 16, 128, CoopMatMatrixUse.MatrixB>;

TensorLayout<2, CoopMatClampMode.Undefined> tensorLayout(uint dim0, uint dim1, uint stride0, uint stride1)
{
    TensorLayout<2, CoopMatClampMode.Undefined> layout;
    return layout.Dimension(dim0, dim1).Stride(stride0, stride1);
}

AMat loadA(uint a_base, TensorLayout<2, CoopMatClampMode.Undefined> layout_a, uint tile_row, uint kk)
{
    return AMat::Load(a_buffer, a_base, layout_a.Slice(tile_row, 128, kk, 16));
}

BMat loadB(uint b_base, TensorLayout<2, CoopMatClampMode.Undefined> layout_b, uint tile_col, uint kk)
{
    return BMat::Load(b_buffer, b_base, layout_b.Slice(kk, 16, tile_col, 128));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 workgroup_id : SV_GroupID)
{
    uint tile_row = workgroup_id.y * 128;
    uint tile_col = workgroup_id.x * (128 * N_TILES);
    uint a_base = workgroup_id.z * push_constants.batch_stride_a;
    uint b_base = workgroup_id.z * push_constants.batch_stride_b;
    uint c_base = workgroup_id.z * push_constants.batch_stride_c;

    AccMat result;
    result.fill(0.0f);
#if N_TILES >= 2
    AccMat result1;
    result1.fill(0.0f);
#endif
#if N_TILES >= 4
    AccMat result2;
    result2.fill(0.0f);
    AccMat result3;
    result3.fill(0.0f);
#endif

    let layout_a = push_constants.op_a == 0
        ? tensorLayout(push_constants.m, push_constants.k, push_constants.a_stride, 1)
        : tensorLayout(push_constants.m, push_constants.k, 1, push_constants.a_stride);
    let layout_b = push_constants.op_b == 0
        ? tensorLayout(push_constants.k, push_constants.n, push_constants.b_stride, 1)
        : tensorLayout(push_constants.k, push_constants.n, 1, push_constants.b_stride);
    let layout_c = tensorLayout(push_constants.m, push_constants.n, push_constants.c_stride, 1);

    uint kk = 0;

#if K_UNROLL >= 4
    for (; kk + 64 <= push_constants.k; kk += 64)
    {
        let a0 = loadA(a_base, layout_a, tile_row, kk + 0);
        let a1 = loadA(a_base, layout_a, tile_row, kk + 16);
        let a2 = loadA(a_base, layout_a, tile_row, kk + 32);
        let a3 = loadA(a_base, layout_a, tile_row, kk + 48);
        result = coopMatMulAdd<float, false>(a0, loadB(b_base, layout_b, tile_col, kk + 0), result);
        result = coopMatMulAdd<float, false>(a1, loadB(b_base, layout_b, tile_col, kk + 16), result);
        result = coopMatMulAdd<float, false>(a2, loadB(b_base, layout_b, tile_col, kk + 32), result);
        result = coopMatMulAdd<float, false>(a3, loadB(b_base, layout_b, tile_col, kk + 48), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a0, loadB(b_base, layout_b, tile_col + 128, kk + 0), result1);
        result1 = coopMatMulAdd<float, false>(a1, loadB(b_base, layout_b, tile_col + 128, kk + 16), result1);
        result1 = coopMatMulAdd<float, false>(a2, loadB(b_base, layout_b, tile_col + 128, kk + 32), result1);
        result1 = coopMatMulAdd<float, false>(a3, loadB(b_base, layout_b, tile_col + 128, kk + 48), result1);
#endif
#if N_TILES >= 4
        result2 = coopMatMulAdd<float, false>(a0, loadB(b_base, layout_b, tile_col + 256, kk + 0), result2);
        result2 = coopMatMulAdd<float, false>(a1, loadB(b_base, layout_b, tile_col + 256, kk + 16), result2);
        result2 = coopMatMulAdd<float, false>(a2, loadB(b_base, layout_b, tile_col + 256, kk + 32), result2);
        result2 = coopMatMulAdd<float, false>(a3, loadB(b_base, layout_b, tile_col + 256, kk + 48), result2);
        result3 = coopMatMulAdd<float, false>(a0, loadB(b_base, layout_b, tile_col + 384, kk + 0), result3);
        result3 = coopMatMulAdd<float, false>(a1, loadB(b_base, layout_b, tile_col + 384, kk + 16), result3);
        result3 = coopMatMulAdd<float, false>(a2, loadB(b_base, layout_b, tile_col + 384, kk + 32), result3);
        result3 = coopMatMulAdd<float, false>(a3, loadB(b_base, layout_b, tile_col + 384, kk + 48), result3);
#endif
    }
#endif

    for (; kk < push_constants.k; kk += 16)
    {
        let a = loadA(a_base, layout_a, tile_row, kk);
        result = coopMatMulAdd<float, false>(a, loadB(b_base, layout_b, tile_col, kk), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a, loadB(b_base, layout_b, tile_col + 128, kk), result1);
#endif
#if N_TILES >= 4
        result2 = coopMatMulAdd<float, false>(a, loadB(b_base, layout_b, tile_col + 256, kk), result2);
        result3 = coopMatMulAdd<float, false>(a, loadB(b_base, layout_b, tile_col + 384, kk), result3);
#endif
    }

    result.Store(c_buffer, c_base, layout_c.Slice(tile_row, 128, tile_col, 128));
#if N_TILES >= 2
    result1.Store(c_buffer, c_base, layout_c.Slice(tile_row, 128, tile_col + 128, 128));
#endif
#if N_TILES >= 4
    result2.Store(c_buffer, c_base, layout_c.Slice(tile_row, 128, tile_col + 256, 128));
    result3.Store(c_buffer, c_base, layout_c.Slice(tile_row, 128, tile_col + 384, 128));
#endif
}
