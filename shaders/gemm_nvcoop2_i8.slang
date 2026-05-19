using namespace linalg;

#ifndef K_UNROLL
#define K_UNROLL 4
#endif

#ifndef N_TILES
#define N_TILES 2
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

[[vk::binding(0, 0)]] StructuredBuffer<int8_t> a_buffer;
[[vk::binding(1, 0)]] StructuredBuffer<int8_t> b_buffer;
[[vk::binding(2, 0)]] RWStructuredBuffer<int> c_buffer;
[[vk::push_constant]] PushConstants push_constants;

typealias AccMat = CoopMat<int, MemoryScope.Workgroup, 128, 128, CoopMatMatrixUse.MatrixAccumulator>;
typealias AMat = CoopMat<int8_t, MemoryScope.Workgroup, 128, 32, CoopMatMatrixUse.MatrixA>;
typealias BMat = CoopMat<int8_t, MemoryScope.Workgroup, 32, 128, CoopMatMatrixUse.MatrixB>;

TensorLayout<2, CoopMatClampMode.Undefined> tensorLayout(uint dim0, uint dim1, uint stride0, uint stride1)
{
    TensorLayout<2, CoopMatClampMode.Undefined> layout;
    return layout.Dimension(dim0, dim1).Stride(stride0, stride1);
}

AMat loadA(uint a_base, TensorLayout<2, CoopMatClampMode.Undefined> layout_a, uint tile_row, uint kk)
{
    let a_slice = layout_a.Slice(tile_row, 128, kk, 32);
    return AMat::Load(a_buffer, a_base, a_slice);
}

BMat loadB(uint b_base, TensorLayout<2, CoopMatClampMode.Undefined> layout_b, uint tile_col, uint kk)
{
    let b_slice = layout_b.Slice(kk, 32, tile_col, 128);
    return BMat::Load(b_buffer, b_base, b_slice);
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
    result.fill(0);
#if N_TILES >= 2
    AccMat result1;
    result1.fill(0);
#endif

    let layout_a = tensorLayout(push_constants.m, push_constants.k, push_constants.a_stride, 1);
    let layout_b = tensorLayout(push_constants.k, push_constants.n, push_constants.b_stride, 1);
    let layout_c = tensorLayout(push_constants.m, push_constants.n, push_constants.c_stride, 1);

    uint kk = 0;
#if K_UNROLL >= 4
    for (; kk + 128 <= push_constants.k; kk += 128)
    {
        let a0 = loadA(a_base, layout_a, tile_row, kk + 0);
        result = coopMatMulAdd<int, false>(a0, loadB(b_base, layout_b, tile_col, kk + 0), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<int, false>(a0, loadB(b_base, layout_b, tile_col + 128, kk + 0), result1);
#endif
        let a1 = loadA(a_base, layout_a, tile_row, kk + 32);
        result = coopMatMulAdd<int, false>(a1, loadB(b_base, layout_b, tile_col, kk + 32), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<int, false>(a1, loadB(b_base, layout_b, tile_col + 128, kk + 32), result1);
#endif
        let a2 = loadA(a_base, layout_a, tile_row, kk + 64);
        result = coopMatMulAdd<int, false>(a2, loadB(b_base, layout_b, tile_col, kk + 64), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<int, false>(a2, loadB(b_base, layout_b, tile_col + 128, kk + 64), result1);
#endif
        let a3 = loadA(a_base, layout_a, tile_row, kk + 96);
        result = coopMatMulAdd<int, false>(a3, loadB(b_base, layout_b, tile_col, kk + 96), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<int, false>(a3, loadB(b_base, layout_b, tile_col + 128, kk + 96), result1);
#endif
    }
#endif

    for (; kk < push_constants.k; kk += 32)
    {
        let a_tail = loadA(a_base, layout_a, tile_row, kk);
        result = coopMatMulAdd<int, false>(a_tail, loadB(b_base, layout_b, tile_col, kk), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<int, false>(a_tail, loadB(b_base, layout_b, tile_col + 128, kk), result1);
#endif
    }

    let c_slice = layout_c.Slice(tile_row, 128, tile_col, 128);
    result.Store(c_buffer, c_base, c_slice);
#if N_TILES >= 2
    let c_slice1 = layout_c.Slice(tile_row, 128, tile_col + 128, 128);
    result1.Store(c_buffer, c_base, c_slice1);
#endif
}
