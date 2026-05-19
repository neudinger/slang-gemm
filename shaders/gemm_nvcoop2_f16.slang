// Slang NV cooperative-matrix matmul shader.
//
// What this shader computes:
//   C = A * B
//
// where:
//   A is an m x k matrix stored as float16 (`half`)
//   B is a k x n matrix stored as float16 (`half`)
//   C is an m x n matrix stored as float32 (`float`)
//
// One Vulkan compute workgroup computes one or more output tiles:
//   N_TILES=1: one 128x128 tile of C
//   N_TILES=2: two neighboring 128x128 tiles, i.e. one 128x256 tile of C
//
// The important idea is that a cooperative matrix is not owned by one thread.
// It is a matrix fragment distributed across the whole GPU workgroup. Slang's
// `CoopMat` operations compile to SPIR-V cooperative-matrix instructions, and
// the driver maps those to NVIDIA tensor-core-like hardware instructions.
//
// Useful references:
//   Slang CoopMat docs:
//     https://docs.shader-slang.org/en/latest/external/core-module-reference/types/coopmat-04/index.html
//   Vulkan VK_NV_cooperative_matrix2 proposal:
//     https://docs.vulkan.org/features/latest/features/proposals/VK_NV_cooperative_matrix2.html
//   Slang source is built by:
//     //shaders:matmul_nvcoop2_square_f16_slang_* in shaders/BUILD.bazel
//   Host-side mode selection and tile-size checks live in:
//     src/main.zig

// The cooperative-matrix types and `coopMatMulAdd` helper are in Slang's
// `linalg` namespace.
using namespace linalg;

// K_UNROLL controls how many 16-wide K slices are explicitly processed inside
// each loop iteration. A larger value exposes more independent work to the
// compiler, but can also increase register pressure.
#ifndef K_UNROLL
#define K_UNROLL 1
#endif

// N_TILES controls how many adjacent 128-wide N tiles one workgroup computes.
// The default keeps the original 128x128 behavior. The optimized
// `vk_matmul_slang_n2_f32_u4` target builds this file with N_TILES=2 and
// K_UNROLL=4.
#ifndef N_TILES
#define N_TILES 1
#endif

// Push constants are tiny values written by the Vulkan host before dispatch.
// They describe the full matrix dimensions and row strides. The shader uses
// them to map each workgroup to a rectangular slice of A, B, and C.
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

// Descriptor bindings:
//   binding 0: input matrix A, row-major half values
//   binding 1: input matrix B, row-major half values
//   binding 2: output matrix C, row-major float values
[[vk::binding(0, 0)]] StructuredBuffer<half> a_buffer;
[[vk::binding(1, 0)]] StructuredBuffer<half> b_buffer;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> c_buffer;
[[vk::push_constant]] PushConstants push_constants;

// Cooperative-matrix fragment shapes:
//
//   AMat:    128 rows x 16 columns, read from A
//   BMat:     16 rows x 128 columns, read from B
//   AccMat: 128 rows x 128 columns, accumulated into C
//
// The multiplication step is:
//   AccMat += AMat * BMat
//
// K advances by 16 each time because AMat and BMat meet on a 16-element K
// dimension. Accumulation is float32 even though A and B are float16.
typealias AccMat = CoopMat<float, MemoryScope.Workgroup, 128, 128, CoopMatMatrixUse.MatrixAccumulator>;
typealias AMat = CoopMat<half, MemoryScope.Workgroup, 128, 16, CoopMatMatrixUse.MatrixA>;
typealias BMat = CoopMat<half, MemoryScope.Workgroup, 16, 128, CoopMatMatrixUse.MatrixB>;

// Build a Slang tensor layout object that describes a 2D row-major matrix.
// `dim0` and `dim1` are logical shape; `stride0` and `stride1` tell Slang how
// to move in memory. For row-major matrices, moving one row advances by the row
// stride and moving one column advances by 1 element.
TensorLayout<2, CoopMatClampMode.Undefined> tensorLayout(uint dim0, uint dim1, uint stride0, uint stride1)
{
    TensorLayout<2, CoopMatClampMode.Undefined> layout;
    return layout.Dimension(dim0, dim1).Stride(stride0, stride1);
}

// Load the 128x16 slice of A for output tile row `tile_row` and K offset `kk`.
AMat loadA(uint a_base, TensorLayout<2, CoopMatClampMode.Undefined> layout_a, uint tile_row, uint kk)
{
    let a_slice = layout_a.Slice(tile_row, 128, kk, 16);
    return AMat::Load(a_buffer, a_base, a_slice);
}

// Load the 16x128 slice of B for output tile column `tile_col` and K offset
// `kk`.
BMat loadB(uint b_base, TensorLayout<2, CoopMatClampMode.Undefined> layout_b, uint tile_col, uint kk)
{
    let b_slice = layout_b.Slice(kk, 16, tile_col, 128);
    return BMat::Load(b_buffer, b_base, b_slice);
}

// Vulkan compute entry point.
//
// The workgroup has 256 threads. With MemoryScope.Workgroup cooperative
// matrices, those 256 threads cooperate on each large matrix fragment.
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 workgroup_id : SV_GroupID)
{
    // `workgroup_id.y` chooses the 128-row band of C.
    // `workgroup_id.x` chooses the 128-column band, widened by N_TILES.
    // `workgroup_id.z` chooses the strided batch.
    uint tile_row = workgroup_id.y * 128;
    uint tile_col = workgroup_id.x * (128 * N_TILES);
    uint a_base = workgroup_id.z * push_constants.batch_stride_a;
    uint b_base = workgroup_id.z * push_constants.batch_stride_b;
    uint c_base = workgroup_id.z * push_constants.batch_stride_c;

    // First accumulator: C[tile_row : tile_row+128,
    //                    tile_col : tile_col+128]
    AccMat result;
    result.fill(0.0f);
#if N_TILES >= 2
    // Optional second accumulator for the neighboring N tile:
    // C[tile_row : tile_row+128, tile_col+128 : tile_col+256].
    // It reuses the same A loads as `result`, but loads a second B tile.
    AccMat result1;
    result1.fill(0.0f);
#endif
#if N_TILES >= 4
    AccMat result2;
    result2.fill(0.0f);
    AccMat result3;
    result3.fill(0.0f);
#endif

    // Tell Slang how the flat buffers should be interpreted as 2D tensors.
    // For transpose/conjugate-transpose real inputs, the stored matrix is
    // transposed and the logical op(A)/op(B) view swaps row/column strides.
    let layout_a = push_constants.op_a == 0
        ? tensorLayout(push_constants.m, push_constants.k, push_constants.a_stride, 1)
        : tensorLayout(push_constants.m, push_constants.k, 1, push_constants.a_stride);
    let layout_b = push_constants.op_b == 0
        ? tensorLayout(push_constants.k, push_constants.n, push_constants.b_stride, 1)
        : tensorLayout(push_constants.k, push_constants.n, 1, push_constants.b_stride);
    let layout_c = tensorLayout(push_constants.m, push_constants.n, push_constants.c_stride, 1);

    uint kk = 0;

#if N_TILES == 1
// Original single-tile path. This intentionally keeps the original code shape
// for N_TILES=1 targets, so the new optimized path does not perturb the older
// Slang variants.
#if K_UNROLL >= 8
    // Process eight 16-wide K slices per loop iteration: 8 * 16 = 128.
    for (; kk + 128 <= push_constants.k; kk += 128)
    {
        let a0 = loadA(a_base, layout_a, tile_row, kk + 0);
        let b0 = loadB(b_base, layout_b, tile_col, kk + 0);
        let a1 = loadA(a_base, layout_a, tile_row, kk + 16);
        let b1 = loadB(b_base, layout_b, tile_col, kk + 16);
        let a2 = loadA(a_base, layout_a, tile_row, kk + 32);
        let b2 = loadB(b_base, layout_b, tile_col, kk + 32);
        let a3 = loadA(a_base, layout_a, tile_row, kk + 48);
        let b3 = loadB(b_base, layout_b, tile_col, kk + 48);
        let a4 = loadA(a_base, layout_a, tile_row, kk + 64);
        let b4 = loadB(b_base, layout_b, tile_col, kk + 64);
        let a5 = loadA(a_base, layout_a, tile_row, kk + 80);
        let b5 = loadB(b_base, layout_b, tile_col, kk + 80);
        let a6 = loadA(a_base, layout_a, tile_row, kk + 96);
        let b6 = loadB(b_base, layout_b, tile_col, kk + 96);
        let a7 = loadA(a_base, layout_a, tile_row, kk + 112);
        let b7 = loadB(b_base, layout_b, tile_col, kk + 112);
        result = coopMatMulAdd<float, false>(a0, b0, result);
        result = coopMatMulAdd<float, false>(a1, b1, result);
        result = coopMatMulAdd<float, false>(a2, b2, result);
        result = coopMatMulAdd<float, false>(a3, b3, result);
        result = coopMatMulAdd<float, false>(a4, b4, result);
        result = coopMatMulAdd<float, false>(a5, b5, result);
        result = coopMatMulAdd<float, false>(a6, b6, result);
        result = coopMatMulAdd<float, false>(a7, b7, result);
    }
#elif K_UNROLL >= 4
    // Process four 16-wide K slices per loop iteration: 4 * 16 = 64.
    for (; kk + 64 <= push_constants.k; kk += 64)
    {
        let a0 = loadA(a_base, layout_a, tile_row, kk + 0);
        let b0 = loadB(b_base, layout_b, tile_col, kk + 0);
        let a1 = loadA(a_base, layout_a, tile_row, kk + 16);
        let b1 = loadB(b_base, layout_b, tile_col, kk + 16);
        let a2 = loadA(a_base, layout_a, tile_row, kk + 32);
        let b2 = loadB(b_base, layout_b, tile_col, kk + 32);
        let a3 = loadA(a_base, layout_a, tile_row, kk + 48);
        let b3 = loadB(b_base, layout_b, tile_col, kk + 48);
        result = coopMatMulAdd<float, false>(a0, b0, result);
        result = coopMatMulAdd<float, false>(a1, b1, result);
        result = coopMatMulAdd<float, false>(a2, b2, result);
        result = coopMatMulAdd<float, false>(a3, b3, result);
    }
#elif K_UNROLL >= 2
    // Process two 16-wide K slices per loop iteration: 2 * 16 = 32.
    for (; kk + 32 <= push_constants.k; kk += 32)
    {
        let a0 = loadA(a_base, layout_a, tile_row, kk + 0);
        let b0 = loadB(b_base, layout_b, tile_col, kk + 0);
        let a1 = loadA(a_base, layout_a, tile_row, kk + 16);
        let b1 = loadB(b_base, layout_b, tile_col, kk + 16);
        result = coopMatMulAdd<float, false>(a0, b0, result);
        result = coopMatMulAdd<float, false>(a1, b1, result);
    }
#endif

    // Tail loop handles any remaining 16-wide K slices. The host requires
    // k % 16 == 0 for this shader mode.
    for (; kk < push_constants.k; kk += 16)
    {
        result = coopMatMulAdd<float, false>(loadA(a_base, layout_a, tile_row, kk), loadB(b_base, layout_b, tile_col, kk), result);
    }

    // Store the completed 128x128 accumulator tile into C.
    let c_slice = layout_c.Slice(tile_row, 128, tile_col, 128);
    result.Store(c_buffer, c_base, c_slice);
#else
// Multi-N-tile path. With N_TILES=2, one workgroup computes two neighboring
// output tiles. This saves A loads: each A fragment feeds both accumulators.
#if K_UNROLL >= 8
    for (; kk + 128 <= push_constants.k; kk += 128)
    {
        let a0 = loadA(a_base, layout_a, tile_row, kk + 0);
        let b0 = loadB(b_base, layout_b, tile_col, kk + 0);
        result = coopMatMulAdd<float, false>(a0, b0, result);
#if N_TILES >= 2
        // Same A fragment, B from the next 128-column tile.
        result1 = coopMatMulAdd<float, false>(a0, loadB(b_base, layout_b, tile_col + 128, kk + 0), result1);
#endif
        let a1 = loadA(a_base, layout_a, tile_row, kk + 16);
        let b1 = loadB(b_base, layout_b, tile_col, kk + 16);
        result = coopMatMulAdd<float, false>(a1, b1, result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a1, loadB(b_base, layout_b, tile_col + 128, kk + 16), result1);
#endif
        let a2 = loadA(a_base, layout_a, tile_row, kk + 32);
        let b2 = loadB(b_base, layout_b, tile_col, kk + 32);
        result = coopMatMulAdd<float, false>(a2, b2, result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a2, loadB(b_base, layout_b, tile_col + 128, kk + 32), result1);
#endif
        let a3 = loadA(a_base, layout_a, tile_row, kk + 48);
        let b3 = loadB(b_base, layout_b, tile_col, kk + 48);
        result = coopMatMulAdd<float, false>(a3, b3, result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a3, loadB(b_base, layout_b, tile_col + 128, kk + 48), result1);
#endif
        let a4 = loadA(a_base, layout_a, tile_row, kk + 64);
        let b4 = loadB(b_base, layout_b, tile_col, kk + 64);
        result = coopMatMulAdd<float, false>(a4, b4, result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a4, loadB(b_base, layout_b, tile_col + 128, kk + 64), result1);
#endif
        let a5 = loadA(a_base, layout_a, tile_row, kk + 80);
        let b5 = loadB(b_base, layout_b, tile_col, kk + 80);
        result = coopMatMulAdd<float, false>(a5, b5, result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a5, loadB(b_base, layout_b, tile_col + 128, kk + 80), result1);
#endif
        let a6 = loadA(a_base, layout_a, tile_row, kk + 96);
        let b6 = loadB(b_base, layout_b, tile_col, kk + 96);
        result = coopMatMulAdd<float, false>(a6, b6, result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a6, loadB(b_base, layout_b, tile_col + 128, kk + 96), result1);
#endif
        let a7 = loadA(a_base, layout_a, tile_row, kk + 112);
        let b7 = loadB(b_base, layout_b, tile_col, kk + 112);
        result = coopMatMulAdd<float, false>(a7, b7, result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a7, loadB(b_base, layout_b, tile_col + 128, kk + 112), result1);
#endif
    }
#elif K_UNROLL >= 4
    // This is the optimized target used by //:vk_matmul_slang_n2_f32_u4.
    // Each loop does four A loads, eight B loads, and eight matrix
    // multiply-adds to update two output tiles.
    for (; kk + 64 <= push_constants.k; kk += 64)
    {
        let a0 = loadA(a_base, layout_a, tile_row, kk + 0);
        let b0 = loadB(b_base, layout_b, tile_col, kk + 0);
        result = coopMatMulAdd<float, false>(a0, b0, result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a0, loadB(b_base, layout_b, tile_col + 128, kk + 0), result1);
#endif
#if N_TILES >= 4
        result2 = coopMatMulAdd<float, false>(a0, loadB(b_base, layout_b, tile_col + 256, kk + 0), result2);
        result3 = coopMatMulAdd<float, false>(a0, loadB(b_base, layout_b, tile_col + 384, kk + 0), result3);
#endif
        let a1 = loadA(a_base, layout_a, tile_row, kk + 16);
        let b1 = loadB(b_base, layout_b, tile_col, kk + 16);
        result = coopMatMulAdd<float, false>(a1, b1, result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a1, loadB(b_base, layout_b, tile_col + 128, kk + 16), result1);
#endif
#if N_TILES >= 4
        result2 = coopMatMulAdd<float, false>(a1, loadB(b_base, layout_b, tile_col + 256, kk + 16), result2);
        result3 = coopMatMulAdd<float, false>(a1, loadB(b_base, layout_b, tile_col + 384, kk + 16), result3);
#endif
        let a2 = loadA(a_base, layout_a, tile_row, kk + 32);
        let b2 = loadB(b_base, layout_b, tile_col, kk + 32);
        result = coopMatMulAdd<float, false>(a2, b2, result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a2, loadB(b_base, layout_b, tile_col + 128, kk + 32), result1);
#endif
#if N_TILES >= 4
        result2 = coopMatMulAdd<float, false>(a2, loadB(b_base, layout_b, tile_col + 256, kk + 32), result2);
        result3 = coopMatMulAdd<float, false>(a2, loadB(b_base, layout_b, tile_col + 384, kk + 32), result3);
#endif
        let a3 = loadA(a_base, layout_a, tile_row, kk + 48);
        let b3 = loadB(b_base, layout_b, tile_col, kk + 48);
        result = coopMatMulAdd<float, false>(a3, b3, result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a3, loadB(b_base, layout_b, tile_col + 128, kk + 48), result1);
#endif
#if N_TILES >= 4
        result2 = coopMatMulAdd<float, false>(a3, loadB(b_base, layout_b, tile_col + 256, kk + 48), result2);
        result3 = coopMatMulAdd<float, false>(a3, loadB(b_base, layout_b, tile_col + 384, kk + 48), result3);
#endif
    }
#elif K_UNROLL >= 2
    for (; kk + 32 <= push_constants.k; kk += 32)
    {
        let a0 = loadA(a_base, layout_a, tile_row, kk + 0);
        let b0 = loadB(b_base, layout_b, tile_col, kk + 0);
        result = coopMatMulAdd<float, false>(a0, b0, result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a0, loadB(b_base, layout_b, tile_col + 128, kk + 0), result1);
#endif
        let a1 = loadA(a_base, layout_a, tile_row, kk + 16);
        let b1 = loadB(b_base, layout_b, tile_col, kk + 16);
        result = coopMatMulAdd<float, false>(a1, b1, result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a1, loadB(b_base, layout_b, tile_col + 128, kk + 16), result1);
#endif
    }
#endif

    // Tail loop for the two-tile path. Again, one A load feeds both output
    // accumulators.
    for (; kk < push_constants.k; kk += 16)
    {
        let a_tail = loadA(a_base, layout_a, tile_row, kk);
        result = coopMatMulAdd<float, false>(a_tail, loadB(b_base, layout_b, tile_col, kk), result);
#if N_TILES >= 2
        result1 = coopMatMulAdd<float, false>(a_tail, loadB(b_base, layout_b, tile_col + 128, kk), result1);
#endif
#if N_TILES >= 4
        result2 = coopMatMulAdd<float, false>(a_tail, loadB(b_base, layout_b, tile_col + 256, kk), result2);
        result3 = coopMatMulAdd<float, false>(a_tail, loadB(b_base, layout_b, tile_col + 384, kk), result3);
#endif
    }

    // Store the first 128x128 output tile.
    let c_slice = layout_c.Slice(tile_row, 128, tile_col, 128);
    result.Store(c_buffer, c_base, c_slice);
#if N_TILES >= 2
    // Store the second 128x128 output tile immediately to the right.
    let c_slice1 = layout_c.Slice(tile_row, 128, tile_col + 128, 128);
    result1.Store(c_buffer, c_base, c_slice1);
#endif
#if N_TILES >= 4
    let c_slice2 = layout_c.Slice(tile_row, 128, tile_col + 256, 128);
    result2.Store(c_buffer, c_base, c_slice2);
    let c_slice3 = layout_c.Slice(tile_row, 128, tile_col + 384, 128);
    result3.Store(c_buffer, c_base, c_slice3);
#endif
#endif
}
