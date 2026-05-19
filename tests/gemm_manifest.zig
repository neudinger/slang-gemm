const std = @import("std");
const gemm = @import("gemm");
const manifest = @import("manifest");

const shape = manifest.Shape{ .m = 128, .n = 128, .k = 128 };
const tensor_shape = manifest.Shape{ .m = 256, .n = 256, .k = 256 };

test "manifest reports runnable FP32 shader case" {
    const case = manifest.Case{
        .api = .regular,
        .op_a = .no_trans,
        .op_b = .no_trans,
        .a_type = .r_32f,
        .b_type = .r_32f,
        .c_type = .r_32f,
        .compute_type = .f32,
        .shape = shape,
    };

    try std.testing.expect(case.cublasSupported());
    try std.testing.expect(case.shaderImplemented());
    try std.testing.expect(case.shaderAvailable());
    try std.testing.expect(case.dispatchAvailable());
    try std.testing.expectEqual(manifest.Status.ok, case.baseStatus());
    try std.testing.expectEqualStrings("classic_sgemm", case.family());
    try std.testing.expectEqual(manifest.ShaderFamily.f32_tf32, case.shaderFamily());
}

test "manifest reports tensor Ex shader implemented and dispatch available" {
    const case = manifest.Case{
        .api = .gemm_ex,
        .op_a = .no_trans,
        .op_b = .no_trans,
        .a_type = .r_16f,
        .b_type = .r_16f,
        .c_type = .r_32f,
        .compute_type = .f32,
        .shape = tensor_shape,
    };

    try std.testing.expect(case.cublasSupported());
    try std.testing.expect(case.shaderImplemented());
    try std.testing.expect(case.hardwareAccelerated());
    try std.testing.expect(case.tileCompatible());
    try std.testing.expect(case.shaderAvailable());
    try std.testing.expect(case.dispatchAvailable());
    try std.testing.expectEqual(manifest.Status.ok, case.baseStatus());
    try std.testing.expectEqualStrings("tensor_ex", case.family());
    try std.testing.expectEqual(manifest.ShaderFamily.f16_bf16_tensor, case.shaderFamily());
}

test "strict tile compatibility distinguishes tensor-aligned shapes" {
    const compatible = manifest.Case{
        .api = .gemm_ex,
        .op_a = .no_trans,
        .op_b = .no_trans,
        .a_type = .r_16f,
        .b_type = .r_16f,
        .c_type = .r_32f,
        .compute_type = .f32,
        .shape = .{ .m = 256, .n = 256, .k = 256 },
    };
    const incompatible = manifest.Case{
        .api = .gemm_ex,
        .op_a = .no_trans,
        .op_b = .no_trans,
        .a_type = .r_16f,
        .b_type = .r_16f,
        .c_type = .r_32f,
        .compute_type = .f32,
        .shape = .{ .m = 256, .n = 128, .k = 256 },
    };

    try std.testing.expect(compatible.tileCompatible());
    try std.testing.expect(!incompatible.tileCompatible());
}

test "manifest reports unsupported cuBLAS enum combinations" {
    const case = manifest.Case{
        .api = .gemm_ex,
        .op_a = .no_trans,
        .op_b = .no_trans,
        .a_type = .r_4f_e2m1,
        .b_type = .r_4f_e2m1,
        .c_type = .r_32f,
        .compute_type = .f32,
        .shape = shape,
    };

    try std.testing.expect(!case.cublasSupported());
    try std.testing.expect(!case.shaderImplemented());
    try std.testing.expectEqual(manifest.Status.cublas_unsupported, case.baseStatus());
    try std.testing.expectEqualStrings("CUDA_R_4F_E2M1", manifest.dataTypeLabel(.r_4f_e2m1));
    try std.testing.expectEqualStrings("CUBLAS_COMPUTE_32F", manifest.computeTypeLabel(.f32));
    try std.testing.expectEqualStrings("N", manifest.opLabel(gemm.Op.no_trans));
}

test "all cuBLAS-supported compact GEMM families have a shader implementation" {
    const cases = [_]manifest.Case{
        .{ .api = .regular, .op_a = .no_trans, .op_b = .no_trans, .a_type = .r_32f, .b_type = .r_32f, .c_type = .r_32f, .compute_type = .f32, .shape = shape },
        .{ .api = .regular, .op_a = .trans, .op_b = .conj_trans, .a_type = .r_64f, .b_type = .r_64f, .c_type = .r_64f, .compute_type = .f64, .shape = shape },
        .{ .api = .regular, .op_a = .conj_trans, .op_b = .no_trans, .a_type = .c_32f, .b_type = .c_32f, .c_type = .c_32f, .compute_type = .f32, .shape = shape },
        .{ .api = .regular, .op_a = .no_trans, .op_b = .conj_trans, .a_type = .c_64f, .b_type = .c_64f, .c_type = .c_64f, .compute_type = .f64, .shape = shape },
        .{ .api = .complex_3m, .op_a = .conj_trans, .op_b = .trans, .a_type = .c_32f, .b_type = .c_32f, .c_type = .c_32f, .compute_type = .f32, .shape = shape },
        .{ .api = .complex_3m, .op_a = .trans, .op_b = .conj_trans, .a_type = .c_64f, .b_type = .c_64f, .c_type = .c_64f, .compute_type = .f64, .shape = shape },
        .{ .api = .gemm_ex, .op_a = .no_trans, .op_b = .no_trans, .a_type = .r_16f, .b_type = .r_16f, .c_type = .r_16f, .compute_type = .f16, .shape = shape },
        .{ .api = .gemm_ex, .op_a = .trans, .op_b = .no_trans, .a_type = .r_16f, .b_type = .r_16f, .c_type = .r_32f, .compute_type = .f32_fast_16f, .shape = shape },
        .{ .api = .gemm_ex, .op_a = .no_trans, .op_b = .trans, .a_type = .r_16bf, .b_type = .r_16bf, .c_type = .r_32f, .compute_type = .f32_fast_16bf, .shape = shape },
        .{ .api = .gemm_ex, .op_a = .conj_trans, .op_b = .no_trans, .a_type = .r_32f, .b_type = .r_32f, .c_type = .r_32f, .compute_type = .f32_fast_tf32, .shape = shape },
        .{ .api = .gemm_ex, .op_a = .no_trans, .op_b = .conj_trans, .a_type = .r_8i, .b_type = .r_8i, .c_type = .r_32i, .compute_type = .i32, .shape = shape },
        .{ .api = .gemm_strided_batched_ex, .op_a = .no_trans, .op_b = .no_trans, .a_type = .r_16f, .b_type = .r_16f, .c_type = .r_32f, .compute_type = .f32, .shape = shape, .batched = true },
    };

    for (cases) |case| {
        try std.testing.expect(case.cublasSupported());
        try std.testing.expect(case.shaderImplemented());
        try std.testing.expect(case.shaderFamily() != .none);
    }
}

test "manifest has no shader unavailable gaps for static cuBLAS-supported matrix" {
    for (manifest.apis) |api| {
        for (manifest.ops) |op_a| {
            for (manifest.ops) |op_b| {
                for (manifest.data_types) |data_type| {
                    for (manifest.compute_types) |compute_type| {
                        const case = manifest.Case{
                            .api = api,
                            .op_a = op_a,
                            .op_b = op_b,
                            .a_type = data_type,
                            .b_type = data_type,
                            .c_type = data_type,
                            .compute_type = compute_type,
                            .shape = shape,
                            .batched = api == .batched or api == .strided_batched or api == .gemm_batched_ex or api == .gemm_strided_batched_ex,
                        };
                        if (case.cublasSupported()) {
                            try std.testing.expect(case.shaderImplemented());
                            try std.testing.expect(case.dispatchAvailable());
                            try std.testing.expectEqual(manifest.Status.ok, case.baseStatus());
                        }
                    }
                }
                const mixed_cases = [_]manifest.Case{
                    .{ .api = api, .op_a = op_a, .op_b = op_b, .a_type = .r_16f, .b_type = .r_16f, .c_type = .r_32f, .compute_type = .f32, .shape = shape },
                    .{ .api = api, .op_a = op_a, .op_b = op_b, .a_type = .r_16bf, .b_type = .r_16bf, .c_type = .r_32f, .compute_type = .f32, .shape = shape },
                    .{ .api = api, .op_a = op_a, .op_b = op_b, .a_type = .r_32f, .b_type = .r_32f, .c_type = .r_32f, .compute_type = .f32_fast_tf32, .shape = shape },
                    .{ .api = api, .op_a = op_a, .op_b = op_b, .a_type = .r_8i, .b_type = .r_8i, .c_type = .r_32i, .compute_type = .i32, .shape = shape },
                };
                for (mixed_cases) |case| {
                    if (case.cublasSupported()) {
                        try std.testing.expect(case.shaderImplemented());
                        try std.testing.expect(case.dispatchAvailable());
                        try std.testing.expectEqual(manifest.Status.ok, case.baseStatus());
                    }
                }
            }
        }
    }
}
