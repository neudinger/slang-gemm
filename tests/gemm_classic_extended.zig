const std = @import("std");
const gemm = @import("gemm");

const testing = std.testing;

test "CUDA data type enum exposes CUDA 13.1 values" {
    try testing.expectEqual(@as(c_int, 0), @intFromEnum(gemm.DataType.r_32f));
    try testing.expectEqual(@as(c_int, 1), @intFromEnum(gemm.DataType.r_64f));
    try testing.expectEqual(@as(c_int, 2), @intFromEnum(gemm.DataType.r_16f));
    try testing.expectEqual(@as(c_int, 14), @intFromEnum(gemm.DataType.r_16bf));
    try testing.expectEqual(@as(c_int, 28), @intFromEnum(gemm.DataType.r_8f_e4m3));
    try testing.expectEqual(@as(c_int, 33), @intFromEnum(gemm.DataType.r_4f_e2m1));
    try testing.expectEqual(@as(c_int, 68), @intFromEnum(gemm.ComputeType.f32));
    try testing.expectEqual(@as(c_int, -1), @intFromEnum(gemm.Algo.default));
    try testing.expectEqual(@as(c_int, 999), @intFromEnum(gemm.Algo.autotune));
}

test "generic gemmEx f32 bytes matches sgemm" {
    const m = 2;
    const n = 3;
    const k = 4;
    var a: [m * k]f32 = undefined;
    var b: [k * n]f32 = undefined;
    var c: [m * n]f32 = undefined;
    var expected: [m * n]f32 = undefined;
    fill(f32, &a, 0.1);
    fill(f32, &b, -0.2);
    fill(f32, &c, 0.3);
    expected = c;

    try gemm.sgemm(.row_major, .no_trans, .no_trans, m, n, k, 1.25, &a, k, &b, n, -0.5, &expected, n);
    try gemm.gemmEx(.{
        .op_a = gemm.Op.no_trans,
        .op_b = gemm.Op.no_trans,
        .m = m,
        .n = n,
        .k = k,
        .alpha = gemm.Scalar.real(1.25),
        .a = bytesConst(&a),
        .a_type = gemm.DataType.r_32f,
        .lda = k,
        .b = bytesConst(&b),
        .b_type = gemm.DataType.r_32f,
        .ldb = n,
        .beta = gemm.Scalar.real(-0.5),
        .c = bytes(&c),
        .c_type = gemm.DataType.r_32f,
        .ldc = n,
        .compute_type = gemm.ComputeType.f32,
        .layout = gemm.Layout.row_major,
    });

    for (expected, c) |e, got| try testing.expectApproxEqAbs(e, got, 1e-5);
}

test "64-bit wrapper aliases match standard wrappers" {
    const m = 2;
    const n = 2;
    const k = 3;
    var a: [m * k]f32 = undefined;
    var b: [k * n]f32 = undefined;
    var c32: [m * n]f32 = undefined;
    var c64: [m * n]f32 = undefined;
    fill(f32, &a, 0.1);
    fill(f32, &b, -0.2);
    fill(f32, &c32, 0.3);
    c64 = c32;

    try gemm.sgemm(.row_major, .no_trans, .no_trans, m, n, k, 1.25, &a, k, &b, n, -0.5, &c32, n);
    try gemm.sgemm64(.row_major, .no_trans, .no_trans, m, n, k, 1.25, &a, k, &b, n, -0.5, &c64, n);
    for (c32, c64) |e, got| try testing.expectApproxEqAbs(e, got, 1e-5);
}

test "generic gemmEx supports strided batched f16 and bf16 inputs" {
    try testLowPrecisionStrided(.r_16f, 2e-2);
    try testLowPrecisionStrided(.r_16bf, 5e-2);
}

test "pointer-array gemmBatchedEx dispatches each batch" {
    const m = 2;
    const n = 2;
    const k = 3;
    var a0: [m * k]f32 = undefined;
    var a1: [m * k]f32 = undefined;
    var b0: [k * n]f32 = undefined;
    var b1: [k * n]f32 = undefined;
    var c0: [m * n]f32 = undefined;
    var c1: [m * n]f32 = undefined;
    var e0: [m * n]f32 = undefined;
    var e1: [m * n]f32 = undefined;
    fill(f32, &a0, 0.1);
    fill(f32, &a1, 0.2);
    fill(f32, &b0, -0.3);
    fill(f32, &b1, -0.4);
    fill(f32, &c0, 0.5);
    fill(f32, &c1, 0.6);
    e0 = c0;
    e1 = c1;

    try gemm.sgemm(.row_major, .no_trans, .no_trans, m, n, k, 1, &a0, k, &b0, n, 0, &e0, n);
    try gemm.sgemm(.row_major, .no_trans, .no_trans, m, n, k, 1, &a1, k, &b1, n, 0, &e1, n);
    const aa = [_][]const u8{ bytesConst(&a0), bytesConst(&a1) };
    const bb = [_][]const u8{ bytesConst(&b0), bytesConst(&b1) };
    const cc = [_][]u8{ bytes(&c0), bytes(&c1) };
    try gemm.gemmBatchedEx(.{
        .op_a = .no_trans,
        .op_b = .no_trans,
        .m = m,
        .n = n,
        .k = k,
        .alpha = gemm.Scalar.real(1),
        .a = &aa,
        .a_type = .r_32f,
        .lda = k,
        .b = &bb,
        .b_type = .r_32f,
        .ldb = n,
        .beta = gemm.Scalar.real(0),
        .c = &cc,
        .c_type = .r_32f,
        .ldc = n,
        .compute_type = .f32,
        .layout = .row_major,
    });
    for (e0, c0) |e, got| try testing.expectApproxEqAbs(e, got, 1e-5);
    for (e1, c1) |e, got| try testing.expectApproxEqAbs(e, got, 1e-5);
}

test "low precision scalar decoders produce finite values" {
    const values = [_]struct { data_type: gemm.DataType, bits: u8 }{
        .{ .data_type = .r_8f_e4m3, .bits = 0b0_0111_000 },
        .{ .data_type = .r_8f_e5m2, .bits = 0b0_01111_00 },
        .{ .data_type = .r_8f_ue8m0, .bits = 127 },
        .{ .data_type = .r_6f_e2m3, .bits = 0b0_01_000 },
        .{ .data_type = .r_6f_e3m2, .bits = 0b0_011_00 },
        .{ .data_type = .r_4f_e2m1, .bits = 0b0_01_0 },
    };
    for (values) |case| {
        const data = [_]u8{case.bits};
        const got = gemm.readScalar(&data, case.data_type, 0);
        try testing.expect(std.math.isFinite(got.re));
    }
}

fn testLowPrecisionStrided(data_type: gemm.DataType, tol: f32) !void {
    const m = 2;
    const n = 2;
    const k = 3;
    const batch_count = 2;
    const a_stride = m * k;
    const b_stride = k * n;
    const c_stride = m * n;
    var a: [a_stride * batch_count]u16 = undefined;
    var b: [b_stride * batch_count]u16 = undefined;
    var c: [c_stride * batch_count]f32 = undefined;
    var expected: [c_stride * batch_count]f32 = undefined;
    for (&a, 0..) |*v, i| v.* = encode(data_type, valueFor(f32, i, 0.1));
    for (&b, 0..) |*v, i| v.* = encode(data_type, valueFor(f32, i, -0.2));
    fill(f32, &c, 0.3);
    expected = c;
    const legacy_input: gemm.GemmType = if (data_type == .r_16f) .f16_f32 else .bf16_f32;

    for (0..batch_count) |batch| {
        const a_slice = a[batch * a_stride ..][0..a_stride];
        const b_slice = b[batch * b_stride ..][0..b_stride];
        try gemm.gemmEx(.{ .op_a = .no_trans, .op_b = .no_trans, .m = m, .n = n, .k = k, .a = a_slice, .b = b_slice, .c = expected[batch * c_stride ..][0..c_stride], .a_layout = .{ .row_stride = k, .col_stride = 1 }, .b_layout = .{ .row_stride = n, .col_stride = 1 }, .c_layout = .{ .row_stride = n, .col_stride = 1 }, .alpha = 1, .beta = 0, .input = legacy_input });
    }

    try gemm.gemmStridedBatchedEx(.{
        .op_a = .no_trans,
        .op_b = .no_trans,
        .m = m,
        .n = n,
        .k = k,
        .alpha = gemm.Scalar.real(1),
        .a = bytesConst(&a),
        .a_type = data_type,
        .lda = k,
        .stride_a = a_stride,
        .b = bytesConst(&b),
        .b_type = data_type,
        .ldb = n,
        .stride_b = b_stride,
        .beta = gemm.Scalar.real(0),
        .c = bytes(&c),
        .c_type = .r_32f,
        .ldc = n,
        .stride_c = c_stride,
        .batch_count = batch_count,
        .compute_type = .f32,
        .layout = .row_major,
    });
    for (expected, c) |e, got| try testing.expectApproxEqAbs(e, got, tol);
}

fn encode(data_type: gemm.DataType, value: f32) u16 {
    return switch (data_type) {
        .r_16f => gemm.f32ToF16Bits(value),
        .r_16bf => gemm.f32ToBf16Bits(value),
        else => unreachable,
    };
}

fn fill(comptime T: type, dst: []T, bias: T) void {
    for (dst, 0..) |*value, i| value.* = valueFor(T, i, bias);
}

fn valueFor(comptime T: type, i: usize, bias: T) T {
    const raw: T = @floatFromInt((i * 19 + 7) % 23);
    return (raw - 11) / 7 + bias;
}

fn bytes(slice: anytype) []u8 {
    return std.mem.asBytes(slice)[0..];
}

fn bytesConst(slice: anytype) []const u8 {
    return std.mem.asBytes(slice)[0..];
}
