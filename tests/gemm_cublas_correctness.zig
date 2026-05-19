const std = @import("std");
const gemm = @import("gemm");
const cublas = @import("cublas");

const testing = std.testing;

test "standard BLAS routines match cuBLAS column-major outputs" {
    if (!cublas.hasCudaDevice()) return error.SkipZigTest;

    const ops = [_]gemm.Op{ .no_trans, .trans, .conj_trans };
    for (ops) |op_a| {
        for (ops) |op_b| {
            try compareReal(f32, op_a, op_b, 2e-5);
            try compareReal(f64, op_a, op_b, 1e-11);
            try compareComplex(gemm.Complex32, op_a, op_b, 3e-5);
            try compareComplex(gemm.Complex64, op_a, op_b, 1e-11);
        }
    }
}

test "cuBLAS GemmEx f32 path matches scalar reference" {
    if (!cublas.hasCudaDevice()) return error.SkipZigTest;

    const m = 4;
    const n = 3;
    const k = 5;
    var a: [m * k]f32 = undefined;
    var b: [k * n]f32 = undefined;
    var expected: [m * n]f32 = undefined;
    var got: [m * n]f32 = undefined;
    fillReal(f32, &a, 0.1);
    fillReal(f32, &b, -0.2);
    fillReal(f32, &expected, 0.3);
    got = expected;

    const alpha: f32 = 1.25;
    const beta: f32 = -0.5;
    try gemm.sgemm(.col_major, .no_trans, .no_trans, m, n, k, alpha, &a, m, &b, k, beta, &expected, m);

    const ctx = try cublas.Context.init();
    defer ctx.deinit();
    const da = try cublas.DeviceBuffer(f32).alloc(a.len);
    defer da.free();
    const db = try cublas.DeviceBuffer(f32).alloc(b.len);
    defer db.free();
    const dc = try cublas.DeviceBuffer(f32).alloc(got.len);
    defer dc.free();
    try da.copyFromHost(&a);
    try db.copyFromHost(&b);
    try dc.copyFromHost(&got);
    try cublas.gemmEx(ctx, .{
        .op_a = .no_trans,
        .op_b = .no_trans,
        .m = m,
        .n = n,
        .k = k,
        .alpha = &alpha,
        .a = anyConst(da.ptr),
        .a_type = .r_32f,
        .lda = m,
        .b = anyConst(db.ptr),
        .b_type = .r_32f,
        .ldb = k,
        .beta = &beta,
        .c = anyMut(dc.ptr),
        .c_type = .r_32f,
        .ldc = m,
        .compute_type = .f32,
    });
    try dc.copyToHost(&got);
    for (expected, got) |e, actual| try testing.expectApproxEqAbs(e, actual, 2e-5);
}

test "cuBLAS strided batched sgemm matches scalar reference" {
    if (!cublas.hasCudaDevice()) return error.SkipZigTest;

    const m = 2;
    const n = 3;
    const k = 4;
    const batch_count = 2;
    const a_stride = m * k;
    const b_stride = k * n;
    const c_stride = m * n;
    var a: [a_stride * batch_count]f32 = undefined;
    var b: [b_stride * batch_count]f32 = undefined;
    var expected: [c_stride * batch_count]f32 = undefined;
    var got: [c_stride * batch_count]f32 = undefined;
    fillReal(f32, &a, 0.1);
    fillReal(f32, &b, -0.2);
    fillReal(f32, &expected, 0.3);
    got = expected;

    const alpha: f32 = 1.0;
    const beta: f32 = 0.0;
    try gemm.sgemmStridedBatched(.col_major, .no_trans, .no_trans, m, n, k, alpha, &a, m, a_stride, &b, k, b_stride, beta, &expected, m, c_stride, batch_count);

    const ctx = try cublas.Context.init();
    defer ctx.deinit();
    const da = try cublas.DeviceBuffer(f32).alloc(a.len);
    defer da.free();
    const db = try cublas.DeviceBuffer(f32).alloc(b.len);
    defer db.free();
    const dc = try cublas.DeviceBuffer(f32).alloc(got.len);
    defer dc.free();
    try da.copyFromHost(&a);
    try db.copyFromHost(&b);
    try dc.copyFromHost(&got);
    try cublas.sgemmStridedBatched(ctx, .no_trans, .no_trans, m, n, k, alpha, da.ptr, m, a_stride, db.ptr, k, b_stride, beta, dc.ptr, m, c_stride, batch_count);
    try dc.copyToHost(&got);
    for (expected, got) |e, actual| try testing.expectApproxEqAbs(e, actual, 2e-5);
}

test "cuBLAS pointer-array batched sgemm matches scalar reference" {
    if (!cublas.hasCudaDevice()) return error.SkipZigTest;

    const m = 2;
    const n = 3;
    const k = 4;
    const batch_count = 2;
    const a_len = m * k;
    const b_len = k * n;
    const c_len = m * n;
    var a0: [a_len]f32 = undefined;
    var a1: [a_len]f32 = undefined;
    var b0: [b_len]f32 = undefined;
    var b1: [b_len]f32 = undefined;
    var expected0: [c_len]f32 = undefined;
    var expected1: [c_len]f32 = undefined;
    var got0: [c_len]f32 = undefined;
    var got1: [c_len]f32 = undefined;
    fillReal(f32, &a0, 0.1);
    fillReal(f32, &a1, 0.7);
    fillReal(f32, &b0, -0.2);
    fillReal(f32, &b1, -0.6);
    fillReal(f32, &expected0, 0.3);
    fillReal(f32, &expected1, -0.4);
    got0 = expected0;
    got1 = expected1;

    const alpha: f32 = 0.75;
    const beta: f32 = -0.25;
    const a_batches = [_][]const f32{ a0[0..], a1[0..] };
    const b_batches = [_][]const f32{ b0[0..], b1[0..] };
    const c_batches = [_][]f32{ expected0[0..], expected1[0..] };
    try gemm.sgemmBatched(.col_major, .no_trans, .no_trans, m, n, k, alpha, &a_batches, m, &b_batches, k, beta, &c_batches, m);

    const ctx = try cublas.Context.init();
    defer ctx.deinit();
    const da0 = try cublas.DeviceBuffer(f32).alloc(a0.len);
    defer da0.free();
    const da1 = try cublas.DeviceBuffer(f32).alloc(a1.len);
    defer da1.free();
    const db0 = try cublas.DeviceBuffer(f32).alloc(b0.len);
    defer db0.free();
    const db1 = try cublas.DeviceBuffer(f32).alloc(b1.len);
    defer db1.free();
    const dc0 = try cublas.DeviceBuffer(f32).alloc(got0.len);
    defer dc0.free();
    const dc1 = try cublas.DeviceBuffer(f32).alloc(got1.len);
    defer dc1.free();
    try da0.copyFromHost(&a0);
    try da1.copyFromHost(&a1);
    try db0.copyFromHost(&b0);
    try db1.copyFromHost(&b1);
    try dc0.copyFromHost(&got0);
    try dc1.copyFromHost(&got1);

    const a_ptrs = [_][*]f32{ da0.ptr, da1.ptr };
    const b_ptrs = [_][*]f32{ db0.ptr, db1.ptr };
    const c_ptrs = [_][*]f32{ dc0.ptr, dc1.ptr };
    const da_array = try cublas.DevicePointerArray(f32).alloc(&a_ptrs);
    defer da_array.free();
    const db_array = try cublas.DevicePointerArray(f32).alloc(&b_ptrs);
    defer db_array.free();
    const dc_array = try cublas.DevicePointerArray(f32).alloc(&c_ptrs);
    defer dc_array.free();

    try cublas.sgemmBatched(ctx, .no_trans, .no_trans, m, n, k, alpha, anyConst(da_array.ptr), m, anyConst(db_array.ptr), k, beta, anyMut(dc_array.ptr), m, batch_count);
    try dc0.copyToHost(&got0);
    try dc1.copyToHost(&got1);
    for (expected0, got0) |e, actual| try testing.expectApproxEqAbs(e, actual, 2e-5);
    for (expected1, got1) |e, actual| try testing.expectApproxEqAbs(e, actual, 2e-5);
}

fn compareReal(comptime T: type, op_a: gemm.Op, op_b: gemm.Op, tol: T) !void {
    const m = 5;
    const n = 4;
    const k = 3;
    const a_rows = gemm.storedRows(op_a, m, k);
    const a_cols = gemm.storedCols(op_a, m, k);
    const b_rows = gemm.storedRows(op_b, k, n);
    const b_cols = gemm.storedCols(op_b, k, n);
    const lda = gemm.minimumLd(.col_major, a_rows, a_cols) + 1;
    const ldb = gemm.minimumLd(.col_major, b_rows, b_cols) + 2;
    const ldc = gemm.minimumLd(.col_major, m, n) + 1;
    const a_len = (a_cols - 1) * lda + a_rows;
    const b_len = (b_cols - 1) * ldb + b_rows;
    const c_len = (n - 1) * ldc + m;

    const a = try testing.allocator.alloc(T, a_len);
    defer testing.allocator.free(a);
    const b = try testing.allocator.alloc(T, b_len);
    defer testing.allocator.free(b);
    const expected = try testing.allocator.alloc(T, c_len);
    defer testing.allocator.free(expected);
    const got = try testing.allocator.alloc(T, c_len);
    defer testing.allocator.free(got);
    fillReal(T, a, 0.1);
    fillReal(T, b, -0.2);
    fillReal(T, expected, 0.3);
    @memcpy(got, expected);

    const alpha: T = 1.25;
    const beta: T = -0.5;
    if (T == f32) {
        try gemm.sgemm(.col_major, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, expected, ldc);
        try runCublasReal(f32, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, got, ldc);
    } else {
        try gemm.dgemm(.col_major, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, expected, ldc);
        try runCublasReal(f64, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, got, ldc);
    }
    for (expected, got) |e, actual| try testing.expectApproxEqAbs(e, actual, tol);
}

fn compareComplex(comptime T: type, op_a: gemm.Op, op_b: gemm.Op, tol: anytype) !void {
    const m = 5;
    const n = 4;
    const k = 3;
    const a_rows = gemm.storedRows(op_a, m, k);
    const a_cols = gemm.storedCols(op_a, m, k);
    const b_rows = gemm.storedRows(op_b, k, n);
    const b_cols = gemm.storedCols(op_b, k, n);
    const lda = gemm.minimumLd(.col_major, a_rows, a_cols) + 1;
    const ldb = gemm.minimumLd(.col_major, b_rows, b_cols) + 2;
    const ldc = gemm.minimumLd(.col_major, m, n) + 1;
    const a_len = (a_cols - 1) * lda + a_rows;
    const b_len = (b_cols - 1) * ldb + b_rows;
    const c_len = (n - 1) * ldc + m;

    const a = try testing.allocator.alloc(T, a_len);
    defer testing.allocator.free(a);
    const b = try testing.allocator.alloc(T, b_len);
    defer testing.allocator.free(b);
    const expected = try testing.allocator.alloc(T, c_len);
    defer testing.allocator.free(expected);
    const got = try testing.allocator.alloc(T, c_len);
    defer testing.allocator.free(got);
    fillComplex(T, a, 0.1);
    fillComplex(T, b, -0.2);
    fillComplex(T, expected, 0.3);
    @memcpy(got, expected);

    const alpha: T = .{ .re = 1.25, .im = -0.25 };
    const beta: T = .{ .re = -0.5, .im = 0.125 };
    if (T == gemm.Complex32) {
        try gemm.cgemm(.col_major, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, expected, ldc);
        try runCublasComplex(gemm.Complex32, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, got, ldc);
    } else {
        try gemm.zgemm(.col_major, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, expected, ldc);
        try runCublasComplex(gemm.Complex64, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, got, ldc);
    }
    for (expected, got) |e, actual| {
        try testing.expectApproxEqAbs(e.re, actual.re, tol);
        try testing.expectApproxEqAbs(e.im, actual.im, tol);
    }
}

fn runCublasReal(comptime T: type, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: T, a: []const T, lda: usize, b: []const T, ldb: usize, beta: T, c: []T, ldc: usize) !void {
    const ctx = try cublas.Context.init();
    defer ctx.deinit();
    const da = try cublas.DeviceBuffer(T).alloc(a.len);
    defer da.free();
    const db = try cublas.DeviceBuffer(T).alloc(b.len);
    defer db.free();
    const dc = try cublas.DeviceBuffer(T).alloc(c.len);
    defer dc.free();
    try da.copyFromHost(a);
    try db.copyFromHost(b);
    try dc.copyFromHost(c);
    if (T == f32) {
        try cublas.sgemm(ctx, op_a, op_b, m, n, k, alpha, da.ptr, lda, db.ptr, ldb, beta, dc.ptr, ldc);
    } else {
        try cublas.dgemm(ctx, op_a, op_b, m, n, k, alpha, da.ptr, lda, db.ptr, ldb, beta, dc.ptr, ldc);
    }
    try dc.copyToHost(c);
}

fn runCublasComplex(comptime T: type, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: T, a: []const T, lda: usize, b: []const T, ldb: usize, beta: T, c: []T, ldc: usize) !void {
    const ctx = try cublas.Context.init();
    defer ctx.deinit();
    const da = try cublas.DeviceBuffer(T).alloc(a.len);
    defer da.free();
    const db = try cublas.DeviceBuffer(T).alloc(b.len);
    defer db.free();
    const dc = try cublas.DeviceBuffer(T).alloc(c.len);
    defer dc.free();
    try da.copyFromHost(a);
    try db.copyFromHost(b);
    try dc.copyFromHost(c);
    if (T == gemm.Complex32) {
        try cublas.cgemm(ctx, op_a, op_b, m, n, k, alpha, da.ptr, lda, db.ptr, ldb, beta, dc.ptr, ldc);
    } else {
        try cublas.zgemm(ctx, op_a, op_b, m, n, k, alpha, da.ptr, lda, db.ptr, ldb, beta, dc.ptr, ldc);
    }
    try dc.copyToHost(c);
}

fn fillReal(comptime T: type, dst: []T, bias: T) void {
    for (dst, 0..) |*value, i| {
        const raw: T = @floatFromInt((i * 19 + 7) % 23);
        value.* = (raw - 11) / 7 + bias;
    }
}

fn fillComplex(comptime T: type, dst: []T, bias: anytype) void {
    for (dst, 0..) |*value, i| {
        value.* = .{
            .re = valueFor(@TypeOf(value.re), i, bias),
            .im = valueFor(@TypeOf(value.im), i + 17, -bias),
        };
    }
}

fn valueFor(comptime T: type, i: usize, bias: T) T {
    const raw: T = @floatFromInt((i * 19 + 7) % 23);
    return (raw - 11) / 7 + bias;
}

fn anyConst(ptr: anytype) ?*const anyopaque {
    return @ptrCast(&ptr[0]);
}

fn anyMut(ptr: anytype) ?*anyopaque {
    return @ptrCast(&ptr[0]);
}
