const std = @import("std");
const gemm = @import("gemm");

const testing = std.testing;

test "sgemm and dgemm cover real transpose variants" {
    const ops = [_]gemm.Op{ .no_trans, .trans, .conj_trans };
    for (ops) |op_a| {
        for (ops) |op_b| {
            try testReal(f32, op_a, op_b, .col_major, 1.25, -0.5, 2e-5);
            try testReal(f32, op_a, op_b, .row_major, -0.75, 0.25, 2e-5);
            try testReal(f64, op_a, op_b, .col_major, 1.25, -0.5, 1e-11);
            try testReal(f64, op_a, op_b, .row_major, -0.75, 0.25, 1e-11);
        }
    }
}

test "cgemm and zgemm cover complex transpose variants" {
    const ops = [_]gemm.Op{ .no_trans, .trans, .conj_trans };
    for (ops) |op_a| {
        for (ops) |op_b| {
            try testComplex(gemm.Complex32, op_a, op_b, .col_major, .{ .re = 0.75, .im = -0.25 }, .{ .re = -0.5, .im = 0.125 }, 3e-5);
            try testComplex(gemm.Complex32, op_a, op_b, .row_major, .{ .re = -0.25, .im = 0.5 }, .{ .re = 0.375, .im = -0.75 }, 3e-5);
            try testComplex(gemm.Complex64, op_a, op_b, .col_major, .{ .re = 0.75, .im = -0.25 }, .{ .re = -0.5, .im = 0.125 }, 1e-11);
            try testComplex(gemm.Complex64, op_a, op_b, .row_major, .{ .re = -0.25, .im = 0.5 }, .{ .re = 0.375, .im = -0.75 }, 1e-11);
        }
    }
}

test "alpha and beta zero behavior" {
    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var c = [_]f32{ 99, -7, 13, 42 };

    try gemm.sgemm(.row_major, .no_trans, .no_trans, 2, 2, 3, 0, &a, 3, &b, 2, 0, &c, 2);
    for (c) |value| try testing.expectEqual(@as(f32, 0), value);

    c = .{ 99, -7, 13, 42 };
    try gemm.sgemm(.row_major, .no_trans, .no_trans, 2, 2, 3, 0, &a, 3, &b, 2, 2, &c, 2);
    try testing.expectApproxEqAbs(@as(f32, 198), c[0], 0);
    try testing.expectApproxEqAbs(@as(f32, -14), c[1], 0);
    try testing.expectApproxEqAbs(@as(f32, 26), c[2], 0);
    try testing.expectApproxEqAbs(@as(f32, 84), c[3], 0);
}

test "strided batched descriptor" {
    const m = 2;
    const n = 3;
    const k = 4;
    const batch_count = 3;
    const a_stride = 16;
    const b_stride = 18;
    const c_stride = 12;

    var a: [a_stride * batch_count]f32 = undefined;
    var b: [b_stride * batch_count]f32 = undefined;
    var c: [c_stride * batch_count]f32 = undefined;
    var expected: [c_stride * batch_count]f32 = undefined;
    fillReal(f32, &a, 0.1);
    fillReal(f32, &b, -0.2);
    fillReal(f32, &c, 0.7);
    expected = c;

    const a_layout: gemm.MatrixLayout = .{ .row_stride = k, .col_stride = 1 };
    const b_layout: gemm.MatrixLayout = .{ .row_stride = k, .col_stride = 1 };
    const c_layout: gemm.MatrixLayout = .{ .row_stride = n, .col_stride = 1 };
    referenceReal(f32, .no_trans, .trans, m, n, k, 1.5, &a, a_layout, a_stride, &b, b_layout, b_stride, -0.25, &expected, c_layout, c_stride, batch_count);

    try gemm.gemm(f32, .{
        .op_a = .no_trans,
        .op_b = .trans,
        .m = m,
        .n = n,
        .k = k,
        .alpha = 1.5,
        .beta = -0.25,
        .a = &a,
        .b = &b,
        .c = &c,
        .a_layout = a_layout,
        .b_layout = b_layout,
        .c_layout = c_layout,
        .batch_count = batch_count,
        .stride_a = a_stride,
        .stride_b = b_stride,
        .stride_c = c_stride,
    });

    for (0..batch_count) |batch| {
        for (0..m) |row| {
            for (0..n) |col| {
                const idx = batch * c_stride + row * n + col;
                try testing.expectApproxEqAbs(expected[idx], c[idx], 1e-5);
            }
        }
    }
}

test "gemmEx f16 and bf16 scalar extension" {
    try testGemmEx(.f16_f32, 2e-2);
    try testGemmEx(.bf16_f32, 5e-2);
}

fn testReal(comptime T: type, op_a: gemm.Op, op_b: gemm.Op, layout: gemm.Layout, alpha: T, beta: T, tol: T) !void {
    const m = 3;
    const n = 4;
    const k = 5;
    const a_rows = gemm.storedRows(op_a, m, k);
    const a_cols = gemm.storedCols(op_a, m, k);
    const b_rows = gemm.storedRows(op_b, k, n);
    const b_cols = gemm.storedCols(op_b, k, n);
    const lda = gemm.minimumLd(layout, a_rows, a_cols) + 2;
    const ldb = gemm.minimumLd(layout, b_rows, b_cols) + 3;
    const ldc = gemm.minimumLd(layout, m, n) + 1;

    const a_len = lenFor(layout, a_rows, a_cols, lda);
    const b_len = lenFor(layout, b_rows, b_cols, ldb);
    const c_len = lenFor(layout, m, n, ldc);
    const a = try testing.allocator.alloc(T, a_len);
    defer testing.allocator.free(a);
    const b = try testing.allocator.alloc(T, b_len);
    defer testing.allocator.free(b);
    const c = try testing.allocator.alloc(T, c_len);
    defer testing.allocator.free(c);
    const expected = try testing.allocator.alloc(T, c_len);
    defer testing.allocator.free(expected);

    fillReal(T, a, 0.3);
    fillReal(T, b, -0.4);
    fillReal(T, c, 0.8);
    @memcpy(expected, c);

    const a_layout = gemm.matrixLayout(layout, lda);
    const b_layout = gemm.matrixLayout(layout, ldb);
    const c_layout = gemm.matrixLayout(layout, ldc);
    referenceReal(T, op_a, op_b, m, n, k, alpha, a, a_layout, 0, b, b_layout, 0, beta, expected, c_layout, 0, 1);
    if (T == f32) {
        try gemm.sgemm(layout, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
        try gemm.dgemm(layout, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    for (0..m) |row| {
        for (0..n) |col| {
            const idx = c_layout.index(row, col);
            try testing.expectApproxEqAbs(expected[idx], c[idx], tol);
        }
    }
}

fn testComplex(comptime T: type, op_a: gemm.Op, op_b: gemm.Op, layout: gemm.Layout, alpha: T, beta: T, tol: anytype) !void {
    const m = 3;
    const n = 4;
    const k = 5;
    const a_rows = gemm.storedRows(op_a, m, k);
    const a_cols = gemm.storedCols(op_a, m, k);
    const b_rows = gemm.storedRows(op_b, k, n);
    const b_cols = gemm.storedCols(op_b, k, n);
    const lda = gemm.minimumLd(layout, a_rows, a_cols) + 2;
    const ldb = gemm.minimumLd(layout, b_rows, b_cols) + 1;
    const ldc = gemm.minimumLd(layout, m, n) + 3;

    const a_len = lenFor(layout, a_rows, a_cols, lda);
    const b_len = lenFor(layout, b_rows, b_cols, ldb);
    const c_len = lenFor(layout, m, n, ldc);
    const a = try testing.allocator.alloc(T, a_len);
    defer testing.allocator.free(a);
    const b = try testing.allocator.alloc(T, b_len);
    defer testing.allocator.free(b);
    const c = try testing.allocator.alloc(T, c_len);
    defer testing.allocator.free(c);
    const expected = try testing.allocator.alloc(T, c_len);
    defer testing.allocator.free(expected);

    fillComplex(T, a, 0.2);
    fillComplex(T, b, -0.1);
    fillComplex(T, c, 0.6);
    @memcpy(expected, c);

    const a_layout = gemm.matrixLayout(layout, lda);
    const b_layout = gemm.matrixLayout(layout, ldb);
    const c_layout = gemm.matrixLayout(layout, ldc);
    referenceComplex(T, op_a, op_b, m, n, k, alpha, a, a_layout, beta, expected, b, b_layout, c_layout);
    if (T == gemm.Complex32) {
        try gemm.cgemm(layout, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
        try gemm.zgemm(layout, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    for (0..m) |row| {
        for (0..n) |col| {
            const idx = c_layout.index(row, col);
            try testing.expectApproxEqAbs(expected[idx].re, c[idx].re, tol);
            try testing.expectApproxEqAbs(expected[idx].im, c[idx].im, tol);
        }
    }
}

fn testGemmEx(input: gemm.GemmType, tol: f32) !void {
    const m = 3;
    const n = 4;
    const k = 5;
    var a: [m * k]u16 = undefined;
    var b: [k * n]u16 = undefined;
    var c: [m * n]f32 = undefined;
    var expected: [m * n]f32 = undefined;

    for (&a, 0..) |*v, i| v.* = gemm.encodeInput(valueFor(f32, i, 0.25), input);
    for (&b, 0..) |*v, i| v.* = gemm.encodeInput(valueFor(f32, i, -0.35), input);
    fillReal(f32, &c, 0.45);
    expected = c;

    for (0..m) |row| {
        for (0..n) |col| {
            var sum: f32 = 0;
            for (0..k) |kk| {
                sum += gemm.decodeInput(a[row * k + kk], input) * gemm.decodeInput(b[kk * n + col], input);
            }
            expected[row * n + col] = 1.25 * sum - 0.5 * expected[row * n + col];
        }
    }

    try gemm.gemmEx(.{
        .op_a = .no_trans,
        .op_b = .no_trans,
        .m = m,
        .n = n,
        .k = k,
        .a = &a,
        .b = &b,
        .c = &c,
        .a_layout = .{ .row_stride = k, .col_stride = 1 },
        .b_layout = .{ .row_stride = n, .col_stride = 1 },
        .c_layout = .{ .row_stride = n, .col_stride = 1 },
        .alpha = 1.25,
        .beta = -0.5,
        .input = input,
    });

    for (expected, c) |e, got| try testing.expectApproxEqAbs(e, got, tol);
}

fn referenceReal(
    comptime T: type,
    op_a: gemm.Op,
    op_b: gemm.Op,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: []const T,
    a_layout: gemm.MatrixLayout,
    stride_a: usize,
    b: []const T,
    b_layout: gemm.MatrixLayout,
    stride_b: usize,
    beta: T,
    c: []T,
    c_layout: gemm.MatrixLayout,
    stride_c: usize,
    batch_count: usize,
) void {
    const used_stride_a = if (stride_a == 0) gemm.matrixExtent(a_layout, gemm.storedRows(op_a, m, k), gemm.storedCols(op_a, m, k)) else stride_a;
    const used_stride_b = if (stride_b == 0) gemm.matrixExtent(b_layout, gemm.storedRows(op_b, k, n), gemm.storedCols(op_b, k, n)) else stride_b;
    const used_stride_c = if (stride_c == 0) gemm.matrixExtent(c_layout, m, n) else stride_c;
    for (0..batch_count) |batch| {
        for (0..m) |row| {
            for (0..n) |col| {
                var sum: T = 0;
                for (0..k) |kk| {
                    const av = switch (op_a) {
                        .no_trans => a[batch * used_stride_a + a_layout.index(row, kk)],
                        .trans, .conj_trans => a[batch * used_stride_a + a_layout.index(kk, row)],
                    };
                    const bv = switch (op_b) {
                        .no_trans => b[batch * used_stride_b + b_layout.index(kk, col)],
                        .trans, .conj_trans => b[batch * used_stride_b + b_layout.index(col, kk)],
                    };
                    sum += av * bv;
                }
                const idx = batch * used_stride_c + c_layout.index(row, col);
                c[idx] = alpha * sum + beta * c[idx];
            }
        }
    }
}

fn referenceComplex(
    comptime T: type,
    op_a: gemm.Op,
    op_b: gemm.Op,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: []const T,
    a_layout: gemm.MatrixLayout,
    beta: T,
    c: []T,
    b: []const T,
    b_layout: gemm.MatrixLayout,
    c_layout: gemm.MatrixLayout,
) void {
    for (0..m) |row| {
        for (0..n) |col| {
            var sum: T = .{ .re = 0, .im = 0 };
            for (0..k) |kk| {
                const av = complexLoad(T, a, a_layout, op_a, row, kk);
                const bv = complexLoad(T, b, b_layout, op_b, kk, col);
                sum = complexAdd(T, sum, complexMul(T, av, bv));
            }
            const idx = c_layout.index(row, col);
            c[idx] = complexAdd(T, complexMul(T, alpha, sum), complexMul(T, beta, c[idx]));
        }
    }
}

fn complexLoad(comptime T: type, data: []const T, layout: gemm.MatrixLayout, op: gemm.Op, row: usize, col: usize) T {
    const value = switch (op) {
        .no_trans => data[layout.index(row, col)],
        .trans, .conj_trans => data[layout.index(col, row)],
    };
    return if (op == .conj_trans) .{ .re = value.re, .im = -value.im } else value;
}

fn complexAdd(comptime T: type, a: T, b: T) T {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}

fn complexMul(comptime T: type, a: T, b: T) T {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}

fn lenFor(layout: gemm.Layout, rows: usize, cols: usize, ld: usize) usize {
    return switch (layout) {
        .row_major => (rows - 1) * ld + cols,
        .col_major => (cols - 1) * ld + rows,
    };
}

fn fillReal(comptime T: type, dst: []T, bias: T) void {
    for (dst, 0..) |*value, i| value.* = valueFor(T, i, bias);
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
